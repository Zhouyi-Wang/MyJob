import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import copy

# 固定随机种子保证复现
torch.manual_seed(42)
np.random.seed(42)

# ---------------------- 1. 物理系统与数据生成 ----------------------
TRUE_C = 0.5
TRUE_K = 2.0
T_MAX = 10.0
N_OBS = 40
NOISE_STD = 0.05

def analytical_solution(t, c=TRUE_C, k=TRUE_K):
    # 欠阻尼解析解 z(0)=1, z'(0)=0
    omega_d = np.sqrt(k - (c**2)/4)
    alpha = c / 2
    A = 1.0
    B = alpha / omega_d
    return np.exp(-alpha * t) * (A * np.cos(omega_d * t) + B * np.sin(omega_d * t))

# 观测数据 0~8
t_obs = np.linspace(0, 8.0, N_OBS)
z_clean = analytical_solution(t_obs)
z_obs = z_clean + np.random.normal(0, NOISE_STD, size=t_obs.shape)

# 张量转换
t_obs_tensor = torch.tensor(t_obs, dtype=torch.float32).unsqueeze(1)
z_obs_tensor = torch.tensor(z_obs, dtype=torch.float32).unsqueeze(1)

# PDE全域配点 0~10
t_pde_tensor = torch.linspace(0, T_MAX, 450, dtype=torch.float32).unsqueeze(1)
t_pde_tensor.requires_grad = True

# 初始边界点 t=0
t_ic = torch.tensor([[0.0]], dtype=torch.float32, requires_grad=True)

# ---------------------- 2. 贝叶斯网络层 ----------------------
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), -3.0))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.full((out_features,), -3.0))

    def forward(self, x, sample=True):
        if sample:
            w_sig = torch.log1p(torch.exp(self.weight_rho))
            b_sig = torch.log1p(torch.exp(self.bias_rho))
            w = self.weight_mu + w_sig * torch.randn_like(self.weight_mu)
            b = self.bias_mu + b_sig * torch.randn_like(self.bias_mu)
        else:
            w = self.weight_mu
            b = self.bias_mu
        return F.linear(x, w, b)

    def kl_divergence(self):
        w_sig = torch.log1p(torch.exp(self.weight_rho))
        b_sig = torch.log1p(torch.exp(self.bias_rho))
        kl_w = 0.5 * torch.sum(w_sig**2 + self.weight_mu**2 - 1.0 - torch.log(w_sig**2))
        kl_b = 0.5 * torch.sum(b_sig**2 + self.bias_mu**2 - 1.0 - torch.log(b_sig**2))
        return kl_w + kl_b

class BPINN(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.l1 = BayesianLinear(1, hidden_dim)
        self.l2 = BayesianLinear(hidden_dim, hidden_dim)
        self.l3 = BayesianLinear(hidden_dim, 1)
        # 物理参数放宽方差，更容易学到真值
        self.c_mu = nn.Parameter(torch.tensor(1.2))
        self.c_rho = nn.Parameter(torch.tensor(-0.3))
        self.k_mu = nn.Parameter(torch.tensor(1.2))
        self.k_rho = nn.Parameter(torch.tensor(-0.3))
        self.log_noise = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x, sample=True):
        x = torch.tanh(self.l1(x, sample))
        x = torch.tanh(self.l2(x, sample))
        return self.l3(x, sample)

    def sample_parameters(self):
        c_sig = torch.log1p(torch.exp(self.c_rho))
        k_sig = torch.log1p(torch.exp(self.k_rho))
        c = self.c_mu + c_sig * torch.randn_like(self.c_mu)
        k = self.k_mu + k_sig * torch.randn_like(self.k_mu)
        sigma = torch.clamp(F.softplus(self.log_noise), min=1e-4)
        return c, k, sigma

    def kl_divergence(self):
        kl_net = self.l1.kl_divergence() + self.l2.kl_divergence() + self.l3.kl_divergence()
        # 放大先验方差，弱化KL惩罚
        prior_std = 3.0
        c_sig = torch.log1p(torch.exp(self.c_rho))
        k_sig = torch.log1p(torch.exp(self.k_rho))
        kl_c = 0.5 * ((c_sig**2 + (self.c_mu - 1.0)**2)/(prior_std**2) - 1 - torch.log((c_sig/prior_std)**2))
        kl_k = 0.5 * ((k_sig**2 + (self.k_mu - 1.0)**2)/(prior_std**2) - 1 - torch.log((k_sig/prior_std)**2))
        return kl_net + kl_c + kl_k

# ---------------------- 3. 训练配置 ----------------------
model = BPINN(hidden_dim=64)
optimizer = optim.Adam(model.parameters(), lr=1.8e-2)
epochs = 1500  # 加长迭代，避免过早收敛假象
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

# 超参调整
lambda_pde = 3.2    # 强化ODE约束，优先拟合振荡
lambda_ic = 1.5     # 初值边界损失权重，锁死波形起点
kl_scale = 2e-5     # 大幅降低KL缩放，弱化正则
S = 14              # 提升MC采样，稳定梯度
NLL_OFFSET = 4.0    # NLL固定偏移，保证loss全程正数

# 日志容器
loss_history = []
nll_history = []
pde_history = []
ic_history = []
kl_history = []

best_loss = float('inf')
best_model_state = None
patience = 60
patience_counter = 0

print("===== 弱化KL正则+加长训练+正向Loss+强化PDE，解决后半段漂移 ====")
for epoch in range(epochs):
    optimizer.zero_grad()
    # KL退火拉长至900轮，前900轮几乎无正则约束
    beta = min(1.0, epoch / 900.0)

    nll_loss = 0.0
    pde_loss = 0.0
    ic_loss = 0.0
    sigma_obs = torch.exp(model.log_noise)

    for _ in range(S):
        c_sample, k_sample, sigma = model.sample_parameters()
        # 1. 观测NLL（增加固定偏移，保证数值恒正）
        z_pred_obs = model(t_obs_tensor, sample=True)
        mse_term = 0.5 * torch.mean((z_obs_tensor - z_pred_obs)**2) / (sigma**2)
        log_sig_term = torch.log(sigma)
        nll_loss += mse_term + log_sig_term + NLL_OFFSET

        # 2. PDE残差（修复梯度shape问题）
        z_pde = model(t_pde_tensor, sample=True)
        dz_dt = torch.autograd.grad(z_pde.sum(), t_pde_tensor, torch.tensor(1.0), create_graph=True, retain_graph=True)[0]
        d2z_dt2 = torch.autograd.grad(dz_dt.sum(), t_pde_tensor, torch.tensor(1.0), create_graph=True, retain_graph=True)[0]
        res = d2z_dt2 + c_sample * dz_dt + k_sample * z_pde
        pde_loss += torch.mean(res**2)

        # 3. 初始边界条件 z(0)=1, z'(0)=0
        z_ic = model(t_ic, sample=True)
        dz_ic = torch.autograd.grad(z_ic.sum(), t_ic, torch.tensor(1.0), create_graph=True, retain_graph=True)[0]
        ic_loss += (z_ic - 1.0)**2 + (dz_ic)**2

    nll_loss /= S
    pde_loss /= S
    ic_loss /= S
    kl_loss = model.kl_divergence() * kl_scale

    # 总损失
    loss = nll_loss + lambda_pde * pde_loss + lambda_ic * ic_loss + beta * kl_loss

    loss.backward()
    optimizer.step()
    scheduler.step()

    # 记录损失
    loss_history.append(loss.item())
    nll_history.append(nll_loss.item())
    pde_history.append(pde_loss.item())
    ic_history.append(ic_loss.item())
    kl_history.append(kl_loss.item() * beta)

    # 早停
    if loss.item() < best_loss:
        best_loss = loss.item()
        best_model_state = copy.deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1
    if patience_counter >= patience:
        print(f"早停触发 Epoch {epoch}, Best Loss: {best_loss:.4f}")
        break

    if epoch % 100 == 0 or epoch == epochs - 1:
        c_est = model.c_mu.item()
        k_est = model.k_mu.item()
        print(f"Epoch {epoch:04d} | Total:{loss.item():.4f} | NLL:{nll_loss.item():.4f} | PDE:{pde_loss.item():.4f} | IC:{ic_loss.item():.4f} | c={c_est:.3f} k={k_est:.3f}")

# 加载最优权重
model.load_state_dict(best_model_state)

# ---------------------- 4. 预测与绘图 ----------------------
t_test = np.linspace(0, T_MAX, 200)
t_test_tensor = torch.tensor(t_test, dtype=torch.float32).unsqueeze(1)
MC_SAMPLES = 500
latent_samples = []
predictive_samples = []
c_samples = []
k_samples = []
sigma_final = torch.exp(model.log_noise).detach().numpy()

with torch.no_grad():
    for _ in range(MC_SAMPLES):
        c_s, k_s, sig = model.sample_parameters()
        z_pred = model(t_test_tensor, sample=True).numpy().flatten()
        latent_samples.append(z_pred)
        noise = np.random.normal(0, sigma_final, size=z_pred.shape)
        predictive_samples.append(z_pred + noise)
        c_samples.append(c_s.item())
        k_samples.append(k_s.item())

latent_samples = np.array(latent_samples)
predictive_samples = np.array(predictive_samples)
c_samples = np.array(c_samples)
k_samples = np.array(k_samples)

mean_latent = np.mean(latent_samples, axis=0)
latent_low = np.percentile(latent_samples, 2.5, axis=0)
latent_up = np.percentile(latent_samples, 97.5, axis=0)
pred_low = np.percentile(predictive_samples, 2.5, axis=0)
pred_up = np.percentile(predictive_samples, 97.5, axis=0)
truth = analytical_solution(t_test)

# 绘图
plt.style.use('default')
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
# 波形图
axs[0,0].fill_between(t_test, pred_low, pred_up, color='#ffddaa', alpha=0.2, label='95% Posterior-Predictive Band')
axs[0,0].fill_between(t_test, latent_low, latent_up, color='#6688ff', alpha=0.3, label='95% Latent Credible Band')
for i in range(30):
    axs[0,0].plot(t_test, latent_samples[i], c='#6688ff', alpha=0.12, lw=1)
axs[0,0].scatter(t_obs, z_obs, c='red', s=15, zorder=5, label='Noisy Observations')
axs[0,0].plot(t_test, truth, 'k--', lw=2, label='Ground Truth Analytical')
axs[0,0].set_title("BPINN Waveform & Uncertainty Bounds", fontsize=14)
axs[0,0].set_xlabel("Time (t)")
axs[0,0].set_ylabel("Displacement (z)")
axs[0,0].legend()
axs[0,0].grid(alpha=0.3)

# 损失曲线（对数坐标，全程正数）
axs[0,1].plot(loss_history, label='Total ELBO Loss', c='purple')
axs[0,1].plot(nll_history, label='NLL Loss', c='#882222', alpha=0.7)
axs[0,1].plot(pde_history, label='PDE Residual Loss', c='#226688', alpha=0.7)
axs[0,1].plot(ic_history, label='IC Boundary Loss', c='#22aa22', alpha=0.7)
axs[0,1].set_yscale('log')
axs[0,1].set_title("Training Loss Trajectories (Log Scale, Positive Only)")
axs[0,1].legend()
axs[0,1].grid(alpha=0.3)

# c后验直方图
axs[1,0].hist(c_samples, bins=40, density=True, alpha=0.7, color='#33aaaa')
axs[1,0].axvline(TRUE_C, c='red', ls='--', lw=2, label=f'True c = {TRUE_C}')
axs[1,0].axvline(np.mean(c_samples), c='blue', lw=2, label=f'Est Mean = {np.mean(c_samples):.3f}')
axs[1,0].set_title("Posterior Distribution of c")
axs[1,0].set_xlabel("c Value")
axs[1,0].legend()

# k后验直方图
axs[1,1].hist(k_samples, bins=40, density=True, alpha=0.7, color='#ee8866')
axs[1,1].axvline(TRUE_K, c='red', ls='--', lw=2, label=f'True k = {TRUE_K}')
axs[1,1].axvline(np.mean(k_samples), c='blue', lw=2, label=f'Est Mean = {np.mean(k_samples):.3f}')
axs[1,1].set_title("Posterior Distribution of k")
axs[1,1].set_xlabel("k Value")
axs[1,1].legend()

plt.tight_layout()
plt.show()

# ---------------------- 验证指标输出 ----------------------
mse = np.mean((mean_latent - truth)**2)
# 区分插值区(0~8)与外推区(8~10)不确定性带宽
mask_dense = t_test <= 8
mask_sparse = t_test > 8
width_dense = np.mean(latent_up[mask_dense] - latent_low[mask_dense])
width_sparse = np.mean(latent_up[mask_sparse] - latent_low[mask_sparse])

# 观测点覆盖概率
count_in = 0
for t_val, z_val in zip(t_obs, z_obs):
    idx = np.abs(t_test - t_val).argmin()
    if pred_low[idx] <= z_val <= pred_up[idx]:
        count_in += 1
cover_rate = count_in / len(z_obs)

c_mean, c_std = np.mean(c_samples), np.std(c_samples)
k_mean, k_std = np.mean(k_samples), np.std(k_samples)
c_recover = abs(c_mean - TRUE_C) < 2 * c_std
k_recover = abs(k_mean - TRUE_K) < 2 * k_std

print("\n" + "="*60)
print("                  BPINN 验证报告")
print("="*60)
print(f"1. 全局波形重建MSE: {mse:.6f}")
print(f"2. 隐变量置信带平均宽度对比：")
print(f"   插值区(0~8): {width_dense:.4f} | 外推区(t>8): {width_sparse:.4f}")
print(f"   判定：{'正常（外推不确定性更大）' if width_sparse > width_dense else '异常'}")
print(f"3. 观测点落在95%预测带覆盖率：{cover_rate * 100:.1f}%（理想≈95%）")
print(f"4. 物理参数恢复：")
print(f"   c: {c_mean:.4f} ± {c_std:.4f} 真值{TRUE_C}，区间覆盖真值：{c_recover}")
print(f"   k: {k_mean:.4f} ± {k_std:.4f} 真值{TRUE_K}，区间覆盖真值：{k_recover}")
print("="*60)