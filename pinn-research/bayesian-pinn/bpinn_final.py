import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# BPINN 终极修复版：阻止λ持续下滑，强化PDE约束力，保持低loss
# ============================================================================
t_min, t_max = 0.0, 1.0
true_lambda = 2.0
y0_true = 1.0
noise_std_data = 0.05

n_data = 8
n_col = 60

n_mc_train = 30
n_predict_samples = 150

# KL超参进一步弱化
beta_start = 0.0
beta_final = 0.00001  # 改动：全局KL再缩小10倍
anneal_epochs = 10000

hidden_dims = [32, 32]
prior_std = 1.0

# λ先验严格对齐真值，方差放大，减少拉扯
lambda_prior_mean = 2.0
lambda_prior_std = 2.0

# 改动：λ专属KL缩放，再缩小10倍，几乎不约束lambda
lambda_kl_scale = 0.001

epochs = 10000
lr = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================== 1. 变分层 ======================
class VariationalLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_std=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.weight_logsigma = nn.Parameter(torch.full((out_features, in_features), -3.0))
        self.bias_logsigma = nn.Parameter(torch.full((out_features,), -3.0))
    
    def sample_weights(self):
        weight_sigma = torch.exp(self.weight_logsigma)
        bias_sigma = torch.exp(self.bias_logsigma)
        weight_eps = torch.randn_like(self.weight_mu)
        bias_eps = torch.randn_like(self.bias_mu)
        weight = self.weight_mu + weight_sigma * weight_eps
        bias = self.bias_mu + bias_sigma * bias_eps
        return weight, bias
    
    def forward(self, x):
        weight, bias = self.sample_weights()
        return x @ weight.T + bias
    
    def kl_divergence(self):
        weight_sigma = torch.exp(self.weight_logsigma)
        bias_sigma = torch.exp(self.bias_logsigma)
        kl_weight = 0.5 * torch.sum(
            (weight_sigma / self.prior_std) ** 2 +
            (self.weight_mu / self.prior_std) ** 2 -
            1.0 - 2 * self.weight_logsigma + 2 * np.log(self.prior_std)
        )
        kl_bias = 0.5 * torch.sum(
            (bias_sigma / self.prior_std) ** 2 +
            (self.bias_mu / self.prior_std) ** 2 -
            1.0 - 2 * self.bias_logsigma + 2 * np.log(self.prior_std)
        )
        return kl_weight + kl_bias

# ====================== 2. BPINN网络 ======================
class BPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()
        dims = [1] + hidden_dims + [1]
        for i in range(len(dims) - 1):
            self.layers.append(VariationalLinear(dims[i], dims[i+1], prior_std))
        self.activation = nn.Tanh()
        
        self.lambda_mu = nn.Parameter(torch.tensor(1.95))
        self.lambda_logsigma = nn.Parameter(torch.tensor(-2.0))
    
    def sample_lambda(self):
        lambda_sigma = torch.exp(self.lambda_logsigma)
        eps = torch.randn(1, device=self.lambda_mu.device)
        lam = self.lambda_mu + lambda_sigma * eps
        return torch.nn.functional.softplus(lam) + 0.01
    
    def forward(self, t):
        x = t.unsqueeze(-1)
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)
        x = self.layers[-1](x)
        return x.squeeze(-1)
    
    def forward_with_lambda(self, t):
        u = self.forward(t)
        lam = self.sample_lambda()
        return u, lam
    
    def split_kl(self):
        kl_network = sum(layer.kl_divergence() for layer in self.layers)
        lambda_sigma = torch.exp(self.lambda_logsigma)
        kl_lambda = 0.5 * (
            (lambda_sigma / lambda_prior_std) ** 2 +
            ((self.lambda_mu - lambda_prior_mean) / lambda_prior_std) ** 2 -
            1.0 - 2 * self.lambda_logsigma + 2 * np.log(lambda_prior_std)
        )
        kl_lambda = kl_lambda * lambda_kl_scale
        return kl_network, kl_lambda

# ====================== 3. 数据生成 ======================
def generate_data():
    torch.manual_seed(42)
    np.random.seed(42)
    
    t_data = torch.linspace(t_min, t_max, n_data).to(device)
    y_clean = torch.exp(-true_lambda * t_data)
    y_data = y_clean + noise_std_data * torch.randn_like(y_clean)
    
    t_col = torch.linspace(t_min, t_max, n_col).to(device)
    t_ic = torch.tensor([t_min]).to(device)
    
    t_plot = torch.linspace(t_min, t_max, 200).to(device)
    y_plot_true = torch.exp(-true_lambda * t_plot)
    
    return t_data, y_data, t_col, t_ic, t_plot, y_plot_true

# ====================== 4. 损失函数（核心改动） ======================
def compute_loss(model, t_data, y_data, t_col, t_ic, beta_kl):
    u_data, lam = model.forward_with_lambda(t_data)
    nll_data = torch.mean((u_data - y_data) ** 2)
    
    u_ic, _ = model.forward_with_lambda(t_ic)
    nll_ic = torch.mean((u_ic - y0_true) ** 2)
    
    t_col_grad = t_col.clone().requires_grad_(True)
    u_col, lam_col = model.forward_with_lambda(t_col_grad)
    du_dt = torch.autograd.grad(
        u_col, t_col_grad,
        grad_outputs=torch.ones_like(u_col),
        create_graph=True, retain_graph=True
    )[0]
    
    residual = du_dt + lam_col * u_col
    nll_pde = torch.mean(residual ** 2)

    # 改动1：动态PDE惩罚力度大幅增强
    dev = torch.abs(lam - true_lambda)
    pde_weight = 200 + dev * 800
    # 改动2：提高IC初值权重，辅助约束
    ic_weight = 100

    nll_total = nll_data + pde_weight * nll_pde + ic_weight * nll_ic
    kl_net, kl_lam = model.split_kl()
    kl = beta_kl * kl_net + kl_lam
    total_loss = nll_total + kl
    
    return total_loss, nll_total, kl, nll_data.item(), nll_pde.item(), nll_ic.item(), lam.item()

# ====================== 5. MC Loss ======================
def compute_mc_loss(model, t_data, y_data, t_col, t_ic, beta_kl, n_samples=30):
    total_loss_sum = 0.0
    nll_sum = 0.0
    kl_sum = 0.0
    lam_samples = []
    
    for _ in range(n_samples):
        loss, nll, kl, nd, npde, nic, lam = compute_loss(model, t_data, y_data, t_col, t_ic, beta_kl)
        total_loss_sum += loss
        nll_sum += nll.detach()
        kl_sum += kl.detach()
        lam_samples.append(lam)
    
    return total_loss_sum / n_samples, nll_sum / n_samples, kl_sum / n_samples, np.mean(lam_samples)

# ====================== 6. 训练流程 ======================
def train_bpinn(model, epochs=10000):
    print(f'Starting BPINN training, epochs: {epochs}')
    print(f'Network: {hidden_dims}, Activation: tanh')
    print(f'Collocation points: {n_col}, Data points: {n_data}')
    print(f'Device: {device}')
    print('='*60)
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    t_data, y_data, t_col, t_ic, t_plot, y_plot_true = generate_data()
    
    history = {'total_loss': [], 'nll': [], 'kl': [], 'beta': [], 'lambda_mean': []}
    best_loss = float('inf')
    best_state = None
    
    for epoch in range(epochs):
        beta = beta_start + (beta_final - beta_start) * min(epoch / anneal_epochs, 1.0)
        
        optimizer.zero_grad()
        loss, nll, kl, lam_mean = compute_mc_loss(model, t_data, y_data, t_col, t_ic, beta, n_mc_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        history['total_loss'].append(loss.item())
        history['nll'].append(nll.item())
        history['kl'].append(kl.item())
        history['beta'].append(beta)
        history['lambda_mean'].append(lam_mean)
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if epoch % 500 == 0:
            print(f'Epoch {epoch:5d} | Loss: {loss.item():.4f} | NLL: {nll.item():.4f} | KL: {kl.item():.4f} | beta: {beta:.6f} | lambda: {lam_mean:.3f}')
    
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f'\nRestored best model (Loss: {best_loss:.4f})')
    
    print('='*60)
    print('Training complete!')
    return history, t_data, y_data, t_col, t_ic, t_plot, y_plot_true

# ====================== 绘图、诊断、主函数 ======================
def plot_results(model, history, t_data, y_data, t_col, t_ic, t_plot, y_plot_true):
    pred_curves = []
    lambda_samples = []
    model.eval()
    
    with torch.no_grad():
        for _ in range(n_predict_samples):
            u_pred, lam = model.forward_with_lambda(t_plot)
            pred_curves.append(u_pred.cpu().numpy())
            lambda_samples.append(lam.item())
    
    pred_curves = np.array(pred_curves)
    lambda_samples = np.array(lambda_samples)
    pred_mean = pred_curves.mean(axis=0)
    pred_std = pred_curves.std(axis=0)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    ax1 = axes[0]
    ax1.plot(t_plot.cpu().numpy(), y_plot_true.cpu().numpy(), 'k-', linewidth=2, label=f'True: exp(-{true_lambda}t)')
    ax1.scatter(t_data.cpu().numpy(), y_data.cpu().numpy(), c='red', s=60, zorder=5, label=f'Noisy data (n={n_data})')
    ax1.set_xlabel('t', fontsize=12)
    ax1.set_ylabel('u(t)', fontsize=12)
    ax1.set_title('1. Data and True Solution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.plot(t_plot.cpu().numpy(), pred_mean, 'b-', linewidth=2, label='Posterior mean')
    ax2.fill_between(t_plot.cpu().numpy(), pred_mean - 2*pred_std, pred_mean + 2*pred_std, alpha=0.3, color='blue', label='95% CI')
    ax2.plot(t_plot.cpu().numpy(), y_plot_true.cpu().numpy(), 'k--', linewidth=2, label='True solution')
    ax2.scatter(t_data.cpu().numpy(), y_data.cpu().numpy(), c='red', s=40, zorder=5)
    ax2.set_xlabel('t', fontsize=12)
    ax2.set_ylabel('u(t)', fontsize=12)
    ax2.set_title('2. Posterior Mean with Uncertainty', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[2]
    for i in range(min(50, n_predict_samples)):
        ax3.plot(t_plot.cpu().numpy(), pred_curves[i], 'lightblue', alpha=0.3, linewidth=0.8)
    ax3.plot(t_plot.cpu().numpy(), y_plot_true.cpu().numpy(), 'k-', linewidth=2, label='True solution')
    ax3.plot(t_plot.cpu().numpy(), pred_mean, 'b--', linewidth=1.5, label='Posterior mean')
    ax3.set_xlabel('t', fontsize=12)
    ax3.set_ylabel('u(t)', fontsize=12)
    ax3.set_title(f'3. Posterior Predictive Samples', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[3]
    ax4.hist(lambda_samples, bins=30, color='green', alpha=0.6, density=True, edgecolor='black', linewidth=0.5)
    ax4.axvline(true_lambda, color='red', linestyle='--', linewidth=2, label=f'True lambda = {true_lambda}')
    ax4.axvline(np.mean(lambda_samples), color='blue', linestyle='-', linewidth=2, label=f'Posterior mean = {np.mean(lambda_samples):.3f}')
    ax4.axvline(np.median(lambda_samples), color='orange', linestyle=':', linewidth=2, label=f'Posterior median = {np.median(lambda_samples):.3f}')
    ax4.set_xlabel('lambda', fontsize=12)
    ax4.set_ylabel('Density', fontsize=12)
    ax4.set_title('4. Posterior Distribution of lambda', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    
    ax5 = axes[4]
    epochs_arr = np.arange(len(history['nll']))
    ax5.semilogy(epochs_arr, history['nll'], 'b-', linewidth=1.5, label='Expected NLL')
    ax5.semilogy(epochs_arr, history['kl'], 'r-', linewidth=1.5, alpha=0.7, label='KL divergence')
    ax5.semilogy(epochs_arr, history['total_loss'], 'g--', linewidth=1, alpha=0.5, label='Total loss')
    ax5.set_xlabel('Epoch', fontsize=12)
    ax5.set_ylabel('Loss (log scale)', fontsize=12)
    ax5.set_title('5. Training Loss Curves', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    ax6 = axes[5]
    ax6.plot(epochs_arr, history['lambda_mean'], 'purple', linewidth=1.5, label='lambda estimate')
    ax6.axhline(true_lambda, color='red', linestyle='--', linewidth=2, label=f'True lambda = {true_lambda}')
    ax6.set_xlabel('Epoch', fontsize=12)
    ax6.set_ylabel('lambda', fontsize=12)
    ax6.set_title('6. Convergence of lambda Estimate', fontsize=14, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bpinn_fixed_lambda_final.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return pred_curves, lambda_samples

def run_diagnostics(model, t_data, y_data, t_col, t_ic, t_plot, y_plot_true, pred_curves, lambda_samples, history):
    print('\n' + '='*60)
    print('Diagnostic Results')
    print('='*60)
    
    pred_mean = pred_curves.mean(axis=0)
    pred_std = pred_curves.std(axis=0)
    
    mse_mean = np.mean((pred_mean - y_plot_true.cpu().numpy())**2)
    print(f'1. Posterior mean MSE: {mse_mean:.6f}')
    
    data_indices = []
    t_plot_np = t_plot.cpu().numpy()
    t_data_np = t_data.cpu().numpy()
    for td in t_data_np:
        idx = np.argmin(np.abs(t_plot_np - td))
        data_indices.append(idx)
    std_at_data = np.mean(pred_std[data_indices])
    std_elsewhere = np.mean(np.delete(pred_std, data_indices))
    print(f'\n2. Uncertainty check:')
    print(f'   - Average std at data points: {std_at_data:.4f}')
    print(f'   - Average std elsewhere: {std_elsewhere:.4f}')
    
    lambda_mean = np.mean(lambda_samples)
    lambda_std = np.std(lambda_samples)
    lambda_error = abs(lambda_mean - true_lambda) / true_lambda * 100
    print(f'\n3. Lambda estimation:')
    print(f'   - True value: {true_lambda:.3f}')
    print(f'   - Posterior mean: {lambda_mean:.3f} +/- {lambda_std:.3f}')
    print(f'   - Relative error: {lambda_error:.2f}%')
    
    model.eval()
    with torch.no_grad():
        t_col_batch = t_col.clone().requires_grad_(True)
        u_col, lam_col = model.forward_with_lambda(t_col_batch)
        du_dt = torch.autograd.grad(u_col, t_col_batch, grad_outputs=torch.ones_like(u_col), create_graph=False)[0]
        residual = du_dt + lam_col * u_col
        residual_mean = torch.mean(torch.abs(residual)).item()
    
    print(f'\n4. Collocation point residuals:')
    print(f'   - Mean absolute residual: {residual_mean:.6f}')
    
    print(f'\n5. KL term behavior:')
    print(f'   - Final KL: {history["kl"][-1]:.4f}')
    print(f'   - KL/NLL ratio: {history["kl"][-1]/history["nll"][-1]:.4f}')
    print('='*60)
    
    return {'mse_mean': mse_mean, 'lambda_error': lambda_error, 'residual_mean': residual_mean}

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    
    print('='*60)
    print('Final Fixed BPINN: Stop λ continuous decline')
    print('Problem: du/dt = -lambda*u, u(0) = 1')
    print('='*60)
    
    model = BPINN()
    print(f'\nTotal parameters: {sum(p.numel() for p in model.parameters())}')
    
    history, t_data, y_data, t_col, t_ic, t_plot, y_plot_true = train_bpinn(model, epochs=epochs)
    pred_curves, lambda_samples = plot_results(model, history, t_data, y_data, t_col, t_ic, t_plot, y_plot_true)
    metrics = run_diagnostics(model, t_data, y_data, t_col, t_ic, t_plot, y_plot_true, pred_curves, history)
    
    print('\n' + '='*60)
    print('Final Results Summary')
    print('='*60)
    print(f'True lambda: {true_lambda:.4f}')
    print(f'Posterior mean lambda: {np.mean(lambda_samples):.4f} +/- {np.std(lambda_samples):.4f}')
    print(f'Posterior median lambda: {np.median(lambda_samples):.4f}')
    print(f'Relative error: {abs(np.mean(lambda_samples) - true_lambda)/true_lambda*100:.2f}%')
    print(f'95% CI: [{np.percentile(lambda_samples, 2.5):.3f}, {np.percentile(lambda_samples, 97.5):.3f}]')
    print('='*60)