import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# ==================== 1. 洛朗级数增强的PINN ====================
class LaurentPINN(nn.Module):
    """
    混合架构：神经网络 + 洛朗级数
    u(t) = NN(t) + sum(a_n / (t - t_singularity)^n)
    """
    def __init__(self, layers, n_laurent_terms=3, singularity=np.pi/2):
        super().__init__()
        self.singularity = singularity
        self.n_laurent_terms = n_laurent_terms
        
        # 标准神经网络部分
        self.net_layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.net_layers.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                self.net_layers.append(nn.Tanh())
        
        # 洛朗级数系数 (可学习)
        self.laurent_coeffs = nn.Parameter(torch.randn(n_laurent_terms) * 0.1)
        self.laurent_offset = nn.Parameter(torch.tensor(0.0))
        
    def neural_net(self, t):
        x = t
        for layer in self.net_layers:
            if isinstance(layer, nn.Linear):
                x = layer(x)
            else:
                x = layer(x)
        return x
    
    def laurent_series(self, t):
        dt = t - self.singularity
        dt = torch.where(torch.abs(dt) < 1e-6, torch.sign(dt) * 1e-6, dt)
        
        laurent_sum = torch.zeros_like(t)
        for n in range(1, self.n_laurent_terms + 1):
            coeff = self.laurent_coeffs[n-1]
            laurent_sum = laurent_sum + coeff / (dt ** n)
        
        return laurent_sum + self.laurent_offset
    
    def forward(self, t):
        nn_output = self.neural_net(t)
        laurent_output = self.laurent_series(t)
        
        distance_to_singularity = torch.abs(t - self.singularity)
        laurent_weight = torch.sigmoid(10.0 * (0.3 - distance_to_singularity))
        nn_weight = 1.0 - laurent_weight
        
        output = nn_weight * nn_output + laurent_weight * laurent_output
        return output


# ==================== 2. 轻量级求解器 ====================
class FastPINNSolver:
    def __init__(self, model, t_domain, u0=0.0, n_collocation=300):
        self.model = model
        self.t_domain = t_domain
        self.u0 = u0
        self.n_collocation = n_collocation
        self.singularity = np.pi / 2
        
    def compute_derivative(self, t):
        t = t.requires_grad_(True)
        u = self.model(t)
        du_dt = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        return u, du_dt
    
    def pde_residual(self, t):
        u, du_dt = self.compute_derivative(t)
        residual = du_dt - (u**2 + 1)
        return residual
    
    def smart_sampling(self, n_points):
        t_min, t_max = self.t_domain
        n_uniform = n_points // 2
        uniform_samples = np.linspace(t_min, t_max, n_uniform)
        
        n_dense = n_points - n_uniform
        dense_center = self.singularity
        dense_range = 0.5
        dense_samples = np.random.normal(dense_center, dense_range/3, n_dense)
        dense_samples = np.clip(dense_samples, t_min, t_max)
        
        samples = np.concatenate([uniform_samples, dense_samples])
        np.random.shuffle(samples)
        return torch.tensor(samples, dtype=torch.float32).view(-1, 1)
    
    def loss_function(self):
        t_collocation = self.smart_sampling(self.n_collocation)
        t_collocation.requires_grad_(True)
        t0 = torch.tensor([[0.0]], dtype=torch.float32, requires_grad=True)
        
        residual = self.pde_residual(t_collocation)
        weights = 1.0 / (1.0 + torch.abs(t_collocation - self.singularity))
        pde_loss = (weights * residual**2).mean()
        
        u0_pred = self.model(t0)
        ic_loss = (u0_pred - self.u0)**2
        
        laurent_reg = 0.01 * torch.sum(self.model.laurent_coeffs**2)
        loss = pde_loss + 5.0 * ic_loss + laurent_reg
        
        return loss, pde_loss.item(), ic_loss.item()
    
    def train(self, epochs=8000, lr=1e-3):
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
        losses = []
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss, pde_loss, ic_loss = self.loss_function()
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
            
            if epoch % 200 == 0:
                print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f} | PDE: {pde_loss:.6f} | IC: {ic_loss:.6f}")
        
        return losses
    
    def predict(self, t):
        self.model.eval()
        with torch.no_grad():
            u_pred = self.model(t)
        return u_pred.numpy()


# ==================== 3. 主程序 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("Laurent-PINN for u' = u^2 + 1")
    print("Domain: [-1.5, 1.5] with singularity at pi/2")
    print("=" * 60)
    
    t_domain = [-1.5, 1.5]
    layers = [1, 32, 32, 1]
    model = LaurentPINN(layers, n_laurent_terms=3, singularity=np.pi/2)
    
    print(f"\nNetwork: {layers}")
    print(f"Laurent terms: 3")
    print(f"Total params: {sum(p.numel() for p in model.parameters())}")
    
    solver = FastPINNSolver(model, t_domain, u0=0.0, n_collocation=200)
    
    print(f"\nTraining (8000 epochs)...")
    losses = solver.train(epochs=12000, lr=5e-3)
    
    t_test = torch.linspace(t_domain[0], t_domain[1], 400).view(-1, 1)
    u_pred = solver.predict(t_test)
    
    t_np = t_test.numpy().flatten()
    u_exact = np.tan(t_np)
    
    mask = np.abs(u_exact) < 30
    t_plot = t_np[mask]
    u_exact_plot = u_exact[mask]
    u_pred_plot = u_pred.flatten()[mask]
    
    # 图1: 损失曲线
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.semilogy(losses, 'b-', linewidth=1.5)
    ax1.axhline(y=0.01, color='r', linestyle='--', linewidth=2, label='Target: 0.01')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (log scale)', fontsize=12)
    ax1.set_title('Training Loss - Laurent-PINN', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('D:/wzy/My_study/surf BPINN/loss_curve.png', dpi=150)
    plt.show(block=True)
    
    # 图2: 解的比较
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    ax2.plot(t_plot, u_exact_plot, 'k-', linewidth=2.5, label='Exact: tan(t)', alpha=0.9)
    ax2.plot(t_plot, u_pred_plot, 'r--', linewidth=2, label='Laurent-PINN Prediction', alpha=0.9)
    ax2.axvline(x=np.pi/2, color='blue', linestyle=':', linewidth=2, alpha=0.7,
                label=f'Singularity: pi/2 = {np.pi/2:.3f}')
    ax2.set_xlabel('t', fontsize=12)
    ax2.set_ylabel('u(t)', fontsize=12)
    ax2.set_title("Solution Comparison: u' = u^2 + 1 with Laurent Series", fontsize=14)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([-25, 25])
    plt.tight_layout()
    plt.savefig('D:/wzy/My_study/surf BPINN/solution_comparison.png', dpi=150)
    plt.show(block=True)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Final Loss: {losses[-1]:.6f}")
    error = np.abs(u_pred_plot - u_exact_plot)
    print(f"Mean Error: {np.mean(error):.6f}")
    print(f"\nLaurent Coefficients:")
    for i, coeff in enumerate(model.laurent_coeffs.detach().numpy()):
        print(f"  a_{i+1} = {coeff:.6f}")
    print(f"  offset = {model.laurent_offset.item():.6f}")
    print(f"\nTheory: tan(t) ~ -1/(t-pi/2) + (t-pi/2)/3 - ...")
    print(f"\nOutput files:")
    print(f"  1. D:/wzy/My_study/surf BPINN/loss_curve.png")
    print(f"  2. D:/wzy/My_study/surf BPINN/solution_comparison.png")
    print("=" * 60)
