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

# ==================== 1. 纯PINN网络 (ResNet + Tanh) ====================
class PurePINN(nn.Module):
    def __init__(self, hidden_size=64, num_layers=6):
        super().__init__()
        self.input_layer = nn.Linear(1, hidden_size)
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        self.output_layer = nn.Linear(hidden_size, 1)
        self.activation = nn.Tanh()
        
    def forward(self, t):
        x = self.activation(self.input_layer(t))
        for layer in self.hidden_layers:
            residual = x
            x = self.activation(layer(x))
            x = x + residual
        output = self.output_layer(x)
        return output


# ==================== 2. 纯PINN求解器 ====================
class PurePINNSolver:
    def __init__(self, model, t_domain, u0=0.0, n_collocation=500):
        self.model = model
        self.t_domain = t_domain
        self.u0 = u0
        self.n_collocation = n_collocation
        self.singularity = np.pi / 2
        
    def compute_derivative(self, t):
        t = t.requires_grad_(True)
        u = self.model(t)
        du_dt = torch.autograd.grad(u.sum(), t, create_graph=True, retain_graph=True)[0]
        return u, du_dt
    
    def pde_residual(self, t):
        u, du_dt = self.compute_derivative(t)
        residual = du_dt - (u**2 + 1)
        return residual, u
    
    def adaptive_sampling(self, n_points):
        t_min, t_max = self.t_domain
        n_uniform = int(n_points * 0.7)
        uniform_samples = np.linspace(t_min, t_max, n_uniform)
        
        n_dense = n_points - n_uniform
        dense_left = np.random.uniform(
            max(t_min, self.singularity - 0.4), 
            self.singularity - 0.02, 
            n_dense // 2
        )
        dense_right = np.random.uniform(
            self.singularity + 0.02, 
            min(t_max, self.singularity + 0.4), 
            n_dense - n_dense // 2
        )
        
        samples = np.concatenate([uniform_samples, dense_left, dense_right])
        np.random.shuffle(samples)
        return torch.tensor(samples, dtype=torch.float32).view(-1, 1)
    
    def robust_loss_function(self):
        t_collocation = self.adaptive_sampling(self.n_collocation)
        t_collocation.requires_grad_(True)
        t0 = torch.tensor([[0.0]], dtype=torch.float32, requires_grad=True)
        
        residual, u = self.pde_residual(t_collocation)
        distance = torch.abs(t_collocation - self.singularity)
        weights = torch.exp(-2.0 / (distance + 0.15))
        
        pde_loss = (weights * torch.log1p(residual**2)).mean()
        pde_loss_mse = (weights * residual**2).mean()
        combined_pde_loss = 0.7 * pde_loss + 0.3 * pde_loss_mse
        
        u0_pred = self.model(t0)
        ic_loss = (u0_pred - self.u0)**2
        u_penalty = torch.mean(torch.relu(torch.abs(u) - 20.0)**2)
        
        loss = combined_pde_loss + 10.0 * ic_loss + 0.001 * u_penalty
        return loss, combined_pde_loss.item(), ic_loss.item()
    
    def train(self, epochs=10000, lr=1e-3):
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2500, gamma=0.5)
        
        losses = []
        pde_losses = []
        ic_losses = []
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss, pde_loss, ic_loss = self.robust_loss_function()
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()
            
            losses.append(loss.item())
            pde_losses.append(pde_loss)
            ic_losses.append(ic_loss)
            
            if epoch % 1000 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch:5d} | Loss: {loss.item():.6f} | PDE: {pde_loss:.6f} | IC: {ic_loss:.6f} | LR: {current_lr:.6f}")
        
        return losses, pde_losses, ic_losses
    
    def predict(self, t):
        self.model.eval()
        with torch.no_grad():
            u_pred = self.model(t)
        return u_pred.numpy()


# ==================== 3. 主程序 ====================
if __name__ == "__main__":
    print("=" * 70)
    print("Pure PINN for u' = u^2 + 1")
    print("Domain: [-1.5, 1.5] with singularity at pi/2")
    print("Method: Pure Physics-Informed Neural Network")
    print("Activation: Tanh | Epochs: 10000")
    print("=" * 70)
    
    t_domain = [-1.5, 1.5]
    model = PurePINN(hidden_size=64, num_layers=6)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nNetwork Architecture:")
    print(f"  Hidden size: 64")
    print(f"  Num layers: 6")
    print(f"  Activation: Tanh")
    print(f"  Total parameters: {total_params}")
    
    solver = PurePINNSolver(model, t_domain, u0=0.0, n_collocation=400)
    
    print(f"\nTraining (10000 epochs)...")
    print("-" * 70)
    losses, pde_losses, ic_losses = solver.train(epochs=10000, lr=8e-4)
    
    t_test = torch.linspace(t_domain[0], t_domain[1], 500).view(-1, 1)
    u_pred = solver.predict(t_test)
    
    t_np = t_test.numpy().flatten()
    u_exact = np.tan(t_np)
    
    mask = np.abs(u_exact) < 50
    t_plot = t_np[mask]
    u_exact_plot = u_exact[mask]
    u_pred_plot = u_pred.flatten()[mask]
    
    # ==================== 两张图画在一起（左右排列）====================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图: 损失曲线
    ax1 = axes[0]
    epochs_range = range(len(losses))
    ax1.semilogy(epochs_range, losses, 'b-', linewidth=1.5, label='Total Loss', alpha=0.8)
    ax1.semilogy(epochs_range, pde_losses, 'g--', linewidth=1, label='PDE Loss', alpha=0.7)
    ax1.semilogy(epochs_range, ic_losses, 'r:', linewidth=1, label='IC Loss', alpha=0.7)
    ax1.axhline(y=0.01, color='orange', linestyle='--', linewidth=2, label='Target: 0.01')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (log scale)', fontsize=12)
    ax1.set_title('Pure PINN Training Loss', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 右图: 解的比较
    ax2 = axes[1]
    ax2.plot(t_plot, u_exact_plot, 'k-', linewidth=2.5, label='Exact: tan(t)', alpha=0.9, zorder=3)
    ax2.plot(t_plot, u_pred_plot, 'r--', linewidth=2, label='PINN Prediction', alpha=0.9, zorder=2)
    ax2.axvline(x=np.pi/2, color='blue', linestyle=':', linewidth=2.5, alpha=0.8, 
                label=f'Singularity: pi/2 = {np.pi/2:.4f}', zorder=1)
    
    error = np.abs(u_pred_plot - u_exact_plot)
    ax2.fill_between(t_plot, u_exact_plot - error, u_exact_plot + error, 
                     alpha=0.15, color='red', label='Error band')
    
    ax2.set_xlabel('t', fontsize=12)
    ax2.set_ylabel('u(t)', fontsize=12)
    ax2.set_title("Solution of u' = u^2 + 1", fontsize=14)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([-30, 30])
    
    plt.tight_layout()
    plt.show()  # 显示图形，等待用户关闭
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Final Total Loss: {losses[-1]:.6f}")
    print(f"Final PDE Loss: {pde_losses[-1]:.6f}")
    print(f"Final IC Loss: {ic_losses[-1]:.6f}")
    
    error = np.abs(u_pred_plot - u_exact_plot)
    print(f"\nError Statistics:")
    print(f"  Mean Absolute Error: {np.mean(error):.6f}")
    print(f"  Max Absolute Error: {np.max(error):.6f}")
    print(f"  RMSE: {np.sqrt(np.mean(error**2)):.6f}")
    
    if losses[-1] < 0.01:
        print(f"\nTarget achieved! Loss < 0.01")
    else:
        print(f"\nTarget not reached. Final loss: {losses[-1]:.6f}")
    print("=" * 70)
