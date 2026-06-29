import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 网络构建 (Network Architecture)
# ==========================================
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        # 输入是时间 t (1维)，输出是 u (1维)
        # 使用一个简单的多层感知机 (MLP)
        self.net = nn.Sequential(
            nn.Linear(1, 20),
            nn.Tanh(),        # PINN 中常用 Tanh 或 Sin 等光滑激活函数，因为需要求导
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )
        
    def forward(self, t):
        return self.net(t)

# 实例化网络
model = PINN()

# ==========================================
# 2. 训练准备与数据生成
# ==========================================
# 并在 [0, 2] 区间内随机生成 100个 点作为物理控制点 (Collocation points)
t_physics = torch.linspace(0, 2, 100).view(-1, 1)
# 关键：设置 requires_grad=True，否则 PyTorch 无法对输入 t 求导
t_physics.requires_grad_(True)

# 初始条件 (Initial Condition, IC): t = 0 时, u = 1
t_ic = torch.tensor([[0.0]], requires_grad=True)
u_ic_true = torch.tensor([[1.0]])

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ==========================================
# 3. 训练过程 (Training Loop)
# ==========================================
epochs = 1000
loss_history = []

for epoch in range(epochs):
    optimizer.zero_grad()
    
    # ---- 3.1 初始条件损失 (Loss_IC) ----
    u_ic_pred = model(t_ic)
    loss_ic = torch.mean((u_ic_pred - u_ic_true) ** 2)
    
    # ---- 3.2 物理方程损失 (Loss_Physics) -> 自动微分 ----
    u_physics_pred = model(t_physics)
    
    # 使用 torch.autograd.grad 进行自动微分，求 du/dt
    du_dt = torch.autograd.grad(
        outputs=u_physics_pred, 
        inputs=t_physics,
        grad_outputs=torch.ones_like(u_physics_pred),
        create_graph=True,  # 保持计算图，以便对损失函数求导更新网络参数
        retain_graph=True
    )[0]
    
    # 物理方程是 du/dt + u = 0。我们希望这个残差 (residual) 越接近 0 越好
    physics_residual = du_dt + u_physics_pred
    loss_physics = torch.mean(physics_residual ** 2)
    
    # ---- 3.3 总损失与反向传播 ----
    total_loss = loss_ic + loss_physics
    total_loss.backward()
    optimizer.step()
    
    # 记录损失
    loss_history.append(total_loss.item())
    
    # 每 200 步打印一次日志
    if (epoch + 1) % 200 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Total Loss: {total_loss.item():.6f}, Loss_IC: {loss_ic.item():.6f}, Loss_Phys: {loss_physics.item():.6f}")

print("\n--- 训练完成 ---\n")

# ==========================================
# 4. 打印网络参数 (Print Parameters)
# ==========================================
print("【部分网络参数示例（第一层的权重前3行）：】")
print(model.net[0].weight[:3]) 
print("-" * 50)

# ==========================================
# 5. 输出 Loss 曲线与结果图像 (Visualization)
# ==========================================
# 测试数据
t_test = torch.linspace(0, 2, 200).view(-1, 1)
with torch.no_grad():
    u_pred = model(t_test).numpy()
t_test_np = t_test.numpy()
u_true = np.exp(-t_test_np) # 真实解 u = e^(-t)

# 开始画图
plt.figure(figsize=(12, 5))

# 子图 1：Loss 下降曲线
plt.subplot(1, 2, 1)
plt.plot(loss_history, label='Total Loss', color='purple')
plt.yscale('log') # 用对数坐标轴看得更清楚
plt.xlabel('Epoch')
plt.ylabel('Loss (Log Scale)')
plt.title('Loss Convergence')
plt.grid(True)
plt.legend()

# 子图 2：真实值 vs 预测值
plt.subplot(1, 2, 2)
plt.plot(t_test_np, u_true, label='Exact Solution ($e^{-t}$)', color='black', linewidth=2)
plt.plot(t_test_np, u_pred, '--', label='PINN Prediction', color='red', linewidth=2)
plt.xlabel('t')
plt.ylabel('u(t)')
plt.title('Comparison of Exact and Predicted Solution')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
