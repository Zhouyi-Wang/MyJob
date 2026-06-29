import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ====================== 1. 模型模块 ======================
class PINN(nn.Module):
    def __init__(self, hidden_dim=20, activation=nn.Tanh()):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, t):
        return self.net(t)

# ====================== 2. 损失计算模块 ======================
def calculate_total_loss(model, t_physics, t_ic, u_ic_true):
    # 初始条件损失
    u_ic_pred = model(t_ic)
    loss_ic = torch.mean((u_ic_pred - u_ic_true) ** 2)

    # 自动求导 du/dt
    u_pred = model(t_physics)
    du_dt = torch.autograd.grad(
        outputs=u_pred,
        inputs=t_physics,
        grad_outputs=torch.ones_like(u_pred),
        create_graph=True,
        retain_graph=True
    )[0]

    # ODE残差 du/dt + u = 0
    residual = du_dt - u_pred**2 - 1
    loss_phys = torch.mean(residual ** 2)
    total_loss = loss_ic + loss_phys
    return total_loss, loss_ic, loss_phys

# ====================== 3. 训练循环模块 ======================
def train_pinn(model, optimizer, epochs, print_interval, t_physics, t_ic, u_ic_true):
    loss_history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        total_loss, loss_ic, loss_phys = calculate_total_loss(model, t_physics, t_ic, u_ic_true)
        total_loss.backward()
        optimizer.step()
        loss_history.append(total_loss.item())

        if (epoch + 1) % print_interval == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Total Loss: {total_loss.item():.6f}, Loss_IC: {loss_ic.item():.6f}, Loss_Phys: {loss_phys.item():.6f}")
    return loss_history

# ====================== 4. 工具可视化模块 ======================
def show_first_layer_weights(model, top_rows=3):
    print("【部分网络参数示例（第一层的权重前3行）：】")
    print(model.net[0].weight[:top_rows])
    print("-" * 50)

def draw_curve_plot(loss_history, model):
    # 测试数据
    t_test = torch.linspace(0, torch.pi/2-0.05, 200).view(-1, 1)
    with torch.no_grad():
        u_pred = model(t_test).numpy()
    t_test_np = t_test.numpy()
    u_true = np.tan(t_test_np)

    plt.figure(figsize=(12, 5))
    # 子图1 Loss曲线
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, label='Total Loss', color='purple')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Log Scale)')
    plt.title('Loss Convergence')
    plt.grid(True)
    plt.legend()

    # 子图2 真实解vs预测
    plt.subplot(1, 2, 2)
    plt.plot(t_test_np, u_true, label='Exact Solution ($\\tan(t)$)', color='black', linewidth=2)
    plt.plot(t_test_np, u_pred, '--', label='PINN Prediction', color='red', linewidth=2)
    plt.xlabel('t')
    plt.ylabel('u(t)')
    plt.title('Comparison of Exact and Predicted Solution')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

# ====================== 5. 主运行入口（统一调参区） ======================
if __name__ == "__main__":
    # 超参数统一修改区
    HIDDEN_SIZE = 20
    ACT_FUNC = nn.Tanh()  # 可改为 nn.Sin() / nn.Softplus()
    TRAIN_EPOCHS = 5000
    LEARNING_RATE = 0.001
    PRINT_STEP = 200
    PHYSICS_POINTS = 1000

    # 1. 初始化模型与优化器
    model = PINN(hidden_dim=HIDDEN_SIZE, activation=ACT_FUNC)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 2. 构造训练采样点
    t_physics = torch.linspace(0, torch.pi/2-0.05, PHYSICS_POINTS).view(-1, 1)
    t_physics.requires_grad_(True)
    t_ic = torch.tensor([[0.0]], requires_grad=True)
    u_ic_true = torch.tensor([[0.0]])

    # 3. 执行训练
    loss_record = train_pinn(
        model=model,
        optimizer=optimizer,
        epochs=TRAIN_EPOCHS,
        print_interval=PRINT_STEP,
        t_physics=t_physics,
        t_ic=t_ic,
        u_ic_true=u_ic_true
    )
    print("\n--- 训练完成 ---\n")

    # 4. 查看权重、绘图
    show_first_layer_weights(model)
    draw_curve_plot(loss_record, model)