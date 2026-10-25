# 第一部分：环境变量（最顶部）
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 第二部分：必须先全部import，再写业务代码
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 解决中文、负号显示乱码
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 1. 拟合微分方程的神经网络
class ODE_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, t):
        return self.model(t)

# 2. PyTorch自动微分求导
def get_derivative(u, t):
    du_dt = torch.autograd.grad(
        outputs=u.sum(),
        inputs=t,
        create_graph=True,
        retain_graph=True
    )[0]
    return du_dt

# 3. 训练初始化（现在torch已经提前导入，不会报错）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = ODE_Net().to(device)
optimizer = optim.Adam(net.parameters(), lr=1e-4)

t_train = torch.linspace(0, 2*np.pi, 500, requires_grad=True).unsqueeze(1).to(device)
loss_history = []
epoch_num = 12000

# 4. 训练循环
print("开始训练，自动微分拟合微分方程...")
for epoch in range(epoch_num):
    optimizer.zero_grad()
    u_pred = net(t_train)
    du_dt_pred = get_derivative(u_pred, t_train)
    # 微分方程：u’ + u = 2sin(t)
    residual = du_dt_pred + u_pred - 2 * torch.sin(t_train)
    loss = torch.mean(residual ** 2)

    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())

    if (epoch + 1) % 500 == 0:
        print(f"Epoch [{epoch+1}/{epoch_num}], Loss = {loss.item():.6f}")

# 5. 打印训练后的最优网络全部参数
print("\n===== 训练完成，网络最优参数 =====")
for name, param in net.named_parameters():
    print(f"\n参数名: {name}")
    print(f"参数形状: {param.shape}")
    print(f"参数值:\n{param.detach().cpu().numpy()}")

# 6. 绘制Loss曲线并保存到本地（必生成图片文件）
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(loss_history, color="#2E86AB", linewidth=1.2)
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss")
ax.set_title("Loss 随训练轮次下降曲线（自动微分PINN）")
ax.grid(alpha=0.3)
ax.set_yscale("log")

# 强制保存图片，不受弹窗限制
plt.savefig("loss_curve.png", dpi=300, bbox_inches="tight")
print("\n✅ Loss图像已保存：项目文件夹下 loss_curve.png")

# 弹窗展示（关闭窗口后程序才结束）
plt.show(block=True)

# 7. 输入任意t输出预测结果
def predict_u(input_t):
    t_tensor = torch.tensor([[input_t]], dtype=torch.float32, requires_grad=True).to(device)
    u_out = net(t_tensor)
    return u_out.detach().cpu().item()

# 测试示例
test_t = 1.5
u_result = predict_u(test_t)
print(f"\n输入 t = {test_t}, 微分方程预测解 u(t) = {u_result:.6f}")

test_ts = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
print("\n批量输入测试：")
for ti in test_ts:
    print(f"t={ti:.2f} → u={predict_u(ti):.6f}")