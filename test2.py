import os
# 解决libiomp5md.dll重复初始化警告
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 4维输入神经网络
class PINN4D(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 1))
    def forward(self, X):
        return self.mlp(X)

model = PINN4D().to(device)
opt = optim.Adam(model.parameters(), lr=1e-4)
D = 0.02
lam = 1.0

# 计算PDE残差
def pde_loss(x, y, z, t):
    x.requires_grad = True
    y.requires_grad = True
    z.requires_grad = True
    t.requires_grad = True
    u = model(torch.cat([x,y,z,t], dim=1))

    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
    u_z = torch.autograd.grad(u.sum(), z, create_graph=True)[0]

    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
    u_zz = torch.autograd.grad(u_z.sum(), z, create_graph=True)[0]

    rhs = D*(u_xx+u_yy+u_zz) + lam * u * (1 - u)
    return torch.mean((u_t - rhs) ** 2)

# 采样函数
def sample_interior(N):
    x = torch.rand(N,1, device=device)
    y = torch.rand(N,1, device=device)
    z = torch.rand(N,1, device=device)
    t = torch.rand(N,1, device=device)
    return x,y,z,t

def sample_ic(N):
    x = torch.rand(N,1, device=device)
    y = torch.rand(N,1, device=device)
    z = torch.rand(N,1, device=device)
    t = torch.zeros_like(x)
    u_true = torch.sin(torch.pi*x)*torch.sin(torch.pi*y)*torch.sin(torch.pi*z)
    return x,y,z,t,u_true

def sample_bc(N):
    def face(xv):
        x = torch.full((N,1), xv, device=device)
        y = torch.rand(N,1, device=device)
        z = torch.rand(N,1, device=device)
        t = torch.rand(N,1, device=device)
        return x,y,z,t
    xs,ys,zs,ts = [],[],[],[]
    for xv in [0,1]:
        x,y,z,t = face(xv)
        xs.append(x);ys.append(y);zs.append(z);ts.append(t)
    for yv in [0,1]:
        x = torch.rand(N,1, device=device)
        y = torch.full((N,1), yv, device=device)
        z = torch.rand(N,1, device=device)
        t = torch.rand(N,1, device=device)
        xs.append(x);ys.append(y);zs.append(z);ts.append(t)
    for zv in [0,1]:
        x = torch.rand(N,1, device=device)
        y = torch.rand(N,1, device=device)
        z = torch.full((N,1), zv, device=device)
        t = torch.rand(N,1, device=device)
        xs.append(x);ys.append(y);zs.append(z);ts.append(t)
    return torch.cat(xs),torch.cat(ys),torch.cat(zs),torch.cat(ts)

# 训练
EPOCHS = 12000
loss_hist = []
N_int, N_ic, N_bc = 1800, 600, 300

print("==== 4D PDE PINN 训练开始 ====")
for ep in range(EPOCHS):
    opt.zero_grad()
    # PDE损失
    xi,yi,zi,ti = sample_interior(N_int)
    loss_pde = pde_loss(xi,yi,zi,ti)
    # 初值损失
    xic,yic,zic,tic,uic = sample_ic(N_ic)
    up_ic = model(torch.cat([xic,yic,zic,tic],1))
    loss_ic = torch.mean((up_ic - uic)**2)
    # 边界损失
    xbc,ybc,zbc,tbc = sample_bc(N_bc)
    up_bc = model(torch.cat([xbc,ybc,zbc,tbc],1))
    loss_bc = torch.mean(up_bc**2)

    loss = loss_pde + 100*loss_ic + 100*loss_bc
    loss.backward()
    opt.step()
    loss_hist.append(loss.item())

    if ep % 500 == 0:
        print(f"Epoch {ep:4d} | Total Loss: {loss.item():.3e} PDE:{loss_pde.item():.2e}")

# ---------------------- 1. 数值批量输出 ----------------------
print("\n==== 四维坐标预测数值输出 ====")
test_pts = torch.tensor([
    [0.1, 0.1, 0.1, 0.1],
    [0.2, 0.3, 0.5, 0.4],
    [0.5, 0.5, 0.5, 0.0],
    [0.5, 0.5, 0.5, 0.2],
    [0.5, 0.5, 0.5, 0.8],
    [0.9, 0.9, 0.9, 0.6]
], dtype=torch.float32, device=device)
u_pred = model(test_pts).cpu().detach().numpy().flatten()
for idx, (pt, val) in enumerate(zip(test_pts.cpu().numpy(), u_pred)):
    x,y,z,t = pt
    print(f"Point{idx+1}: (x={x:.1f}, y={y:.1f}, z={z:.1f}, t={t:.1f}) | u = {val:.6f}")

# ---------------------- 2. 绘图：损失曲线 + 固定t切片热力图 ----------------------
# 损失曲线
plt.figure(figsize=(8,3))
plt.plot(loss_hist)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.yscale("log")
plt.title("4D PDE Training Loss Curve")
plt.grid(True)
plt.show()

# 固定时间切片可视化 z=0.5 截面
def draw_slice(t_fix):
    n = 25
    xg, yg = np.meshgrid(np.linspace(0,1,n), np.linspace(0,1,n))
    zg = np.full_like(xg, 0.5)
    tg = np.full_like(xg, t_fix)
    grid = np.stack([xg, yg, zg, tg], -1).reshape(-1,4)
    grid_t = torch.tensor(grid, dtype=torch.float32, device=device)
    u = model(grid_t).cpu().detach().numpy().reshape(n,n)
    plt.figure()
    im = plt.imshow(u, extent=[0,1,0,1], origin="lower", cmap="jet")
    plt.colorbar(im)
    plt.title(f"4D PDE slice t={t_fix:.2f}, z=0.5, u(x,y)")
    plt.xlabel("x"); plt.ylabel("y")
    plt.show()

draw_slice(0.2)
draw_slice(0.8)