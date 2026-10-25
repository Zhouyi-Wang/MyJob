# 🧠 Pure Physics-Informed Neural Network

Standard PINN implementation with **ResNet backbone** and adaptive collocation point sampling.

## 📝 Overview

This is a clean, well-documented implementation of the classic PINN approach. It serves as:

- A **baseline** for comparing with advanced PINN variants
- An **educational reference** for understanding PINN fundamentals
- A **starting point** for new PINN research

## 🏗️ Architecture

```
Input (t)
  ↓
Linear(1 → hidden)
  ↓
[Residual Block × N]   ← Each: Linear → Tanh → Linear + skip connection
  ↓
Linear(hidden → 1)
  ↓
Output (u(t))
```

### Key Features

| Feature | Description |
|---------|-------------|
| **ResNet Backbone** | Deep network with residual connections for stable training |
| **Adaptive Sampling** | Dynamically adds collocation points in high-error regions |
| **PDE Residual Loss** | Physics constraint via automatic differentiation |
| **Boundary / Initial Loss** | Data constraints |

## 🧪 Problem Setup

- **ODE**: du/dt = u² + 1 (blow-up equation, used as benchmark)
- **Initial condition**: u(0) = 0
- **Domain**: t ∈ [0, π/2 - ε]
- **Network**: 6 hidden layers, 64 units each

## 🚀 Usage

```bash
python pure_pinn.py
```

Outputs:
- Training loss curves
- Solution comparison with analytical solution
- Error analysis

## 📐 PINN Loss Function

The total loss combines:

```
L_total = w_data · L_data + w_pde · L_pde + w_ic · L_ic
```

Where:
- `L_data` — MSE on observed data points
- `L_pde` — PDE residual at collocation points
- `L_ic` — Initial/boundary condition loss

---

🔙 [Back to PINN Research](../README.md) · 🏠 [Main README](../../README.md)
