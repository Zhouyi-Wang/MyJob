# 🔮 Bayesian Physics-Informed Neural Network

Bayesian PINN implementation using **variational inference** for uncertainty quantification in PDE-constrained learning.

## 📝 Overview

Traditional PINNs provide point estimates of PDE solutions without uncertainty measures. This implementation extends PINNs with Bayesian neural networks, enabling:

- **Uncertainty estimates** for predictions
- **Calibrated confidence intervals**
- **Robust parameter inference** under noise

## 🛠️ Architecture

```
Input (t)
    ↓
[Variational Linear × N]  ← Each layer has μ and σ parameters
    ↓
Output (u(t)) + uncertainty
```

### Key Components

| Component | Description |
|-----------|-------------|
| `VariationalLinear` | Custom layer with learnable mean and variance for weights |
| **KL Annealing** | Gradually increase KL weight to prevent posterior collapse |
| **MC Sampling** | Multiple forward passes for uncertainty estimation |
| **PDE Residual** | Physics-constrained loss with collocation points |

## 🧪 Experiment Details

### Problem Setup
- **ODE**: du/dt = λ·u (exponential decay/growth)
- **True λ**: 2.0
- **Initial condition**: u(0) = 1.0
- **Noise level**: 5% Gaussian noise on data

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Hidden dimensions | [32, 32] |
| Prior std | 1.0 |
| Learning rate | 1e-3 |
| Epochs | 10,000 |
| MC samples (train) | 30 |
| MC samples (predict) | 150 |
| Collocation points | 60 |
| Data points | 8 |

## 🚀 Usage

```bash
python bpinn_final.py
```

The script will:
1. Train the Bayesian PINN
2. Track λ estimation and loss convergence
3. Generate prediction intervals via Monte Carlo sampling
4. Save comparison plots to `../results/`

## 📈 Expected Results

- **λ inference**: Converges close to true value (2.0)
- **Uncertainty**: Well-calibrated bands that widen in data-sparse regions
- **Physics consistency**: PDE residual remains low throughout training

---

🔙 [Back to PINN Research](../README.md) · 🏠 [Main README](../../README.md)
