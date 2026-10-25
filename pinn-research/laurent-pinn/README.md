# 📐 Laurent Series-Enhanced PINN

A **hybrid neural network + analytic series** approach to solving PDEs with singular solutions.

## 📝 Overview

Standard PINNs struggle near singularities (blow-up, discontinuities, etc.). This project introduces **Laurent PINN**, which augments the neural network with learnable Laurent series terms that naturally capture singular behavior.

### Key Idea

```
u(t) = NN(t) + Σ(aₙ / (t - t₀)ⁿ)
        ↓              ↓
    Smooth part    Singular part
  (handled by NN)  (handled by Laurent series)
```

An **adaptive gating mechanism** smoothly transitions between the two components based on proximity to the singularity.

## 🎯 Features

| Feature | Description |
|---------|-------------|
| **Hybrid Architecture** | Combines universal approximation of NN with analytic precision of series expansions |
| **Adaptive Gating** | Sigmoid-based weighting that activates Laurent terms near singularities |
| **Learnable Coefficients** | Laurent series coefficients are trained end-to-end |
| **Singularity Awareness** | Explicitly models singular behavior instead of forcing NN to learn it |

## 🧪 Problem Setup

- **ODE**: du/dt = u² + 1 (solution blows up at t = π/2)
- **Singularity**: t₀ = π/2 ≈ 1.5708
- **Laurent terms**: 3 negative-power terms
- **Network layers**: [1, 50, 50, 50, 1]

## 🚀 Usage

```bash
python laurent_pinn.py
```

The script compares:
1. **Pure PINN** — baseline residual network
2. **Laurent PINN** — hybrid NN + Laurent series

## 📊 Expected Results

| Metric | Pure PINN | Laurent PINN |
|--------|-----------|--------------|
| Accuracy near singularity | Poor (large error) | Excellent |
| Solution smoothness | Oscillations near blow-up | Clean analytic behavior |
| Training stability | May diverge | More stable |

---

🔙 [Back to PINN Research](../README.md) · 🏠 [Main README](../../README.md)
