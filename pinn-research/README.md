# 🔬 Physics-Informed Neural Networks Research

This directory contains my research implementations of **Physics-Informed Neural Networks (PINNs)** — a class of neural networks that incorporate physical laws (described by partial differential equations) into the learning process.

## 📁 Subprojects

| Project | Description | Key Techniques |
|---------|-------------|----------------|
| [**Bayesian PINN**](./bayesian-pinn/) | Variational Bayesian approach to PINNs with uncertainty quantification | Variational Inference, Monte Carlo Dropout, KL Divergence |
| [**Laurent PINN**](./laurent-pinn/) | Hybrid NN + Laurent series architecture for singular PDEs | Laurent Series, Singularity Handling, Adaptive Gating |
| [**Pure PINN**](./pure-pinn/) | Standard PINN implementation with ResNet backbone | Residual Networks, Adaptive Sampling, PDE Constraints |
| [**Uncertainty Analysis**](./uncertainty-analysis/) | Tools for exploring and visualizing uncertainty bands | Confidence Intervals, Visualization, Statistical Analysis |

## 🎯 Research Goals

1. **Uncertainty Quantification** — Develop robust methods for estimating prediction uncertainty in PINNs
2. **Singularity Handling** — Improve PINN performance near singular points of PDE solutions
3. **Efficient Training** — Explore adaptive sampling and curriculum learning strategies
4. **Benchmarking** — Systematic comparison of different PINN variants across problem classes

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r ../requirements.txt

# Run Bayesian PINN experiment
cd bayesian-pinn
python bpinn_final.py

# Run Laurent PINN experiment
cd ../laurent-pinn
python laurent_pinn.py
```

## 📊 Sample Results

| Method | Problem Type | Key Advantage |
|--------|-------------|---------------|
| Pure PINN | Smooth PDEs | Simple, fast convergence |
| Bayesian PINN | Noisy data | Provides uncertainty estimates |
| Laurent PINN | Singular PDEs | Accurate near singularities |

---

🔙 [Back to Main README](../README.md)
