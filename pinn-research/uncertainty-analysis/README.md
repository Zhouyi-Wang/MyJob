# 📊 Uncertainty Analysis & Visualization

Tools for **exploring, visualizing, and analyzing uncertainty bands** in PINN predictions.

## 📝 Overview

This module provides utilities for:

- Generating and plotting **confidence intervals**
- Comparing **uncertainty estimates** across methods
- **Statistical analysis** of prediction variance
- **Calibration checks** for uncertainty quantification

## 🔧 Tools

| Script | Purpose |
|--------|---------|
| `uncertainty_bands.py` | Generate and visualize uncertainty bands from Bayesian PINN samples |

## 📊 Analysis Methods

### 1. **Confidence Interval Calculation**
- Mean ± 2σ (Gaussian assumption)
- Percentile-based (empirical, more robust)

### 2. **Uncertainty Decomposition**
- Epistemic uncertainty (model uncertainty)
- Aleatoric uncertainty (data noise)

### 3. **Calibration Metrics**
- Coverage probability
- Interval width analysis
- Sharpness vs. calibration trade-off

## 🚀 Usage

```bash
python uncertainty_bands.py
```

Generates plots showing:
- Prediction mean with uncertainty bands
- Individual sample trajectories
- Uncertainty growth patterns

---

🔙 [Back to PINN Research](../README.md) · 🏠 [Main README](../../README.md)
