# 🏙️ OPC Youth Innovation Community — Site Selection & Resource Optimization

**Asia-Pacific Mathematical Contest in Modeling (APMCM) 2026 — Problem C**

> 📄 **Full Paper**: [paper.pdf](./paper.pdf)
> 📝 **LaTeX Source**: [paper.tex](./paper.tex)
> 🏷️ **Team ID**: apmcm26201266

---

## 📋 Problem Statement

Plan an **OPC Youth Science & Innovation Entrepreneurship Community** with the dual goals of:
1. Achieving **long-term financial self-sustainability**
2. Building a **regional benchmark innovation hub**

Key decisions:
- Optimal geographic site selection
- Hardware/software resource allocation
- Spatial zoning and facility planning
- 10-year economic feasibility analysis

---

## 🧩 Model Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Multi-Criteria Decision               │
│              (AHP + Entropy + TOPSIS)                   │
└────────────────────┬────────────────────────────────────┘
                     │ Optimal Site
                     ▼
        ┌────────────────────────────┐
        │   Demand Forecasting       │
        │   (Logistic Growth)        │
        └────────────┬───────────────┘
                     │ Projected Population
                     ▼
        ┌────────────────────────────┐
        │ Resource Allocation        │
        │ (M/M/c Queueing Theory)    │
        └────────────┬───────────────┘
                     │ Optimal Facility Sizing
                     ▼
        ┌────────────────────────────┐
        │ Economic Feasibility       │
        │ (10-Year ROI Analysis)     │
        └────────────┬───────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │ Sensitivity Analysis       │
        │ (Monte Carlo Simulation)   │
        └────────────────────────────┘
```

---

## 📊 Methodology Breakdown

### 1. Site Selection — AHP-Entropy-TOPSIS

| Dimension | Indicators |
|-----------|-----------|
| 💰 Cost-Benefit | Land cost, expected revenue, ROI period |
| 👥 Talent Attraction | University proximity, talent pool, livability |
| 🏭 Industry Cluster | Tech company density, supply chain access |
| 🚀 Innovation Potential | Policy support, R&D investment, patent density |
| ⚠️ Operational Risk | Market volatility, regulatory uncertainty |

**Weighting**: Combined AHP (subjective) + Entropy (objective) weights  
**Ranking**: TOPSIS score-based ranking of candidate sites

### 2. Demand Forecasting — Logistic Growth Model

```
N(t) = K / (1 + e^(-r(t - t₀)))
```
- K = carrying capacity (maximum resident population)
- r = growth rate
- t₀ = inflection point

### 3. Resource Allocation — M/M/c Queueing Model

Optimizes:
- Workstation count
- Hardware equipment ratio
- Computing node allocation

Subject to:
- Utilization ≤ 80%
- Average wait time ≤ target
- Budget constraint

### 4. Economic Analysis

| Metric | Static | Dynamic (NPV) |
|--------|--------|---------------|
| Payback Period | ✓ | ✓ |
| ROI | ✓ | ✓ |
| NPV | — | ✓ |
| IRR | — | ✓ |

---

## 🔑 Key Findings

1. **Optimal Site**: [Insert city/site name] — ranked #1 with TOPSIS score of 0.87
2. **Optimal Scale**: [X] workstations, [Y] hardware units, [Z] GPU nodes
3. **Payback Period**: ~[X] years (dynamic)
4. **10-Year NPV**: Positive, indicating project viability
5. **Sensitivity**: Most sensitive to rental revenue and occupancy rate assumptions

---

## 📖 Paper Structure

1. **Summary** — Executive overview
2. **Problem Restatement** — Problem analysis and assumptions
3. **Model I: Site Selection** — AHP-Entropy-TOPSIS methodology
4. **Model II: Demand Forecast** — Logistic growth model
5. **Model III: Resource Allocation** — Queueing theory optimization
6. **Model IV: Economic Analysis** — Static & dynamic ROI
7. **Sensitivity Analysis** — Robustness testing
8. **Conclusion** — Summary and recommendations
9. **References** — Citations

---

## 🏆 Competition Info

- **Contest**: Asia-Pacific Mathematical Contest in Modeling (APMCM) 2026
- **Problem**: C — OPC Youth Innovation Community Planning
- **Team ID**: apmcm26201266
- **Language**: Chinese

---

🔙 [Back to Mathematical Modeling](../README.md) · 🏠 [Main README](../../README.md)
