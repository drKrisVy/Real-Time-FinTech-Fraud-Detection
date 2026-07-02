# Real-Time Fintech Fraud Detection

A fraud detection pipeline built on the [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) mobile-money simulation dataset (6.3M+ transactions), combining behavioral & graph-based feature engineering, a soft-voting XGBoost + LightGBM ensemble, and a Redis Streams real-time inference engine.

## Results

| Metric | Score |
|---|---|
| PR-AUC (out-of-time test set) | **0.9868** |
| Precision (fraud class) | **95%** |
| Recall (fraud class) | **99%** |
| F1 (fraud class) | **0.97** |
| Optimal decision threshold | 0.982 |
| Test set size | 552,504 transactions |
| Measured inference latency | **~0.5ms avg / ~0.8ms P95** (model only), ~2-4ms end-to-end via Redis |

Training is fully deterministic (`n_jobs=1`, `deterministic=True` on LightGBM), so these results reproduce exactly on every run.

## Overview

- **EDA** — Class imbalance, fraud-by-transaction-type breakdown (confirms fraud is confined to `TRANSFER`/`CASH_OUT`), amount distributions, and correlation analysis on raw ledger fields.
- **Feature Engineering** — 6 purely behavioral, **causal** signals: transaction velocity, hour-of-day, sender/receiver graph degree, and transaction-to-wealth ratio. No raw ledger balances used directly, avoiding label leakage. Graph-degree features are computed as running/expanding counts (not global counts) so no row ever sees information from a transaction that happens later in time.
- **Chronological Splitting** — Trained on the past, tested on the future (80/20 time-based split), avoiding lookahead bias from a random split.
- **Modeling** — Soft-voting ensemble of XGBoost and LightGBM, tuned for class imbalance via `scale_pos_weight`, trained single-threaded for full reproducibility.
- **Interpretability** — Feature importance and SHAP analysis; on this dataset, `amount_to_balance_ratio` drives ~97% of model decisions, with the graph-degree features contributing marginally — a finding worth noting rather than hiding.
- **Threshold Optimization** — Decision threshold chosen by maximizing F1 on the precision-recall curve, rather than an arbitrary cutoff; that threshold is used directly in evaluation and in the Redis inference engine.
- **Real-Time Inference** — A Redis Streams producer/consumer architecture, with measured (not estimated) latency benchmarking and a train/test data drift check.

## Tech Stack

`Python` · `pandas` / `numpy` · `scikit-learn` · `XGBoost` · `LightGBM` · `Redis Streams` · `matplotlib` / `seaborn`

## Project Structure

```
fintech-fraud-detection/
├── notebooks/
│   └── fraud_detection.ipynb   # Full pipeline: EDA → features → training → real-time engine
├── requirements.txt
└── README.md
```

## Getting Started

```bash
git clone https://github.com/YOUR_USERNAME/fintech-fraud-detection.git
cd fintech-fraud-detection
pip install -r requirements.txt
jupyter notebook notebooks/fraud_detection.ipynb
```

The notebook downloads the PaySim dataset automatically via `kagglehub` — no manual download needed. Module 6 requires a local Redis server (installed automatically in the notebook on Debian/Ubuntu-based environments, e.g. Colab).

## Dataset

[PaySim1](https://www.kaggle.com/datasets/ealaxi/paysim1) — a synthetic dataset simulating mobile money transactions, generated from real financial transaction logs, with injected fraudulent behavior.

## License

MIT
