# Real-Time Fintech Fraud Detection

A fraud detection pipeline built on the [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) mobile-money simulation dataset (6.3M+ transactions), combining behavioral & graph-based feature engineering, a soft-voting XGBoost + LightGBM ensemble, and a Redis Streams real-time inference engine.

## Results

| Metric | Score |
|---|---|
| PR-AUC (out-of-time test set) | **0.9929** |
| Recall | **99%** |
| Test set size | 550K transactions |
| Median inference latency | **~20ms** per transaction |

## Overview

- **EDA & Feature Engineering** — Filtered 6.3M+ transactions down to fraud-relevant types (`TRANSFER`, `CASH_OUT`), then engineered 6 purely behavioral signals: transaction velocity, hour-of-day, sender/receiver graph degree (to capture multi-node fraud rings), and transaction-to-wealth ratio. No raw ledger balances are used directly, avoiding label leakage.
- **Chronological Splitting** — Trained on the past, tested on the future (80/20 time-based split) to simulate real production conditions rather than a random split that would leak future information.
- **Modeling** — A soft-voting ensemble of XGBoost and LightGBM, each tuned for the heavy class imbalance in fraud data via `scale_pos_weight`.
- **Real-Time Inference** — A Redis Streams-based producer/consumer architecture: a simulated banking API streams live transactions, and a fraud engine consumer scores each one through the trained ensemble in real time.

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
