# Time-Series Trading Research Lab — Roadmap & Checklist

This document is the **master plan** for the time-series ML trading research lab.
Use it as a **living checklist** to track progress and ensure research discipline.

---

## Guiding Principles (Read First)

- **No data leakage** — only information available at time *t* may be used.
- **Time-aware evaluation** — no random shuffling; walk-forward splits only.
- **Baselines first** — ML models must beat naive/statistical baselines out-of-sample.
- **Two scoreboards**:
  - Predictive quality (MAE, accuracy, calibration)
  - Economic value (PnL, Sharpe, drawdown)

---

## Phase 0 — Project Foundation & Data Integrity
**Goal:** A reproducible research lab that loads real data and runs end-to-end.

### Checklist
- [x] Project scaffold created
- [x] Conda environment (Python 3.11.x) configured
- [x] VS Code interpreter linked

### Data & Features
- [x] Load OHLC data (sorted, deduplicated, indexed)
- [x] Handle missing dates / NaNs explicitly
- [x] Define corporate action assumptions (no adjustments vs adjusted prices)
- [x] Compute log returns (core signal)
- [ ] Optional: intraday range features (H–L, C–O)
- [ ] Optional: rolling volatility features

### Windowing & Targets
- [x] Generic window builder (W, H)
- [x] Regression targets (next-day return)
- [ ] Multi-step cumulative return targets
- [ ] Sequence-to-sequence targets

---

## Phase 1 — First Forecasting Models (COMPLETED)
**Goal:** Validate the ML pipeline with working deep models.

### Checklist
- [x] MLP forecaster implemented
- [x] LSTM forecaster implemented
- [x] Train/test split (time-based)
- [x] MAE / RMSE evaluation
- [x] Forecast visualization

---

## Phase 1.5 — Experimental Discipline & Baselines (NEXT)
**Goal:** Upgrade the lab from demos to research-grade experiments.

### 1. Walk-Forward Evaluation
- [ ] Rolling train/test splits
- [ ] Configurable split definitions
- [ ] Per-split metric tracking
- [ ] Aggregate summary statistics

### 2. Baseline Models (First-Class)
- [ ] Naive zero-return forecast
- [ ] Persistence / last-value model
- [ ] Rolling mean forecast
- [ ] Linear / Ridge regression on lags

### 3. Metrics Upgrade
**Predictive**
- [ ] Directional accuracy (hit rate)
- [ ] Prediction–realization correlation
- [ ] Error by volatility regime

**Trading**
- [ ] Equity curve generation
- [ ] Sharpe ratio
- [ ] Max drawdown
- [ ] Turnover proxy

### 4. Minimal Trading Backtest
- [ ] Prediction → position mapping
- [ ] Transaction cost parameter
- [ ] Slippage assumption
- [ ] Strategy PnL calculation

### 5. Standard Experiment Report
- [ ] Config + metrics saved per run
- [ ] Summary tables (CSV/JSON)
- [ ] Automatic plots (equity, metrics over time)

---

## Phase 2 — Multi-Horizon Forecasting
**Goal:** Understand how horizon length affects predictability and tradability.

### Checklist
- [ ] Horizons: 1, 5, 10, 20, 60 days
- [ ] Cumulative return targets
- [ ] Sequence forecasting targets
- [ ] Walk-forward evaluation per horizon
- [ ] Horizon comparison summary

---

## Phase 3 — Classification Research Track
**Goal:** Test classification-based signals vs regression.

### Tasks
- [ ] Up / down next-day classifier
- [ ] 3-class direction classifier
- [ ] Large-move / event classifier

### Evaluation
- [ ] Accuracy / ROC-AUC / F1
- [ ] Probability calibration
- [ ] Confidence-filtered trading rules

---

## Phase 4 — Regime Detection & Conditioning
**Goal:** Identify market states and conditional performance.

### Regime Construction
- [ ] Volatility-based regimes
- [ ] Trend vs range regimes
- [ ] Clustering-based regimes
- [ ] (Optional) HMM regimes

### Analysis
- [ ] Regime visualization on price
- [ ] Forecast skill by regime
- [ ] Strategy performance by regime

---

## Phase 5 — Anomaly Detection & Risk Overlays
**Goal:** Detect abnormal periods and manage tail risk.

### Checklist
- [ ] Z-score volatility shocks
- [ ] Isolation Forest anomalies
- [ ] Drawdown alignment analysis
- [ ] Risk-off overlay backtests

---

## Phase 6 — Robustness & Research Hygiene
**Goal:** Ensure results are not artifacts.

### Robustness
- [ ] Sensitivity to window size
- [ ] Sensitivity to feature sets
- [ ] Sensitivity to cost assumptions
- [ ] Stability across time splits

### Hygiene
- [ ] Fixed random seeds
- [ ] Config versioning
- [ ] Environment export

---

## Phase 7 — Scaling & Advanced Research (Optional)
**Goal:** Approach professional systematic research.

### Ideas
- [ ] Probabilistic forecasts (quantiles)
- [ ] Portfolio-level signals
- [ ] Hyperparameter optimization
- [ ] Experiment tracking (MLflow / W&B)

---

## Status Summary
- [x] Phase 0 — Foundation
- [x] Phase 1 — First models
- [ ] Phase 1.5 — Experimental discipline
- [ ] Phase 2 — Multi-horizon
- [ ] Phase 3 — Classification
- [ ] Phase 4 — Regimes
- [ ] Phase 5 — Anomalies
- [ ] Phase 6 — Robustness
- [ ] Phase 7 — Scaling

---

**Rule of thumb:**
> *If a model does not beat baselines across walk-forward splits and survive transaction costs, it is not a signal — it is noise.*

