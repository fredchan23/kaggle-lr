# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Run in the conda `ml` environment. Activate with:
```bash
conda activate ml
```

## Running the Notebook

All work lives in `moneyball-starter-code-share.ipynb`. Run cells sequentially:

```bash
jupyter notebook moneyball-starter-code-share.ipynb
# or headless execution:
jupyter nbconvert --to notebook --execute moneyball-starter-code-share.ipynb --output moneyball-starter-code-share.ipynb
```

Kaggle submissions require `~/.kaggle/kaggle.json` credentials.

## Project Overview

Kaggle competition (`sctpdsai-m-3-ds-2-f-coaching-money-ball-analytics`) to predict MLB team wins (`W`) from historical team statistics. Training: 1,812 samples × 51 columns (`data/data.csv`). Test: 453 samples (`data/predict.csv`).

## Architecture & Data Flow

```
data.csv + predict.csv
    → Feature Engineering (adds 12 computed baseball features)
    → Feature Selection (55 features available in both train/predict sets)
    → 80/20 Train/Test Split (random_state=42)
    → StandardScaler (numeric stats only — NOT era_*/decade_* one-hot columns)
    → Model Training (LinearRegression + RidgeCV)
    → Model Registry (dict of models + their metrics)
    → submit_model() → timestamped CSV + optional Kaggle API submission
```

## Key Design Patterns

**Selective Scaling:** `era_1`–`era_8` and `decade_1910`–`decade_2010` one-hot columns are excluded from StandardScaler. Only continuous numeric features are scaled.

**Model Registry:** All trained models are registered in `models_config` dict with their metrics. The `submit_model(model_name, ...)` function handles prediction + CSV generation + optional Kaggle submission for any registered model. Add new models here to enable batch submission.

**Feature Parity:** Engineered features are computed on both `data_df` and `predict_df` before any train/test split, ensuring the scaler is fit only on training data but the same feature set exists in both.

**Timestamped Outputs:** Submission CSVs are named `submission_{ModelName}_{YYYYMMDD_HHMMSS}.csv` to preserve history without overwrites.

## Current Model Performance

| Version | Model | Test MAE | Test R² | Kaggle Score |
|---------|-------|----------|---------|--------------|
| v0 | LinearRegression (43 features, no ERA) | 2.8918 | 0.9172 | — |
| v1 | LinearRegression (55 features + engineered) | 2.8067 | 0.9224 | — |
| v3 | RidgeCV α=1.0 (55 features) | 2.7989 | 0.9227 | 3.04701 |
| v6 | RidgeCV α=1.0 (67 features) | 2.8177 | 0.9219 | 3.04416 |
| v11 (Current) | RidgeCV α=1.0 (69 features) | 2.8169 | 0.9219 | — |

## Engineered Features

24 features computed from raw stats in both train and predict sets (67 total features):

| Group | Features |
|---|---|
| Existing | `run_diff`, `run_diff_per_game`, `win_expectancy`, `hr_rate`, `bb_rate`, `so_rate`, `hit_rate`, `extra_base_hits`, `xbh_rate`, `steal_value`, `baserunning_value`, `pitching_whip_proxy` |
| Pythagorean | `win_exp_pythagorean` (PythagoPat), `win_exp_pythagenport` (Davenport scoring-env exponent), `win_exp_park_adj` (BPF/PPF-corrected) |
| Batting quality | `obp_proxy`, `iso`, `contact_rate`, `babip` |
| Pitching (DI) | `fip_proxy`, `k_bb_ratio` |
| Park-adjusted | `park_adj_run_diff` (uses `BPF`/`PPF`) |
| Efficiency rates | `cg_rate`, `sho_rate`, `dp_rate`, `e_rate` |

## Extending the Pipeline

To add a new model (e.g., Lasso, ElasticNet):
1. Train it using the existing `X_train_scaled`, `y_train`
2. Add to `models_config` with `model`, `test_mae`, `test_r2` keys
3. Run the batch submission cell — it will auto-generate CSVs for all registered models
