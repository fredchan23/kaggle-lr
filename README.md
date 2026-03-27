# Kaggle Linear Regression: MLB Wins Prediction

This workspace contains a linear-regression pipeline for predicting MLB team wins from historical team statistics. The main workflow lives in `moneyball-starter-code-share.ipynb` and produces `submission_predict.csv` for Kaggle submission.

## Project Files

- `moneyball-starter-code-share.ipynb`: main training, evaluation, plotting, and submission notebook.
- `mlb_wins_prediction_lr.ipynb`: alternate notebook variant.
- `data/data.csv`: training dataset.
- `data/predict.csv`: prediction dataset used for submission generation.
- `submission_predict.csv`: generated prediction file.

## Current Pipeline

The current notebook trains a `LinearRegression` model on:

- baseball counting stats
- existing derived run-rate features
- one-hot encoded era and decade indicators
- additional engineered baseball features such as run differential, win expectancy, rate stats, baserunning value, and a WHIP proxy

Numeric non-indicator columns are standardized with `StandardScaler`, while `era_*` and `decade_*` columns remain unchanged.

## Model Version History

Use this section as the canonical experiment log. Update it every time the model or feature set changes.

| Version | Change | Feature Count | Train MAE | Test MAE | Test RMSE | Test R² | Submission Output | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| v0 | Removed `ERA` from the default feature list to reduce redundant pitching information | 43 | 2.6630 | 2.8918 | 3.6451 | 0.9172 | `submission_predict.csv` | Metrics were effectively unchanged, suggesting `ERA` was not adding useful signal in the linear setup. |
| v1 | Added engineered baseball features: `run_diff`, `run_diff_per_game`, `win_expectancy`, `hr_rate`, `bb_rate`, `so_rate`, `hit_rate`, `extra_base_hits`, `xbh_rate`, `steal_value`, `baserunning_value`, `pitching_whip_proxy` | 55 | 2.6107 | 2.8067 | 3.5308 | 0.9224 | `submission_predict.csv` | Previous baseline. Linear model with engineered features. |
| v3 | Introduced RidgeCV for regularization comparison; refactored submission logic into a future-proof model registry & pipeline framework | 55 | 2.6155 | 2.7989 | 3.5235 | 0.9227 | `submission_RidgeCV_*.csv` | **Current version.** RidgeCV selected α=1.0 via 5-fold CV on MAE, yielding marginal improvement over LinearRegression (ΔTest MAE: −0.0078, ΔR²: +0.0003). Most importantly, introduced centralized model registry for easy comparison and switching between models without duplicating submission logic. Timestamped submission files prevent overwrites. |

## Why Feature Count Increased To 55

The notebook now builds the final feature list as:

```python
candidate_features = base_features + engineered_features
available_features = [
	col for col in candidate_features
	if col in data_df.columns and col in predict_df.columns
]
```

That means the feature count increased because engineered columns are created for both `data_df` and `predict_df`, then only the columns available in both datasets are kept.

## Recommended Workflow For Each New Model Version

When changing features, preprocessing, or model type:

1. Update the relevant notebook cell.
2. Rerun feature creation and train/test split.
3. Rerun scaling.
4. Rerun model fitting.
5. Rerun evaluation.
6. Rerun submission generation.
7. Record the new metrics in the version-history table above.
8. Commit the change in Git with one experiment per commit.

This keeps the model code, metrics, and generated submission aligned.

## v3 Model Registry & Submission Pipeline

Starting with **v3**, the notebook introduces a **centralized model registry** to support rapid comparison of multiple model variants without duplicating submission logic.

### Key Components

**1. Model Registry** (defines `models_config` dict)
- Stores all trained models with their test metrics and hyperparameters
- Makes it trivial to add new regularized variants (Lasso, ElasticNet, etc.)
- Enables side-by-side comparison

**2. `submit_model()` Function**
- Single reusable function for generating predictions and submission CSVs
- Automatically scales features using the fitted scaler
- Generates timestamped filenames to prevent overwrites
- Optional Kaggle API integration

**3. Batch Submission Cell**
- Generates submissions for **all registered models** in one cell execution
- No code changes needed to add new models to submission pipeline

**4. Deployment Selection Cell**
- Comment/uncomment the desired model for deployment
- Simplifies switching between LinearRegression and RidgeCV (or future variants)

### Extending v3 for New Models

To add a new regularized model variant:

```python
# In the model registry definition:
'ElasticNetCV': {
    'model': elasticnet_cv,
    'test_mae': elasticnet_test_mae,
    'test_r2': elasticnet_test_r2,
    'l1_ratio': 0.5,
    'version': 'v4_elastic',
}
```

Then run the batch submission cell—no additional code changes needed. The submission cell will automatically generate `submission_ElasticNetCV_*.csv`.

## Git Tracking For Model Evolution

Git should be used to track each model version so every metric change can be tied back to a concrete code change.

### Initialization

From the project root:

```bash
git init
git add README.md .gitignore moneyball-starter-code-share.ipynb mlb_wins_prediction_lr.ipynb
git commit -m "Initialize MLB wins prediction experiments"
```

### Commit Pattern

Use one commit per experiment. Example commit messages:

```bash
git commit -am "v0 drop ERA from linear feature set"
git commit -am "v1 add engineered baseball features"
git commit -am "v3 introduce RidgeCV and future-proof model registry"
```

### What To Record For Every Version

For each new version, document at least:

- version name
- modeling change
- feature count
- training MAE
- test MAE
- test RMSE
- test R²
- submission filename
- Kaggle public score, if submitted
- short conclusion

### Practical Rule

Do not mix multiple experimental ideas in one commit. One versioned change per commit makes it possible to explain why metrics moved.

## Next Sensible Versions

1. Compare the current linear model against `RidgeCV` using cross-validated MAE.
2. Test whether any engineered features are redundant by removing them one group at a time.
3. Add Kaggle public leaderboard scores to the version table after each submission.