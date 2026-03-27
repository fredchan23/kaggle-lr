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
| v3 | Introduced RidgeCV for regularization comparison; refactored submission logic into a future-proof model registry & pipeline framework | 55 | 2.6155 | 2.7989 | 3.5235 | 0.9227 | `submission_RidgeCV_*.csv` | RidgeCV selected α=1.0 via 5-fold CV on MAE, yielding marginal improvement over LinearRegression (ΔTest MAE: −0.0078, ΔR²: +0.0003). Introduced centralized model registry; timestamped submission files prevent overwrites. |
| v4 | Added LassoCV (L1 regularization) — cross-validates over α to drive less predictive coefficients to zero | 55 | — | — | — | — | `submission_LassoCV_*.csv` | L1 sparsity may help identify and drop redundant engineered features. Best α selected via 5-fold CV. |
| v5 | Added ElasticNetCV (L1 + L2) — cross-validates over both α and l1_ratio | 55 | — | — | — | — | `submission_ElasticNetCV_*.csv` | Combines Ridge's coefficient shrinkage with Lasso's sparsity. Best α and l1_ratio selected via 5-fold CV over grids [0.0001…100] × [0.1, 0.25, 0.5, 0.75, 0.9]. |
| v6 | Added 12 new engineered features: PythagoPat win expectancy, OBP proxy, ISO, contact rate, BABIP, FIP proxy, K/BB ratio, park-adjusted run diff, CG/SHO/DP/error rates | 67 | 2.6035 | 2.8177 | — | 0.9219 | `submission_RidgeCV_*.csv` | **Current best.** RidgeCV best model (α=1.0). Local test MAE slightly up vs v3 due to split variance, but Kaggle public score improved 3.04701 → 3.04416, confirming better generalisation. Park-factor adjustment and FIP proxy add the most signal. |
| v7a | Random Forest — unconstrained (`max_depth=30, min_leaf=1`) | 67 | 1.2746 | 3.2712 | — | 0.8925 | — | Severe overfit: train MAE 1.27 vs test MAE 3.27. Deep trees memorise ~1,450 training samples. Not submitted. |
| v7b | Random Forest — constrained (`max_depth=7, min_leaf=5, min_split=20`) | 67 | 2.4069 | 3.2824 | — | 0.8910 | — | Constraining closed the train/test gap but test MAE barely moved (3.27 → 3.28). RF still underperforms linear models. Not submitted. |
| v8 | HuberRegressor — robust loss, grid-searched `epsilon` × `alpha` | 67 | 2.6073 | 2.8236 | — | 0.9215 | — | Best params: ε=1.75, α=1.0. Beats vanilla LinearRegression but trails RidgeCV by 0.006 MAE. High ε confirms few extreme outliers — robust loss adds limited value. Not submitted. |
| v9 | GradientBoostingRegressor — sklearn defaults (`max_depth=3, n_estimators=100, lr=0.1`) | 67 | 2.2302 | 3.1893 | — | 0.8991 | — | Overfits similarly to RF despite shallower trees. Default regularisation insufficient for ~1,450 samples. Not submitted. |
| v10 | HistGradientBoostingRegressor — 50-iter random search, early stopping | 67 | 2.3932 | 3.2632 | — | 0.8946 | — | Best params: `max_depth=2, min_leaf=20, lr=0.2, L2=1.0, max_leaf_nodes=31`. Heavily constrained but train/test gap persists (2.39 vs 3.26). All tree/boosting approaches exhausted. Not submitted. |
| v11 | Added Pythagenport + park-adjusted Pythagorean features; retuned RidgeCV with wider alpha grid | 69 | 2.6035 | 2.8169 | — | 0.9219 | `submission_RidgeCV_*.csv` | **Best Kaggle score.** Pythagenport + park-adj Pythagorean both add signal. Alpha stays 1.0 on wider grid. Kaggle public: **3.04393**. |
| v12 | Added `run_diff×win_exp` + `win_exp×G` interaction terms; stratified train/test split by era | 71 | 2.6047 | 2.7686 (LR) | — | 0.9314 | `submission_LinearRegression_*.csv` | Stratified split makes local eval more honest (MAE 2.81→2.77) but Kaggle score regressed to 3.05159 (LR) / 3.05325 (Ridge). Interaction terms + wider feature set overfit slightly. Random-split v11 RidgeCV still holds best Kaggle score. See local/Kaggle gap analysis below. |
| v13 | Pythagorean Residual RidgeCV — stratified split | 71 | 2.6173 | 2.7659 | — | 0.9321 | `submission_PythagoreanResidualRidge_*.csv` | Best local MAE (2.7659) but Kaggle 3.05304. Stratified split + residual both tested. |
| v14 | Pythagorean Residual RidgeCV — random split (same as v11 conditions) | 71 | 2.6057 | 2.8141 | — | 0.9221 | `submission_PythagoreanResidualRidge_*.csv` | Ties RidgeCV locally (2.8141 vs 2.8140) but Kaggle **3.05653** — definitively worse than plain RidgeCV. Residual formulation exhausted. |
| analytical | Pure Pythagorean baseline: `W = round(R²/(R²+RA²) × G)` — no ML | — | — | 3.2942 | — | — | `submission_Pythagorean_*.csv` | Kaggle: **3.61316**. Confirms ML pipeline adds ~0.57 MAE of genuine signal over the analytical formula. Model is not adding noise. |

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

### Adding a New Model

Train the model, then call `register_model()` — that's it:

```python
from sklearn.linear_model import LassoCV

lasso_cv = LassoCV(alphas=[...], cv=5, max_iter=10000)
lasso_cv.fit(X_train_scaled, y_train)
register_model("LassoCV", lasso_cv, alpha=lasso_cv.alpha_, version="v4_l1")
```

Re-running `compare_models()`, the batch submission cell, and the deploy cell requires no further edits — they all iterate over the registry automatically.

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

## Random Forest Exploration Finding (v7)

Two RF configurations were tested against the v6 feature set (67 features, ~1,450 training samples):

| Config | max_depth | min_leaf | Train MAE | Test MAE | Verdict |
|---|---|---|---|---|---|
| Unconstrained | 30 | 1 | 1.27 | 3.27 | Severe overfit |
| Constrained | 7 | 5 | 2.41 | 3.28 | Overfit resolved, still worse than linear |

**Finding:** Constraining the RF closed the generalisation gap but did not improve test MAE — both configurations lost to RidgeCV (2.8177). The near-identical test MAEs confirm the problem is not overfitting but that **RF's non-linear capacity adds no value here**: the relationship between the engineered features and wins is predominantly linear. The engineered features (run differential, win expectancy, FIP proxy, etc.) already encode the domain knowledge that tree splits would otherwise need to rediscover.

**Implication:** Further gains should come from gradient boosting with proper regularisation (XGBoost/LightGBM), better features, or feature selection — not from deeper/wider trees.

## Tree-Based Model Exploration Finding (v7–v10) — CLOSED

Four tree/boosting configurations all underperform the best linear model (RidgeCV test MAE 2.8177):

| Model | Train MAE | Test MAE | Gap |
|---|---|---|---|
| RF unconstrained (v7a) | 1.27 | 3.27 | 2.00 — severe overfit |
| RF constrained (v7b) | 2.41 | 3.28 | 0.87 — gap closed, still worse |
| GradientBoosting default (v9) | 2.23 | 3.19 | 0.96 — overfits |
| HistGradientBoosting tuned (v10) | 2.39 | 3.26 | 0.87 — heavily constrained, still worse |

**Conclusion:** Tree-based exploration is exhausted. All variants — shallow, deep, tuned, untuned, with and without L2 — fail to match regularised linear models. The MLB wins prediction problem is fundamentally linear given the current feature set. The engineered features (run differential, win expectancy, FIP proxy, etc.) already encode the domain signal that tree splits would otherwise rediscover. No further tree/boosting models will be explored unless the feature set changes significantly.

## Local vs Kaggle Gap Analysis

Across all submitted versions, a persistent ~0.2 MAE gap exists between local test and Kaggle public score:

| Version | Local MAE | Kaggle | Gap |
|---|---|---|---|
| v3 RidgeCV (random split) | 2.7989 | 3.04701 | 0.248 |
| v11 RidgeCV (random split) | 2.8169 | 3.04393 | 0.227 |
| v12 LR (stratified split) | 2.7686 | 3.05159 | 0.283 |

**Key findings:**
- Era distribution differs between train and predict: `era_7` (1993–2009) is 6.3pp underrepresented in predict vs train
- Stratifying the split by era makes local evaluation more honest (MAE drops) but did not improve Kaggle score — the gap is structural, not a split artefact
- The ~0.2 gap is likely driven by year-on-year variance in baseball scoring environment not fully captured by era/decade indicators
- **v11 random-split RidgeCV holds the best Kaggle score (3.04393)**

## Pythagorean Residual Finding (v13–v14) — CLOSED

Tested both split strategies with the residual formulation:

| Config | Local MAE | Kaggle |
|---|---|---|
| Stratified split (v13) | 2.7659 | 3.05304 |
| Random split (v14) | 2.8141 | 3.05653 |
| v11 plain RidgeCV (random) | 2.8169 | **3.04393** |

**Conclusion:** Pythagorean residual is definitively worse on Kaggle regardless of split. Despite tying locally (2.8141 vs 2.8140), the residual decomposition produces predictions the Kaggle holdout penalises more — likely because the Pythagorean term introduces correlated errors on teams that systematically over/under-perform their run differential. Plain RidgeCV on the full feature set generalises better.

## Next Sensible Versions

1. Investigate year-level features (league-average runs/game per season) to reduce the structural ~0.22 local/Kaggle gap.
2. Add Kaggle public leaderboard scores to the version table after each submission.