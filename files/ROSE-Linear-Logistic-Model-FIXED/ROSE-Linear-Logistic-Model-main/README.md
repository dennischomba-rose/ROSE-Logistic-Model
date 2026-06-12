# ROSE-Linear-Logistic-Model

## Overview

Loan default prediction models for ROSE Women's Foundation borrowers, using traditional features (V1) and composite score-based features (V2).

**Status: Final model selection confirmed (Jan 2026).** All 20 models have been trained, evaluated, and validated against the master analysis workbook (`Model_Evaluation__ROSE_Logisitc.xlsx`). The two best models are committed to `models/v2/`.

## 🏆 Confirmed Best Models

| Rank | Model | ROC-AUC | KS Statistic | Accuracy | F1 | Features |
|---|---|---|---|---|---|---|
| #1 (discrimination) | **LightGBM V2 Feature Set C** | **0.6599** | 0.3149 | 0.6548 | 0.5246 | 10 |
| #1 (rank ordering) | **XGBoost V2 Feature Set A** | 0.6490 | **0.3582** | 0.6310 | 0.4746 | 4 |

- **LightGBM V2 Feature Set C** — best overall discrimination (ROC-AUC). Uses the 4 composite scores plus Age Group, Education, CRB Class, Living, Logic on Income, and Marital status.
- **XGBoost V2 Feature Set A** — best risk rank-ordering (KS statistic), which is the metric that matters for a lending scorecard. Uses **only the 4 composite scores**, making it the most interpretable and the recommended model for the production scorecard. Gains-chart validation showed strong monotone rank ordering across score bands.

Both models are saved (with their encoders, scaler, and feature lists) in:

```
models/v2/best_roc_auc__LightGBM_V2_Feature_Set_C.joblib
models/v2/best_ks__XGBoost_V2_Feature_Set_A.joblib
```

### Key finding: composite scores beat raw categoricals

V2 Feature Set A (composite scores only) outperforms all 5 V1 models on ROC-AUC for 4 of 5 algorithms. Distilling 10+ raw categorical features into four 0–100 scores gives the models cleaner signal *and* makes individual decisions explainable to loan officers.

### Scorecard insight (from gains-chart validation)

The XGBoost Feature Set A score separates borrowers into clean risk bands: the best band shows ~6% default probability while approving 26% of good accounts; the worst bands show 76–92% default probability. A slight non-monotonicity exists at band 7 of Feature Set C, but those accounts fall below any realistic approval cut-off.

### CRB insight

CRB class alone is a weak and counterintuitive signal in this population: `Legacy_Gd` borrowers had the **highest** observed default rate (43.5%), `Legacy_Dflt` 42.6%, while `Active_L_M` had the lowest (27.1%) and unscored/new borrowers 30.9%. Lacking a credit history should not be penalised in this lending context.

## Repository Structure

```
├── notebooks/
│   ├── eda.ipynb                                     # Exploratory Data Analysis
│   ├── loan_default_models.ipynb                     # V1 Models (12 traditional features)
│   ├── loan_default_models_v2_composite_scores.ipynb # V2 Models (composite scores)
│   └── model_evaluation.ipynb                        # Comprehensive model evaluation
├── scripts/
│   ├── train_and_evaluate.py                         # ✅ Reproducible pipeline: trains all 20
│   │                                                 #    models, regenerates results CSVs,
│   │                                                 #    saves the two best models
│   └── predict.py                                    # Score new applicants with a saved model
├── models/
│   ├── model_comparison.csv                          # V1 results (regenerated, confirmed)
│   ├── model_evaluation_results.csv                  # All 20 models, ranked by ROC-AUC
│   └── v2/
│       ├── best_roc_auc__LightGBM_V2_Feature_Set_C.joblib
│       └── best_ks__XGBoost_V2_Feature_Set_A.joblib
└── Github Original Data.csv                          # Dataset (559 borrowers, 101 columns)
```

## Reproducing the results

```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost imbalanced-learn scipy joblib

python scripts/train_and_evaluate.py
```

This trains all 20 models (5 algorithms × {V1, V2-A, V2-B, V2-C}), prints the full leaderboard, writes `models/model_evaluation_results.csv` and `models/model_comparison.csv`, and saves the two best model bundles. All numbers are reproducible with `random_state=42` and match the confirmed analysis workbook exactly.

## Scoring new applicants

```bash
python scripts/predict.py models/v2/best_ks__XGBoost_V2_Feature_Set_A.joblib new_applicants.csv
```

The input CSV needs the same raw columns as the training data (the script generates intermediate features and composite scores automatically). Output: `predictions.csv` with probability of default per applicant.

## Methodology

1. Load dataset (559 borrowers, 38.6% default rate)
2. Generate intermediate features and 4 composite scores (see below)
3. Stratified 70/15/15 train/val/test split (`random_state=42`)
4. Standard-scale features; apply SMOTE to the training set only
5. Train 5 algorithms per feature set with consistent hyperparameters
6. Evaluate on the held-out test set: Accuracy, Precision, Recall, F1, ROC-AUC, KS

> ⚠️ Note on the preprocessing of intermediate features: the confirmed run (the one matching the master workbook) uses the intermediate-feature definitions in `model_evaluation.ipynb`, which differ subtly from the V2 notebook (the `Savings` column rather than `Savings Categorical`, and a `Regular Income` column that is absent from the data). `scripts/train_and_evaluate.py` replicates the confirmed run exactly and documents these choices inline. A future improvement is to switch to `Savings Categorical` and `Regular Income Brackets` and re-validate.

## Composite Scores (V2 models)

1. **Financial Resilience Score (0–100)** — Extra Income (35%), Expense Ratio (30%), Income Diversity (20%), Savings (15%)
2. **Business Quality Score (0–100)** — Rent Payment (45%), Utility Expenses (30%), Business Affordability (25%)
3. **Stability Score (0–100)** — School Fees (40%), Regular Income (30%), Income Streams (30%)
4. **Expense Management Score (0–100)** — Expense Ratio (50%), Affordability HH (35%), Utility (15%)

## Feature Sets

- **V1 (12):** Extra Income Brackets, Rent Category, School Fees Category, Age Group, Education, Loan Access, CRB Class, Income Diversity, Utility Category, Expense Ratio, Affordability HH, Living
- **V2 Set A (4):** the 4 composite scores only ← *recommended for scorecard*
- **V2 Set B (8):** Set A + Age Group, Education, CRB Class, Living
- **V2 Set C (10):** Set B + Logic on Income, Marital status ← *best ROC-AUC*

## Full Leaderboard (test set, confirmed)

See `models/model_evaluation_results.csv`. Top 5 by ROC-AUC:

| Model | ROC-AUC | KS |
|---|---|---|
| LightGBM V2 Feature Set C | 0.6599 | 0.3149 |
| XGBoost V2 Feature Set A | 0.6490 | 0.3582 |
| CatBoost V2 Feature Set A | 0.6340 | 0.2740 |
| LightGBM V2 Feature Set A | 0.6280 | 0.3053 |
| CatBoost V2 Feature Set B | 0.6256 | 0.2308 |

## Reproducibility

All models use `random_state=42`. The same preprocessing, split, and SMOTE application are used across all 20 models, so results are directly comparable.

## License

[Add license information]

## Contributors

[Add contributor information]
