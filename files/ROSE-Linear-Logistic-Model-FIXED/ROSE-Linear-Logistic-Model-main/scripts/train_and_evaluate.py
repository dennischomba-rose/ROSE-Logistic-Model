"""
ROSE Women's Foundation - Loan Default Model Training & Evaluation
===================================================================
Reproduces the confirmed analysis pipeline (validated against
Model_Evaluation__ROSE_Logisitc.xlsx, Jan 2026):

- Trains all 20 models (5 V1 + 15 V2 across feature sets A/B/C)
- Evaluates 6 metrics: Accuracy, Precision, Recall, F1, ROC-AUC, KS
- Saves the two confirmed BEST models:
    1. LightGBM V2 Feature Set C  -> best ROC-AUC
    2. XGBoost  V2 Feature Set A  -> best KS statistic
- Regenerates models/model_comparison.csv and
  models/model_evaluation_results.csv from the actual run.

Usage:
    python scripts/train_and_evaluate.py
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TARGET = "Defaulted"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_PATH = os.path.join(ROOT, "Github Original Data.csv")
MODELS_DIR = os.path.join(ROOT, "models")
V2_DIR = os.path.join(MODELS_DIR, "v2")
os.makedirs(V2_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------
V1_FEATURES = [
    "Extra Income Brackets", "Categorize Rent Payment",
    "School Fees Categorical", "Age Group", "Education", "Loan Access",
    "CRB Class", "Logic on Income", "Categorizing Utility Expenses",
    "Expense Relative to Income", "Affordability (HH)", "Living",
]
COMPOSITES = [
    "Financial_Resilience_Score", "Business_Quality_Score",
    "Stability_Score", "Expense_Management_Score",
]
FEATURES_A = COMPOSITES
FEATURES_B = COMPOSITES + ["Age Group", "Education", "CRB Class", "Living"]
FEATURES_C = FEATURES_B + ["Logic on Income", "Marital status"]


# ---------------------------------------------------------------------------
# Composite score functions (from eda.ipynb Section 9.6)
# ---------------------------------------------------------------------------
def calculate_financial_resilience(row):
    score = 0
    extra_income = str(row.get("Extra_Income_Brackets", "")).lower()
    if "moderate" in extra_income or "high" in extra_income:
        score += 35 * 1.0
    elif "low" in extra_income and "no" not in extra_income:
        score += 35 * 0.3
    else:
        score += 35 * 0.6
    expense_ratio = str(row.get("Expense_Ratio", "")).lower()
    if "1/3" in expense_ratio:
        score += 30 * 1.0
    elif "half" in expense_ratio:
        score += 30 * 0.7
    elif "2/3" in expense_ratio and "more" not in expense_ratio:
        score += 30 * 0.4
    else:
        score += 30 * 0.5
    income_div = str(row.get("Income_Diversity", "")).lower()
    if "full" in income_div:
        score += 20 * 1.0
    elif "regular" in income_div:
        score += 20 * 0.7
    elif "extra" in income_div:
        score += 20 * 0.5
    else:
        score += 20 * 0.6
    savings = str(row.get("Savings_Category", "")).lower()
    if "high" in savings:
        score += 15 * 1.0
    elif "low" in savings and "no" not in savings:
        score += 15 * 0.8
    else:
        score += 15 * 0.85
    return score


def calculate_business_quality(row):
    score = 0
    rent = str(row.get("Rent_Category", "")).lower()
    if "high" in rent:
        score += 45 * 1.0
    elif "low" in rent and "no" not in rent:
        score += 45 * 0.5
    else:
        score += 45 * 0.6
    utility = str(row.get("Utility_Category", "")).lower()
    if "high" in utility:
        score += 30 * 1.0
    elif "low" in utility and "no" not in utility:
        score += 30 * 0.5
    else:
        score += 30 * 0.7
    afford = str(row.get("Affordability_Business", "")).lower()
    if "profitable" in afford:
        score += 25 * 1.0
    else:
        score += 25 * 0.5
    return score


def calculate_stability(row):
    score = 0
    school = str(row.get("SchoolFees_Category", "")).lower()
    if "high" in school:
        score += 40 * 1.0
    elif "low" in school and "no" not in school:
        score += 40 * 0.5
    else:
        score += 40 * 0.9
    regular = str(row.get("Regular_Income_Brackets", "")).lower()
    if "moderate" in regular or "high" in regular:
        score += 30 * 1.0
    elif "low" in regular and "no" not in regular:
        score += 30 * 1.1
    else:
        score += 30 * 0.85
    income_div = str(row.get("Income_Diversity", "")).lower()
    if "full" in income_div:
        score += 30 * 1.0
    elif "regular" in income_div:
        score += 30 * 0.8
    elif "extra" in income_div:
        score += 30 * 0.6
    else:
        score += 30 * 0.7
    return min(score, 100)


def calculate_expense_management(row):
    score = 0
    expense_ratio = str(row.get("Expense_Ratio", "")).lower()
    if "1/3" in expense_ratio:
        score += 50 * 1.0
    elif "half" in expense_ratio:
        score += 50 * 0.7
    elif "2/3" in expense_ratio and "more" not in expense_ratio:
        score += 50 * 0.4
    else:
        score += 50 * 0.5
    afford = str(row.get("Affordability_HH", "")).lower()
    if "profitable" in afford:
        score += 35 * 1.0
    else:
        score += 35 * 0.5
    utility = str(row.get("Utility_Category", "")).lower()
    if "high" in utility:
        score += 15 * 1.0
    elif "low" in utility and "no" not in utility:
        score += 15 * 0.5
    else:
        score += 15 * 0.7
    return score


def prepare_intermediates(df):
    """Intermediate features exactly as in model_evaluation.ipynb (the run
    that produced the confirmed results in Model_Evaluation__ROSE_Logisitc.xlsx)."""
    df["Affordability_Business"] = df.get(
        "Affordability", pd.Series("Unknown", index=df.index)).fillna("Unknown")
    df["Affordability_HH"] = df.get(
        "Affordability (HH)", pd.Series("Unknown", index=df.index)).fillna("Unknown")
    df["Extra_Income_Brackets"] = df.get(
        "Extra Income Brackets",
        pd.Series("No Extra Income", index=df.index)).fillna("No Extra Income")
    df["Expense_Ratio"] = df.get(
        "Expense Relative to Income",
        pd.Series("Unknown", index=df.index)).fillna("Unknown")
    df["Income_Diversity"] = df.get(
        "Logic on Income", pd.Series("Unknown", index=df.index)).fillna("Unknown")
    # NOTE: the confirmed run used the (empty) 'Savings' column, not
    # 'Savings Categorical' -> every row becomes 'No Savings'.
    df["Savings_Category"] = df.get(
        "Savings", pd.Series("No Savings", index=df.index)).fillna("No Savings")
    df["Rent_Category"] = df.get(
        "Categorize Rent Payment",
        pd.Series("No Rent", index=df.index)).fillna("No Rent")
    df["Utility_Category"] = df.get(
        "Categorizing Utility Expenses",
        pd.Series("None", index=df.index)).fillna("None")
    df["SchoolFees_Category"] = df.get(
        "School Fees Categorical",
        pd.Series("None", index=df.index)).fillna("None")
    # NOTE: the confirmed run looked for 'Regular Income' (absent) -> 'Unknown'.
    if "Regular Income" in df.columns:
        regular = pd.to_numeric(df["Regular Income"], errors="coerce").fillna(0)
        df["Regular_Income_Brackets"] = np.where(
            regular == 0, "No Regular Income",
            np.where(regular < 10000, "Low Regular Income",
                     np.where(regular < 20000, "Moderate Regular Income",
                              "High Regular Income")))
    else:
        df["Regular_Income_Brackets"] = "Unknown"
    return df


def generate_composites(df):
    df["Financial_Resilience_Score"] = df.apply(calculate_financial_resilience, axis=1)
    df["Business_Quality_Score"] = df.apply(calculate_business_quality, axis=1)
    df["Stability_Score"] = df.apply(calculate_stability, axis=1)
    df["Expense_Management_Score"] = df.apply(calculate_expense_management, axis=1)
    return df


def calculate_ks(y_true, y_prob):
    ks, _ = stats.ks_2samp(y_prob[y_true == 1], y_prob[y_true == 0])
    return ks


def prepare_data(df, features):
    available = [f for f in features if f in df.columns]
    work = df[available + [TARGET]].copy()
    is_cat = {c: (work[c].dtype == "object"
                  or pd.api.types.is_string_dtype(work[c])) for c in available}
    for col in available:
        if is_cat[col]:
            mode = work[col].mode()
            work[col] = work[col].fillna(mode[0] if len(mode) else "Unknown")
        else:
            work[col] = work[col].fillna(work[col].median())
    encoders = {}
    for col in available:
        if is_cat[col]:
            le = LabelEncoder()
            work[col] = le.fit_transform(work[col].astype(str))
            encoders[col] = le
    X, y = work[available], work[TARGET]
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15 / 0.85,
        random_state=RANDOM_STATE, stratify=y_temp)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    X_tr_sm, y_tr_sm = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train_s, y_train)
    return X_tr_sm, X_test_s, y_tr_sm, y_test, encoders, scaler, available


def make_models(suffix):
    return {
        f"Logistic Regression {suffix}": LogisticRegression(
            random_state=RANDOM_STATE, max_iter=1000, C=0.1),
        f"Random Forest {suffix}": RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1),
        f"XGBoost {suffix}": XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, eval_metric="logloss"),
        f"LightGBM {suffix}": LGBMClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, verbose=-1),
        f"CatBoost {suffix}": CatBoostClassifier(
            iterations=100, depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, verbose=False),
    }


def evaluate(model, X_test, y_test, name):
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred),
        "F1-Score": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, prob),
        "KS Statistic": calculate_ks(y_test.values, prob),
    }


def main():
    df = pd.read_csv(DATA_PATH, encoding="latin-1")
    print(f"Dataset: {df.shape}, default rate {df[TARGET].mean()*100:.1f}%")
    df = prepare_intermediates(df)
    df = generate_composites(df)

    results, trained = [], {}

    print("\n--- V1 (12 traditional features) ---")
    Xtr, Xte, ytr, yte, enc, sc, feats = prepare_data(df, V1_FEATURES)
    bundles = {"V1": (Xtr, Xte, ytr, yte, enc, sc, feats)}
    for name, model in make_models("V1").items():
        model.fit(Xtr, ytr)
        m = evaluate(model, Xte, yte, name)
        results.append(m)
        trained[name] = (model, "V1")
        print(f"  {name}: ROC-AUC={m['ROC-AUC']:.4f} KS={m['KS Statistic']:.4f}")

    for set_name, feats_def in [("A", FEATURES_A), ("B", FEATURES_B), ("C", FEATURES_C)]:
        print(f"\n--- V2 Feature Set {set_name} ({len(feats_def)} features) ---")
        Xtr, Xte, ytr, yte, enc, sc, feats = prepare_data(df, feats_def)
        bundles[set_name] = (Xtr, Xte, ytr, yte, enc, sc, feats)
        for name, model in make_models(f"V2 Feature Set {set_name}").items():
            model.fit(Xtr, ytr)
            m = evaluate(model, Xte, yte, name)
            results.append(m)
            trained[name] = (model, set_name)
            print(f"  {name}: ROC-AUC={m['ROC-AUC']:.4f} KS={m['KS Statistic']:.4f}")

    res = pd.DataFrame(results).round(4).sort_values("ROC-AUC", ascending=False)
    res.to_csv(os.path.join(MODELS_DIR, "model_evaluation_results.csv"), index=False)
    res[res["Model"].str.endswith("V1")].drop(columns=[]).to_csv(
        os.path.join(MODELS_DIR, "model_comparison.csv"), index=False)

    best_auc = res.iloc[0]
    best_ks = res.loc[res["KS Statistic"].idxmax()]
    print("\n" + "=" * 70)
    print(f"BEST ROC-AUC : {best_auc['Model']}  "
          f"(AUC={best_auc['ROC-AUC']:.4f}, KS={best_auc['KS Statistic']:.4f})")
    print(f"BEST KS      : {best_ks['Model']}  "
          f"(AUC={best_ks['ROC-AUC']:.4f}, KS={best_ks['KS Statistic']:.4f})")
    print("=" * 70)

    for label, row in [("best_roc_auc", best_auc), ("best_ks", best_ks)]:
        name = row["Model"]
        model, set_key = trained[name]
        _, _, _, _, enc, sc, feats = bundles[set_key]
        out = os.path.join(V2_DIR, f"{label}__{name.replace(' ', '_')}.joblib")
        joblib.dump({"model": model, "encoders": enc, "scaler": sc,
                     "features": feats, "metrics": row.to_dict()}, out)
        print(f"Saved {label}: {out}")

    return res


if __name__ == "__main__":
    main()
