"""
ROSE Women's Foundation - Loan Default Prediction
==================================================
Loads either of the two confirmed best models and scores new applicants.

Best models (saved in models/v2/ by scripts/train_and_evaluate.py):
  1. best_roc_auc__LightGBM_V2_Feature_Set_C.joblib
     - Best discrimination overall (ROC-AUC 0.6599)
     - Needs: 4 composite scores + Age Group, Education, CRB Class,
       Living, Logic on Income, Marital status
  2. best_ks__XGBoost_V2_Feature_Set_A.joblib
     - Best risk rank-ordering (KS 0.3582) - recommended for the
       lending scorecard since it uses ONLY the 4 composite scores.

Usage:
    python scripts/predict.py models/v2/best_ks__XGBoost_V2_Feature_Set_A.joblib data.csv
"""

import sys

import joblib
import pandas as pd

from train_and_evaluate import generate_composites, prepare_intermediates


def predict(model_path, data_path):
    bundle = joblib.load(model_path)
    model = bundle["model"]
    encoders = bundle["encoders"]
    scaler = bundle["scaler"]
    features = bundle["features"]

    df = pd.read_csv(data_path, encoding="latin-1")
    df = prepare_intermediates(df)
    df = generate_composites(df)

    X = df[features].copy()
    for col in features:
        if col in encoders:
            le = encoders[col]
            known = set(le.classes_)
            X[col] = X[col].astype(str).map(
                lambda v: v if v in known else le.classes_[0])
            X[col] = le.transform(X[col])
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(X[col].median())

    X_scaled = scaler.transform(X)
    prob_default = model.predict_proba(X_scaled)[:, 1]
    out = df[["ID Number"]].copy() if "ID Number" in df.columns else pd.DataFrame(index=df.index)
    out["Probability_of_Default"] = prob_default
    out["Predicted_Default"] = (prob_default >= 0.5).astype(int)
    return out


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    result = predict(sys.argv[1], sys.argv[2])
    print(result.to_string(index=False))
    result.to_csv("predictions.csv", index=False)
    print("\nSaved to predictions.csv")
