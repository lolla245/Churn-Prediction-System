"""
Train an XGBoost churn prediction model on the Telco Customer Churn dataset,
compute SHAP values for interpretability, and save artifacts for the API.
"""

import os
import pandas as pd
import numpy as np
import joblib
import json
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from xgboost import XGBClassifier

DATA_PATH = "data/telco_churn.csv"
MODEL_DIR = "model"


def load_and_clean(path):
    df = pd.read_csv(path)

    # Drop ID column
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # TotalCharges has some blank strings -> convert to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def encode_features(df):
    """Label-encode all categorical columns, save the encoders for reuse."""
    encoders = {}
    df_enc = df.copy()

    categorical_cols = df_enc.select_dtypes(include=["object"]).columns.tolist()

    for col in categorical_cols:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        encoders[col] = {cls: int(idx) for idx, cls in enumerate(le.classes_)}

    return df_enc, encoders, categorical_cols


def main():
    print("Loading data...")
    df = load_and_clean(DATA_PATH)

    df_enc, encoders, categorical_cols = encode_features(df)

    X = df_enc.drop(columns=["Churn"])
    y = df_enc["Churn"]

    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training XGBoost model...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC: {auc:.4f}")
    print(classification_report(y_test, preds))

    # SHAP explainer (TreeExplainer works natively with XGBoost)
    print("Computing SHAP background values...")
    explainer = shap.TreeExplainer(model)

    # Save artifacts
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, f"{MODEL_DIR}/churn_model.joblib")
    joblib.dump(explainer, f"{MODEL_DIR}/shap_explainer.joblib")

    metadata = {
        "feature_names": feature_names,
        "categorical_cols": categorical_cols,
        "encoders": encoders,
        "metrics": {"accuracy": acc, "roc_auc": auc},
        # Sensible defaults for unseen categorical values
        "defaults": {
            col: df[col].iloc[0] if col in df.columns else None
            for col in df.columns
            if col != "Churn"
        },
    }

    with open(f"{MODEL_DIR}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print("Saved model, explainer, and metadata to /model")


if __name__ == "__main__":
    main()