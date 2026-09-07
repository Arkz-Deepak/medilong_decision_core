"""
src/train.py - Project MediLong: ML Fusion & Clinical Safety Squad
Author: Deepak (Arkz-Deepak) — ML Fusion & Clinical Safety Squad (ML Engineer)
Copyright (c) 2026 Deepak (Arkz-Deepak). All rights reserved.
Licensed under the MIT License.

This script translates the exploratory notebook (notebooks/0_eda.ipynb) into a modular,
production-grade training and evaluation pipeline for the MediLong Decision Core.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import shap


# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_FILE = os.path.join(DATA_DIR, "PCOS_data_without_infertility.xlsx")


def load_and_preprocess_data(data_path: str = DATA_FILE):
    """
    Loads and cleans the Kaggle Indian PCOS dataset.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    # 1. Load data
    df = pd.read_excel(data_path, sheet_name="Full_new")

    # 2. Clean column headers
    df.columns = df.columns.str.strip()

    # 3. Drop non-predictive / identifier columns
    columns_to_drop = ['Sl. No', 'Patient File No.', 'Unnamed: 44']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

    # 4. Handle problematic object columns needing numeric conversion
    problematic_cols = ['II    beta-HCG(mIU/mL)', 'AMH(ng/mL)']
    for col in problematic_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 5. Compute and fill missing values with median
    feature_medians = df.median(numeric_only=True).to_dict()
    df = df.fillna(feature_medians)

    # 6. Separate features and target
    target_col = 'PCOS (Y/N)'
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col], axis=1)

    return X, y, feature_medians


def check_safety_guardrails(row):
    """
    Deterministic clinical safety guardrails based on FOGSI/Rotterdam criteria.
    Hypothyroidism and Hyperprolactinemia mimic PCOS symptoms and must trigger
    a differential diagnosis safety override.
    """
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]

    overrides = []
    # TSH > 4.5 uIU/mL indicates suspected hypothyroidism
    if row.get('TSH (mIU/L)', 0) > 4.5:
        overrides.append("OVERRIDE: TSH > 4.5 mIU/L (Suspected Hypothyroidism)")

    # Prolactin > 25.0 ng/mL indicates hyperprolactinemia
    if row.get('PRL(ng/mL)', 0) > 25.0:
        overrides.append("OVERRIDE: PRL > 25.0 ng/mL (Suspected Hyperprolactinemia)")

    return {
        "safety_override": len(overrides) > 0,
        "reasons": overrides
    }


def decision_core(patient_record: pd.DataFrame, explainer: shap.TreeExplainer, calibrated_model: CalibratedClassifierCV):
    """
    Core decision function that combines deterministic guardrails,
    calibrated probability classification, and SHAP explainability.
    """
    # 1. Run deterministic guardrails
    safety_override = check_safety_guardrails(patient_record)

    # 2. Predict calibrated probability (class 1: PCOS positive)
    risk_prob = float(calibrated_model.predict_proba(patient_record)[0, 1])

    # 3. Categorize risk
    if risk_prob < 0.30:
        risk_category = "Low"
    elif risk_prob <= 0.65:
        risk_category = "Moderate"
    else:
        risk_category = "High"

    # 4. Extract SHAP Local Feature Contributions
    shap_vals = explainer(patient_record)
    contributions = [
        {"feature": feat, "value": float(val) if not pd.isna(val) else None, "contribution": float(s_val)}
        for feat, val, s_val in zip(patient_record.columns, shap_vals.data[0], shap_vals.values[0])
    ]

    # Sort positive (risk enhancers) and negative (risk reducers/protective)
    positive_contribs = sorted(
        [c for c in contributions if c["contribution"] > 0],
        key=lambda x: x["contribution"],
        reverse=True
    )
    negative_contribs = sorted(
        [c for c in contributions if c["contribution"] < 0],
        key=lambda x: x["contribution"]
    )

    # Top 2 positive and top 2 negative features
    top_shap = []
    if positive_contribs:
        top_shap.append(positive_contribs[:2])
    if negative_contribs:
        top_shap.append(negative_contribs[:2])

    payload = {
        "risk_probability": round(risk_prob, 4),
        "risk_category": risk_category,
        "safety_override": safety_override,
        "shap_contributions": top_shap
    }

    return payload


def train_and_evaluate():
    """
    Executes model training, calibration, evaluation, artifact saving, and test runs.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=" * 60)
    print("Project MediLong - Decision Core Model Training & Calibration")
    print("=" * 60)

    # 1. Preprocessing
    print(f"[*] Loading dataset from: {DATA_FILE}")
    X, y, medians = load_and_preprocess_data()
    feature_names = list(X.columns)
    print(f"[*] Total dataset size: {len(X)} samples, {len(feature_names)} features")
    print(f"[*] Class distribution: Non-PCOS (0) = {(y == 0).sum()}, PCOS (1) = {(y == 1).sum()}")

    # 2. Train / Test Split
    print("[*] Splitting dataset (80% train, 20% test, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=44, stratify=y
    )
    print(f"[*] Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # 3. Model Definition & Calibration
    print("[*] Training base XGBoost classifier (n_estimators=100, max_depth=4)...")
    base_xgb = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        eval_metric='logloss',
        random_state=42
    )

    print("[*] Applying Isotonic Probability Calibration (5-fold CV)...")
    calibrated_model = CalibratedClassifierCV(base_xgb, method='isotonic', cv=5)
    calibrated_model.fit(X_train, y_train)

    # Also fit base XGBoost standalone for SHAP TreeExplainer
    base_xgb.fit(X_train, y_train)

    # 4. Evaluation
    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS (Test Set)")
    print("=" * 60)
    y_pred = calibrated_model.predict(X_test)
    y_prob = calibrated_model.predict_proba(X_test)[:, 1]

    clf_report = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(clf_report)
    print(f"ROC-AUC Score: {roc_auc:.4f} ({roc_auc})")

    # 5. Fit SHAP Explainer
    print("\n[*] Initializing SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(base_xgb)

    # 6. Test Decision Core on Test Samples
    print("\n" + "=" * 60)
    print("DECISION CORE VERIFICATION")
    print("=" * 60)
    sample_patient = X_test.iloc[[0]]
    res_1 = decision_core(sample_patient, explainer, calibrated_model)
    print("\n--- Test Case 1: Standard Sample Inference ---")
    print(json.dumps(res_1, indent=2))

    override_patient = sample_patient.copy()
    override_patient['TSH (mIU/L)'] = 6.2
    res_2 = decision_core(override_patient, explainer, calibrated_model)
    print("\n--- Test Case 2: Guardrail Override Test (TSH=6.2) ---")
    print(json.dumps(res_2, indent=2))

    # 7. Persist Artifacts
    print("\n[*] Saving model artifacts and metadata to 'models/' directory...")
    joblib.dump(calibrated_model, os.path.join(MODELS_DIR, "calibrated_model.joblib"))
    joblib.dump(base_xgb, os.path.join(MODELS_DIR, "base_xgb.joblib"))

    with open(os.path.join(MODELS_DIR, "feature_names.json"), "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    with open(os.path.join(MODELS_DIR, "feature_medians.json"), "w", encoding="utf-8") as f:
        clean_medians = {k: float(v) for k, v in medians.items()}
        json.dump(clean_medians, f, indent=2)

    # Save metrics report
    metrics = {
        "dataset_size": len(X),
        "test_size": len(X_test),
        "train_size": len(X_train),
        "class_distribution": {"0": int((y == 0).sum()), "1": int((y == 1).sum())},
        "roc_auc": float(roc_auc),
        "classification_report": clf_report
    }
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("[OK] All artifacts successfully trained and saved!")
    return calibrated_model, base_xgb, explainer, metrics


if __name__ == "__main__":
    train_and_evaluate()
