"""
src/decision_core.py - Project MediLong: Unified Decision Core Service
Author: Deepak (Arkz-Deepak) — ML Fusion & Clinical Safety Squad (ML Engineer)
Copyright (c) 2026 Deepak (Arkz-Deepak). All rights reserved.
Licensed under the MIT License.

This module implements the complete Decision Core contract slice for Project MediLong.
It is built to integrate seamlessly with:
  1. Backend & Cloud Squad (FastAPI /predict orchestration, Firebase persistence, signed PDF summary)
  2. Frontend & UI/UX Squad (SHAP waterfall/bar charts, safety override banners, stub-mode disclosures)
  3. Data Pipelines Squad (Pillar 2 Ultrasound CV Stub, Pillar 3 Lab OCR/NLP 10 biomarkers)

Contract Slice Owned:
  decision_core (risk_probability, risk_category, safety_override, shap_contributions)
"""

import os
import json
import time
import hashlib
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field

import shap
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

# Authorship provenance and engine build telemetry signature
__author__ = "Deepak (Arkz-Deepak)"
__copyright__ = "Copyright (c) 2026 Deepak (Arkz-Deepak)"
__license__ = "MIT"
__engine_signature__ = "arkz-ml-decision-v1.0.4-8f2c7a"
__engine_id__ = f"arkz-core-{hashlib.sha256(b'Arkz-Deepak-ML-Fusion-MediLong').hexdigest()[:12]}"


# Directory setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


# ==============================================================================
# Pydantic Schemas for Multi-Pillar Contract & Decision Core Slice
# ==============================================================================

class Pillar1Demographics(BaseModel):
    """Pillar 1: Patient Demographics, Vitals, and Clinical History (Intake Form)"""
    age: Optional[float] = Field(None, description="Age in years")
    weight_kg: Optional[float] = Field(None, description="Weight in Kilograms")
    height_cm: Optional[float] = Field(None, description="Height in Centimeters")
    bmi: Optional[float] = Field(None, description="Body Mass Index (kg/m^2)")
    blood_group: Optional[int] = Field(None, description="Encoded Blood Group (11-18)")
    pulse_rate_bpm: Optional[float] = Field(None, description="Pulse rate in beats per minute")
    rr_breaths_min: Optional[float] = Field(None, description="Respiratory Rate in breaths/min")
    hb_g_dl: Optional[float] = Field(None, description="Hemoglobin in g/dl")
    cycle_ri: Optional[int] = Field(None, description="Cycle regularity: 2 (Regular), 4 (Irregular)")
    cycle_length_days: Optional[float] = Field(None, description="Menstrual cycle length in days")
    marriage_status_yrs: Optional[float] = Field(None, description="Years married")
    pregnant: Optional[int] = Field(None, description="Currently pregnant: 1 (Yes), 0 (No)")
    abortions_count: Optional[int] = Field(None, description="Number of abortions")
    hip_inch: Optional[float] = Field(None, description="Hip measurement in inches")
    waist_inch: Optional[float] = Field(None, description="Waist measurement in inches")
    waist_hip_ratio: Optional[float] = Field(None, description="Waist to Hip Ratio")
    weight_gain: Optional[int] = Field(None, description="Weight gain symptom: 1 (Yes), 0 (No)")
    hair_growth: Optional[int] = Field(None, description="Excessive hair growth (Hirsutism): 1 (Yes), 0 (No)")
    skin_darkening: Optional[int] = Field(None, description="Acanthosis Nigricans / Skin darkening: 1 (Yes), 0 (No)")
    hair_loss: Optional[int] = Field(None, description="Hair thinning / loss: 1 (Yes), 0 (No)")
    pimples: Optional[int] = Field(None, description="Acne / Pimples: 1 (Yes), 0 (No)")
    fast_food: Optional[float] = Field(None, description="Frequent fast food consumption: 1 (Yes), 0 (No)")
    reg_exercise: Optional[int] = Field(None, description="Regular physical exercise: 1 (Yes), 0 (No)")
    bp_systolic: Optional[float] = Field(None, description="Systolic Blood Pressure (mmHg)")
    bp_diastolic: Optional[float] = Field(None, description="Diastolic Blood Pressure (mmHg)")


class Pillar2Ultrasound(BaseModel):
    """Pillar 2: Ultrasound Computer Vision Analysis / CV Stub"""
    mode: str = Field("stub", description="Operational mode: 'stub' or 'live'")
    follicle_count: Optional[float] = Field(None, description="Total detected ovarian follicles")
    follicle_no_l: Optional[float] = Field(None, description="Left Ovary Follicle Count")
    follicle_no_r: Optional[float] = Field(None, description="Right Ovary Follicle Count")
    avg_f_size_l_mm: Optional[float] = Field(None, description="Average follicle size left (mm)")
    avg_f_size_r_mm: Optional[float] = Field(None, description="Average follicle size right (mm)")
    endometrium_mm: Optional[float] = Field(None, description="Endometrium thickness (mm)")
    volume_ml: Optional[float] = Field(None, description="Ovarian volume in ml")
    confidence: Optional[float] = Field(1.0, description="CV model confidence score (0.0 - 1.0)")


class Pillar3Labs(BaseModel):
    """Pillar 3: Lab Biomarkers extracted via OCR/NLP (10 key biomarkers)"""
    tsh_miu_l: Optional[float] = Field(None, description="Thyroid Stimulating Hormone (mIU/L)")
    prl_ng_ml: Optional[float] = Field(None, description="Prolactin (ng/mL)")
    amh_ng_ml: Optional[float] = Field(None, description="Anti-Müllerian Hormone (ng/mL)")
    fsh_miu_ml: Optional[float] = Field(None, description="Follicle Stimulating Hormone (mIU/mL)")
    lh_miu_ml: Optional[float] = Field(None, description="Luteinizing Hormone (mIU/mL)")
    fsh_lh_ratio: Optional[float] = Field(None, description="FSH / LH ratio")
    vit_d3_ng_ml: Optional[float] = Field(None, description="Vitamin D3 (ng/mL)")
    prg_ng_ml: Optional[float] = Field(None, description="Progesterone (ng/mL)")
    rbs_mg_dl: Optional[float] = Field(None, description="Random Blood Sugar (mg/dL)")
    beta_hcg_i: Optional[float] = Field(None, description="I beta-HCG (mIU/mL)")
    beta_hcg_ii: Optional[float] = Field(None, description="II beta-HCG (mIU/mL)")
    source_confidence: Optional[float] = Field(1.0, description="OCR extraction source confidence score")


class UnifiedIntakeRequest(BaseModel):
    """
    Unified JSON contract payload ingested from Backend's /predict orchestration.
    Can accept either nested 3-pillar structures or a flat dictionary of raw features.
    """
    patient_id: Optional[str] = Field(None, description="Optional unique patient identifier")
    pillar_1_demographics: Optional[Pillar1Demographics] = None
    pillar_2_ultrasound: Optional[Pillar2Ultrasound] = None
    pillar_3_labs: Optional[Pillar3Labs] = None
    raw_features: Optional[Dict[str, Any]] = Field(
        None, description="Direct tabular features dictionary (for direct ML testing)"
    )


class SafetyOverrideResponse(BaseModel):
    """Deterministic Clinical Safety Guardrail Result"""
    safety_override: bool = Field(..., description="True if a hard clinical rule triggered an override")
    reasons: List[str] = Field(default_factory=list, description="List of clinical override reasons")
    recommendations: List[str] = Field(
        default_factory=list, description="Recommended clinical differential next steps"
    )


class ShapContributionItem(BaseModel):
    """Individual feature contribution item for Frontend waterfall/bar chart"""
    feature: str = Field(..., description="Clinical feature name")
    value: Optional[float] = Field(None, description="Patient's actual feature value")
    contribution: float = Field(..., description="SHAP value (impact on model log-odds)")
    impact_direction: str = Field(
        ..., description="'increases_risk' (positive SHAP) or 'decreases_risk' (negative SHAP)"
    )


class DecisionCoreResponse(BaseModel):
    """
    Final contract slice owned by ML Fusion Squad:
    decision_core (risk_probability, risk_category, safety_override, shap_contributions)
    """
    risk_probability: float = Field(
        ..., description="Calibrated risk probability between 0.0000 and 1.0000 (Isotonic Regression)"
    )
    risk_category: str = Field(
        ..., description="Risk category classification: 'Low' (<0.30), 'Moderate' (0.30-0.65), 'High' (>0.65)"
    )
    safety_override: SafetyOverrideResponse = Field(
        ..., description="Pre-model deterministic clinical safety guardrail evaluation"
    )
    shap_contributions: List[List[Dict[str, Any]]] = Field(
        ..., description="Notebook-compatible top positive and negative SHAP contributions"
    )
    waterfall_items: List[ShapContributionItem] = Field(
        ..., description="Structured flat list of top SHAP items for Frontend waterfall chart rendering"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Metadata including latency, pillar 2 stub flag, and calibration"
    )


# ==============================================================================
# Clinical Safety Guardrail Layer (FOGSI / Rotterdam Criteria)
# ==============================================================================

class ClinicalGuardrails:
    """
    Deterministic clinical safety guardrails engine based on FOGSI / Rotterdam guidelines.
    Executes BEFORE model inference to catch conditions that mimic PCOS symptoms
    (such as Hypothyroidism and Hyperprolactinemia) and issues hard clinical overrides.
    """

    @staticmethod
    def evaluate(record: Union[pd.DataFrame, pd.Series, Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(record, pd.DataFrame):
            data = record.iloc[0].to_dict()
        elif isinstance(record, pd.Series):
            data = record.to_dict()
        else:
            data = record

        overrides = []
        recommendations = []

        # 1. TSH Check (Thyroid Stimulating Hormone)
        # FOGSI Rule: TSH > 4.5 uIU/mL indicates Hypothyroidism (must exclude before PCOS diagnosis)
        tsh_val = data.get('TSH (mIU/L)')
        if tsh_val is not None and not pd.isna(tsh_val):
            try:
                if float(tsh_val) > 4.5:
                    overrides.append("OVERRIDE: TSH > 4.5 mIU/L (Suspected Hypothyroidism)")
                    recommendations.append(
                        "Order full Thyroid Profile (Free T3, Free T4, Anti-TPO) to rule out primary hypothyroidism."
                    )
            except (ValueError, TypeError):
                pass

        # 2. PRL Check (Prolactin)
        # FOGSI Rule: Prolactin > 25.0 ng/mL indicates Hyperprolactinemia (can induce amenorrhea/anovulation)
        prl_val = data.get('PRL(ng/mL)')
        if prl_val is not None and not pd.isna(prl_val):
            try:
                if float(prl_val) > 25.0:
                    overrides.append("OVERRIDE: PRL > 25.0 ng/mL (Suspected Hyperprolactinemia)")
                    recommendations.append(
                        "Repeat serum Prolactin fasting levels; consider pituitary MRI if persistently elevated."
                    )
            except (ValueError, TypeError):
                pass

        # 3. Pregnancy / Beta-HCG Check (amenorrhea differential)
        beta_hcg_i = data.get('I   beta-HCG(mIU/mL)')
        beta_hcg_ii = data.get('II    beta-HCG(mIU/mL)')
        for b_val in [beta_hcg_i, beta_hcg_ii]:
            if b_val is not None and not pd.isna(b_val):
                try:
                    if float(b_val) > 25.0:
                        overrides.append("OVERRIDE: Beta-HCG elevated (>25 mIU/mL, Pregnancy suspected)")
                        recommendations.append(
                            "Rule out pregnancy via confirmatory urine pregnancy test and obstetric evaluation."
                        )
                        break
                except (ValueError, TypeError):
                    pass

        has_override = len(overrides) > 0

        return {
            "safety_override": has_override,
            "reasons": overrides,
            "recommendations": recommendations
        }


# ==============================================================================
# Decision Core Engine (Model Ingestion, Calibration & SHAP)
# ==============================================================================

class DecisionCoreService:
    """
    Singleton / callable service managing model loading, feature fusion across 3 pillars,
    guardrail verification, calibrated inference, and SHAP explainability.
    """

    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        self.calibrated_model: Optional[CalibratedClassifierCV] = None
        self.base_xgb: Optional[XGBClassifier] = None
        self.explainer: Optional[shap.TreeExplainer] = None
        self.feature_names: List[str] = []
        self.feature_medians: Dict[str, float] = {}
        self.load_artifacts()

    def load_artifacts(self):
        """Loads trained models and feature metadata from models directory."""
        calibrated_path = os.path.join(self.models_dir, "calibrated_model.joblib")
        base_xgb_path = os.path.join(self.models_dir, "base_xgb.joblib")
        feat_names_path = os.path.join(self.models_dir, "feature_names.json")
        medians_path = os.path.join(self.models_dir, "feature_medians.json")

        if not all(os.path.exists(p) for p in [calibrated_path, base_xgb_path, feat_names_path, medians_path]):
            # If not found, train on-the-fly via src.train
            from src.train import train_and_evaluate
            print("[*] Model artifacts not found. Training model now...")
            self.calibrated_model, self.base_xgb, self.explainer, _ = train_and_evaluate()
            with open(feat_names_path, "r", encoding="utf-8") as f:
                self.feature_names = json.load(f)
            with open(medians_path, "r", encoding="utf-8") as f:
                self.feature_medians = json.load(f)
            return

        self.calibrated_model = joblib.load(calibrated_path)
        self.base_xgb = joblib.load(base_xgb_path)
        with open(feat_names_path, "r", encoding="utf-8") as f:
            self.feature_names = json.load(f)
        with open(medians_path, "r", encoding="utf-8") as f:
            self.feature_medians = json.load(f)

        self.explainer = shap.TreeExplainer(self.base_xgb)

    def parse_request_to_dataframe(self, request: Union[UnifiedIntakeRequest, Dict[str, Any]]) -> pd.DataFrame:
        """
        Maps nested 3-pillar inputs or flat feature dictionary into the 41-feature DataFrame
        expected by the calibrated XGBoost model. Fills any missing or null values
        with dataset median baselines.
        """
        # Start with default median values for all 41 features
        row_data = dict(self.feature_medians)

        # 1. Direct flat features if provided
        raw = None
        if isinstance(request, UnifiedIntakeRequest):
            if request.raw_features:
                raw = request.raw_features
        elif isinstance(request, dict) and "raw_features" in request and request["raw_features"]:
            raw = request["raw_features"]
        elif isinstance(request, dict) and any(col in request for col in self.feature_names):
            raw = request

        if raw:
            for k, v in raw.items():
                if k in self.feature_names and v is not None and not pd.isna(v):
                    try:
                        row_data[k] = float(v)
                    except (ValueError, TypeError):
                        pass
            df = pd.DataFrame([row_data], columns=self.feature_names)
            return df

        # 2. Extract from 3 Pillars
        p1 = None
        p2 = None
        p3 = None

        if isinstance(request, UnifiedIntakeRequest):
            p1 = request.pillar_1_demographics
            p2 = request.pillar_2_ultrasound
            p3 = request.pillar_3_labs
        elif isinstance(request, dict):
            if "pillar_1_demographics" in request and request["pillar_1_demographics"]:
                p1 = Pillar1Demographics(**request["pillar_1_demographics"])
            if "pillar_2_ultrasound" in request and request["pillar_2_ultrasound"]:
                p2 = Pillar2Ultrasound(**request["pillar_2_ultrasound"])
            if "pillar_3_labs" in request and request["pillar_3_labs"]:
                p3 = Pillar3Labs(**request["pillar_3_labs"])

        # Map Pillar 1 (Demographics)
        if p1:
            mapping_p1 = {
                'Age (yrs)': p1.age,
                'Weight (Kg)': p1.weight_kg,
                'Height(Cm)': p1.height_cm,
                'BMI': p1.bmi,
                'Blood Group': p1.blood_group,
                'Pulse rate(bpm)': p1.pulse_rate_bpm,
                'RR (breaths/min)': p1.rr_breaths_min,
                'Hb(g/dl)': p1.hb_g_dl,
                'Cycle(R/I)': p1.cycle_ri,
                'Cycle length(days)': p1.cycle_length_days,
                'Marraige Status (Yrs)': p1.marriage_status_yrs,
                'Pregnant(Y/N)': p1.pregnant,
                'No. of aborptions': p1.abortions_count,
                'Hip(inch)': p1.hip_inch,
                'Waist(inch)': p1.waist_inch,
                'Waist:Hip Ratio': p1.waist_hip_ratio,
                'Weight gain(Y/N)': p1.weight_gain,
                'hair growth(Y/N)': p1.hair_growth,
                'Skin darkening (Y/N)': p1.skin_darkening,
                'Hair loss(Y/N)': p1.hair_loss,
                'Pimples(Y/N)': p1.pimples,
                'Fast food (Y/N)': p1.fast_food,
                'Reg.Exercise(Y/N)': p1.reg_exercise,
                'BP _Systolic (mmHg)': p1.bp_systolic,
                'BP _Diastolic (mmHg)': p1.bp_diastolic
            }
            # Auto-calculate BMI if missing but height and weight provided
            if mapping_p1['BMI'] is None and p1.weight_kg and p1.height_cm and p1.height_cm > 0:
                mapping_p1['BMI'] = round(p1.weight_kg / ((p1.height_cm / 100.0) ** 2), 2)
            # Auto-calculate Waist:Hip Ratio if missing
            if mapping_p1['Waist:Hip Ratio'] is None and p1.waist_inch and p1.hip_inch and p1.hip_inch > 0:
                mapping_p1['Waist:Hip Ratio'] = round(p1.waist_inch / p1.hip_inch, 3)

            for col, val in mapping_p1.items():
                if val is not None and not pd.isna(val):
                    row_data[col] = float(val)

        # Map Pillar 2 (Ultrasound CV Stub / Live)
        if p2:
            f_left = p2.follicle_no_l
            f_right = p2.follicle_no_r
            # If only total follicle_count is provided, split between left and right
            if (f_left is None or f_right is None) and p2.follicle_count is not None:
                f_left = float(p2.follicle_count // 2)
                f_right = float(p2.follicle_count - f_left)

            mapping_p2 = {
                'Follicle No. (L)': f_left,
                'Follicle No. (R)': f_right,
                'Avg. F size (L) (mm)': p2.avg_f_size_l_mm,
                'Avg. F size (R) (mm)': p2.avg_f_size_r_mm,
                'Endometrium (mm)': p2.endometrium_mm
            }
            for col, val in mapping_p2.items():
                if val is not None and not pd.isna(val):
                    row_data[col] = float(val)

        # Map Pillar 3 (Lab Biomarkers)
        if p3:
            fsh = p3.fsh_miu_ml
            lh = p3.lh_miu_ml
            fsh_lh = p3.fsh_lh_ratio
            if fsh_lh is None and fsh is not None and lh is not None and lh > 0:
                fsh_lh = round(fsh / lh, 2)

            mapping_p3 = {
                'TSH (mIU/L)': p3.tsh_miu_l,
                'PRL(ng/mL)': p3.prl_ng_ml,
                'AMH(ng/mL)': p3.amh_ng_ml,
                'FSH(mIU/mL)': fsh,
                'LH(mIU/mL)': lh,
                'FSH/LH': fsh_lh,
                'Vit D3 (ng/mL)': p3.vit_d3_ng_ml,
                'PRG(ng/mL)': p3.prg_ng_ml,
                'RBS(mg/dl)': p3.rbs_mg_dl,
                'I   beta-HCG(mIU/mL)': p3.beta_hcg_i,
                'II    beta-HCG(mIU/mL)': p3.beta_hcg_ii
            }
            for col, val in mapping_p3.items():
                if val is not None and not pd.isna(val):
                    row_data[col] = float(val)

        df = pd.DataFrame([row_data], columns=self.feature_names)
        return df

    def evaluate(self, request: Union[UnifiedIntakeRequest, Dict[str, Any], pd.DataFrame]) -> DecisionCoreResponse:
        """
        Executes the full Decision Core inference pipeline:
          1. Deterministic clinical safety guardrails (veto / override check)
          2. Multi-pillar feature fusion & robust median imputation
          3. Calibrated probability inference (Isotonic Regression)
          4. Three-tier clinical risk categorization ('Low', 'Moderate', 'High')
          5. Local SHAP feature attribution generation for UI waterfall & bar charts
        """
        start_time = time.time()

        # 1. Transform to 41-feature DataFrame
        if isinstance(request, pd.DataFrame):
            df_record = request[self.feature_names].copy() if all(c in request.columns for c in self.feature_names) else request
        else:
            df_record = self.parse_request_to_dataframe(request)

        # 2. Run deterministic clinical guardrails
        guardrail_raw = ClinicalGuardrails.evaluate(df_record)
        safety_override = SafetyOverrideResponse(
            safety_override=guardrail_raw["safety_override"],
            reasons=guardrail_raw["reasons"],
            recommendations=guardrail_raw["recommendations"]
        )

        # 3. Predict calibrated probability
        prob_raw = self.calibrated_model.predict_proba(df_record)[0, 1]
        risk_probability = round(float(prob_raw), 4)

        # 4. Categorize risk
        if risk_probability < 0.30:
            risk_category = "Low"
        elif risk_probability <= 0.65:
            risk_category = "Moderate"
        else:
            risk_category = "High"

        # 5. Compute SHAP feature attributions
        shap_vals = self.explainer(df_record)
        raw_values = shap_vals.data[0]
        impact_values = shap_vals.values[0]

        all_contributions: List[Dict[str, Any]] = []
        for feat, val, s_val in zip(self.feature_names, raw_values, impact_values):
            val_clean = float(val) if not pd.isna(val) else None
            s_clean = float(s_val)
            direction = "increases_risk" if s_clean > 0 else "decreases_risk"
            all_contributions.append({
                "feature": feat,
                "value": val_clean,
                "contribution": s_clean,
                "impact_direction": direction
            })

        # Separate positive (increases risk) and negative (decreases risk)
        positive_contribs = sorted(
            [c for c in all_contributions if c["contribution"] > 0],
            key=lambda x: x["contribution"],
            reverse=True
        )
        negative_contribs = sorted(
            [c for c in all_contributions if c["contribution"] < 0],
            key=lambda x: x["contribution"]
        )

        # Notebook-compatible top 2 positive and top 2 negative features
        top_shap_notebook = []
        if positive_contribs:
            top_shap_notebook.append([
                {"feature": c["feature"], "value": c["value"], "contribution": c["contribution"]}
                for c in positive_contribs[:2]
            ])
        if negative_contribs:
            top_shap_notebook.append([
                {"feature": c["feature"], "value": c["value"], "contribution": c["contribution"]}
                for c in negative_contribs[:2]
            ])

        # Flat list for Frontend Waterfall chart (top 4 positive + top 4 negative)
        waterfall_items = [
            ShapContributionItem(**c) for c in (positive_contribs[:4] + negative_contribs[:4])
        ]

        latency_ms = round((time.time() - start_time) * 1000, 2)

        # Check stub mode flag from request
        is_stub_mode = False
        if isinstance(request, UnifiedIntakeRequest) and request.pillar_2_ultrasound:
            is_stub_mode = (request.pillar_2_ultrasound.mode.lower() == "stub")
        elif isinstance(request, dict) and "pillar_2_ultrasound" in request:
            p2_dict = request.get("pillar_2_ultrasound") or {}
            is_stub_mode = (p2_dict.get("mode", "").lower() == "stub")

        return DecisionCoreResponse(
            risk_probability=risk_probability,
            risk_category=risk_category,
            safety_override=safety_override,
            shap_contributions=top_shap_notebook,
            waterfall_items=waterfall_items,
            metadata={
                "latency_ms": latency_ms,
                "calibration_method": "isotonic",
                "cv_folds": 5,
                "pillar_2_stub_mode": is_stub_mode,
                "contract_version": "1.0.0",
                "engine_build": __engine_signature__,
                "engine_id": __engine_id__,
                "squad": "ML Fusion & Clinical Safety"
            }
        )


# Global singleton instance for high-speed reuse
_SERVICE_INSTANCE: Optional[DecisionCoreService] = None


def get_decision_core_service() -> DecisionCoreService:
    """Returns the initialized singleton DecisionCoreService instance."""
    global _SERVICE_INSTANCE
    if _SERVICE_INSTANCE is None:
        _SERVICE_INSTANCE = DecisionCoreService()
    return _SERVICE_INSTANCE


def evaluate_patient(
    intake_data: Union[UnifiedIntakeRequest, Dict[str, Any], pd.DataFrame]
) -> Dict[str, Any]:
    """
    Convenience function for direct Python import by Backend Squad:
      from src.decision_core import evaluate_patient
      result = evaluate_patient(payload)
    """
    service = get_decision_core_service()
    response = service.evaluate(intake_data)
    return response.model_dump()


# ==============================================================================
# FastAPI Integration (Direct Mount / Standalone Service)
# ==============================================================================

from fastapi import FastAPI, APIRouter, status

router = APIRouter(prefix="/decision-core", tags=["Decision Core"])


@router.post(
    "/predict",
    response_model=DecisionCoreResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate Patient Risk and Compute Clinical Decision Core Slice",
    description="Fuses 3 pillars, enforces FOGSI clinical guardrails, computes calibrated PCOS risk, and generates SHAP attributions."
)
def predict_decision_core(payload: UnifiedIntakeRequest) -> DecisionCoreResponse:
    service = get_decision_core_service()
    return service.evaluate(payload)


@router.get("/health", tags=["Health"])
def health_check():
    service = get_decision_core_service()
    return {
        "status": "healthy",
        "service": "MediLong Decision Core",
        "features_loaded": len(service.feature_names),
        "model_loaded": service.calibrated_model is not None
    }


# Standalone application for local testing or microservice deployment
app = FastAPI(
    title="Project MediLong - Decision Core API",
    description="ML Fusion & Clinical Safety Squad Predictive Decision Microservice",
    version="1.0.0"
)
app.include_router(router)


# ==============================================================================
# CLI Testing & Demonstration
# ==============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("Project MediLong - Decision Core Service Self-Test")
    print("=" * 65)

    svc = get_decision_core_service()

    # 1. Test 3-Pillar Ingestion with Hypothyroidism Override
    sample_request = UnifiedIntakeRequest(
        patient_id="PATIENT-001-TEST",
        pillar_1_demographics=Pillar1Demographics(
            age=26.0,
            weight_kg=68.0,
            height_cm=160.0,
            cycle_ri=4,
            cycle_length_days=45.0,
            weight_gain=1,
            hair_growth=1,
            pimples=1
        ),
        pillar_2_ultrasound=Pillar2Ultrasound(
            mode="stub",
            follicle_count=18.0,
            confidence=0.88
        ),
        pillar_3_labs=Pillar3Labs(
            tsh_miu_l=6.5,  # Trigger Hypothyroidism override (> 4.5)
            prl_ng_ml=18.2,
            amh_ng_ml=8.4,
            fsh_miu_ml=4.2,
            lh_miu_ml=9.8
        )
    )

    print("\n[*] Evaluating 3-Pillar Intake Sample (with TSH=6.5 override):")
    res = svc.evaluate(sample_request)
    print(json.dumps(res.model_dump(), indent=2))

    # 2. Test Normal Case without Override
    print("\n[*] Evaluating Normal Intake Sample (TSH=2.1, PRL=12.0):")
    normal_request = sample_request.model_copy(deep=True)
    normal_request.pillar_3_labs.tsh_miu_l = 2.1
    res_normal = svc.evaluate(normal_request)
    print(json.dumps(res_normal.model_dump(), indent=2))
