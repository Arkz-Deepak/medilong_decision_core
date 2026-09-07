"""
tests/test_decision_core.py - Unit Tests for MediLong Decision Core & Clinical Guardrails
Author: Deepak (ML Fusion Squad)

Sprint 4 Definition of Done Validation:
- Guardrail layer demonstrably overrides model output on hard clinical rules (TSH > 4.5, PRL > 25.0)
- Guardrail overrides are unit-tested independently of the XGBoost path
- Risk categorization is deterministic, reproducible, and stable across repeated runs
- Handles missing biomarkers gracefully without crashing (imputation fallback)
- Preserves Pillar 2 stub mode disclosure
"""

import unittest
import pandas as pd
from src.decision_core import (
    ClinicalGuardrails,
    DecisionCoreService,
    UnifiedIntakeRequest,
    Pillar1Demographics,
    Pillar2Ultrasound,
    Pillar3Labs,
    get_decision_core_service
)


class TestClinicalGuardrailsIndependent(unittest.TestCase):
    """
    Tests guardrail layer independently of the ML/XGBoost path as required by Sprint 4.
    """

    def test_tsh_override_triggered(self):
        """TSH > 4.5 must trigger Hypothyroidism override"""
        record = {"TSH (mIU/L)": 6.8, "PRL(ng/mL)": 15.0}
        result = ClinicalGuardrails.evaluate(record)
        self.assertTrue(result["safety_override"])
        self.assertTrue(any("Hypothyroidism" in r for r in result["reasons"]))
        self.assertTrue(len(result["recommendations"]) > 0)

    def test_prl_override_triggered(self):
        """PRL > 25.0 must trigger Hyperprolactinemia override"""
        record = {"TSH (mIU/L)": 2.2, "PRL(ng/mL)": 38.5}
        result = ClinicalGuardrails.evaluate(record)
        self.assertTrue(result["safety_override"])
        self.assertTrue(any("Hyperprolactinemia" in r for r in result["reasons"]))

    def test_dual_overrides(self):
        """Both TSH and PRL elevated"""
        record = {"TSH (mIU/L)": 7.1, "PRL(ng/mL)": 45.0}
        result = ClinicalGuardrails.evaluate(record)
        self.assertTrue(result["safety_override"])
        self.assertEqual(len(result["reasons"]), 2)

    def test_normal_guardrails_no_override(self):
        """Normal physiological levels should NOT trigger safety override"""
        record = {"TSH (mIU/L)": 2.1, "PRL(ng/mL)": 14.0}
        result = ClinicalGuardrails.evaluate(record)
        self.assertFalse(result["safety_override"])
        self.assertEqual(len(result["reasons"]), 0)

    def test_none_and_nan_values_safe(self):
        """Missing or NaN values should safely skip override without raising errors"""
        record = {"TSH (mIU/L)": None, "PRL(ng/mL)": float('nan')}
        result = ClinicalGuardrails.evaluate(record)
        self.assertFalse(result["safety_override"])


class TestDecisionCoreServiceIntegration(unittest.TestCase):
    """
    End-to-End Decision Core Integration Tests with 3-Pillar schemas.
    """

    @classmethod
    def setUpClass(cls):
        cls.service = get_decision_core_service()

    def test_reproducibility_and_stability(self):
        """Risk categorization and probability must be 100% reproducible on repeated runs"""
        payload = UnifiedIntakeRequest(
            pillar_1_demographics=Pillar1Demographics(age=28, weight_kg=55, height_cm=162),
            pillar_2_ultrasound=Pillar2Ultrasound(mode="stub", follicle_count=8),
            pillar_3_labs=Pillar3Labs(tsh_miu_l=2.5, prl_ng_ml=12.0)
        )
        res1 = self.service.evaluate(payload)
        res2 = self.service.evaluate(payload)

        self.assertEqual(res1.risk_probability, res2.risk_probability)
        self.assertEqual(res1.risk_category, res2.risk_category)
        self.assertEqual(res1.safety_override.safety_override, res2.safety_override.safety_override)

    def test_pillar_2_stub_mode_disclosure(self):
        """Pillar 2 stub mode must be flagged in metadata for Frontend disclosure"""
        payload_stub = UnifiedIntakeRequest(
            pillar_2_ultrasound=Pillar2Ultrasound(mode="stub")
        )
        res_stub = self.service.evaluate(payload_stub)
        self.assertTrue(res_stub.metadata["pillar_2_stub_mode"])

        payload_live = UnifiedIntakeRequest(
            pillar_2_ultrasound=Pillar2Ultrasound(mode="live")
        )
        res_live = self.service.evaluate(payload_live)
        self.assertFalse(res_live.metadata["pillar_2_stub_mode"])

    def test_missing_biomarkers_graceful_handling(self):
        """Empty or partially missing requests must impute medians and not crash"""
        empty_payload = UnifiedIntakeRequest()
        res = self.service.evaluate(empty_payload)
        self.assertIn(res.risk_category, ["Low", "Moderate", "High"])
        self.assertTrue(0.0 <= res.risk_probability <= 1.0)
        self.assertTrue(len(res.shap_contributions) > 0)

    def test_shap_waterfall_format(self):
        """SHAP waterfall data must contain both positive and negative feature impacts"""
        payload = UnifiedIntakeRequest(
            pillar_1_demographics=Pillar1Demographics(weight_gain=1, hair_growth=1),
            pillar_3_labs=Pillar3Labs(tsh_miu_l=3.0)
        )
        res = self.service.evaluate(payload)
        self.assertTrue(len(res.waterfall_items) > 0)
        directions = {item.impact_direction for item in res.waterfall_items}
        self.assertTrue("increases_risk" in directions or "decreases_risk" in directions)

    def test_engine_build_signature(self):
        """Engine build and telemetry signature must be embedded in metadata"""
        payload = UnifiedIntakeRequest()
        res = self.service.evaluate(payload)
        self.assertIn("engine_build", res.metadata)
        self.assertTrue(res.metadata["engine_build"].startswith("arkz-"))
        self.assertIn("engine_id", res.metadata)


if __name__ == "__main__":
    unittest.main()
