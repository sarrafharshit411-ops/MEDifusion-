"""
MediFusion AI - Multimodal Fusion Pipeline Audit & XAI Test Suite
Validates:
1. Conflicting prediction resolution (Normal X-ray + Malaria symptoms/vitals -> Malaria, NOT Pneumonia).
2. Concordant prediction synthesis (Pneumonia image + Pneumonia symptoms/vitals -> Pneumonia).
3. Dynamic gating weights and 3x3 attention matrix (verifies no hardcoded 45/35/20).
4. Full XAI payload: Grad-CAM overlay, SHAP feature attributions, and marginal contributions.
5. Missing modality resilience.
"""

import os
import json
import pytest
import numpy as np
from fastapi.testclient import TestClient

from api.main import app
from api.model_loader import registry


@pytest.fixture(scope="module")
def client():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, "models")
    registry.load_all(models_dir)
    return TestClient(app)


def test_conflicting_signals_resolution(client):
    """
    AUDIT TEST 1:
    When Radiology predicts Normal (83%) and Symptoms/Clinical predict Malaria,
    the fusion model MUST NOT predict Pneumonia (which was a previous shortcut bug).
    It must resolve in favor of Malaria with dynamic gating weights.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sample_img_path = os.path.join(project_root, "frontend", "sample_xray.png")
    with open(sample_img_path, "rb") as f:
        img_bytes = f.read()

    files = {
        "file": ("sample_xray.png", img_bytes, "image/png"),
        "symptoms": (None, json.dumps({
            "fever": 1, "chills": 1, "sweating": 1, "headache": 1, "vomiting": 1, "fatigue": 1
        })),
        "clinical_data": (None, json.dumps({
            "age": 45, "temperature": 39.0, "heart_rate": 105,
            "bp_systolic": 120, "bp_diastolic": 80, "respiratory_rate": 18,
            "oxygen_saturation": 97
        }))
    }

    res = client.post("/api/predict/multimodal", files=files)
    assert res.status_code == 200, res.text
    data = res.json()

    # 1. Predicted class MUST be Malaria, NOT Pneumonia!
    assert data["predicted_class"] == "Malaria", f"Expected Malaria, got {data['predicted_class']}"
    
    # 2. Check individual predictions
    img_preds = {p["disease"]: p["probability"] for p in data["individual_predictions"]["image"]}
    assert img_preds["NORMAL"] > 0.70, f"Expected Normal > 70%, got {img_preds['NORMAL']}"

    sym_preds = {p["disease"]: p["probability"] for p in data["individual_predictions"]["symptoms"]}
    assert sym_preds["Malaria"] > 0.50, f"Expected Malaria > 50%, got {sym_preds['Malaria']}"

    # 3. Dynamic gating weights MUST NOT be hardcoded [0.45, 0.35, 0.20]
    gating = data["gating_weights"]
    assert gating["image"] > 0 and gating["symptoms"] > 0 and gating["clinical"] > 0
    assert not (abs(gating["image"] - 0.45) < 1e-4 and abs(gating["symptoms"] - 0.35) < 1e-4 and abs(gating["clinical"] - 0.20) < 1e-4), \
        "Gating weights appear hardcoded to 45/35/20!"

    # 4. Gating weights must sum to approximately 1.0
    total_gating = sum(gating.values())
    assert abs(total_gating - 1.0) < 1e-2, f"Gating weights should sum to 1.0, got {total_gating}"

    # 5. Multimodal XAI Validation
    xai = data["xai"]
    assert "Discordant" in xai["modality_concordance"]
    assert "decision_reasoning" in xai and len(xai["decision_reasoning"]) > 30
    assert xai["gradcam_overlay_base64"] is not None
    assert xai["top_clinical_features"] is not None and len(xai["top_clinical_features"]) > 0

    # Cross-attention matrix must be 3x3
    matrix = xai["cross_attention"]["matrix"]
    assert len(matrix) == 3 and all(len(row) == 3 for row in matrix)


def test_concordant_pneumonia_synthesis(client):
    """
    AUDIT TEST 2:
    When all active modalities exhibit Pneumonia markers (Pneumonia radiograph,
    Pneumonia symptoms, distress vitals), the model must predict Pneumonia with
    high confidence and Concordant Consensus.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pneumonia_img_path = os.path.join(project_root, "frontend", "images", "samples", "pneumonia_xray.jpeg")
    with open(pneumonia_img_path, "rb") as f:
        img_bytes = f.read()

    files = {
        "file": ("pneumonia_xray.jpeg", img_bytes, "image/jpeg"),
        "symptoms": (None, json.dumps({
            "cough": 1, "rapid_breathing": 1, "fever": 1, "fatigue": 1, "chills": 1
        })),
        "clinical_data": (None, json.dumps({
            "age": 55, "temperature": 39.2, "heart_rate": 108,
            "bp_systolic": 130, "bp_diastolic": 85, "respiratory_rate": 26,
            "oxygen_saturation": 90
        }))
    }

    res = client.post("/api/predict/multimodal", files=files)
    assert res.status_code == 200, res.text
    data = res.json()

    assert data["predicted_class"] == "Pneumonia"
    assert data["confidence"] > 0.85, f"Expected confidence > 85%, got {data['confidence']}"
    assert "Concordant" in data["xai"]["modality_concordance"]
    assert data["xai"]["gradcam_overlay_base64"] is not None


def test_single_modality_image_only(client):
    """
    AUDIT TEST 3:
    When only an image is provided, image gating must be 1.0, other gates 0.0.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sample_img_path = os.path.join(project_root, "frontend", "sample_xray.png")
    with open(sample_img_path, "rb") as f:
        img_bytes = f.read()

    files = {
        "file": ("sample_xray.png", img_bytes, "image/png"),
    }

    res = client.post("/api/predict/multimodal", files=files)
    assert res.status_code == 200, res.text
    data = res.json()

    assert data["predicted_class"] == "Normal"
    gating = data["gating_weights"]
    assert gating["image"] == 1.0
    assert gating["symptoms"] == 0.0
    assert gating["clinical"] == 0.0
    assert "Single" in data["xai"]["modality_concordance"]


def test_single_modality_symptoms_only(client):
    """
    AUDIT TEST 4:
    When only symptoms are provided, symptom gating must be 1.0.
    """
    files = {
        "symptoms": (None, json.dumps({
            "fever": 1, "chills": 1, "sweating": 1, "headache": 1, "vomiting": 1
        })),
    }

    res = client.post("/api/predict/multimodal", files=files)
    assert res.status_code == 200, res.text
    data = res.json()

    assert data["predicted_class"] == "Malaria"
    gating = data["gating_weights"]
    assert gating["image"] == 0.0
    assert gating["symptoms"] == 1.0
    assert gating["clinical"] == 0.0


if __name__ == "__main__":
    pytest.main(["-v", __file__])
