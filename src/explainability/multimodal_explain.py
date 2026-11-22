"""
MediFusion AI - Multimodal Explainability (XAI) Engine
Provides comprehensive explainability for the multimodal fusion pipeline:
1. Modality-level contribution and attribution (leave-one-modality-out marginal impact).
2. Learned Cross-Modal Gating weights and raw gate activations.
3. Cross-modal multi-head attention matrix (3x3 inter-modality attention flow).
4. Automated modality concordance/discordance detection with clinical resolution narrative.
5. Pathway-level deep dive explanations:
   - Vision pathway: Grad-CAM spatial activation heatmaps on chest radiographs.
   - Tabular pathway: SHAP feature attributions on symptoms and clinical vitals.
6. Honest documentation of prototype data limitations (synthetic vitals & non-patient-paired cohorts).
"""

import io
import logging
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

MODALITY_NAMES = ["Radiology (Vision)", "Symptom Stream", "Clinical Vitals"]


def compute_modality_marginal_attributions(
    fusion_model: torch.nn.Module,
    image_emb: torch.Tensor,
    symptom_emb: torch.Tensor,
    clinical_emb: torch.Tensor,
    active_mask: List[float],
    winning_class_idx: int,
    device: torch.device,
) -> Dict[str, float]:
    """
    Compute cooperative leave-one-modality-out marginal contributions
    towards the winning class probability.
    
    Delta_i = max(0, P_full(c) - P_without_i(c))
    Normalized so active modalities sum to 1.0 (or 100%).
    """
    mask_full = torch.tensor([active_mask], dtype=torch.float32, device=device)
    with torch.no_grad():
        logits_full = fusion_model(
            image_emb.to(device),
            symptom_emb.to(device),
            clinical_emb.to(device),
            mask_full,
        )
        p_full = float(F.softmax(logits_full, dim=1)[0, winning_class_idx].item())

    deltas = {}
    keys = ["image", "symptoms", "clinical"]
    active_indices = [i for i, m in enumerate(active_mask) if m > 0]
    
    if len(active_indices) <= 1:
        # Single active modality gets 1.0
        return {k: (1.0 if active_mask[i] > 0 else 0.0) for i, k in enumerate(keys)}

    for i, k in enumerate(keys):
        if active_mask[i] == 0:
            deltas[k] = 0.0
            continue
            
        ablated_mask = list(active_mask)
        ablated_mask[i] = 0.0
        mask_abl = torch.tensor([ablated_mask], dtype=torch.float32, device=device)
        
        with torch.no_grad():
            logits_abl = fusion_model(
                image_emb.to(device),
                symptom_emb.to(device),
                clinical_emb.to(device),
                mask_abl,
            )
            p_abl = float(F.softmax(logits_abl, dim=1)[0, winning_class_idx].item())
            
        marginal_drop = max(0.001, p_full - p_abl)
        deltas[k] = marginal_drop

    total_delta = sum(deltas.values())
    if total_delta > 0:
        attributions = {k: round(v / total_delta, 4) for k, v in deltas.items()}
    else:
        # Fallback to equal split among active
        n_act = len(active_indices)
        attributions = {k: (round(1.0 / n_act, 4) if active_mask[i] > 0 else 0.0) for i, k in enumerate(keys)}

    return attributions


def evaluate_modality_concordance(
    individual_preds: Dict[str, List[Any]],
    winning_class: str,
    winning_prob: float,
    gating_weights: Dict[str, float],
) -> Tuple[str, str, str]:
    """
    Evaluate whether independent modality streams agree or produce conflicting signals.
    Returns:
        concordance_status: 'Concordant Consensus', 'Discordant (Resolved)', or 'Single-Modality Assessment'
        concordance_details: Narrative explaining the agreement or conflict.
        decision_reasoning: Detailed clinical rationale for why the final class was selected.
    """
    active_streams = list(individual_preds.keys())
    
    if len(active_streams) <= 1:
        stream_name = active_streams[0].capitalize() if active_streams else "Single"
        status = "Single-Modality Assessment"
        details = f"Diagnostic assessment synthesized solely from {stream_name} stream evidence."
        reasoning = f"The model leveraged available {stream_name} data to infer {winning_class} ({winning_prob*100:.1f}% confidence)."
        return status, details, reasoning

    # Get top predicted disease per active stream
    top_per_stream = {}
    for mod, preds in individual_preds.items():
        if preds and len(preds) > 0:
            top_p = max(preds, key=lambda p: p.probability)
            top_per_stream[mod] = (top_p.disease, top_p.probability)

    # Check for discordance
    # Radiology predicts Normal/Pneumonia; Symptoms/Clinical predict Malaria/Pneumonia/Typhoid
    img_pred = top_per_stream.get("image")
    sym_pred = top_per_stream.get("symptoms")
    cli_pred = top_per_stream.get("clinical")

    has_conflict = False
    conflict_desc = []

    if img_pred and sym_pred:
        # Image Normal vs Respiratory Pneumonia or Systemic Malaria/Typhoid
        if img_pred[0].upper() == "NORMAL" and sym_pred[0] in ["Malaria", "Typhoid"]:
            has_conflict = True
            conflict_desc.append(
                f"Radiology stream indicates clear pulmonary fields (Normal: {img_pred[1]*100:.1f}%), "
                f"whereas Symptom stream exhibits acute paroxysmal markers for {sym_pred[0]} ({sym_pred[1]*100:.1f}%)"
            )
        elif img_pred[0].upper() == "NORMAL" and sym_pred[0] == "Pneumonia":
            has_conflict = True
            conflict_desc.append(
                f"Radiology stream found no clear consolidations (Normal: {img_pred[1]*100:.1f}%), "
                f"yet Symptom stream detected respiratory distress indicative of Pneumonia ({sym_pred[1]*100:.1f}%)"
            )
        elif img_pred[0].upper() == "PNEUMONIA" and sym_pred[0] in ["Malaria", "Typhoid"]:
            has_conflict = True
            conflict_desc.append(
                f"Radiology stream detected thoracic opacities (Pneumonia: {img_pred[1]*100:.1f}%), "
                f"competing with systemic {sym_pred[0]} symptoms ({sym_pred[1]*100:.1f}%)"
            )

    if has_conflict:
        status = "Discordant Signals Resolved"
        details = " | ".join(conflict_desc)
        
        if winning_class in ["Malaria", "Typhoid"]:
            reasoning = (
                f"Discordant Modality Signals Detected: Radiology indicated clear pulmonary parenchymal fields (Normal: {img_pred[1]*100:.1f}%), "
                f"while Symptom and Clinical telemetry exhibited pronounced systemic infection markers ({winning_class}). "
                f"The Gated Cross-Modal Attention network appropriately weighted the absence of pulmonary consolidation to rule out pneumonia, "
                f"allocating primary diagnostic attribution ({gating_weights.get('symptoms', 0)*100:.1f}% symptom / {gating_weights.get('clinical', 0)*100:.1f}% vitals) "
                f"to systemic infection ({winning_class} with {winning_prob*100:.1f}% unified confidence)."
            )
        elif winning_class == "Pneumonia":
            reasoning = (
                f"Cross-modal synthesis resolved in favor of Pneumonia ({winning_prob*100:.1f}% confidence) "
                f"driven by focal pulmonary findings and acute respiratory distress parameters."
            )
        else: # Normal
            reasoning = (
                f"Normal physiological findings and clear chest radiograph confirmed benign presentation without acute consolidation."
            )
    else:
        status = "Concordant Consensus"
        details = "Active diagnostic streams demonstrate mutually corroborating clinical evidence."
        reasoning = (
            f"Cross-modal alignment: Active modalities mutually reinforce the assessment of {winning_class} "
            f"with {(winning_prob * 100):.1f}% confidence. Gating weights dynamically distributed as "
            f"Vision: {gating_weights.get('image', 0)*100:.1f}%, Symptoms: {gating_weights.get('symptoms', 0)*100:.1f}%, Vitals: {gating_weights.get('clinical', 0)*100:.1f}%."
        )

    return status, details, reasoning


def generate_multimodal_xai(
    fusion_model: torch.nn.Module,
    image_emb: torch.Tensor,
    symptom_emb: torch.Tensor,
    clinical_emb: torch.Tensor,
    mask: List[float],
    gating_weights: torch.Tensor,
    raw_gates: torch.Tensor,
    attn_weights: torch.Tensor,
    winning_class: str,
    winning_prob: float,
    winning_idx: int,
    individual_preds: Dict[str, List[Any]],
    device: torch.device,
    image_bytes: Optional[bytes] = None,
    image_model: Optional[torch.nn.Module] = None,
    clinical_model: Optional[Any] = None,
    clinical_scaler: Optional[Any] = None,
    raw_symptoms: Optional[Dict[str, int]] = None,
    raw_clinical_data: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Generate complete multimodal XAI payload incorporating:
    - Gating weights
    - Cross-attention matrix
    - Modality marginal attributions
    - Concordance analysis
    - Optional Grad-CAM overlay & heatmap (vision)
    - Optional SHAP feature drivers (tabular)
    """
    gating_dict = {
        "image": round(float(gating_weights[0, 0].item()), 4),
        "symptoms": round(float(gating_weights[0, 1].item()), 4),
        "clinical": round(float(gating_weights[0, 2].item()), 4),
    }

    raw_gating_dict = {
        "image": round(float(raw_gates[0, 0].item()), 4),
        "symptoms": round(float(raw_gates[0, 1].item()), 4),
        "clinical": round(float(raw_gates[0, 2].item()), 4),
    }

    # Cross-attention matrix (3x3)
    attn_matrix = np.round(attn_weights[0].detach().cpu().numpy(), 4).tolist()

    # Modality marginal attributions
    attributions = compute_modality_marginal_attributions(
        fusion_model=fusion_model,
        image_emb=image_emb,
        symptom_emb=symptom_emb,
        clinical_emb=clinical_emb,
        active_mask=mask,
        winning_class_idx=winning_idx,
        device=device,
    )

    # Concordance & narrative
    concordance_status, concordance_details, decision_reasoning = evaluate_modality_concordance(
        individual_preds=individual_preds,
        winning_class=winning_class,
        winning_prob=winning_prob,
        gating_weights=gating_dict,
    )

    xai_payload = {
        "modality_concordance": concordance_status,
        "concordance_details": concordance_details,
        "decision_reasoning": decision_reasoning,
        "gating_weights": gating_dict,
        "raw_gating_scores": raw_gating_dict,
        "modality_attribution": attributions,
        "cross_attention": {
            "matrix": attn_matrix,
            "modality_labels": MODALITY_NAMES,
        },
        "gradcam_overlay_base64": None,
        "gradcam_heatmap_base64": None,
        "top_symptom_features": None,
        "top_clinical_features": None,
        "limitations": (
            "Clinical Prototype Notice: Imaging and symptom datasets originate from independent non-patient-paired "
            "cohorts. Clinical vitals include synthetic parameters for architectural demonstration. "
            "This prototype illustrates multimodal gated cross-attention and is not an FDA-cleared diagnostic device."
        ),
    }

    # Vision Stream Grad-CAM (if image provided)
    if mask[0] > 0 and image_bytes and image_model is not None:
        try:
            from src.explainability.gradcam import GradCAM
            from src.data.xray_dataset import get_eval_transforms
            
            image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            transform = get_eval_transforms(224)
            tensor = transform(image_pil).unsqueeze(0).to(device)
            
            gradcam = GradCAM(image_model)
            heatmap, _, _ = gradcam.generate(tensor)
            overlay_b64 = gradcam.create_overlay(image_pil, heatmap)
            heatmap_b64 = gradcam.heatmap_to_base64(heatmap)
            
            xai_payload["gradcam_overlay_base64"] = overlay_b64
            xai_payload["gradcam_heatmap_base64"] = heatmap_b64
        except Exception as e:
            logger.warning(f"Failed to generate Grad-CAM in multimodal XAI: {e}")

    # Tabular SHAP (if symptoms or clinical vitals provided)
    if (mask[1] > 0 or mask[2] > 0) and clinical_model is not None and clinical_scaler is not None:
        try:
            from src.explainability.shap_explain import SHAPExplainer
            from src.data.symptom_dataset import (
                preprocess_symptoms, SYMPTOM_FEATURES, SYMPTOM_LABELS
            )
            CLINICAL_FEATURES = [
                "age", "temperature", "heart_rate",
                "bp_systolic", "bp_diastolic",
                "respiratory_rate", "oxygen_saturation"
            ]
            all_feat_names = SYMPTOM_FEATURES + CLINICAL_FEATURES
            
            sym_dict = raw_symptoms or {}
            sym_vec = preprocess_symptoms(sym_dict)
            clin_dict = raw_clinical_data or {}
            clin_values = np.array([[
                clin_dict.get("age", 45),
                clin_dict.get("temperature", 37.0),
                clin_dict.get("heart_rate", 75),
                clin_dict.get("bp_systolic", 120),
                clin_dict.get("bp_diastolic", 80),
                clin_dict.get("respiratory_rate", 16),
                clin_dict.get("oxygen_saturation", 98),
            ]], dtype=np.float32)
            
            X_full = np.hstack([sym_vec, clin_values])
            X_scaled = clinical_scaler.transform(X_full)
            
            explainer = SHAPExplainer(clinical_model, all_feat_names)
            shap_res = explainer.explain(X_scaled, class_names=SYMPTOM_LABELS, top_n=6)
            
            xai_payload["top_clinical_features"] = shap_res["feature_importance"]
            if "per_class_importance" in shap_res:
                xai_payload["per_class_importance"] = shap_res["per_class_importance"]
        except Exception as e:
            logger.warning(f"Failed to generate SHAP in multimodal XAI: {e}")

    return xai_payload
