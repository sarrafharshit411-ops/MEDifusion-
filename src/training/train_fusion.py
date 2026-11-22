"""
MediFusion AI - Fusion Model Training Pipeline
Trains both ConcatFusion and GatedAttentionFusion models.

NOTE: Since the two datasets are NOT patient-paired, we create synthetic pairings
for demonstration:
- For "Pneumonia" class: pair X-ray embeddings with symptom rows both labeled Pneumonia
- For "Normal" class: use only image embeddings (symptom/clinical set to zero)
- For "Malaria"/"Typhoid": use only symptom/clinical embeddings (image set to zero)

This clearly demonstrates the fusion architecture while being honest about data limitations.
"""

import os
import sys
import logging
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.xray_dataset import load_xray_data
from src.data.symptom_dataset import (
    load_symptom_data, generate_synthetic_clinical_features,
    SYMPTOM_FEATURES, SYMPTOM_LABELS,
)
from src.models.image_model import ChestXRayModel
from src.models.fusion_model import (
    ConcatFusion, GatedAttentionFusion,
    UNIFIED_CLASSES, NUM_UNIFIED_CLASSES,
)
from src.utils.helpers import load_config, set_seed, get_device, ensure_dir, save_metrics, setup_logging

logger = logging.getLogger(__name__)

# Unified labels: Normal=0, Pneumonia=1, Malaria=2, Typhoid=3


def generate_fusion_data(config: dict, device: torch.device):
    """
    Generate training data for the fusion model by extracting embeddings
    from trained single-modality models and creating balanced, clinically realistic pairings.
    
    Clinical alignment:
    - Normal: Normal image with healthy baseline vitals and no acute symptoms, or normal image alone.
    - Pneumonia: Pneumonia image with acute respiratory symptoms and clinical distress vitals, or partial streams.
    - Malaria: Clear lung fields (Normal image) with systemic paroxysmal symptoms (fever, chills, headache) and malaria vitals, or partial streams.
    - Typhoid: Clear lung fields (Normal image) with enteric fever symptoms and vitals, or partial streams.
    
    Applies diverse modality masking patterns across all classes to prevent the network from
    learning shortcut correlations with modality availability.
    """
    import joblib
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models_dir = os.path.join(project_root, config["paths"]["models_dir"])
    cache_path = os.path.join(models_dir, "xray_embeddings_cache.npz")
    
    # --- Image embeddings ---
    if os.path.exists(cache_path):
        logger.info(f"Loading cached X-ray embeddings from {cache_path}...")
        cache = np.load(cache_path)
        normal_img_embs = cache["normal"]
        pneumonia_img_embs = cache["pneumonia"]
    else:
        logger.info("Extracting image embeddings from raw dataset...")
        image_model = ChestXRayModel(num_classes=2, pretrained=False)
        image_model.load_state_dict(
            torch.load(os.path.join(models_dir, "image_model.pth"), map_location=device, weights_only=True)
        )
        image_model.to(device)
        image_model.eval()
        
        train_loader, _, _, _ = load_xray_data(
            data_root=config["dataset"]["xray_path"],
            input_size=config["image_model"]["input_size"],
            val_split=0.0001,
            batch_size=config["image_model"]["batch_size"],
            seed=config["training"]["seed"],
        )
        
        normal_img_list = []
        pneumonia_img_list = []
        with torch.no_grad():
            for images, labels in train_loader:
                images = images.to(device)
                emb = image_model.get_embedding(images).cpu().numpy()
                lbls = labels.numpy()
                normal_img_list.append(emb[lbls == 0])
                pneumonia_img_list.append(emb[lbls == 1])
                
        normal_img_embs = np.vstack(normal_img_list)
        pneumonia_img_embs = np.vstack(pneumonia_img_list)
        np.savez_compressed(cache_path, normal=normal_img_embs, pneumonia=pneumonia_img_embs)
        
    logger.info(f"Available X-ray embeddings: {len(normal_img_embs)} Normal, {len(pneumonia_img_embs)} Pneumonia")
    
    # --- Symptom & Clinical data ---
    X_train_sym, X_test_sym, y_train_sym, y_test_sym, le, sym_info = load_symptom_data(
        csv_path=config["dataset"]["symptom_path"],
        test_size=0.2,
        seed=config["training"]["seed"],
    )
    
    clinical_train, _ = generate_synthetic_clinical_features(
        X_train_sym, seed=config["training"]["seed"]
    )
    clin_train_full = np.hstack([X_train_sym, clinical_train])
    
    # Load clinical scaler so training features match inference scaling exactly
    scaler_path = os.path.join(models_dir, "clinical_scaler.joblib")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(clin_train_full)
        joblib.dump(scaler, scaler_path)
        
    clin_train_scaled = scaler.transform(clin_train_full)
    
    # Stratify by symptom classes: 0=Malaria, 1=Pneumonia, 2=Typhoid
    malaria_sym = X_train_sym[y_train_sym == 0]
    malaria_clin = clin_train_scaled[y_train_sym == 0]
    
    pneumonia_sym = X_train_sym[y_train_sym == 1]
    pneumonia_clin = clin_train_scaled[y_train_sym == 1]
    
    typhoid_sym = X_train_sym[y_train_sym == 2]
    typhoid_clin = clin_train_scaled[y_train_sym == 2]
    
    # Synthesize healthy normal vitals
    rng = np.random.RandomState(config["training"]["seed"])
    n_norm_pool = len(normal_img_embs)
    norm_age = rng.normal(42, 14, n_norm_pool).clip(18, 80)
    norm_temp = rng.normal(36.8, 0.25, n_norm_pool).clip(36.2, 37.3)
    norm_hr = rng.normal(72, 6, n_norm_pool).clip(60, 85)
    norm_bpsys = rng.normal(120, 7, n_norm_pool).clip(105, 135)
    norm_bpdia = rng.normal(80, 5, n_norm_pool).clip(70, 88)
    norm_rr = rng.normal(16, 2, n_norm_pool).clip(12, 20)
    norm_spo2 = rng.normal(98.5, 0.7, n_norm_pool).clip(96, 100)
    healthy_vitals = np.column_stack([norm_age, norm_temp, norm_hr, norm_bpsys, norm_bpdia, norm_rr, norm_spo2])
    healthy_sym = np.zeros((n_norm_pool, 15), dtype=np.float32)
    healthy_clin_scaled = scaler.transform(np.hstack([healthy_sym, healthy_vitals]))
    
    all_image_emb = []
    all_symptom_emb = []
    all_clinical_emb = []
    all_masks = []
    all_unified_labels = []
    
    N_PER_CLASS = 1800
    
    # 0) NORMAL CLASS:
    for i in range(N_PER_CLASS):
        img = normal_img_embs[i % len(normal_img_embs)]
        clin = healthy_clin_scaled[i % len(healthy_clin_scaled)]
        sym = healthy_sym[i % len(healthy_sym)]
        r = i % 10
        if r < 4:
            m = [1.0, 0.0, 0.0]; sym = np.zeros(15, dtype=np.float32); clin = np.zeros(22, dtype=np.float32)
        elif r < 7:
            m = [1.0, 1.0, 1.0]
        elif r < 9:
            m = [1.0, 0.0, 1.0]; sym = np.zeros(15, dtype=np.float32)
        else:
            m = [0.0, 0.0, 1.0]; img = np.zeros(512, dtype=np.float32); sym = np.zeros(15, dtype=np.float32)
        all_image_emb.append(img); all_symptom_emb.append(sym); all_clinical_emb.append(clin)
        all_masks.append(m); all_unified_labels.append(0)
        
    # 1) PNEUMONIA CLASS:
    for i in range(N_PER_CLASS):
        img = pneumonia_img_embs[i % len(pneumonia_img_embs)]
        sym = pneumonia_sym[i % len(pneumonia_sym)]
        clin = pneumonia_clin[i % len(pneumonia_clin)]
        r = i % 10
        if r < 5:
            m = [1.0, 1.0, 1.0]
        elif r < 7:
            m = [1.0, 0.0, 0.0]; sym = np.zeros(15, dtype=np.float32); clin = np.zeros(22, dtype=np.float32)
        elif r < 9:
            m = [0.0, 1.0, 1.0]; img = np.zeros(512, dtype=np.float32)
        else:
            m = [0.0, 1.0, 0.0]; img = np.zeros(512, dtype=np.float32); clin = np.zeros(22, dtype=np.float32)
        all_image_emb.append(img); all_symptom_emb.append(sym); all_clinical_emb.append(clin)
        all_masks.append(m); all_unified_labels.append(1)
        
    # 2) MALARIA CLASS:
    for i in range(N_PER_CLASS):
        # In real clinical presentation, malaria patient has clear lungs on X-ray
        img = normal_img_embs[(i + 400) % len(normal_img_embs)]
        sym = malaria_sym[i % len(malaria_sym)]
        clin = malaria_clin[i % len(malaria_clin)]
        r = i % 10
        if r < 5:
            m = [1.0, 1.0, 1.0] # Normal radiograph + Malaria systemic infection
        elif r < 8:
            m = [0.0, 1.0, 1.0]; img = np.zeros(512, dtype=np.float32)
        elif r < 9:
            m = [0.0, 1.0, 0.0]; img = np.zeros(512, dtype=np.float32); clin = np.zeros(22, dtype=np.float32)
        else:
            m = [1.0, 1.0, 0.0]; clin = np.zeros(22, dtype=np.float32)
        all_image_emb.append(img); all_symptom_emb.append(sym); all_clinical_emb.append(clin)
        all_masks.append(m); all_unified_labels.append(2)
        
    # 3) TYPHOID CLASS:
    for i in range(N_PER_CLASS):
        # In real clinical presentation, typhoid enteric fever has clear lungs on X-ray
        img = normal_img_embs[(i + 800) % len(normal_img_embs)]
        sym = typhoid_sym[i % len(typhoid_sym)]
        clin = typhoid_clin[i % len(typhoid_clin)]
        r = i % 10
        if r < 5:
            m = [1.0, 1.0, 1.0] # Normal radiograph + Typhoid infection
        elif r < 8:
            m = [0.0, 1.0, 1.0]; img = np.zeros(512, dtype=np.float32)
        elif r < 9:
            m = [0.0, 1.0, 0.0]; img = np.zeros(512, dtype=np.float32); clin = np.zeros(22, dtype=np.float32)
        else:
            m = [1.0, 1.0, 0.0]; clin = np.zeros(22, dtype=np.float32)
        all_image_emb.append(img); all_symptom_emb.append(sym); all_clinical_emb.append(clin)
        all_masks.append(m); all_unified_labels.append(3)
    
    # Convert to tensors
    data = {
        "image_emb": torch.tensor(np.array(all_image_emb), dtype=torch.float32),
        "symptom_emb": torch.tensor(np.array(all_symptom_emb), dtype=torch.float32),
        "clinical_emb": torch.tensor(np.array(all_clinical_emb), dtype=torch.float32),
        "masks": torch.tensor(np.array(all_masks), dtype=torch.float32),
        "labels": torch.tensor(all_unified_labels, dtype=torch.long),
    }
    
    logger.info(f"Balanced fusion dataset generated: {len(all_unified_labels)} total samples")
    for i, cls in enumerate(UNIFIED_CLASSES):
        count = sum(1 for l in all_unified_labels if l == i)
        logger.info(f"  {cls}: {count} samples")
    
    return data


def train_fusion_model(config: dict = None) -> Dict:
    """Train both ConcatFusion and GatedAttentionFusion models."""
    setup_logging()
    
    if config is None:
        config = load_config()
    
    set_seed(config["training"]["seed"])
    device = get_device()
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models_dir = os.path.join(project_root, config["paths"]["models_dir"])
    reports_dir = os.path.join(project_root, config["paths"]["reports_dir"])
    ensure_dir(models_dir)
    ensure_dir(reports_dir)
    
    fusion_config = config["fusion_model"]
    
    # Generate fusion data
    data = generate_fusion_data(config, device)
    
    # Train/val split
    n = len(data["labels"])
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(config["training"]["seed"]))
    val_size = int(n * 0.15)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    
    train_dataset = TensorDataset(
        data["image_emb"][train_idx],
        data["symptom_emb"][train_idx],
        data["clinical_emb"][train_idx],
        data["masks"][train_idx],
        data["labels"][train_idx],
    )
    val_dataset = TensorDataset(
        data["image_emb"][val_idx],
        data["symptom_emb"][val_idx],
        data["clinical_emb"][val_idx],
        data["masks"][val_idx],
        data["labels"][val_idx],
    )
    
    train_loader = DataLoader(train_dataset, batch_size=fusion_config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=fusion_config["batch_size"], shuffle=False)
    
    results = {}
    
    for model_name, ModelClass in [("concat", ConcatFusion), ("gated", GatedAttentionFusion)]:
        logger.info(f"\n{'='*50}\nTraining {model_name} fusion model\n{'='*50}")
        
        model = ModelClass(
            image_dim=fusion_config["image_embedding_dim"],
            symptom_dim=fusion_config["symptom_embedding_dim"],
            clinical_dim=fusion_config["clinical_embedding_dim"],
            num_classes=NUM_UNIFIED_CLASSES,
            dropout=fusion_config["dropout"],
        )
        model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=fusion_config["learning_rate"])
        
        best_val_acc = 0
        
        for epoch in range(1, fusion_config["epochs"] + 1):
            # Train
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for img_e, sym_e, cli_e, mask, labels in train_loader:
                img_e, sym_e, cli_e = img_e.to(device), sym_e.to(device), cli_e.to(device)
                mask, labels = mask.to(device), labels.to(device)
                
                optimizer.zero_grad()
                logits = model(img_e, sym_e, cli_e, mask)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * labels.size(0)
                train_correct += (logits.argmax(1) == labels).sum().item()
                train_total += labels.size(0)
            
            # Validate
            model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for img_e, sym_e, cli_e, mask, labels in val_loader:
                    img_e, sym_e, cli_e = img_e.to(device), sym_e.to(device), cli_e.to(device)
                    mask, labels = mask.to(device), labels.to(device)
                    
                    logits = model(img_e, sym_e, cli_e, mask)
                    val_correct += (logits.argmax(1) == labels).sum().item()
                    val_total += labels.size(0)
            
            val_acc = val_correct / val_total if val_total > 0 else 0
            
            if epoch % 5 == 0 or epoch == 1:
                logger.info(
                    f"Epoch {epoch}/{fusion_config['epochs']} | "
                    f"Train Loss: {train_loss/train_total:.4f}, "
                    f"Acc: {train_correct/train_total:.4f} | "
                    f"Val Acc: {val_acc:.4f}"
                )
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    model.state_dict(),
                    os.path.join(models_dir, f"fusion_{model_name}.pth")
                )
        
        logger.info(f"{model_name} fusion: Best val accuracy = {best_val_acc:.4f}")
        results[model_name] = {
            "best_val_accuracy": best_val_acc,
            "model_path": os.path.join(models_dir, f"fusion_{model_name}.pth"),
        }
    
    save_metrics(results, os.path.join(reports_dir, "fusion_model_metrics.json"))
    return results


if __name__ == "__main__":
    train_fusion_model()
