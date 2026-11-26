# MediFusion AI 🧬
### *Multimodal Clinical Decision-Support & Disease-Assessment System*

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-185a9d.svg?style=flat)](https://xgboost.readthedocs.io/)
[![Explainability](https://img.shields.io/badge/Explainability-Grad--CAM%20%7C%20SHAP-blueviolet.svg?style=flat)](https://github.com/slundberg/shap)
[![License](https://img.shields.io/badge/License-Educational%20Use%20Only-orange.svg?style=flat)](#license)

---

## ⚠️ Important Mentions & Medical Disclaimers

> [!CAUTION]
> **EDUCATIONAL & RESEARCH PROTOTYPE ONLY**  
> MediFusion AI is an academic and research proof-of-concept designed to explore multimodal deep learning, explainability, and clinical decision support.  
> - **NOT A MEDICAL DEVICE:** This system is **NOT** an FDA/CE-cleared diagnostic medical device.
> - **NOT A SUBSTITUTE FOR PROFESSIONAL JUDGMENT:** It should **NEVER** be used as a definitive diagnostic tool or as a replacement for clinical consultation with a qualified, licensed healthcare professional.
> - **EMERGENCY NOTICE:** In the event of a medical emergency, immediately seek urgent medical attention from emergency services.

### Dataset Acknowledgement & Architecture Disclosures
- **Cohort Notice:** The imaging dataset (Chest X-Rays) and symptom datasets originate from independent open-source repositories and are **not patient-paired** (i.e., they do not share a common patient ID or longitudinal electronic health record).
- **Synthetic Physiological Features:** The clinical vitals modality (blood pressure, oxygen saturation, temperature, respiratory rate, etc.) integrates synthetic parameters to demonstrate how multimodal gating handles tabular clinical vitals alongside unstructured imaging.
- **Multimodal Fusion Demonstration:** The fusion network evaluates architectural feasibility (Gated Attention Fusion with modality masking for handling missing clinical data) rather than an FDA-validated clinical pipeline.

---

## 🌟 Key Highlights & Capabilities

MediFusion AI is built on a modular tri-stream deep learning architecture that fuses unstructured radiologic imagery with structured clinical biomarkers and differential symptom indicators:

1. **Radiological Assessment (ResNet-18 + Grad-CAM)**
   - High-throughput chest radiograph classification (Normal vs. Pneumonia) fine-tuned on clinical X-rays.
   - Built-in **Radiograph Validation Filter** (`image_validator.py`) to detect and reject invalid/non-radiologic uploads, corrupted files, and inverted contrast errors.
   - **Grad-CAM (Gradient-weighted Class Activation Mapping)** generates interpretable spatial heatmaps highlighting the exact pulmonary regions driving model inference.

2. **General Disease 41-Class Diagnostic Ensemble**
   - Comprehensive multi-class diagnostic engine covering **41 systemic diseases** and **132 distinct symptoms**.
   - Generates confidence-ranked differential diagnoses, matched symptom breakdowns, and probability distributions.

3. **Integrated Clinical Pharmacology & Knowledge Base**
   - Actionable clinical pharmacology recommendations, disease overviews, lifestyle precautions, and standard treatment context extracted from clinical knowledge bases.

4. **Gated Attention Multimodal Fusion**
   - Learnable cross-modality attention mechanism fusing imaging embeddings (512-d), symptom vectors (15-d), and clinical vitals (22-d).
   - Robust **Modality Masking**: dynamically handles partial or missing clinical data (e.g., assessing a patient when only symptoms or only imaging is available).

5. **SHAP Tree Interpretability**
   - SHAP (SHapley Additive exPlanations) values decompose tabular clinical models to attribute exact credit and risk factors to specific symptoms and physiological vitals.

6. **Interactive Multi-Page Web Platform**
   - Modern medical dashboard built with Vanilla CSS/JS and Chart.js, featuring real-time diagnostic reporting, interactive symptom pickers, and visual heatmaps.

---

## 🏗️ System Architecture

```
                                  ┌────────────────────────┐
                                  │   Patient Evaluation   │
                                  └───────────┬────────────┘
                                              │
         ┌────────────────────────────────────┼────────────────────────────────────┐
         │                                    │                                    │
         ▼                                    ▼                                    ▼
┌──────────────────┐               ┌──────────────────────┐             ┌──────────────────────┐
│   Chest X-Ray    │               │     132 Symptoms     │             │ 7 Physiological Vitals│
│ (DICOM/PNG/JPEG) │               │  (Differential Match)│             │ (BP, O2, HR, Temp, RR)│
└────────┬─────────┘               └──────────┬───────────┘             └──────────┬───────────┘
         │                                    │                                    │
    Validation                           Preprocessing                         Standard
    & ResNet18                            & Embedding                           Scaling
         │                                    │                                    │
   512-d Embedding                      15-d Embedding                       22-d Embedding
         │                                    │                                    │
         └────────────────────────┬───────────┴────────────────────────────────────┘
                                  │
                                  ▼
                     ┌───────────────────────────┐
                     │   Gated Attention Fusion  │  ◄── Modality Availability Mask
                     │  (Cross-Modal Synthesis)  │
                     └────────────┬──────────────┘
                                  │
                                  ▼
                     ┌───────────────────────────┐
                     │   Differential Diagnosis   │
                     │  & Clinical Risk Profile  │
                     └────────────┬──────────────┘
                                  │
                 ┌────────────────┴────────────────┐
                 ▼                                 ▼
    ┌───────────────────────────┐     ┌───────────────────────────┐
    │     Grad-CAM Heatmap      │     │      SHAP Risk Credit     │
    │  (Radiology Localization) │     │  (Feature-Level Impact)   │
    └───────────────────────────┘     └───────────────────────────┘
```

---

## 🔬 Multimodal Fusion Audit & Explainable AI (XAI) Architecture

### 1. Root Cause Analysis: Conflicting Signals vs. Artificial Shortcuts
During rigorous pipeline auditing, a critical prediction failure was diagnosed where individual single-modality models produced conflicting predictions:
- **Radiology:** NORMAL (83%)
- **Symptoms:** Malaria (99%)
- **Clinical Vitals:** Malaria (96%)

Yet the fusion model predicted **Pneumonia with an artificial 99% confidence score** and hardcoded gating weights (`45% / 35% / 20%`).

#### Root Causes Identified:
1. **Mask-Memorization Shortcut in Data Generation:**  
   In the legacy fusion training routine (`train_fusion.py`), the all-active mask `[1, 1, 1]` was generated *exclusively* for Pneumonia samples (Normal used `[1, 0, 0]`, while Malaria and Typhoid used `[0, 1, 1]`). The fusion network bypassed the underlying representations entirely, memorizing the binary mask pattern: `image_present + symptoms_present == Pneumonia` with 100% correlation.
2. **Hardcoded Frontend Gating Weights:**  
   `frontend/js/multimodal.js` previously contained hardcoded gating values (`wImg = 45%, wSym = 35%, wCli = 20%`) rather than extracting learned gating activations from the model.
3. **Clinical Feature Scaling Mismatch:**  
   The training pipeline passed raw unscaled vitals (e.g., systolic BP ~120, HR ~80) to the clinical embedding branch, whereas inference in `api/main.py` passed `StandardScaler`-transformed vitals (mean ~0, std ~1), causing severe distribution shift in the gating network.

### 2. Architectural & Mathematical Fixes
1. **Dynamic Gating Normalization:**
   The fusion network applies a modality-wise gating layer with sigmoid activation, followed by masked dynamic normalization:
   $$\mathbf{g} = \sigma(W_g [\mathbf{h}_{\text{img}}, \mathbf{h}_{\text{sym}}, \mathbf{h}_{\text{cli}}] + \mathbf{b}_g) \odot \mathbf{m}$$
   $$\tilde{g}_i = \frac{g_i}{\sum_{j=1}^M g_j + \epsilon}$$
   where $\mathbf{m} \in \{0, 1\}^3$ is the availability mask, ensuring weights are mathematically valid and sum to $1.0$ across active modalities.
2. **Cross-Modal Attention Extraction:**
   A multi-head attention mechanism ($d_{\text{model}}=64, \text{heads}=4$) computes inter-modality attention flow:
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
   The resulting $3 \times 3$ attention weight matrix is extracted and exposed via the API to visualize cross-modal information transfer.
3. **Cross-Modality Data Synthesis:**
   Retrained `GatedAttentionFusion` on cross-modality pairings (e.g., normal radiographs paired with malaria/typhoid systemic vitals, matching clinical reality where non-respiratory infections present with clear lungs).

### 3. Integrated Multimodal Explainability (XAI) Suite
The platform features an end-to-end Explainable AI suite providing transparency at both the fusion and pathway levels:
- **Modality Concordance Analysis:** Automatically evaluates agreement across independent streams, classifying the diagnostic context as:
  - `Concordant Consensus`: All active streams mutually corroborate the diagnostic hypothesis.
  - `Discordant Signals Resolved`: Conflicting modality indicators (e.g., clear pulmonary fields vs. acute systemic febrile markers) are explicitly highlighted with automated clinical resolution reasoning.
  - `Single-Modality Assessment`: Graceful single-stream inference with modality masking.
- **Marginal Modality Attribution (Leave-One-Out Shapley Impact):** Dynamically measures the marginal contribution of each modality to the winning class probability:
  $$\Delta_i = \max(0, P_{\text{full}}(c^*) - P_{-i}(c^*))$$
- **Interactive $3 \times 3$ Cross-Attention Heatmap:** Real-time visual matrix detailing how the Vision, Symptom, and Clinical vitals streams attend to one another.
- **Dual-Pathway Deep Dive:**
  - *Radiology Pathway:* Grad-CAM heatmaps highlighting pulmonary opacities and consolidations.
  - *Biomarker Pathway:* SHAP feature importance rankings quantifying the impact of individual symptoms and vitals.
- **Clinical Prototype Disclosures:** Complete transparency regarding non-patient-paired training cohorts and synthetic physiological vitals.

---

| Domain | Technology | Details |
|--------|-----------|---------|
| **Deep Learning** | PyTorch & Torchvision | ResNet-18 Transfer Learning & Custom Gated Attention |
| **Machine Learning** | XGBoost & Scikit-Learn | Calibrated multi-class classification and scalers |
| **Model Explainability**| Grad-CAM & SHAP | Spatial activations and Shapley feature attribution |
| **Backend API** | FastAPI, Uvicorn, Pydantic | Asynchronous, validated RESTful endpoints |
| **Data Processing** | NumPy, Pandas, Joblib | Efficient tensor manipulation & pipeline serialization |
| **Frontend UI** | HTML5, Modern Vanilla CSS, JS | Multi-page responsive UI with Chart.js visualization |
| **Containerization** | Docker & Docker Compose | Multi-platform containerized deployment |

---

## 📁 Repository Structure

```
.
├── api/
│   ├── main.py                      # FastAPI application entry point
│   ├── model_loader.py              # Singleton model registry & cache
│   ├── general_disease.py           # 41-class general disease API routes
│   └── schemas.py                   # Pydantic request/response schemas
├── configs/
│   └── config.yaml                  # Global training and model hyperparameters
├── data/
│   ├── Training.csv                 # 41-disease training dataset
│   ├── Testing.csv                  # Multi-class evaluation dataset
│   ├── disease_medicine_recommendation.csv # Pharmacology guidance data
│   ├── disease_symptom.csv          # Symptom-to-disease mappings
│   └── medications.csv              # Medication reference catalog
├── frontend/
│   ├── index.html                   # MediFusion landing page & system overview
│   ├── image-analysis.html          # Radiology Grad-CAM workspace
│   ├── symptom-analysis.html        # 132-symptom diagnosis & pharmacology hub
│   ├── multimodal.html              # Multi-stream gated fusion workspace
│   ├── insights.html                # Model performance & SHAP analytics
│   ├── about.html                   # Research documentation & ethics
│   ├── css/                         # Modular CSS design system
│   └── js/                          # Client-side API connectors & chart rendering
├── models/
│   ├── image_model.pth              # Trained ResNet-18 radiograph weights
│   ├── symptom_model.joblib         # XGBoost symptom classifier
│   ├── clinical_model.joblib        # XGBoost clinical vitals classifier
│   ├── fusion_gated.pth             # Gated Attention Multimodal network
│   ├── general_disease_ensemble.joblib # 41-disease ensemble model
│   ├── disease_knowledge_base.json  # Comprehensive clinical knowledge base
│   └── symptoms_132_metadata.json   # Symptom definitions and metadata
├── src/
│   ├── data/                        # Dataset loaders and transformers
│   ├── explainability/              # Grad-CAM and SHAP implementations
│   ├── models/                      # PyTorch and XGBoost architecture definitions
│   ├── training/                    # Training routines and metrics computation
│   └── utils/                       # Radiograph validator and helpers
├── Dockerfile                       # Production container specification
├── docker-compose.yml               # Service orchestration
├── requirements.txt                 # Project dependencies
└── run_training.py                  # End-to-end model training orchestrator
```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/sarrafharshit411-ops/MEDifusion-.git
cd MEDifusion-
```

### 2. Set Up Environment & Install Dependencies
It is recommended to use Python 3.10+ in a virtual environment:
```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Run the Web Application
Start the FastAPI server:
```bash
uvicorn api.main:app --reload --port 8000
```

Once running, launch your browser and navigate to:
- **Web Dashboard:** [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc API Documentation:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 🔌 API Endpoints Reference

| Route | Method | Description |
|---|---|---|
| `/api/health` | `GET` | Health status and verification of loaded model weights |
| `/api/predict/image` | `POST` | Chest X-Ray inference (Normal vs. Pneumonia) with radiograph validation |
| `/api/predict/symptoms` | `POST` | Core symptom-based multi-class probability assessment |
| `/api/predict/clinical` | `POST` | Disease prediction combining symptoms with physiological vitals |
| `/api/predict/multimodal` | `POST` | Unified multimodal inference via Gated Attention Fusion |
| `/api/explain/image` | `POST` | Generates Grad-CAM spatial activation heatmaps |
| `/api/explain/clinical` | `POST` | Computes SHAP values and feature attribution charts |
| `/api/general-disease/predict` | `POST` | 41-class differential diagnosis from 132 symptoms + pharmacology |
| `/api/general-disease/symptoms`| `GET` | Comprehensive list of recognized symptoms and categorical metadata |

---

## 🐳 Docker Deployment

To launch the full system inside an isolated container:

```bash
docker-compose up --build
```
The application will be accessible at `http://localhost:8000`.

---

## 📊 Model Evaluation & Metrics

The metrics below are transcribed from the repository's committed evaluation artifacts and the IMDAI Milestone 4 report. They have not been independently re-run during this documentation update. Results use different datasets and evaluation splits and are not directly comparable.

| Model / Task | Accuracy | Macro F1 | ROC-AUC | Evaluation context |
|---|---:|---:|---:|---|
| Chest X-ray (ResNet-18) | 85.10% | 82.38% | 95.40% | Held-out test set, n=624 |
| Brain tumour MRI (ResNet-18) | 95.50% | 95.42% | 98.23% | Four classes, support 400/class |
| Skin lesion / HAM10000 (ResNet-18) | 53.85% | 50.95% | 91.74% | Seven imbalanced classes |
| Diabetic retinopathy (ResNet-18) | 10.79% | 17.25% | 71.90% | Five classes; No-DR dominates |
| Blood-cell classification (ResNet-18) | 87.86% | 88.15% | 95.41% | Four classes |
| Symptom model (XGBoost) | 87.20% | 87.20% | 96.91% | Test set, n=1,000 |
| Clinical model (XGBoost) | 87.10% | 87.08% | 97.09% | Test set, n=1,000; synthetic clinical features |
| Fusion — concatenation | Not reported as test | Not reported | Not reported | Best validation accuracy 93.24% |
| Fusion — gated attention | Not reported as test | Not reported | Not reported | Best validation accuracy 93.80% |

**Interpretation and limitations**
- The 93.80% gated-fusion figure is validation accuracy, not held-out test accuracy.
- The retinopathy pathway is currently weak: the stored report records 10.79% accuracy and zero precision, recall and F1 for the No-DR class.
- Skin-lesion performance is modest and affected by class imbalance.
- Chest X-ray accuracy hides class-level differences: Normal recall is 61.11%, while Pneumonia recall is 99.49%.
- Imaging and symptom datasets are independent and are not patient-paired. Physiological clinical values used for fusion include synthetic data. Fusion results therefore demonstrate an academic architecture, not clinical effectiveness.

---

## 📑 Academic Project Report & Documentation

The current IMDAI Milestone 4 report is the single canonical project report:

- **Milestone 4 Final Report (PDF):** [`docs/MediFusion_AI_Milestone4_Final_Report.pdf`](./docs/MediFusion_AI_Milestone4_Final_Report.pdf)
- **System architecture:** [`docs/medifusion_system_architecture.png`](./docs/medifusion_system_architecture.png)

Superseded comprehensive-report files have been removed. The report describes the project as an academic/research prototype, not a clinically validated diagnostic device.

---

## 👨‍💻 Developer & Contributor

**Sole developer and contributor: Harshit Raj**  
**Enrollment:** S24CSEU0733  
**GitHub:** [sarrafharshit411-ops](https://github.com/sarrafharshit411-ops)

MediFusion AI is maintained as an individual project in this repository. This repository-level ownership statement does not make claims about historical authorship of every source file; see Git commit history for change-level attribution.

### Project ownership and technical responsibilities

| Area | Repository implementation |
|---|---|
| Multimodal architecture | `src/models/fusion_model.py` — concatenation baseline and gated-attention fusion |
| Fusion training | `src/training/train_fusion.py` — embedding extraction, synthetic cross-modal pairing, modality-mask generation and validation |
| Fusion explainability | `src/explainability/multimodal_explain.py` — modality concordance, leave-one-out marginal attribution, gating outputs, attention matrix and pathway explanation hooks |
| API integration | `api/main.py`, `api/schemas.py` — multimodal inference route and structured response payload |
| Frontend integration | `frontend/multimodal.html`, `frontend/js/multimodal.js` — multimodal workflow and XAI visualisation |
| Audit tests | `tests/test_audit_fusion.py` — conflicting-signal, concordant-signal and missing-modality scenarios |

### Fusion implementation details

- Embedding inputs: 512-dimensional image, 15-dimensional symptom and 22-dimensional clinical representations.
- The current `GatedAttentionFusion` class defaults to a 128-dimensional shared projection and two attention heads. These values reflect the current source-code defaults.
- Learnable per-modality sigmoid gates are masked by modality availability and normalized across active streams.
- Cross-modal attention returns an attention matrix for the three modality tokens.
- The concatenation model is retained as a baseline.
- The training pipeline uses synthetic, non-patient-paired cross-modal combinations. These are for architecture demonstration only.

### Repository structure

The source code, model artifacts, evaluation reports and documentation are maintained in this repository. See the directory tree above and the linked Milestone 4 report for the detailed system description.

---

## 👤 Dataset Acknowledgements

- **Chest X-Ray Images (Pneumonia):** Paul Mooney, Kaggle.
- **Symptom-Based Disease Prediction Dataset:** MiltonMacGyver, Kaggle.
- Other image-domain dataset references and evaluation details are documented in the source and report artifacts.
- Dataset creators retain their respective rights and terms. This repository's project ownership does not imply ownership of the underlying datasets.
## 📄 License

This project is licensed strictly for **Educational and Academic Research Purposes**. It is not licensed or approved for clinical diagnosis or direct medical intervention.


