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

All evaluation reports and metrics are strictly calculated from validated test partitions:
- **Radiology Stream (ResNet-18):** Macro F1 > 0.88, ROC-AUC > 0.94 on held-out test radiography.
- **Symptom Classifier:** Multi-class log-loss minimization across binary indicators.
- **Multimodal Fusion:** Gated cross-modality synthesis improves classification resilience when individual modalities suffer from noise or absence.

---

## 📑 Academic Project Report & Documentation

The comprehensive university-level academic project report and architectural diagrams are available in the [`docs/`](./docs) directory:
- **Comprehensive Project Report (PDF):** [`docs/MediFusion_AI_Comprehensive_Project_Report.pdf`](./docs/MediFusion_AI_Comprehensive_Project_Report.pdf)
- **Comprehensive Project Report (Word DOCX):** [`docs/MediFusion_AI_Comprehensive_Project_Report.docx`](./docs/MediFusion_AI_Comprehensive_Project_Report.docx)
- **High-Resolution Architecture Diagram:** [`docs/medifusion_system_architecture.png`](./docs/medifusion_system_architecture.png)

---

## 👨‍💻 My Contribution — Harshit Raj

### Contribution Summary Card

| Attribute | Details |
| :--- | :--- |
| **Contributor** | **Harshit Raj** |
| **Role** | **Multimodal AI & Fusion Integration** |
| **Focus Areas** | Multimodal Deep Learning, Gated Cross-Modal Attention, Feature Representation, Cross-Modal Explainability (XAI) |
| **Core Modules & Files** | [`src/models/fusion_model.py`](./src/models/fusion_model.py), [`src/training/train_fusion.py`](./src/training/train_fusion.py), [`src/explainability/multimodal_explain.py`](./src/explainability/multimodal_explain.py), [`api/schemas.py`](./api/schemas.py), [`api/main.py`](./api/main.py), [`frontend/multimodal.html`](./frontend/multimodal.html), [`frontend/js/multimodal.js`](./frontend/js/multimodal.js), [`tests/test_audit_fusion.py`](./tests/test_audit_fusion.py) |
| **Primary Methodologies** | Learnable Gating Networks ($\sigma(W_g \mathbf{h} + \mathbf{b}) \odot \mathbf{m}$), Multi-Head Cross-Modal Attention ($4$ heads, $d_{\text{model}}=64$), Shapley Marginal Attributions, Grad-CAM, TreeSHAP |

---

### Key Technical Contributions & Deliverables

#### 1. Multimodal AI & Unified Fusion Pipeline Architecture
- **Unified Pipeline Design:** Designed, audited, and integrated the end-to-end multimodal fusion pipeline in [`api/main.py`](./api/main.py) and [`src/models/fusion_model.py`](./src/models/fusion_model.py), unifying heterogeneous clinical data streams:
  - **Vision Pathway:** 512-dimensional convolutional feature embeddings extracted from ResNet-18 chest radiographs.
  - **Symptom Pathway:** 15-dimensional differential symptom indicator representations.
  - **Clinical Telemetry Pathway:** 22-dimensional structured physiological vitals (blood pressure, heart rate, oxygen saturation, temperature, respiratory rate).
- **Dynamic Modality Availability Masking:** Integrated an adaptive binary availability mask $\mathbf{m} \in \{0, 1\}^3$ allowing the system to perform robust inference across incomplete or single-modality patient inputs (radiograph-only, symptoms-only, or full tri-modality telemetry) without pipeline failure or artificial data imputation.

#### 2. Gated Attention Fusion & Modality Weighting
- **Dynamic Gating Network Implementation:** Developed and integrated the `GatedAttentionFusion` network in [`src/models/fusion_model.py`](./src/models/fusion_model.py) featuring modality-specific linear projections and dynamic normalization:
  $$\mathbf{g} = \sigma(W_g [\mathbf{h}_{\text{img}}, \mathbf{h}_{\text{sym}}, \mathbf{h}_{\text{cli}}] + \mathbf{b}_g) \odot \mathbf{m}, \quad \tilde{g}_i = \frac{g_i}{\sum_{j=1}^M g_j + \epsilon}$$
  ensuring all gating weights are dynamically inferred, mathematically valid, and sum to 1.0 across active modalities rather than relying on arbitrary fixed constants.
- **Resolution of Conflicting Signals Anomaly:** Investigated and resolved the critical architectural flaw where conflicting single-modality outputs (Normal radiograph 83% vs. Malaria symptoms 99% / vitals 96%) previously triggered an erroneous 99% Pneumonia prediction. Overhauled synthetic training generation in [`src/training/train_fusion.py`](./src/training/train_fusion.py) to eliminate mask-shortcut memorization and enforce realistic cross-modality pairings (e.g., clear pulmonary fields paired with acute systemic febrile infections).

#### 3. Cross-Modal Interaction & Shared Feature Representation
- **Cross-Modal Multi-Head Attention:** Built an inter-modality multi-head attention block ($4$ heads, $d_{\text{model}}=64$) capturing rich cross-stream dependencies and contextual interactions between radiographic visual features and tabular clinical parameters.
- **Feature Scaling & Distribution Alignment:** Diagnosed and corrected a severe feature scaling mismatch where the clinical pathway was previously trained on unscaled vitals but evaluated against `StandardScaler`-transformed inputs. Harmonized the preprocessing pipeline across training and inference routines.

#### 4. Fusion-Level Explainable AI (XAI) Suite
- **Comprehensive Explainability Module:** Engineered [`src/explainability/multimodal_explain.py`](./src/explainability/multimodal_explain.py) to bring transparency to the fusion decision-making process:
  - **Modality Concordance Evaluation:** Designed automated clinical concordance detection categorizing inputs into `Concordant Consensus` vs. `Discordant Signals Resolved`, delivering evidence-based clinical reasoning when imaging and systemic telemetry diverge.
  - **Leave-One-Out Marginal Attributions:** Implemented cooperative Shapley-style marginal impact scoring ($\Delta_i = \max(0, P_{\text{full}}(c^*) - P_{-i}(c^*))$) to quantify the exact contribution of each modality to the winning diagnostic class.
  - **$3 \times 3$ Attention Flow Extraction:** Exposing full inter-modality attention weight matrices through API schemas to visualize inter-stream information routing.
  - **Dual-Pathway Explainability Hook:** Coupled fusion outputs with Grad-CAM spatial heatmaps for thoracic radiograph localization and SHAP TreeExplainer feature importance rankings for vital signs.

#### 5. Frontend Multimodal UI & Integration Testing
- **Interactive Multimodal Reasoning Interface:** Connected the frontend ([`frontend/multimodal.html`](./frontend/multimodal.html), [`frontend/js/multimodal.js`](./frontend/js/multimodal.js)) to dynamic API responses—eliminating hardcoded weights, rendering real-time gating bars, concordance status banners, interactive $3 \times 3$ attention heatmaps, and dual deep-dive XAI tabs.
- **Automated Test Suite:** Authored [`tests/test_audit_fusion.py`](./tests/test_audit_fusion.py) covering conflicting signal resolution, concordant pneumonia synthesis, and single-modality masking under pytest.

---

## 👤 Author & Acknowledgements

- **Developer:** [Harshit Raj](https://github.com/sarrafharshit411-ops)
- **Repository:** [https://github.com/sarrafharshit411-ops/MEDifusion-](https://github.com/sarrafharshit411-ops/MEDifusion-)
- **Datasets:**
  - *Chest X-Ray Images (Pneumonia)* by Paul Mooney (Kaggle)
  - *Disease Symptom Prediction Dataset* (Kaggle)

---

## 📄 License

This project is licensed strictly for **Educational and Academic Research Purposes**. It is not licensed or approved for clinical diagnosis or direct medical intervention.


