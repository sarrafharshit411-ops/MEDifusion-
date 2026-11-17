/**
 * MediFusion AI — Benchmarks & Performance Analytics Controller
 * Dynamically loads all model evaluation metrics, renders comparative tables,
 * per-class diagnostic breakdowns, and interactive confusion matrices / ROC curves.
 */

(function () {
    "use strict";

    const MODEL_META = {
        "Brain Tumor MRI": {
            id: "brain_tumor",
            icon: "🧠",
            domain: "Neuro-Oncology",
            architecture: "ResNet-18 (AMP FP16)",
            dataset: "Kaggle: Brain Tumor MRI (7,023 scans)",
            classes: ["Glioma", "Meningioma", "No Tumor", "Pituitary"],
            cmImage: "/static/reports/brain_tumor_confusion_matrix.png",
            rocImage: "/static/reports/brain_tumor_roc_curves.png",
            color: "#f59e0b",
            status: "Complete",
        },
        "Blood Cell Classification": {
            id: "blood_cell",
            icon: "🩸",
            domain: "Hematology Cytology",
            architecture: "ResNet-18 (AMP FP16)",
            dataset: "Kaggle: Blood Cell Images (12,444 samples)",
            classes: ["Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"],
            cmImage: "/static/reports/blood_cell_confusion_matrix.png",
            rocImage: "/static/reports/blood_cell_roc_curves.png",
            color: "#8b5cf6",
            status: "Complete",
        },
        "Image-Only": {
            id: "chest_xray",
            name: "Chest Radiography (Pneumonia)",
            icon: "🫁",
            domain: "Pulmonology",
            architecture: "ResNet-18",
            dataset: "Kaggle: Chest X-Ray Pneumonia (5,856 radiographs)",
            classes: ["NORMAL", "PNEUMONIA"],
            cmImage: "/static/reports/image_confusion_matrix.png",
            rocImage: "/static/reports/image_roc_curves.png",
            color: "#0d9488",
            status: "Complete",
        },
        "Symptom-Only": {
            id: "symptoms",
            name: "Symptom Intelligence",
            icon: "📋",
            domain: "Infectious Disease Triage",
            architecture: "XGBoost Multi-Softprob",
            dataset: "Clinical Symptom Matrix (15 features)",
            classes: ["Malaria", "Pneumonia", "Typhoid"],
            cmImage: "/static/reports/symptom_confusion_matrix.png",
            rocImage: "/static/reports/symptom_roc_curves.png",
            color: "#0284c7",
            status: "Complete",
        },
        "Clinical-Only": {
            id: "clinical",
            name: "Clinical Vitals Assessment",
            icon: "🩺",
            domain: "Critical Care Vitals",
            architecture: "Standardized XGBoost",
            dataset: "Physiological Distress Signals (7 vitals)",
            classes: ["Malaria", "Pneumonia", "Typhoid"],
            cmImage: "/static/reports/symptom_confusion_matrix.png",
            rocImage: "/static/reports/symptom_roc_curves.png",
            color: "#6366f1",
            status: "Complete",
        },
        "Skin Cancer (HAM10000)": {
            id: "skin_cancer",
            icon: "🔬",
            domain: "Dermatopathology",
            architecture: "ResNet-18 (AMP FP16, Weighted)",
            dataset: "HAM10000 Dermatoscopy (10,015 lesions)",
            classes: ["Actinic Keratoses", "BCC", "Benign Keratosis", "Dermatofibroma", "Melanoma", "Nevi", "Vascular"],
            cmImage: "/static/reports/skin_cancer_confusion_matrix.png",
            rocImage: "/static/reports/skin_cancer_roc_curves.png",
            color: "#ef4444",
            status: "Complete",
        },
        "Diabetic Retinopathy": {
            id: "retinopathy",
            icon: "👁️",
            domain: "Ophthalmology Fundus",
            architecture: "ResNet-18 (AMP FP16, OneCycleLR)",
            dataset: "Kaggle: Diabetic Retinopathy Arranged (35,126 fundus scans)",
            classes: ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"],
            cmImage: "/static/reports/retinopathy_confusion_matrix.png",
            rocImage: "/static/reports/retinopathy_roc_curves.png",
            color: "#3b82f6",
            status: "Complete",
        }
    };

    let allMetricsData = null;
    let selectedModelKey = "Brain Tumor MRI";

    document.addEventListener("DOMContentLoaded", () => {
        initTabs();
        loadAllMetrics();
        setupExportButtons();
    });

    async function loadAllMetrics() {
        try {
            const res = await fetch("/static/reports/all_model_metrics.json?v=" + Date.now());
            if (res.ok) {
                allMetricsData = await res.json();
            } else {
                console.warn("Could not load reports JSON directly, using static fallback.");
                allMetricsData = getStaticFallbackMetrics();
            }
        } catch (e) {
            console.warn("Network error loading metrics JSON, using fallback data:", e);
            allMetricsData = getStaticFallbackMetrics();
        }

        renderKPICards(allMetricsData);
        renderBenchmarksTable(allMetricsData);
        renderModelDetail(selectedModelKey, allMetricsData);
    }

    function renderKPICards(data) {
        let maxAcc = 0;
        let maxAccModel = "";
        let maxRoc = 0;
        let maxRocModel = "";
        let totalModels = Object.keys(data).length;
        let totalSamples = 0;

        for (const [name, m] of Object.entries(data)) {
            if (m.accuracy && m.accuracy > maxAcc) {
                maxAcc = m.accuracy;
                maxAccModel = name.replace("-Only", "");
            }
            if (m.roc_auc && m.roc_auc > maxRoc) {
                maxRoc = m.roc_auc;
                maxRocModel = name.replace("-Only", "");
            }
            if (m.per_class) {
                for (const c of Object.values(m.per_class)) {
                    totalSamples += c.support || 0;
                }
            }
        }

        const kpiMaxAcc = document.getElementById("kpiMaxAcc");
        const kpiMaxAccLabel = document.getElementById("kpiMaxAccLabel");
        const kpiMaxRoc = document.getElementById("kpiMaxRoc");
        const kpiMaxRocLabel = document.getElementById("kpiMaxRocLabel");
        const kpiTotalModels = document.getElementById("kpiTotalModels");
        const kpiTotalSamples = document.getElementById("kpiTotalSamples");

        if (kpiMaxAcc) kpiMaxAcc.textContent = `${(maxAcc * 100).toFixed(1)}%`;
        if (kpiMaxAccLabel) kpiMaxAccLabel.textContent = `Top Model: ${maxAccModel}`;
        if (kpiMaxRoc) kpiMaxRoc.textContent = maxRoc.toFixed(4);
        if (kpiMaxRocLabel) kpiMaxRocLabel.textContent = `Top Discrimination: ${maxRocModel}`;
        if (kpiTotalModels) kpiTotalModels.textContent = totalModels.toString();
        if (kpiTotalSamples) kpiTotalSamples.textContent = `${totalSamples.toLocaleString()} Scans`;
    }

    function renderBenchmarksTable(data) {
        const tbody = document.getElementById("benchmarksTableBody");
        if (!tbody) return;

        let rowsHtml = "";
        const priorityOrder = [
            "Brain Tumor MRI",
            "Blood Cell Classification",
            "Image-Only",
            "Symptom-Only",
            "Clinical-Only",
            "Skin Cancer (HAM10000)",
            "Diabetic Retinopathy"
        ];

        // Gather all keys including any dynamically returned
        const keys = [...new Set([...priorityOrder, ...Object.keys(data)])];

        for (const key of keys) {
            const m = data[key];
            const meta = MODEL_META[key] || {
                icon: "🩺",
                domain: "General Medical",
                architecture: m?.model || "Machine Learning",
                dataset: "Medical Dataset",
                classes: [],
                color: "#0d9488",
                status: m ? "Evaluated" : "Pending"
            };

            const displayName = meta.name || key;
            const hasData = m && m.accuracy !== undefined;

            const acc = hasData ? (m.accuracy * 100).toFixed(1) + "%" : "Pending";
            const prec = hasData && m.precision_macro ? (m.precision_macro * 100).toFixed(1) + "%" : "—";
            const recall = hasData && m.recall_macro ? (m.recall_macro * 100).toFixed(1) + "%" : "—";
            const f1 = hasData && m.f1_macro ? (m.f1_macro * 100).toFixed(1) + "%" : "—";
            const rocAuc = hasData && m.roc_auc ? m.roc_auc.toFixed(4) : "—";
            const prAuc = hasData && m.pr_auc ? m.pr_auc.toFixed(4) : "—";

            const accVal = hasData ? m.accuracy * 100 : 0;
            const badgeClass = hasData ? (accVal > 90 ? "badge-success" : accVal > 80 ? "badge-teal" : "badge-amber") : "badge-slate";

            rowsHtml += `
                <tr class="benchmark-row ${key === selectedModelKey ? 'row-selected' : ''}" data-model="${key}">
                    <td>
                        <div style="display:flex; align-items:center; gap:0.65rem;">
                            <span style="font-size:1.3rem;">${meta.icon}</span>
                            <div>
                                <strong style="color:var(--text-highlight); font-size:0.92rem;">${displayName}</strong>
                                <div style="font-size:0.72rem; color:var(--text-muted);">${meta.domain}</div>
                            </div>
                        </div>
                    </td>
                    <td>
                        <span style="font-size:0.82rem; font-family:'JetBrains Mono', monospace; color:var(--text-secondary);">${meta.architecture}</span>
                    </td>
                    <td>
                        <div style="display:flex; align-items:center; gap:0.5rem;">
                            <span style="font-size:0.95rem; font-weight:800; color:${hasData ? 'var(--text-highlight)' : 'var(--text-muted)'}; font-family:'JetBrains Mono', monospace;">${acc}</span>
                            ${hasData ? `
                            <div style="width:50px; height:6px; background:rgba(255,255,255,0.08); border-radius:3px; overflow:hidden;">
                                <div style="height:100%; width:${accVal}%; background:${meta.color}; border-radius:3px;"></div>
                            </div>` : ''}
                        </div>
                    </td>
                    <td style="font-family:'JetBrains Mono', monospace; font-size:0.85rem;">${prec}</td>
                    <td style="font-family:'JetBrains Mono', monospace; font-size:0.85rem;">${recall}</td>
                    <td style="font-family:'JetBrains Mono', monospace; font-size:0.85rem; font-weight:700; color:var(--text-highlight);">${f1}</td>
                    <td style="font-family:'JetBrains Mono', monospace; font-size:0.85rem; color:${meta.color}; font-weight:700;">${rocAuc}</td>
                    <td style="font-family:'JetBrains Mono', monospace; font-size:0.85rem;">${prAuc}</td>
                    <td>
                        <span class="badge ${badgeClass}">${hasData ? 'Validated' : 'Training...'}</span>
                    </td>
                    <td>
                        <button class="btn btn-outline btn-sm btn-inspect" data-model="${key}" style="padding:0.3rem 0.65rem; font-size:0.75rem;">
                            Inspect
                        </button>
                    </td>
                </tr>
            `;
        }

        tbody.innerHTML = rowsHtml;

        // Attach click listeners to inspect rows
        tbody.querySelectorAll(".benchmark-row").forEach(row => {
            row.addEventListener("click", () => {
                const modelKey = row.dataset.model;
                selectModel(modelKey);
            });
        });
    }

    function initTabs() {
        const tabs = document.querySelectorAll(".model-selector-pill");
        tabs.forEach(tab => {
            tab.addEventListener("click", () => {
                tabs.forEach(t => t.classList.remove("active"));
                tab.classList.add("active");
                const modelKey = tab.dataset.model;
                selectModel(modelKey);
            });
        });
    }

    function selectModel(modelKey) {
        selectedModelKey = modelKey;

        // Update selected row in table
        document.querySelectorAll(".benchmark-row").forEach(r => {
            r.classList.toggle("row-selected", r.dataset.model === modelKey);
        });

        // Update tab pills
        document.querySelectorAll(".model-selector-pill").forEach(t => {
            t.classList.toggle("active", t.dataset.model === modelKey);
        });

        if (allMetricsData) {
            renderModelDetail(modelKey, allMetricsData);
        }
    }

    function renderModelDetail(modelKey, data) {
        const m = data[modelKey];
        const meta = MODEL_META[modelKey] || {
            icon: "🩺",
            domain: "Diagnostic Domain",
            architecture: "ResNet-18",
            dataset: "Clinical Dataset",
            classes: [],
            cmImage: "/static/reports/model_comparison.png",
            rocImage: "/static/reports/model_comparison.png",
            color: "#0d9488"
        };

        const titleEl = document.getElementById("detailModelTitle");
        const domainEl = document.getElementById("detailModelDomain");
        const archEl = document.getElementById("detailModelArch");
        const datasetEl = document.getElementById("detailModelDataset");
        const cmImgEl = document.getElementById("detailCmImg");
        const rocImgEl = document.getElementById("detailRocImg");
        const perClassBody = document.getElementById("detailPerClassBody");

        if (titleEl) titleEl.innerHTML = `<span style="font-size:1.4rem; margin-right:0.4rem;">${meta.icon}</span> ${meta.name || modelKey}`;
        if (domainEl) domainEl.textContent = meta.domain;
        if (archEl) archEl.textContent = meta.architecture;
        if (datasetEl) datasetEl.textContent = meta.dataset;

        // Visual matrices with cache busters
        const ts = Date.now();
        if (cmImgEl) {
            cmImgEl.src = `${meta.cmImage}?v=${ts}`;
            cmImgEl.alt = `${modelKey} Confusion Matrix`;
        }
        if (rocImgEl) {
            rocImgEl.src = `${meta.rocImage}?v=${ts}`;
            rocImgEl.alt = `${modelKey} ROC Curves`;
        }

        // Render Per-Class Table
        if (perClassBody) {
            if (m && m.per_class) {
                let html = "";
                for (const [clsName, stats] of Object.entries(m.per_class)) {
                    const p = (stats.precision * 100).toFixed(1);
                    const r = (stats.recall * 100).toFixed(1);
                    const f1 = (stats["f1-score"] !== undefined ? stats["f1-score"] : stats.f1) * 100;
                    const supp = stats.support !== undefined ? stats.support : "—";

                    html += `
                        <tr>
                            <td><strong style="color:var(--text-highlight);">${clsName}</strong></td>
                            <td style="font-family:'JetBrains Mono',monospace;">${p}%</td>
                            <td style="font-family:'JetBrains Mono',monospace;">${r}%</td>
                            <td style="font-family:'JetBrains Mono',monospace; font-weight:700; color:${meta.color};">${f1.toFixed(1)}%</td>
                            <td style="font-family:'JetBrains Mono',monospace; color:var(--text-muted);">${typeof supp === 'number' ? supp.toLocaleString() : supp}</td>
                        </tr>
                    `;
                }
                perClassBody.innerHTML = html;
            } else {
                perClassBody.innerHTML = `
                    <tr>
                        <td colspan="5" style="text-align:center; padding:1.5rem; color:var(--text-muted);">
                            ${meta.status === "In Training / Active" ? "Model is currently training on GPU. Per-class metrics will appear upon validation." : "No per-class breakdown available."}
                        </td>
                    </tr>
                `;
            }
        }
    }

    function setupExportButtons() {
        const btnExportJson = document.getElementById("btnExportJson");
        if (btnExportJson) {
            btnExportJson.addEventListener("click", () => {
                if (!allMetricsData) return;
                const blob = new Blob([JSON.stringify(allMetricsData, null, 2)], { type: "application/json" });
                const url = URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = url;
                a.download = `medifusion_ai_model_benchmarks_${new Date().toISOString().slice(0, 10)}.json`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            });
        }

        const btnPrintReport = document.getElementById("btnPrintReport");
        if (btnPrintReport) {
            btnPrintReport.addEventListener("click", () => {
                window.print();
            });
        }
    }

    function getStaticFallbackMetrics() {
        return {
            "Brain Tumor MRI": {
                "accuracy": 0.9550,
                "precision_macro": 0.9578,
                "recall_macro": 0.9550,
                "f1_macro": 0.9542,
                "roc_auc": 0.9823,
                "pr_auc": 0.9785,
                "per_class": {
                    "Glioma": { "precision": 0.932, "recall": 0.941, "f1-score": 0.936, "support": 300 },
                    "Meningioma": { "precision": 0.948, "recall": 0.925, "f1-score": 0.936, "support": 306 },
                    "No Tumor": { "precision": 0.985, "recall": 0.990, "f1-score": 0.987, "support": 405 },
                    "Pituitary": { "precision": 0.966, "recall": 0.964, "f1-score": 0.965, "support": 300 }
                }
            },
            "Blood Cell Classification": {
                "accuracy": 0.8786,
                "precision_macro": 0.9018,
                "recall_macro": 0.8785,
                "f1_macro": 0.8815,
                "roc_auc": 0.9541,
                "pr_auc": 0.9410,
                "per_class": {
                    "Eosinophil": { "precision": 0.892, "recall": 0.843, "f1-score": 0.867, "support": 623 },
                    "Lymphocyte": { "precision": 0.951, "recall": 0.965, "f1-score": 0.958, "support": 623 },
                    "Monocyte": { "precision": 0.912, "recall": 0.890, "f1-score": 0.901, "support": 623 },
                    "Neutrophil": { "precision": 0.852, "recall": 0.816, "f1-score": 0.834, "support": 624 }
                }
            },
            "Image-Only": {
                "accuracy": 0.8510,
                "precision_macro": 0.8981,
                "recall_macro": 0.8030,
                "f1_macro": 0.8238,
                "roc_auc": 0.9540,
                "pr_auc": 0.9656,
                "per_class": {
                    "NORMAL": { "precision": 0.986, "recall": 0.611, "f1-score": 0.755, "support": 234 },
                    "PNEUMONIA": { "precision": 0.810, "recall": 0.995, "f1-score": 0.893, "support": 390 }
                }
            },
            "Symptom-Only": {
                "accuracy": 0.8720,
                "precision_macro": 0.8721,
                "recall_macro": 0.8720,
                "f1_macro": 0.8720,
                "roc_auc": 0.9691,
                "pr_auc": 0.9433,
                "per_class": {
                    "Malaria": { "precision": 0.838, "recall": 0.841, "f1-score": 0.840, "support": 333 },
                    "Pneumonia": { "precision": 0.903, "recall": 0.895, "f1-score": 0.899, "support": 333 },
                    "Typhoid": { "precision": 0.875, "recall": 0.880, "f1-score": 0.878, "support": 334 }
                }
            },
            "Clinical-Only": {
                "accuracy": 0.8710,
                "precision_macro": 0.8707,
                "recall_macro": 0.8710,
                "f1_macro": 0.8708,
                "roc_auc": 0.9709,
                "pr_auc": 0.9450,
                "per_class": {
                    "Malaria": { "precision": 0.852, "recall": 0.865, "f1-score": 0.858, "support": 333 },
                    "Pneumonia": { "precision": 0.881, "recall": 0.872, "f1-score": 0.876, "support": 333 },
                    "Typhoid": { "precision": 0.879, "recall": 0.876, "f1-score": 0.877, "support": 334 }
                }
            },
            "Skin Cancer (HAM10000)": {
                "accuracy": 0.5385,
                "precision_macro": 0.4674,
                "recall_macro": 0.7691,
                "f1_macro": 0.5095,
                "roc_auc": 0.9174,
                "pr_auc": 0.8240,
                "per_class": {
                    "Melanoma": { "precision": 0.421, "recall": 0.782, "f1-score": 0.547, "support": 111 },
                    "Melanocytic Nevi": { "precision": 0.865, "recall": 0.463, "f1-score": 0.603, "support": 670 },
                    "Basal Cell Carcinoma": { "precision": 0.485, "recall": 0.892, "f1-score": 0.628, "support": 51 },
                    "Benign Keratosis": { "precision": 0.412, "recall": 0.684, "f1-score": 0.514, "support": 109 },
                    "Actinic Keratoses": { "precision": 0.280, "recall": 0.875, "f1-score": 0.424, "support": 32 },
                    "Vascular Lesions": { "precision": 0.385, "recall": 0.857, "f1-score": 0.531, "support": 14 },
                    "Dermatofibroma": { "precision": 0.429, "recall": 0.833, "f1-score": 0.566, "support": 12 }
                }
            },
            "Diabetic Retinopathy": {
                "accuracy": 0.1079,
                "precision_macro": 0.1256,
                "recall_macro": 0.4774,
                "f1_macro": 0.1725,
                "roc_auc": 0.7190,
                "pr_auc": 0.3832,
                "per_class": {
                    "No DR": { "precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 2572 },
                    "Mild": { "precision": 0.079, "recall": 0.780, "f1-score": 0.144, "support": 241 },
                    "Moderate": { "precision": 0.202, "recall": 0.139, "f1-score": 0.164, "support": 541 },
                    "Severe": { "precision": 0.101, "recall": 0.633, "f1-score": 0.174, "support": 79 },
                    "Proliferative DR": { "precision": 0.246, "recall": 0.835, "f1-score": 0.380, "support": 79 }
                }
            }
        };
    }
})();
