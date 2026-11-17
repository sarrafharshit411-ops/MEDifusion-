/**
 * MediFusion AI — Model Insights & Explainability Module
 * Enhanced with dynamic Advanced Imaging Model performance cards.
 */

(function () {
    "use strict";

    document.addEventListener("DOMContentLoaded", () => {
        setupInteractivePipeline();
        loadLiveExplainabilityPreview();
        loadAdvancedImageModelMetrics();
    });

    function setupInteractivePipeline() {
        const nodes = document.querySelectorAll(".pipeline-node-card");
        const detailBox = document.getElementById("pipelineNodeDetails");

        const descriptions = {
            vision: {
                title: "ResNet18 Vision Backbone",
                desc: "Processes 224×224 medical images through deep convolutional residual blocks. Frozen early layers maintain robust low-level spatial features while layer3/4 are fine-tuned for domain-specific pathology recognition. Extracts a dense 512-dimensional latent embedding used for both classification and Grad-CAM explainability.",
                tech: "PyTorch • Transfer Learning • Grad-CAM Hooked • 5 Imaging Domains",
            },
            symptoms: {
                title: "XGBoost Symptom Intelligence",
                desc: "Evaluates multi-label binary symptom vectors (15 dimensions) across gradient-boosted decision trees. Calculates conditional leaf distribution and provides direct interpretable SHAP game-theoretic feature contributions.",
                tech: "XGBoost 3.0 • Multi-Softprob • 15 Feature Space",
            },
            clinical: {
                title: "Standardized Clinical Signal Encoder",
                desc: "Scales continuous vital signs (SpO₂, heart rate, body temperature, blood pressures, respiratory rate) and joins them with patient symptom profiles to gauge acute physiological distress.",
                tech: "StandardScaler • XGBoost • SHAP TreeExplainer",
            },
            fusion: {
                title: "Gated Multi-Head Cross-Attention",
                desc: "Projects heterogenous embeddings into a shared 128-dimensional latent space. Learns dynamic per-modality importance gates and cross-attends across tokens. Gracefully supports missing modalities via attention key padding masks.",
                tech: "PyTorch MultiheadAttention • Modality Masking • Dynamic Gating",
            },
        };

        nodes.forEach((node) => {
            node.addEventListener("click", () => {
                nodes.forEach((n) => n.classList.remove("active"));
                node.classList.add("active");
                const key = node.dataset.node;
                const info = descriptions[key];
                if (detailBox && info) {
                    detailBox.innerHTML = `
                        <div class="glass-card" style="padding: 1.5rem; border-color: var(--border-accent); animation: fade-in-up 0.3s ease;">
                            <span class="badge badge-teal" style="margin-bottom:0.75rem;">${info.tech}</span>
                            <h3 style="font-size:1.25rem; font-weight:700; margin-bottom:0.5rem; color:var(--text-highlight);">${info.title}</h3>
                            <p style="font-size:0.9rem; color:var(--text-secondary); line-height:1.6;">${info.desc}</p>
                        </div>
                    `;
                }
            });
        });
    }

    async function loadLiveExplainabilityPreview() {
        const shapContainer = document.getElementById("liveShapContainer");
        if (!shapContainer) return;

        try {
            const data = await MediFusionAPI.explainClinical(
                { fever: 1, cough: 1, rapid_breathing: 1, fatigue: 1, chills: 1 },
                { age: 55, temperature: 39.2, heart_rate: 108, bp_systolic: 130, bp_diastolic: 85, respiratory_rate: 26, oxygen_saturation: 91 }
            );

            if (data && data.plot_base64) {
                shapContainer.innerHTML = `
                    <img src="data:image/png;base64,${data.plot_base64}" alt="SHAP Feature Importance" style="width:100%; border-radius:var(--radius-md); border:1px solid var(--border-normal);">
                `;
            }
        } catch (e) {
            console.error("Could not load SHAP preview:", e);
        }
    }

    // ── Advanced Image Model Metrics ─────────────────────────────

    const MODEL_DISPLAY_INFO = {
        chest_xray: {
            icon: "🫁", title: "Chest X-Ray", subtitle: "Pneumonia Detection",
            color: "#2dd4bf", gradient: "linear-gradient(135deg, rgba(45,212,191,0.12), rgba(45,212,191,0.03))"
        },
        brain_tumor: {
            icon: "🧠", title: "Brain Tumor MRI", subtitle: "Neoplasm Classification",
            color: "#f59e0b", gradient: "linear-gradient(135deg, rgba(245,158,11,0.12), rgba(245,158,11,0.03))"
        },
        skin_cancer: {
            icon: "🔬", title: "Skin Cancer", subtitle: "HAM10000 Lesion Analysis",
            color: "#ef4444", gradient: "linear-gradient(135deg, rgba(239,68,68,0.12), rgba(239,68,68,0.03))"
        },
        retinopathy: {
            icon: "👁️", title: "Diabetic Retinopathy", subtitle: "Retinal Fundus Grading",
            color: "#3b82f6", gradient: "linear-gradient(135deg, rgba(59,130,246,0.12), rgba(59,130,246,0.03))"
        },
        blood_cell: {
            icon: "🩸", title: "Blood Cell", subtitle: "Hematological Microscopy",
            color: "#8b5cf6", gradient: "linear-gradient(135deg, rgba(139,92,246,0.12), rgba(139,92,246,0.03))"
        },
    };

    async function loadAdvancedImageModelMetrics() {
        const grid = document.getElementById("advancedImageModelsGrid");
        if (!grid) return;

        try {
            // Fetch model status and metadata from the API
            const [statusRes, metadataRes] = await Promise.all([
                fetch("/api/image-models/status").then(r => r.ok ? r.json() : null).catch(() => null),
                fetch("/api/image-models/metadata").then(r => r.ok ? r.json() : null).catch(() => null),
            ]);

            // Also try to fetch stored metrics from the reports directory
            const metricsMap = {};
            const modelKeys = ["chest_xray", "brain_tumor", "skin_cancer", "retinopathy", "blood_cell"];

            for (const key of modelKeys) {
                try {
                    const res = await fetch(`/static/reports/${key}_image_model_metrics.json`);
                    if (res.ok) {
                        metricsMap[key] = await res.json();
                    }
                } catch (e) { /* skip */ }
            }

            // Also fetch all_model_metrics.json for chest xray (it has Image-Only key)
            try {
                const allRes = await fetch("/static/reports/all_model_metrics.json");
                if (allRes.ok) {
                    const allMetrics = await allRes.json();
                    if (allMetrics["Image-Only"]) {
                        metricsMap["chest_xray"] = {
                            test_accuracy: allMetrics["Image-Only"].accuracy,
                            ...allMetrics["Image-Only"]
                        };
                    }
                }
            } catch (e) { /* skip */ }

            // Build cards
            let html = "";
            let loadedCount = 0;

            for (const key of modelKeys) {
                const info = MODEL_DISPLAY_INFO[key] || {};
                const metrics = metricsMap[key];
                const meta = metadataRes ? metadataRes[key] : null;
                const statusModel = statusRes?.models?.find(m => m.model_key === key);
                const isLoaded = statusModel?.loaded ?? false;

                if (isLoaded) loadedCount++;

                const accuracy = metrics?.test_accuracy ?? metrics?.accuracy ?? null;
                const f1 = metrics?.f1_macro ?? null;
                const rocAuc = metrics?.roc_auc ?? null;
                const numClasses = meta?.classes ? Object.keys(meta.classes).length : (statusModel?.num_classes ?? "—");
                const totalImages = meta?.total_images ?? metrics?.train_size ?? "—";

                html += `
                <div class="glass-card" style="padding:1.35rem; background:${info.gradient}; border-color:${isLoaded ? info.color + '40' : 'var(--border-subtle)'}; transition:all 0.3s ease; position:relative; overflow:hidden;">
                    <!-- Status dot -->
                    <div style="position:absolute; top:10px; right:10px; width:8px; height:8px; border-radius:50%; background:${isLoaded ? '#22c55e' : '#64748b'};"></div>

                    <!-- Header -->
                    <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:1rem;">
                        <div style="width:42px; height:42px; border-radius:10px; background:rgba(255,255,255,0.06); display:flex; align-items:center; justify-content:center; font-size:1.4rem;">${info.icon}</div>
                        <div>
                            <div style="font-size:0.95rem; font-weight:700; color:var(--text-highlight);">${info.title}</div>
                            <div style="font-size:0.72rem; color:var(--text-muted);">${info.subtitle} • ${numClasses} Classes</div>
                        </div>
                    </div>

                    ${accuracy !== null ? `
                    <!-- Metrics Grid -->
                    <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:0.65rem; margin-bottom:0.85rem;">
                        <div style="text-align:center;">
                            <div style="font-size:1.35rem; font-weight:800; color:${info.color}; font-family:'JetBrains Mono',monospace;">${(accuracy * 100).toFixed(1)}%</div>
                            <div style="font-size:0.65rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.06em;">Accuracy</div>
                        </div>
                        <div style="text-align:center;">
                            <div style="font-size:1.35rem; font-weight:800; color:var(--text-secondary); font-family:'JetBrains Mono',monospace;">${f1 !== null ? (f1 * 100).toFixed(1) + '%' : '—'}</div>
                            <div style="font-size:0.65rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.06em;">F1-Score</div>
                        </div>
                        <div style="text-align:center;">
                            <div style="font-size:1.35rem; font-weight:800; color:var(--text-secondary); font-family:'JetBrains Mono',monospace;">${rocAuc !== null ? rocAuc.toFixed(3) : '—'}</div>
                            <div style="font-size:0.65rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.06em;">ROC-AUC</div>
                        </div>
                    </div>

                    <!-- Accuracy Bar -->
                    <div style="height:5px; background:rgba(100,116,139,0.15); border-radius:3px; overflow:hidden;">
                        <div style="height:100%; width:${accuracy * 100}%; background:${info.color}; border-radius:3px; transition:width 1.2s ease;"></div>
                    </div>
                    <div style="display:flex; justify-content:space-between; margin-top:0.4rem;">
                        <span style="font-size:0.65rem; color:var(--text-muted);">${typeof totalImages === 'number' ? totalImages.toLocaleString() + ' images' : totalImages}</span>
                        <span style="font-size:0.65rem; color:${isLoaded ? '#22c55e' : 'var(--text-muted)'}; font-weight:600;">${isLoaded ? '● Model Loaded' : '○ Not Loaded'}</span>
                    </div>
                    ` : `
                    <!-- Not trained yet -->
                    <div style="text-align:center; padding:1rem 0;">
                        <div style="font-size:0.82rem; color:var(--text-muted); line-height:1.6;">
                            ${isLoaded ? '<span style="color:#22c55e; font-weight:600;">● Model Loaded</span> — Metrics pending evaluation' : 'Model not yet trained. Run training to generate metrics.'}
                        </div>
                    </div>
                    `}
                </div>
                `;
            }

            grid.innerHTML = html;

        } catch (e) {
            console.error("Failed to load advanced image model metrics:", e);
            grid.innerHTML = `
                <div class="glass-card" style="padding:1.5rem; text-align:center; grid-column: 1 / -1;">
                    <p style="font-size:0.85rem; color:var(--text-muted);">Unable to load model metrics. Start the API server to view performance data.</p>
                </div>
            `;
        }
    }
})();
