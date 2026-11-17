/**
 * MediFusion AI — Advanced Multi-Model Image Analysis Module
 * Handles model selection, drag/drop, multi-endpoint inference,
 * clinical context display, and Grad-CAM inspection.
 */

(function () {
    "use strict";

    // ── State ──
    let currentFile = null;
    let originalDataUrl = null;
    let gradcamOverlayBase64 = null;
    let gradcamHeatmapBase64 = null;
    let zoomLevel = 1;
    let selectedModelKey = null;
    let modelsMetadata = {};
    let modelsStatus = {};

    // ── Model Definitions (fallback if API unavailable) ──
    const MODEL_DEFS = [
        {
            key: "chest_xray",
            title: "Chest X-Ray",
            icon: "🫁",
            classes: "2-class",
            imageType: "grayscale",
            description: "Pneumonia Detection",
            path: "/chest-xray",
        },
        {
            key: "brain_tumor",
            title: "Brain Tumor MRI",
            icon: "🧠",
            classes: "4-class",
            imageType: "grayscale",
            description: "Intracranial Neoplasm",
            path: "/brain-tumor",
        },
        {
            key: "skin_cancer",
            title: "Skin Cancer",
            icon: "🔬",
            classes: "7-class",
            imageType: "color",
            description: "Dermatological Lesion",
            path: "/skin-cancer",
        },
        {
            key: "retinopathy",
            title: "Retinopathy",
            icon: "👁",
            classes: "5-class",
            imageType: "color",
            description: "DR Severity Grading",
            path: "/retinopathy",
        },
        {
            key: "blood_cell",
            title: "Blood Cell",
            icon: "🩸",
            classes: "4-class",
            imageType: "color",
            description: "Hematological Microscopy",
            path: "/blood-cell",
        },
    ];

    function setStepperStep(stepNumber, state) {
        for (let i = 1; i <= 5; i++) {
            const stepEl = document.getElementById(`stepNode${i}`);
            const dividerEl = document.getElementById(`stepDiv${i}`);
            if (!stepEl) continue;
            if (i < stepNumber) {
                stepEl.className = "stepper-step completed";
                if (dividerEl) dividerEl.className = "stepper-divider completed";
            } else if (i === stepNumber) {
                stepEl.className = `stepper-step ${state}`;
                if (dividerEl) dividerEl.className = "stepper-divider";
            } else {
                stepEl.className = "stepper-step";
                if (dividerEl) dividerEl.className = "stepper-divider";
            }
        }
    }

    document.addEventListener("DOMContentLoaded", () => {
        setStepperStep(1, "active");
        initModelSelector();
        setupUpload();
        setupControls();
        loadModelsStatus();
        initClinicalTabs();
        initThumbnailClicks();
    });

    function initClinicalTabs() {
        const tabCtx   = document.getElementById("iaTabContext");
        const tabAct   = document.getElementById("iaTabAction");
        const paneCtx  = document.getElementById("iaPaneContext");
        const paneAct  = document.getElementById("iaPaneAction");
        if (!tabCtx || !tabAct) return;
        tabCtx.addEventListener("click", () => {
            tabCtx.classList.add("active"); tabAct.classList.remove("active");
            paneCtx.classList.add("active"); paneAct.classList.remove("active");
        });
        tabAct.addEventListener("click", () => {
            tabAct.classList.add("active"); tabCtx.classList.remove("active");
            paneAct.classList.add("active"); paneCtx.classList.remove("active");
        });
    }

    function initThumbnailClicks() {
        const th = ["thumbOrig","thumbHeat","thumbOverlay"];
        const modes = ["original","heatmap","overlay"];
        th.forEach((id, i) => {
            const el = document.getElementById(id);
            if (el) el.addEventListener("click", () => setViewerMode(modes[i]));
        });
    }

    // ── Model Selector ──

    async function loadModelsStatus() {
        try {
            const statusData = await MediFusionAPI.getImageModelsStatus();
            if (statusData && statusData.models) {
                statusData.models.forEach(m => {
                    modelsStatus[m.model_key] = m.loaded;
                });
                updateModelCards();
            }
        } catch (e) {
            console.warn("Could not load image model status:", e);
        }

        try {
            const metaData = await MediFusionAPI.getImageModelsMetadata();
            if (metaData) {
                modelsMetadata = metaData;
            }
        } catch (e) {
            console.warn("Could not load image model metadata:", e);
        }
    }

    function initModelSelector() {
        const grid = document.getElementById("modelSelectorGrid");
        if (!grid) return;

        grid.innerHTML = "";

        MODEL_DEFS.forEach(def => {
            const card = document.createElement("div");
            card.className = "model-card";
            card.dataset.modelKey = def.key;

            card.innerHTML = `
                <div class="model-card-status not-loaded" id="status-${def.key}"></div>
                <div class="model-card-icon">${def.icon}</div>
                <div class="model-card-title">${def.title}</div>
                <div class="model-card-classes">${def.classes} • ${def.description}</div>
            `;

            card.addEventListener("click", () => selectModel(def.key));
            grid.appendChild(card);
        });
    }

    function updateModelCards() {
        MODEL_DEFS.forEach(def => {
            const statusDot = document.getElementById(`status-${def.key}`);
            if (statusDot) {
                const loaded = modelsStatus[def.key] || false;
                statusDot.className = `model-card-status ${loaded ? "loaded" : "not-loaded"}`;
            }
        });
    }

    function selectModel(modelKey) {
        selectedModelKey = modelKey;

        // Update card selection
        document.querySelectorAll(".model-card").forEach(card => {
            card.classList.toggle("selected", card.dataset.modelKey === modelKey);
        });

        // Update dropzone text
        const def = MODEL_DEFS.find(d => d.key === modelKey);
        const dropzoneMain = document.getElementById("dropzoneMainText");
        const dropzoneSub = document.getElementById("dropzoneSubText");
        const dropzoneNotice = document.getElementById("dropzoneNotice");
        const analyzeBtn = document.getElementById("runImageAnalysisBtn");
        const btnText = document.getElementById("imageBtnText");
        const backboneTag = document.getElementById("modelBackboneTag");

        if (dropzoneMain) dropzoneMain.textContent = `Drag & drop ${def ? def.title : "medical"} image`;
        if (dropzoneSub) dropzoneSub.textContent = `${def?.imageType === "grayscale" ? "Grayscale" : "Color"} JPEG, PNG • Max 15MB`;
        if (dropzoneNotice) {
            dropzoneNotice.innerHTML = `
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
                <span>${def ? def.description : "Analysis"} pipeline selected</span>
            `;
        }

        if (btnText) {
            if (currentFile) {
                btnText.textContent = `Analyze with ${def?.title || "Model"}`;
                if (analyzeBtn) analyzeBtn.disabled = false;
            } else {
                btnText.textContent = `Upload Image for ${def?.title || "Analysis"}`;
            }
        }

        if (backboneTag) backboneTag.textContent = "ResNet18";

        // Update model info box
        const infoBox = document.getElementById("modelInfoBox");
        if (infoBox && def) {
            const meta = modelsMetadata[modelKey] || {};
            infoBox.innerHTML = `
                <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                    <div>
                        <strong style="color:var(--text-secondary);">${def.title}:</strong>
                        ${meta.subtitle || def.description}. ${def.classes} classification using ${meta.backbone || "ResNet-18"} transfer learning.
                    </div>
                </div>
                <div style="margin-top:0.65rem;">
                    <a href="${def.path || '#'}" class="btn btn-secondary" style="display:inline-flex; align-items:center; gap:0.4rem; font-size:0.75rem; padding:0.3rem 0.75rem; border-color:rgba(45,212,191,0.5); color:var(--accent-teal);">
                        <span>Launch Dedicated ${def.title} Studio</span>
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6M15 3h6v6M10 14L21 3"/></svg>
                    </a>
                </div>
            `;
        }
    }

    // ── Upload & Image Handling ──

    function setupUpload() {
        const dropzone = document.getElementById("imageDropzone");
        const fileInput = document.getElementById("imageFileInput");
        const removeBtn = document.getElementById("removeImageBtn");
        const analyzeBtn = document.getElementById("runImageAnalysisBtn");

        if (!dropzone || !fileInput) return;

        dropzone.addEventListener("click", () => fileInput.click());

        dropzone.addEventListener("dragover", (e) => {
            e.preventDefault();
            dropzone.classList.add("drag-active");
        });

        dropzone.addEventListener("dragleave", () => {
            dropzone.classList.remove("drag-active");
        });

        dropzone.addEventListener("drop", (e) => {
            e.preventDefault();
            dropzone.classList.remove("drag-active");
            if (e.dataTransfer.files.length > 0) {
                loadFile(e.dataTransfer.files[0]);
            }
        });

        fileInput.addEventListener("change", (e) => {
            if (e.target.files.length > 0) {
                loadFile(e.target.files[0]);
            }
        });

        if (removeBtn) {
            removeBtn.addEventListener("click", (e) => {
                e.stopPropagation();
                clearImage();
            });
        }

        const loadSampleBtn = document.getElementById("loadSampleBtn");
        if (loadSampleBtn) {
            loadSampleBtn.addEventListener("click", (e) => {
                e.stopPropagation();
                loadSampleImage();
            });
        }

        if (analyzeBtn) {
            analyzeBtn.addEventListener("click", runImageInference);
        }
    }

    async function loadSampleImage() {
        if (!selectedModelKey) {
            selectModel("brain_tumor");
        }
        const sampleFiles = {
            chest_xray: "chest_xray.png",
            brain_tumor: "brain_tumor.jpg",
            skin_cancer: "skin_cancer.jpg",
            retinopathy: "retinopathy.jpg",
            blood_cell: "blood_cell.jpeg"
        };
        const filename = sampleFiles[selectedModelKey] || "brain_tumor.jpg";
        const url = `/static/images/samples/${filename}`;

        try {
            MediFusionUI.showToast("info", `Loading sample ${selectedModelKey.replace('_', ' ')} scan...`);
            const res = await fetch(url);
            if (!res.ok) throw new Error("Sample file not found");
            const blob = await res.blob();
            const file = new File([blob], filename, { type: blob.type || "image/jpeg" });
            await loadFile(file);
            MediFusionUI.showToast("success", `Sample image loaded for ${selectedModelKey.replace('_', ' ')}! Click Analyze to run inference.`);
        } catch (err) {
            console.error("Failed to load sample image:", err);
            MediFusionUI.showToast("error", "Failed to load sample image.");
        }
    }

    async function loadFile(file) {
        if (!file.type.startsWith("image/")) {
            MediFusionUI.showToast("error", "Please select a valid image file (PNG, JPG, JPEG).");
            return;
        }

        // Only run strict radiograph validation for chest_xray mode
        if (selectedModelKey === "chest_xray" && window.ImageValidator) {
            const dropzone = document.getElementById("imageDropzone");
            MediFusionUI.showToast("info", "Validating radiograph structure...");

            try {
                const result = await ImageValidator.validate(file);
                if (!result.valid) {
                    const title = ImageValidator.getErrorTitle(result.code);
                    MediFusionUI.showToast("error", result.error, 6000);
                    if (dropzone) {
                        MediFusionUI.showValidationError(dropzone, title, result.error, 6000);
                    }
                    const fileInput = document.getElementById("imageFileInput");
                    if (fileInput) fileInput.value = "";
                    return;
                }
            } catch (err) {
                console.warn("Image validation error, allowing upload:", err);
            }
        }

        loadFileDirectly(file);
    }

    function loadFileDirectly(file) {
        const dropzone = document.getElementById("imageDropzone");
        if (dropzone) {
            const existingOverlay = dropzone.querySelector(".validation-error-overlay");
            if (existingOverlay) existingOverlay.remove();
        }

        currentFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            originalDataUrl = e.target.result;
            const previewImg = document.getElementById("previewImg");
            const dropzoneInitial = document.getElementById("dropzoneInitial");
            const previewContainer = document.getElementById("previewContainer");
            const analyzeBtn = document.getElementById("runImageAnalysisBtn");
            const fileMeta = document.getElementById("fileMetadataPill");
            const btnText = document.getElementById("imageBtnText");

            if (previewImg) previewImg.src = originalDataUrl;
            if (dropzoneInitial) dropzoneInitial.style.display = "none";
            if (previewContainer) previewContainer.style.display = "flex";

            if (selectedModelKey) {
                if (analyzeBtn) analyzeBtn.disabled = false;
                const def = MODEL_DEFS.find(d => d.key === selectedModelKey);
                if (btnText) btnText.textContent = `Analyze with ${def?.title || "Model"}`;
            } else {
                if (btnText) btnText.textContent = "Select a Model First";
            }

            if (fileMeta) {
                const sizeKb = (file.size / 1024).toFixed(1);
                fileMeta.textContent = `${file.name} • ${sizeKb} KB`;
            }
            setStepperStep(2, "active");
        };
        reader.readAsDataURL(file);
    }

    function clearImage() {
        currentFile = null;
        originalDataUrl = null;
        gradcamOverlayBase64 = null;
        gradcamHeatmapBase64 = null;

        const dropzoneInitial = document.getElementById("dropzoneInitial");
        const previewContainer = document.getElementById("previewContainer");
        const fileInput = document.getElementById("imageFileInput");
        const analyzeBtn = document.getElementById("runImageAnalysisBtn");
        const btnText = document.getElementById("imageBtnText");

        if (dropzoneInitial) dropzoneInitial.style.display = "block";
        if (previewContainer) previewContainer.style.display = "none";
        if (fileInput) fileInput.value = "";
        if (analyzeBtn) analyzeBtn.disabled = true;
        if (btnText) btnText.textContent = selectedModelKey ? "Upload an Image" : "Select a Model to Begin";

        const emptyStage = document.getElementById("emptyStage");
        const resultsStage = document.getElementById("imageResultsStage");
        if (emptyStage) emptyStage.style.display = "flex";
        if (resultsStage) resultsStage.style.display = "none";
        setStepperStep(1, "active");
    }

    // ── Inference ──

    async function runImageInference() {
        if (!currentFile) {
            MediFusionUI.showToast("error", "Please upload an image first.");
            return;
        }
        if (!selectedModelKey) {
            MediFusionUI.showToast("error", "Please select an imaging model first.");
            return;
        }

        const analyzeBtn = document.getElementById("runImageAnalysisBtn");
        const btnText = document.getElementById("imageBtnText");
        const scannerBeam = document.getElementById("imageScannerBeam");
        const emptyStage = document.getElementById("emptyStage");
        const resultsStage = document.getElementById("imageResultsStage");

        if (analyzeBtn) analyzeBtn.disabled = true;
        if (btnText) btnText.textContent = "Analyzing...";
        if (scannerBeam) scannerBeam.style.display = "block";

        setStepperStep(3, "active");

        try {
            let predData, explainData;

            if (selectedModelKey === "chest_xray") {
                // Use original endpoints for backward compatibility
                [predData, explainData] = await Promise.all([
                    MediFusionAPI.predictImage(currentFile),
                    MediFusionAPI.explainImage(currentFile),
                ]);
                // Normalize chest_xray response to match advanced format
                predData = normalizeChestXrayResponse(predData);
            } else {
                // Advanced model endpoints
                [predData, explainData] = await Promise.all([
                    MediFusionAPI.predictImageAdvanced(selectedModelKey, currentFile),
                    MediFusionAPI.explainImageAdvanced(selectedModelKey, currentFile).catch(err => {
                        console.warn("Grad-CAM not available for this model:", err);
                        return null;
                    }),
                ]);
            }

            if (emptyStage) emptyStage.style.display = "none";
            if (resultsStage) {
                resultsStage.style.display = "flex";
                resultsStage.style.flexDirection = "column";
                setTimeout(() => resultsStage.scrollIntoView({ behavior: "smooth", block: "nearest" }), 50);
            }

            renderResults(predData);
            renderGradCAM(explainData);

            setStepperStep(5, "completed");
            MediFusionUI.showToast("success", "Analysis complete with clinical context.");
        } catch (err) {
            console.error("Image analysis error:", err);
            const msg = err.message || "Unable to complete image analysis.";
            MediFusionUI.showToast("error", msg, 6000);
            setStepperStep(2, "active");
        } finally {
            if (analyzeBtn) analyzeBtn.disabled = false;
            const def = MODEL_DEFS.find(d => d.key === selectedModelKey);
            if (btnText) btnText.textContent = `Analyze with ${def?.title || "Model"}`;
            if (scannerBeam) scannerBeam.style.display = "none";
        }
    }

    function normalizeChestXrayResponse(data) {
        // Convert original chest_xray response to AdvancedImagePredictionResponse format
        const meta = modelsMetadata["chest_xray"] || {};
        const classesMeta = meta.classes || {};

        const predictions = (data.predictions || []).map(p => {
            const cmeta = classesMeta[p.disease] || {};
            return {
                class_id: p.disease,
                display_name: cmeta.display_name || p.disease,
                probability: p.probability,
                severity: cmeta.severity || (p.disease === "PNEUMONIA" ? "high" : "none"),
                color: cmeta.color || "#3b82f6",
                description: cmeta.description || "",
                action: cmeta.action || "",
            };
        });

        predictions.sort((a, b) => b.probability - a.probability);

        const primary = predictions[0] || {};
        return {
            model_key: "chest_xray",
            model_title: meta.title || "Chest X-Ray Analysis",
            backbone: "ResNet-18",
            predicted_class: data.predicted_class,
            predicted_display_name: primary.display_name || data.predicted_class,
            confidence: data.confidence,
            severity: primary.severity || "none",
            predictions: predictions,
            clinical_context: primary.description || "",
            recommended_action: primary.action || "",
        };
    }

    // ── Result Rendering ──

    function renderResults(data) {
        // Model label
        const modelLabel = document.getElementById("resultModelLabel");
        if (modelLabel) modelLabel.textContent = data.model_title || data.model_key;

        // Predicted condition
        const conditionEl = document.getElementById("imagePredCondition");
        if (conditionEl) conditionEl.textContent = data.predicted_display_name || data.predicted_class;

        // Confidence (premium orb)
        const confEl = document.getElementById("imagePredConfidence");
        if (confEl) confEl.textContent = `${(data.confidence * 100).toFixed(1)}%`;

        // Severity pill (ia-* system)
        const severityBadge = document.getElementById("imageSeverityBadge");
        if (severityBadge) {
            const sev = (data.severity || "none").toLowerCase();
            const labels = { critical: "Critical", high: "High Severity", moderate: "Moderate", low: "Low Risk", none: "Normal", info: "Informational" };
            severityBadge.className = `ia-severity-pill ia-sev-${sev}`;
            severityBadge.innerHTML = `<span class="ia-sev-dot"></span>${labels[sev] || sev}`;
        }

        // Predictions list with animated bars (ia-* classes)
        const listContainer = document.getElementById("predictionsListContainer");
        if (listContainer && data.predictions) {
            listContainer.innerHTML = "";
            data.predictions.forEach((p, idx) => {
                const pct = (p.probability * 100).toFixed(1);
                const row = document.createElement("div");
                row.className = `ia-prob-row${idx === 0 ? " ia-top" : ""}`;
                row.innerHTML = `
                    <div class="ia-rank-badge">${idx + 1}</div>
                    <div class="ia-prob-body">
                        <div class="ia-prob-label">${p.display_name || p.class_id}</div>
                        <div class="ia-prob-track"><div class="ia-prob-fill" style="background:${p.color || '#2dd4bf'}; width:0%;"></div></div>
                    </div>
                    <div class="ia-prob-pct">${pct}%</div>
                `;
                listContainer.appendChild(row);
                const fill = row.querySelector(".ia-prob-fill");
                if (fill) setTimeout(() => { fill.style.width = `${pct}%`; }, 80 + idx * 70);
            });
        }

        // Class count tag
        const countTag = document.getElementById("classCountTag");
        if (countTag && data.predictions) {
            countTag.textContent = `${data.predictions.length}-Class Probabilities`;
        }

        // Secondary differential diagnosis callout
        const diffCallout = document.getElementById("imageDifferentialCallout");
        const diffText = document.getElementById("differentialText");
        if (diffCallout && diffText && data.predictions && data.predictions.length > 1) {
            const p2 = data.predictions[1];
            diffText.textContent = `${p2.display_name || p2.class_id} (${(p2.probability * 100).toFixed(1)}%)`;
            diffCallout.style.display = "inline-flex";
        } else if (diffCallout) {
            diffCallout.style.display = "none";
        }

        // Clinical context (tabs)
        const contextText = document.getElementById("clinicalContextText");
        const actionText  = document.getElementById("actionText");
        if (contextText) contextText.textContent = data.clinical_context || "";
        if (actionText)  actionText.textContent  = data.recommended_action || "";
    }

    function renderGradCAM(explainData) {
        const origLayer    = document.getElementById("viewerOrigImg");
        const heatLayer    = document.getElementById("viewerHeatmapImg");
        const overlayLayer = document.getElementById("viewerOverlayImg");
        const thumbOrig    = document.getElementById("thumbOrig");
        const thumbHeat    = document.getElementById("thumbHeat");
        const thumbOverlay = document.getElementById("thumbOverlay");
        const statusDot    = document.getElementById("gradcamStatusDot");
        const statusText   = document.getElementById("gradcamStatusText");

        // Always set original
        if (origLayer) { origLayer.src = originalDataUrl || ""; origLayer.style.display = "block"; origLayer.classList.remove("hidden"); }
        if (thumbOrig)  { const i = thumbOrig.querySelector("img"); if (i) i.src = originalDataUrl || ""; }

        if (explainData && explainData.gradcam_overlay_base64) {
            gradcamOverlayBase64 = explainData.gradcam_overlay_base64;
            gradcamHeatmapBase64 = explainData.gradcam_heatmap_base64;

            if (heatLayer)    heatLayer.src    = `data:image/png;base64,${gradcamHeatmapBase64}`;
            if (overlayLayer) overlayLayer.src = `data:image/png;base64,${gradcamOverlayBase64}`;

            if (thumbHeat)    { const i = thumbHeat.querySelector("img");    if (i) i.src = `data:image/png;base64,${gradcamHeatmapBase64}`; }
            if (thumbOverlay) { const i = thumbOverlay.querySelector("img"); if (i) i.src = `data:image/png;base64,${gradcamOverlayBase64}`; }

            if (statusDot)  statusDot.className   = "ia-status-dot on";
            if (statusText) statusText.textContent = "Attention map generated";
        } else {
            if (heatLayer)    heatLayer.src    = originalDataUrl || "";
            if (overlayLayer) overlayLayer.src = originalDataUrl || "";
            if (statusDot)  statusDot.className   = "ia-status-dot";
            if (statusText) statusText.textContent = "Grad-CAM unavailable";
        }

        setViewerMode("overlay");
    }

    // ── Viewer Controls ──

    function setupControls() {
        // Tab buttons in viewer header
        const tabBtns = document.querySelectorAll(".ia-tab-btn");
        tabBtns.forEach((btn) => {
            btn.addEventListener("click", () => {
                tabBtns.forEach((b) => b.classList.remove("active"));
                btn.classList.add("active");
                setViewerMode(btn.dataset.mode);
            });
        });

        const zoomIn = document.getElementById("zoomInBtn");
        const zoomOut = document.getElementById("zoomOutBtn");
        const zoomReset = document.getElementById("zoomResetBtn");
        const zoomLabel = document.getElementById("zoomLevelLabel");

        function applyZoom() {
            document.querySelectorAll(".ia-stage-img").forEach((img) => {
                img.style.transform = `scale(${zoomLevel})`;
            });
            if (zoomLabel) zoomLabel.textContent = `${Math.round(zoomLevel * 100)}%`;
        }

        if (zoomIn) {
            zoomIn.addEventListener("click", () => {
                zoomLevel = Math.min(zoomLevel + 0.25, 3.0);
                applyZoom();
            });
        }
        if (zoomOut) {
            zoomOut.addEventListener("click", () => {
                zoomLevel = Math.max(zoomLevel - 0.25, 0.6);
                applyZoom();
            });
        }
        if (zoomReset) {
            zoomReset.addEventListener("click", () => {
                zoomLevel = 1.0;
                applyZoom();
            });
        }
    }

    function setViewerMode(mode) {
        const origLayer    = document.getElementById("viewerOrigImg");
        const heatLayer    = document.getElementById("viewerHeatmapImg");
        const overlayLayer = document.getElementById("viewerOverlayImg");
        const thumbOrig    = document.getElementById("thumbOrig");
        const thumbHeat    = document.getElementById("thumbHeat");
        const thumbOverlay = document.getElementById("thumbOverlay");

        // Update tab buttons
        document.querySelectorAll(".ia-tab-btn").forEach(b => b.classList.remove("active"));
        [thumbOrig, thumbHeat, thumbOverlay].forEach(t => { if (t) t.classList.remove("active"); });

        [origLayer, heatLayer, overlayLayer].forEach(img => { if (img) img.classList.add("hidden"); });

        if (mode === "original") {
            document.querySelectorAll('.ia-tab-btn[data-mode="original"]').forEach(b => b.classList.add("active"));
            if (thumbOrig) thumbOrig.classList.add("active");
            if (origLayer) { origLayer.classList.remove("hidden"); origLayer.style.display = "block"; }
        } else if (mode === "heatmap") {
            document.querySelectorAll('.ia-tab-btn[data-mode="heatmap"]').forEach(b => b.classList.add("active"));
            if (thumbHeat) thumbHeat.classList.add("active");
            if (heatLayer) { heatLayer.classList.remove("hidden"); heatLayer.style.display = "block"; }
        } else if (mode === "overlay") {
            document.querySelectorAll('.ia-tab-btn[data-mode="overlay"]').forEach(b => b.classList.add("active"));
            if (thumbOverlay) thumbOverlay.classList.add("active");
            if (overlayLayer) { overlayLayer.classList.remove("hidden"); overlayLayer.style.display = "block"; }
        }
    }

    function applyZoom(images) {
        images.forEach((img) => {
            img.style.transform = `scale(${zoomLevel})`;
        });
    }
})();
