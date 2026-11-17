/**
 * MediFusion AI — Dedicated Modality Page Controller v3.0
 * Powers dedicated single-model image classification workspaces
 * with unified 5-step workflow, instant inference, Grad-CAM overlays,
 * sample loading, and clinical guidance.
 */

(function () {
    "use strict";

    window.MediFusionModalityApp = {
        init: function (config) {
            const modelKey    = config.modelKey;
            const endpointRaw = config.endpoint;
            const sampleFile  = config.sampleFile;
            const modelTitle  = config.title;

            // Resolve standard endpoint mapping
            const endpoint = (function () {
                if (endpointRaw.includes("/advanced/brain_tumor") || endpointRaw.includes("/brain_tumor")) return "/api/predict/image/brain-tumor";
                if (endpointRaw.includes("/advanced/skin_cancer") || endpointRaw.includes("/skin_cancer")) return "/api/predict/image/skin-cancer";
                if (endpointRaw.includes("/advanced/retinopathy") || endpointRaw.includes("/retinopathy")) return "/api/predict/image/retinopathy";
                if (endpointRaw.includes("/advanced/blood_cell") || endpointRaw.includes("/blood_cell")) return "/api/predict/image/blood-cell";
                return endpointRaw;
            })();

            let currentFile     = null;
            let currentImageSrc = null;
            let gradcamOverlay  = null;
            let gradcamHeatmap  = null;
            let activeViewMode  = "original";
            let zoomLevel       = 1;

            // ── DOM refs ──
            const dropzone         = document.getElementById("dropzone");
            const fileInput        = document.getElementById("fileInput");
            const previewImg       = document.getElementById("previewImg");
            const previewContainer = document.getElementById("previewContainer");
            const dropzonePrompt   = document.getElementById("dropzonePrompt");
            const analyzeBtn       = document.getElementById("analyzeBtn");
            const loadSampleBtn    = document.getElementById("loadSampleBtn");
            const clearBtn         = document.getElementById("clearBtn");
            const scannerBeam      = document.getElementById("scannerBeam");
            const fileInfoPill     = document.getElementById("fileInfoPill");
            const resultsSection   = document.getElementById("resultsSection");
            const emptySection     = document.getElementById("emptySection");

            const predCondition    = document.getElementById("predCondition");
            const predConfidence   = document.getElementById("predConfidence");
            const severityBadge    = document.getElementById("severityBadge");
            const classProbabilitiesList = document.getElementById("classProbabilitiesList");

            const clinicalContextText   = document.getElementById("clinicalContextText");
            const recommendedActionText = document.getElementById("recommendedActionText");
            const tabContext   = document.getElementById("tabBtnContext");
            const tabAction    = document.getElementById("tabBtnAction");
            const paneContext  = document.getElementById("paneContext");
            const paneAction   = document.getElementById("paneAction");

            const viewerOrigImg    = document.getElementById("viewerOrigImg");
            const viewerHeatmapImg = document.getElementById("viewerHeatmapImg");
            const viewerOverlayImg = document.getElementById("viewerOverlayImg");
            const viewerStage      = document.getElementById("viewerStage");
            const thumbOrig        = document.getElementById("thumbOrig");
            const thumbHeat        = document.getElementById("thumbHeat");
            const thumbOverlay     = document.getElementById("thumbOverlay");
            const gradcamStatusDot  = document.getElementById("gradcamStatusDot");
            const gradcamStatusText = document.getElementById("gradcamStatusText");
            const btnViewOriginal  = document.getElementById("btnViewOriginal");
            const btnViewOverlay   = document.getElementById("btnViewOverlay");
            const btnViewHeatmap   = document.getElementById("btnViewHeatmap");
            const viewerTabOrig    = document.getElementById("viewerTabOrig");
            const viewerTabHeat    = document.getElementById("viewerTabHeat");
            const viewerTabOver    = document.getElementById("viewerTabOver");
            const gradcamStatusDot2  = document.getElementById("gradcamStatusDot2");
            const gradcamStatusText2 = document.getElementById("gradcamStatusText2");
            const diffCallout        = document.getElementById("differentialCallout");
            const diffText           = document.getElementById("differentialText");

            // ── Workflow Stepper Controller ──
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

            // Initialize stepper to Step 1
            setStepperStep(1, "active");

            // ── Clinical tabs ──
            function initClinicalTabs() {
                if (!tabContext || !tabAction) return;
                tabContext.addEventListener("click", () => {
                    tabContext.classList.add("active"); tabAction.classList.remove("active");
                    if (paneContext) paneContext.classList.add("active");
                    if (paneAction)  paneAction.classList.remove("active");
                });
                tabAction.addEventListener("click", () => {
                    tabAction.classList.add("active"); tabContext.classList.remove("active");
                    if (paneAction)  paneAction.classList.add("active");
                    if (paneContext) paneContext.classList.remove("active");
                });
            }
            initClinicalTabs();

            // ── Dropzone setup ──
            if (dropzone && fileInput) {
                dropzone.addEventListener("click", () => fileInput.click());
                dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("drag-active"); });
                dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag-active"));
                dropzone.addEventListener("drop", (e) => {
                    e.preventDefault(); dropzone.classList.remove("drag-active");
                    if (e.dataTransfer.files && e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
                });
                fileInput.addEventListener("change", (e) => {
                    if (e.target.files && e.target.files[0]) handleFile(e.target.files[0]);
                });
            }

            if (loadSampleBtn) {
                loadSampleBtn.addEventListener("click", async (e) => {
                    e.stopPropagation();
                    try {
                        if (window.MediFusionUI) MediFusionUI.showToast("info", `Loading sample ${modelTitle} scan...`);
                        const res = await fetch(`/static/images/samples/${sampleFile}`);
                        if (!res.ok) throw new Error("Sample file not found");
                        const blob = await res.blob();
                        const file = new File([blob], sampleFile, { type: blob.type || "image/jpeg" });
                        await handleFile(file);
                        if (window.MediFusionUI) MediFusionUI.showToast("success", "Sample image loaded! Ready to analyze.");
                    } catch (err) {
                        if (window.MediFusionUI) MediFusionUI.showToast("error", "Could not load sample image.");
                    }
                });
            }

            if (clearBtn) clearBtn.addEventListener("click", (e) => { e.stopPropagation(); resetUI(); });
            if (analyzeBtn) analyzeBtn.addEventListener("click", runAnalysis);

            if (btnViewOriginal) btnViewOriginal.addEventListener("click", () => setImageViewMode("original"));
            if (btnViewOverlay)  btnViewOverlay.addEventListener("click",  () => setImageViewMode("overlay"));
            if (btnViewHeatmap)  btnViewHeatmap.addEventListener("click",  () => setImageViewMode("heatmap"));
            if (viewerTabOrig)   viewerTabOrig.addEventListener("click",   () => setImageViewMode("original"));
            if (viewerTabOver)   viewerTabOver.addEventListener("click",   () => setImageViewMode("overlay"));
            if (viewerTabHeat)   viewerTabHeat.addEventListener("click",   () => setImageViewMode("heatmap"));
            if (thumbOrig)    thumbOrig.addEventListener("click",    () => setImageViewMode("original"));
            if (thumbHeat)    thumbHeat.addEventListener("click",    () => setImageViewMode("heatmap"));
            if (thumbOverlay) thumbOverlay.addEventListener("click", () => setImageViewMode("overlay"));

            // Zoom controls
            const zoomInBtn    = document.getElementById("zoomInBtn");
            const zoomOutBtn   = document.getElementById("zoomOutBtn");
            const zoomResetBtn = document.getElementById("zoomResetBtn");
            const zoomLabel    = document.getElementById("zoomLevelLabel");

            function applyZoom() {
                const imgs = viewerStage ? viewerStage.querySelectorAll("img") : [];
                imgs.forEach(img => { img.style.transform = `scale(${zoomLevel})`; });
                if (zoomLabel) zoomLabel.textContent = `${Math.round(zoomLevel * 100)}%`;
            }

            if (zoomInBtn)    zoomInBtn.addEventListener("click",    () => { zoomLevel = Math.min(zoomLevel + 0.25, 3); applyZoom(); });
            if (zoomOutBtn)   zoomOutBtn.addEventListener("click",   () => { zoomLevel = Math.max(zoomLevel - 0.25, 0.5); applyZoom(); });
            if (zoomResetBtn) zoomResetBtn.addEventListener("click", () => { zoomLevel = 1; applyZoom(); });

            // ── Handle file ──
            async function handleFile(file) {
                if (!file.type.startsWith("image/")) {
                    if (window.MediFusionUI) MediFusionUI.showToast("error", "Please upload a valid image file (PNG, JPG, JPEG).");
                    return;
                }
                currentFile = file;
                const reader = new FileReader();
                reader.onload = (e) => {
                    currentImageSrc = e.target.result;
                    if (previewImg)       previewImg.src = currentImageSrc;
                    if (previewContainer) previewContainer.style.display = "block";
                    if (dropzonePrompt)   dropzonePrompt.style.display = "none";
                    if (fileInfoPill)     fileInfoPill.textContent = `${file.name} • ${(file.size / 1024).toFixed(1)} KB`;
                    if (analyzeBtn) {
                        analyzeBtn.disabled = false;
                        analyzeBtn.innerHTML = `<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="5 3 19 12 5 21 5 3"/></svg><span>Run Neural Analysis</span>`;
                    }
                    if (clearBtn) clearBtn.style.display = "inline-flex";
                    if (thumbOrig) {
                        const tImg = thumbOrig.querySelector("img");
                        if (tImg) tImg.src = currentImageSrc;
                    }

                    // Stepper: Upload Completed, Preview Active
                    setStepperStep(2, "active");
                };
                reader.readAsDataURL(file);
            }

            // ── View mode ──
            function setImageViewMode(mode) {
                activeViewMode = mode;
                [btnViewOriginal, btnViewOverlay, btnViewHeatmap, viewerTabOrig, viewerTabHeat, viewerTabOver].forEach(btn => { if (btn) btn.classList.remove("active"); });
                [thumbOrig, thumbHeat, thumbOverlay].forEach(t => { if (t) t.classList.remove("active"); });

                if (mode === "original") {
                    if (btnViewOriginal) btnViewOriginal.classList.add("active");
                    if (viewerTabOrig)   viewerTabOrig.classList.add("active");
                    if (thumbOrig)       thumbOrig.classList.add("active");
                    if (viewerOrigImg)    { viewerOrigImg.classList.remove("hidden"); viewerOrigImg.style.display = "block"; viewerOrigImg.src = currentImageSrc; }
                    if (viewerHeatmapImg) viewerHeatmapImg.classList.add("hidden");
                    if (viewerOverlayImg) viewerOverlayImg.classList.add("hidden");
                } else if (mode === "overlay" && gradcamOverlay) {
                    if (btnViewOverlay) btnViewOverlay.classList.add("active");
                    if (viewerTabOver)   viewerTabOver.classList.add("active");
                    if (thumbOverlay)   thumbOverlay.classList.add("active");
                    if (viewerOrigImg)    viewerOrigImg.classList.add("hidden");
                    if (viewerHeatmapImg) viewerHeatmapImg.classList.add("hidden");
                    if (viewerOverlayImg) { viewerOverlayImg.classList.remove("hidden"); viewerOverlayImg.style.display = "block"; viewerOverlayImg.src = `data:image/png;base64,${gradcamOverlay}`; }
                } else if (mode === "heatmap" && gradcamHeatmap) {
                    if (btnViewHeatmap) btnViewHeatmap.classList.add("active");
                    if (viewerTabHeat)   viewerTabHeat.classList.add("active");
                    if (thumbHeat)      thumbHeat.classList.add("active");
                    if (viewerOrigImg)    viewerOrigImg.classList.add("hidden");
                    if (viewerHeatmapImg) { viewerHeatmapImg.classList.remove("hidden"); viewerHeatmapImg.style.display = "block"; viewerHeatmapImg.src = `data:image/png;base64,${gradcamHeatmap}`; }
                    if (viewerOverlayImg) viewerOverlayImg.classList.add("hidden");
                }
            }

            // ── Inference ──
            async function runAnalysis() {
                if (!currentFile) return;
                if (analyzeBtn) { analyzeBtn.disabled = true; analyzeBtn.innerHTML = `<div class="btn-spinner"></div> Analyzing...`; }
                if (scannerBeam) scannerBeam.style.display = "block";

                // Stepper: Preview Completed, AI Processing Active
                setStepperStep(3, "active");

                try {
                    const formData = new FormData();
                    formData.append("file", currentFile);

                    const predPromise = fetch(endpoint, { method: "POST", body: formData }).then(async r => {
                        if (!r.ok) { const err = await r.json().catch(() => ({})); throw new Error(err.detail || `Server error (${r.status})`); }
                        return r.json();
                    });

                    const gradcamFormData = new FormData();
                    gradcamFormData.append("file", currentFile);
                    gradcamFormData.append("model_key", modelKey);

                    const gradcamPromise = fetch("/api/explain/image/advanced", { method: "POST", body: gradcamFormData })
                        .then(r => r.ok ? r.json() : null).catch(() => null);

                    const [predData, gradcamData] = await Promise.all([predPromise, gradcamPromise]);

                    renderResults(predData);
                    renderGradCAM(gradcamData);

                    // Stepper: AI Processing Completed, Prediction & Insights Completed
                    setStepperStep(5, "completed");

                    if (window.MediFusionUI) MediFusionUI.showToast("success", "Clinical inference completed successfully!");
                } catch (err) {
                    if (window.MediFusionUI) MediFusionUI.showToast("error", `Analysis Failed: ${err.message}`);
                    setStepperStep(2, "active");
                } finally {
                    if (scannerBeam) scannerBeam.style.display = "none";
                    if (analyzeBtn) {
                        analyzeBtn.disabled = false;
                        analyzeBtn.innerHTML = `<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="5 3 19 12 5 21 5 3"/></svg><span>Re-Analyze</span>`;
                    }
                }
            }

            // ── Render Results ──
            function renderResults(data) {
                if (emptySection) emptySection.style.display = "none";
                if (resultsSection) {
                    resultsSection.style.display = "flex";
                    resultsSection.style.flexDirection = "column";
                    setTimeout(() => resultsSection.scrollIntoView({ behavior: "smooth", block: "nearest" }), 50);
                }

                // Secondary differential callout
                if (diffCallout && diffText && data.predictions && data.predictions.length > 1) {
                    const p2 = data.predictions[1];
                    diffText.textContent = `${p2.display_name || p2.class_id} (${((p2.probability || 0) * 100).toFixed(1)}%)`;
                    diffCallout.style.display = "inline-flex";
                } else if (diffCallout) {
                    diffCallout.style.display = "none";
                }

                if (predCondition)  predCondition.textContent = data.predicted_display_name || data.predicted_class || "—";
                if (predConfidence) predConfidence.textContent = `${((data.confidence || 0) * 100).toFixed(1)}%`;

                if (severityBadge) {
                    const sev = (data.severity || "info").toLowerCase();
                    const labels = { critical: "Critical", high: "High Severity", moderate: "Moderate", low: "Low Risk", none: "Normal", info: "Informational" };
                    severityBadge.className = `ia-severity-pill ia-sev-${sev}`;
                    severityBadge.innerHTML = `<span class="ia-sev-dot"></span>${labels[sev] || sev}`;
                }

                if (clinicalContextText)   clinicalContextText.textContent   = data.clinical_context   || "No acute pathology flagged.";
                if (recommendedActionText) recommendedActionText.textContent = data.recommended_action || "Consult with a specialist physician.";

                if (classProbabilitiesList && data.predictions) {
                    classProbabilitiesList.innerHTML = "";
                    data.predictions.forEach((p, idx) => {
                        const pct = (p.probability * 100).toFixed(1);
                        const row = document.createElement("div");
                        row.className = `ia-prob-row${idx === 0 ? " ia-top" : ""}`;
                        row.innerHTML = `
                            <div class="ia-rank-badge">${idx + 1}</div>
                            <div class="ia-prob-body">
                                <div class="ia-prob-label">${p.display_name || p.class_id}</div>
                                <div class="ia-prob-track"><div class="ia-prob-fill" style="background:${p.color || '#0284c7'}; width:0%;"></div></div>
                            </div>
                            <div class="ia-prob-pct">${pct}%</div>
                        `;
                        classProbabilitiesList.appendChild(row);
                        const fill = row.querySelector(".ia-prob-fill");
                        if (fill) setTimeout(() => { fill.style.width = `${pct}%`; }, 80 + idx * 70);
                    });
                }
            }

            // ── Render Grad-CAM ──
            function renderGradCAM(gradcamData) {
                if (viewerOrigImg) { viewerOrigImg.src = currentImageSrc || ""; viewerOrigImg.classList.remove("hidden"); }
                if (viewerHeatmapImg) viewerHeatmapImg.classList.add("hidden");
                if (viewerOverlayImg) viewerOverlayImg.classList.add("hidden");

                if (gradcamData && gradcamData.gradcam_overlay_base64) {
                    gradcamOverlay = gradcamData.gradcam_overlay_base64;
                    gradcamHeatmap = gradcamData.gradcam_heatmap_base64;

                    if (btnViewOverlay) btnViewOverlay.disabled = false;
                    if (btnViewHeatmap) btnViewHeatmap.disabled = false;
                    if (viewerTabOver)  viewerTabOver.disabled = false;
                    if (viewerTabHeat)  viewerTabHeat.disabled = false;

                    if (gradcamStatusDot2)  gradcamStatusDot2.className   = "ia-status-dot on";
                    if (gradcamStatusText2) gradcamStatusText2.textContent = "Attention map generated";

                    if (thumbHeat    && gradcamHeatmap) { const img = thumbHeat.querySelector("img");    if (img) img.src = `data:image/png;base64,${gradcamHeatmap}`; }
                    if (thumbOverlay && gradcamOverlay) { const img = thumbOverlay.querySelector("img"); if (img) img.src = `data:image/png;base64,${gradcamOverlay}`; }

                    if (gradcamStatusDot)  gradcamStatusDot.className   = "ia-status-dot on";
                    if (gradcamStatusText) gradcamStatusText.textContent = "Attention map generated";

                    setTimeout(() => setImageViewMode("overlay"), 200);
                } else {
                    if (gradcamStatusDot)  gradcamStatusDot.className   = "ia-status-dot";
                    if (gradcamStatusText) gradcamStatusText.textContent = "Grad-CAM unavailable";
                    setImageViewMode("original");
                }
            }

            // ── Reset ──
            function resetUI() {
                currentFile = null; currentImageSrc = null;
                gradcamOverlay = null; gradcamHeatmap = null; zoomLevel = 1;
                if (fileInput)        fileInput.value = "";
                if (previewImg)       previewImg.src = "";
                if (previewContainer) previewContainer.style.display = "none";
                if (dropzonePrompt)   dropzonePrompt.style.display = "block";
                if (fileInfoPill)     fileInfoPill.textContent = "No image selected";
                if (analyzeBtn) { analyzeBtn.disabled = true; analyzeBtn.innerHTML = `<span>Select Image First</span>`; }
                if (clearBtn)       clearBtn.style.display = "none";
                if (resultsSection) resultsSection.style.display = "none";
                if (emptySection)   emptySection.style.display = "block";
                if (btnViewOverlay) btnViewOverlay.disabled = true;
                if (btnViewHeatmap) btnViewHeatmap.disabled = true;
                setStepperStep(1, "active");
            }
        }
    };
})();
