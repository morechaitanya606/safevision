/**
 * SafeVision — Frontend Application Logic
 * Handles tab switching, file uploads, AJAX detection calls,
 * webcam controls, and results rendering.
 */

document.addEventListener('DOMContentLoaded', () => {
    // =====================
    //  Tab Navigation
    // =====================
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.tab;

            tabButtons.forEach(b => b.classList.remove('active'));
            tabContents.forEach(t => t.classList.remove('active'));

            btn.classList.add('active');
            document.getElementById(`${tab}Tab`).classList.add('active');
        });
    });

    // =====================
    //  Image Detection
    // =====================
    const imageDropZone = document.getElementById('imageDropZone');
    const imageInput = document.getElementById('imageInput');
    const imagePreview = document.getElementById('imagePreview');
    const previewImg = document.getElementById('previewImg');
    const removeImageBtn = document.getElementById('removeImage');
    const detectImageBtn = document.getElementById('detectImageBtn');
    const imageResultZone = document.getElementById('imageResultZone');
    const imageResult = document.getElementById('imageResult');
    const resultImg = document.getElementById('resultImg');
    const imageResultsPanel = document.getElementById('imageResultsPanel');

    let selectedImageFile = null;

    // Click to upload
    imageDropZone.addEventListener('click', (e) => {
        if (e.target === removeImageBtn || e.target.closest('.remove-btn')) return;
        imageInput.click();
    });

    imageInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleImageFile(e.target.files[0]);
        }
    });

    // Drag and drop
    imageDropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        imageDropZone.classList.add('drag-over');
    });

    imageDropZone.addEventListener('dragleave', () => {
        imageDropZone.classList.remove('drag-over');
    });

    imageDropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        imageDropZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            handleImageFile(e.dataTransfer.files[0]);
        }
    });

    function handleImageFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file.');
            return;
        }
        selectedImageFile = file;
        const url = URL.createObjectURL(file);
        previewImg.src = url;
        imagePreview.style.display = 'flex';
        detectImageBtn.disabled = false;
    }

    removeImageBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        selectedImageFile = null;
        previewImg.src = '';
        imagePreview.style.display = 'none';
        detectImageBtn.disabled = true;
        imageInput.value = '';
    });

    // Detect image
    detectImageBtn.addEventListener('click', async () => {
        if (!selectedImageFile) return;

        showLoading();
        detectImageBtn.disabled = true;

        const formData = new FormData();
        formData.append('file', selectedImageFile);

        try {
            const response = await fetch('/detect/image', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                // Show result image
                resultImg.src = data.result_image + '?t=' + Date.now();
                imageResult.style.display = 'flex';
                imageResultZone.querySelector('.placeholder-content').style.display = 'none';

                // Show results panel
                renderImageResults(data.detection);
                imageResultsPanel.style.display = 'block';
            } else {
                alert('Detection failed: ' + (data.error || 'Unknown error'));
            }
        } catch (error) {
            alert('Error connecting to server: ' + error.message);
        } finally {
            hideLoading();
            detectImageBtn.disabled = false;
        }
    });

    function renderImageResults(detection) {
        const statusClass = detection.overall_safe ? 'safe' :
            detection.safety_status === 'PARTIAL' ? 'warning' : 'unsafe';
        const statusIcon = detection.overall_safe ? '✅' :
            detection.safety_status === 'PARTIAL' ? '⚠️' : '🚫';
        const statusText = detection.overall_safe ? 'ALL SAFE' :
            detection.safety_status === 'PARTIAL' ? 'PARTIAL COMPLIANCE' :
            detection.safety_status === 'NO_PERSON' ? 'NO PERSON DETECTED' : 'UNSAFE';

        let html = `
            <div class="results-header">
                <h3>Detection Results</h3>
                <span class="safety-badge ${statusClass}">${statusIcon} ${statusText}</span>
            </div>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">${detection.persons_detected}</div>
                    <div class="stat-label">Persons</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${detection.goggles_detected}</div>
                    <div class="stat-label">With Goggles</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${detection.persons_detected - detection.goggles_detected}</div>
                    <div class="stat-label">Without Goggles</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${detection.other_ppe.length}</div>
                    <div class="stat-label">Other PPE</div>
                </div>
            </div>
        `;

        if (detection.persons && detection.persons.length > 0) {
            html += '<div class="person-list">';
            detection.persons.forEach(person => {
                const isSafe = person.wearing_goggles;
                const icon = isSafe ? '🛡️' : '⚠️';
                const cls = isSafe ? 'safe' : 'unsafe';
                const conf = person.goggles_confidence ?
                    ` (${Math.round(person.goggles_confidence * 100)}% conf.)` : '';
                const ppeList = person.ppe_items.length > 0 ?
                    ` • PPE: ${person.ppe_items.join(', ')}` : '';

                html += `
                    <div class="person-item">
                        <div class="person-info">
                            <div class="person-icon ${cls}">${icon}</div>
                            <div>
                                <div class="person-name">Person ${person.id}</div>
                                <div class="person-status">
                                    ${isSafe ? 'Wearing Goggles' + conf : 'No Goggles Detected!'}${ppeList}
                                </div>
                            </div>
                        </div>
                        <span class="safety-badge ${cls}">
                            ${isSafe ? '✅ SAFE' : '🚫 UNSAFE'}
                        </span>
                    </div>
                `;
            });
            html += '</div>';
        }

        if (detection.other_ppe.length > 0) {
            html += `
                <div style="margin-top: 16px; padding: 12px 16px; background: rgba(255,180,0,0.08); border-radius: 8px; border: 1px solid rgba(255,180,0,0.2);">
                    <strong style="color: #FFB400;">Other PPE Detected:</strong>
                    <span style="color: #94a3b8;">${detection.other_ppe.map(p => p.charAt(0).toUpperCase() + p.slice(1)).join(', ')}</span>
                </div>
            `;
        }

        imageResultsPanel.innerHTML = html;
    }

    // =====================
    //  Video Detection
    // =====================
    const videoDropZone = document.getElementById('videoDropZone');
    const videoInput = document.getElementById('videoInput');
    const videoPreview = document.getElementById('videoPreview');
    const previewVideo = document.getElementById('previewVideo');
    const removeVideoBtn = document.getElementById('removeVideo');
    const detectVideoBtn = document.getElementById('detectVideoBtn');
    const videoResultZone = document.getElementById('videoResultZone');
    const videoResult = document.getElementById('videoResult');
    const resultVideo = document.getElementById('resultVideo');
    const videoProgress = document.getElementById('videoProgress');
    const progressFill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');
    const videoResultsPanel = document.getElementById('videoResultsPanel');

    let selectedVideoFile = null;

    videoDropZone.addEventListener('click', (e) => {
        if (e.target === removeVideoBtn || e.target.closest('.remove-btn')) return;
        videoInput.click();
    });

    videoInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleVideoFile(e.target.files[0]);
        }
    });

    videoDropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        videoDropZone.classList.add('drag-over');
    });

    videoDropZone.addEventListener('dragleave', () => {
        videoDropZone.classList.remove('drag-over');
    });

    videoDropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        videoDropZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            handleVideoFile(e.dataTransfer.files[0]);
        }
    });

    function handleVideoFile(file) {
        if (!file.type.startsWith('video/')) {
            alert('Please select a video file.');
            return;
        }
        selectedVideoFile = file;
        const url = URL.createObjectURL(file);
        previewVideo.src = url;
        videoPreview.style.display = 'flex';
        detectVideoBtn.disabled = false;
    }

    removeVideoBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        selectedVideoFile = null;
        previewVideo.src = '';
        videoPreview.style.display = 'none';
        detectVideoBtn.disabled = true;
        videoInput.value = '';
    });

    detectVideoBtn.addEventListener('click', async () => {
        if (!selectedVideoFile) return;

        detectVideoBtn.disabled = true;
        videoProgress.style.display = 'block';
        progressFill.style.width = '0%';
        progressText.textContent = 'Uploading and processing video...';

        // Simulate progress
        let progress = 0;
        const progressTimer = setInterval(() => {
            progress = Math.min(progress + Math.random() * 5, 90);
            progressFill.style.width = `${progress}%`;
            progressText.textContent = `Processing... ${Math.round(progress)}%`;
        }, 500);

        const formData = new FormData();
        formData.append('file', selectedVideoFile);

        try {
            const response = await fetch('/detect/video', {
                method: 'POST',
                body: formData
            });

            clearInterval(progressTimer);
            progressFill.style.width = '100%';
            progressText.textContent = 'Complete!';

            const data = await response.json();

            if (data.success) {
                resultVideo.src = data.result_video + '?t=' + Date.now();
                videoResult.style.display = 'flex';
                videoResultZone.querySelector('.placeholder-content').style.display = 'none';

                renderVideoResults(data.detection);
                videoResultsPanel.style.display = 'block';
            } else {
                alert('Processing failed: ' + (data.error || 'Unknown error'));
            }
        } catch (error) {
            clearInterval(progressTimer);
            alert('Error connecting to server: ' + error.message);
        } finally {
            detectVideoBtn.disabled = false;
            setTimeout(() => {
                videoProgress.style.display = 'none';
            }, 2000);
        }
    });

    function renderVideoResults(detection) {
        const safetyScore = detection.safety_score || 0;
        const statusClass = safetyScore >= 80 ? 'safe' :
            safetyScore >= 40 ? 'warning' : 'unsafe';
        const statusIcon = safetyScore >= 80 ? '✅' :
            safetyScore >= 40 ? '⚠️' : '🚫';

        let html = `
            <div class="results-header">
                <h3>Video Analysis Results</h3>
                <span class="safety-badge ${statusClass}">${statusIcon} Safety Score: ${safetyScore}%</span>
            </div>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">${detection.total_frames}</div>
                    <div class="stat-label">Total Frames</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${detection.frames_with_persons}</div>
                    <div class="stat-label">Frames w/ Persons</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${detection.frames_with_goggles}</div>
                    <div class="stat-label">Frames w/ Goggles</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${detection.max_persons_in_frame}</div>
                    <div class="stat-label">Max Persons</div>
                </div>
            </div>
            <div style="padding: 16px; background: rgba(255,255,255,0.03); border-radius: 8px; border: 1px solid rgba(255,255,255,0.06);">
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <span style="color: #94a3b8; font-size: 0.85rem;">Safety Compliance</span>
                    <span style="font-weight: 700; color: ${safetyScore >= 80 ? 'var(--safe-color)' : safetyScore >= 40 ? 'var(--warning-color)' : 'var(--unsafe-color)'};">${safetyScore}%</span>
                </div>
                <div style="width: 100%; height: 10px; background: rgba(255,255,255,0.1); border-radius: 100px; overflow: hidden;">
                    <div style="width: ${safetyScore}%; height: 100%; background: ${safetyScore >= 80 ? 'var(--gradient-primary)' : safetyScore >= 40 ? 'linear-gradient(90deg, #ffa502, #ff6348)' : 'linear-gradient(90deg, #ff4757, #c0392b)'}; border-radius: 100px; transition: width 1s ease;"></div>
                </div>
            </div>
        `;

        videoResultsPanel.innerHTML = html;
    }

    // =====================
    //  Webcam Detection
    // =====================
    const startWebcamBtn = document.getElementById('startWebcamBtn');
    const stopWebcamBtn = document.getElementById('stopWebcamBtn');
    const webcamStream = document.getElementById('webcamStream');
    const webcamPlaceholder = document.getElementById('webcamPlaceholder');
    const webcamOverlay = document.getElementById('webcamOverlay');

    startWebcamBtn.addEventListener('click', async () => {
        try {
            const response = await fetch('/detect/webcam/start', { method: 'POST' });
            const data = await response.json();

            if (data.success) {
                webcamStream.src = '/video_feed?t=' + Date.now();
                webcamStream.style.display = 'block';
                webcamPlaceholder.style.display = 'none';
                webcamOverlay.style.display = 'block';

                startWebcamBtn.style.display = 'none';
                stopWebcamBtn.style.display = 'flex';
            }
        } catch (error) {
            alert('Failed to start webcam: ' + error.message);
        }
    });

    stopWebcamBtn.addEventListener('click', async () => {
        try {
            await fetch('/detect/webcam/stop', { method: 'POST' });
        } catch (e) {
            // Ignore
        }

        webcamStream.src = '';
        webcamStream.style.display = 'none';
        webcamPlaceholder.style.display = 'flex';
        webcamOverlay.style.display = 'none';

        startWebcamBtn.style.display = 'flex';
        stopWebcamBtn.style.display = 'none';
    });

    // =====================
    //  Loading Overlay
    // =====================
    function showLoading() {
        document.getElementById('loadingOverlay').style.display = 'flex';
    }

    function hideLoading() {
        document.getElementById('loadingOverlay').style.display = 'none';
    }

    // =====================
    //  Health Check
    // =====================
    async function checkHealth() {
        try {
            const response = await fetch('/health');
            const data = await response.json();
            const statusEl = document.getElementById('modelStatus');

            if (data.status === 'healthy') {
                statusEl.innerHTML = `
                    <span class="status-dot"></span>
                    <span>Model Ready ${data.world_model ? '(YOLO-World)' : '(YOLOv8)'}</span>
                `;
            }
        } catch (e) {
            // Server not ready yet
        }
    }

    checkHealth();
});
