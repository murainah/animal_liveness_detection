// Global variables
let selectedFile = null;
let outputFilename = null;
let statsInterval = null;


document.addEventListener('DOMContentLoaded', function() {
    setupEventListeners();
});

function setupEventListeners() {
    // File input
    const fileInput = document.getElementById('video-input');
    fileInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop
    const uploadArea = document.getElementById('upload-area');
    
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        this.classList.add('drag-over');
    });
    
    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        this.classList.remove('drag-over');
    });
    
    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        this.classList.remove('drag-over');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            handleFileSelect({ target: { files: files } });
        }
    });
    
    uploadArea.addEventListener('click', function() {
        fileInput.click();
    });
}

function switchTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Remove active from all buttons
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(tabName + '-tab').classList.add('active');
    
    // Activate button
    event.target.classList.add('active');
}

function handleFileSelect(event) {
    const file = event.target.files[0];
    
    if (!file) return;
    
    // Check file type
    const validTypes = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska'];
    if (!validTypes.includes(file.type)) {
        alert('Invalid file type. Please upload MP4, AVI, MOV, or MKV files.');
        return;
    }
    
    // Check file size (500MB max)
    const maxSize = 500 * 1024 * 1024;
    if (file.size > maxSize) {
        alert('File too large. Maximum size is 500MB.');
        return;
    }
    
    selectedFile = file;
    
    // Display file info
    document.getElementById('filename').textContent = file.name;
    document.getElementById('filesize').textContent = formatFileSize(file.size);
    document.getElementById('upload-area').style.display = 'none';
    document.getElementById('file-info').style.display = 'block';
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

function resetUpload() {
    selectedFile = null;
    outputFilename = null;
    
    document.getElementById('video-input').value = '';
    document.getElementById('upload-area').style.display = 'block';
    document.getElementById('file-info').style.display = 'none';
    document.getElementById('progress-section').style.display = 'none';
    document.getElementById('results-section').style.display = 'none';
}

async function uploadVideo() {
    if (!selectedFile) {
        alert('Please select a video file first.');
        return;
    }
    
    // Get configuration
    const livestockType = document.getElementById('livestock-type').value;
    // Fixed confidence at 45% (optimal)
    
    // Prepare form data
    const formData = new FormData();
    formData.append('video', selectedFile);
    formData.append('livestock_type', livestockType);
    
    // Show progress section
    document.getElementById('file-info').style.display = 'none';
    document.getElementById('progress-section').style.display = 'block';
    
    // Simulate progress (real progress would need server-sent events or websockets)
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress += 2;
        if (progress >= 90) {
            clearInterval(progressInterval);
        }
        updateProgress(progress, 'Processing video...');
    }, 500);
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        clearInterval(progressInterval);
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Upload failed');
        }
        
        const result = await response.json();
        
        // Complete progress
        updateProgress(100, 'Processing complete!');
        
        // Show results after a short delay
        setTimeout(() => {
            displayResults(result);
        }, 1000);
        
    } catch (error) {
        clearInterval(progressInterval);
        alert('Error processing video: ' + error.message);
        resetUpload();
    }
}

function updateProgress(percentage, message) {
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');
    
    progressFill.style.width = percentage + '%';
    progressText.textContent = message + ' (' + percentage + '%)';
}

function displayResults(result) {
    const summary = result.summary;
    outputFilename = result.output_video;
    
    // Update alert summary
    const alertSummary = document.getElementById('alert-summary');
    const alertIcon = document.getElementById('alert-icon');
    const alertTitle = document.getElementById('alert-title');
    const alertMessage = document.getElementById('alert-message');
    
    if (summary.stillness_alert) {
        alertSummary.className = 'alert-summary alert';
        alertIcon.textContent = '⚠️';
        alertTitle.textContent = 'Alert: Animal(s) Still Too Long';
        alertMessage.textContent = summary.alert_message;
    } else {
        alertSummary.className = 'alert-summary success';
        alertIcon.textContent = '✅';
        alertTitle.textContent = 'No Issues Detected';
        alertMessage.textContent = summary.alert_message;
    }
    
    // Update statistics
    document.getElementById('total-frames').textContent = summary.total_frames.toLocaleString();
    document.getElementById('max-animals').textContent = summary.max_animals_in_frame || 0;
    document.getElementById('motion-frames').textContent = summary.frames_with_motion.toLocaleString();
    document.getElementById('motion-percentage').textContent = summary.motion_percentage.toFixed(1) + '%';
    document.getElementById('processing-time').textContent = summary.processing_time.toFixed(1) + 's';
    document.getElementById('processing-fps').textContent = summary.fps_processed.toFixed(1);
    
    // Hide progress, show results
    document.getElementById('progress-section').style.display = 'none';
    document.getElementById('results-section').style.display = 'block';
}

function downloadVideo() {
    if (!outputFilename) {
        alert('No processed video available.');
        return;
    }
    
    window.location.href = '/download/' + outputFilename;
}

// Camera functions
async function startCamera() {
    const livestockType = document.getElementById('livestock-type').value;
    const cameraIndex = document.getElementById('camera-index').value;
    // Fixed confidence at 45% (optimal)
    
    try {
        const response = await fetch('/camera/start', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                livestock_type: livestockType,
                camera_index: parseInt(cameraIndex)
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Failed to start camera');
        }
        
        // Update UI
        document.getElementById('start-camera-btn').disabled = true;
        document.getElementById('stop-camera-btn').disabled = false;
        document.getElementById('camera-feed-section').style.display = 'block';
        
        // Set camera feed source
        document.getElementById('camera-feed').src = '/camera/feed?t=' + Date.now();
        
        // Start updating stats
        statsInterval = setInterval(updateCameraStats, 500);
        
    } catch (error) {
        alert('Error starting camera: ' + error.message);
    }
}

async function stopCamera() {
    try {
        const response = await fetch('/camera/stop', {
            method: 'POST'
        });
        
        if (!response.ok) {
            throw new Error('Failed to stop camera');
        }
        
        // Update UI
        document.getElementById('start-camera-btn').disabled = false;
        document.getElementById('stop-camera-btn').disabled = true;
        document.getElementById('camera-feed-section').style.display = 'none';
        
        // Clear camera feed
        document.getElementById('camera-feed').src = '';
        
        // Stop updating stats
        if (statsInterval) {
            clearInterval(statsInterval);
            statsInterval = null;
        }
        
    } catch (error) {
        alert('Error stopping camera: ' + error.message);
    }
}

async function updateCameraStats() {
    try {
        const response = await fetch('/camera/stats');
        
        if (!response.ok) return;
        
        const stats = await response.json();
        
        if (!stats.active) {
            stopCamera();
            return;
        }
        
        // Update stats display
        document.getElementById('live-total').textContent = stats.animal_count || 0;
        document.getElementById('live-moving').textContent = stats.animals_with_motion || 0;
        document.getElementById('live-still').textContent = 
            (stats.animal_count || 0) - (stats.animals_with_motion || 0);
        
        // Update detection details
        const detailsDiv = document.getElementById('detection-details');
        if (stats.bounding_boxes && stats.bounding_boxes.length > 0) {
            let html = '<h4>Individual Detections:</h4>';
            stats.bounding_boxes.forEach((box, index) => {
                const motionClass = box.has_motion ? 'moving' : '';
                const motionText = box.has_motion ? '✓ Moving' : '✗ Still';
                html += `
                    <div class="detection-item ${motionClass}">
                        <strong>Animal #${box.id}</strong> - 
                        Confidence: ${(box.confidence * 100).toFixed(1)}% - 
                        ${motionText}
                    </div>
                `;
            });
            detailsDiv.innerHTML = html;
        } else {
            detailsDiv.innerHTML = '<p>No animals detected</p>';
        }
        
    } catch (error) {
        console.error('Error updating stats:', error);
    }
}

// Cleanup on page unload
window.addEventListener('beforeunload', function() {
    if (statsInterval) {
        stopCamera();
    }
});
