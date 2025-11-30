// Brain Tumor Segmentation - Frontend JavaScript

// Global variables
let currentSessionId = null;
let uploadedFiles = {
    t1: null,
    t1ce: null,
    t2: null,
    flair: null
};

// DOM Elements
const elements = {
    // File inputs
    t1Input: document.getElementById('t1-upload'),
    t1ceInput: document.getElementById('t1ce-upload'),
    t2Input: document.getElementById('t2-upload'),
    flairInput: document.getElementById('flair-upload'),

    // Buttons
    uploadBtn: document.getElementById('uploadBtn'),
    clearBtn: document.getElementById('clearBtn'),
    newScanBtn: document.getElementById('newScanBtn'),
    retryBtn: document.getElementById('retryBtn'),

    // Sections
    uploadSection: document.getElementById('uploadSection'),
    processingSection: document.getElementById('processingSection'),
    resultsSection: document.getElementById('resultsSection'),
    errorSection: document.getElementById('errorSection'),

    // Progress
    uploadProgress: document.getElementById('uploadProgress'),
    uploadProgressFill: document.getElementById('uploadProgressFill'),
    uploadProgressText: document.getElementById('uploadProgressText'),

    // Processing
    processingText: document.getElementById('processingText'),
    step1: document.getElementById('step1'),
    step2: document.getElementById('step2'),
    step3: document.getElementById('step3'),
    step4: document.getElementById('step4'),

    // Status
    statusIndicator: document.getElementById('statusIndicator'),

    // Results
    statsGrid: document.getElementById('statsGrid'),
    visualizationContainer: document.getElementById('visualizationContainer'),

    // Error
    errorMessage: document.getElementById('errorMessage')
};

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    console.log('Brain Tumor Segmentation App initialized');

    // Check system status
    checkStatus();

    // Setup file inputs
    setupFileInputs();

    // Setup event listeners
    setupEventListeners();
});

// Check system status
async function checkStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();

        if (data.model_loaded) {
            updateStatusIndicator(true, `Model Ready (${data.device})`);
        } else {
            updateStatusIndicator(false, 'Model Not Loaded');
            showToast('Warning: Model not loaded. Please check server logs.', 'error');
        }
    } catch (error) {
        console.error('Failed to check status:', error);
        updateStatusIndicator(false, 'Connection Error');
        showToast('Failed to connect to server', 'error');
    }
}

// Update status indicator
function updateStatusIndicator(isReady, text) {
    const statusDot = elements.statusIndicator.querySelector('.status-dot');
    const statusText = elements.statusIndicator.querySelector('.status-text');

    if (isReady) {
        statusDot.style.background = '#10b981';
        elements.statusIndicator.style.background = 'rgba(16, 185, 129, 0.1)';
        elements.statusIndicator.style.borderColor = 'rgba(16, 185, 129, 0.2)';
        statusText.style.color = '#10b981';
    } else {
        statusDot.style.background = '#ef4444';
        elements.statusIndicator.style.background = 'rgba(239, 68, 68, 0.1)';
        elements.statusIndicator.style.borderColor = 'rgba(239, 68, 68, 0.2)';
        statusText.style.color = '#ef4444';
    }

    statusText.textContent = text;
}

// Setup file inputs
function setupFileInputs() {
    const modalities = ['t1', 't1ce', 't2', 'flair'];

    modalities.forEach(mod => {
        const input = elements[`${mod}Input`];
        const label = document.querySelector(`label[for="${mod}-upload"]`);
        const filename = document.getElementById(`${mod}-filename`);

        input.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                uploadedFiles[mod] = file;
                filename.textContent = truncateFilename(file.name, 20);
                label.classList.add('has-file');

                // Show checkmark
                const statusDiv = document.getElementById(`${mod}-status`);
                statusDiv.textContent = '✓ Ready';
                statusDiv.style.color = '#10b981';

                checkAllFilesUploaded();
            }
        });
    });
}

// Setup event listeners
function setupEventListeners() {
    // Upload button
    elements.uploadBtn.addEventListener('click', handleUpload);

    // Clear button
    elements.clearBtn.addEventListener('click', handleClear);

    // New scan button
    elements.newScanBtn.addEventListener('click', handleNewScan);

    // Retry button
    elements.retryBtn.addEventListener('click', handleRetry);

    // Download buttons (delegated)
    document.addEventListener('click', (e) => {
        if (e.target.closest('.btn-download')) {
            const button = e.target.closest('.btn-download');
            const filename = button.dataset.file;
            handleDownload(filename);
        }
    });
}

// Check if all files are uploaded
function checkAllFilesUploaded() {
    const allUploaded = Object.values(uploadedFiles).every(file => file !== null);
    elements.uploadBtn.disabled = !allUploaded;
}

// Truncate filename
function truncateFilename(name, maxLength) {
    if (name.length <= maxLength) return name;
    const extension = name.split('.').pop();
    const nameWithoutExt = name.substring(0, name.lastIndexOf('.'));
    const truncated = nameWithoutExt.substring(0, maxLength - extension.length - 4) + '...';
    return truncated + '.' + extension;
}

// Handle file upload
async function handleUpload() {
    try {
        // Show progress
        elements.uploadProgress.classList.remove('hidden');
        elements.uploadBtn.disabled = true;

        // Create FormData
        const formData = new FormData();
        formData.append('t1', uploadedFiles.t1);
        formData.append('t1ce', uploadedFiles.t1ce);
        formData.append('t2', uploadedFiles.t2);
        formData.append('flair', uploadedFiles.flair);

        // Update progress
        updateProgress(30, 'Uploading files...');

        // Upload files
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Upload failed');
        }

        const data = await response.json();
        currentSessionId = data.session_id;

        updateProgress(100, 'Upload complete!');
        showToast('Files uploaded successfully', 'success');

        // Start inference
        setTimeout(() => {
            startInference();
        }, 500);

    } catch (error) {
        console.error('Upload error:', error);
        showToast('Upload failed: ' + error.message, 'error');
        elements.uploadProgress.classList.add('hidden');
        elements.uploadBtn.disabled = false;
    }
}

// Update progress bar
function updateProgress(percent, text) {
    elements.uploadProgressFill.style.width = `${percent}%`;
    elements.uploadProgressText.textContent = text;
}

// Start inference
async function startInference() {
    try {
        // Hide upload section
        elements.uploadSection.classList.add('hidden');
        elements.processingSection.classList.remove('hidden');

        // Animate processing steps
        animateProcessingSteps();

        // Run inference
        const response = await fetch(`/api/predict/${currentSessionId}`, {
            method: 'POST'
        });

        if (!response.ok) {
            throw new Error('Inference failed');
        }

        const data = await response.json();

        // Show results
        displayResults(data);

    } catch (error) {
        console.error('Inference error:', error);
        showError('Inference failed: ' + error.message);
    }
}

// Animate processing steps
async function animateProcessingSteps() {
    const steps = [
        { element: elements.step1, text: 'Loading volumes...', delay: 0 },
        { element: elements.step2, text: 'Normalizing data...', delay: 2000 },
        { element: elements.step3, text: 'Running inference...', delay: 4000 },
        { element: elements.step4, text: 'Generating results...', delay: 8000 }
    ];

    for (const step of steps) {
        await new Promise(resolve => setTimeout(resolve, step.delay));
        elements.processingText.textContent = step.text;
        step.element.classList.add('active');

        // Complete previous steps
        steps.forEach(s => {
            if (s !== step && s.delay < step.delay) {
                s.element.classList.remove('active');
                s.element.classList.add('completed');
                s.element.querySelector('.step-icon').textContent = '✓';
            }
        });
    }
}

// Display results
function displayResults(data) {
    // Hide processing
    elements.processingSection.classList.add('hidden');
    elements.resultsSection.classList.remove('hidden');

    // Update statistics
    const stats = data.statistics;
    document.getElementById('wtVolume').textContent = formatNumber(stats.wt_voxels) + ' voxels';
    document.getElementById('wtPercentage').textContent = stats.wt_percentage.toFixed(2) + '%';

    document.getElementById('tcVolume').textContent = formatNumber(stats.tc_voxels) + ' voxels';
    document.getElementById('tcPercentage').textContent = stats.tc_percentage.toFixed(2) + '%';

    document.getElementById('etVolume').textContent = formatNumber(stats.et_voxels) + ' voxels';
    document.getElementById('etPercentage').textContent = stats.et_percentage.toFixed(2) + '%';

    // Display visualizations
    displayVisualizations(data.visualizations);

    // Show success toast
    showToast('Segmentation complete! Results are ready.', 'success');

    // Scroll to results
    elements.resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// Display visualizations
function displayVisualizations(visualizations) {
    elements.visualizationContainer.innerHTML = '';

    visualizations.forEach((viz, index) => {
        const div = document.createElement('div');
        div.className = 'visualization-item';

        const title = document.createElement('h5');
        title.textContent = `Slice ${viz.slice}`;

        const img = document.createElement('img');
        img.src = `data:image/png;base64,${viz.image}`;
        img.alt = `Segmentation visualization for slice ${viz.slice}`;

        div.appendChild(title);
        div.appendChild(img);
        elements.visualizationContainer.appendChild(div);
    });
}

// Format number with commas
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

// Handle download
async function handleDownload(filename) {
    try {
        const url = `/api/download/${currentSessionId}/${filename}`;
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);

        showToast(`Downloading ${filename}...`, 'info');
    } catch (error) {
        console.error('Download error:', error);
        showToast('Download failed: ' + error.message, 'error');
    }
}

// Handle clear
function handleClear() {
    // Reset uploaded files
    uploadedFiles = {
        t1: null,
        t1ce: null,
        t2: null,
        flair: null
    };

    // Clear file inputs
    const modalities = ['t1', 't1ce', 't2', 'flair'];
    modalities.forEach(mod => {
        elements[`${mod}Input`].value = '';
        const label = document.querySelector(`label[for="${mod}-upload"]`);
        label.classList.remove('has-file');
        document.getElementById(`${mod}-filename`).textContent = 'No file chosen';
        document.getElementById(`${mod}-status`).textContent = '';
    });

    // Disable upload button
    elements.uploadBtn.disabled = true;

    // Hide progress
    elements.uploadProgress.classList.add('hidden');
    updateProgress(0, '');

    showToast('All files cleared', 'info');
}

// Handle new scan
async function handleNewScan() {
    // Cleanup session
    if (currentSessionId) {
        try {
            await fetch(`/api/cleanup/${currentSessionId}`, {
                method: 'DELETE'
            });
        } catch (error) {
            console.error('Cleanup error:', error);
        }
    }

    // Reset state
    currentSessionId = null;
    handleClear();

    // Reset processing steps
    [elements.step1, elements.step2, elements.step3, elements.step4].forEach(step => {
        step.classList.remove('active', 'completed');
        step.querySelector('.step-icon').textContent = '⏳';
    });

    // Show upload section
    elements.resultsSection.classList.add('hidden');
    elements.uploadSection.classList.remove('hidden');

    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Handle retry
function handleRetry() {
    elements.errorSection.classList.add('hidden');
    elements.uploadSection.classList.remove('hidden');
}

// Show error
function showError(message) {
    elements.processingSection.classList.add('hidden');
    elements.errorSection.classList.remove('hidden');
    elements.errorMessage.textContent = message;
}

// Show toast notification
function showToast(message, type = 'info') {
    const toastContainer = document.getElementById('toastContainer');

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;

    const icons = {
        success: '✓',
        error: '✗',
        info: 'ℹ'
    };

    toast.innerHTML = `
        <span class="toast-icon">${icons[type] || 'ℹ'}</span>
        <span class="toast-message">${message}</span>
    `;

    toastContainer.appendChild(toast);

    // Auto-remove after 5 seconds
    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => {
            toast.remove();
        }, 300);
    }, 5000);
}

// Add slideOut animation
const style = document.createElement('style');
style.textContent = `
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);
