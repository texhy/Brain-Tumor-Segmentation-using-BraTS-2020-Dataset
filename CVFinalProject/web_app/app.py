"""
Flask Web Application for Brain Tumor Segmentation
Provides a web interface for uploading MRI scans and running inference.
"""

import os
import sys
import json
import uuid
import shutil
import numpy as np
import nibabel as nib
import torch
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import UNet3D
from src.transforms import z_score_normalize
from src.utils import load_checkpoint

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB max upload
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MODEL_PATH'] = '../best_model.pth'

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Global model variable
model = None
device = None

def initialize_model():
    """Load the trained model."""
    global model, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = UNet3D(in_channels=4, num_classes=3, use_transformer=True).to(device)
    
    # Load checkpoint
    model_path = os.path.join(os.path.dirname(__file__), app.config['MODEL_PATH'])
    if os.path.exists(model_path):
        load_checkpoint(model_path, model)
        model.eval()
        print(f"Model loaded from: {model_path}")
    else:
        print(f"Warning: Model not found at {model_path}")
        print("Please ensure best_model.pth is in the correct location")
    
    return model is not None

def sliding_window_inference(volume, patch_size=128, overlap=0.5):
    """
    Perform sliding window inference on full volume.
    
    Args:
        volume: (4, D, H, W) normalized MRI volume
        patch_size: Size of patches
        overlap: Overlap ratio
        
    Returns:
        (3, D, H, W) segmentation prediction
    """
    _, d, h, w = volume.shape
    stride = int(patch_size * (1 - overlap))
    
    # Initialize output and count map
    output = np.zeros((3, d, h, w), dtype=np.float32)
    count_map = np.zeros((d, h, w), dtype=np.float32)
    
    with torch.no_grad():
        for z in range(0, max(1, d - patch_size + 1), stride):
            for y in range(0, max(1, h - patch_size + 1), stride):
                for x in range(0, max(1, w - patch_size + 1), stride):
                    # Extract patch
                    z_end = min(z + patch_size, d)
                    y_end = min(y + patch_size, h)
                    x_end = min(x + patch_size, w)
                    
                    patch = volume[:, z:z_end, y:y_end, x:x_end]
                    
                    # Pad if necessary
                    if patch.shape != (4, patch_size, patch_size, patch_size):
                        padded = np.zeros((4, patch_size, patch_size, patch_size), dtype=np.float32)
                        padded[:, :patch.shape[1], :patch.shape[2], :patch.shape[3]] = patch
                        patch = padded
                    
                    # Convert to tensor
                    patch_tensor = torch.from_numpy(patch).unsqueeze(0).float().to(device)
                    
                    # Forward pass
                    pred = model(patch_tensor)
                    pred = torch.sigmoid(pred).cpu().numpy()[0]
                    
                    # Add to output
                    actual_shape = (min(patch_size, d - z), min(patch_size, h - y), min(patch_size, w - x))
                    output[:, z:z_end, y:y_end, x:x_end] += pred[:, :actual_shape[0], :actual_shape[1], :actual_shape[2]]
                    count_map[z:z_end, y:y_end, x:x_end] += 1
    
    # Average overlapping predictions
    count_map[count_map == 0] = 1
    output = output / count_map[np.newaxis, ...]
    
    return output

def create_visualization(volume_data, seg_data, slice_idx, modality='flair'):
    """
    Create visualization of segmentation overlay.
    
    Args:
        volume_data: Original MRI volume (4, D, H, W)
        seg_data: Segmentation prediction (3, D, H, W)
        slice_idx: Slice index to visualize
        modality: Which modality to display (default: flair)
        
    Returns:
        Base64 encoded PNG image
    """
    modality_map = {'t1': 0, 't1ce': 1, 't2': 2, 'flair': 3}
    mod_idx = modality_map.get(modality, 3)
    
    # Get slice
    mri_slice = volume_data[mod_idx, :, :, slice_idx]
    
    # Combine segmentation masks with different colors
    wt_mask = seg_data[0, :, :, slice_idx] > 0.5
    tc_mask = seg_data[1, :, :, slice_idx] > 0.5
    et_mask = seg_data[2, :, :, slice_idx] > 0.5
    
    # Create RGB overlay
    overlay = np.zeros((*mri_slice.shape, 3), dtype=np.float32)
    overlay[wt_mask] = [0, 1, 0]  # Green for whole tumor
    overlay[tc_mask] = [1, 1, 0]  # Yellow for tumor core
    overlay[et_mask] = [1, 0, 0]  # Red for enhancing tumor
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original MRI
    axes[0].imshow(mri_slice, cmap='gray')
    axes[0].set_title(f'{modality.upper()} MRI (Slice {slice_idx})', fontsize=12)
    axes[0].axis('off')
    
    # Segmentation only
    axes[1].imshow(overlay)
    axes[1].set_title('Tumor Segmentation', fontsize=12)
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(mri_slice, cmap='gray')
    axes[2].imshow(overlay, alpha=0.5)
    axes[2].set_title('Overlay', fontsize=12)
    axes[2].axis('off')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Enhancing Tumor (ET)'),
        Patch(facecolor='yellow', label='Tumor Core (TC)'),
        Patch(facecolor='green', label='Whole Tumor (WT)')
    ]
    axes[2].legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    
    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return img_base64

@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')

@app.route('/api/status')
def status():
    """Get system status."""
    return jsonify({
        'model_loaded': model is not None,
        'device': str(device),
        'cuda_available': torch.cuda.is_available()
    })

@app.route('/api/upload', methods=['POST'])
def upload_files():
    """
    Handle file uploads.
    Expects: t1, t1ce, t2, flair NIfTI files
    """
    try:
        # Check if all required files are present
        required_files = ['t1', 't1ce', 't2', 'flair']
        for mod in required_files:
            if mod not in request.files:
                return jsonify({'error': f'Missing {mod} file'}), 400
        
        # Create unique session ID
        session_id = str(uuid.uuid4())
        session_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        os.makedirs(session_folder, exist_ok=True)
        
        # Save files
        file_paths = {}
        for mod in required_files:
            file = request.files[mod]
            if file.filename == '':
                return jsonify({'error': f'No file selected for {mod}'}), 400
            
            filename = secure_filename(f"{mod}.nii.gz")
            filepath = os.path.join(session_folder, filename)
            file.save(filepath)
            file_paths[mod] = filepath
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': 'Files uploaded successfully'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/<session_id>', methods=['POST'])
def predict(session_id):
    """
    Run inference on uploaded files.
    """
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        session_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        if not os.path.exists(session_folder):
            return jsonify({'error': 'Session not found'}), 404
        
        # Load all modalities
        modalities = ['t1', 't1ce', 't2', 'flair']
        volumes = []
        affine = None
        
        for mod in modalities:
            filepath = os.path.join(session_folder, f"{mod}.nii.gz")
            nii = nib.load(filepath)
            volume = nii.get_fdata().astype(np.float32)
            volumes.append(volume)
            
            if affine is None:
                affine = nii.affine
        
        # Stack modalities
        volume = np.stack(volumes, axis=0)  # (4, D, H, W)
        original_shape = volume.shape[1:]
        
        # Normalize
        volume = z_score_normalize(volume)
        
        # Run inference
        print(f"Running inference on volume shape: {volume.shape}")
        prediction = sliding_window_inference(volume, patch_size=128, overlap=0.5)
        
        # Save results
        results_folder = os.path.join(app.config['RESULTS_FOLDER'], session_id)
        os.makedirs(results_folder, exist_ok=True)
        
        # Threshold predictions
        wt = (prediction[0] > 0.5).astype(np.uint8)
        tc = (prediction[1] > 0.5).astype(np.uint8)
        et = (prediction[2] > 0.5).astype(np.uint8)
        
        # Create combined segmentation (BraTS format)
        combined = np.zeros_like(wt, dtype=np.uint8)
        combined[et == 1] = 4  # Enhancing tumor
        combined[(tc == 1) & (et == 0)] = 1  # Tumor core (non-enhancing)
        combined[(wt == 1) & (tc == 0)] = 2  # Edema
        
        # Save segmentation
        seg_path = os.path.join(results_folder, 'segmentation.nii.gz')
        nii_seg = nib.Nifti1Image(combined, affine)
        nib.save(nii_seg, seg_path)
        
        # Save individual masks
        for name, mask in [('wt', wt), ('tc', tc), ('et', et)]:
            mask_path = os.path.join(results_folder, f'{name}.nii.gz')
            nii_mask = nib.Nifti1Image(mask, affine)
            nib.save(nii_mask, mask_path)
        
        # Calculate statistics
        total_voxels = np.prod(original_shape)
        wt_volume = np.sum(wt)
        tc_volume = np.sum(tc)
        et_volume = np.sum(et)
        
        # Find middle slice with tumor
        wt_slices = np.where(np.any(wt, axis=(0, 1)))[0]
        middle_slice = wt_slices[len(wt_slices) // 2] if len(wt_slices) > 0 else original_shape[2] // 2
        
        # Create visualizations for multiple slices
        slice_indices = [
            max(0, middle_slice - 10),
            middle_slice,
            min(original_shape[2] - 1, middle_slice + 10)
        ]
        
        visualizations = []
        for idx in slice_indices:
            img_base64 = create_visualization(volume, prediction, idx, 'flair')
            visualizations.append({
                'slice': idx,
                'image': img_base64
            })
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'statistics': {
                'total_voxels': int(total_voxels),
                'wt_voxels': int(wt_volume),
                'tc_voxels': int(tc_volume),
                'et_voxels': int(et_volume),
                'wt_percentage': float(wt_volume / total_voxels * 100),
                'tc_percentage': float(tc_volume / total_voxels * 100),
                'et_percentage': float(et_volume / total_voxels * 100)
            },
            'visualizations': visualizations,
            'volume_shape': original_shape
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/download/<session_id>/<filename>')
def download_result(session_id, filename):
    """Download result file."""
    try:
        results_folder = os.path.join(app.config['RESULTS_FOLDER'], session_id)
        filepath = os.path.join(results_folder, secure_filename(filename))
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(filepath, as_attachment=True)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cleanup/<session_id>', methods=['DELETE'])
def cleanup_session(session_id):
    """Clean up session files."""
    try:
        # Remove upload folder
        upload_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        if os.path.exists(upload_folder):
            shutil.rmtree(upload_folder)
        
        # Remove results folder
        results_folder = os.path.join(app.config['RESULTS_FOLDER'], session_id)
        if os.path.exists(results_folder):
            shutil.rmtree(results_folder)
        
        return jsonify({'success': True})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("Brain Tumor Segmentation Web Application")
    print("=" * 60)
    
    # Initialize model
    if initialize_model():
        print("\n✅ Model loaded successfully!")
        print(f"✅ Device: {device}")
        print("\nStarting web server...")
        print("Access the application at: http://localhost:5000")
        print("=" * 60)
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n❌ Failed to load model!")
        print("Please ensure best_model.pth is in the correct location")
        print("Expected path: ../best_model.pth relative to web_app/")
