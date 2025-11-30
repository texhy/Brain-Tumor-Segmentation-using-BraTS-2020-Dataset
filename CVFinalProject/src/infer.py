"""
Inference script for BraTS 2020 brain tumor segmentation.
"""

import os
import argparse
import numpy as np
import nibabel as nib
import torch
from tqdm import tqdm

from src.model import UNet3D
from src.transforms import z_score_normalize
from src.utils import load_checkpoint


def sliding_window_inference(model, volume, patch_size=128, overlap=0.5, device='cuda'):
    """
    Perform sliding window inference on full volume.
    
    Args:
        model: Trained segmentation model
        volume: (4, D, H, W) full MRI volume
        patch_size: Size of patches (default: 128)
        overlap: Overlap ratio (default: 0.5)
        device: Device to run inference on
        
    Returns:
        (3, D, H, W) segmentation prediction
    """
    model.eval()
    
    _, d, h, w = volume.shape
    stride = int(patch_size * (1 - overlap))
    
    # Initialize output volume and count map for averaging
    output = np.zeros((3, d, h, w), dtype=np.float32)
    count_map = np.zeros((d, h, w), dtype=np.float32)
    
    # Calculate number of patches
    z_steps = max(1, (d - patch_size) // stride + 1)
    y_steps = max(1, (h - patch_size) // stride + 1)
    x_steps = max(1, (w - patch_size) // stride + 1)
    
    total_patches = z_steps * y_steps * x_steps
    
    print(f"Processing {total_patches} patches...")
    
    with torch.no_grad():
        pbar = tqdm(total=total_patches, desc="Inference")
        
        for z in range(0, d - patch_size + 1, stride):
            for y in range(0, h - patch_size + 1, stride):
                for x in range(0, w - patch_size + 1, stride):
                    # Extract patch
                    patch = volume[:, z:z+patch_size, y:y+patch_size, x:x+patch_size]
                    
                    # Convert to tensor and add batch dimension
                    patch_tensor = torch.from_numpy(patch).unsqueeze(0).float().to(device)
                    
                    # Forward pass
                    pred = model(patch_tensor)
                    
                    # Apply sigmoid to get probabilities
                    pred = torch.sigmoid(pred).cpu().numpy()[0]
                    
                    # Add to output with averaging for overlapping regions
                    output[:, z:z+patch_size, y:y+patch_size, x:x+patch_size] += pred
                    count_map[z:z+patch_size, y:y+patch_size, x:x+patch_size] += 1
                    
                    pbar.update(1)
        
        pbar.close()
    
    # Average overlapping predictions
    count_map[count_map == 0] = 1  # Avoid division by zero
    output = output / count_map[np.newaxis, ...]
    
    return output


def load_volume(subject_path):
    """
    Load all four modalities for a subject.
    
    Args:
        subject_path: Path to subject directory
        
    Returns:
        volume: (4, D, H, W) array with T1, T1ce, T2, FLAIR
        affine: Affine matrix from NIfTI file
    """
    subject_id = os.path.basename(subject_path)
    modalities = ['t1', 't1ce', 't2', 'flair']
    volumes = []
    affine = None
    
    print(f"Loading subject: {subject_id}")
    
    for modality in modalities:
        file_path = os.path.join(subject_path, f"{subject_id}_{modality}.nii.gz")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing modality file: {file_path}")
        
        nii = nib.load(file_path)
        volume = nii.get_fdata().astype(np.float32)
        volumes.append(volume)
        
        if affine is None:
            affine = nii.affine
    
    # Stack modalities
    volume = np.stack(volumes, axis=0)  # Shape: (4, D, H, W)
    
    print(f"Volume shape: {volume.shape}")
    
    return volume, affine


def save_prediction(prediction, output_dir, subject_id, affine):
    """
    Save prediction as NIfTI files.
    
    Args:
        prediction: (3, D, H, W) segmentation prediction
        output_dir: Output directory
        subject_id: Subject identifier
        affine: Affine matrix for NIfTI file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Threshold predictions to binary masks
    wt = (prediction[0] > 0.5).astype(np.uint8)
    tc = (prediction[1] > 0.5).astype(np.uint8)
    et = (prediction[2] > 0.5).astype(np.uint8)
    
    # Save individual masks
    masks = {
        'wt': wt,  # Whole Tumor
        'tc': tc,  # Tumor Core
        'et': et   # Enhancing Tumor
    }
    
    for mask_name, mask_data in masks.items():
        output_path = os.path.join(output_dir, f"{subject_id}_{mask_name}.nii.gz")
        nii = nib.Nifti1Image(mask_data, affine)
        nib.save(nii, output_path)
        print(f"Saved: {output_path}")
    
    # Create combined segmentation (BraTS format)
    # 0=background, 1=NCR/NET, 2=ED, 4=ET
    combined = np.zeros_like(wt, dtype=np.uint8)
    combined[et == 1] = 4  # Enhancing tumor
    combined[(tc == 1) & (et == 0)] = 1  # Tumor core (non-enhancing)
    combined[(wt == 1) & (tc == 0)] = 2  # Edema
    
    combined_path = os.path.join(output_dir, f"{subject_id}_seg.nii.gz")
    nii = nib.Nifti1Image(combined, affine)
    nib.save(nii, combined_path)
    print(f"Saved: {combined_path}")


def main(checkpoint_path, input_path, output_dir, use_transformer=True):
    """
    Main inference function.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        input_path: Path to subject directory or directory containing subjects
        output_dir: Output directory for predictions
        use_transformer: Whether model uses transformer (default: True)
    """
    print("=" * 60)
    print("BraTS 2020 Brain Tumor Segmentation - Inference")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model
    print(f"\nLoading model from: {checkpoint_path}")
    model = UNet3D(in_channels=4, num_classes=3, use_transformer=use_transformer).to(device)
    load_checkpoint(checkpoint_path, model)
    model.eval()
    
    print("Model loaded successfully!")
    
    # Check if input is a single subject or directory of subjects
    if os.path.isdir(input_path):
        # Check if it's a single subject directory
        subject_id = os.path.basename(input_path)
        if subject_id.startswith('BraTS'):
            # Single subject
            subjects = [(input_path, subject_id)]
        else:
            # Directory containing multiple subjects
            subjects = []
            for subject_id in os.listdir(input_path):
                subject_path = os.path.join(input_path, subject_id)
                if os.path.isdir(subject_path) and subject_id.startswith('BraTS'):
                    subjects.append((subject_path, subject_id))
    else:
        raise ValueError(f"Input path must be a directory: {input_path}")
    
    print(f"\nFound {len(subjects)} subject(s) to process")
    
    # Process each subject
    for subject_path, subject_id in subjects:
        print("\n" + "=" * 60)
        print(f"Processing: {subject_id}")
        print("=" * 60)
        
        try:
            # Load volume
            volume, affine = load_volume(subject_path)
            
            # Normalize
            print("Normalizing volume...")
            volume = z_score_normalize(volume)
            
            # Run inference
            print("Running sliding window inference...")
            prediction = sliding_window_inference(
                model, volume, patch_size=128, overlap=0.5, device=device
            )
            
            # Save predictions
            print("Saving predictions...")
            save_prediction(prediction, output_dir, subject_id, affine)
            
            print(f"\nCompleted: {subject_id}")
            
        except Exception as e:
            print(f"Error processing {subject_id}: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("Inference Complete!")
    print("=" * 60)
    print(f"\nPredictions saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on BraTS 2020 data")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--input", type=str, required=True,
                       help="Path to subject directory or directory containing subjects")
    parser.add_argument("--output", type=str, required=True,
                       help="Output directory for predictions")
    parser.add_argument("--no_transformer", action="store_true",
                       help="Model was trained without transformer")
    
    args = parser.parse_args()
    
    main(args.checkpoint, args.input, args.output, use_transformer=not args.no_transformer)
