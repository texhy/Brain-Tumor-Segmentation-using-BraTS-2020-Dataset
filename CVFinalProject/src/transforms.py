"""
Data preprocessing and augmentation transforms for BraTS 2020 dataset.
"""

import numpy as np
from typing import Tuple, List
import random


def z_score_normalize(volume: np.ndarray) -> np.ndarray:
    """
    Apply z-score normalization independently to each modality.
    
    Args:
        volume: Input volume of shape (C, D, H, W) where C is number of modalities
        
    Returns:
        Normalized volume with mean ~0 and std ~1 per channel
    """
    normalized = np.zeros_like(volume, dtype=np.float32)
    
    for c in range(volume.shape[0]):
        channel = volume[c]
        mean = np.mean(channel)
        std = np.std(channel)
        
        # Avoid division by zero
        if std > 1e-8:
            normalized[c] = (channel - mean) / std
        else:
            normalized[c] = channel - mean
    
    return normalized


def extract_patches(volume: np.ndarray, mask: np.ndarray, patch_size: int = 128, 
                   overlap: float = 0.5) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Extract overlapping 3D patches from volume and mask.
    
    Args:
        volume: Input volume of shape (C, D, H, W)
        mask: Segmentation mask of shape (D, H, W) or (C, D, H, W)
        patch_size: Size of cubic patches (default: 128)
        overlap: Overlap ratio between patches (default: 0.5)
        
    Returns:
        List of (volume_patch, mask_patch) tuples
    """
    patches = []
    
    # Handle mask shape
    if mask.ndim == 3:
        mask = mask[np.newaxis, ...]  # Add channel dimension
    
    _, d, h, w = volume.shape
    
    # Calculate stride based on overlap
    stride = int(patch_size * (1 - overlap))
    
    # Extract patches with sliding window
    for z in range(0, d - patch_size + 1, stride):
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                vol_patch = volume[:, z:z+patch_size, y:y+patch_size, x:x+patch_size]
                mask_patch = mask[:, z:z+patch_size, y:y+patch_size, x:x+patch_size]
                
                patches.append((vol_patch, mask_patch))
    
    return patches


def apply_augmentation(volume: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply random augmentations to volume and mask.
    
    Augmentations include:
    - Random flips (along each axis)
    - Random 90-degree rotations
    - Gaussian noise
    - Intensity shifts
    
    Args:
        volume: Input volume of shape (C, D, H, W)
        mask: Segmentation mask of shape (C, D, H, W)
        
    Returns:
        Augmented (volume, mask) tuple
    """
    vol_aug = volume.copy()
    mask_aug = mask.copy()
    
    # Random flips along each spatial axis
    if random.random() > 0.5:
        vol_aug = np.flip(vol_aug, axis=1)  # Flip along D
        mask_aug = np.flip(mask_aug, axis=1)
    
    if random.random() > 0.5:
        vol_aug = np.flip(vol_aug, axis=2)  # Flip along H
        mask_aug = np.flip(mask_aug, axis=2)
    
    if random.random() > 0.5:
        vol_aug = np.flip(vol_aug, axis=3)  # Flip along W
        mask_aug = np.flip(mask_aug, axis=3)
    
    # Random 90-degree rotation in axial plane (H-W plane)
    k = random.randint(0, 3)  # 0, 90, 180, or 270 degrees
    if k > 0:
        vol_aug = np.rot90(vol_aug, k=k, axes=(2, 3))
        mask_aug = np.rot90(mask_aug, k=k, axes=(2, 3))
    
    # Gaussian noise (only to volume, not mask)
    if random.random() > 0.5:
        noise_std = random.uniform(0.01, 0.1)
        noise = np.random.normal(0, noise_std, vol_aug.shape).astype(np.float32)
        vol_aug = vol_aug + noise
    
    # Intensity shift (only to volume, not mask)
    if random.random() > 0.5:
        shift = random.uniform(-0.1, 0.1)
        vol_aug = vol_aug + shift
    
    # Intensity scaling (only to volume, not mask)
    if random.random() > 0.5:
        scale = random.uniform(0.9, 1.1)
        vol_aug = vol_aug * scale
    
    return vol_aug.copy(), mask_aug.copy()


def center_crop(volume: np.ndarray, target_size: Tuple[int, int, int]) -> np.ndarray:
    """
    Center crop volume to target size.
    
    Args:
        volume: Input volume of shape (C, D, H, W)
        target_size: Target spatial dimensions (D, H, W)
        
    Returns:
        Cropped volume
    """
    _, d, h, w = volume.shape
    td, th, tw = target_size
    
    # Calculate crop indices
    d_start = (d - td) // 2
    h_start = (h - th) // 2
    w_start = (w - tw) // 2
    
    return volume[:, d_start:d_start+td, h_start:h_start+th, w_start:w_start+tw]


def resize(volume: np.ndarray, target_size: Tuple[int, int, int]) -> np.ndarray:
    """
    Resize volume to target size using nearest neighbor interpolation.
    
    Args:
        volume: Input volume of shape (C, D, H, W)
        target_size: Target spatial dimensions (D, H, W)
        
    Returns:
        Resized volume
    """
    from scipy.ndimage import zoom
    
    _, d, h, w = volume.shape
    td, th, tw = target_size
    
    # Calculate zoom factors
    zoom_factors = [1, td/d, th/h, tw/w]
    
    return zoom(volume, zoom_factors, order=1)
