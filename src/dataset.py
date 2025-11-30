"""
BraTS 2020 Dataset class for loading and preprocessing MRI volumes.
"""

import os
import random
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from typing import Tuple, List
from src.transforms import z_score_normalize, extract_patches, apply_augmentation


def _resolve_nifti_path(subject_path: str, file_stem: str) -> str:
    """
    Locate a NIfTI file supporting both .nii.gz and .nii extensions.
    
    Args:
        subject_path: Directory that stores the subject files.
        file_stem: Filename without extension (e.g., BraTS20_Training_001_t1).
    
    Returns:
        Path to the discovered file.
    """
    for ext in ('.nii.gz', '.nii'):
        candidate = os.path.join(subject_path, f"{file_stem}{ext}")
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Missing NIfTI file for stem '{file_stem}' in {subject_path}")


class BraTSDataset(Dataset):
    """
    Dataset class for BraTS 2020 brain tumor segmentation data.
    
    Loads multi-modal MRI volumes (T1, T1ce, T2, FLAIR) and segmentation masks,
    applies preprocessing and augmentation, and extracts 3D patches.
    """
    
    def __init__(self, data_dir: str, subject_ids: List[str], patch_size: int = 128,
                 overlap: float = 0.5, augment: bool = False, test_mode: bool = False):
        """
        Initialize BraTS dataset.
        
        Args:
            data_dir: Root directory containing BraTS 2020 subjects
            subject_ids: List of subject IDs to include in this dataset
            patch_size: Size of 3D patches to extract (default: 128)
            overlap: Overlap ratio for patch extraction (default: 0.5)
            augment: Whether to apply data augmentation (default: False)
            test_mode: If True, limit to 1 subject for testing (default: False)
        """
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.overlap = overlap
        self.augment = augment
        self.test_mode = test_mode
        
        # Limit subjects in test mode
        if test_mode:
            self.subject_ids = subject_ids[:1]
        else:
            self.subject_ids = subject_ids
        
        # Pre-extract all patches from all subjects
        self.patches = []
        self._extract_all_patches()
    
    def _extract_all_patches(self):
        """Extract patches from all subjects and store in memory."""
        for subject_id in self.subject_ids:
            try:
                volume, mask = self.load_subject(subject_id)
                
                # Normalize volume
                volume = z_score_normalize(volume)
                
                # Extract patches
                subject_patches = extract_patches(volume, mask, self.patch_size, self.overlap)
                
                # In test mode, limit number of patches
                if self.test_mode:
                    subject_patches = subject_patches[:10]
                
                self.patches.extend(subject_patches)
                
            except Exception as e:
                print(f"Warning: Failed to load subject {subject_id}: {e}")
                continue
    
    def load_subject(self, subject_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load all four modalities and segmentation mask for a subject.
        
        Args:
            subject_id: Subject identifier (e.g., 'BraTS20_Training_001')
            
        Returns:
            volume: (4, D, H, W) array with T1, T1ce, T2, FLAIR
            mask: (3, D, H, W) array with segmentation labels
        """
        subject_path = os.path.join(self.data_dir, subject_id)
        
        # Load all four modalities
        modalities = ['t1', 't1ce', 't2', 'flair']
        volumes = []
        
        for modality in modalities:
            file_stem = f"{subject_id}_{modality}"
            file_path = _resolve_nifti_path(subject_path, file_stem)
            
            nii = nib.load(file_path)
            volume = nii.get_fdata().astype(np.float32)
            volumes.append(volume)
        
        # Stack modalities along channel dimension
        volume = np.stack(volumes, axis=0)  # Shape: (4, D, H, W)
        
        # Load segmentation mask
        mask_path = _resolve_nifti_path(subject_path, f"{subject_id}_seg")
        
        mask_nii = nib.load(mask_path)
        mask = mask_nii.get_fdata().astype(np.int32)
        
        # Convert BraTS labels to binary masks for WT, TC, ET
        # BraTS labels: 0=background, 1=NCR/NET, 2=ED, 4=ET
        # WT (Whole Tumor) = 1, 2, 4
        # TC (Tumor Core) = 1, 4
        # ET (Enhancing Tumor) = 4
        wt = (mask > 0).astype(np.float32)
        tc = ((mask == 1) | (mask == 4)).astype(np.float32)
        et = (mask == 4).astype(np.float32)
        
        # Stack masks along channel dimension
        mask = np.stack([wt, tc, et], axis=0)  # Shape: (3, D, H, W)
        
        return volume, mask
    
    def __len__(self) -> int:
        """Return total number of patches."""
        return len(self.patches)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single patch.
        
        Args:
            idx: Patch index
            
        Returns:
            volume_patch: (4, D, H, W) tensor
            mask_patch: (3, D, H, W) tensor
        """
        volume_patch, mask_patch = self.patches[idx]
        
        # Apply augmentation if enabled
        if self.augment:
            volume_patch, mask_patch = apply_augmentation(volume_patch, mask_patch)
        
        # Convert to tensors
        volume_tensor = torch.from_numpy(volume_patch).float()
        mask_tensor = torch.from_numpy(mask_patch).float()
        
        return volume_tensor, mask_tensor


def select_subjects(data_dir: str, num_train: int, num_val: int, 
                   log_file: str = 'subject_selection.log', seed: int = 42) -> Tuple[List[str], List[str]]:
    """
    Randomly select training and validation subjects from BraTS dataset.
    
    Args:
        data_dir: Root directory containing BraTS subjects
        num_train: Number of training subjects (20-30)
        num_val: Number of validation subjects (typically 5)
        log_file: Path to log file for recording selected subjects
        seed: Random seed for reproducibility
        
    Returns:
        train_subjects: List of training subject IDs
        val_subjects: List of validation subject IDs
    """
    # Get all subject directories
    all_subjects = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('BraTS')]
    
    # Set random seed for reproducibility
    random.seed(seed)
    random.shuffle(all_subjects)
    
    # Select subjects
    train_subjects = all_subjects[:num_train]
    val_subjects = all_subjects[num_train:num_train + num_val]
    
    # Log subject selection
    with open(log_file, 'w') as f:
        f.write("BraTS 2020 Subject Selection\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Random Seed: {seed}\n")
        f.write(f"Total Subjects Available: {len(all_subjects)}\n\n")
        
        f.write(f"Training Subjects ({len(train_subjects)}):\n")
        for subject in train_subjects:
            f.write(f"  - {subject}\n")
        
        f.write(f"\nValidation Subjects ({len(val_subjects)}):\n")
        for subject in val_subjects:
            f.write(f"  - {subject}\n")
    
    print(f"Subject selection logged to {log_file}")
    print(f"Selected {len(train_subjects)} training and {len(val_subjects)} validation subjects")
    
    return train_subjects, val_subjects

