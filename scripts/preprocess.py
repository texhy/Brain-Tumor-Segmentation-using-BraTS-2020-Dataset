"""
Preprocessing script to verify BraTS 2020 dataset structure and compute statistics.
"""

import os
import argparse
import numpy as np
import nibabel as nib
from tqdm import tqdm
for modality in modalities:
    file_stem = f"{subject_id}_{modality}"
    try:
        _resolve_nifti_path(subject_path, file_stem)
    except FileNotFoundError:
        missing_files.append((subject_id, modality))
        subject_valid = False


def verify_dataset(data_dir):
    """
    Verify dataset structure and check for missing files.
    
    Args:
        data_dir: Root directory containing BraTS subjects
    """
    print("=" * 60)
    print("BraTS 2020 Dataset Verification")
    print("=" * 60)
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        return False
    
    # Get all subject directories
    subjects = [d for d in os.listdir(data_dir) 
               if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('BraTS')]
    
    print(f"\nFound {len(subjects)} subjects")
    
    if len(subjects) == 0:
        print("Error: No BraTS subjects found in directory")
        return False
    
    # Check each subject
    modalities = ['t1', 't1ce', 't2', 'flair', 'seg']
    missing_files = []
    valid_subjects = []
    
    print("\nVerifying subjects...")
    for subject_id in tqdm(subjects):
        subject_path = os.path.join(data_dir, subject_id)
        subject_valid = True
        
        for modality in modalities:
            file_stem = f"{subject_id}_{modality}"
            
            try:
                _resolve_nifti_path(subject_path, file_stem)
            except FileNotFoundError:
                missing_files.append((subject_id, modality))
                subject_valid = False
        
        if subject_valid:
            valid_subjects.append(subject_id)
    
    # Report results
    print("\n" + "=" * 60)
    print("Verification Results")
    print("=" * 60)
    print(f"Total subjects: {len(subjects)}")
    print(f"Valid subjects: {len(valid_subjects)}")
    print(f"Subjects with missing files: {len(subjects) - len(valid_subjects)}")
    
    if missing_files:
        print("\nMissing files:")
        for subject_id, modality in missing_files[:10]:  # Show first 10
            print(f"  {subject_id}: {modality}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")
    
    return len(valid_subjects) > 0


def compute_statistics(data_dir, num_samples=10):
    """
    Compute dataset statistics (mean, std, shape distribution).
    
    Args:
        data_dir: Root directory containing BraTS subjects
        num_samples: Number of subjects to sample for statistics
    """
    print("\n" + "=" * 60)
    print("Computing Dataset Statistics")
    print("=" * 60)
    
    subjects = [d for d in os.listdir(data_dir) 
               if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('BraTS')]
    
    # Sample subjects
    import random
    random.seed(42)
    sampled_subjects = random.sample(subjects, min(num_samples, len(subjects)))
    
    print(f"\nSampling {len(sampled_subjects)} subjects for statistics...")
    
    modalities = ['t1', 't1ce', 't2', 'flair']
    shapes = []
    intensities = {mod: [] for mod in modalities}
    label_counts = {'background': 0, 'ncr': 0, 'ed': 0, 'et': 0}
    
    for subject_id in tqdm(sampled_subjects):
        subject_path = os.path.join(data_dir, subject_id)
        
        # Load modalities
        for modality in modalities:
            file_stem = f"{subject_id}_{modality}"
            
            try:
                file_path = _resolve_nifti_path(subject_path, file_stem)
                nii = nib.load(file_path)
                volume = nii.get_fdata()
                
                # Record shape
                if modality == 't1':
                    shapes.append(volume.shape)
                
                # Record intensity statistics (non-zero voxels only)
                non_zero = volume[volume > 0]
                if len(non_zero) > 0:
                    intensities[modality].append({
                        'mean': np.mean(non_zero),
                        'std': np.std(non_zero),
                        'min': np.min(non_zero),
                        'max': np.max(non_zero)
                    })
            except Exception as e:
                print(f"Error loading {subject_id} {modality}: {e}")
        
        # Load segmentation
        try:
            seg_path = _resolve_nifti_path(subject_path, f"{subject_id}_seg")
            seg_nii = nib.load(seg_path)
            seg = seg_nii.get_fdata()
            
            label_counts['background'] += np.sum(seg == 0)
            label_counts['ncr'] += np.sum(seg == 1)
            label_counts['ed'] += np.sum(seg == 2)
            label_counts['et'] += np.sum(seg == 4)
        except Exception as e:
            print(f"Error loading {subject_id} segmentation: {e}")
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    
    # Shape statistics
    print("\nVolume Shapes:")
    unique_shapes = list(set(shapes))
    for shape in unique_shapes:
        count = shapes.count(shape)
        print(f"  {shape}: {count} subjects")
    
    # Intensity statistics
    print("\nIntensity Statistics (non-zero voxels):")
    for modality in modalities:
        if intensities[modality]:
            means = [s['mean'] for s in intensities[modality]]
            stds = [s['std'] for s in intensities[modality]]
            mins = [s['min'] for s in intensities[modality]]
            maxs = [s['max'] for s in intensities[modality]]
            
            print(f"\n  {modality.upper()}:")
            print(f"    Mean: {np.mean(means):.2f} ± {np.std(means):.2f}")
            print(f"    Std: {np.mean(stds):.2f} ± {np.std(stds):.2f}")
            print(f"    Min: {np.mean(mins):.2f}")
            print(f"    Max: {np.mean(maxs):.2f}")
    
    # Label distribution
    print("\nLabel Distribution:")
    total_voxels = sum(label_counts.values())
    for label, count in label_counts.items():
        percentage = (count / total_voxels) * 100 if total_voxels else 0
        print(f"  {label}: {percentage:.2f}%")


def main(data_dir, verify_only=False):
    """
    Main preprocessing function.
    
    Args:
        data_dir: Root directory containing BraTS subjects
        verify_only: If True, only verify dataset structure
    """
    # Verify dataset
    is_valid = verify_dataset(data_dir)
    
    if not is_valid:
        print("\nDataset verification failed!")
        return
    
    # Compute statistics if requested
    if not verify_only:
        compute_statistics(data_dir)
    
    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and verify BraTS 2020 dataset")
    parser.add_argument("--data_dir", type=str, default="./data/BraTS2020",
                       help="Path to BraTS 2020 dataset directory")
    parser.add_argument("--verify_only", action="store_true",
                       help="Only verify dataset structure, skip statistics")
    
    args = parser.parse_args()
    
    main(args.data_dir, args.verify_only)

