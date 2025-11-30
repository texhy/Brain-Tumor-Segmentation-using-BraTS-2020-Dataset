"""
Script to download BraTS 2020 dataset using kagglehub.
"""

import os
import shutil
import kagglehub


def download_brats_dataset(target_dir: str = "./data/BraTS2020"):
    """
    Download BraTS 2020 dataset from Kaggle using kagglehub.
    
    Args:
        target_dir: Target directory to store the dataset
    """
    print("Downloading BraTS 2020 dataset from Kaggle...")
    print("This may take several minutes depending on your connection speed.")
    
    # Download dataset using kagglehub
    path = kagglehub.dataset_download("awsaf49/brats20-dataset-training-validation")
    
    print(f"Dataset downloaded to: {path}")
    
    # Create target directory if it doesn't exist
    os.makedirs(os.path.dirname(target_dir), exist_ok=True)
    
    # Move or symlink to target directory
    if os.path.exists(target_dir):
        print(f"Target directory {target_dir} already exists.")
        response = input("Do you want to overwrite it? (yes/no): ")
        if response.lower() != 'yes':
            print("Using existing dataset directory.")
            return target_dir
        shutil.rmtree(target_dir)
    
    # Copy dataset to target location
    print(f"Moving dataset to {target_dir}...")
    shutil.copytree(path, target_dir)
    
    print(f"Dataset successfully set up at: {target_dir}")
    
    # Verify dataset structure
    if os.path.exists(target_dir):
        subjects = [d for d in os.listdir(target_dir) 
                   if os.path.isdir(os.path.join(target_dir, d)) and d.startswith('BraTS')]
        print(f"Found {len(subjects)} subjects in the dataset")
    
    return target_dir


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download BraTS 2020 dataset")
    parser.add_argument("--target_dir", type=str, default="./data/BraTS2020",
                       help="Target directory for dataset (default: ./data/BraTS2020)")
    
    args = parser.parse_args()
    
    download_brats_dataset(args.target_dir)
