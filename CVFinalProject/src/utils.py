"""
Utility functions for training, checkpointing, and AWS S3 integration.
"""

import os
import torch
import torch.nn as nn
import boto3
from botocore.exceptions import ClientError
from typing import Dict, Optional
import time


class DiceLoss(nn.Module):
    """Soft Dice loss for segmentation."""
    
    def __init__(self, smooth: float = 1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Args:
            pred: (B, C, D, H, W) predicted logits
            target: (B, C, D, H, W) ground truth masks
            
        Returns:
            Dice loss value
        """
        # Apply sigmoid to predictions
        pred = torch.sigmoid(pred)
        
        # Flatten spatial dimensions
        pred = pred.view(pred.size(0), pred.size(1), -1)
        target = target.view(target.size(0), target.size(1), -1)
        
        # Compute Dice coefficient
        intersection = (pred * target).sum(dim=2)
        union = pred.sum(dim=2) + target.sum(dim=2)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Return Dice loss (1 - Dice)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """Combined Dice Loss and Cross-Entropy Loss."""
    
    def __init__(self, dice_weight: float = 0.5, ce_weight: float = 0.5):
        super(CombinedLoss, self).__init__()
        
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            pred: (B, C, D, H, W) predicted logits
            target: (B, C, D, H, W) ground truth masks
            
        Returns:
            Combined loss value
        """
        dice = self.dice_loss(pred, target)
        ce = self.ce_loss(pred, target)
        
        return self.dice_weight * dice + self.ce_weight * ce


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, 
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                   epoch: int, train_loss: float, val_loss: float,
                   val_dice_scores: Dict[str, float], config: dict,
                   subject_ids: Dict[str, list], save_path: str):
    """
    Save training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Learning rate scheduler state
        epoch: Current epoch number
        train_loss: Training loss
        val_loss: Validation loss
        val_dice_scores: Dictionary of validation Dice scores
        config: Configuration dictionary
        subject_ids: Dictionary with 'train' and 'val' subject lists
        save_path: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_dice_scores': val_dice_scores,
        'config': config,
        'subject_ids': subject_ids
    }
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(checkpoint_path: str, model: nn.Module, 
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> int:
    """
    Load training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        
    Returns:
        Epoch number from checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Use strict=False to handle dynamic components (like PositionalEncoding)
    # that might not be in the model definition at initialization time
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    if missing_keys:
        print(f"Warning: Missing keys in model: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    
    print(f"Checkpoint loaded from {checkpoint_path} (epoch {epoch})")
    
    return epoch


def upload_to_s3(local_path: str, bucket: str, s3_path: str, max_retries: int = 3):
    """
    Upload file to S3 bucket with retry logic.
    
    Args:
        local_path: Local file path
        bucket: S3 bucket name
        s3_path: S3 object key
        max_retries: Maximum number of retry attempts
    """
    s3_client = boto3.client('s3')
    
    for attempt in range(max_retries):
        try:
            s3_client.upload_file(local_path, bucket, s3_path)
            print(f"Successfully uploaded {local_path} to s3://{bucket}/{s3_path}")
            return
        except ClientError as e:
            print(f"Upload attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"Failed to upload after {max_retries} attempts")
                raise


def download_from_s3(bucket: str, s3_path: str, local_path: str):
    """
    Download file from S3 bucket.
    
    Args:
        bucket: S3 bucket name
        s3_path: S3 object key
        local_path: Local file path to save to
    """
    s3_client = boto3.client('s3')
    
    try:
        s3_client.download_file(bucket, s3_path, local_path)
        print(f"Successfully downloaded s3://{bucket}/{s3_path} to {local_path}")
    except ClientError as e:
        print(f"Failed to download from S3: {e}")
        raise


def check_s3_checkpoint(bucket: str, s3_path: str) -> bool:
    """
    Check if checkpoint exists on S3.
    
    Args:
        bucket: S3 bucket name
        s3_path: S3 object key
        
    Returns:
        True if checkpoint exists, False otherwise
    """
    s3_client = boto3.client('s3')
    
    try:
        s3_client.head_object(Bucket=bucket, Key=s3_path)
        return True
    except ClientError:
        return False


def check_for_pretrained(checkpoint_dir: str, bucket: Optional[str] = None, 
                        s3_path: Optional[str] = None) -> Optional[str]:
    """
    Check for pretrained checkpoint locally and on S3.
    
    Args:
        checkpoint_dir: Local checkpoint directory
        bucket: S3 bucket name (optional)
        s3_path: S3 checkpoint path (optional)
        
    Returns:
        Path to pretrained checkpoint if found, None otherwise
    """
    # Check locally first
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        if checkpoints:
            latest_checkpoint = os.path.join(checkpoint_dir, sorted(checkpoints)[-1])
            print(f"Pretrained model detected locally: {latest_checkpoint}")
            return latest_checkpoint
    
    # Check S3 if configured
    if bucket and s3_path:
        if check_s3_checkpoint(bucket, s3_path):
            print(f"Pretrained model detected on S3: s3://{bucket}/{s3_path}")
            local_path = os.path.join(checkpoint_dir, 'pretrained.pth')
            os.makedirs(checkpoint_dir, exist_ok=True)
            download_from_s3(bucket, s3_path, local_path)
            return local_path
    
    return None


def compute_dice_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Compute Dice score for binary segmentation.
    
    Args:
        pred: Predicted logits or probabilities
        target: Ground truth binary mask
        threshold: Threshold for binarizing predictions
        
    Returns:
        Dice score
    """
    pred = (torch.sigmoid(pred) > threshold).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    dice = (2.0 * intersection) / union
    return dice.item()
