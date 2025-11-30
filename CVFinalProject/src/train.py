"""
Training script for BraTS 2020 brain tumor segmentation.
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from dotenv import load_dotenv

from src.dataset import BraTSDataset, select_subjects
from src.model import UNet3D
from src.utils import (
    CombinedLoss, save_checkpoint, load_checkpoint,
    upload_to_s3, check_for_pretrained, compute_dice_score
)


def train_epoch(model, dataloader, optimizer, scaler, criterion, 
                accumulation_steps, device, epoch):
    """
    Train for one epoch with AMP and gradient accumulation.
    
    Args:
        model: Model to train
        dataloader: Training data loader
        optimizer: Optimizer
        scaler: GradScaler for mixed precision
        criterion: Loss function
        accumulation_steps: Number of steps to accumulate gradients
        device: Device to train on
        epoch: Current epoch number
        
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, (volumes, masks) in enumerate(pbar):
        volumes = volumes.to(device)
        masks = masks.to(device)
        
        # Forward pass with mixed precision
        with autocast():
            outputs = model(volumes)
            loss = criterion(outputs, masks)
            loss = loss / accumulation_steps
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Optimizer step every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            # Unscale gradients before clipping
            scaler.unscale_(optimizer)
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validate_epoch(model, dataloader, criterion, device, epoch):
    """
    Validate for one epoch.
    
    Args:
        model: Model to validate
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number
        
    Returns:
        avg_loss: Average validation loss
        dice_scores: Dictionary with WT, TC, ET Dice scores
    """
    model.eval()
    total_loss = 0.0
    dice_wt, dice_tc, dice_et = 0.0, 0.0, 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
    
    with torch.no_grad():
        for volumes, masks in pbar:
            volumes = volumes.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(volumes)
            loss = criterion(outputs, masks)
            
            total_loss += loss.item()
            
            # Compute Dice scores for each class
            dice_wt += compute_dice_score(outputs[:, 0], masks[:, 0])
            dice_tc += compute_dice_score(outputs[:, 1], masks[:, 1])
            dice_et += compute_dice_score(outputs[:, 2], masks[:, 2])
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    dice_scores = {
        'WT': dice_wt / len(dataloader),
        'TC': dice_tc / len(dataloader),
        'ET': dice_et / len(dataloader)
    }
    
    return avg_loss, dice_scores


def main(config_path, test_run=False, resume=False):
    """
    Main training function.
    
    Args:
        config_path: Path to configuration YAML file
        test_run: If True, run in test mode (1 subject, 1 epoch)
        resume: If True, resume from latest checkpoint
    """
    # Load environment variables
    load_dotenv()
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 60)
    print("BraTS 2020 Brain Tumor Segmentation - Training")
    print("=" * 60)
    
    # Override config for test mode
    if test_run:
        print("\n[TEST MODE] Running with 1 subject and 1 epoch")
        config['training']['epochs'] = 1
        config['data']['num_train_subjects'] = 1
        config['data']['num_val_subjects'] = 1
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        torch.backends.cudnn.benchmark = True
    
    # Select subjects
    print(f"\nSelecting subjects...")
    train_subjects, val_subjects = select_subjects(
        config['data']['data_dir'],
        config['data']['num_train_subjects'],
        config['data']['num_val_subjects'],
        log_file='subject_selection.log'
    )
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = BraTSDataset(
        data_dir=config['data']['data_dir'],
        subject_ids=train_subjects,
        patch_size=config['data']['patch_size'],
        overlap=config['data']['overlap'],
        augment=True,
        test_mode=test_run
    )
    
    val_dataset = BraTSDataset(
        data_dir=config['data']['data_dir'],
        subject_ids=val_subjects,
        patch_size=config['data']['patch_size'],
        overlap=config['data']['overlap'],
        augment=False,
        test_mode=test_run
    )
    
    print(f"Training patches: {len(train_dataset)}")
    print(f"Validation patches: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    # Initialize model
    print("\nInitializing model...")
    model = UNet3D(
        in_channels=config['model']['in_channels'],
        num_classes=config['model']['num_classes'],
        use_transformer=config['model']['use_transformer']
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['optimization']['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs']
    )
    
    # Initialize loss function
    criterion = CombinedLoss(
        dice_weight=config['training']['dice_weight'],
        ce_weight=config['training']['ce_weight']
    )
    
    # Initialize mixed precision scaler
    scaler = GradScaler()
    
    # Check for pretrained or resume checkpoint
    start_epoch = 0
    checkpoint_dir = config['checkpoint']['save_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if resume:
        print("\nChecking for checkpoint to resume...")
        pretrained_path = check_for_pretrained(
            checkpoint_dir,
            config['aws'].get('s3_bucket'),
            config['aws'].get('s3_checkpoint_path')
        )
        
        if pretrained_path:
            print("Pretrained model detected. Loading weights...")
            start_epoch = load_checkpoint(pretrained_path, model, optimizer, scheduler)
            print(f"Resuming from epoch {start_epoch + 1}")
        else:
            print("No checkpoint found. Starting from scratch.")
    else:
        # Check for pretrained weights (not resume)
        pretrained_path = check_for_pretrained(
            checkpoint_dir,
            config['aws'].get('s3_bucket'),
            config['aws'].get('s3_checkpoint_path')
        )
        
        if pretrained_path:
            print("Pretrained model detected. Loading weights...")
            load_checkpoint(pretrained_path, model)
            print("Loaded pretrained weights (optimizer state not loaded)")
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, config['training']['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")
        print("-" * 60)
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler, criterion,
            config['training']['accumulation_steps'], device, epoch + 1
        )
        
        # Validate
        val_loss, dice_scores = validate_epoch(
            model, val_loader, criterion, device, epoch + 1
        )
        
        # Update learning rate
        scheduler.step()
        
        # Print metrics
        print(f"\nEpoch {epoch + 1} Results:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Dice Scores - WT: {dice_scores['WT']:.4f}, "
              f"TC: {dice_scores['TC']:.4f}, ET: {dice_scores['ET']:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % config['checkpoint']['save_frequency'] == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"checkpoint_epoch_{epoch + 1}.pth"
            )
            
            save_checkpoint(
                model, optimizer, scheduler, epoch + 1,
                train_loss, val_loss, dice_scores, config,
                {'train': train_subjects, 'val': val_subjects},
                checkpoint_path
            )
            
            # Upload to S3 if configured
            if config['aws'].get('s3_bucket'):
                try:
                    s3_path = os.path.join(
                        config['aws']['s3_checkpoint_path'],
                        f"checkpoint_epoch_{epoch + 1}.pth"
                    )
                    
                    # Set up AWS credentials from .env
                    os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('ACCESS_KEY_ID', '')
                    os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('SECRET_ACCESS_KEY', '')
                    os.environ['AWS_DEFAULT_REGION'] = os.getenv('REGION', 'us-east-1')
                    
                    upload_to_s3(
                        checkpoint_path,
                        config['aws']['s3_bucket'],
                        s3_path
                    )
                except Exception as e:
                    print(f"Warning: Failed to upload to S3: {e}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(checkpoint_dir, "best_model.pth")
                save_checkpoint(
                    model, optimizer, scheduler, epoch + 1,
                    train_loss, val_loss, dice_scores, config,
                    {'train': train_subjects, 'val': val_subjects},
                    best_path
                )
                print(f"  New best model saved! (Val Loss: {val_loss:.4f})")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    if test_run:
        print("\n[TEST MODE] Test run completed successfully!")
        print("All core functionality validated:")
        print("  - Data loading")
        print("  - Forward pass")
        print("  - Loss computation")
        print("  - Backpropagation")
        print("  - Checkpoint saving")
        print("\nYou can now run full training with:")
        print(f"  python src/train.py --config {config_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BraTS 2020 segmentation model")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to configuration YAML file")
    parser.add_argument("--test_run", action="store_true",
                       help="Run in test mode (1 subject, 1 epoch)")
    parser.add_argument("--resume", action="store_true",
                       help="Resume training from latest checkpoint")
    
    args = parser.parse_args()
    
    main(args.config, args.test_run, args.resume)
