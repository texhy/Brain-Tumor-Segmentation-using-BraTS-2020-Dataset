import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.transformer import ContextualTransformerBlock
from src.model.unet3d import UNet3D

def test_cot_stability():
    print("Testing CoT Block Stability in FP16...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = ContextualTransformerBlock(channels=512, reduced_channels=256).to(device)
    model.train()
    
    # Create random input
    x = torch.randn(2, 512, 16, 16, 16).to(device)
    
    # Forward pass in AMP
    with autocast():
        output = model(x)
        
    if torch.isnan(output).any():
        print("❌ CoT Block produced NaNs!")
        return False
    else:
        print("✅ CoT Block stable (No NaNs)")
        return True

def test_unet_stability():
    print("\nTesting UNet3D Stability in FP16...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = UNet3D(in_channels=4, num_classes=3, use_transformer=True).to(device)
    model.train()
    
    # Create random input
    x = torch.randn(2, 4, 32, 32, 32).to(device)
    
    # Forward pass in AMP
    with autocast():
        output = model(x)
        
    if torch.isnan(output).any():
        print("❌ UNet3D produced NaNs!")
        return False
    else:
        print("✅ UNet3D stable (No NaNs)")
        return True

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available, testing on CPU (AMP might be ignored)")
    
    cot_stable = test_cot_stability()
    unet_stable = test_unet_stability()
    
    if cot_stable and unet_stable:
        print("\n🎉 All stability tests passed!")
        sys.exit(0)
    else:
        print("\n💥 Stability tests failed!")
        sys.exit(1)
