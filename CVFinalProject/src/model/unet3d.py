"""
3D U-Net architecture for brain tumor segmentation.
"""

import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    """3D convolution block with BatchNorm and ReLU."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super(ConvBlock3D, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        return x


class DownBlock3D(nn.Module):
    """Encoder block with convolution and downsampling."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super(DownBlock3D, self).__init__()
        
        self.conv_block = ConvBlock3D(in_channels, out_channels)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
    
    def forward(self, x):
        skip = self.conv_block(x)
        x = self.pool(skip)
        return x, skip


class UpBlock3D(nn.Module):
    """Decoder block with upsampling and skip connections."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super(UpBlock3D, self).__init__()
        
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = ConvBlock3D(in_channels + out_channels, out_channels)
    
    def forward(self, x, skip):
        x = self.up(x)
        
        # Concatenate skip connection
        x = torch.cat([x, skip], dim=1)
        
        x = self.conv_block(x)
        return x


class UNet3D(nn.Module):
    """
    3D U-Net with 4 encoder-decoder levels and optional Contextual Transformer.
    
    Architecture:
    - Encoder: [32, 64, 128, 256] channels
    - Bottleneck: 512 channels (with optional CT Block)
    - Decoder: [256, 128, 64, 32] channels
    - Output: 3 channels (WT, TC, ET)
    """
    
    def __init__(self, in_channels: int = 4, num_classes: int = 3, use_transformer: bool = True):
        super(UNet3D, self).__init__()
        
        self.use_transformer = use_transformer
        
        # Initial convolution
        self.init_conv = ConvBlock3D(in_channels, 32)
        
        # Encoder path
        self.down1 = DownBlock3D(32, 64)
        self.down2 = DownBlock3D(64, 128)
        self.down3 = DownBlock3D(128, 256)
        self.down4 = DownBlock3D(256, 512)
        
        # Bottleneck
        self.bottleneck = ConvBlock3D(512, 512)
        
        # Contextual Transformer (optional)
        if use_transformer:
            from src.model.transformer import ContextualTransformerBlock
            self.ct_block = ContextualTransformerBlock(channels=512, reduced_channels=256, num_heads=8)
        
        # Decoder path
        self.up1 = UpBlock3D(512, 256)
        self.up2 = UpBlock3D(256, 128)
        self.up3 = UpBlock3D(128, 64)
        self.up4 = UpBlock3D(64, 32)
        
        # Output layer
        self.output = nn.Conv3d(32, num_classes, kernel_size=1)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: (B, 4, D, H, W) input volume
            
        Returns:
            (B, 3, D, H, W) segmentation logits
        """
        # Initial convolution
        x = self.init_conv(x)
        
        # Encoder
        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        x, skip3 = self.down3(x)
        x, skip4 = self.down4(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Apply Contextual Transformer if enabled
        if self.use_transformer:
            x = self.ct_block(x)
        
        # Decoder
        x = self.up1(x, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)
        
        # Output
        x = self.output(x)
        
        return x
