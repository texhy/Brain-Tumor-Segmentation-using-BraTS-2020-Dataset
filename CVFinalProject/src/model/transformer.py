"""
Contextual Transformer block for 3D feature enhancement.
Based on "Brain Tumor Segmentation in MRI Images with 3D U-Net and Contextual Transformer"
"""

import torch
import torch.nn as nn


class ContextualTransformerBlock3D(nn.Module):
    """
    3D Contextual Transformer (CoT) Block as described in the paper.
    
    This block integrates static and dynamic contextual information:
    1. Static Context (K¹): Derived from k×k×k convolution on Keys
    2. Dynamic Context (K²): Derived from attention mechanism (V * A)
    3. Final output merges both static and dynamic contexts
    
    Pipeline:
    1. Input → K, V, Q via 1×1×1 convolutions
    2. K → Static Context (K¹) via k×k×k convolution
    3. [K¹, Q] → Attention Matrix (A) via two 1×1×1 convolutions
    4. V * A → Dynamic Context (K²)
    5. Merge K¹ and K² → Output
    """
    
    def __init__(self, channels: int = 512, kernel_size: int = 3):
        """
        Initialize CoT Block.
        
        Args:
            channels: Number of input/output channels
            kernel_size: Kernel size for static context convolution (k in paper)
        """
        super(ContextualTransformerBlock3D, self).__init__()
        
        self.channels = channels
        self.kernel_size = kernel_size
        
        # 1×1×1 convolutions to derive K, V, Q from input
        self.key_conv = nn.Conv3d(channels, channels, kernel_size=1)
        self.value_conv = nn.Conv3d(channels, channels, kernel_size=1)
        self.query_conv = nn.Conv3d(channels, channels, kernel_size=1)
        
        # k×k×k convolution for static context (K¹)
        # Paper uses this to capture local contextual information
        padding = kernel_size // 2
        self.static_context_conv = nn.Conv3d(
            channels, channels, 
            kernel_size=kernel_size, 
            padding=padding,
            groups=1  # Standard convolution
        )
        
        # Two 1×1×1 convolutions for attention matrix
        # Input: concatenated [K¹, Q] with 2*channels
        # Output: attention weights with same spatial dims
        self.attention_conv1 = nn.Conv3d(2 * channels, channels, kernel_size=1)
        self.attention_conv2 = nn.Conv3d(channels, channels, kernel_size=1)
        
        # Final 1×1×1 convolution to merge static and dynamic contexts
        self.output_conv = nn.Conv3d(2 * channels, channels, kernel_size=1)
        
        
        # GroupNorm for stability (works with any batch size)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        
    def forward(self, x):
        """
        Forward pass through CoT block.
        
        Args:
            x: (B, C, D, H, W) input features
            
        Returns:
            (B, C, D, H, W) output features with contextual information
        """
        B, C, D, H, W = x.shape
        identity = x
        
        # Step 1: Derive K, V, Q from input
        K = self.key_conv(x)      # (B, C, D, H, W)
        V = self.value_conv(x)    # (B, C, D, H, W)
        Q = self.query_conv(x)    # (B, C, D, H, W)
        
        # Step 2: Static Context (K¹) via k×k×k convolution on Keys
        K1 = self.static_context_conv(K)  # (B, C, D, H, W)
        
        # Step 3: Attention Matrix (A)
        # Concatenate K¹ and Q along channel dimension
        K1_Q = torch.cat([K1, Q], dim=1)  # (B, 2C, D, H, W)
        
        # Apply two 1×1×1 convolutions to compute attention
        A = self.attention_conv1(K1_Q)    # (B, C, D, H, W)
        A = nn.functional.relu(A)
        A = self.attention_conv2(A)       # (B, C, D, H, W)
        # Scale attention scores to prevent overflow in FP16
        A = A * 0.1
        
        # Use softmax instead of sigmoid for better numerical stability
        # Force float32 for softmax calculation to prevent FP16 overflow
        A_flat = A.view(B, C, -1).float()
        A = torch.softmax(A_flat, dim=-1).view(B, C, D, H, W).type_as(x)
        
        # Step 4: Dynamic Context (K²) via element-wise multiplication
        K2 = V * A  # (B, C, D, H, W)
        
        # Step 5: Merge static (K¹) and dynamic (K²) contexts
        merged = torch.cat([K1, K2], dim=1)  # (B, 2C, D, H, W)
        output = self.output_conv(merged)     # (B, C, D, H, W)
        output = self.norm(output)
        
        # Add small epsilon to prevent NaN
        output = torch.clamp(output, min=-10, max=10)
        
        # Residual connection
        output = output + identity
        
        return output


class ContextualTransformerBlock(nn.Module):
    """
    Wrapper for CoT block with channel reduction/expansion as in original code.
    
    This maintains compatibility with the existing UNet3D architecture
    while using the correct CoT implementation internally.
    """
    
    def __init__(self, channels: int = 512, reduced_channels: int = 256, num_heads: int = 8):
        """
        Initialize wrapper CoT block.
        
        Args:
            channels: Input/output channels (512 for bottleneck)
            reduced_channels: Channels for CoT processing (256)
            num_heads: Not used in paper's CoT, kept for compatibility
        """
        super(ContextualTransformerBlock, self).__init__()
        
        self.channels = channels
        self.reduced_channels = reduced_channels
        
        # Channel reduction
        self.reduce = nn.Conv3d(channels, reduced_channels, kernel_size=1)
        
        # Core CoT block (paper's architecture)
        self.cot_block = ContextualTransformerBlock3D(
            channels=reduced_channels,
            kernel_size=3  # Paper uses 3×3×3 for static context
        )
        
        # Channel expansion
        self.expand = nn.Conv3d(reduced_channels, channels, kernel_size=1)
    
    def forward(self, x):
        """
        Forward pass through CoT wrapper.
        
        Args:
            x: (B, 512, D, H, W) bottleneck features
            
        Returns:
            (B, 512, D, H, W) enhanced features
        """
        identity = x
        
        # Channel reduction
        x = self.reduce(x)  # (B, 256, D, H, W)
        
        # Apply CoT block
        x = self.cot_block(x)  # (B, 256, D, H, W)
        
        # Channel expansion
        x = self.expand(x)  # (B, 512, D, H, W)
        
        # Residual connection
        x = x + identity
        
        return x
