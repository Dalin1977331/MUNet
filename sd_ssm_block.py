import torch
import torch.nn as nn
import torch.nn.functional as F

class SCConv(nn.Module):
    """
    Spatial and Channel Reconstruction Convolution (SCConv)
    - SRU: Spatial Reconstruction Unit
    - CRU: Channel Reconstruction Unit
    """
    def __init__(self, in_channels, out_channels):
        super(SCConv, self).__init__()
        # Spatial Reconstruction Unit (SRU)
        self.spatial_reconstruction = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        # Channel Reconstruction Unit (CRU)
        self.channel_reconstruction = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )

    def forward(self, x):
        x = self.spatial_reconstruction(x)  # Reduce spatial redundancy
        x = self.channel_reconstruction(x)  # Reduce channel redundancy
        return x


class DWConv(nn.Module):
    """
    Depthwise Separable Convolution (DWConv)
    - Efficient feature compression
    """
    def __init__(self, in_channels, out_channels):
        super(DWConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class SDConv(nn.Module):
    """
    SD-Conv: Combination of SCConv and DWConv with dilated convolutions
    """
    def __init__(self, in_channels, out_channels, dilation=1):
        super(SDConv, self).__init__()
        self.scconv = SCConv(in_channels, out_channels)
        self.dwconv = DWConv(out_channels, out_channels)
        self.dilated_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)

    def forward(self, x):
        x = self.scconv(x)
        x = self.dwconv(x)
        x = self.dilated_conv(x)
        return x


class SDSSMBlock(nn.Module):
    """
    SD-SSM Block: Combines two branches (X1 and X2) for local and global feature extraction
    """
    def __init__(self, in_channels, out_channels):
        super(SDSSMBlock, self).__init__()
        
        # X1 Branch: Local feature extraction with multi-scale SD-Conv
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.sdconv1 = SDConv(in_channels, out_channels, dilation=1)
        self.sdconv2 = SDConv(out_channels, out_channels, dilation=2)
        self.sdconv3 = SDConv(out_channels, out_channels, dilation=3)

        # X2 Branch: Global feature extraction with LayerNorm, Linear, SD-Conv, and SS2D
        self.layer_norm = nn.LayerNorm([in_channels, 1, 1])  # Normalize along the channel axis
        self.linear1 = nn.Linear(in_channels, in_channels)
        self.silu = nn.SiLU()
        self.sdconv4 = SDConv(in_channels, out_channels, dilation=2)
        self.ss2d = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        # Fusion: Combine features from both branches
        self.fusion_linear = nn.Linear(out_channels, out_channels)
        self.final_layer_norm = nn.LayerNorm([out_channels, 1, 1])
        self.output_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Split input into two branches
        x1, x2 = torch.chunk(x, 2, dim=1)  # Split along the channel dimension

        # X1 Branch
        x1 = self.bn1(x1)
        x1 = self.sdconv1(x1)
        x1 = self.sdconv2(x1)
        x1 = self.sdconv3(x1)

        # X2 Branch
        x2 = self.layer_norm(x2)
        x2 = self.linear1(x2.permute(0, 2, 3, 1))  # Adjust shape for Linear layer
        x2 = self.silu(x2).permute(0, 3, 1, 2)  # Restore shape
        x2 = self.sdconv4(x2)
        x2 = self.ss2d(x2)

        # Fusion
        fused = x1 + x2  # Add features from both branches
        fused = self.fusion_linear(fused.permute(0, 2, 3, 1))  # Adjust shape for Linear layer
        fused = self.final_layer_norm(fused).permute(0, 3, 1, 2)  # Restore shape
        output = self.output_conv(fused)

        # Residual connection
        output += x  # Add input as residual
        return output
