import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torchvision import transforms,models

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(out_channels)
        
        self.conv_out = nn.Conv2d(out_channels * 5, out_channels, 1, bias=False)
        self.bn_out = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        size = x.size()[2:]
        
        conv1 = self.relu(self.bn1(self.conv1(x)))
        conv2 = self.relu(self.bn2(self.conv2(x)))
        conv3 = self.relu(self.bn3(self.conv3(x)))
        conv4 = self.relu(self.bn4(self.conv4(x)))
        
        pool = self.pool(x)
        pool = self.relu(self.bn5(self.conv5(pool)))
        pool = nn.functional.interpolate(pool, size=size, mode='bilinear', align_corners=False)
        
        out = torch.cat([conv1, conv2, conv3, conv4, pool], dim=1)
        out = self.relu(self.bn_out(self.conv_out(out)))
        
        return out
        
class ResNetEncoder(nn.Module):
    """
    ResNet-34 encoder for feature extraction at multiple scales.
    Returns features at 4 different resolutions for skip connections.
    """
    def __init__(self, pretrained=True):
        super(ResNetEncoder, self).__init__()
        
        resnet34 = models.resnet34(pretrained=pretrained)
        
        # Extract ResNet stages
        self.stage1 = nn.Sequential(*list(resnet34.children())[:5])  # Output: 64 channels
        self.stage2 = nn.Sequential(*list(resnet34.children())[5:6])  # Output: 128 channels
        self.stage3 = nn.Sequential(*list(resnet34.children())[6:7])  # Output: 256 channels
        self.stage4 = nn.Sequential(*list(resnet34.children())[7:8])  # Output: 512 channels
    
    def forward(self, x):
        """
        Returns features at 4 different scales for skip connections."""
        
        feat_stage1 = self.stage1(x)   # 1/4 resolution, 64 channels
        feat_stage2 = self.stage2(feat_stage1)  # 1/8 resolution, 128 channels
        feat_stage3 = self.stage3(feat_stage2)  # 1/16 resolution, 256 channels
        feat_stage4 = self.stage4(feat_stage3)  # 1/32 resolution, 512 channels
        
        return feat_stage1, feat_stage2, feat_stage3, feat_stage4

class ConvBlock(nn.Module):
    """
    Basic convolutional block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        return x
    
class UpsampleBlock(nn.Module):
    """
    Upsampling block: TransposeConv -> BN -> ReLU -> ConvBlock
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=0, output_padding=0):
        super(UpsampleBlock, self).__init__()
        
        self.transpose_conv = nn.ConvTranspose2d(
            in_channels, out_channels, 
            kernel_size=kernel_size, 
            stride=stride,
            padding=padding, 
            output_padding=output_padding
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv_block = ConvBlock(out_channels, out_channels)
    
    def forward(self, x):
        x = self.transpose_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv_block(x)
        
        return x


class FPNBranch(nn.Module):
    """
    Feature Pyramid Network branch for multi-scale feature fusion.
    """
    def __init__(self, in_channels, upsample_factor):
        super(FPNBranch, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=upsample_factor, mode='bilinear', align_corners=False)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.upsample(x)
        
        return x


class CRowNet(nn.Module):
    """
    CRowNet: Lightweight Crop Row Detection Network
    
    Architecture:
    - Encoder: ResNet-34 backbone
    - ASPP: Multi-scale context aggregation
    - Decoder: Progressive upsampling with skip connections
    - Multi-scale FPN: Feature fusion at multiple resolutions
    
    Args:
        in_channels (int): Number of input channels (default: 3 for RGB)
        out_channels (int): Number of output channels (default: 1 for binary segmentation)
    """
    def __init__(self, in_channels=3, out_channels=1):
        super(CRowNet, self).__init__()
        
        # Encoder (ResNet-34)
        self.encoder = ResNetEncoder(pretrained=True)
        
        # ASPP module for multi-scale context
        self.aspp = ASPP(in_channels=512, out_channels=256)
        
        # Decoder with progressive upsampling
        self.upsample1 = UpsampleBlock(256, 256, padding=1, output_padding=1)  # 256 -> 256 channels
        self.upsample2 = UpsampleBlock(256, 128, padding=1, output_padding=1)  # 256 -> 128 channels
        self.upsample3 = UpsampleBlock(128, 64, padding=1, output_padding=1)   # 128 -> 64 channels
        self.upsample4 = UpsampleBlock(64, 64, padding=1, output_padding=1)    # 64 -> 64 channels
        
        # Multi-scale FPN branches
        self.fpn_branch1 = FPNBranch(in_channels=256, upsample_factor=8)  # From stage3
        self.fpn_branch2 = FPNBranch(in_channels=128, upsample_factor=4)  # From stage2
        self.fpn_branch3 = FPNBranch(in_channels=64, upsample_factor=2)   # From stage1
        self.fpn_branch4 = FPNBranch(in_channels=64, upsample_factor=1)   # From decoder
        
        # Final upsampling and prediction
        self.final_upsample = UpsampleBlock(256, 64, kernel_size=2, stride=2)  # Fused features -> 64 channels
        self.final_conv = ConvBlock(64, 32)
        self.output_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass through CRowNet.
        
        Args:
            x (torch.Tensor): Input image tensor [B, C, H, W]
        
        Returns:
            torch.Tensor: Segmentation prediction [B, out_channels, H, W]
        """
        # Encoder: Extract multi-scale features
        feat_stage1, feat_stage2, feat_stage3, feat_stage4 = self.encoder(x)
        
        # Apply ASPP to deepest features
        feat_aspp = self.aspp(feat_stage4)
        
        # Decoder: Progressive upsampling with skip connections
        decoder_feat = self.upsample1(feat_aspp)
        decoder_feat = decoder_feat + feat_stage3  # Skip connection from stage3
        feat_stage3_skip = decoder_feat
        
        decoder_feat = self.upsample2(decoder_feat)
        decoder_feat = decoder_feat + feat_stage2  # Skip connection from stage2
        feat_stage2_skip = decoder_feat
        
        decoder_feat = self.upsample3(decoder_feat)
        decoder_feat = decoder_feat + feat_stage1  # Skip connection from stage1
        feat_stage1_skip = decoder_feat
        
        decoder_feat = self.upsample4(decoder_feat)
        
        # Multi-scale FPN feature fusion
        fpn_feat1 = self.fpn_branch1(feat_stage3_skip)  # From stage3
        fpn_feat2 = self.fpn_branch2(feat_stage2_skip)  # From stage2
        fpn_feat3 = self.fpn_branch3(feat_stage1_skip)  # From stage1
        fpn_feat4 = self.fpn_branch4(decoder_feat)      # From decoder
        
        # Concatenate multi-scale features
        fused_features = torch.cat([fpn_feat4, fpn_feat3, fpn_feat2, fpn_feat1], dim=1)  # 256 channels total
        
        # Final prediction
        output = self.final_upsample(fused_features)
        output = self.final_conv(output)
        output = self.output_conv(output)
        
        return output
 
