# modules.py
import torch
import torch.nn as nn
from config import Config

class ImprovedConvBlock3D(nn.Module):
    """
    Improved 3D Convolutional block with:
    - Instance Normalization (instead of Batch Norm)
    - Leaky ReLU (instead of ReLU)
    
    Based on nnU-Net methodology (Isensee et al., 2021)
    """
    def __init__(self, in_channels, out_channels):
        super(ImprovedConvBlock3D, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)  # Instance Norm
        self.act1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)  # Leaky ReLU
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        self.act2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
    
    def forward(self, x):
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))
        return x


class Improved3DUNet(nn.Module):
    """
    Improved 3D U-Net for medical image segmentation.
    
    Based on:
    - Original 3D U-Net: Çiçek et al. (2016)
    - Improvements from nnU-Net: Isensee et al. (2021)
    
    Key improvements:
    1. Instance Normalization instead of Batch Normalization
    2. Leaky ReLU (slope=0.01) instead of ReLU
    3. Deep supervision (optional)
    
    Args:
        in_channels: Number of input channels (1 for grayscale MRI)
        num_classes: Number of segmentation classes (6 for HipMRI)
        base_features: Number of features in first layer (32)
        deep_supervision: Enable deep supervision outputs
    """
    def __init__(self, 
                 in_channels=1, 
                 num_classes=6, 
                 base_features=32,
                 deep_supervision=True):
        super(Improved3DUNet, self).__init__()
        
        self.deep_supervision = deep_supervision
        
        # Encoder (Contracting Path)
        self.enc1 = ImprovedConvBlock3D(in_channels, base_features)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.enc2 = ImprovedConvBlock3D(base_features, base_features * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.enc3 = ImprovedConvBlock3D(base_features * 2, base_features * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.enc4 = ImprovedConvBlock3D(base_features * 4, base_features * 8)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = ImprovedConvBlock3D(base_features * 8, base_features * 16)
        
        # Decoder (Expanding Path)
        self.upconv4 = nn.ConvTranspose3d(base_features * 16, base_features * 8, 
                                          kernel_size=2, stride=2)
        self.dec4 = ImprovedConvBlock3D(base_features * 16, base_features * 8)
        
        self.upconv3 = nn.ConvTranspose3d(base_features * 8, base_features * 4, 
                                          kernel_size=2, stride=2)
        self.dec3 = ImprovedConvBlock3D(base_features * 8, base_features * 4)
        
        self.upconv2 = nn.ConvTranspose3d(base_features * 4, base_features * 2, 
                                          kernel_size=2, stride=2)
        self.dec2 = ImprovedConvBlock3D(base_features * 4, base_features * 2)
        
        self.upconv1 = nn.ConvTranspose3d(base_features * 2, base_features, 
                                          kernel_size=2, stride=2)
        self.dec1 = ImprovedConvBlock3D(base_features * 2, base_features)
        
        # Main output
        self.out = nn.Conv3d(base_features, num_classes, kernel_size=1)
        
        # Deep supervision outputs (auxiliary outputs from intermediate decoder levels)
        if self.deep_supervision:
            self.deep_out1 = nn.Conv3d(base_features * 2, num_classes, kernel_size=1)
            self.deep_out2 = nn.Conv3d(base_features * 4, num_classes, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)  # 1/1 resolution
        enc2 = self.enc2(self.pool1(enc1))  # 1/2 resolution
        enc3 = self.enc3(self.pool2(enc2))  # 1/4 resolution
        enc4 = self.enc4(self.pool3(enc3))  # 1/8 resolution
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))  # 1/16 resolution
        
        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)  # Skip connection
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        # Main output
        out = self.out(dec1)
        
        # Deep supervision outputs
        if self.deep_supervision and self.training:
            # Upsample auxiliary outputs to match input size
            deep_out1 = self.deep_out1(dec2)
            deep_out1 = nn.functional.interpolate(deep_out1, size=out.shape[2:], 
                                                  mode='trilinear', align_corners=False)
            
            deep_out2 = self.deep_out2(dec3)
            deep_out2 = nn.functional.interpolate(deep_out2, size=out.shape[2:], 
                                                  mode='trilinear', align_corners=False)
            
            return out, deep_out1, deep_out2
        
        return out


def get_model():
    """
    Create and return the Improved 3D U-Net model
    """
    model = Improved3DUNet(
        in_channels=Config.IN_CHANNELS,
        num_classes=Config.NUM_CLASSES,
        base_features=Config.BASE_FEATURES,
        deep_supervision=Config.DEEP_SUPERVISION
    )
    return model


if __name__ == "__main__":
    # Test the model
    from config import Config
    
    print("Testing Improved 3D U-Net...")
    model = get_model()
    
    # Test forward pass
    x = torch.randn(1, 1, 96, 96, 96)  # Batch, Channel, D, H, W
    
    model.train()  # Training mode (with deep supervision)
    if Config.DEEP_SUPERVISION:
        out, deep1, deep2 = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Main output shape: {out.shape}")
        print(f"Deep supervision output 1: {deep1.shape}")
        print(f"Deep supervision output 2: {deep2.shape}")
    else:
        out = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {out.shape}")
    
    # Test inference mode
    model.eval()
    with torch.no_grad():
        out = model(x)
        print(f"\nInference mode output: {out.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB")