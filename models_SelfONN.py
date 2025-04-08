import torch
import torch.nn as nn
import torch.nn.functional as F
from fastonn import SelfONN1d as SelfONN1dlayer

class DownsampleConvBlock(nn.Module):
    """Convolutional block with optional downsampling."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, downsample=False, dropout=False, q=3):
        super().__init__()
        self.dropout = dropout
        self.conv = SelfONN1dlayer(in_channels, out_channels, kernel_size, stride, padding, q=q)
        self.relu = nn.Tanh()
        self.norm = nn.InstanceNorm1d(out_channels)
        self.downsample = downsample
        if downsample:
            self.downsample_layer = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout_layer = nn.Dropout(0.25)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)

        if self.dropout:
            out = self.dropout_layer(out)

        if self.downsample:
            out = self.downsample_layer(out)

        return out
    

class UpsampleConvBlock(nn.Module):
    """Block with upsampling followed by a convolutional block."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, upsample=False, dropout=False, q=3):
        super().__init__()
        self.dropout = dropout
        self.conv = SelfONN1dlayer(in_channels, out_channels, kernel_size, stride, padding, q=q)
        self.relu = nn.Tanh()
        self.norm = nn.InstanceNorm1d(out_channels)
        self.upsample = upsample
        if upsample:
            self.upsample_layer = nn.Upsample(scale_factor=2, mode='nearest')
        self.dropout_layer = nn.Dropout(0.25)

    def forward(self, x):
        if self.upsample:
            x = self.upsample_layer(x)

        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)

        if self.dropout:
            out = self.dropout_layer(out)

        return out
    

##########################################################################################################
# Generator and Discriminator with simple downsample and upsample blocks.
##########################################################################################################
class Generator(nn.Module):
    def __init__(self,q):
        super().__init__()

        # Downsampling
        self.time_down1 = DownsampleConvBlock(2, 16, downsample=True, q=q)
        self.time_down2 = DownsampleConvBlock(16, 32, downsample=True, q=q)
        self.time_down3 = DownsampleConvBlock(32, 64, downsample=True, dropout=True, q=q)
        self.time_down4 = DownsampleConvBlock(64, 64, downsample=True, dropout=True, q=q)
        self.time_down5 = DownsampleConvBlock(64, 64, downsample=True, dropout=True, q=q)
        
        # Upsampling 
        self.time_up1 = UpsampleConvBlock(64, 64, upsample=True, dropout=True, q=q)
        self.time_up2 = UpsampleConvBlock(64, 64, upsample=True, dropout=True, q=q)
        self.time_up3 = UpsampleConvBlock(64, 32, upsample=True, q=q)
        self.time_up4 = UpsampleConvBlock(32, 16, upsample=True, q=q)
        self.time_up5 = UpsampleConvBlock(16, 16, upsample=True, q=q)

        # Output Branch
        self.time_output_conv = nn.Conv1d(16, 2, kernel_size=3, stride=1, padding=1)

    def forward(self, time_input):
        # Encoder
        time_features_d1 = self.time_down1(time_input) # 16 x 512
        time_features_d2 = self.time_down2(time_features_d1) # 32 x 256       
        time_features_d3 = self.time_down3(time_features_d2) # 64 x 128
        time_features_d4 = self.time_down4(time_features_d3) # 64 x 64
        time_features_d5 = self.time_down5(time_features_d4) # 64 x 32 # Bottleneck

        # Decoder
        time_features_u1 = self.time_up1(time_features_d5) # 64 x 64
        time_features_u2 = self.time_up2(time_features_u1) # 64 x 128
        time_features_u3 = self.time_up3(time_features_u2) # 32 x 256
        time_features_u4 = self.time_up4(time_features_u3) # 16 x 512
        time_features_u5 = self.time_up5(time_features_u4) # 16 x 1024

        # output
        time_output = self.time_output_conv(time_features_u5) # 2 x 1024

        return time_output
    

class Discriminator(nn.Module):
    def __init__(self,q):
        super().__init__()
        # Time domain branch with downsampling
        self.time_down1 = DownsampleConvBlock(2, 16, downsample=True, q=q)
        self.time_down2 = DownsampleConvBlock(16, 32, downsample=True, q=q)
        self.time_down3 = DownsampleConvBlock(32, 32, downsample=True, q=q)
        self.time_down4 = DownsampleConvBlock(32, 64, downsample=True, dropout=True, q=q)
        self.time_down5 = DownsampleConvBlock(64, 64, downsample=True, dropout=True, q=q)
        self.time_down6 = DownsampleConvBlock(64, 16, downsample=True, q=q)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(16, 1),
            #nn.Sigmoid()
        )
        
    def forward(self, time_input):
        # Process time-domain input
        time_features = self.time_down1(time_input) # 16 x 512
        time_features = self.time_down2(time_features) # 32 x 256
        time_features = self.time_down3(time_features) # 32 x 128
        time_features = self.time_down4(time_features) # 64 x 64
        time_features = self.time_down5(time_features) # 64 x 32
        time_features = self.time_down6(time_features) # 64 x 16
        
        # Classify combined features as real or fake
        classification = self.classifier(time_features)

        return classification
    

##########################################################################################################
##########################################################################################################

if __name__ == '__main__':
    # Example inputs
    time_input = torch.randn(16, 2, 1024)  # (Batch size, Channels, Length)
    freq_input = torch.randn(16, 7, 256)   # (Batch size, Channels, Length)

    # Initialize the model
    G = Generator(q=3)
    D = Discriminator(q=3)

    # Forward pass
    time_output = G(time_input)
    output = D(time_input)

    print("_" * 32)
    print("Time Domain Models.")
    print(f"Time output shape: {time_output.shape}")  # Expected: [16, 2, 1024]
    print(f"Discriminator output shape: {output.shape}")  # Expected: [16, 1]
    print("_" * 32)

    # Calculate the total number of parameters
    total_params = sum(p.numel() for p in G.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params}")

    total_params_k = total_params / 1000
    print(f"Total number of parameters (K): {total_params_k:.2f}K")

