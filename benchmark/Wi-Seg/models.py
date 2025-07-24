import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + residual)
    
class EncoderWithSkips(nn.Module):
    def __init__(self, in_channels=512):
        super(EncoderWithSkips, self).__init__()
        self.skips = []

        layers = []
        self.encoder_layers = nn.ModuleList()
        curr_channels = in_channels
        for out_channels in [512, 256, 256, 128, 64, 32]:
            layers = nn.Sequential(
                nn.ConvTranspose2d(curr_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.ReLU()
            )
            self.encoder_layers.append(layers)
            curr_channels = out_channels
        layers = nn.Sequential(
            nn.ConvTranspose2d(curr_channels, 3, kernel_size=[3, 4], stride=[1, 2], padding=1),
            nn.ReLU()
        )
        self.encoder_layers.append(layers)
        curr_channels = 3

        self.conv1 = nn.Conv2d(curr_channels, curr_channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(curr_channels, curr_channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.resblock = ResBlock(curr_channels)

    def forward(self, x):
        skips = []
        for layer in self.encoder_layers:
            x = layer(x)
            skips.append(x)  # Save skip after each transposed conv

        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.resblock(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class UNetDecoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, channel_list=[32, 64, 128, 256, 512]):
        super().__init__()
        self.depth = len(channel_list)

        # Down path
        self.down_convs = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        prev_ch = in_channels
        for ch in channel_list:
            self.down_convs.append(DoubleConv(prev_ch, ch))
            prev_ch = ch

        # Up path
        self.up_convs = nn.ModuleList()
        self.up_trans = nn.ModuleList()
        for ch in reversed(channel_list[:-1]):
            self.up_trans.append(nn.ConvTranspose2d(prev_ch, ch, kernel_size=2, stride=2))
            self.up_convs.append(DoubleConv(prev_ch, ch))
            prev_ch = ch

        self.final = nn.Sequential(
            nn.Conv2d(channel_list[0], out_channels, kernel_size=1),
            # nn.Sigmoid()  # Assuming binary segmentation
        )

    def forward(self, x):
        encoder_outs = []

        # Encoder
        for down in self.down_convs:
            x = down(x)
            encoder_outs.append(x)
            x = self.pool(x)

        x = encoder_outs.pop()  # deepest feature map

        # Decoder
        for up_trans, up_conv in zip(self.up_trans, self.up_convs):
            skip = encoder_outs.pop()
            x = up_trans(x)

            # Padding if necessary (to match skip connection size)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

            x = torch.cat([skip, x], dim=1)
            x = up_conv(x)

        return self.final(x)


class WiSegUNet(nn.Module):
    def __init__(self, in_channels=103275, reduced_channels=512):
        super(WiSegUNet, self).__init__()
        self.reducer = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.encoder = EncoderWithSkips(in_channels=reduced_channels)
        self.decoder = UNetDecoder()

    def forward(self, x):
        x = self.reducer(x)             # [B, 512, 3, 2]
        x = self.encoder(x)      # x: encoded features, skips: for U-Net
        x = self.decoder(x)
        return x

if __name__ == "__main__":
    # Example usage
    model = WiSegUNet()
    dummy_input = torch.randn(1, 103275, 3, 2)
    output = model(dummy_input)
    print("output:", output.shape)  # [1, 1, 192, 256]