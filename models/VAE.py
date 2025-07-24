from typing import List

from torch import nn
import torch

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False, upsample=False):
        super(ResidualBlock, self).__init__()
        self.downsample = downsample
        self.upsample = upsample

        stride = 2 if downsample or upsample else 1

        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        elif upsample:
            self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1) if not upsample else \
                     nn.ConvTranspose2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = nn.Sequential()
        if in_channels != out_channels or downsample or upsample:
            if upsample:
                self.skip = nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
                    nn.BatchNorm2d(out_channels)
                )
            elif downsample:
                self.skip = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, stride=2),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.skip = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class Encoder(nn.Module):
    def __init__(self, channels, latent_dim):
        super(Encoder, self).__init__()
        self.blocks = nn.Sequential(*[
            ResidualBlock(channels[i], channels[i+1], downsample=True)
            for i in range(len(channels) - 1)
        ])
        self.flatten = nn.Flatten()
        self.feature_dim = channels[-1] * 12 * 16
        self.fc_mu = nn.Linear(self.feature_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.feature_dim, latent_dim)

    def forward(self, x):
        x = self.blocks(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, channels, latent_dim):
        super(Decoder, self).__init__()
        self.feature_dim = channels[0] * 12 * 16
        self.fc = nn.Linear(latent_dim, self.feature_dim)
        self.reshape_channels = channels[0]
        self.blocks = nn.Sequential(*[
            ResidualBlock(channels[i], channels[i+1], upsample=True)
            for i in range(len(channels) - 2)
        ])
        self.final = nn.ConvTranspose2d(channels[-2], channels[-1], 4, stride=2, padding=1)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, self.reshape_channels, 12, 16)
        x = self.blocks(x)
        x = self.final(x)
        return x

class VAE(nn.Module):
    def __init__(self, enc_channels: List[int], dec_channels: List[int], latent_dim: int):
        super(VAE, self).__init__()
        self.encoder = Encoder(channels=enc_channels, latent_dim=latent_dim)
        self.decoder = Decoder(channels=dec_channels, latent_dim=latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        # logvar = torch.clamp(logvar, -10, 10)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar
    
if __name__ == "__main__":
    model = VAE()
    x = torch.randn(1, 1, 192, 256)  # Example input
    recon, mu, logvar = model(x)
    print("Reconstructed shape:", recon.shape)
    print("Mu shape:", mu.shape)
    print("Logvar shape:", logvar.shape)