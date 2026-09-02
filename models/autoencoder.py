import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = F.silu(x)

        x = self.conv2(x)
        x = self.norm2(x)

        return F.silu(x + residual)


class Encoder(nn.Module):
    def __init__(self, latent_channels=4):
        super().__init__()
        self.model = nn.Sequential(
            ResBlock(1, 64),
            nn.Conv2d(64, 128, 4, 2, 1),
            ResBlock(128, 128),
            nn.Conv2d(128, 256, 4, 2, 1),
            ResBlock(256, 256),
            nn.Conv2d(256, latent_channels, 4, 2, 1),
        )

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, latent_channels=4):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 256, 4, 2, 1),
            ResBlock(256, 256),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            ResBlock(128, 128),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            ResBlock(64, 64),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.model(z)


class Autoencoder(nn.Module):
    def __init__(self, latent_channels=4):
        super().__init__()
        self.encoder = Encoder(latent_channels)
        self.decoder = Decoder(latent_channels)

    def forward(self, x):
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction, z


def autoencoder_loss(reconstruction, target):
    return F.l1_loss(reconstruction, target)