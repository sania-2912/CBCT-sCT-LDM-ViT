import math
import torch
import torch.nn as nn

from models.autoencoder import ResBlock
from models.vit import ViTBottleneck

class ConditionEncoder(nn.Module):
    """Encodes the CBCT conditioning image into a spatial feature map."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(256, 128, 4, 2, 1),
        )

    def forward(self, x):
        return self.encoder(x)


class TimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding (DDPM-style)."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=1)


class ConditionalDiffusionModel(nn.Module):
    """Noise-prediction network: conv down/up path with a ViT bottleneck,
    conditioned on the CBCT image and the diffusion timestep."""

    def __init__(self, latent_channels=4):
        super().__init__()
        self.condition_encoder = ConditionEncoder()

        self.time_embedding = nn.Sequential(
            TimeEmbedding(256),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
        )

        self.input_conv = nn.Conv2d(latent_channels + 128, 128, 3, padding=1)
        self.down1 = ResBlock(128, 256)
        self.downsample = nn.Conv2d(256, 256, 4, 2, 1)
        self.vit = ViTBottleneck(256)
        self.upsample = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.output = nn.Conv2d(128, latent_channels, 3, padding=1)

    def forward(self, z, t, cbct):
        cond = self.condition_encoder(cbct)

        x = torch.cat([z, cond], dim=1)
        x = self.input_conv(x)
        x = self.down1(x)

        t_emb = self.time_embedding(t)
        x = x + t_emb[:, :, None, None]

        skip = x

        x = self.downsample(x)
        x = self.vit(x)
        x = self.upsample(x)

        x = x + skip[:, :128]

        return self.output(x)