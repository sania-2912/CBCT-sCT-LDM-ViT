import torch.nn as nn
from einops import rearrange


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_dim=512):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x)
        x = residual + attn_out

        residual = x
        x = self.norm2(x)
        x = residual + self.mlp(x)

        return x


class ViTBottleneck(nn.Module):
    """Stack of transformer blocks applied over a flattened spatial feature map."""

    def __init__(self, channels=256, depth=4, heads=8, mlp_dim=512):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(channels, heads, mlp_dim) for _ in range(depth)
        ])

    def forward(self, x):
        B, C, H, W = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")

        for block in self.blocks:
            x = block(x)

        return rearrange(x, "b (h w) c -> b c h w", h=H, w=W)