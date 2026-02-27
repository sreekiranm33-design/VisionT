import torch
import torch.nn as nn
from timm.layers import DropPath, LayerScale

import config as cfg


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x):
        batch_size, num_tokens, channels = x.shape
        qkv = (
            self.qkv(x)
            .reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(batch_size, num_tokens, channels)
        out = self.proj(out)
        return out


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, drop_path, layer_scale_init):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim=dim, num_heads=num_heads)
        self.ls1 = LayerScale(dim, init_values=layer_scale_init)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim=dim, mlp_ratio=mlp_ratio)
        self.ls2 = LayerScale(dim, init_values=layer_scale_init)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class SteelViT(nn.Module):
    def __init__(
        self,
        num_classes=cfg.NUM_CLASSES,
        in_channels=cfg.IN_CHANNELS,
        embed_dim=cfg.EMBED_DIM,
        num_heads=cfg.NUM_HEADS,
        depth=cfg.DEPTH,
        mlp_ratio=cfg.MLP_RATIO,
        patch_size=cfg.PATCH_SIZE,
        drop_path_rate=cfg.DROP_PATH_RATE,
        layer_scale_init=cfg.LAYER_SCALE_INIT,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )

        self.patch_embed = nn.Conv2d(
            128, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True
        )

        tokens_per_side = 16 // patch_size
        num_patches = tokens_per_side * tokens_per_side
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        drop_path_rates = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path_rates[i],
                    layer_scale_init=layer_scale_init,
                )
                for i in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        cls_repr = self.norm(x[:, 0])
        logits = self.head(cls_repr)
        return logits
