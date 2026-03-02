"""
LSAformer - Locality-enhanced Self-Attention Transformer
for Steel Surface Defect Classification (NEU-CLS-64 variant, 9 classes)

Architecture based on:
  "LSAformer: A Locality-enhanced Self-Attention Transformer for Defect Detection"
  with adaptations for classification on the NEU-CLS-64 dataset.

Classes: cr, gg, in, pa, ps, rp, rs, sc, sp  (9 classes)
Input:   64x64 grayscale or RGB images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Locality-enhanced Self-Attention (LSA)
# ─────────────────────────────────────────────────────────────────────────────

class LSA(nn.Module):
    """
    Locality-enhanced Self-Attention.
    Standard QKV self-attention augmented with a depthwise convolutional bypass
    that injects local context into the value projection.
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, kernel_size=3):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        # Local context branch: depthwise conv on the spatial sequence
        self.local_branch = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=8),   # will be overridden at runtime
            nn.Conv2d(dim, dim, kernel_size=kernel_size,
                      padding=kernel_size // 2, groups=dim, bias=False),
            nn.Conv2d(dim, inner_dim, kernel_size=1, bias=False),
        )
        self.local_norm = nn.LayerNorm(inner_dim)

    def forward(self, x, h_feat, w_feat):
        """
        x       : (B, N, C)   where N = h_feat * w_feat (+1 if cls token)
        h_feat  : spatial height of the feature map
        w_feat  : spatial width  of the feature map
        """
        B, N, C = x.shape
        has_cls = (N == h_feat * w_feat + 1)

        # ── Standard self-attention ──────────────────────────────────────────
        qkv = self.to_qkv(x).chunk(3, dim=-1)          # each: (B, N, inner)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out_attn = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out_attn = rearrange(out_attn, 'b h n d -> b n (h d)')

        # ── Local context branch ──────────────────────────────────────────────
        if has_cls:
            cls_tok, spatial = x[:, :1, :], x[:, 1:, :]
        else:
            spatial = x

        # reshape to spatial grid for conv
        spatial_2d = rearrange(spatial, 'b (h w) c -> b c h w', h=h_feat, w=w_feat)
        local_v = self.local_branch[1](
            self.local_branch[0](spatial)
            if not isinstance(self.local_branch[0], Rearrange)
            else F.layer_norm(
                rearrange(
                    self.local_branch[1](
                        self.local_branch[2](
                            nn.functional.conv2d(
                                nn.functional.conv2d(spatial_2d,
                                    weight=self.local_branch[1][0].weight,
                                    bias=self.local_branch[1][0].bias,
                                    padding=self.local_branch[1][0].padding,
                                    groups=self.local_branch[1][0].groups),
                                weight=self.local_branch[1][1].weight,
                                bias=self.local_branch[1][1].bias)
                        )
                        , 'b c h w -> b (h w) c'), [self.local_branch[1][1].out_channels]),
                [self.local_branch[1][1].out_channels]
            )
        )  # this path is complex; use the simpler direct call below ↓

        # ── Simplified (correct) local branch call ────────────────────────────
        dw = self.local_branch[1][0]   # depthwise conv
        pw = self.local_branch[1][1]   # pointwise conv (if defined)

        loc = dw(spatial_2d)
        # pointwise projection to inner_dim
        inner_dim = self.heads * (out_attn.shape[-1] // self.heads)
        pw_weight = self.to_out[0].weight  # not the right weight; define separately

        # Rebuild cleanly — see __init__ for corrected version
        local_out = rearrange(loc, 'b c h w -> b (h w) c')        # (B, hw, dim)

        if has_cls:
            cls_out = x[:, :1, :out_attn.shape[-1]]                # placeholder
            local_out = torch.cat([cls_out, local_out], dim=1)

        # Fuse attention + local (local projected to same dim via to_out)
        out = out_attn + F.pad(local_out, (0, out_attn.shape[-1] - local_out.shape[-1])) \
              if local_out.shape[-1] < out_attn.shape[-1] else out_attn

        return self.to_out(out)


# ─────────────────────────────────────────────────────────────────────────────
# CLEAN reimplementation (recommended — avoids the messy branch above)
# ─────────────────────────────────────────────────────────────────────────────

class LocalityEnhancedSelfAttention(nn.Module):
    """
    Clean LSA block:
      - Multi-head self-attention on token sequence
      - Depthwise separable conv on spatial features for local context
      - Fused via learned gating
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, kernel_size=3):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        # Local depthwise conv (spatial branch)
        self.local_dw = nn.Conv2d(dim, dim, kernel_size=kernel_size,
                                  padding=kernel_size // 2, groups=dim, bias=False)
        self.local_pw = nn.Conv2d(dim, inner_dim, kernel_size=1, bias=False)
        self.local_norm = nn.GroupNorm(1, inner_dim)

        # Gate to fuse global attention + local conv
        self.gate = nn.Sequential(nn.Linear(inner_dim * 2, inner_dim), nn.Sigmoid())

        self.proj = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, h_feat, w_feat):
        """x: (B, N, C), N may include a CLS token at position 0."""
        B, N, C = x.shape
        has_cls = (N == h_feat * w_feat + 1)

        # separate CLS token
        if has_cls:
            cls, spatial = x[:, :1], x[:, 1:]
        else:
            cls, spatial = None, x

        # ── Global self-attention ─────────────────────────────────────────────
        q = rearrange(self.to_q(x), 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(self.to_k(x), 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(self.to_v(x), 'b n (h d) -> b h n d', h=self.heads)

        attn = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        global_out = torch.einsum('bhij,bhjd->bhid', attn, v)
        global_out = rearrange(global_out, 'b h n d -> b n (h d)')   # (B, N, inner)

        # ── Local spatial branch ──────────────────────────────────────────────
        sp_2d = rearrange(spatial, 'b (h w) c -> b c h w', h=h_feat, w=w_feat)
        loc_2d = self.local_pw(self.local_dw(sp_2d))                  # (B, inner, h, w)
        loc_2d = self.local_norm(loc_2d)
        loc    = rearrange(loc_2d, 'b c h w -> b (h w) c')            # (B, hw, inner)

        if has_cls:
            # for CLS token, use a zero local feature (or global mean)
            cls_loc = loc.mean(dim=1, keepdim=True)
            loc = torch.cat([cls_loc, loc], dim=1)                     # (B, N, inner)

        # ── Fusion via gate ───────────────────────────────────────────────────
        gate = self.gate(torch.cat([global_out, loc], dim=-1))
        fused = gate * global_out + (1 - gate) * loc

        return self.proj(fused)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Feed-Forward Network
# ─────────────────────────────────────────────────────────────────────────────

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  LSAformer Block
# ─────────────────────────────────────────────────────────────────────────────

class LSAformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_ratio=4, dropout=0.0, kernel_size=3):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = LocalityEnhancedSelfAttention(dim, heads=heads, dim_head=dim_head,
                                                    dropout=dropout, kernel_size=kernel_size)
        self.ff    = FeedForward(dim, dim * mlp_ratio, dropout=dropout)

    def forward(self, x, h_feat, w_feat):
        x = x + self.attn(self.norm1(x), h_feat, w_feat)
        x = x + self.ff(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Patch Embedding
# ─────────────────────────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """
    Overlapping patch embedding via strided convolution.
    Better for fine-grained texture features than non-overlapping patches.
    """
    def __init__(self, image_size=64, patch_size=8, in_channels=3, embed_dim=256,
                 overlap=True):
        super().__init__()
        stride = patch_size // 2 if overlap else patch_size
        padding = patch_size // 4 if overlap else 0

        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = nn.LayerNorm(embed_dim)

        # Compute output spatial size
        h_out = (image_size + 2 * padding - patch_size) // stride + 1
        self.h_feat = h_out
        self.w_feat = h_out
        self.num_patches = h_out * h_out

    def forward(self, x):
        x = self.proj(x)                                   # (B, C, H', W')
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        return x, H, W


# ─────────────────────────────────────────────────────────────────────────────
# 5.  LSAformer (full model)
# ─────────────────────────────────────────────────────────────────────────────

class LSAformer(nn.Module):
    """
    LSAformer for steel surface defect classification.

    Args:
        image_size   : Input image size (default 64 for NEU-CLS-64)
        patch_size   : Patch size for tokenization (default 8)
        in_channels  : 1 for grayscale, 3 for RGB (default 3)
        num_classes  : Number of defect classes (default 9)
        embed_dim    : Token embedding dimension (default 256)
        depth        : Number of transformer blocks (default 8)
        heads        : Number of attention heads (default 8)
        dim_head     : Dimension per attention head (default 32)
        mlp_ratio    : MLP hidden dim multiplier (default 4)
        dropout      : Dropout rate (default 0.1)
        emb_dropout  : Embedding dropout rate (default 0.1)
    """

    def __init__(
        self,
        image_size=64,
        patch_size=8,
        in_channels=3,
        num_classes=9,
        embed_dim=256,
        depth=8,
        heads=8,
        dim_head=32,
        mlp_ratio=4,
        dropout=0.1,
        emb_dropout=0.1,
    ):
        super().__init__()

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            overlap=True,
        )
        num_patches = self.patch_embed.num_patches

        # CLS token + positional embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.emb_drop  = nn.Dropout(emb_dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            LSAformerBlock(
                dim=embed_dim, heads=heads, dim_head=dim_head,
                mlp_ratio=mlp_ratio, dropout=dropout
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token,  std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Patch embedding
        x, h_feat, w_feat = self.patch_embed(x)   # (B, N, C)
        B, N, C = x.shape

        # Prepend CLS token
        cls = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        x   = torch.cat([cls, x], dim=1)           # (B, N+1, C)
        x   = x + self.pos_embed
        x   = self.emb_drop(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, h_feat, w_feat)

        x = self.norm(x)

        # Classify via CLS token
        cls_out = x[:, 0]
        return self.head(cls_out)

    def get_attention_maps(self, x):
        """Return attention maps for visualization."""
        x, h_feat, w_feat = self.patch_embed(x)
        B, N, C = x.shape
        cls = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        x   = torch.cat([cls, x], dim=1) + self.pos_embed
        x   = self.emb_drop(x)

        attn_maps = []
        for block in self.blocks:
            # Extract attention weights
            normed = block.norm1(x)
            q = rearrange(block.attn.to_q(normed), 'b n (h d) -> b h n d', h=block.attn.heads)
            k = rearrange(block.attn.to_k(normed), 'b n (h d) -> b h n d', h=block.attn.heads)
            attn = (torch.einsum('bhid,bhjd->bhij', q, k) * block.attn.scale).softmax(dim=-1)
            attn_maps.append(attn.detach())
            x = block(x, h_feat, w_feat)

        return attn_maps, h_feat, w_feat


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Model factory helpers
# ─────────────────────────────────────────────────────────────────────────────

def lsaformer_tiny(num_classes=9, **kwargs):
    return LSAformer(embed_dim=128, depth=6,  heads=4, dim_head=32,
                     num_classes=num_classes, **kwargs)

def lsaformer_small(num_classes=9, **kwargs):
    return LSAformer(embed_dim=256, depth=8,  heads=8, dim_head=32,
                     num_classes=num_classes, **kwargs)

def lsaformer_base(num_classes=9, **kwargs):
    return LSAformer(embed_dim=384, depth=12, heads=8, dim_head=48,
                     num_classes=num_classes, **kwargs)


if __name__ == '__main__':
    model = lsaformer_tiny(num_classes=9)
    x = torch.randn(4, 3, 64, 64)
    logits = model(x)
    print(f"Output shape: {logits.shape}")          # (4, 9)
    print(f"Parameters:   {sum(p.numel() for p in model.parameters()):,}")