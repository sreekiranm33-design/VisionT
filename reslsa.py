"""
LSAformer + ResNet-18 Backbone + Frequency Enhancement Branch
for Steel Surface Defect Classification (NEU-CLS-64, 9 classes)

Architecture:
  ResNet-18 stem (pretrained ImageNet) → feature maps
  → Project to embed_dim → tokenize → CLS token + pos embed
  → 6x LSAformer blocks (3-branch: global attn + local conv + FFT freq)
  → classification head

Frequency Enhancement Branch — how it works:
  1. spatial tokens → reshape to (B, C, 8, 8)
  2. rfft2 → complex spectrum (B, C, 8, 5)
  3. element-wise multiply by learnable complex weight mask
     (model learns which frequencies matter for each defect)
  4. irfft2 → back to (B, C, 8, 8)  [frequency-filtered spatial features]
  5. 1×1 conv + GroupNorm + GELU → (B, inner_dim, 8, 8)
  6. flatten → (B, N, inner_dim)

3-way fusion gate (softmax, not sigmoid):
  gate_weights = softmax( Linear([global, local, freq]) → 3 )
  out = w_g * global + w_l * local + w_f * freq

Why freq helps steel defects specifically:
  rp (rolled pits)  → periodic pattern → sharp FFT peak at specific frequency
  cr (crazing)      → irregular cracks → spread low-freq energy
  sp (spalling)     → coarse pits      → dominant low-frequency components
  ps (pitted surf)  → fine uniform pit → dominant mid-high frequency components
  These signatures are invisible in spatial domain but clear in freq domain.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torchvision.models import resnet18, ResNet18_Weights


# ─────────────────────────────────────────────────────────────────────────────
# 1.  ResNet-18 CNN Backbone
# ─────────────────────────────────────────────────────────────────────────────

class ResNetBackbone(nn.Module):
    """
    Uses ResNet-18 layers 1+2 as a CNN feature extractor.

    For 64×64 input:
      stem (conv1+bn+relu+maxpool): 64×64 → 16×16, 64ch
      layer1 (2 residual blocks):  16×16 → 16×16, 64ch
      layer2 (2 residual blocks):  16×16 →  8×8, 128ch

    Output: (B, 128, 8, 8) → 64 spatial tokens of dim 128
    """

    def __init__(self, in_channels=1, pretrained=True):
        super().__init__()

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base    = resnet18(weights=weights)

        if in_channels != 3:
            orig_conv = base.conv1
            new_conv  = nn.Conv2d(
                in_channels, 64,
                kernel_size=7, stride=2, padding=3, bias=False
            )
            if pretrained:
                new_conv.weight.data = orig_conv.weight.data.mean(dim=1, keepdim=True)
            base.conv1 = new_conv

        self.stem   = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.out_channels = 128

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 2.  CNN → Token projection
# ─────────────────────────────────────────────────────────────────────────────

class CNNTokenizer(nn.Module):
    """(B, C_cnn, H, W) → (B, H*W, embed_dim)"""

    def __init__(self, cnn_channels=128, embed_dim=256):
        super().__init__()
        self.proj = nn.Conv2d(cnn_channels, embed_dim, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        return x, H, W


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Frequency Enhancement Branch
# ─────────────────────────────────────────────────────────────────────────────

class FrequencyEnhancementBranch(nn.Module):
    """
    Frequency-domain feature enhancement via learnable 2D FFT filtering.

    The learnable complex weight mask acts as a trainable bandpass filter —
    it learns to amplify defect-characteristic frequencies and suppress noise.

    Args:
        dim       : input channel dimension (= embed_dim)
        inner_dim : output dimension (= dim_head * heads, matches other branches)
        h, w      : spatial grid size (8, 8 for 64×64 input)
    """

    def __init__(self, dim, inner_dim, h=8, w=8):
        super().__init__()
        self.h = h
        self.w = w

        # rfft2 output shape: (H, W//2+1) exploiting conjugate symmetry
        freq_h = h
        freq_w = w // 2 + 1   # = 5 for w=8

        # Learnable complex weight mask — initialised as identity (1+0j)
        # Model learns which frequencies to amplify/suppress per channel
        self.weight_real = nn.Parameter(torch.ones(1, dim, freq_h, freq_w))
        self.weight_imag = nn.Parameter(torch.zeros(1, dim, freq_h, freq_w))

        # Channel mixing after IFFT
        self.proj = nn.Conv2d(dim, inner_dim, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(1, inner_dim)
        self.act  = nn.GELU()

    def forward(self, spatial_tokens, h_feat, w_feat):
        """
        Args:
            spatial_tokens : (B, N, C)  — patch tokens only (no CLS)
            h_feat, w_feat : spatial grid size
        Returns:
            (B, N, inner_dim)
        """
        # 1. Reshape tokens → 2D spatial grid
        x = rearrange(spatial_tokens, 'b (h w) c -> b c h w',
                      h=h_feat, w=w_feat)              # (B, C, H, W)

        # 2. Forward FFT → complex frequency domain
        x_freq = torch.fft.rfft2(x, norm='ortho')      # (B, C, H, W//2+1) complex

        # 3. Apply learnable complex filter (element-wise multiply)
        weight = torch.complex(self.weight_real, self.weight_imag)
        x_freq = x_freq * weight

        # 4. Inverse FFT → spatial domain (now frequency-enhanced)
        x_enh = torch.fft.irfft2(x_freq, s=(h_feat, w_feat),
                                  norm='ortho')         # (B, C, H, W) real

        # 5. Project + norm + activate
        x_enh = self.act(self.norm(self.proj(x_enh)))  # (B, inner_dim, H, W)

        # 6. Flatten → token sequence
        return rearrange(x_enh, 'b c h w -> b (h w) c')  # (B, N, inner_dim)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  LSA with 3-Branch Fusion
# ─────────────────────────────────────────────────────────────────────────────

class LocalityEnhancedSelfAttention(nn.Module):
    """
    Three parallel branches fused by a learned softmax gate:

      Branch 1 — Global Self-Attention  : long-range spatial context
      Branch 2 — Local Depthwise Conv   : short-range texture details
      Branch 3 — Frequency Enhancement  : FFT learnable filter → IFFT

    Fusion:
      gate_weights = softmax( MLP([global ∥ local ∥ freq]) )  →  (B, N, 3)
      out = w_g·global + w_l·local + w_f·freq

    Softmax gate ensures weights always sum to 1, preventing any branch
    from dominating and keeping training stable.
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0,
                 kernel_size=3, h=8, w=8):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads    = heads
        self.dim_head = dim_head
        self.scale    = dim_head ** -0.5

        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        # Branch 2: local depthwise conv (unchanged from original)
        self.local_dw   = nn.Conv2d(dim, dim, kernel_size=kernel_size,
                                    padding=kernel_size // 2, groups=dim, bias=False)
        self.local_pw   = nn.Conv2d(dim, inner_dim, kernel_size=1, bias=False)
        self.local_norm = nn.GroupNorm(1, inner_dim)

        # Branch 3: frequency enhancement (new)
        self.freq_branch = FrequencyEnhancementBranch(
            dim=dim, inner_dim=inner_dim, h=h, w=w
        )

        # 3-way softmax gate
        # Input:  [global ∥ local ∥ freq]  shape (B, N, inner_dim*3)
        # Output: 3 blending weights        shape (B, N, 3)
        self.gate = nn.Sequential(
            nn.Linear(inner_dim * 3, inner_dim * 3 // 4),
            nn.GELU(),
            nn.Linear(inner_dim * 3 // 4, 3),
            nn.Softmax(dim=-1),
        )

        self.proj = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, h_feat, w_feat):
        B, N, C = x.shape
        has_cls = (N == h_feat * w_feat + 1)

        if has_cls:
            cls_tok, spatial = x[:, :1], x[:, 1:]
        else:
            cls_tok, spatial = None, x

        # ── Branch 1: Global self-attention ───────────────────────────────────
        q = rearrange(self.to_q(x), 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(self.to_k(x), 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(self.to_v(x), 'b n (h d) -> b h n d', h=self.heads)

        attn       = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn       = attn.softmax(dim=-1)
        global_out = rearrange(
            torch.einsum('bhij,bhjd->bhid', attn, v),
            'b h n d -> b n (h d)'
        )                                              # (B, N, inner_dim)

        # ── Branch 2: Local depthwise conv ────────────────────────────────────
        sp_2d  = rearrange(spatial, 'b (h w) c -> b c h w', h=h_feat, w=w_feat)
        loc    = rearrange(
            self.local_norm(self.local_pw(self.local_dw(sp_2d))),
            'b c h w -> b (h w) c'
        )                                              # (B, N_spatial, inner_dim)
        if has_cls:
            loc = torch.cat([loc.mean(dim=1, keepdim=True), loc], dim=1)

        # ── Branch 3: Frequency enhancement ──────────────────────────────────
        freq = self.freq_branch(spatial, h_feat, w_feat)  # (B, N_spatial, inner_dim)
        if has_cls:
            freq = torch.cat([freq.mean(dim=1, keepdim=True), freq], dim=1)

        # ── 3-way softmax gate ─────────────────────────────────────────────────
        gate_w = self.gate(torch.cat([global_out, loc, freq], dim=-1))  # (B, N, 3)

        fused = (gate_w[..., 0:1] * global_out +
                 gate_w[..., 1:2] * loc        +
                 gate_w[..., 2:3] * freq)               # (B, N, inner_dim)

        return self.proj(fused)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  FeedForward (unchanged)
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
# 6.  LSAformerBlock (passes h, w to LESA for freq branch sizing)
# ─────────────────────────────────────────────────────────────────────────────

class LSAformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_ratio=4,
                 dropout=0.0, kernel_size=3, h=8, w=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = LocalityEnhancedSelfAttention(
            dim, heads=heads, dim_head=dim_head,
            dropout=dropout, kernel_size=kernel_size, h=h, w=w
        )
        self.ff    = FeedForward(dim, dim * mlp_ratio, dropout=dropout)

    def forward(self, x, h_feat, w_feat):
        x = x + self.attn(self.norm1(x), h_feat, w_feat)
        x = x + self.ff(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 7.  LSAformerResNet — full model
# ─────────────────────────────────────────────────────────────────────────────

class LSAformerResNet(nn.Module):
    """
    Hybrid CNN-Transformer with Frequency Enhancement for steel defect classification.

    Each LSAformer block now has THREE parallel branches:
      1. Global Self-Attention  — spatial long-range context
      2. Local Depthwise Conv   — spatial short-range texture
      3. Frequency Enhancement  — FFT learnable filter → frequency-domain features

    Args:
        in_channels     : 1 (grayscale) or 3 (RGB)
        num_classes     : number of output classes
        embed_dim       : transformer embedding dimension
        depth           : number of LSAformer blocks
        heads           : attention heads
        dim_head        : dimension per head
        mlp_ratio       : FFN expansion ratio
        dropout         : dropout in transformer
        emb_dropout     : embedding dropout
        pretrained      : use ImageNet pretrained ResNet-18 weights
        freeze_backbone : freeze CNN backbone during warmup
    """

    def __init__(
        self,
        in_channels=1,
        num_classes=9,
        embed_dim=256,
        depth=6,
        heads=8,
        dim_head=32,
        mlp_ratio=4,
        dropout=0.1,
        emb_dropout=0.1,
        pretrained=True,
        freeze_backbone=False,
    ):
        super().__init__()

        self.h_feat = 8
        self.w_feat = 8
        num_patches = self.h_feat * self.w_feat   # 64

        # ── CNN Backbone ──────────────────────────────────────────────────────
        self.backbone  = ResNetBackbone(in_channels=in_channels, pretrained=pretrained)
        self.tokenizer = CNNTokenizer(cnn_channels=128, embed_dim=embed_dim)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # ── Transformer ───────────────────────────────────────────────────────
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.emb_drop  = nn.Dropout(emb_dropout)

        self.blocks = nn.ModuleList([
            LSAformerBlock(
                dim=embed_dim, heads=heads, dim_head=dim_head,
                mlp_ratio=mlp_ratio, dropout=dropout,
                h=self.h_feat, w=self.w_feat,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # ── Classification head ───────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

        self._init_transformer_weights()

    def _init_transformer_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        feat             = self.backbone(x)
        tokens, h, w     = self.tokenizer(feat)

        B   = tokens.shape[0]
        cls = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        x   = torch.cat([cls, tokens], dim=1)
        x   = x + self.pos_embed
        x   = self.emb_drop(x)

        for block in self.blocks:
            x = block(x, h, w)

        x = self.norm(x)
        return self.head(x[:, 0])

    def get_attention_maps(self, x):
        feat         = self.backbone(x)
        tokens, h, w = self.tokenizer(feat)
        B            = tokens.shape[0]
        cls          = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        x            = torch.cat([cls, tokens], dim=1) + self.pos_embed
        x            = self.emb_drop(x)

        attn_maps = []
        for block in self.blocks:
            normed = block.norm1(x)
            q = rearrange(block.attn.to_q(normed), 'b n (h d) -> b h n d',
                          h=block.attn.heads)
            k = rearrange(block.attn.to_k(normed), 'b n (h d) -> b h n d',
                          h=block.attn.heads)
            attn = (torch.einsum('bhid,bhjd->bhij', q, k) * block.attn.scale
                    ).softmax(dim=-1)
            attn_maps.append(attn.detach())
            x = block(x, h, w)

        return attn_maps, h, w

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
        print("Backbone unfrozen — all parameters trainable.")

    def param_groups(self, lr_backbone=1e-5, lr_transformer=3e-4):
        backbone_params    = list(self.backbone.parameters()) + \
                             list(self.tokenizer.parameters())
        transformer_params = [self.cls_token, self.pos_embed] + \
                             list(self.blocks.parameters()) + \
                             list(self.norm.parameters()) + \
                             list(self.head.parameters())
        return [
            {'params': backbone_params,    'lr': lr_backbone},
            {'params': transformer_params, 'lr': lr_transformer},
        ]


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Factory helpers
# ─────────────────────────────────────────────────────────────────────────────

def lsaformer_resnet_tiny(num_classes=9, **kwargs):
    """tiny: embed_dim=128, depth=6, heads=4"""
    return LSAformerResNet(
        embed_dim=128, depth=6, heads=4, dim_head=32,
        num_classes=num_classes, **kwargs
    )

def lsaformer_resnet_small(num_classes=9, **kwargs):
    """small: embed_dim=256, depth=6, heads=8"""
    return LSAformerResNet(
        embed_dim=256, depth=6, heads=8, dim_head=32,
        num_classes=num_classes, **kwargs
    )

def lsaformer_resnet_base(num_classes=9, **kwargs):
    """base: embed_dim=384, depth=8, heads=8"""
    return LSAformerResNet(
        embed_dim=384, depth=8, heads=8, dim_head=48,
        num_classes=num_classes, **kwargs
    )


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("LSAformer + ResNet-18 + Frequency Enhancement Branch")
    print("=" * 60)

    for size, fn in [('tiny', lsaformer_resnet_tiny),
                     ('small', lsaformer_resnet_small)]:
        model     = fn(num_classes=9, in_channels=1, pretrained=False)
        x         = torch.randn(4, 1, 64, 64)
        logits    = model(x)
        total     = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        freq_p    = sum(
            p.numel()
            for block in model.blocks
            for _, p in block.attn.freq_branch.named_parameters()
        )
        print(f"\nlsaformer_resnet_{size}")
        print(f"  Output shape : {logits.shape}")
        print(f"  Total params : {total:,}")
        print(f"  Trainable    : {trainable:,}")
        print(f"  Freq branch  : {freq_p:,} params across all {len(model.blocks)} blocks")
    print()