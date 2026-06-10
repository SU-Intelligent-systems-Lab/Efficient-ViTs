"""
EfficientFormer building blocks.

This module implements the "dimension-consistent" blocks of EfficientFormer
(Li et al., NeurIPS 2022, "EfficientFormer: Vision Transformers at MobileNet
Speed"). The central idea of the paper is that a network should spend most of
its depth in a hardware-friendly 4D convolutional form and switch to the 3D
token-attention form only at the very end, where the spatial resolution is
small enough that attention is cheap.

Two MetaBlock flavours implement this split:

- ``MB4D`` (4D MetaBlock): operates on ``(B, C, H, W)`` feature maps. Its token
  mixer is a parameter-free 3x3 average pool (PoolFormer-style) and its MLP is
  two 1x1 convolutions with BatchNorm -- everything folds into convolutions for
  fast inference.
- ``MB3D`` (3D MetaBlock): operates on ``(B, N, C)`` token sequences. It is a
  standard pre-norm Transformer block (LayerNorm + multi-head self-attention
  with a learned bias + LayerNorm + Linear MLP), used only at the tail of the
  last stage.

Supporting layers here:

- ``ConvStem``          : patch-embedding stem (two 3x3 stride-2 convs, /4).
- ``StageDownsample``   : 3x3 stride-2 conv between stages (/2).
- ``PoolMixer``         : PoolFormer token mixer ``avgpool(x) - x``.
- ``ConvMLP``           : 1x1 conv MLP for MB4D.
- ``EfficientFormerAttention`` : MB3D multi-head self-attention with learned bias.
- ``LinearMLP``         : Linear MLP for MB3D.

Key equations (paper Sec. 4.1)
------------------------------
- MB4D:  x = x + Pool(x);   x = x + ConvMLP(x)
- MB3D:  x = x + MHSA(LN(x));  x = x + LinearMLP(LN(x))
- MHSA:  softmax( Q K^T / sqrt(d) + b ) V,  with ``b`` a learned per-(i,j) bias.

LayerScale
----------
Each residual branch is scaled by a learnable per-channel factor ``gamma``
initialised to ``1e-5`` (LayerScale, Touvron et al.). This keeps the residual
contributions tiny at initialisation, which stabilises training of the deeper
variants. It does not change the architecture's character.

Fixed resolution
----------------
The MB3D attention bias ``b`` is a full ``(num_heads, N, N)`` table with
``N = resolution^2`` fixed at construction time, so EfficientFormer -- like
EfficientViT -- is a fixed-input-size model and is excluded from multi-scale
sampling.

Example
-------
>>> import torch
>>> from models.efficientformer_block import MB4D, MB3D
>>>
>>> x4d = torch.randn(2, 64, 24, 24)
>>> MB4D(dim=64)(x4d).shape
torch.Size([2, 64, 24, 24])
>>>
>>> x3d = torch.randn(2, 36, 448)          # 36 = 6 * 6 tokens
>>> MB3D(dim=448, num_heads=8, qk_dim=32, resolution=6)(x3d).shape
torch.Size([2, 36, 448])
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .common import ensure_positive_int
from .stochastic import DropPath


__all__ = [
    "ConvStem",
    "StageDownsample",
    "PoolMixer",
    "ConvMLP",
    "MB4D",
    "EfficientFormerAttention",
    "LinearMLP",
    "MB3D",
]


class ConvStem(nn.Module):
    """
    Patch-embedding stem: two 3x3 stride-2 convolutions (4x downsample).

    EfficientFormer replaces a ViT's single large-kernel patch embedding with
    two small overlapping strided convolutions, each followed by BatchNorm and
    GELU. The paper identifies the non-overlapping large-kernel embedding as a
    mobile latency bottleneck; this stem is faster and downsamples by 4x:

        in_channels -> out_channels/2 -> out_channels
        H -> H/2 -> H/4

    Parameters
    ----------
    in_channels:
        Number of input image channels.

    out_channels:
        Output channel dimension (stage-1 width). The intermediate width is
        ``out_channels // 2``.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        ensure_positive_int(in_channels, "in_channels")
        ensure_positive_int(out_channels, "out_channels")

        mid = out_channels // 2
        self.conv1 = nn.Conv2d(in_channels, mid, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(mid, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Input shape:  (B, in_channels, H, W)
        Output shape: (B, out_channels, H/4, W/4)
        """
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        return x


class StageDownsample(nn.Module):
    """
    Between-stage downsampler: a single 3x3 stride-2 convolution + BatchNorm.

    Halves the spatial resolution and projects to the next stage's channel
    width. Used at the boundary between two stages of MB4D blocks.

    Parameters
    ----------
    in_channels:
        Channels entering the downsampler.

    out_channels:
        Channels leaving the downsampler.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        ensure_positive_int(in_channels, "in_channels")
        ensure_positive_int(out_channels, "out_channels")

        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        """
        Input shape:  (B, in_channels,  H,   W  )
        Output shape: (B, out_channels, H/2, W/2)
        """
        return self.bn(self.proj(x))


class PoolMixer(nn.Module):
    """
    PoolFormer token mixer: a 3x3 average pool minus the identity.

    The mixer returns ``avgpool(x) - x`` so that, when wrapped in the MB4D
    residual ``x + gamma * PoolMixer(x)``, the update simplifies to the
    standard PoolFormer relation ``x + gamma * (avgpool(x) - x)``. Average
    pooling is parameter-free and extremely cheap, which is the point: most of
    EfficientFormer's depth uses it instead of attention.

    Parameters
    ----------
    pool_size:
        Side length of the average-pooling window. Default 3 (stride 1, "same"
        padding, so the spatial size is preserved).
    """

    def __init__(self, pool_size: int = 3) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(
            kernel_size=pool_size,
            stride=1,
            padding=pool_size // 2,
            count_include_pad=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Input shape:  (B, C, H, W)
        Output shape: (B, C, H, W)   -- returns ``avgpool(x) - x``.
        """
        return self.pool(x) - x


class ConvMLP(nn.Module):
    """
    Convolutional MLP for MB4D: 1x1 conv -> BN -> GELU -> 1x1 conv -> BN.

    The "all conv, all BN" design means the whole MLP can be folded into two
    convolutions at inference time (paper Observation 3). Operates on 4D feature
    maps without leaving the spatial layout.

    Parameters
    ----------
    dim:
        Input/output channel dimension.

    mlp_ratio:
        Hidden-dimension expansion ratio. Default 4.0.
    """

    def __init__(self, dim: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        ensure_positive_int(dim, "dim")
        hidden = int(dim * mlp_ratio)

        self.fc1 = nn.Conv2d(dim, hidden, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden, dim, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Input shape:  (B, C, H, W)
        Output shape: (B, C, H, W)
        """
        x = self.act(self.bn1(self.fc1(x)))
        x = self.bn2(self.fc2(x))
        return x


class MB4D(nn.Module):
    """
    4D MetaBlock: PoolFormer token mixer + convolutional MLP, both residual.

    Equations (with per-channel LayerScale ``gamma1, gamma2``):

        x = x + DropPath( gamma1 * PoolMixer(x) )
        x = x + DropPath( gamma2 * ConvMLP(x)   )

    Operates entirely on ``(B, C, H, W)`` tensors. This is the workhorse block
    that fills most of EfficientFormer's depth.

    Parameters
    ----------
    dim:
        Channel dimension (preserved).

    mlp_ratio:
        Expansion ratio for the convolutional MLP. Default 4.0.

    drop_path:
        Stochastic-depth probability for the two residual branches.
    """

    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop_path: float = 0.0) -> None:
        super().__init__()
        ensure_positive_int(dim, "dim")

        self.token_mixer = PoolMixer(pool_size=3)
        self.mlp = ConvMLP(dim, mlp_ratio)
        self.drop_path = DropPath(drop_path)

        # LayerScale: per-channel learnable scale, init 1e-5 (tiny residuals).
        self.gamma1 = nn.Parameter(1e-5 * torch.ones(dim))
        self.gamma2 = nn.Parameter(1e-5 * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        """
        Input shape:  (B, C, H, W)
        Output shape: (B, C, H, W)
        """
        # gamma reshaped to (1, C, 1, 1) so it scales per channel across H, W.
        x = x + self.drop_path(self.gamma1.view(1, -1, 1, 1) * self.token_mixer(x))
        x = x + self.drop_path(self.gamma2.view(1, -1, 1, 1) * self.mlp(x))
        return x


class EfficientFormerAttention(nn.Module):
    """
    MB3D multi-head self-attention with a learned (fixed-resolution) bias.

    A standard multi-head attention with two EfficientFormer specifics:

    1. Asymmetric head widths: queries and keys use ``qk_dim`` channels per
       head while values use ``dim // num_heads``, so the concatenated value
       width matches ``dim`` and the output projection is square.
    2. A learned attention bias ``b`` of shape ``(num_heads, N, N)`` is added to
       the logits before softmax (paper Eq. 6). Because ``N = resolution^2`` is
       fixed, this module only accepts a token sequence of exactly that length.

        attn = softmax( (Q K^T) / sqrt(qk_dim) + b )
        out  = attn V

    Parameters
    ----------
    dim:
        Token embedding dimension.

    num_heads:
        Number of attention heads. Default 8.

    qk_dim:
        Per-head query/key dimension. Default 32.

    resolution:
        Side length of the (square) token grid. The bias table is built for
        ``N = resolution * resolution`` tokens.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qk_dim: int = 32,
        resolution: int = 7,
    ) -> None:
        super().__init__()
        ensure_positive_int(dim, "dim")
        ensure_positive_int(num_heads, "num_heads")
        ensure_positive_int(qk_dim, "qk_dim")
        ensure_positive_int(resolution, "resolution")
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")

        self.num_heads = num_heads
        self.qk_dim = qk_dim
        self.v_dim = dim // num_heads  # per-head value width
        self.scale = qk_dim**-0.5      # 1 / sqrt(qk_dim)
        self.resolution = resolution

        self.q = nn.Linear(dim, num_heads * qk_dim, bias=False)
        self.k = nn.Linear(dim, num_heads * qk_dim, bias=False)
        self.v = nn.Linear(dim, num_heads * self.v_dim, bias=False)
        self.proj = nn.Linear(num_heads * self.v_dim, dim)

        # Learned attention bias: one scalar per (head, query_pos, key_pos).
        num_tokens = resolution * resolution
        self.attention_bias = nn.Parameter(torch.zeros(num_heads, num_tokens, num_tokens))

    def forward(self, x: Tensor) -> Tensor:
        """
        Input shape:  (B, N, dim)   with N == resolution^2
        Output shape: (B, N, dim)
        """
        batch_size, num_tokens, _ = x.shape
        expected = self.resolution * self.resolution
        if num_tokens != expected:
            raise ValueError(
                "EfficientFormer attention is fixed-resolution: expected "
                f"N={expected} tokens (resolution {self.resolution}), got N={num_tokens}."
            )

        # Project and split into heads. Each: (B, num_heads, N, head_dim).
        q = self.q(x).reshape(batch_size, num_tokens, self.num_heads, self.qk_dim).permute(0, 2, 1, 3)
        k = self.k(x).reshape(batch_size, num_tokens, self.num_heads, self.qk_dim).permute(0, 2, 1, 3)
        v = self.v(x).reshape(batch_size, num_tokens, self.num_heads, self.v_dim).permute(0, 2, 1, 3)

        # Scaled dot-product scores + learned bias, then softmax over keys.
        # (B, H, N, qk) @ (B, H, qk, N) -> (B, H, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + self.attention_bias.unsqueeze(0)
        attn = attn.softmax(dim=-1)

        # Weighted sum of values, merge heads, project back to dim.
        # (B, H, N, N) @ (B, H, N, v) -> (B, H, N, v) -> (B, N, H*v)
        out = (attn @ v).transpose(1, 2).reshape(batch_size, num_tokens, self.num_heads * self.v_dim)
        return self.proj(out)


class LinearMLP(nn.Module):
    """
    Linear MLP for MB3D: Linear -> GELU -> Linear (operates on tokens).

    Parameters
    ----------
    dim:
        Token embedding dimension.

    mlp_ratio:
        Hidden-dimension expansion ratio. Default 4.0.
    """

    def __init__(self, dim: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        ensure_positive_int(dim, "dim")
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Input shape:  (B, N, dim)
        Output shape: (B, N, dim)
        """
        return self.fc2(self.act(self.fc1(x)))


class MB3D(nn.Module):
    """
    3D MetaBlock: pre-norm Transformer block over tokens, both branches residual.

    Equations (with per-channel LayerScale ``gamma1, gamma2``):

        x = x + DropPath( gamma1 * Attention(LayerNorm(x)) )
        x = x + DropPath( gamma2 * LinearMLP(LayerNorm(x)) )

    Operates on ``(B, N, C)`` token sequences. Used only at the tail of the last
    stage, where ``N`` is small.

    Parameters
    ----------
    dim:
        Token embedding dimension.

    num_heads:
        Number of attention heads. Default 8.

    qk_dim:
        Per-head query/key dimension. Default 32.

    mlp_ratio:
        MLP expansion ratio. Default 4.0.

    resolution:
        Side length of the token grid (fixes the attention-bias size).

    drop_path:
        Stochastic-depth probability for the two residual branches.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qk_dim: int = 32,
        mlp_ratio: float = 4.0,
        resolution: int = 7,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        ensure_positive_int(dim, "dim")

        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientFormerAttention(dim, num_heads, qk_dim, resolution)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = LinearMLP(dim, mlp_ratio)
        self.drop_path = DropPath(drop_path)

        # LayerScale: per-channel scale over the C dimension of (B, N, C).
        self.gamma1 = nn.Parameter(1e-5 * torch.ones(dim))
        self.gamma2 = nn.Parameter(1e-5 * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        """
        Input shape:  (B, N, dim)
        Output shape: (B, N, dim)
        """
        x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x
