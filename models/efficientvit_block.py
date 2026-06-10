"""
EfficientViT attention and the EfficientViT building block.

This module implements EfficientViT's token mixer -- Cascaded Group Attention
inside a Local Window -- and the residual block that wraps it. EfficientViT
operates entirely on spatial ``(B, C, H, W)`` feature maps (no token reshaping),
which is why its attention lives here rather than in
:mod:`models.transformer` (that module is reserved for the generic
sequence-Transformer used by MobileViT).

Two ideas define EfficientViT's attention:

1. Cascaded Group Attention (CGA)
   Multi-head attention usually feeds the *same* input to every head, which is
   wasteful because heads learn redundant projections. CGA instead splits the
   channels into ``num_heads`` groups and gives each head only its own slice.
   Crucially the heads are *cascaded*: the output of head ``i`` is added to the
   input slice of head ``i + 1``. This grows the effective receptive field head
   by head while keeping each head's Q/K/V projection small, cutting both
   computation and parameter redundancy.

2. Local Window Attention
   Attention is computed within non-overlapping square windows of size
   ``window_resolution`` rather than globally, so the cost is ``O(window^2)``
   per window instead of ``O((H*W)^2)``. When the feature map is no larger than
   one window, the whole map is attended at once.

Relative-position bias (fixed resolution!)
-------------------------------------------
CGA adds a learned bias ``b[i, j]`` to the attention logits for every ordered
pair of spatial positions ``(i, j)`` within a window:

    attn = softmax( Q^T K / sqrt(key_dim) + b )

The bias depends only on the *relative offset* between the two positions, so
positions sharing an offset share a bias value. This is realised with:

- ``attention_biases``    : a learnable table ``(num_heads, num_unique_offsets)``.
- ``attention_bias_idxs`` : a constant index map ``(N, N)`` with ``N = window^2``
                            that gathers the right bias for each ``(i, j)`` pair.

Because ``attention_bias_idxs`` is built for one specific window resolution,
**EfficientViT attention is fixed-resolution**: the spatial size of every stage
is determined at construction time and asserted at run time. Mixed-size /
variable-resolution training is therefore not supported for EfficientViT (see
:class:`LocalWindowAttention.forward`'s size check). This is the reason the
model registry marks EfficientViT as *not* dynamic-input-safe.

Example
-------
>>> import torch
>>> from models.efficientvit_block import EfficientViTBlock
>>>
>>> x = torch.randn(2, 128, 14, 14)            # feature map at resolution 14
>>> block = EfficientViTBlock(
...     embed_dim=128, key_dim=16, num_heads=4, attn_ratio=2.0,
...     resolution=14, window_resolution=7, kernels=(5, 5, 5, 5),
... )
>>> block(x).shape
torch.Size([2, 128, 14, 14])
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .common import ensure_positive_int
from .efficientvit_layers import FFN
from .layers import ConvBN
from .stochastic import Residual


__all__ = [
    "CascadedGroupAttention",
    "LocalWindowAttention",
    "EfficientViTBlock",
]


class CascadedGroupAttention(nn.Module):
    r"""
    Cascaded Group Attention (CGA) over a single square window.

    The input channels ``dim`` are split into ``num_heads`` equal groups. Each
    head ``i`` works on its own slice and, for ``i > 0``, also receives the
    output of head ``i - 1`` added in (the "cascade"). Within a head:

    1. A ``ConvBN`` projects the slice to ``[Q | K | V]`` with channel widths
       ``[key_dim, key_dim, value_dim]`` where ``value_dim = attn_ratio * key_dim``.
    2. A depthwise conv is applied to ``Q`` (a cheap local enhancement of the
       query, with a per-head kernel size from ``kernels``).
    3. Q, K, V are flattened to ``(B, c, N)`` with ``N = H*W`` and attention is
       computed with the learned relative-position bias:
           attn = softmax( (Q^T K) / sqrt(key_dim) + bias )   # (B, N, N)
           out  = V @ attn^T                                  # (B, value_dim, N)
    4. The per-head outputs are concatenated and projected back to ``dim``.

    Parameters
    ----------
    dim:
        Number of input/output channels. Must be divisible by ``num_heads``.

    key_dim:
        Channel width of the per-head query and key.

    num_heads:
        Number of cascaded heads/groups.

    attn_ratio:
        Multiplier giving the per-head value width: ``value_dim = round(attn_ratio * key_dim)``.

    resolution:
        Window side length. The attention bias table is built for an
        ``N = resolution * resolution`` token grid, fixing the resolution.

    kernels:
        Per-head kernel size for the depthwise conv applied to the query.
        Must provide at least ``num_heads`` entries.
    """

    def __init__(
        self,
        dim: int,
        key_dim: int,
        num_heads: int = 8,
        attn_ratio: float = 4.0,
        resolution: int = 14,
        kernels: Sequence[int] = (5, 5, 5, 5),
    ) -> None:
        super().__init__()
        ensure_positive_int(dim, "dim")
        ensure_positive_int(key_dim, "key_dim")
        ensure_positive_int(num_heads, "num_heads")
        ensure_positive_int(resolution, "resolution")
        if dim % num_heads != 0:
            raise ValueError(
                f"dim={dim} must be divisible by num_heads={num_heads}."
            )
        if len(kernels) < num_heads:
            raise ValueError(
                f"kernels must provide at least num_heads={num_heads} entries, "
                f"got {len(kernels)}."
            )

        self.num_heads = num_heads
        self.scale = key_dim**-0.5  # 1 / sqrt(key_dim), the attention scaling
        self.key_dim = key_dim
        self.value_dim = int(attn_ratio * key_dim)  # per-head value width
        self.attn_ratio = attn_ratio

        head_in = dim // num_heads

        # Per-head Q/K/V projection and depthwise enhancement of the query.
        qkvs: list[nn.Module] = []
        dws: list[nn.Module] = []
        for i in range(num_heads):
            # head_in -> [key_dim (Q) | key_dim (K) | value_dim (V)]
            qkvs.append(ConvBN(head_in, self.key_dim * 2 + self.value_dim))
            # Depthwise conv on the query slice (per-head kernel size).
            dws.append(
                ConvBN(
                    self.key_dim,
                    self.key_dim,
                    kernel_size=kernels[i],
                    stride=1,
                    padding=kernels[i] // 2,
                    groups=self.key_dim,
                )
            )
        self.qkvs = nn.ModuleList(qkvs)
        self.dws = nn.ModuleList(dws)

        # Output projection: concatenated head values -> dim. The final BN is
        # zero-initialised so the residual branch starts as an identity.
        self.proj = nn.Sequential(
            nn.ReLU(),
            ConvBN(self.value_dim * num_heads, dim, bn_weight_init=0.0),
        )

        # ---- Relative-position bias tables (fixed for this resolution) ----
        # Enumerate every spatial position in the resolution x resolution grid.
        points = list(itertools.product(range(resolution), range(resolution)))
        num_points = len(points)  # N = resolution^2
        attention_offsets: dict[tuple[int, int], int] = {}
        idxs: list[int] = []
        for p1 in points:
            for p2 in points:
                # Absolute relative offset between the two positions.
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])

        # One learnable bias per (head, unique offset).
        self.attention_biases = nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets))
        )
        # Constant gather map: idxs[i, j] -> which unique offset pair (i, j) is.
        # Shape (N, N); used to expand the table to (num_heads, N, N) per call.
        self.register_buffer(
            "attention_bias_idxs",
            torch.LongTensor(idxs).view(num_points, num_points),
            persistent=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply cascaded group attention to one window.

        Input shape:  (B, dim, H, W)   with H * W == resolution^2
        Output shape: (B, dim, H, W)
        """
        batch_size, _, height, width = x.shape

        # Expand the bias table to (num_heads, N, N) for this window.
        attn_bias = self.attention_biases[:, self.attention_bias_idxs]

        # Split channels into one slice per head: each (B, dim/num_heads, H, W).
        feats_in = x.chunk(self.num_heads, dim=1)
        feats_out: list[Tensor] = []
        feat = feats_in[0]

        for i, qkv in enumerate(self.qkvs):
            # Cascade: fold the previous head's output into this head's input.
            if i > 0:
                feat = feat + feats_in[i]

            # Project to Q, K, V along the channel dim.
            # (B, head_in, H, W) -> (B, key_dim*2 + value_dim, H, W)
            feat = qkv(feat)
            q, k, v = feat.view(batch_size, -1, height, width).split(
                [self.key_dim, self.key_dim, self.value_dim], dim=1
            )

            # Cheap depthwise enhancement of the query, then flatten to tokens.
            q = self.dws[i](q)
            # (B, c, H, W) -> (B, c, N), N = H*W
            q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)

            # Attention logits with relative-position bias:
            #   (B, N, key_dim) @ (B, key_dim, N) -> (B, N, N)
            attn = (q.transpose(-2, -1) @ k) * self.scale + attn_bias[i]
            attn = attn.softmax(dim=-1)

            # Weighted sum of values: (B, value_dim, N) @ (B, N, N) -> (B, value_dim, N)
            # then reshape back to a feature map.
            feat = (v @ attn.transpose(-2, -1)).view(
                batch_size, self.value_dim, height, width
            )
            feats_out.append(feat)

        # Concatenate per-head outputs and project back to dim channels.
        return self.proj(torch.cat(feats_out, dim=1))


class LocalWindowAttention(nn.Module):
    r"""
    Cascaded group attention applied within local square windows.

    When the feature map is larger than ``window_resolution``, it is split into
    non-overlapping windows of that size, attention is computed independently in
    each window, and the windows are stitched back together. When the map fits
    in a single window, attention is applied directly.

    The spatial size of the input is fixed: ``forward`` asserts ``H == W ==
    resolution``. This mirrors the reference implementation and is what makes
    EfficientViT incompatible with variable-resolution (multi-scale) training.

    Parameters
    ----------
    dim:
        Number of input/output channels.

    key_dim:
        Per-head query/key width passed to :class:`CascadedGroupAttention`.

    num_heads:
        Number of cascaded heads.

    attn_ratio:
        Value-width multiplier passed to :class:`CascadedGroupAttention`.

    resolution:
        Spatial side length of the feature map at this stage.

    window_resolution:
        Side length of the local attention window. The effective window is
        ``min(window_resolution, resolution)``.

    kernels:
        Per-head depthwise kernel sizes for the query enhancement.
    """

    def __init__(
        self,
        dim: int,
        key_dim: int,
        num_heads: int = 8,
        attn_ratio: float = 4.0,
        resolution: int = 14,
        window_resolution: int = 7,
        kernels: Sequence[int] = (5, 5, 5, 5),
    ) -> None:
        super().__init__()
        ensure_positive_int(dim, "dim")
        ensure_positive_int(resolution, "resolution")
        if window_resolution <= 0:
            raise ValueError("window_resolution must be a positive integer.")

        self.dim = dim
        self.num_heads = num_heads
        self.resolution = resolution
        self.window_resolution = window_resolution

        # The effective window cannot exceed the feature-map resolution.
        effective_window = min(window_resolution, resolution)
        self.attn = CascadedGroupAttention(
            dim,
            key_dim,
            num_heads,
            attn_ratio=attn_ratio,
            resolution=effective_window,
            kernels=kernels,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Input shape:  (B, dim, resolution, resolution)
        Output shape: (B, dim, resolution, resolution)
        """
        height = width = self.resolution
        batch_size, channels, height_in, width_in = x.shape
        if not (height == height_in and width == width_in):
            raise ValueError(
                "EfficientViT uses fixed-resolution attention: expected input "
                f"feature size {(height, width)}, got {(height_in, width_in)}. "
                "Build the model with the image size you intend to run."
            )

        if height <= self.window_resolution and width <= self.window_resolution:
            # The whole map is a single window.
            return self.attn(x)

        # ---- Window partition (B,C,H,W) -> (B*nH*nW, C, w, w) ----
        # Work in channels-last to make the window reshape readable.
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)

        # Pad H and W up to a multiple of the window size if necessary.
        pad_b = (self.window_resolution - height % self.window_resolution) % self.window_resolution
        pad_r = (self.window_resolution - width % self.window_resolution) % self.window_resolution
        padding = pad_b > 0 or pad_r > 0
        if padding:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))

        padded_h, padded_w = height + pad_b, width + pad_r
        num_h = padded_h // self.window_resolution
        num_w = padded_w // self.window_resolution

        # (B, nH*w, nW*w, C) -> (B, nH, w, nW, w, C) -> (B*nH*nW, w, w, C) -> (.., C, w, w)
        x = (
            x.view(batch_size, num_h, self.window_resolution, num_w, self.window_resolution, channels)
            .transpose(2, 3)
            .reshape(batch_size * num_h * num_w, self.window_resolution, self.window_resolution, channels)
            .permute(0, 3, 1, 2)
        )

        # Attention within every window.
        x = self.attn(x)

        # ---- Window reverse: inverse of the partition above ----
        x = (
            x.permute(0, 2, 3, 1)
            .view(batch_size, num_h, num_w, self.window_resolution, self.window_resolution, channels)
            .transpose(2, 3)
            .reshape(batch_size, padded_h, padded_w, channels)
        )
        if padding:
            x = x[:, :height, :width].contiguous()
        return x.permute(0, 3, 1, 2)  # back to (B, C, H, W)


class EfficientViTBlock(nn.Module):
    r"""
    A basic EfficientViT building block (the "sandwich" layout).

    The block sandwiches the attention token mixer between two local
    feed-forward sub-layers, each piece wrapped in a residual connection:

        x -> Residual(dw 3x3)
          -> Residual(FFN)
          -> Residual(LocalWindowAttention)     # the token mixer
          -> Residual(dw 3x3)
          -> Residual(FFN)

    The depthwise convolutions inject local inductive bias cheaply, while the
    single attention mixer captures longer-range dependencies. The depthwise
    convs here are zero-BN-initialised so each residual branch starts as an
    identity.

    Parameters
    ----------
    embed_dim:
        Channel dimension of the block (preserved end to end).

    key_dim:
        Per-head query/key width in the attention mixer.

    num_heads:
        Number of cascaded attention heads.

    attn_ratio:
        Value-width multiplier in the attention mixer.

    resolution:
        Spatial side length at this stage (fixed; see :class:`LocalWindowAttention`).

    window_resolution:
        Local attention window side length.

    kernels:
        Per-head depthwise kernel sizes for the query enhancement.

    token_mixer:
        Which token mixer to use. Only ``"s"`` (self-attention) is implemented,
        matching the reference EfficientViT.
    """

    def __init__(
        self,
        embed_dim: int,
        key_dim: int,
        num_heads: int = 8,
        attn_ratio: float = 4.0,
        resolution: int = 14,
        window_resolution: int = 7,
        kernels: Sequence[int] = (5, 5, 5, 5),
        token_mixer: str = "s",
    ) -> None:
        super().__init__()
        ensure_positive_int(embed_dim, "embed_dim")

        # Local sub-layer 1: depthwise conv + FFN, both residual.
        self.dw0 = Residual(
            ConvBN(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1,
                   groups=embed_dim, bn_weight_init=0.0)
        )
        self.ffn0 = Residual(FFN(embed_dim, int(embed_dim * 2)))

        # Token mixer.
        if token_mixer == "s":
            self.mixer = Residual(
                LocalWindowAttention(
                    embed_dim, key_dim, num_heads,
                    attn_ratio=attn_ratio,
                    resolution=resolution,
                    window_resolution=window_resolution,
                    kernels=kernels,
                )
            )
        else:
            raise ValueError(
                f"Unsupported token_mixer {token_mixer!r}; only 's' (self-attention) "
                "is implemented."
            )

        # Local sub-layer 2: depthwise conv + FFN, both residual.
        self.dw1 = Residual(
            ConvBN(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1,
                   groups=embed_dim, bn_weight_init=0.0)
        )
        self.ffn1 = Residual(FFN(embed_dim, int(embed_dim * 2)))

    def forward(self, x: Tensor) -> Tensor:
        """
        Input shape:  (B, embed_dim, resolution, resolution)
        Output shape: (B, embed_dim, resolution, resolution)
        """
        return self.ffn1(self.dw1(self.mixer(self.ffn0(self.dw0(x)))))
