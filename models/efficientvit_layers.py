"""
EfficientViT-specific low-level layers.

This module holds the convolutional building blocks that are particular to
EfficientViT (Liu et al., CVPR 2023, "EfficientViT: Memory Efficient Vision
Transformer with Cascaded Group Attention"). The generic ``ConvBN`` /
``SqueezeExcite`` primitives live in :mod:`models.layers`; the attention
machinery lives in :mod:`models.efficientvit_block`. What is left here are the
pieces unique to EfficientViT's macro design:

- ``PatchEmbed``  : the overlapping-conv patch embedding stem (16x downsample).
- ``FFN``         : the conv feed-forward network used as a residual sub-layer.
- ``PatchMerging``: the strided, SE-gated downsampling block between stages.
- ``EfficientViTSubsample``: the full inter-stage transition
  (local block -> patch merging -> local block).

Design notes
------------
EfficientViT keeps *every* tensor in the spatial ``(B, C, H, W)`` layout and
uses BatchNorm everywhere (no LayerNorm), so all of these layers are built from
``ConvBN``. Two BatchNorm-initialisation conventions appear:

- ``bn_weight_init=1`` (default): a normal conv-bn.
- ``bn_weight_init=0``: used for the *last* BN of a residual branch so the
  branch starts as a no-op (``x + 0``). EfficientViT applies this to the
  second conv of every :class:`FFN`, which is why ``FFN`` sets it explicitly.

Example
-------
>>> import torch
>>> from models.efficientvit_layers import PatchEmbed, PatchMerging
>>>
>>> x = torch.randn(2, 3, 224, 224)
>>> stem = PatchEmbed(in_channels=3, embed_dim=64)
>>> feat = stem(x)                 # 224 / 16 = 14
>>> feat.shape
torch.Size([2, 64, 14, 14])
>>> merge = PatchMerging(dim=64, out_dim=128)
>>> merge(feat).shape              # stride-2 downsample
torch.Size([2, 128, 7, 7])
"""

from __future__ import annotations

from torch import Tensor, nn

from .common import ensure_positive_int
from .layers import ConvBN, SqueezeExcite
from .stochastic import Residual


__all__ = [
    "PatchEmbed",
    "FFN",
    "PatchMerging",
    "EfficientViTSubsample",
]


class PatchEmbed(nn.Sequential):
    """
    EfficientViT patch-embedding stem: four 3x3 stride-2 convolutions.

    Unlike a standard ViT, which tokenises with a single large-kernel
    non-overlapping convolution, EfficientViT uses a stack of four small
    overlapping strided convolutions. Each halves the spatial resolution, so
    the stem downsamples by a factor of ``2^4 = 16`` overall, matching the
    ``patch_size=16`` of the ViT it replaces while being much cheaper and more
    expressive at high resolution.

    Channel schedule (with ``embed_dim = C``):

        in_channels -> C/8 -> C/4 -> C/2 -> C

    Spatial schedule (with input ``H``):

        H -> H/2 -> H/4 -> H/8 -> H/16

    Parameters
    ----------
    in_channels:
        Number of input image channels (3 for RGB).

    embed_dim:
        Output channel dimension ``C`` of the stem (stage-1 width). Must be
        divisible by 8 so the intermediate widths ``C/8, C/4, C/2`` are
        integers.
    """

    def __init__(self, in_channels: int, embed_dim: int) -> None:
        ensure_positive_int(in_channels, "in_channels")
        ensure_positive_int(embed_dim, "embed_dim")
        if embed_dim % 8 != 0:
            raise ValueError(
                f"embed_dim must be divisible by 8, got {embed_dim} "
                "(the stem uses widths embed_dim/8, /4, /2)."
            )

        super().__init__(
            # H -> H/2, channels in -> C/8
            ConvBN(in_channels, embed_dim // 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # H/2 -> H/4, C/8 -> C/4
            ConvBN(embed_dim // 8, embed_dim // 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # H/4 -> H/8, C/4 -> C/2
            ConvBN(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # H/8 -> H/16, C/2 -> C
            ConvBN(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
        )


class FFN(nn.Module):
    """
    Convolutional feed-forward network (point-wise expand -> ReLU -> project).

    EfficientViT's FFN is two 1x1 convolutions with a ReLU in between, the
    convolutional analogue of a Transformer MLP operating on a ``(B, C, H, W)``
    feature map:

        x -> ConvBN(C -> hidden) -> ReLU -> ConvBN(hidden -> C, bn_weight_init=0)

    The final BatchNorm is initialised with weight 0 so that, when this FFN is
    wrapped in a :class:`~models.stochastic.Residual`, the whole branch starts
    as an identity mapping.

    Parameters
    ----------
    dim:
        Input and output channel dimension ``C``.

    hidden_dim:
        Hidden (expanded) channel dimension. Typically ``2 * dim`` in
        EfficientViT.
    """

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        ensure_positive_int(dim, "dim")
        ensure_positive_int(hidden_dim, "hidden_dim")

        self.pw1 = ConvBN(dim, hidden_dim)
        self.act = nn.ReLU()
        # bn_weight_init=0 makes the residual branch start at zero.
        self.pw2 = ConvBN(hidden_dim, dim, bn_weight_init=0.0)

    def forward(self, x: Tensor) -> Tensor:
        """
        Input shape:  (B, C, H, W)
        Output shape: (B, C, H, W)
        """
        return self.pw2(self.act(self.pw1(x)))


class PatchMerging(nn.Module):
    """
    Strided, SE-gated downsampling block between EfficientViT stages.

    PatchMerging halves the spatial resolution and changes the channel count
    from ``dim`` to ``out_dim``. It follows an inverted-bottleneck shape with a
    Squeeze-and-Excitation gate on the wide hidden tensor:

        x -> ConvBN(dim -> 4*dim, 1x1)            # expand
          -> ReLU
          -> ConvBN(4*dim -> 4*dim, 3x3 s2, dw)   # depthwise, stride 2 (H,W /2)
          -> ReLU
          -> SqueezeExcite(4*dim, 0.25)           # channel attention
          -> ConvBN(4*dim -> out_dim, 1x1)        # project

    The 4x expansion before the depthwise conv gives the SE block a wide
    representation to recalibrate, which the paper found important for
    information-preserving downsampling.

    Parameters
    ----------
    dim:
        Input channel dimension.

    out_dim:
        Output channel dimension after merging.
    """

    def __init__(self, dim: int, out_dim: int) -> None:
        super().__init__()
        ensure_positive_int(dim, "dim")
        ensure_positive_int(out_dim, "out_dim")

        hidden_dim = int(dim * 4)
        self.conv1 = ConvBN(dim, hidden_dim, kernel_size=1)
        self.act = nn.ReLU()
        # Depthwise (groups=hidden_dim), 3x3, stride 2 -> halves H and W.
        self.conv2 = ConvBN(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=hidden_dim,
        )
        self.se = SqueezeExcite(hidden_dim, rd_ratio=0.25)
        self.conv3 = ConvBN(hidden_dim, out_dim, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Input shape:  (B, dim,     H,   W  )
        Output shape: (B, out_dim, H/2, W/2)
        """
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.se(x)
        x = self.conv3(x)
        return x


class EfficientViTSubsample(nn.Module):
    """
    Full inter-stage transition: local block -> patch merging -> local block.

    Between two EfficientViT stages the resolution is halved and the width is
    increased. EfficientViT surrounds the :class:`PatchMerging` step with a
    "local block" (a residual depthwise conv followed by a residual FFN) on
    each side -- one at the old width/resolution and one at the new
    width/resolution:

        [Residual(dw 3x3) -> Residual(FFN)]   at dim=in_dim
        -> PatchMerging(in_dim -> out_dim)    (H, W halved)
        [Residual(dw 3x3) -> Residual(FFN)]   at dim=out_dim

    Note on BatchNorm init: the depthwise convs *here* use the default
    ``bn_weight_init=1`` (matching the reference implementation), whereas the
    depthwise convs inside :class:`~models.efficientvit_block.EfficientViTBlock`
    use ``bn_weight_init=0``. This subtle difference is preserved deliberately.

    Parameters
    ----------
    in_dim:
        Channel dimension entering the transition.

    out_dim:
        Channel dimension leaving the transition.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        ensure_positive_int(in_dim, "in_dim")
        ensure_positive_int(out_dim, "out_dim")

        # Local block at the old (in_dim) width and resolution.
        self.pre = nn.Sequential(
            Residual(ConvBN(in_dim, in_dim, kernel_size=3, stride=1, padding=1, groups=in_dim)),
            Residual(FFN(in_dim, int(in_dim * 2))),
        )
        # Strided SE downsample: in_dim -> out_dim, H,W halved.
        self.merge = PatchMerging(in_dim, out_dim)
        # Local block at the new (out_dim) width and resolution.
        self.post = nn.Sequential(
            Residual(ConvBN(out_dim, out_dim, kernel_size=3, stride=1, padding=1, groups=out_dim)),
            Residual(FFN(out_dim, int(out_dim * 2))),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Input shape:  (B, in_dim,  H,   W  )
        Output shape: (B, out_dim, H/2, W/2)
        """
        x = self.pre(x)
        x = self.merge(x)
        x = self.post(x)
        return x
