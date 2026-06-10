"""
This module provides the MobileViT-style sequence Transformer stack:
- SelfAttention: multi-head scaled dot-product self-attention
- MLP: position-wise feed-forward network
- EncoderBlock: one Transformer encoder block (pre-norm variant)
- TransformerEncoder: a stack of encoder blocks

It also re-exports :class:`DropPath` from :mod:`models.stochastic` for backward
compatibility, since stochastic depth used to live in this file. The canonical
home for ``DropPath`` (and the related ``Residual`` wrapper) is now
``models/stochastic.py`` so that every model family shares one implementation.

This file is intentionally scoped to the *generic sequence Transformer* used by
MobileViT, where attention is computed over a token sequence of shape
``(B, S, E)``. The window/cascaded 2D attention of EfficientViT and the
fixed-resolution token attention of EfficientFormer are different enough that
they live with their own blocks (``efficientvit_block.py`` and
``efficientformer_block.py``) rather than being forced into this module.

Shape conventions
-----------------
Throughout this file:
- B = batch size
- S = sequence length (number of tokens)
- E = embedding dimension
- H = number of attention heads
- D = per-head dimension, with E = H * D

A Transformer encoder block follows this pre-normalization structure:
1. Layer normalization (before attention).
2. Multi-head self-attention.
3. Residual connection over the attention output.
4. Layer normalization (before the MLP).
5. MLP expansion: E -> hidden = E * mlp_ratio.
6. Activation (GELU by default) and dropout.
7. MLP projection: hidden -> E.
8. Residual connection over the MLP output.

Scaled dot-product attention
----------------------------
For queries Q, keys K, values V (each of shape (B, H, S, D)):

    Attention(Q, K, V) = softmax( (Q @ K^T) / sqrt(D) ) @ V

The 1/sqrt(D) scaling keeps the magnitudes of the dot products from
growing with D, which would otherwise push the softmax into very flat
regions with tiny gradients.

Example
-------
>>> import torch
>>> from models.transformer import EncoderBlock
>>>
>>> x = torch.randn(2, 64, 96)  # (B, S, E)
>>> block = EncoderBlock(embed_dim=96, n_heads=4, mlp_ratio=2.0)
>>>
>>> y, attn = block(x, return_attention_weights=True)
>>> y.shape
torch.Size([2, 64, 96])
>>> attn.shape
torch.Size([2, 4, 64, 64])  # (B, H, S, S)
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

# DropPath now lives in the shared stochastic-depth module. It is re-exported
# here so existing imports (``from models.transformer import DropPath``) keep
# working unchanged.
from .stochastic import DropPath


__all__ = [
    "DropPath",
    "SelfAttention",
    "MLP",
    "EncoderBlock",
    "TransformerEncoder",
]


class SelfAttention(nn.Module):
    """
    Multi-head scaled dot-product self-attention.

    The input tokens are linearly projected into queries (Q), keys (K),
    and values (V), split into H heads of dimension D = E / H, and
    attended in parallel:

        Attention(Q, K, V) = softmax( (Q @ K^T) / sqrt(D) ) @ V

    Per-head outputs are concatenated to a single tensor of width E and
    passed through a final linear projection.

    Parameters
    ----------
    embed_dim:
        Input and output embedding dimension E.

    n_heads:
        Number of attention heads H. embed_dim must be divisible by
        n_heads, giving per-head dimension D = E / H.

    attention_dropout:
        Dropout probability applied to the attention probabilities (after
        softmax, before multiplying by V).

    projection_dropout:
        Dropout probability applied after the output linear projection.

    qkv_bias:
        Whether to use bias in the combined Q/K/V linear projection.

    Notes
    -----
    Input shape:  (B, S, E)
    Output shape: (B, S, E)
    Optional attention-weight shape: (B, H, S, S)
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()

        if embed_dim <= 0:
            raise ValueError("embed_dim must be a positive integer.")

        if n_heads <= 0:
            raise ValueError("n_heads must be a positive integer.")

        if embed_dim % n_heads != 0:
            raise ValueError(
                f"embed_dim={embed_dim} must be divisible by n_heads={n_heads}."
            )

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        # Precomputed 1/sqrt(D) scaling factor used in attention.
        self.scale = self.head_dim**-0.5

        # One Linear(E -> 3E) computes the Q, K, and V projections jointly;
        # this is faster than three separate Linear(E -> E) calls.
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attention_dropout = nn.Dropout(attention_dropout)

        # Output projection after the heads are concatenated.
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.projection_dropout = nn.Dropout(projection_dropout)

    def forward(
        self,
        x: Tensor,
        return_attention_weights: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        """
        Apply multi-head self-attention.

        Parameters
        ----------
        x:
            Input tensor with shape (B, S, E).

        return_attention_weights:
            If True, return the post-softmax attention probabilities
            (before attention dropout) with shape (B, H, S, S).

        Returns
        -------
        output:
            Tensor with shape (B, S, E).

        attention_weights:
            Tensor with shape (B, H, S, S) if requested; otherwise None.
        """
        if x.ndim != 3:
            raise ValueError(
                f"SelfAttention expects input shape (B, S, E), but got {tuple(x.shape)}."
            )

        batch_size, seq_len, embed_dim = x.shape

        if embed_dim != self.embed_dim:
            raise ValueError(
                f"Expected embed_dim={self.embed_dim}, but got input with "
                f"last dimension {embed_dim}."
            )

        # Project tokens into Q, K, V jointly.
        # (B, S, E) -> (B, S, 3E)
        qkv = self.qkv(x)

        # Split the last dim into (3, H, D) so we can separate Q/K/V and
        # the attention heads.
        # (B, S, 3E) -> (B, S, 3, H, D)
        qkv = qkv.reshape(
            batch_size,
            seq_len,
            3,
            self.n_heads,
            self.head_dim,
        )

        # Reorder so the leading dim selects Q/K/V and heads come before
        # sequence (each head attends across all tokens independently).
        # (B, S, 3, H, D) -> (3, B, H, S, D)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # Three tensors each of shape (B, H, S, D).
        q, k, v = qkv.unbind(dim=0)

        # Scaled dot-product attention scores per head:
        #   scores[..., i, j] = (q_i . k_j) / sqrt(D)
        # Shapes:
        #   q       : (B, H, S, D)
        #   k.T     : (B, H, D, S)
        #   product : (B, H, S, S)
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Softmax over the keys dimension turns each row into a probability
        # distribution over which tokens this query attends to.
        attention_probs = attention_scores.softmax(dim=-1)

        # Snapshot the probabilities before dropout so the caller sees the
        # actual attention pattern, not the dropped-out version.
        attention_weights = attention_probs if return_attention_weights else None
        attention_probs = self.attention_dropout(attention_probs)

        # Weighted sum over values per head:
        # (B, H, S, S) @ (B, H, S, D) -> (B, H, S, D)
        out = torch.matmul(attention_probs, v)

        # Move heads back next to features and merge into E = H * D.
        # (B, H, S, D) -> (B, S, H, D) -> (B, S, E)
        out = out.transpose(1, 2).contiguous()
        out = out.reshape(batch_size, seq_len, self.embed_dim)

        # Final linear projection across the concatenated heads.
        out = self.projection(out)
        out = self.projection_dropout(out)

        return out, attention_weights


class MLP(nn.Module):
    """
    Position-wise feed-forward network used inside a Transformer block.

    The same two-layer network is applied independently to every token,
    so it can mix features within a token but not across tokens (that is
    what self-attention is for).

    Math
    ----
        h = activation(x W_1 + b_1)   # expand:  E -> hidden
        y = h W_2 + b_2               # project: hidden -> E

    with hidden = embed_dim * mlp_ratio.

    Parameters
    ----------
    embed_dim:
        Input and output embedding dimension E.

    mlp_ratio:
        Expansion ratio for the hidden dimension.
        For example, mlp_ratio=4.0 gives hidden = 4 * E.

    dropout:
        Dropout probability applied after each linear layer.

    activation_layer:
        Activation layer class. Default is nn.GELU.
    """

    def __init__(
        self,
        embed_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        activation_layer: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()

        if embed_dim <= 0:
            raise ValueError("embed_dim must be a positive integer.")

        if mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive.")

        hidden_dim = int(embed_dim * mlp_ratio)

        # Expansion: E -> hidden.
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.activation = activation_layer()
        self.dropout1 = nn.Dropout(dropout)

        # Projection: hidden -> E.
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the MLP.

        Parameters
        ----------
        x:
            Input tensor with shape (B, S, E).

        Returns
        -------
        Tensor
            Output tensor with shape (B, S, E).
        """
        # (B, S, E) -> (B, S, hidden)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)

        # (B, S, hidden) -> (B, S, E)
        x = self.fc2(x)
        x = self.dropout2(x)

        return x


class EncoderBlock(nn.Module):
    """
    Transformer encoder block (pre-norm variant).

    Math
    ----
        x' = x  + DropPath( Attention( LayerNorm(x ) ) )
        y  = x' + DropPath(    MLP   ( LayerNorm(x') ) )

    Pre-norm (normalization applied before each sublayer rather than
    after) is the convention used by recent Vision Transformer-style
    architectures. It gives more stable training than post-norm at
    moderate depths because gradients can flow through the residual path
    without first being passed through a normalization layer.

    Parameters
    ----------
    embed_dim:
        Input and output embedding dimension.

    n_heads:
        Number of attention heads.

    mlp_ratio:
        Expansion ratio for the MLP hidden dimension.

    attention_dropout:
        Dropout probability applied to attention probabilities.

    projection_dropout:
        Dropout probability applied after the attention output projection.

    mlp_dropout:
        Dropout probability used inside the MLP.

    drop_path:
        Stochastic depth probability for the residual branches. The same
        value is used for both the attention and MLP residuals.

    qkv_bias:
        Whether to use bias in the Q/K/V projection.

    activation_layer:
        Activation layer class used in the MLP. Default is nn.GELU.

    norm_layer:
        Normalization layer class. Default is nn.LayerNorm.
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        drop_path: float = 0.0,
        qkv_bias: bool = True,
        activation_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()

        # Attention sublayer: norm -> attention -> stochastic depth.
        self.norm1 = norm_layer(embed_dim)
        self.attention = SelfAttention(
            embed_dim=embed_dim,
            n_heads=n_heads,
            attention_dropout=attention_dropout,
            projection_dropout=projection_dropout,
            qkv_bias=qkv_bias,
        )
        self.drop_path1 = DropPath(drop_path)

        # MLP sublayer: norm -> MLP -> stochastic depth.
        self.norm2 = norm_layer(embed_dim)
        self.mlp = MLP(
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            dropout=mlp_dropout,
            activation_layer=activation_layer,
        )
        self.drop_path2 = DropPath(drop_path)

    def forward(
        self,
        x: Tensor,
        return_attention_weights: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        """
        Apply one Transformer encoder block.

        Parameters
        ----------
        x:
            Input tensor with shape (B, S, E).

        return_attention_weights:
            If True, return attention probabilities from this block.

        Returns
        -------
        output:
            Tensor with shape (B, S, E).

        attention_weights:
            Tensor with shape (B, H, S, S) if requested; otherwise None.
        """
        # Attention sublayer with pre-normalization and residual connection.
        attention_out, attention_weights = self.attention(
            self.norm1(x),
            return_attention_weights=return_attention_weights,
        )
        x = x + self.drop_path1(attention_out)

        # MLP sublayer with pre-normalization and residual connection.
        mlp_out = self.mlp(self.norm2(x))
        x = x + self.drop_path2(mlp_out)

        return x, attention_weights


class TransformerEncoder(nn.Module):
    """
    Stack of Transformer encoder blocks.

    Applies depth identical EncoderBlocks in sequence. The stochastic
    depth probability of each block is interpolated linearly from 0 at
    the first block to drop_path at the last:

        drop_path_i = drop_path * i / (depth - 1)     for i = 0, ..., depth - 1

    This is a common schedule: early layers are not regularized so they
    can learn cleanly, while later layers are regularized more strongly.

    Parameters
    ----------
    embed_dim:
        Input and output embedding dimension.

    depth:
        Number of encoder blocks in the stack.

    n_heads:
        Number of attention heads.

    mlp_ratio:
        Expansion ratio for the MLP hidden dimension.

    attention_dropout:
        Dropout probability applied to attention probabilities.

    projection_dropout:
        Dropout probability applied after the attention output projection.

    mlp_dropout:
        Dropout probability used inside the MLP.

    drop_path:
        Maximum stochastic depth probability. With depth > 1, per-block
        drop-path values are linearly increased from 0 to this value.

    qkv_bias:
        Whether to use bias in the Q/K/V projection.

    activation_layer:
        Activation layer class used in the MLP.

    norm_layer:
        Normalization layer class. Default is nn.LayerNorm.
    """

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        drop_path: float = 0.0,
        qkv_bias: bool = True,
        activation_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()

        if depth <= 0:
            raise ValueError("depth must be a positive integer.")

        # Linear schedule of drop-path rates: 0, ..., drop_path. The first
        # block is never dropped; the last block is dropped at the full
        # configured rate.
        drop_path_rates = torch.linspace(0, drop_path, depth).tolist()

        self.blocks = nn.ModuleList(
            [
                EncoderBlock(
                    embed_dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    attention_dropout=attention_dropout,
                    projection_dropout=projection_dropout,
                    mlp_dropout=mlp_dropout,
                    drop_path=drop_path_rates[i],
                    qkv_bias=qkv_bias,
                    activation_layer=activation_layer,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

    def forward(
        self,
        x: Tensor,
        return_attention_weights: bool = False,
    ) -> tuple[Tensor, list[Tensor] | None]:
        """
        Apply all encoder blocks in sequence.

        Parameters
        ----------
        x:
            Input tensor with shape (B, S, E).

        return_attention_weights:
            If True, return a list of attention maps, one per block.

        Returns
        -------
        output:
            Tensor with shape (B, S, E).

        attention_weights:
            List of attention tensors, each with shape (B, H, S, S), if
            requested; otherwise None.
        """
        all_attention_weights: list[Tensor] = []

        for block in self.blocks:
            x, attention_weights = block(
                x,
                return_attention_weights=return_attention_weights,
            )

            if return_attention_weights and attention_weights is not None:
                all_attention_weights.append(attention_weights)

        if return_attention_weights:
            return x, all_attention_weights

        return x, None
