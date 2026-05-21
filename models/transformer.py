"""
This module provides:
- SelfAttention: multi-head self-attention
- MLP: feed-forward network used after attention
- DropPath: stochastic depth for residual branches
- EncoderBlock: one Transformer encoder block
- TransformerEncoder: a stack of encoder blocks

A Transformer encoder block follows this structure:
1. Layer normalization: normalizes the input sequence before attention.
2. Multi-head self-attention: computes relationships between all tokens using several attention heads.
3. Attention residual connection: adds the attention output back to the original input.
4. Layer normalization: normalizes the updated sequence before the feed-forward network.
5. MLP expansion: expands the embedding dimension using a feed-forward layer.
6. Activation and dropout: applies nonlinearity and regularization.
7. MLP projection: projects the hidden features back to the original embedding dimension.
8. MLP residual connection: adds the MLP output back to the sequence.

Example
-------
>>> import torch
>>> from models.transformer import EncoderBlock
>>>
>>> x = torch.randn(2, 64, 96)  # (batch, sequence_length, embed_dim)
>>> block = EncoderBlock(embed_dim=96, n_heads=4, mlp_ratio=2.0)
>>>
>>> y, attn = block(x, return_attention_weights=True)
>>> y.shape
torch.Size([2, 64, 96])
>>> attn.shape
torch.Size([2, 4, 64, 64])
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


__all__ = [
    "DropPath",
    "SelfAttention",
    "MLP",
    "EncoderBlock",
    "TransformerEncoder",
]


class DropPath(nn.Module):
    """
    Stochastic depth regularization.

    Unlike nn.Dropout, DropPath drops entire residual branches per sample.
    This is commonly used in modern Vision Transformer-style architecture.

    Parameters
    ----------
    drop_prob:
        Probability of dropping the residual branch.
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()

        if not 0.0 <= drop_prob <= 1.0:
            raise ValueError("drop_prob must be between 0 and 1.")

        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply stochastic depth.

        During evaluation, or when drop_prob is zero, the input is returned unchanged.
        """
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)

        random_tensor = keep_prob + torch.rand(
            shape,
            dtype=x.dtype,
            device=x.device,
        )
        random_tensor.floor_()

        return x.div(keep_prob) * random_tensor


class SelfAttention(nn.Module):
    """
    Multi-head self-attention.

    Parameters
    ----------
    embed_dim:
        Input and output embedding dimension.

    n_heads:
        Number of attention heads. embed_dim must be divisible by n_heads.

    attention_dropout:
        Dropout probability applied to the attention probabilities.

    projection_dropout:
        Dropout probability applied after the output projection.

    qkv_bias:
        Whether to use bias in the combined query/key/value projection.

    Notes
    -----
    Input shape: (B, S, E)
    Output shape: (B, S, E)

    where:
    - B = batch size
    - S = sequence length
    - E = embedding dimension
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
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attention_dropout = nn.Dropout(attention_dropout)

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
            If True, return attention probabilities before dropout.

        Returns
        -------
        output:
            Tensor with shape (B, S, E).

        attention_weights:
            Tensor with shape (B, n_heads, S, S) if requested; otherwise None.
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

        qkv = self.qkv(x)
        qkv = qkv.reshape(
            batch_size,
            seq_len,
            3,
            self.n_heads,
            self.head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv.unbind(dim=0)

        attention_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attention_probs = attention_scores.softmax(dim=-1)

        attention_weights = attention_probs if return_attention_weights else None
        attention_probs = self.attention_dropout(attention_probs)

        out = torch.matmul(attention_probs, v)
        out = out.transpose(1, 2).contiguous()
        out = out.reshape(batch_size, seq_len, self.embed_dim)

        out = self.projection(out)
        out = self.projection_dropout(out)

        return out, attention_weights


class MLP(nn.Module):
    """
    Feed-forward network used inside a Transformer encoder block.

    Parameters
    ----------
    embed_dim:
        Input and output embedding dimension.

    mlp_ratio:
        Expansion ratio for the hidden dimension.
        For example, mlp_ratio=4.0 gives hidden_dim = 4 * embed_dim.

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

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.activation = activation_layer()
        self.dropout1 = nn.Dropout(dropout)

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
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.dropout2(x)

        return x


class EncoderBlock(nn.Module):
    """
    Transformer encoder block.

    The block uses the standard pre-normalization structure:

    LayerNorm -> SelfAttention -> residual
    LayerNorm -> MLP -> residual

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
        Dropout probability applied after attention output projection.

    mlp_dropout:
        Dropout probability used inside the MLP.

    drop_path:
        Stochastic depth probability for residual branches.

    qkv_bias:
        Whether to use bias in the query/key/value projection.

    activation_layer:
        Activation layer class used in the MLP.

    norm_layer:
        Normalization layer class.
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

        self.norm1 = norm_layer(embed_dim)
        self.attention = SelfAttention(
            embed_dim=embed_dim,
            n_heads=n_heads,
            attention_dropout=attention_dropout,
            projection_dropout=projection_dropout,
            qkv_bias=qkv_bias,
        )
        self.drop_path1 = DropPath(drop_path)

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
            If True, return attention probabilities from the attention layer.

        Returns
        -------
        output:
            Tensor with shape (B, S, E).

        attention_weights:
            Tensor with shape (B, n_heads, S, S) if requested; otherwise None.
        """
        attention_out, attention_weights = self.attention(
            self.norm1(x),
            return_attention_weights=return_attention_weights,
        )
        x = x + self.drop_path1(attention_out)

        mlp_out = self.mlp(self.norm2(x))
        x = x + self.drop_path2(mlp_out)

        return x, attention_weights


class TransformerEncoder(nn.Module):
    """
    Stack of Transformer encoder blocks.

    Parameters
    ----------
    embed_dim:
        Input and output embedding dimension.

    depth:
        Number of encoder blocks.

    n_heads:
        Number of attention heads.

    mlp_ratio:
        Expansion ratio for the MLP hidden dimension.

    attention_dropout:
        Dropout probability applied to attention probabilities.

    projection_dropout:
        Dropout probability applied after attention output projection.

    mlp_dropout:
        Dropout probability used inside the MLP.

    drop_path:
        Maximum stochastic depth probability. If depth > 1, drop-path values
        are linearly increased from 0 to this value.

    qkv_bias:
        Whether to use bias in the query/key/value projection.

    activation_layer:
        Activation layer class used in the MLP.

    norm_layer:
        Normalization layer class.
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
        Apply all encoder blocks.

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
            List of attention tensors if requested; otherwise None.
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
