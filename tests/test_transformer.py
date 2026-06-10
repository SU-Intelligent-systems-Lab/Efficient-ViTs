import pytest
import torch

from models.transformer import (
    SelfAttention,
    MLP,
    EncoderBlock,
    TransformerEncoder,
    DropPath,
)


def test_self_attention_output_shape():
    x = torch.randn(2, 64, 96)
    layer = SelfAttention(embed_dim=96, n_heads=4)

    y, attn = layer(x)

    assert y.shape == (2, 64, 96)
    assert attn is None


def test_self_attention_returns_attention_weights():
    x = torch.randn(2, 64, 96)
    layer = SelfAttention(embed_dim=96, n_heads=4)

    y, attn = layer(x, return_attention_weights=True)

    assert y.shape == (2, 64, 96)
    assert attn is not None
    assert attn.shape == (2, 4, 64, 64)


def test_self_attention_invalid_heads():
    with pytest.raises(ValueError):
        SelfAttention(embed_dim=95, n_heads=4)


def test_self_attention_invalid_input_shape():
    x = torch.randn(2, 96)
    layer = SelfAttention(embed_dim=96, n_heads=4)

    with pytest.raises(ValueError):
        layer(x)


def test_mlp_output_shape():
    x = torch.randn(2, 64, 96)
    mlp = MLP(embed_dim=96, mlp_ratio=2.0)

    y = mlp(x)

    assert y.shape == (2, 64, 96)


def test_encoder_block_output_shape():
    x = torch.randn(2, 64, 96)
    block = EncoderBlock(embed_dim=96, n_heads=4, mlp_ratio=2.0)

    y, attn = block(x)

    assert y.shape == (2, 64, 96)
    assert attn is None


def test_encoder_block_returns_attention_weights():
    x = torch.randn(2, 64, 96)
    block = EncoderBlock(embed_dim=96, n_heads=4, mlp_ratio=2.0)

    y, attn = block(x, return_attention_weights=True)

    assert y.shape == (2, 64, 96)
    assert attn is not None
    assert attn.shape == (2, 4, 64, 64)


def test_transformer_encoder_output_shape():
    x = torch.randn(2, 64, 96)
    encoder = TransformerEncoder(
        embed_dim=96,
        depth=3,
        n_heads=4,
        mlp_ratio=2.0,
    )

    y, attn = encoder(x)

    assert y.shape == (2, 64, 96)
    assert attn is None


def test_transformer_encoder_returns_attention_weights():
    x = torch.randn(2, 64, 96)
    encoder = TransformerEncoder(
        embed_dim=96,
        depth=3,
        n_heads=4,
        mlp_ratio=2.0,
    )

    y, attn = encoder(x, return_attention_weights=True)

    assert y.shape == (2, 64, 96)
    assert attn is not None
    assert len(attn) == 3
    assert attn[0].shape == (2, 4, 64, 64)


def test_drop_path_eval_mode_returns_same_tensor():
    x = torch.randn(2, 64, 96)
    drop_path = DropPath(drop_prob=0.5)
    drop_path.eval()

    y = drop_path(x)

    assert torch.equal(x, y)
