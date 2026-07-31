import pytest
import torch
from src.models.xlstm import xLSTMConfig, CausalConv1d, mLSTMBlock, sLSTMBlock


def test_xlstm_config_defaults():
    config = xLSTMConfig()
    assert config.vocab_size == 50257
    assert config.n_embd == 768
    assert config.n_layer == 12
    assert config.block_size == 1024
    assert config.use_pos_emb is True


def test_xlstm_config_block_pattern_parsing():
    # Test "mlstm" pattern
    cfg_m = xLSTMConfig(n_layer=4, block_type_pattern="mlstm")
    assert cfg_m.get_block_types() == ["mlstm", "mlstm", "mlstm", "mlstm"]

    # Test "slstm" pattern
    cfg_s = xLSTMConfig(n_layer=4, block_type_pattern="slstm")
    assert cfg_s.get_block_types() == ["slstm", "slstm", "slstm", "slstm"]

    # Test "7:1" pattern
    cfg_71 = xLSTMConfig(n_layer=9, block_type_pattern="7:1")
    expected_71 = [
        "mlstm", "mlstm", "mlstm", "mlstm", "mlstm", "mlstm", "mlstm", "slstm", "mlstm"
    ]
    assert cfg_71.get_block_types() == expected_71

    # Test explicit list pattern
    custom_pattern = ["mlstm", "slstm", "mlstm", "slstm"]
    cfg_list = xLSTMConfig(n_layer=4, block_type_pattern=custom_pattern)
    assert cfg_list.get_block_types() == custom_pattern


def test_causal_conv1d_shape_and_causality():
    batch_size = 2
    seq_len = 16
    channels = 32
    kernel_size = 4

    conv = CausalConv1d(channels=channels, kernel_size=kernel_size)
    x = torch.randn(batch_size, seq_len, channels)
    out = conv(x)

    assert out.shape == (batch_size, seq_len, channels)

    # Test causality: changing x[:, 5:, :] should not affect out[:, :5, :]
    x_mod = x.clone()
    x_mod[:, 5:, :] += 10.0
    out_mod = conv(x_mod)

    torch.testing.assert_close(out[:, :5, :], out_mod[:, :5, :])


def test_mlstm_block_forward():
    config = xLSTMConfig(n_embd=64, num_heads=4, block_size=32)
    block = mLSTMBlock(config)
    x = torch.randn(2, 16, 64)
    out = block(x)

    assert out.shape == (2, 16, 64)


def test_slstm_block_forward():
    config = xLSTMConfig(n_embd=64, num_heads=4, block_size=32)
    block = sLSTMBlock(config)
    x = torch.randn(2, 16, 64)
    out = block(x)

    assert out.shape == (2, 16, 64)
