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


def test_xlstm_model_forward_and_loss():
    from src.models.xlstm import xLSTMModel

    config = xLSTMConfig(
        vocab_size=100,
        n_embd=64,
        n_layer=2,
        block_size=32,
        num_heads=4,
        block_type_pattern="7:1",
    )
    model = xLSTMModel(config)

    idx = torch.randint(0, 100, (2, 16))
    targets = torch.randint(0, 100, (2, 16))

    # Test forward pass with targets
    logits, loss = model(idx, targets)
    assert logits.shape == (2, 16, 100)
    assert loss is not None
    assert loss.item() > 0

    # Test backward pass
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient"

    # Test forward pass without targets (inference)
    model.zero_grad()
    logits_inf, loss_inf = model(idx)
    assert logits_inf.shape == (2, 1, 100)
    assert loss_inf is None


def test_xlstm_model_without_pos_emb():
    from src.models.xlstm import xLSTMModel

    config = xLSTMConfig(
        vocab_size=100,
        n_embd=64,
        n_layer=2,
        block_size=32,
        num_heads=4,
        use_pos_emb=False,
    )
    model = xLSTMModel(config)
    idx = torch.randint(0, 100, (2, 16))
    logits, loss = model(idx, idx)
    assert logits.shape == (2, 16, 100)


def test_xlstm_model_configure_optimizers():
    from src.models.xlstm import xLSTMModel

    config = xLSTMConfig(vocab_size=100, n_embd=64, n_layer=2, block_size=32)
    model = xLSTMModel(config)

    optimizer = model.configure_optimizers(
        weight_decay=0.1, learning_rate=1e-3, betas=(0.9, 0.95), device_type="cpu"
    )
    assert isinstance(optimizer, torch.optim.AdamW)
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["weight_decay"] == 0.1
    assert optimizer.param_groups[1]["weight_decay"] == 0.0


def test_mlstm_block_step():
    config = xLSTMConfig(n_embd=64, num_heads=4, block_size=32)
    block = mLSTMBlock(config)
    block.eval()

    x_seq = torch.randn(2, 5, 64)
    # Sequence forward
    out_seq = block(x_seq)

    # Step-by-step forward
    state = None
    step_outputs = []
    for t in range(5):
        x_t = x_seq[:, t, :]
        out_t, state = block.step(x_t, state)
        step_outputs.append(out_t.unsqueeze(1))

    out_steps = torch.cat(step_outputs, dim=1)
    assert out_steps.shape == (2, 5, 64)


def test_slstm_block_step():
    config = xLSTMConfig(n_embd=64, num_heads=4, block_size=32)
    block = sLSTMBlock(config)
    block.eval()

    x_seq = torch.randn(2, 5, 64)
    # Sequence forward
    out_seq = block(x_seq)

    # Step-by-step forward
    state = None
    step_outputs = []
    for t in range(5):
        x_t = x_seq[:, t, :]
        out_t, state = block.step(x_t, state)
        step_outputs.append(out_t.unsqueeze(1))

    out_steps = torch.cat(step_outputs, dim=1)
    assert out_steps.shape == (2, 5, 64)


def test_xlstm_model_generate():
    from src.models.xlstm import xLSTMModel

    config = xLSTMConfig(
        vocab_size=100,
        n_embd=64,
        n_layer=2,
        block_size=32,
        num_heads=4,
        block_type_pattern="7:1",
    )
    model = xLSTMModel(config)
    model.eval()

    idx = torch.randint(0, 100, (2, 8))
    max_new_tokens = 6

    generated = model.generate(idx, max_new_tokens=max_new_tokens, temperature=1.0, top_k=10)
    assert generated.shape == (2, 8 + max_new_tokens)
    assert (generated[:, :8] == idx).all()


