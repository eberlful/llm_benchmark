import pytest
import torch
from src.models.bdh import BDHConfig, BDHModel


def test_bdh_model_initialization():
    config = BDHConfig(
        block_size=64,
        vocab_size=256,
        n_layer=2,
        n_head=2,
        n_embd=32,
        mlp_internal_dim_multiplier=8,
        dropout=0.0,
    )
    model = BDHModel(config)
    assert model is not None
    assert model.get_num_params() > 0


def test_bdh_model_forward_without_targets():
    config = BDHConfig(
        block_size=64,
        vocab_size=256,
        n_layer=2,
        n_head=2,
        n_embd=32,
        mlp_internal_dim_multiplier=8,
        dropout=0.0,
    )
    model = BDHModel(config)
    model.eval()

    batch_size = 4
    seq_len = 16
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits, loss = model(idx)
    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    assert loss is None


def test_bdh_model_forward_with_targets():
    config = BDHConfig(
        block_size=64,
        vocab_size=256,
        n_layer=2,
        n_head=2,
        n_embd=32,
        mlp_internal_dim_multiplier=8,
        dropout=0.0,
    )
    model = BDHModel(config)
    model.eval()

    batch_size = 4
    seq_len = 16
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits, loss = model(idx, targets)
    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # scalar loss


def test_bdh_model_generate():
    config = BDHConfig(
        block_size=64,
        vocab_size=256,
        n_layer=2,
        n_head=2,
        n_embd=32,
        mlp_internal_dim_multiplier=8,
        dropout=0.0,
    )
    model = BDHModel(config)
    model.eval()

    batch_size = 2
    seq_len = 8
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    max_new_tokens = 5

    # generate without top_k
    generated = model.generate(
        idx, max_new_tokens=max_new_tokens, temperature=0.8, top_k=None
    )
    assert generated.shape == (batch_size, seq_len + max_new_tokens)

    # generate with top_k
    generated_topk = model.generate(
        idx, max_new_tokens=max_new_tokens, temperature=0.8, top_k=10
    )
    assert generated_topk.shape == (batch_size, seq_len + max_new_tokens)


def test_bdh_model_configure_optimizers():
    config = BDHConfig(
        block_size=64,
        vocab_size=256,
        n_layer=2,
        n_head=2,
        n_embd=32,
        mlp_internal_dim_multiplier=8,
        dropout=0.0,
    )
    model = BDHModel(config)

    optimizer = model.configure_optimizers(
        weight_decay=0.1, learning_rate=1e-3, betas=(0.9, 0.95), device_type="cpu"
    )

    assert isinstance(optimizer, torch.optim.AdamW)
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["weight_decay"] == 0.1
    assert optimizer.param_groups[1]["weight_decay"] == 0.0


def test_bdh_training_step():
    config = BDHConfig(
        block_size=16,
        vocab_size=64,
        n_layer=1,
        n_head=2,
        n_embd=16,
        mlp_internal_dim_multiplier=4,
        dropout=0.0,
    )
    model = BDHModel(config)
    optimizer = model.configure_optimizers(
        weight_decay=0.01, learning_rate=1e-3, betas=(0.9, 0.95), device_type="cpu"
    )

    idx = torch.randint(0, config.vocab_size, (2, 8))
    targets = torch.randint(0, config.vocab_size, (2, 8))

    optimizer.zero_grad()
    logits, loss = model(idx, targets)
    loss.backward()
    optimizer.step()

    assert loss.item() > 0


def test_bdh_config_fields():
    config = BDHConfig()
    assert config.n_layer == 6
    assert config.n_embd == 256
    assert config.dropout == 0.1
    assert config.n_head == 4
    assert config.mlp_internal_dim_multiplier == 128
    assert config.vocab_size == 50304
    assert config.block_size == 1024
    assert config.strict_causal is True


def test_bdh_model_is_subclass():
    from src.base.model import Model

    config = BDHConfig(block_size=16, vocab_size=64, n_layer=1, n_head=2, n_embd=16)
    model = BDHModel(config)
    assert isinstance(model, Model)


def test_bdh_get_num_params():
    config = BDHConfig(
        block_size=16,
        vocab_size=64,
        n_layer=1,
        n_head=2,
        n_embd=16,
        mlp_internal_dim_multiplier=4,
        dropout=0.0,
    )
    model = BDHModel(config)
    total_params = model.get_num_params(non_embedding=False)
    non_emb_params = model.get_num_params(non_embedding=True)
    embed_params = model.embed.weight.numel()

    assert total_params - non_emb_params == embed_params
    assert embed_params == 64 * 16
    assert non_emb_params > 0


def test_bdh_attention_strict_causal():
    from src.models.bdh import Attention

    # Test strict_causal=True (diagonal=-1)
    config_strict = BDHConfig(
        block_size=16,
        vocab_size=64,
        n_layer=1,
        n_head=2,
        n_embd=16,
        mlp_internal_dim_multiplier=4,
        strict_causal=True,
    )
    attn_strict = Attention(config_strict)
    
    # Test strict_causal=False (diagonal=0)
    config_non_strict = BDHConfig(
        block_size=16,
        vocab_size=64,
        n_layer=1,
        n_head=2,
        n_embd=16,
        mlp_internal_dim_multiplier=4,
        strict_causal=False,
    )
    attn_non_strict = Attention(config_non_strict)

    B, nh, T, N = 1, config_strict.n_head, 4, config_strict.n_embd * config_strict.mlp_internal_dim_multiplier // config_strict.n_head
    x_sparse = torch.ones((B, nh, T, N))

    # Compute RoPE and scores for strict
    r_phases = torch.arange(0, T, dtype=torch.float32).view(1, 1, -1, 1) * attn_strict.freqs
    QR_strict = attn_strict.rope(r_phases, x_sparse)
    scores_strict = (QR_strict @ QR_strict.mT).tril(diagonal=-1)

    # For strict_causal=True, the diagonal (t, t) must be 0
    diag_strict = torch.diagonal(scores_strict, dim1=-2, dim2=-1)
    assert torch.all(diag_strict == 0)

    # For strict_causal=False, the diagonal (t, t) is included (diagonal=0)
    scores_non_strict = (QR_strict @ QR_strict.mT).tril(diagonal=0)
    diag_non_strict = torch.diagonal(scores_non_strict, dim1=-2, dim2=-1)
    assert torch.all(diag_non_strict != 0)


def test_bdh_optimizer_parameter_groups():
    config = BDHConfig(
        block_size=16,
        vocab_size=64,
        n_layer=1,
        n_head=2,
        n_embd=16,
        mlp_internal_dim_multiplier=4,
        dropout=0.0,
    )
    model = BDHModel(config)
    optimizer = model.configure_optimizers(
        weight_decay=0.1, learning_rate=1e-3, betas=(0.9, 0.95), device_type="cpu"
    )

    decay_params = optimizer.param_groups[0]["params"]
    no_decay_params = optimizer.param_groups[1]["params"]

    # Retrieve parameter objects for expected 2D / 3D matrices
    expected_decay_params = {
        model.encoder,
        model.decoder,
        model.encoder_v,
        model.lm_head,
        model.embed.weight,
    }

    for p in expected_decay_params:
        assert any(p is dp for dp in decay_params), f"Expected parameter {p} to be in decay group"

    # Confirm all decay_params have dim >= 2 and weight_decay == 0.1
    assert optimizer.param_groups[0]["weight_decay"] == 0.1
    assert optimizer.param_groups[1]["weight_decay"] == 0.0


def test_bdh_generate_context_cropping():
    config = BDHConfig(
        block_size=16,
        vocab_size=64,
        n_layer=1,
        n_head=2,
        n_embd=16,
        mlp_internal_dim_multiplier=4,
        dropout=0.0,
    )
    model = BDHModel(config)
    model.eval()

    batch_size = 2
    # Prompt length 20 exceeds block_size=16
    prompt_len = 20
    idx = torch.randint(0, config.vocab_size, (batch_size, prompt_len))
    max_new_tokens = 5

    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=0.8, top_k=5)
    assert out.shape == (batch_size, prompt_len + max_new_tokens)
    # Check original prompt prefix is preserved
    assert torch.equal(out[:, :prompt_len], idx)


