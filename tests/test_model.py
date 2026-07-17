import pytest
import torch
from src.models.gpt import GPTModel, GPTConfig

def test_gpt_model_initialization():
    config = GPTConfig(
        block_size=64,
        vocab_size=256,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        bias=True
    )
    model = GPTModel(config)
    assert model is not None
    assert model.get_num_params() > 0

def test_gpt_model_forward_without_targets():
    config = GPTConfig(
        block_size=64,
        vocab_size=256,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        bias=True
    )
    model = GPTModel(config)
    model.eval()
    
    batch_size = 4
    seq_len = 16
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    logits, loss = model(idx)
    assert logits.shape == (batch_size, 1, config.vocab_size)
    assert loss is None

def test_gpt_model_forward_with_targets():
    config = GPTConfig(
        block_size=64,
        vocab_size=256,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        bias=True
    )
    model = GPTModel(config)
    model.eval()
    
    batch_size = 4
    seq_len = 16
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    logits, loss = model(idx, targets)
    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # scalar loss

def test_gpt_model_generate():
    config = GPTConfig(
        block_size=64,
        vocab_size=256,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        bias=True
    )
    model = GPTModel(config)
    model.eval()
    
    batch_size = 2
    seq_len = 8
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    max_new_tokens = 10
    
    # generate without top_k
    generated = model.generate(idx, max_new_tokens=max_new_tokens, temperature=0.8, top_k=None)
    assert generated.shape == (batch_size, seq_len + max_new_tokens)
    
    # generate with top_k
    generated_topk = model.generate(idx, max_new_tokens=max_new_tokens, temperature=0.8, top_k=10)
    assert generated_topk.shape == (batch_size, seq_len + max_new_tokens)

def test_gpt_model_configure_optimizers():
    config = GPTConfig(
        block_size=64,
        vocab_size=256,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        bias=True
    )
    model = GPTModel(config)
    
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=1e-3,
        betas=(0.9, 0.95),
        device_type="cpu"
    )
    
    assert isinstance(optimizer, torch.optim.AdamW)
    assert len(optimizer.param_groups) == 2
    # group 0 should have weight_decay = 0.1, group 1 should have weight_decay = 0.0
    assert optimizer.param_groups[0]['weight_decay'] == 0.1
    assert optimizer.param_groups[1]['weight_decay'] == 0.0
