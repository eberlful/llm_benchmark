import pytest
import torch
import math
from src.models.pam import (
    REAL, IMAG,
    real_part, imag_part, stack_complex, scale_complex,
    cmul, cconj, cabs, cnormalize, to_real_concat,
    fused_decay_matrix,
    ModReLU, ModSwish, PhaseModulatedActivation,
    ComplexLinear, ComplexNorm, ComplexEmbed, ComplexPosEmbed,
    ComplexGatedUnit, V11PAMLayer, V11Block, PAMConfig
)

def test_complex_math_helpers():
    # Setup test tensors
    # a = 3 + 4i, b = 1 - 2i
    a = torch.tensor([3.0, 4.0])
    b = torch.tensor([1.0, -2.0])
    
    assert real_part(a) == 3.0
    assert imag_part(a) == 4.0
    
    # stack_complex
    stacked = stack_complex(torch.tensor(3.0), torch.tensor(4.0))
    assert torch.allclose(stacked, a)
    
    # cmul: (3 + 4i)(1 - 2i) = (3*1 - 4*-2) + i(3*-2 + 4*1) = 11 - 2i
    prod = cmul(a, b)
    assert torch.allclose(prod, torch.tensor([11.0, -2.0]))
    
    # cconj
    conj = cconj(a)
    assert torch.allclose(conj, torch.tensor([3.0, -4.0]))
    
    # cabs: sqrt(3^2 + 4^2) = 5
    assert torch.allclose(cabs(a), torch.tensor(5.0))
    
    # cnormalize: (3/5) + i(4/5)
    normalized = cnormalize(a)
    assert torch.allclose(normalized, torch.tensor([0.6, 0.8]))
    
    # to_real_concat
    # If z shape is [D, 2]
    z = torch.stack([a, b], dim=0) # [2, 2]
    concat = to_real_concat(z) # [4]
    assert torch.allclose(concat, torch.tensor([3.0, 1.0, 4.0, -2.0]))

def test_decay_matrix():
    # 2 heads, seq_len = 3
    decay_gamma = torch.tensor([[0.5, 0.8, 0.9], [0.9, 0.9, 0.9]])
    decay_mat = fused_decay_matrix(decay_gamma, 3)
    
    # Let's check head 0:
    # gamma = [0.5, 0.8, 0.9]
    # decay_mat[t, s] = prod_{j=s+1}^t gamma_j for t >= s
    # t=0, s=0: 1.0
    # t=1, s=0: gamma[1] = 0.8
    # t=1, s=1: 1.0
    # t=2, s=0: gamma[1]*gamma[2] = 0.8 * 0.9 = 0.72
    # t=2, s=1: gamma[2] = 0.9
    # t=2, s=2: 1.0
    expected_head_0 = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.8, 1.0, 0.0],
        [0.72, 0.9, 1.0]
    ])
    assert torch.allclose(decay_mat[0], expected_head_0, atol=1e-4)

def test_activations():
    z = torch.randn(4, 8, 2)
    
    mod_relu = ModReLU(8)
    out_relu = mod_relu(z)
    assert out_relu.shape == (4, 8, 2)
    
    mod_swish = ModSwish(8)
    out_swish = mod_swish(z)
    assert out_swish.shape == (4, 8, 2)
    
    phase_mod = PhaseModulatedActivation(8)
    out_pm = phase_mod(z)
    assert out_pm.shape == (4, 8, 2)

def test_complex_layers():
    x = torch.randn(4, 16, 8, 2) # [B, T, dim, 2]
    
    linear = ComplexLinear(8, 12)
    out_linear = linear(x)
    assert out_linear.shape == (4, 16, 12, 2)
    
    norm = ComplexNorm(8)
    out_norm = norm(x)
    assert out_norm.shape == (4, 16, 8, 2)
    # Check that magnitude is stabilized but phase direction is intact
    mag_in = cabs(x)
    mag_out = cabs(out_norm)
    # Phase preservation check: direction on unit circle is same
    phase_in = x / (mag_in.unsqueeze(-1) + 1e-8)
    phase_out = out_norm / (mag_out.unsqueeze(-1) + 1e-8)
    assert torch.allclose(phase_in, phase_out, atol=1e-4)
    
    embed = ComplexEmbed(100, 8)
    ids = torch.randint(0, 100, (4, 16))
    out_embed = embed(ids)
    assert out_embed.shape == (4, 16, 8, 2)
    
    pos_embed = ComplexPosEmbed(32, 8)
    out_pos = pos_embed(out_embed)
    assert out_pos.shape == (4, 16, 8, 2)

def test_cgu():
    cgu = ComplexGatedUnit(dim=8, expand=3)
    x = torch.randn(4, 16, 8, 2)
    out = cgu(x)
    assert out.shape == (4, 16, 8, 2)

def _verify_equivalence_for_config(cfg, name, atol=2e-3):
    torch.manual_seed(42)
    layer = V11PAMLayer(cfg).eval()
    
    B, T = 2, 32
    # Generate mock inputs
    x = torch.randn(B, T, cfg.dim, 2) * 0.5
    
    with torch.no_grad():
        # Parallel forward pass
        parallel_out, parallel_state = layer(x)
        
        # Recurrent forward pass
        recurrent_steps = []
        state = None
        for t in range(T):
            step_out, state = layer(x[:, t:t+1], state=state, step_offset=t)
            recurrent_steps.append(step_out)
        recurrent_out = torch.cat(recurrent_steps, dim=1)
        
    diff = (parallel_out - recurrent_out).abs().max().item()
    assert diff < atol, f"[{name}] parallel and recurrent forms do not match: max|diff| = {diff:.2e}"

def test_pam_layer_parallel_recurrent_equivalence():
    # We test the equivalence for different configurations corresponding to:
    # 1. Baseline (fused_e3=False/True, n_states=1)
    # 2. E1 (decay_mode='per_channel')
    # 3. E2 (write_mode='delta')
    # 4. E3 (n_states=2, fused_e3=False and True)
    
    configs = {
        "baseline": PAMConfig(dim=16, n_head=2, head_dim=8, max_seq_len=64, chunk_size=8),
        "E1_per_channel": PAMConfig(dim=16, n_head=2, head_dim=8, max_seq_len=64, chunk_size=8, decay_mode='per_channel'),
        "E2_delta": PAMConfig(dim=16, n_head=2, head_dim=8, max_seq_len=64, delta_chunk=8, write_mode='delta'),
        "E3_multistate_unfused": PAMConfig(dim=16, n_head=2, head_dim=8, max_seq_len=64, chunk_size=8, n_states=2, fused_e3=False),
        "E3_multistate_fused": PAMConfig(dim=16, n_head=2, head_dim=8, max_seq_len=64, chunk_size=8, n_states=2, fused_e3=True)
    }
    
    for name, cfg in configs.items():
        _verify_equivalence_for_config(cfg, name)

def test_v11_block():
    cfg = PAMConfig(dim=16, n_head=2, head_dim=8, max_seq_len=64, chunk_size=8)
    block = V11Block(cfg)
    
    x = torch.randn(2, 10, 16, 2)
    out, state = block(x)
    assert out.shape == (2, 10, 16, 2)
