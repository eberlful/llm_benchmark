import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

# ── Split-real vocabulary ─────────────────────────────────────────────────────

REAL = 0  # last-axis index for the real part of [..., dim, 2]
IMAG = 1  # last-axis index for the imaginary part


def real_part(z: torch.Tensor) -> torch.Tensor:
    """Real slice of a split-real complex tensor."""
    return z[..., REAL]


def imag_part(z: torch.Tensor) -> torch.Tensor:
    """Imaginary slice of a split-real complex tensor."""
    return z[..., IMAG]


def stack_complex(real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
    """Pack real and imag into split-real layout [..., 2]."""
    return torch.stack([real, imag], dim=-1)


def scale_complex(z: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Multiply complex z by a real scale that has no complex axis.

    Appends trailing dims to `scale` until it broadcasts over z, so the same
    factor applies to real and imag together (dropout, protect gate, etc.).
    """
    while scale.dim() < z.dim():
        scale = scale.unsqueeze(-1)
    return z * scale


def as_complex_dropout_mask(dropout_module: nn.Dropout, like_z: torch.Tensor) -> torch.Tensor:
    """Bernoulli keep-mask over every axis except the complex last dim.

    nn.Dropout on complex tensors would drop real/imag independently; we build a
    mask on shape[:-1] and apply it with scale_complex so a token is kept or
    dropped as one complex value.
    """
    return dropout_module(torch.ones(like_z.shape[:-1], device=like_z.device, dtype=like_z.dtype))


# ── Complex Arithmetic (split-real: [..., dim, 2]) ───────────────────────────

def cmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Complex multiplication: (a_r + i·a_i)(b_r + i·b_i)"""
    return torch.stack([
        a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1],
        a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0],
    ], dim=-1)


def cconj(x: torch.Tensor) -> torch.Tensor:
    """Complex conjugate: real - i * imag"""
    return torch.stack([x[..., 0], -x[..., 1]], dim=-1)


def cabs(x: torch.Tensor) -> torch.Tensor:
    """Complex magnitude: sqrt(real^2 + imag^2 + epsilon)"""
    return torch.sqrt(x[..., 0].square() + x[..., 1].square() + 1e-8)


def cnormalize(x: torch.Tensor) -> torch.Tensor:
    """Normalize complex tensor to unit magnitude."""
    return x / cabs(x).unsqueeze(-1)


def to_real_concat(x: torch.Tensor) -> torch.Tensor:
    """Concatenate real and imaginary components along the last dimension."""
    return torch.cat([x[..., 0], x[..., 1]], dim=-1)


def fused_decay_matrix(decay_gamma: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Pure PyTorch implementation of the log-cumsum-sub-mask-exp decay matrix computation.

    decay_gamma has shape [batch_size * num_heads, seq_len] or similar.
    Returns decay matrix of shape [batch_size * num_heads, seq_len, seq_len] where
    decay_matrix[t, s] = prod_{j=s+1}^{t} gamma_j if t >= s else 0.
    """
    log_gamma = torch.log(decay_gamma + 1e-6)
    cum_neg_log_gamma = torch.cumsum(-log_gamma, dim=-1)
    log_decay = (cum_neg_log_gamma.unsqueeze(-1) - cum_neg_log_gamma.unsqueeze(-2)).transpose(-1, -2)
    causal = torch.tril(torch.ones(seq_len, seq_len, device=decay_gamma.device))
    log_decay = log_decay * causal + (1 - causal) * (-1e4)
    return torch.exp(log_decay.clamp(max=0.0))


# ── Complex Modules & Activations ───────────────────────────────────────────

class ModReLU(nn.Module):
    """Phase-preserving activation: threshold on magnitude, phase untouched."""

    def __init__(self, dim: int):
        super().__init__()
        self.bias = nn.Parameter(torch.full((dim,), -0.1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mag = torch.sqrt(z[..., 0].square() + z[..., 1].square() + 1e-8)
        activated = F.relu(mag + self.bias)
        phase = z / (mag.unsqueeze(-1) + 1e-8)
        return phase * activated.unsqueeze(-1)


class ModSwish(nn.Module):
    """Smooth phase-preserving activation: Swish on magnitude, phase untouched."""

    def __init__(self, dim: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.ones(dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mag = torch.sqrt(z[..., 0].square() + z[..., 1].square() + 1e-8)
        activated = mag * torch.sigmoid(self.beta * mag + self.bias)
        phase = z / (mag.unsqueeze(-1) + 1e-8)
        return phase * activated.unsqueeze(-1)


class PhaseModulatedActivation(nn.Module):
    """Activation that couples magnitude and phase."""

    def __init__(self, dim: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.ones(dim))
        self.phase_alpha = nn.Parameter(torch.zeros(dim))
        self.phase_beta = nn.Parameter(torch.zeros(dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        magnitude = cabs(z)
        activated_magnitude = magnitude * torch.sigmoid(self.beta * magnitude + self.bias)
        # preserve direction on the unit circle before scaling by activated magnitude
        phase = z / (magnitude.unsqueeze(-1) + 1e-8)
        rotation_angle = self.phase_alpha * magnitude + self.phase_beta
        rotation = torch.stack([rotation_angle.cos(), rotation_angle.sin()], dim=-1)
        phase = cmul(phase, rotation)
        return phase * activated_magnitude.unsqueeze(-1)


def _build_activation(name: str, dim: int) -> nn.Module:
    if name == 'swish':
        return ModSwish(dim)
    elif name == 'phase_mod':
        return PhaseModulatedActivation(dim)
    return ModReLU(dim)


class ComplexLinear(nn.Module):
    """Complex linear via split real/imag matmuls with orthogonal init."""

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        init_scale = (2 / (in_dim + out_dim)) ** 0.5
        self.weight_real = nn.Parameter(torch.empty(out_dim, in_dim))
        self.weight_imag = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.orthogonal_(self.weight_real, gain=init_scale)
        nn.init.orthogonal_(self.weight_imag, gain=init_scale)
        if bias:
            self.bias_real = nn.Parameter(torch.zeros(out_dim))
            self.bias_imag = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real_in, imag_in = x[..., 0], x[..., 1]
        real_out = F.linear(real_in, self.weight_real) - F.linear(imag_in, self.weight_imag)
        imag_out = F.linear(real_in, self.weight_imag) + F.linear(imag_in, self.weight_real)
        if self.bias_real is not None:
            real_out = real_out + self.bias_real
            imag_out = imag_out + self.bias_imag
        return torch.stack([real_out, imag_out], dim=-1)


class ComplexNorm(nn.Module):
    """Stabilize complex token vectors without rotating their phase."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mag = torch.sqrt(z[..., 0].square() + z[..., 1].square() + 1e-8)
        rms = torch.sqrt(mag.square().mean(dim=-1, keepdim=True) + self.eps)
        scaled = (mag / rms) * self.scale
        phase = z / (mag.unsqueeze(-1) + 1e-8)
        return phase * scaled.unsqueeze(-1)


class ComplexEmbed(nn.Module):
    """Embed tokens into complex space: real + imaginary components."""

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.dim = dim
        self.embed_real = nn.Embedding(vocab_size, dim)
        self.embed_imag = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.embed_real.weight, std=0.02)
        nn.init.normal_(self.embed_imag.weight, std=0.02)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.embed_real(ids), self.embed_imag(ids)], dim=-1)


class ComplexPosEmbed(nn.Module):
    """Learned absolute position embed added to token embed before the stack."""

    def __init__(self, max_seq_len: int, dim: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.pos_embed = nn.Embedding(max_seq_len, dim)
        nn.init.normal_(self.pos_embed.weight, std=0.02)

    def forward(self, z: torch.Tensor, step_offset: int = 0) -> torch.Tensor:
        # z: [B, T, dim, 2]
        seq_len = z.shape[1]
        position_end = step_offset + seq_len
        if position_end > self.max_seq_len:
            raise ValueError(
                f"Position range [{step_offset}, {position_end}) exceeds max_seq_len "
                f"{self.max_seq_len}"
            )
        position_ids = torch.arange(step_offset, position_end, device=z.device)
        position_embed = self.pos_embed(position_ids)  # [T, dim]
        return z + position_embed.unsqueeze(0).unsqueeze(-1)


def build_rope_cache(max_len: int, head_dim: int) -> torch.Tensor:
    """Complex RoPE: e^{i·m·theta_k} for positions m and frequency bands k."""
    inverse_freqs = 1.0 / (10000.0 ** (torch.arange(head_dim).float() / head_dim))
    positions = torch.arange(max_len).float()
    angles = positions.unsqueeze(1) * inverse_freqs.unsqueeze(0)
    return torch.stack([angles.cos(), angles.sin()], dim=-1)


class ComplexGatedUnit(nn.Module):
    """Channel mixer inside one token (not across time)."""

    def __init__(self, dim: int, expand: int = 3, activation: str = 'modrelu'):
        super().__init__()
        hidden_dim = dim * expand
        self.gate_proj = ComplexLinear(dim, hidden_dim, bias=False)
        self.up_proj = ComplexLinear(dim, hidden_dim, bias=False)
        self.down_proj = ComplexLinear(hidden_dim, dim, bias=False)
        self.act = _build_activation(activation, hidden_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(z)
        up = self.act(self.up_proj(z))
        gmag = torch.sqrt(gate[..., 0].square() + gate[..., 1].square() + 1e-8)
        gate_mag = torch.sigmoid(gmag)
        phase = gate / (gmag.unsqueeze(-1) + 1e-8)
        pr, pi = phase[..., 0], phase[..., 1]
        ur, ui = up[..., 0], up[..., 1]
        out_r = (pr * ur - pi * ui) * gate_mag
        out_i = (pr * ui + pi * ur) * gate_mag
        gated = torch.stack([out_r, out_i], dim=-1)
        return self.down_proj(gated)


def _complex_triangular_solve(mass_real, mass_imag, write_real, write_imag, identity):
    """Solve (I + M) update = write for complex update, M strictly lower-tri."""
    chunk_len = mass_real.shape[-1]
    system_real = (identity + mass_real).float()
    system_imag = mass_imag.float()
    top = torch.cat([system_real, -system_imag], dim=-1)
    bot = torch.cat([system_imag, system_real], dim=-1)
    system_matrix = torch.cat([top, bot], dim=-2)
    rhs = torch.cat([write_real.float(), write_imag.float()], dim=-2)
    solution = torch.linalg.solve(system_matrix, rhs)
    update_real, update_imag = solution[..., :chunk_len, :], solution[..., chunk_len:, :]
    return update_real.to(write_real.dtype), update_imag.to(write_imag.dtype)


# ── Config Class ─────────────────────────────────────────────────────────────

@dataclass
class PAMConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    head_dim: int = 64
    dropout: float = 0.0
    bias: bool = True

    # Internal mappings or overrides
    max_seq_len: int = None
    dim: int = None
    n_layers: int = None
    n_heads: int = None

    # Pluggable parameters
    expand: int = 3
    use_rope: bool = True
    use_gsp: bool = True
    fused_qkv: bool = True
    qk_norm: bool = False
    activation: str = 'swish'
    decay_mode: str = 'head'
    write_mode: str = 'additive'
    n_states: int = 1
    chunk_size: int = 256
    delta_chunk: int = 64
    base_dt_bias: float = -4.0
    protect_gate_bias: float = -3.0
    gamma_floor: float = 0.0
    gate_surprisal_lambda: float = 0.0
    gate_surprisal_tau: float = 1.0
    gate_surprisal_sign: float = 1.0
    fused_e3: bool = True
    recompute_pam_chunks: bool = False

    def __post_init__(self):
        if self.max_seq_len is None:
            self.max_seq_len = self.block_size
        if self.dim is None:
            self.dim = self.n_embd
        if self.n_layers is None:
            self.n_layers = self.n_layer
        if self.n_heads is None:
            self.n_heads = self.n_head


# ── Phase-Associative Memory Layer ────────────────────────────────────────────

class V11PAMLayer(nn.Module):
    """Matrix-state memory with complex-conjugate retrieval and pluggable dynamics."""

    def __init__(self, cfg: PAMConfig, layer_idx: int = 0):
        super().__init__()
        self.num_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        inner = cfg.n_heads * cfg.head_dim
        self.inner_dim = inner
        self.dim = cfg.dim
        self.fused_qkv = cfg.fused_qkv
        self.use_rope = cfg.use_rope
        self.use_gsp = cfg.use_gsp
        self.qk_norm = cfg.qk_norm
        self.decay_mode = cfg.decay_mode
        self.write_mode = cfg.write_mode
        self.n_states = cfg.n_states
        self.delta_chunk = cfg.delta_chunk
        self.fused_e3 = getattr(cfg, 'fused_e3', True)
        self.recompute_pam_chunks = getattr(cfg, 'recompute_pam_chunks', False)

        if cfg.fused_qkv:
            self.qkv_proj = ComplexLinear(cfg.dim, 3 * inner, bias=False)
        else:
            self.q_proj = ComplexLinear(cfg.dim, inner, bias=False)
            self.k_proj = ComplexLinear(cfg.dim, inner, bias=False)
            self.v_proj = ComplexLinear(cfg.dim, inner, bias=False)
        self.o_proj = ComplexLinear(inner, cfg.dim, bias=False)

        # Decay projection
        decay_out = cfg.n_heads * (cfg.head_dim if cfg.decay_mode == 'per_channel' else 1)
        self.dt_proj = nn.Linear(cfg.dim * 2, decay_out)
        if cfg.decay_mode == 'per_channel':
            self.dt_bias = nn.Parameter(torch.zeros(cfg.n_heads, cfg.head_dim) + cfg.base_dt_bias)
        else:
            self.dt_bias = nn.Parameter(torch.zeros(cfg.n_heads) + cfg.base_dt_bias)

        self.gate_content_aware = getattr(cfg, 'gate_content_aware', False)
        self.protect_gate_bias = getattr(cfg, 'protect_gate_bias', -3.0)
        self.routing_content_aware = getattr(cfg, 'routing_content_aware', False)
        self.state_compete = getattr(cfg, 'state_compete', False)
        self.phase_init = getattr(cfg, 'phase_init', 'zero')
        self.route_balance_lambda = getattr(cfg, 'route_balance_lambda', 0.0)
        self.gamma_floor = getattr(cfg, 'gamma_floor', 0.0)
        self.gate_surprisal_lambda = getattr(cfg, 'gate_surprisal_lambda', 0.0)
        
        if cfg.use_gsp:
            gate_in = cfg.dim * 2 if self.gate_content_aware else cfg.dim
            self.protect_gate = nn.Linear(gate_in, cfg.n_heads)
            nn.init.constant_(self.protect_gate.bias, self.protect_gate_bias)

        if cfg.write_mode == 'delta':
            self.beta_proj = nn.Linear(cfg.dim, cfg.n_heads)
            nn.init.constant_(self.beta_proj.bias, 0.0)

        if cfg.n_states > 1:
            state_dt_spread = getattr(cfg, 'state_dt_spread', 2.0)
            offs = torch.linspace(-state_dt_spread, state_dt_spread, cfg.n_states)
            self.state_dt_offset = nn.Parameter(offs.clone())
            route_in = cfg.dim * 2 if self.routing_content_aware else cfg.dim
            self.phase_proj = nn.Linear(route_in, cfg.n_heads * cfg.n_states)
            if self.state_compete:
                self.score_proj = nn.Linear(route_in, cfg.n_heads * cfg.n_states)
                nn.init.zeros_(self.score_proj.weight)
                nn.init.zeros_(self.score_proj.bias)
            self._init_phase_proj()

        if cfg.use_rope:
            self.register_buffer(
                'rope_cache',
                build_rope_cache(cfg.max_seq_len, cfg.head_dim),
                persistent=False,
            )

        self.dropout = nn.Dropout(cfg.dropout)
        self.chunk_size = cfg.chunk_size
        _causal_size = cfg.chunk_size if cfg.chunk_size > 0 else cfg.max_seq_len
        self.register_buffer(
            '_causal',
            torch.tril(torch.ones(_causal_size, _causal_size)),
            persistent=False,
        )
        self._route_aux = None
        self._gate_prob_bt = None

    def _apply_gamma_floor(self, base_decay: torch.Tensor) -> torch.Tensor:
        if self.gamma_floor and self.gamma_floor > 0.0:
            return self.gamma_floor + (1.0 - self.gamma_floor) * base_decay
        return base_decay

    def _init_phase_proj(self):
        if self.n_states <= 1:
            return
        num_memory_states = self.n_states
        num_heads = self.num_heads
        if self.phase_init == 'spread':
            nn.init.zeros_(self.phase_proj.weight)
            biases = torch.zeros(num_heads * num_memory_states)
            for head_idx in range(num_heads):
                for state_idx in range(num_memory_states):
                    if num_memory_states == 3:
                        biases[head_idx * num_memory_states + state_idx] = [0.0, 2 * math.pi / 3, -2 * math.pi / 3][state_idx]
                    else:
                        biases[head_idx * num_memory_states + state_idx] = state_idx * 2 * math.pi / num_memory_states
            with torch.no_grad():
                self.phase_proj.bias.copy_(biases)
        elif self.phase_init == 'ortho':
            nn.init.orthogonal_(self.phase_proj.weight)
            nn.init.zeros_(self.phase_proj.bias)
        else:
            nn.init.zeros_(self.phase_proj.weight)
            nn.init.zeros_(self.phase_proj.bias)

    def _routing_input(self, x: torch.Tensor) -> torch.Tensor:
        return to_real_concat(x) if self.routing_content_aware else cabs(x)

    def _phase_and_alpha(self, x: torch.Tensor):
        batch_size, seq_len = x.shape[0], x.shape[1]
        num_heads, num_memory_states = self.num_heads, self.n_states
        routing_input = self._routing_input(x)
        retrieval_phase = self.phase_proj(routing_input).view(
            batch_size, seq_len, num_heads, num_memory_states
        )
        if self.state_compete:
            routing_scores = self.score_proj(routing_input).view(
                batch_size, seq_len, num_heads, num_memory_states
            )
            routing_weights = F.softmax(routing_scores, dim=-1) * num_memory_states
        else:
            routing_weights = torch.ones(
                batch_size, seq_len, num_heads, num_memory_states,
                device=x.device, dtype=x.dtype,
            )
        return retrieval_phase, routing_weights

    def _route_balance_loss(self, routing_weights: torch.Tensor):
        balance_lambda = self.route_balance_lambda
        if balance_lambda <= 0 or not self.state_compete or not self.training:
            return None
        routing_prob = routing_weights / self.n_states
        mean_routing_prob = routing_prob.mean(dim=(0, 1))
        entropy = -(mean_routing_prob * (mean_routing_prob + 1e-8).log()).sum(dim=-1)
        return -balance_lambda * entropy.mean()

    def _project(self, x: torch.Tensor, step_offset: int):
        batch_size, seq_len, _, _ = x.shape
        num_heads, head_dim = self.num_heads, self.head_dim
        if self.fused_qkv:
            qkv = self.qkv_proj(x).view(batch_size, seq_len, 3, num_heads, head_dim, 2)
            queries = qkv[:, :, 0].transpose(1, 2).contiguous()
            keys = qkv[:, :, 1].transpose(1, 2).contiguous()
            values = qkv[:, :, 2].transpose(1, 2).contiguous()
        else:
            queries = self.q_proj(x).view(batch_size, seq_len, num_heads, head_dim, 2).transpose(1, 2).contiguous()
            keys = self.k_proj(x).view(batch_size, seq_len, num_heads, head_dim, 2).transpose(1, 2).contiguous()
            values = self.v_proj(x).view(batch_size, seq_len, num_heads, head_dim, 2).transpose(1, 2).contiguous()

        if self.use_rope:
            position_end = step_offset + seq_len
            if position_end > self.rope_cache.shape[0]:
                self.register_buffer(
                    'rope_cache',
                    build_rope_cache(position_end * 2, head_dim).to(x.device),
                    persistent=False,
                )
            rope_positions = self.rope_cache[step_offset:position_end].to(dtype=x.dtype)
            queries = cmul(queries, rope_positions)
            keys = cmul(keys, rope_positions)

        if self.qk_norm:
            queries = cnormalize(queries)
            keys = cnormalize(keys)
        return queries, keys, values

    def _gamma_and_vprime(self, x: torch.Tensor, values: torch.Tensor, state_offset: float = 0.0):
        batch_size, seq_len = x.shape[0], x.shape[1]
        num_heads, head_dim = self.num_heads, self.head_dim
        x_flat = to_real_concat(x)
        if self.decay_mode == 'per_channel':
            decay_logits = self.dt_proj(x_flat).view(batch_size, seq_len, num_heads, head_dim)
            softplus_dt = F.softplus(decay_logits + self.dt_bias + state_offset)
            softplus_dt = softplus_dt.permute(0, 2, 1, 3).contiguous()
        else:
            decay_logits = self.dt_proj(x_flat)
            softplus_dt = F.softplus(decay_logits + self.dt_bias + state_offset)
            softplus_dt = softplus_dt.transpose(1, 2).contiguous()

        base_decay = self._apply_gamma_floor(torch.exp(-softplus_dt))
        if self.use_gsp:
            gate_input = to_real_concat(x) if self.gate_content_aware else cabs(x)
            protect_prob = torch.sigmoid(self.protect_gate(gate_input)).transpose(1, 2)
            if self.gate_surprisal_lambda > 0 and self.training:
                self._gate_prob_bt = protect_prob.mean(dim=1)
            if self.decay_mode == 'per_channel':
                protect_prob_expanded = protect_prob.unsqueeze(-1)
                decay_gamma = base_decay * (1 - protect_prob_expanded) + protect_prob_expanded
            else:
                decay_gamma = base_decay * (1 - protect_prob) + protect_prob
            protected_values = scale_complex(values, 1 - protect_prob)
        else:
            decay_gamma = base_decay
            protected_values = values
        return decay_gamma, protected_values

    @staticmethod
    def _dual_form_block(scaled_queries, keys, protected_values, decay_gamma, causal_mask):
        batch_size, num_heads, seq_len = decay_gamma.shape
        decay_gamma_flat = decay_gamma.reshape(batch_size * num_heads, seq_len)
        decay_matrix = fused_decay_matrix(decay_gamma_flat, seq_len).reshape(
            batch_size, num_heads, seq_len, seq_len
        )
        query_real, query_imag = scaled_queries[..., 0], scaled_queries[..., 1]
        key_real, key_imag = keys[..., 0], keys[..., 1]
        score_real = query_real @ key_real.transpose(-1, -2) + query_imag @ key_imag.transpose(-1, -2)
        score_imag = query_imag @ key_real.transpose(-1, -2) - query_real @ key_imag.transpose(-1, -2)
        weighted_real, weighted_imag = score_real * decay_matrix, score_imag * decay_matrix
        value_real, value_imag = protected_values[..., 0], protected_values[..., 1]
        output_real = weighted_real @ value_real - weighted_imag @ value_imag
        output_imag = weighted_real @ value_imag + weighted_imag @ value_real
        output = torch.stack([output_real, output_imag], dim=-1)
        decay_last_row = decay_matrix[:, :, -1, :]
        write_value_real = value_real * decay_last_row.unsqueeze(-1)
        write_value_imag = value_imag * decay_last_row.unsqueeze(-1)
        state_real = write_value_real.transpose(-1, -2) @ key_real + write_value_imag.transpose(-1, -2) @ key_imag
        state_imag = write_value_imag.transpose(-1, -2) @ key_real - write_value_real.transpose(-1, -2) @ key_imag
        memory_state = torch.stack([state_real, state_imag], dim=-1)
        return output, memory_state

    def _forward_chunked_head(self, queries, keys, protected_values, decay_gamma, head_dim):
        batch_size, num_heads, seq_len = queries.shape[:3]
        chunk_size = self.chunk_size
        query_scale = head_dim ** -0.5
        scaled_queries = queries * query_scale
        memory_state = queries.new_zeros(batch_size, num_heads, head_dim, head_dim, 2)
        outputs = []
        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_len)
            chunk_len = chunk_end - chunk_start
            queries_chunk = scaled_queries[:, :, chunk_start:chunk_end]
            keys_chunk = keys[:, :, chunk_start:chunk_end]
            values_chunk = protected_values[:, :, chunk_start:chunk_end]
            decay_gamma_chunk = decay_gamma[:, :, chunk_start:chunk_end]
            causal = self._causal[:chunk_len, :chunk_len]
            output_chunk, state_chunk = self._dual_form_block(
                queries_chunk, keys_chunk, values_chunk, decay_gamma_chunk, causal
            )
            log_decay = torch.log(decay_gamma_chunk + 1e-6)
            cumulative_decay = torch.exp(torch.cumsum(log_decay, dim=-1))
            if chunk_start > 0:
                state_real, state_imag = memory_state[..., 0], memory_state[..., 1]
                query_real_chunk, query_imag_chunk = queries_chunk[..., 0], queries_chunk[..., 1]
                carried_real = (
                    state_real @ query_real_chunk.transpose(-1, -2)
                    - state_imag @ query_imag_chunk.transpose(-1, -2)
                ).transpose(-1, -2)
                carried_imag = (
                    state_real @ query_imag_chunk.transpose(-1, -2)
                    + state_imag @ query_real_chunk.transpose(-1, -2)
                ).transpose(-1, -2)
                cumulative_decay_expanded = cumulative_decay.unsqueeze(-1)
                output_chunk = output_chunk + torch.stack(
                    [carried_real * cumulative_decay_expanded, carried_imag * cumulative_decay_expanded],
                    dim=-1,
                )
            outputs.append(output_chunk)
            total_decay = cumulative_decay[:, :, -1]
            memory_state = memory_state * total_decay[..., None, None, None] + state_chunk
        return torch.cat(outputs, dim=2), memory_state

    def _forward_chunked_perchannel(self, queries, keys, protected_values, decay_gamma, head_dim):
        batch_size, num_heads, seq_len = queries.shape[:3]
        chunk_size = self.chunk_size
        query_scale = head_dim ** -0.5
        memory_state = queries.new_zeros(batch_size, num_heads, head_dim, head_dim, 2)
        outputs = []
        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_len)
            chunk_len = chunk_end - chunk_start
            queries_chunk = queries[:, :, chunk_start:chunk_end]
            keys_chunk = keys[:, :, chunk_start:chunk_end]
            values_chunk = protected_values[:, :, chunk_start:chunk_end]
            decay_gamma_chunk = decay_gamma[:, :, chunk_start:chunk_end]

            log_decay = torch.log(decay_gamma_chunk.clamp_min(1e-6)).float()
            cumulative_log_decay = torch.cumsum(log_decay, dim=2)
            cumulative_log_decay = cumulative_log_decay.clamp(min=-30.0)
            alpha = torch.exp(cumulative_log_decay)
            inv_alpha = torch.exp(-cumulative_log_decay)
            cumulative_log_total = cumulative_log_decay[:, :, -1:, :]
            decay_tail = torch.exp(cumulative_log_total - cumulative_log_decay)
            alpha = alpha.to(queries.dtype)
            inv_alpha = inv_alpha.to(queries.dtype)
            decay_tail = decay_tail.to(queries.dtype)
            alpha_total = torch.exp(cumulative_log_total).to(queries.dtype)

            folded_queries = queries_chunk * alpha.unsqueeze(-1) * query_scale
            folded_keys = keys_chunk * inv_alpha.unsqueeze(-1)
            query_real, query_imag = folded_queries[..., 0], folded_queries[..., 1]
            key_real, key_imag = folded_keys[..., 0], folded_keys[..., 1]
            score_real = query_real @ key_real.transpose(-1, -2) + query_imag @ key_imag.transpose(-1, -2)
            score_imag = query_imag @ key_real.transpose(-1, -2) - query_real @ key_imag.transpose(-1, -2)
            causal = self._causal[:chunk_len, :chunk_len]
            score_real, score_imag = score_real * causal, score_imag * causal
            value_real, value_imag = values_chunk[..., 0], values_chunk[..., 1]
            output_real = score_real @ value_real - score_imag @ value_imag
            output_imag = score_real @ value_imag + score_imag @ value_real
            output_chunk = torch.stack([output_real, output_imag], dim=-1)

            if chunk_start > 0:
                queries_with_decay = queries_chunk * alpha.unsqueeze(-1) * query_scale
                state_real, state_imag = memory_state[..., 0], memory_state[..., 1]
                query_real_decay, query_imag_decay = queries_with_decay[..., 0], queries_with_decay[..., 1]
                carried_real = (
                    query_real_decay @ state_real.transpose(-1, -2)
                    - query_imag_decay @ state_imag.transpose(-1, -2)
                )
                carried_imag = (
                    query_real_decay @ state_imag.transpose(-1, -2)
                    + query_imag_decay @ state_real.transpose(-1, -2)
                )
                output_chunk = output_chunk + torch.stack([carried_real, carried_imag], dim=-1)

            outputs.append(output_chunk)

            decayed_keys = keys_chunk * decay_tail.unsqueeze(-1)
            decayed_key_real, decayed_key_imag = decayed_keys[..., 0], decayed_keys[..., 1]
            state_real = value_real.transpose(-1, -2) @ decayed_key_real + value_imag.transpose(-1, -2) @ decayed_key_imag
            state_imag = value_imag.transpose(-1, -2) @ decayed_key_real - value_real.transpose(-1, -2) @ decayed_key_imag
            state_chunk = torch.stack([state_real, state_imag], dim=-1)
            alpha_total_squeezed = alpha_total.squeeze(2)
            memory_state = memory_state * alpha_total_squeezed.unsqueeze(2).unsqueeze(-1) + state_chunk
        return torch.cat(outputs, dim=2), memory_state

    def _forward_delta(self, queries, keys, protected_values, decay_gamma, write_beta, head_dim):
        batch_size, num_heads, seq_len = queries.shape[:3]
        chunk_size = self.delta_chunk
        query_scale = head_dim ** -0.5
        memory_state = queries.new_zeros(batch_size, num_heads, head_dim, head_dim, 2)
        outputs = []
        identity = torch.eye(chunk_size, device=queries.device, dtype=torch.float32)
        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_len)
            chunk_len = chunk_end - chunk_start
            queries_chunk = queries[:, :, chunk_start:chunk_end]
            keys_chunk = keys[:, :, chunk_start:chunk_end]
            values_chunk = protected_values[:, :, chunk_start:chunk_end]
            decay_gamma_chunk = decay_gamma[:, :, chunk_start:chunk_end]
            write_beta_chunk = write_beta[:, :, chunk_start:chunk_end]

            decay_gamma_flat = decay_gamma_chunk.reshape(batch_size * num_heads, chunk_len)
            decay_matrix = fused_decay_matrix(decay_gamma_flat, chunk_len).reshape(
                batch_size, num_heads, chunk_len, chunk_len
            )
            log_decay = torch.log(decay_gamma_chunk + 1e-6)
            cumulative_alpha = torch.exp(torch.cumsum(log_decay, dim=-1))

            key_real, key_imag = keys_chunk[..., 0], keys_chunk[..., 1]
            query_real, query_imag = queries_chunk[..., 0], queries_chunk[..., 1]
            key_gram_real = key_real @ key_real.transpose(-1, -2) + key_imag @ key_imag.transpose(-1, -2)
            key_gram_imag = key_imag @ key_real.transpose(-1, -2) - key_real @ key_imag.transpose(-1, -2)
            strict_lower = torch.tril(torch.ones(chunk_len, chunk_len, device=queries.device), -1)
            decay_masked = decay_matrix * strict_lower
            mass_real = write_beta_chunk.unsqueeze(-1) * decay_masked * key_gram_real
            mass_imag = write_beta_chunk.unsqueeze(-1) * decay_masked * key_gram_imag

            value_real, value_imag = values_chunk[..., 0], values_chunk[..., 1]
            if chunk_start > 0:
                state_real, state_imag = memory_state[..., 0], memory_state[..., 1]
                state_key_real = (
                    key_real @ state_real.transpose(-1, -2) - key_imag @ state_imag.transpose(-1, -2)
                )
                state_key_imag = (
                    key_real @ state_imag.transpose(-1, -2) + key_imag @ state_real.transpose(-1, -2)
                )
                state_key_real = state_key_real * cumulative_alpha.unsqueeze(-1)
                state_key_imag = state_key_imag * cumulative_alpha.unsqueeze(-1)
                write_real = write_beta_chunk.unsqueeze(-1) * (value_real - state_key_real)
                write_imag = write_beta_chunk.unsqueeze(-1) * (value_imag - state_key_imag)
            else:
                write_real = write_beta_chunk.unsqueeze(-1) * value_real
                write_imag = write_beta_chunk.unsqueeze(-1) * value_imag

            update_real, update_imag = _complex_triangular_solve(
                mass_real, mass_imag, write_real, write_imag, identity[:chunk_len, :chunk_len]
            )

            query_key_real = query_real @ key_real.transpose(-1, -2) + query_imag @ key_imag.transpose(-1, -2)
            query_key_imag = query_imag @ key_real.transpose(-1, -2) - query_real @ key_imag.transpose(-1, -2)
            causal_inclusive = self._causal[:chunk_len, :chunk_len]
            projection_real = (decay_matrix * causal_inclusive) * query_key_real
            projection_imag = (decay_matrix * causal_inclusive) * query_key_imag
            output_real = (projection_real @ update_real - projection_imag @ update_imag) * query_scale
            output_imag = (projection_real @ update_imag + projection_imag @ update_real) * query_scale
            output_chunk = torch.stack([output_real, output_imag], dim=-1)
            
            if chunk_start > 0:
                state_real, state_imag = memory_state[..., 0], memory_state[..., 1]
                scaled_query_real = query_real * query_scale * cumulative_alpha.unsqueeze(-1)
                scaled_query_imag = query_imag * query_scale * cumulative_alpha.unsqueeze(-1)
                carried_real = (
                    scaled_query_real @ state_real.transpose(-1, -2)
                    - scaled_query_imag @ state_imag.transpose(-1, -2)
                )
                carried_imag = (
                    scaled_query_real @ state_imag.transpose(-1, -2)
                    + scaled_query_imag @ state_real.transpose(-1, -2)
                )
                output_chunk = output_chunk + torch.stack([carried_real, carried_imag], dim=-1)
            outputs.append(output_chunk)

            cumulative_total = cumulative_alpha[:, :, -1:]
            decay_tail = cumulative_total / (cumulative_alpha + 1e-12)
            update_decayed_real = update_real * decay_tail.unsqueeze(-1)
            update_decayed_imag = update_imag * decay_tail.unsqueeze(-1)
            state_real = update_decayed_real.transpose(-1, -2) @ key_real + update_decayed_imag.transpose(-1, -2) @ key_imag
            state_imag = update_decayed_imag.transpose(-1, -2) @ key_real - update_decayed_real.transpose(-1, -2) @ key_imag
            state_chunk = torch.stack([state_real, state_imag], dim=-1)
            memory_state = memory_state * cumulative_total.unsqueeze(-1).unsqueeze(-1) + state_chunk
        return torch.cat(outputs, dim=2), memory_state

    def _forward_multistate(self, x, queries, keys, protected_values, head_dim):
        batch_size, seq_len = x.shape[0], x.shape[1]
        num_heads, num_memory_states = self.num_heads, self.n_states
        query_scale = head_dim ** -0.5
        retrieval_phase, routing_weights = self._phase_and_alpha(x)
        retrieval_phase = retrieval_phase.permute(0, 2, 3, 1)
        routing_weights = routing_weights.permute(0, 2, 3, 1)
        self._route_aux = self._route_balance_loss(routing_weights.permute(0, 3, 1, 2))
        output_sum = None
        state_list = []
        for state_idx in range(num_memory_states):
            decay_gamma_state, protected_values_state = self._gamma_and_vprime(
                x, protected_values, state_offset=self.state_dt_offset[state_idx]
            )
            if self.decay_mode == 'per_channel':
                output_state, memory_state = self._forward_chunked_perchannel(
                    queries, keys, protected_values_state, decay_gamma_state, head_dim
                )
            elif self.chunk_size > 0 and seq_len > self.chunk_size:
                output_state, memory_state = self._forward_chunked_head(
                    queries, keys, protected_values_state, decay_gamma_state, head_dim
                )
            else:
                scaled_queries = queries * query_scale
                output_state, memory_state = self._dual_form_block(
                    scaled_queries, keys, protected_values_state, decay_gamma_state,
                    self._causal[:seq_len, :seq_len],
                )
            rotation_real = routing_weights[:, :, state_idx] * torch.cos(retrieval_phase[:, :, state_idx])
            rotation_imag = routing_weights[:, :, state_idx] * torch.sin(retrieval_phase[:, :, state_idx])
            rotation = torch.stack([rotation_real, rotation_imag], dim=-1)
            output_state = cmul(output_state, rotation.unsqueeze(-2))
            output_sum = output_state if output_sum is None else output_sum + output_state
            state_list.append(memory_state)
        return output_sum, torch.stack(state_list, dim=0)

    def _gamma_all_and_vprime(self, x, values):
        batch_size, seq_len = x.shape[0], x.shape[1]
        num_heads, num_memory_states = self.num_heads, self.n_states
        x_flat = to_real_concat(x)
        decay_logits = self.dt_proj(x_flat)
        state_offsets = self.state_dt_offset.view(num_memory_states, 1, 1, 1)
        softplus_dt = F.softplus(decay_logits + self.dt_bias + state_offsets)
        softplus_dt = softplus_dt.permute(0, 1, 3, 2).contiguous()
        base_decay = self._apply_gamma_floor(torch.exp(-softplus_dt))
        if self.use_gsp:
            gate_input = to_real_concat(x) if self.gate_content_aware else cabs(x)
            protect_prob = torch.sigmoid(self.protect_gate(gate_input)).transpose(1, 2)
            if self.gate_surprisal_lambda > 0 and self.training:
                self._gate_prob_bt = protect_prob.mean(dim=1)
            decay_gamma_all = base_decay * (1 - protect_prob) + protect_prob
            protected_values = scale_complex(values, 1 - protect_prob)
        else:
            decay_gamma_all = base_decay
            protected_values = values
        return decay_gamma_all, protected_values

    def _fused_chunk_step(
        self, queries_chunk, keys_chunk, protected_values_chunk,
        decay_gamma_chunk, retrieval_phase_chunk, routing_weights_chunk,
        memory_state, is_first_chunk,
    ):
        num_memory_states = decay_gamma_chunk.shape[0]
        chunk_len = decay_gamma_chunk.shape[-1]
        query_real, query_imag = real_part(queries_chunk), imag_part(queries_chunk)
        key_real, key_imag = real_part(keys_chunk), imag_part(keys_chunk)
        score_real = query_real @ key_real.transpose(-1, -2) + query_imag @ key_imag.transpose(-1, -2)
        score_imag = query_imag @ key_real.transpose(-1, -2) - query_real @ key_imag.transpose(-1, -2)

        batch_heads_states = num_memory_states * decay_gamma_chunk.shape[1] * decay_gamma_chunk.shape[2]
        decay_matrix = fused_decay_matrix(
            decay_gamma_chunk.reshape(batch_heads_states, chunk_len), chunk_len
        ).reshape(decay_gamma_chunk.shape + (chunk_len,))
        cos_phase = torch.cos(retrieval_phase_chunk)
        sin_phase = torch.sin(retrieval_phase_chunk)
        decay_real = ((routing_weights_chunk * cos_phase).unsqueeze(-1) * decay_matrix).sum(dim=0)
        decay_imag = ((routing_weights_chunk * sin_phase).unsqueeze(-1) * decay_matrix).sum(dim=0)

        weighted_real = score_real * decay_real - score_imag * decay_imag
        weighted_imag = score_real * decay_imag + score_imag * decay_real
        value_real, value_imag = real_part(protected_values_chunk), imag_part(protected_values_chunk)
        output_real = weighted_real @ value_real - weighted_imag @ value_imag
        output_imag = weighted_real @ value_imag + weighted_imag @ value_real
        output_chunk = stack_complex(output_real, output_imag)

        log_decay = torch.log(decay_gamma_chunk + 1e-6)
        cumulative_decay = torch.exp(torch.cumsum(log_decay, dim=-1))

        if not is_first_chunk:
            state_real, state_imag = real_part(memory_state), imag_part(memory_state)
            query_real_states = query_real.unsqueeze(0)
            query_imag_states = query_imag.unsqueeze(0)
            carried_real = (
                state_real @ query_real_states.transpose(-1, -2)
                - state_imag @ query_imag_states.transpose(-1, -2)
            ).transpose(-1, -2)
            carried_imag = (
                state_real @ query_imag_states.transpose(-1, -2)
                + state_imag @ query_real_states.transpose(-1, -2)
            ).transpose(-1, -2)
            combined_real = routing_weights_chunk * cos_phase * cumulative_decay
            combined_imag = routing_weights_chunk * sin_phase * cumulative_decay
            routed_real = (carried_real * combined_real.unsqueeze(-1) - carried_imag * combined_imag.unsqueeze(-1)).sum(dim=0)
            routed_imag = (carried_real * combined_imag.unsqueeze(-1) + carried_imag * combined_real.unsqueeze(-1)).sum(dim=0)
            output_chunk = output_chunk + stack_complex(routed_real, routed_imag)

        decay_last = decay_matrix[:, :, :, -1, :]
        value_real_states = value_real.unsqueeze(0)
        value_imag_states = value_imag.unsqueeze(0)
        write_value_real = value_real_states * decay_last.unsqueeze(-1)
        write_value_imag = value_imag_states * decay_last.unsqueeze(-1)
        key_real_states = key_real.unsqueeze(0)
        key_imag_states = key_imag.unsqueeze(0)
        state_real = write_value_real.transpose(-1, -2) @ key_real_states + write_value_imag.transpose(-1, -2) @ key_imag_states
        state_imag = write_value_imag.transpose(-1, -2) @ key_real_states - write_value_real.transpose(-1, -2) @ key_imag_states
        state_chunk = stack_complex(state_real, state_imag)
        total_decay = cumulative_decay[:, :, :, -1]
        memory_state_new = memory_state * total_decay[..., None, None, None] + state_chunk
        return output_chunk, memory_state_new

    def _forward_multistate_fused(self, x, queries, keys, values, head_dim):
        batch_size, seq_len = x.shape[0], x.shape[1]
        num_heads, num_memory_states = self.num_heads, self.n_states
        chunk_size = self.chunk_size if self.chunk_size > 0 else seq_len
        query_scale = head_dim ** -0.5
        retrieval_phase, routing_weights = self._phase_and_alpha(x)
        retrieval_phase = retrieval_phase.permute(3, 0, 2, 1)
        routing_weights = routing_weights.permute(3, 0, 2, 1)
        self._route_aux = self._route_balance_loss(
            routing_weights.permute(1, 3, 2, 0)
        )
        decay_gamma_all, protected_values = self._gamma_all_and_vprime(x, values)
        scaled_queries = queries * query_scale
        recompute = getattr(self, 'recompute_pam_chunks', False) and self.training

        memory_state = queries.new_zeros(num_memory_states, batch_size, num_heads, head_dim, head_dim, 2)
        outputs = []
        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_len)
            queries_chunk = scaled_queries[:, :, chunk_start:chunk_end]
            keys_chunk = keys[:, :, chunk_start:chunk_end]
            protected_values_chunk = protected_values[:, :, chunk_start:chunk_end]
            decay_gamma_chunk = decay_gamma_all[:, :, :, chunk_start:chunk_end]
            retrieval_phase_chunk = retrieval_phase[:, :, :, chunk_start:chunk_end]
            routing_weights_chunk = routing_weights[:, :, :, chunk_start:chunk_end]
            is_first_chunk = chunk_start == 0
            if recompute:
                output_chunk, memory_state = grad_checkpoint(
                    self._fused_chunk_step,
                    queries_chunk, keys_chunk, protected_values_chunk,
                    decay_gamma_chunk, retrieval_phase_chunk, routing_weights_chunk,
                    memory_state, is_first_chunk,
                    use_reentrant=False,
                )
            else:
                output_chunk, memory_state = self._fused_chunk_step(
                    queries_chunk, keys_chunk, protected_values_chunk,
                    decay_gamma_chunk, retrieval_phase_chunk, routing_weights_chunk,
                    memory_state, is_first_chunk,
                )
            outputs.append(output_chunk)

        return torch.cat(outputs, dim=2), memory_state

    def forward(self, x, state=None, step_offset: int = 0):
        batch_size, seq_len, _, _ = x.shape
        num_heads, head_dim = self.num_heads, self.head_dim
        queries, keys, values = self._project(x, step_offset)

        if state is None and seq_len > 1:
            if self.n_states > 1:
                use_fused = (
                    self.fused_e3
                    and self.decay_mode != 'per_channel'
                    and self.write_mode == 'additive'
                )
                if use_fused:
                    output, new_state = self._forward_multistate_fused(x, queries, keys, values, head_dim)
                else:
                    output, new_state = self._forward_multistate(x, queries, keys, values, head_dim)
            elif self.write_mode == 'delta':
                decay_gamma, protected_values = self._gamma_and_vprime(x, values)
                write_beta = torch.sigmoid(self.beta_proj(cabs(x))).transpose(1, 2)
                output, new_state = self._forward_delta(
                    queries, keys, protected_values, decay_gamma, write_beta, head_dim
                )
            elif self.decay_mode == 'per_channel':
                decay_gamma, protected_values = self._gamma_and_vprime(x, values)
                output, new_state = self._forward_chunked_perchannel(
                    queries, keys, protected_values, decay_gamma, head_dim
                )
            else:
                decay_gamma, protected_values = self._gamma_and_vprime(x, values)
                if self.chunk_size > 0 and seq_len > self.chunk_size:
                    output, new_state = self._forward_chunked_head(
                        queries, keys, protected_values, decay_gamma, head_dim
                    )
                else:
                    scaled_queries = queries * (head_dim ** -0.5)
                    output, new_state = self._dual_form_block(
                        scaled_queries, keys, protected_values, decay_gamma,
                        self._causal[:seq_len, :seq_len],
                    )
        else:
            output, new_state = self._recurrent(x, queries, keys, values, state, head_dim)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.inner_dim, 2)
        out = self.o_proj(output)
        if self.training:
            dropout_mask = as_complex_dropout_mask(self.dropout, out)
            out = scale_complex(out, dropout_mask)
        return out, new_state

    def _recurrent(self, x, queries, keys, values, state, head_dim):
        batch_size, seq_len = x.shape[0], x.shape[1]
        num_heads, num_memory_states = self.num_heads, self.n_states
        query_scale = head_dim ** -0.5
        write_beta = None
        if self.write_mode == 'delta':
            write_beta = torch.sigmoid(self.beta_proj(cabs(x))).transpose(1, 2)
        if self.n_states > 1:
            retrieval_phase, routing_weights = self._phase_and_alpha(x)
            retrieval_phase = retrieval_phase.permute(0, 2, 3, 1)
            routing_weights = routing_weights.permute(0, 2, 3, 1)
            self._route_aux = self._route_balance_loss(routing_weights.permute(0, 3, 1, 2))

        if state is None:
            if self.n_states > 1:
                memory_state = torch.zeros(
                    num_memory_states, batch_size, num_heads, head_dim, head_dim, 2,
                    device=x.device, dtype=x.dtype,
                )
            else:
                memory_state = torch.zeros(
                    batch_size, num_heads, head_dim, head_dim, 2,
                    device=x.device, dtype=x.dtype,
                )
        else:
            memory_state = state

        output_steps = []
        for time_idx in range(seq_len):
            token_input = x[:, time_idx:time_idx + 1]
            key_t = keys[:, :, time_idx]
            query_t = queries[:, :, time_idx] * query_scale
            value_t = values[:, :, time_idx]
            if self.n_states > 1:
                output_accum = None
                new_states = []
                for state_idx in range(num_memory_states):
                    decay_gamma_state, protected_values_state = self._gamma_and_vprime(
                        token_input, values[:, :, time_idx:time_idx + 1],
                        state_offset=self.state_dt_offset[state_idx],
                    )
                    decay_gamma_t = decay_gamma_state[:, :, 0]
                    output_state, state_new = self._recur_step_additive(
                        memory_state[state_idx], decay_gamma_t,
                        protected_values_state[:, :, 0], key_t, query_t,
                    )
                    rotation_real = routing_weights[:, :, state_idx, time_idx] * torch.cos(
                        retrieval_phase[:, :, state_idx, time_idx]
                    )
                    rotation_imag = routing_weights[:, :, state_idx, time_idx] * torch.sin(
                        retrieval_phase[:, :, state_idx, time_idx]
                    )
                    rotation = stack_complex(rotation_real, rotation_imag)
                    output_state = cmul(output_state, rotation.unsqueeze(-2))
                    output_accum = output_state if output_accum is None else output_accum + output_state
                    new_states.append(state_new)
                output_steps.append(output_accum)
                memory_state = torch.stack(new_states, dim=0)
                continue

            decay_gamma, protected_values = self._gamma_and_vprime(
                token_input, values[:, :, time_idx:time_idx + 1]
            )
            decay_gamma_t = decay_gamma[:, :, 0]
            protected_value_t = protected_values[:, :, 0]
            if self.write_mode == 'delta':
                output_step, memory_state = self._recur_step_delta(
                    memory_state, decay_gamma_t, protected_value_t, key_t, query_t,
                    write_beta[:, :, time_idx],
                )
            else:
                output_step, memory_state = self._recur_step_additive(
                    memory_state, decay_gamma_t, protected_value_t, key_t, query_t,
                )
            output_steps.append(output_step)

        output = torch.stack(output_steps, dim=2)
        return output, memory_state

    def _recur_step_additive(self, memory_state, decay_gamma, value_t, key_t, query_t):
        key_conj = stack_complex(real_part(key_t), -imag_part(key_t)).unsqueeze(-3)
        outer_real = (
            real_part(value_t).unsqueeze(-1) * real_part(key_conj)
            - imag_part(value_t).unsqueeze(-1) * imag_part(key_conj)
        )
        outer_imag = (
            real_part(value_t).unsqueeze(-1) * imag_part(key_conj)
            + imag_part(value_t).unsqueeze(-1) * real_part(key_conj)
        )
        outer_product = stack_complex(outer_real, outer_imag)
        if decay_gamma.dim() == memory_state.dim() - 3:
            decay_factor = decay_gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        else:
            decay_factor = decay_gamma.unsqueeze(-2).unsqueeze(-1)
        memory_state = memory_state * decay_factor + outer_product
        state_query_real = (
            real_part(memory_state) * real_part(query_t).unsqueeze(-2)
            - imag_part(memory_state) * imag_part(query_t).unsqueeze(-2)
        )
        state_query_imag = (
            real_part(memory_state) * imag_part(query_t).unsqueeze(-2)
            + imag_part(memory_state) * real_part(query_t).unsqueeze(-2)
        )
        output = stack_complex(state_query_real.sum(dim=-1), state_query_imag.sum(dim=-1))
        return output, memory_state

    def _recur_step_delta(self, memory_state, decay_gamma, value_t, key_t, query_t, write_beta_t):
        decay_factor = decay_gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        memory_state = memory_state * decay_factor
        predicted_real = (
            memory_state[..., 0] * key_t[..., 0].unsqueeze(-2)
            - memory_state[..., 1] * key_t[..., 1].unsqueeze(-2)
        ).sum(dim=-1)
        predicted_imag = (
            memory_state[..., 0] * key_t[..., 1].unsqueeze(-2)
            + memory_state[..., 1] * key_t[..., 0].unsqueeze(-2)
        ).sum(dim=-1)
        beta_expanded = write_beta_t.unsqueeze(-1)
        update_real = beta_expanded * (value_t[..., 0] - predicted_real)
        update_imag = beta_expanded * (value_t[..., 1] - predicted_imag)
        update = torch.stack([update_real, update_imag], dim=-1)
        key_conj = torch.stack([key_t[..., 0], -key_t[..., 1]], dim=-1)
        outer_real = (
            update[..., 0].unsqueeze(-1) * key_conj[..., 0].unsqueeze(-2)
            - update[..., 1].unsqueeze(-1) * key_conj[..., 1].unsqueeze(-2)
        )
        outer_imag = (
            update[..., 0].unsqueeze(-1) * key_conj[..., 1].unsqueeze(-2)
            + update[..., 1].unsqueeze(-1) * key_conj[..., 0].unsqueeze(-2)
        )
        memory_state = memory_state + torch.stack([outer_real, outer_imag], dim=-1)
        state_query_real = (
            memory_state[..., 0] * query_t[..., 0].unsqueeze(-2)
            - memory_state[..., 1] * query_t[..., 1].unsqueeze(-2)
        )
        state_query_imag = (
            memory_state[..., 0] * query_t[..., 1].unsqueeze(-2)
            + memory_state[..., 1] * query_t[..., 0].unsqueeze(-2)
        )
        output = stack_complex(state_query_real.sum(dim=-1), state_query_imag.sum(dim=-1))
        return output, memory_state


# ── Residual Block Wrapper ────────────────────────────────────────────────────

class V11Block(nn.Module):
    """Pre-norm residual: CGU (channel mix) + PAM (sequence/memory mix)."""

    def __init__(self, cfg: PAMConfig, layer_idx: int = 0):
        super().__init__()
        self.norm1 = ComplexNorm(cfg.dim)
        self.cgu = ComplexGatedUnit(cfg.dim, cfg.expand, activation=cfg.activation)
        self.cgu_scale = nn.Parameter(torch.tensor(1.0))
        self.cgu_dropout = nn.Dropout(cfg.dropout)
        self.norm2 = ComplexNorm(cfg.dim)
        self.pam = V11PAMLayer(cfg, layer_idx=layer_idx)
        self.pam_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, pam_state=None, step_offset: int = 0):
        cgu_out = self.cgu(self.norm1(x))
        if self.training:
            drop = as_complex_dropout_mask(self.cgu_dropout, cgu_out)
            cgu_out = scale_complex(cgu_out, drop)
        x = x + cgu_out * self.cgu_scale
        pam_out, new_state = self.pam(self.norm2(x), state=pam_state, step_offset=step_offset)
        x = x + pam_out * self.pam_scale
        return x, new_state
