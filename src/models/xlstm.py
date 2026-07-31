import math
import inspect
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.base.model import Model


@dataclass
class xLSTMConfig:
    """
    Configuration dataclass for xLSTM models.
    """
    vocab_size: int = 50257
    n_embd: int = 768
    n_layer: int = 12
    block_size: int = 1024
    dropout: float = 0.0
    bias: bool = True
    use_pos_emb: bool = True
    conv_kernel_size: int = 4
    num_heads: int = 4
    expand_factor: int = 2
    block_type_pattern: Union[str, List[str]] = "7:1"

    def get_block_types(self) -> List[str]:
        """
        Parses `block_type_pattern` into a list of block type strings ("mlstm" or "slstm")
        of length `n_layer`.
        """
        if isinstance(self.block_type_pattern, list):
            if len(self.block_type_pattern) == self.n_layer:
                return list(self.block_type_pattern)
            pattern = []
            while len(pattern) < self.n_layer:
                pattern.extend(self.block_type_pattern)
            return pattern[: self.n_layer]

        pattern_str = str(self.block_type_pattern).lower().strip()
        if pattern_str == "mlstm":
            return ["mlstm"] * self.n_layer
        elif pattern_str == "slstm":
            return ["slstm"] * self.n_layer
        elif pattern_str == "7:1":
            unit = ["mlstm"] * 7 + ["slstm"]
            pattern = []
            while len(pattern) < self.n_layer:
                pattern.extend(unit)
            return pattern[: self.n_layer]
        else:
            raise ValueError(f"Unknown block_type_pattern: {self.block_type_pattern}")


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. """
    def __init__(self, ndim: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalConv1d(nn.Module):
    """
    1D Causal Convolution with left padding.
    """
    def __init__(self, channels: int, kernel_size: int, bias: bool = True):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=0,
            groups=channels,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (B, T, C)
        B, T, C = x.shape
        x_trans = x.transpose(1, 2)  # (B, C, T)
        # Pad left by (kernel_size - 1) on time dimension
        x_pad = F.pad(x_trans, (self.kernel_size - 1, 0))
        out = self.conv(x_pad)  # (B, C, T)
        return out.transpose(1, 2)  # (B, T, C)

    def step(self, x_t: torch.Tensor, conv_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Step-by-step causal convolution for autoregressive inference.
        x_t: (B, C) or (B, 1, C)
        conv_state: (B, kernel_size - 1, C)
        Returns:
            out_t: (B, C)
            new_conv_state: (B, kernel_size - 1, C)
        """
        if x_t.dim() == 2:
            x_t = x_t.unsqueeze(1)  # (B, 1, C)
        B, _, C = x_t.shape
        if conv_state is None:
            conv_state = torch.zeros(B, self.kernel_size - 1, C, device=x_t.device, dtype=x_t.dtype)

        # Concatenate past conv_state with current step x_t along time dim: (B, kernel_size, C)
        window = torch.cat([conv_state, x_t], dim=1)
        window_trans = window.transpose(1, 2)  # (B, C, K)
        out_trans = self.conv(window_trans)   # (B, C, 1)
        out_t = out_trans.squeeze(2)          # (B, C)

        new_conv_state = window[:, 1:, :]      # slide window by 1 step
        return out_t, new_conv_state


class mLSTMBlock(nn.Module):
    """
    Matrix LSTM (mLSTM) Residual Block with causal 1D conv,
    exponential gating, and parallel causal matrix covariance formulation.
    """
    def __init__(self, config: xLSTMConfig):
        super().__init__()
        self.config = config
        self.n_embd = config.n_embd
        self.num_heads = config.num_heads
        self.head_dim = config.n_embd // config.num_heads

        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.causal_conv = CausalConv1d(config.n_embd, kernel_size=config.conv_kernel_size, bias=config.bias)

        # Projections for Query, Key, Value, Input Gate, Forget Gate
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.i_proj = nn.Linear(config.n_embd, config.num_heads, bias=config.bias)
        self.f_proj = nn.Linear(config.n_embd, config.num_heads, bias=config.bias)

        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LayerNorm & Residual
        residual = x
        x_norm = self.ln_1(x)

        # Causal Conv1d pre-processing
        x_conv = self.causal_conv(x_norm)
        x_act = F.silu(x_conv)

        B, T, C = x_act.shape

        # Projections
        q = self.q_proj(x_act).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, NH, T, DH)
        k = self.k_proj(x_act).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, NH, T, DH)
        v = self.v_proj(x_act).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, NH, T, DH)

        i = self.i_proj(x_act).transpose(1, 2)  # (B, NH, T)
        f = self.f_proj(x_act).transpose(1, 2)  # (B, NH, T)

        # Exponential input gate & Sigmoid forget gate with log-stabilization
        log_f = F.logsigmoid(f)  # (B, NH, T)
        log_f_cum = torch.cumsum(log_f, dim=-1)  # (B, NH, T)

        # Causal matrix formulation: decay matrix D[t, j] = exp(log_f_cum[t] - log_f_cum[j] + i[j])
        # Log space representation: S[t, j] = (q_t @ k_j^T) / sqrt(DH) + log_f_cum[t] - log_f_cum[j] + i[j]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, NH, T, T)

        # Construct log decay matrix: log_D[t, j] = log_f_cum[t] - log_f_cum[j] + i[j]
        log_f_diff = log_f_cum.unsqueeze(-1) - log_f_cum.unsqueeze(-2)  # (B, NH, T, T), [t, j] is log_f_cum[t] - log_f_cum[j]
        log_decay = log_f_diff + i.unsqueeze(-2)  # (B, NH, T, T)

        # Total log matrix
        total_scores = scores + log_decay

        # Apply causal mask (j <= t)
        causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        total_scores = total_scores.masked_fill(~causal_mask, float('-inf'))

        # Max-stabilized Softmax along j dimension
        max_scores, _ = torch.max(total_scores, dim=-1, keepdim=True)
        max_scores = torch.clamp(max_scores, min=0.0)  # Avoid NaN when all masked out
        weights = torch.exp(total_scores - max_scores)
        weights = weights.masked_fill(~causal_mask, 0.0)

        # Output calculation
        out_heads = torch.matmul(weights, v)  # (B, NH, T, DH)
        out_heads = out_heads.transpose(1, 2).contiguous().view(B, T, C)

        out = self.c_proj(out_heads)
        out = self.dropout(out)
        return residual + out

    def step(
        self,
        x_t: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Recurrent step for autoregressive generation.
        x_t: (B, C)
        state: Dict containing 'conv_state', 'C_matrix', 'n_vector'
        """
        B, C = x_t.shape
        if state is None:
            state = {
                'conv_state': None,
                'C_matrix': torch.zeros(B, self.num_heads, self.head_dim, self.head_dim, device=x_t.device, dtype=x_t.dtype),
                'n_vector': torch.zeros(B, self.num_heads, self.head_dim, 1, device=x_t.device, dtype=x_t.dtype),
            }

        x_norm = self.ln_1(x_t)
        x_conv, new_conv_state = self.causal_conv.step(x_norm, state['conv_state'])
        x_act = F.silu(x_conv)

        q = self.q_proj(x_act).view(B, self.num_heads, self.head_dim, 1)  # (B, NH, DH, 1)
        k = self.k_proj(x_act).view(B, self.num_heads, self.head_dim, 1)  # (B, NH, DH, 1)
        v = self.v_proj(x_act).view(B, self.num_heads, self.head_dim, 1)  # (B, NH, DH, 1)

        i_gate = torch.exp(self.i_proj(x_act)).unsqueeze(-1).unsqueeze(-1)  # (B, NH, 1, 1)
        f_gate = torch.sigmoid(self.f_proj(x_act)).unsqueeze(-1).unsqueeze(-1)  # (B, NH, 1, 1)

        # State updates: C_t = f_t * C_{t-1} + i_t * (v_t @ k_t^T)
        C_prev = state['C_matrix']
        n_prev = state['n_vector']

        C_new = f_gate * C_prev + i_gate * torch.matmul(v, k.transpose(-2, -1))
        n_new = f_gate * n_prev + i_gate * k

        # Recurrent Output: h_t = (C_t @ q_t) / max(|n_t^T @ q_t|, 1)
        nq = torch.abs(torch.matmul(n_new.transpose(-2, -1), q))  # (B, NH, 1, 1)
        denom = torch.clamp(nq, min=1.0)
        num = torch.matmul(C_new, q)  # (B, NH, DH, 1)

        h_t = (num / denom).squeeze(-1)  # (B, NH, DH)
        h_t = h_t.view(B, C)

        out = self.c_proj(h_t)
        out = self.dropout(out)

        new_state = {
            'conv_state': new_conv_state,
            'C_matrix': C_new,
            'n_vector': n_new,
        }
        return x_t + out, new_state


class sLSTMBlock(nn.Module):
    """
    Scalar LSTM (sLSTM) Residual Block with exponential gating,
    max-stabilization, and scalar recurrent state updates.
    """
    def __init__(self, config: xLSTMConfig):
        super().__init__()
        self.config = config
        self.n_embd = config.n_embd
        self.num_heads = config.num_heads
        self.head_dim = config.n_embd // config.num_heads

        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.causal_conv = CausalConv1d(config.n_embd, kernel_size=config.conv_kernel_size, bias=config.bias)

        # Projections for Cell input z, Input gate i, Forget gate f, Output gate o
        self.z_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.i_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.f_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.ln_1(x)

        x_conv = self.causal_conv(x_norm)
        x_act = F.silu(x_conv)

        B, T, C = x_act.shape

        z = self.z_proj(x_act)  # (B, T, C)
        i = self.i_proj(x_act)  # (B, T, C)
        f = self.f_proj(x_act)  # (B, T, C)
        o = torch.sigmoid(self.o_proj(x_act))  # (B, T, C)

        # Sequential scan over T with max-stabilization for sLSTM
        c = torch.zeros(B, C, device=x.device, dtype=x.dtype)
        n = torch.zeros(B, C, device=x.device, dtype=x.dtype)
        m = torch.zeros(B, C, device=x.device, dtype=x.dtype)

        h_outputs = []

        for t in range(T):
            i_t = i[:, t, :]
            f_t = f[:, t, :]
            z_t = z[:, t, :]
            o_t = o[:, t, :]

            # Max-stabilizer state update: m_t = max(f_t + m_{t-1}, i_t)
            m_new = torch.maximum(f_t + m, i_t)

            # Exponential terms
            exp_f = torch.exp(f_t + m - m_new)
            exp_i = torch.exp(i_t - m_new)

            c = exp_f * c + exp_i * z_t
            n = exp_f * n + exp_i

            m = m_new

            # Stabilized cell state output
            c_norm = c / torch.clamp(n, min=1e-6)
            h_t = o_t * c_norm
            h_outputs.append(h_t.unsqueeze(1))

        h_seq = torch.cat(h_outputs, dim=1)  # (B, T, C)

        out = self.c_proj(h_seq)
        out = self.dropout(out)
        return residual + out

    def step(
        self,
        x_t: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Recurrent step for sLSTM autoregressive generation.
        x_t: (B, C)
        state: Dict containing 'conv_state', 'c', 'n', 'm'
        """
        B, C = x_t.shape
        if state is None:
            state = {
                'conv_state': None,
                'c': torch.zeros(B, C, device=x_t.device, dtype=x_t.dtype),
                'n': torch.zeros(B, C, device=x_t.device, dtype=x_t.dtype),
                'm': torch.zeros(B, C, device=x_t.device, dtype=x_t.dtype),
            }

        x_norm = self.ln_1(x_t)
        x_conv, new_conv_state = self.causal_conv.step(x_norm, state['conv_state'])
        x_act = F.silu(x_conv)

        z_t = self.z_proj(x_act)
        i_t = self.i_proj(x_act)
        f_t = self.f_proj(x_act)
        o_t = torch.sigmoid(self.o_proj(x_act))

        m_prev = state['m']
        c_prev = state['c']
        n_prev = state['n']

        m_new = torch.maximum(f_t + m_prev, i_t)
        exp_f = torch.exp(f_t + m_prev - m_new)
        exp_i = torch.exp(i_t - m_new)

        c_new = exp_f * c_prev + exp_i * z_t
        n_new = exp_f * n_prev + exp_i

        c_norm = c_new / torch.clamp(n_new, min=1e-6)
        h_t = o_t * c_norm

        out = self.c_proj(h_t)
        out = self.dropout(out)

        new_state = {
            'conv_state': new_conv_state,
            'c': c_new,
            'n': n_new,
            'm': m_new,
        }
        return x_t + out, new_state


class xLSTMModel(Model):
    """
    xLSTM Architecture for language modeling inheriting from src.base.model.Model.
    Supports mLSTM and sLSTM blocks, parallel training forward pass, and recurrent step generation.
    """
    def __init__(self, config: xLSTMConfig):
        super().__init__()
        self.config = config

        block_types = config.get_block_types()
        blocks = []
        for b_type in block_types:
            if b_type == "mlstm":
                blocks.append(mLSTMBlock(config))
            elif b_type == "slstm":
                blocks.append(sLSTMBlock(config))
            else:
                raise ValueError(f"Unknown block type: {b_type}")

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd) if config.use_pos_emb else None,
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList(blocks),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Weight initialization
        self.apply(self._init_weights)
        # Scaled residual projection initialization
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and self.transformer.wpe is not None:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        tok_emb = self.transformer.wte(idx)  # (b, t, n_embd)
        if self.transformer.wpe is not None:
            pos = torch.arange(0, t, dtype=torch.long, device=device)  # (t)
            pos_emb = self.transformer.wpe(pos)  # (t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)
        else:
            x = self.transformer.drop(tok_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)  # (b, t, vocab_size)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])  # (b, 1, vocab_size)
            loss = None

        return logits, loss

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
        device_type: str,
    ) -> torch.optim.Optimizer:
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        # 2D or higher tensors get weight decay; 1D tensors (biases, LayerNorms) do not
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        return optimizer

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Sequence completion starting from context sequence idx.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

