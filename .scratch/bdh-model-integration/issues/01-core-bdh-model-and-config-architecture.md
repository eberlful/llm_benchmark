# 01 — Core BDH Model & Config Architecture

**What to build:** Instantiate the Dragon Hatchling (`BDHModel`) architecture and configuration (`BDHConfig`) inheriting from the generic model base class, supporting forward logit computation and strict paper-causal linear Rotary Position Embedding (RoPE) attention (`tril(diagonal=-1)`).

**Blocked by:** None — can start immediately

**Status:** completed

- [x] `BDHConfig` defined with dataclass fields (`n_layer`, `n_embd`, `dropout`, `n_head`, `mlp_internal_dim_multiplier`, `vocab_size`, `block_size`, `strict_causal`)
- [x] `Attention` implemented using RoPE and lower-triangular causal scores (`diagonal=-1` when `strict_causal=True`)
- [x] `BDHModel` inherits from `src.base.model.Model` and implements `forward(idx, targets)` returning `(logits, loss)`
- [x] `get_num_params()` returns total parameters (with non-embedding subtraction option)
