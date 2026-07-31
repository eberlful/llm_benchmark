# 02 — xLSTM Model Class and Parallel Training Forward Pass

**What to build:**
Implement `xLSTMModel` inheriting from `src.base.model.Model`. Construct embedding layers (with optional `use_pos_emb`), residual block stack, pure PyTorch vectorized parallel causal forward pass computing `(logits, loss)`, and `configure_optimizers(...)`.

**Blocked by:** 01 — Core xLSTM Config and Block Layer Primitives

**Status:** completed

- [x] `xLSTMModel` inherits from `Model` and instantiates token embeddings `wte`, optional position embeddings `wpe`, block stack, final `LayerNorm`, and `lm_head`.
- [x] Parallel causal matrix forward pass computes logits `(B, T, vocab_size)` and cross-entropy loss when `targets` are provided.
- [x] `configure_optimizers(...)` splits parameters into decay and non-decay groups (LayerNorm, 1D biases excluded from weight decay).
