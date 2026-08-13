# 02 — Parameter Optimizer Grouping & Autoregressive Generation

**What to build:** Configure AdamW parameter decay groups separating 2D projection matrices from 1D/LayerNorm parameters, and enable token continuation generation with temperature scaling and top-k filtering.

**Blocked by:** 01 — Core BDH Model & Config Architecture

**Status:** completed

- [x] `configure_optimizers()` groups 2D parameters (`encoder`, `decoder`, `encoder_v`, `lm_head`, `embed.weight`) for weight decay and 1D/LayerNorm/bias parameters for zero weight decay
- [x] `generate(idx, max_new_tokens, temperature, top_k)` generates sequence continuations autoregressively
- [x] Sequence context cropping enforces `block_size` limits during generation
