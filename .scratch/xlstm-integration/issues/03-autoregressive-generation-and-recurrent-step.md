# 03 — Autoregressive Generation Step and Recurrent State Update

**What to build:**
Implement stateful token-by-token `generate(...)` for `xLSTMModel` using recurrent state updates (`step`) for mLSTM and sLSTM cells.

**Blocked by:** 02 — xLSTM Model Class and Parallel Training Forward Pass

**Status:** completed

- [x] `mLSTMBlock` and `sLSTMBlock` implement step-by-step `step(x, state)` method maintaining memory state matrices/vectors across time steps $t$.
- [x] `xLSTMModel.generate(idx, max_new_tokens, temperature, top_k)` generates completion sequence index tensors.
