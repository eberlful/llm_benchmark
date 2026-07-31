# 01 — Core xLSTM Config and Block Layer Primitives

**What to build:**
Implement `xLSTMConfig` with block pattern parsing, causal 1D convolution pre-processing (`CausalConv1d`), and basic `mLSTMBlock` and `sLSTMBlock` layer building blocks in `src/models/xlstm.py`.

**Blocked by:** None — can start immediately

**Status:** completed

- [x] `xLSTMConfig` dataclass handles default values and parses `block_type_pattern` (`"mlstm"`, `"slstm"`, `"7:1"`, or list).
- [x] `CausalConv1d` applies 1D causal convolution with left padding.
- [x] `mLSTMBlock` and `sLSTMBlock` classes initialized with projection layers and LayerNorm residual connections.
