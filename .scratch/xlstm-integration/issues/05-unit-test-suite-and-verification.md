# 05 — xLSTM Unit Test Suite and Verification

**What to build:**
Comprehensive unit test suite in `tests/test_xlstm.py` testing config parsing, model forward pass shapes, loss backward pass, generation, and CLI dictionary instantiation.

**Blocked by:** 04 — CLI Model Integration and Benchmark YAML Configurations

**Status:** completed

- [x] `tests/test_xlstm.py` tests `xLSTMConfig` and block pattern resolution.
- [x] `tests/test_xlstm.py` tests `xLSTMModel` forward pass returning `(logits, loss)` with correct shape `(B, T, vocab_size)`.
- [x] `tests/test_xlstm.py` tests gradient backward pass.
- [x] `tests/test_xlstm.py` tests `generate(...)` output shapes.
- [x] `tests/test_xlstm.py` tests `create_model_from_dict` with `type: "xlstm"`.
- [x] `uv run pytest tests/test_xlstm.py` and `uv run pytest tests/` run green.
