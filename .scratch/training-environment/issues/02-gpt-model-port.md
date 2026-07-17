# 02 — GPT Model Architecture Port

**What to build:**
A standardized model interface and a port of the `GPT` model architecture from `old/model.py`. The model must inherit from a common `Model` base class and support forward passes (returning logits and loss), token generation, and optimizer configuration.

**Blocked by:** None — can start immediately

**Status:** completed

- [x] A generic `Model` base class is created at `src/base/model.py` which inherits from `torch.nn.Module`.
- [x] A concrete `GPTModel` (and associated config/components) is created at `src/models/gpt.py` inheriting from `Model`.
- [x] `GPTModel` implements `forward(idx, targets=None)` returning a tuple of `(logits, loss)`.
- [x] `GPTModel` implements `generate(idx, max_new_tokens, temperature, top_k)` returning generated token indices.
- [x] `GPTModel` implements `configure_optimizers(weight_decay, learning_rate, betas, device_type)` returning a configured optimizer.
- [x] A script or test is run to verify that a forward pass and token generation succeed on the ported model with dummy input.

