# 02 — PAM Model wrapper and generation

**What to build:**
Implement `PAMModel` class inheriting from `src.base.model.Model` that wraps the layers, manages recurrent states during generation, computes cross-entropy loss if targets are provided, and configures the optimizer parameter groups.

The model should:
- Support parameter initialization using custom orthogonal initialization for complex maps.
- Handle state routing: training runs in parallel mode; evaluation/inference runs in recurrent mode when states are provided.
- Implement the `Model.generate` signature using cached recurrent state step-by-step updates for optimal performance.
- Implement `configure_optimizers` grouping 2D parameters ending in `weight`, `weight_real`, and `weight_imag` for decay, while excluding biases, normalizations, and scalar offsets.

**Blocked by:** 01 — PAM Model core implementation

**Status:** resolved
**Triage Role:** resolved

- [x] `PAMModel` wraps `V11Block` layers.
- [x] `PAMModel.forward(idx, targets=None)` handles logits computation and calculates Cross Entropy loss if `targets` is not None.
- [x] `PAMModel.generate(idx, max_new_tokens, temperature, top_k)` uses efficient recurrent states.
- [x] `PAMModel.configure_optimizers` groups parameters correctly based on our decay rules.
