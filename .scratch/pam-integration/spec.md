# Spec: Phase-Associative Memory (PAM) Model Integration

**Status:** ready-for-agent
**Triage Role:** ready-for-agent

## Problem Statement

The benchmark environment currently only evaluates standard real-valued architectures (specifically, GPT). In natural language processing, semantic expressions are highly contextual and indeterminate prior to interpretation. Replicating this quantum-like contextuality using standard real-valued architectures and softmax attention requires simulating complex phase relationships using higher parameter counts, which causes inefficiencies. We need a way to run benchmarks on a natively complex-valued, attention-free, recurrence-based architecture that uses a fixed-size matrix state to evaluate its scaling dynamics and efficiency compared to real-valued architectures.

## Solution

Integrate Phase-Associative Memory (PAM) into the benchmark environment. PAM uses complex-valued token embeddings, a complex matrix state $S_t \in \mathbb{C}^{d \times d}$ per head, and retrieves associations using the conjugate inner product $\operatorname{Re}\langle K_i^* \mid Q_t \rangle$. By implementing this in a self-contained, standard-compliant model interface (`PAMModel`), we will enable standard training runs, evaluation, and autoregressive text generation using our existing `Trainer` and CLI interface.

## User Stories

1. As a machine learning researcher, I want to configure a PAM model in a YAML configuration file using a `type: pam` property under the `model` block, so that I can easily toggle between architectures.
2. As a model designer, I want the config loader to map standard benchmark fields like `block_size` and `n_embd` to PAM's internal sequence length and dimensionality attributes, so that I don't have to rewrite model configuration blocks.
3. As a developer, I want all core complex math helpers (multiplication, conjugation, magnitude, normalization) to run on split-real tensors of shape `[..., 2]`, so that they run compatibly on standard hardware devices (CPU, CUDA, MPS).
4. As a researcher, I want the PAM layer to implement parallel chunked execution during training, so that I can train models in $O(T^2)$ time utilizing parallel GPU resources.
5. As a researcher, I want the PAM layer to support recurrent $O(1)$ token execution during generation, so that text generation remains efficient with a fixed-size state matrix and no linear-growing Key-Value cache.
6. As a developer, I want the PAM model to compute standard cross-entropy loss over targets if provided during the forward pass, so that the training loop can calculate gradients and log loss.
7. As an optimizer engine, I want custom parameter grouping that excludes biases, normalization scales, and decay biases (like `dt_bias` and `protect_gate.bias`) from weight decay, so that training converges stably.
8. As a test suite, I want comprehensive unit tests for complex math blocks, activation functions, and layer shapes, so that I can ensure the correctness of the mathematical logic.
9. As a test suite, I want integration tests running a full dummy training run, evaluation, and inference on a tiny PAM model configuration, so that I know the CLI works end-to-end.

## Implementation Decisions

- **New Module `src/models/pam.py`**:
  - Expose `PAMConfig` with fields mapping to the model's structural parameters (dimensions, layers, heads, etc.).
  - Implement a complex math library working on split-real float tensors (final dimension of size 2).
  - Implement activation layers including magnitude-based ModReLU and ModSwish.
  - Implement a `ComplexGatedUnit` channel-mixing block.
  - Implement `V11PAMLayer` representing sequence mixing via matrix state updates ($S_t = \gamma_t S_{t-1} + V'_t \otimes K_t^*$) and conjugate inner-product retrieval ($Y_t = S_t \tilde{Q}_t$).
  - Implement a custom `configure_optimizers` method grouping 2D weights for decay and keeping 1D parameters, biases, and special decay variables out of decay.
  - Expose a `generate` method using recurrent state step-by-step updates for $O(1)$ inference.

- **CLI Compatibility (`src/cli.py`)**:
  - Update YAML config parsing to instantiate `PAMModel` when `type: pam` is selected, defaulting to `GPTModel` otherwise.
  - Standardize configuration property names by mapping `block_size` to `max_seq_len` and `n_embd` to `dim`.

- **Glossary (`CONTEXT.md`)**:
  - Formally register Phase-Associative Memory, Complex Gated Unit, and Complex Representation in the domain glossary.

## Testing Decisions

- **Testing Seam 1 (Unit Level)**:
  - Verify individual tensor shapes and operations inside `src/models/pam.py`.
  - Assert that complex multiplication and normalization behave correctly against expected manual calculations.
  - Seam prior art: `tests/test_model.py`.

- **Testing Seam 2 (Inference State Caching)**:
  - Feed a prompt sequence and ensure that the recurrent generation form matches the parallel generation form exactly at the logits level, proving the correctness of the recurrent state updates.

- **Testing Seam 3 (CLI / Trainer End-to-End)**:
  - Use `CliRunner` to execute end-to-end tiny training, validation, and generation pipelines for `type: pam`.
  - Seam prior art: `tests/test_cli.py`.

## Out of Scope

- Integrating external hardware-specific CUDA kernels for block-real gemms or parallel scans.
- Pretraining a large model (e.g. 100M parameters) or downloading the full WikiText-103 dataset as part of unit tests.
- Somatic or biological validation of quantum/human semantic structures.

## Further Notes

The default initialization bias for the protect gate in GSP will be initialized to `-3.0` and the decay bias `dt_bias` will be initialized to `-4.0` as specified in the V11 defaults from `qllm2` to ensure slow initial decay and model stability.
