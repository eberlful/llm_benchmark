Status: ready-for-agent

# Dragon Hatchling (BDH) Model Integration Specification

## Problem Statement

Researchers and developers benchmarking language model architectures need a way to train, evaluate, and compare the Dragon Hatchling (BDH) model—a scale-free biologically inspired architecture based on local distributed graph dynamics—against existing Transformer (GPT) and State Space Model (PAM, xLSTM) baselines. Without native integration into the training platform, evaluating BDH's unique properties (such as sparse positive activations, Hebbian synaptic re-weighting, and linear positional attention) requires manual unintegrated scripts.

## Solution

Integrate the BDH architecture as a first-class Model within the model training platform. Provide configuration schemas, CLI model creation and training hooks, strict paper-causal linear attention with Rotary Position Embeddings (RoPE), AdamW optimizer parameter grouping, and standardized training YAML configurations for both CPU testing and Shakespeare dataset benchmarks.

## User Stories

1. As a machine learning researcher, I want to instantiate the BDH model architecture via configuration files, so that I can compare its training loss and scaling laws against GPT and xLSTM baselines.
2. As a benchmark operator, I want to execute training runs using a unified CLI command (`train`), so that I can launch BDH training without writing custom runner code.
3. As a model developer, I want BDH parameters split into weight-decayed (2D projection matrices) and non-decayed (1D vectors and normalization layers) optimizer groups, so that AdamW training optimizes parameters effectively according to modern practices.
4. As an AI engineer, I want BDH autoregressive generation (`generate`) to support temperature sampling and top-k filtering, so that I can evaluate text generation quality during or after training.
5. As a biological neural network researcher, I want BDH attention to support strict paper causality ($\sum_{\tau < t}$), so that key-value accumulation strictly models past-context updates as described in the theoretical BDH graph equations.
6. As a developer, I want lightweight CPU training configurations for BDH, so that I can rapidly run integration and smoke tests in CI/CD environments.

## Implementation Decisions

- **Architecture & Model Interface**: Subclass the generic base Model interface directly for the BDH model implementation. The model manages embedding lookup, LayerNorm pre-processing, multi-layer sparse feed-forward ReLU projections, linear RoPE attention, and output language modeling head projections.
- **Config Dataclass**: Introduce a dedicated configuration object (`BDHConfig`) encapsulating hidden layer count, embedding dimension, dropout probability, attention head count, MLP internal dimension multiplier, vocabulary size, maximum context block size, and strict causality flags.
- **Attention Mechanism**: Implement linear positional attention using Rotary Position Embeddings (RoPE). Support strict lower-triangular causal masking (`tril(diagonal=-1)`) by default to exclude self-token attention at step $t$ from past-context accumulation, matching Definition 4 of the BDH paper, while leaving strict causality togglable via configuration.
- **Feed-Forward ReLU-Lowrank Projection**: Implement sparse positive activations ($x_{sparse}, y_{sparse}$) using low-rank encoder/decoder parameter matrices combined with non-negative ReLU activation thresholds and elementwise product interactions.
- **Optimizer Parameter Grouping**: Separate model parameters in `configure_optimizers()` such that 2D weight matrices (encoder, decoder, encoder_v, lm_head, embed) receive weight decay, while 1D vectors, LayerNorm parameters, and biases receive zero weight decay.
- **CLI & Workflow Hooks**: Extend CLI model factory functions (`instantiate_model` and `create_model_from_dict`) to recognize `bdh` model configuration types and output rich status notifications during model setup.
- **Domain Terms & ADRs**: Record canonical BDH domain terms (*Dragon Hatchling (BDH)*, *BDH-GPU*, *ReLU-lowrank*, *Sparse Positive Activation*) in the domain glossary (`CONTEXT.md`) and publish Architectural Decision Record 0009 documenting the design rationale.

## Testing Decisions

- **Test Philosophy**: Tests must focus strictly on observable external behavior and contracts (tensor shapes, loss values, optimizer parameter groups, generated output shapes) rather than internal state internals.
- **Tested Components**:
  - Model initialization and parameter count verification.
  - Forward pass without targets (logits shape check `(B, T, V)`).
  - Forward pass with targets (logits shape check `(B, T, V)` and scalar loss output).
  - Autoregressive generation (output shape check `(B, T + max_new_tokens)` under standard and top-k sampling).
  - Optimizer configuration (AdamW parameter group count and weight decay assignments).
  - End-to-end single-step optimization and loss backward pass.
- **Prior Art**: Follow test patterns established in `tests/test_gpt.py`, `tests/test_pam.py`, `tests/test_xlstm.py`, and `tests/test_cli.py`.

## Out of Scope

- Custom CUDA or Triton C++ extensions (pure PyTorch implementation is used for full portability and `torch.compile` compatibility).
- Spiking-neuron asynchronous graph simulation engine (BDH-GPU state-space tensor formulation is implemented).
- Multi-GPU distributed data parallel (DDP) orchestrators beyond existing trainer capabilities.

## Further Notes

- BDH exhibits Transformer-like scaling laws while maintaining monosemantic sparse positive activations (~5% sparsity).
- Reference code and paper details are documented in `paper/THE DRAGON HATCHLING.md` and `docs/adr/0009-bdh-dragon-hatchling-integration.md`.
