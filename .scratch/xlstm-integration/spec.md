# xLSTM Model Integration Spec

**Status:** completed

## Problem Statement

The user wants to evaluate and benchmark the **xLSTM** (Extended Long Short-Term Memory) sequence modeling architecture alongside GPT and PAM language models in `llm_benchmark`. Currently, the environment only supports Transformer-based GPT and complex-valued Phase-Associative Memory (PAM) models.

## Solution

Integrate `xLSTMModel` into `src/models/xlstm.py`, supporting both **mLSTM** (Matrix LSTM with covariance update and parallel training) and **sLSTM** (Scalar LSTM with exponential gating, max-stabilized state tracking, and memory mixing) residual blocks. Provide pure PyTorch parallel training, recurrent step inference generation, CLI registration (`model.type: "xlstm"`), YAML benchmark configs, and unit tests.

## User Stories

1. As an LLM researcher, I want to run xLSTM training runs using YAML configurations, so that I can compare its training loss and throughput against GPT and PAM models.
2. As a benchmark operator, I want to configure the layer pattern of xLSTM (such as pure mLSTM, pure sLSTM, or hybrid 7:1 ratio), so that I can evaluate different block composition strategies.
3. As a developer, I want mLSTM training to run in pure PyTorch without external native CUDA/Triton dependencies, so that training works portably across CPU and GPU hardware and supports `torch.compile`.
4. As a language model consumer, I want `xLSTMModel` to support autoregressive text generation (`generate`), so that I can sample tokens token-by-token using recurrent state updates.
5. As a test engineer, I want automated unit tests verifying xLSTM model forward passes, loss computation, and generation, so that regressions are prevented.

## Implementation Decisions

- **Block Variants**: Implement both mLSTM and sLSTM blocks within a unified residual block architecture.
- **Parallel & Recurrent Modes**: Use sequence-wide lower-triangular causal decay matrices for vectorized parallel training of mLSTM, and step-by-step state matrix updates for `generate(...)`.
- **Configuration Schema**: `xLSTMConfig` with `block_type_pattern` supporting preset string ratios (e.g. `"mlstm"`, `"slstm"`, `"7:1"`) or explicit lists.
- **Positional Embeddings**: Controlled by `use_pos_emb: true/false` (default `true`).
- **CLI & Benchmark Integration**: Register `model.type: "xlstm"` in `src/cli.py` (`create_model_from_dict`), and supply `configs/train_shakespeare_xlstm.yaml` and `configs/train_cpu_light_xlstm.yaml`.
- **ADR & Glossary**: Updated `CONTEXT.md` and recorded `docs/adr/0008-xlstm-pure-pytorch-parallel-formulation.md`.

## Testing Decisions

- Test high-level model interface via `create_model_from_dict` and `Model` base class (`forward` and `generate`).
- Test internal module equivalency between parallel matrix formulation and recurrent step state update for mLSTM.
- Existing prior art: `tests/test_pam.py` and `tests/test_gpt.py`.

## Out of Scope

- External CUDA/Triton native kernels (focus is pure PyTorch portability and `torch.compile` compatibility).
- Bi-directional sequence modeling (xLSTM in this project is strictly causal language modeling).

## Further Notes

- All code execution uses `uv`.
