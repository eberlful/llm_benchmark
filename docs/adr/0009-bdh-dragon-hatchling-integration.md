# 9. Dragon Hatchling (BDH) Model Architecture Integration

Date: 2026-08-13

## Status

Accepted

## Context

The Dragon Hatchling (BDH) paper introduces a biologically inspired, scale-free sequence modeling architecture. BDH formulates attention and feed-forward transformations as local graph dynamics across $n$ neuron particles, using linear Rotary Position Embedding (RoPE) attention, sparse non-negative activation functions (ReLU), and low-rank projection parameter matrices ($E, D_x, D_y$).

To compare BDH against existing architectures (GPT, PAM, xLSTM) in `llm_benchmark`, we need a clean, standard PyTorch implementation compatible with the `Model` interface, CLI tools, and training engine.

## Decision

1. **Architecture Implementation**: Implement `BDHModel` in `src/models/bdh.py` inheriting directly from `src.base.model.Model`.
2. **Causal Attention**: Maintain strict paper causality (`tril(diagonal=-1)`) by default in `Attention.forward` to exclude self-token attention during past-context key-value accumulation ($\sum_{\tau < t}$), while adding a `strict_causal` configuration flag.
3. **Parameter Optimization**: Configure AdamW optimizer parameter groups in `configure_optimizers()` by decaying 2D parameter matrices (`encoder`, `decoder`, `encoder_v`, `lm_head`, `embed.weight`) while excluding 1D parameters, LayerNorm weights, and buffers.
4. **CLI & Benchmark Integration**: Register `bdh` in `src/cli.py` and provide standard YAML configs `configs/train_cpu_light_bdh.yaml` and `configs/train_shakespeare_bdh.yaml`.

## Consequences

- Full compatibility with the `llm_benchmark` training, logging, and evaluation infrastructure.
- High training portability across CPU and CUDA execution backends without custom CUDA extensions.
- Standardized benchmarking of BDH scaling and performance against Transformer and SSM baselines.
