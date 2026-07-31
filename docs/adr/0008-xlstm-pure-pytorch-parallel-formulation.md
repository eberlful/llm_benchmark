# 8. Pure PyTorch Parallel Causal Formulation for xLSTM

Date: 2026-07-31

## Status

Accepted

## Context

Integrating xLSTM into `llm_benchmark` requires an efficient forward pass for training mLSTM blocks over sequences. Official xLSTM implementations often rely on custom CUDA or Triton kernels for maximum hardware throughput. However, requiring compiled C++/Triton extensions limits portability across hardware (e.g. CPU vs CUDA) and complicates execution with `torch.compile`.

## Decision

We decide to implement mLSTM training using a pure PyTorch vectorized **Parallel Causal Matrix formulation**. For autoregressive sequence generation (`generate`), we implement a **Recurrent Step formulation**.

## Consequences

- High portability: Runs seamlessly on CPU and GPU backends.
- Full `torch.compile` compatibility without native C++ compilation steps.
- Native maintenance and debugging within standard PyTorch tensor ops.
