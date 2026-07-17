Status: ready-for-agent

# Spec: Model Training Environment

## Problem Statement

The user wants to create a flexible, modular, and extensible training environment for different language model architectures. Currently, the code in `old/` is a set of hardcoded scripts (like `train.py`, `model.py`) optimized for minGPT on specific datasets. Developing and comparing new architectures is difficult because there are no clear, decoupled interfaces for trainer logic, datasets, neural network models, logging, or generic callback hooks.

## Solution

Build a modular Python library under `src/` using clean abstractions for the core components of machine learning experiments:
1.  **Trainer**: Orchestrates the loop (forward/backward/stepping) and triggers callbacks.
2.  **Model**: Base class extending `torch.nn.Module` with a standardized contract for optimization, forward passes, and generation.
3.  **Dataset**: Base class for data loading with a custom `get_batch` method.
4.  **Callback**: Base class for event-driven hooks (`on_train_start`, `on_train_end`, `on_step_start`, `on_step_end`, `on_eval_start`, `on_eval_end`). Loggers (TensorBoard and Terminal/File) are subclasses of `Callback`.

A CLI powered by `typer` and `rich` will expose three commands:
*   `train`: Start a training run from a YAML configuration file with CLI parameter overrides, writing output logs and checkpoints to a run-specific directory.
*   `eval`: Load a trained model checkpoint and run validation evaluation.
*   `inference`: Load a model checkpoint and sample text generation given a prompt.

## User Stories

1.  As a machine learning engineer, I want a `train` command, so that I can start a new training run of a language model.
2.  As a machine learning engineer, I want to specify a YAML configuration file, so that my hyperparameter configurations are reproducible.
3.  As a machine learning engineer, I want to override config values via CLI parameters, so that I can quickly experiment without modifying the config file.
4.  As a machine learning engineer, I want training logs, checkpoints, and metrics to be saved to a run-specific directory, so that I can compare different experiments.
5.  As a machine learning engineer, I want to log metrics to TensorBoard, so that I can visualize training curves like loss and learning rate.
6.  As a machine learning engineer, I want logs to be printed to the terminal, so that I can monitor progress in real-time.
7.  As a machine learning engineer, I want terminal logs to also be written to a file in the output folder, so that I have a persistent text record of the training stdout.
8.  As a machine learning engineer, I want an `eval` command, so that I can evaluate a trained model's performance on the validation split.
9.  As a machine learning engineer, I want an `inference` command, so that I can generate text completions from a model checkpoint.
10. As a machine learning engineer, I want the CLI output to be visually appealing and rich with emojis, so that it's easy to read and trace.
11. As a developer, I want generic base classes for callbacks, loggers, models, trainer, and data, so that I can easily support new architectures or datasets in the future.
12. As a developer, I want the system to auto-tokenize and download the Shakespeare dataset if it is missing, so that I don't have to manually execute a prepare step.

## Implementation Decisions

### Core Abstractions (Base Classes)

*   `Callback` (`src/base/callback.py`):
    *   Defines lifecycle hooks: `on_train_start(trainer)`, `on_train_end(trainer)`, `on_step_start(trainer)`, `on_step_end(trainer, metrics)`, `on_eval_start(trainer)`, and `on_eval_end(trainer, metrics)`.
*   `Logger` (`src/base/logger.py`):
    *   Subclasses `Callback` to handle metric and event logging.
*   `Model` (`src/base/model.py`):
    *   Inherits from `torch.nn.Module`.
    *   Exposes `forward(idx, targets=None) -> (logits, loss)`.
    *   Exposes `generate(idx, max_new_tokens, temperature, top_k) -> idx`.
    *   Exposes `configure_optimizers(weight_decay, learning_rate, betas, device_type) -> Optimizer`.
*   `Dataset` (`src/base/dataset.py`):
    *   Exposes `get_batch(split, batch_size, block_size, device) -> (X, Y)`.
    *   Implements or triggers automated raw data downloading and encoding/preparation.

### Core Modules

*   `Trainer` (`src/trainer.py`):
    *   Accepts model, dataset, configuration, and a list of callbacks/loggers.
    *   Manages gradient accumulation steps, learning rate decay, mixed precision training context, gradient scaling, optimization steps, and compilation settings.
    *   Orchestrates evaluation intervals and runs.
*   `GPTModel` (`src/models/gpt.py`):
    *   A port of the GPT implementation from `old/model.py` adapted to inherit from `Model`.
*   `ShakespeareDataset` (`src/datasets/shakespeare.py`):
    *   Loads the Shakespeare corpus from `train.bin` and `val.bin`. If the directory/files do not exist, it downloads `input.txt` and tokenizes it using `tiktoken` automatically.
*   `TerminalLogger` (`src/callbacks/terminal_logger.py`):
    *   Prints pretty progress logs to terminal with emojis using `typer`/`rich`. Writes identical text log output to `out.log` in the run's output directory.
*   `TensorBoardLogger` (`src/callbacks/tensorboard_logger.py`):
    *   Writes scalar metrics (loss, learning rate, step time, etc.) to TensorBoard event files in the run's output directory.
*   `CheckpointCallback` (`src/callbacks/checkpoint.py`):
    *   Saves the model state dictionary, optimizer state, configuration, and best loss to `best_ckpt.pt` and `last_ckpt.pt` in the run's output directory.

### CLI (`src/cli.py` and `main.py`)

*   `train` command:
    *   Accepts `--config` path.
    *   Accepts `--run-name` (defaults to `run_YYYYMMDD_HHMMSS`).
    *   Accepts optional CLI overrides for hyperparameters (e.g., `--learning-rate`, `--batch-size`).
    *   Creates the output directory and instantiates all modules before launching `trainer.train()`.
*   `eval` command:
    *   Accepts `--checkpoint` path.
    *   Loads the model weights and configurations, and runs evaluation on the validation split.
*   `inference` command:
    *   Accepts `--checkpoint` path.
    *   Accepts `--prompt` (as string or path to a file).
    *   Accepts `--max-new-tokens`, `--temperature`, and `--top-k`.
    *   Generates and prints output text.

## Testing Decisions

*   We will test the system primarily at two seams:
    1.  **CLI Command Interface (highest seam)**: Using `typer.testing.CliRunner` to execute commands (`train`, `eval`, `inference`) with dummy settings and checking return code, print statements, and expected log files.
    2.  **Trainer Hooks (unit seam)**: Injecting mock dataset, model, and callback instances to assert that all trainer hooks are executed at correct steps.
*   A good test verifies external behavior: that a training command successfully writes TB logs, prints progress, generates files, and saves checkpoints, and that the inference command decodes tokens back to strings correctly.

## Out of Scope

*   Distributed Data Parallel (DDP) multi-GPU training.
*   Weights & Biases integration.
*   Advanced learning rate scheduling policies (other than cosine decay with warmup).

## Further Notes

*   Ensure code is fully typed and adheres to standard clean Python formatting.
