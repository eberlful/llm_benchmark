# 05 — CLI Interface (train, eval, inference)

**What to build:**
A CLI entry point exposing `train`, `eval`, and `inference` commands using `typer` and `rich`, wrapping all modules together.

**Blocked by:** 04 — TensorBoard Logger & Checkpointing

**Status:** completed

- [x] A Typer CLI app is created at `src/cli.py` exposing commands `train`, `eval`, and `inference`.
- [x] The `train` command parses a YAML configuration file path, applies CLI overrides (learning rate, steps, batch size), creates a unique output directory under `runs/`, builds the pipeline, and calls `trainer.train()`.
- [x] The `eval` command loads a specified model checkpoint and computes the final validation loss.
- [x] The `inference` command loads a checkpoint and generates a string sequence starting from a specified prompt (accepts file or string prompt, temperature, top-k).
- [x] Emojis and visual formatting are used throughout the command line outputs.
- [x] The root `main.py` is updated to invoke the Typer CLI app.
