# 05 — CLI Interface (train, eval, inference)

**What to build:**
A CLI entry point exposing `train`, `eval`, and `inference` commands using `typer` and `rich`, wrapping all modules together.

**Blocked by:** 04 — TensorBoard Logger & Checkpointing

**Status:** ready-for-agent

- [ ] A Typer CLI app is created at `src/cli.py` exposing commands `train`, `eval`, and `inference`.
- [ ] The `train` command parses a YAML configuration file path, applies CLI overrides (learning rate, steps, batch size), creates a unique output directory under `runs/`, builds the pipeline, and calls `trainer.train()`.
- [ ] The `eval` command loads a specified model checkpoint and computes the final validation loss.
- [ ] The `inference` command loads a checkpoint and generates a string sequence starting from a specified prompt (accepts file or string prompt, temperature, top-k).
- [ ] Emojis and visual formatting are used throughout the command line outputs.
- [ ] The root `main.py` is updated to invoke the Typer CLI app.
