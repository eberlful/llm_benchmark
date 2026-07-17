# 03 — Trainer & Terminal Logger

**What to build:**
The core training coordinator (`Trainer`) and a callback-based `TerminalLogger` to format stdout nicely using emojis and log execution metrics to a persistent file.

**Blocked by:** 01 — Shakespeare Dataset with Automated Download & Prep, 02 — GPT Model Architecture Port

**Status:** ready-for-agent

- [ ] A generic `Callback` base class is created at `src/base/callback.py` defining lifecycle hooks (`on_train_start`, `on_train_end`, `on_step_start`, `on_step_end`, `on_eval_start`, `on_eval_end`).
- [ ] A `Logger` base class inheriting from `Callback` is created at `src/base/logger.py`.
- [ ] The `Trainer` class is created at `src/trainer.py` orchestrating the standard training steps (autocast context, scaling gradients, backward pass, step, optimizer zeroing, cosine decay schedule, periodic eval runs).
- [ ] A `TerminalLogger` is implemented at `src/callbacks/terminal_logger.py` subclassing `Logger`. It prints metrics with emojis using Typer/Rich and mirrors terminal output to `out.log` in the run directory.
- [ ] A test script can successfully run 5 steps of `GPTModel` training on `ShakespeareDataset` with logs printed and recorded.
