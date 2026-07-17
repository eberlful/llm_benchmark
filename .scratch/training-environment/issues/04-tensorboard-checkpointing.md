# 04 — TensorBoard Logger & Checkpointing

**What to build:**
A TensorBoard logging callback and a model checkpointing callback to record metrics visually and save intermediate model states during evaluation.

**Blocked by:** 03 — Trainer & Terminal Logger

**Status:** ready-for-agent

- [ ] A `TensorBoardLogger` is created at `src/callbacks/tensorboard_logger.py` subclassing `Logger` using `SummaryWriter` to write metrics into the output directory.
- [ ] A `CheckpointCallback` is created at `src/callbacks/checkpoint.py` subclassing `Callback` to serialize training metadata (state dicts of model and optimizer, config, steps, best validation loss) as `best_ckpt.pt` and `last_ckpt.pt` in the output directory.
- [ ] The `Trainer` invokes these callbacks correctly during evaluation and step phases.
- [ ] Verification script shows that runs generate TensorBoard files and checkpoint files successfully.
