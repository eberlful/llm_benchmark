# 03 — CLI integration and YAML config

**What to build:**
Update the CLI tool in `src/cli.py` to instantiate `PAMModel` when `type: pam` is provided in the configuration. Define a training configuration file for Shakespeare using the PAM model.

Requirements:
- Parse `type` field from the YAML model configuration, defaulting to `gpt`.
- Map `block_size` to `max_seq_len` and `n_embd` to `dim` in the model configuration dictionary when instantiating `PAMModel`.
- Create `configs/train_shakespeare_pam.yaml` with a tiny PAM model suitable for Shakespeare training.

**Blocked by:** 02 — PAM Model wrapper and generation

**Status:** resolved
**Triage Role:** resolved

- [x] CLI command `train` supports instantiating `PAMModel`.
- [x] CLI command `eval` supports loading `PAMModel` from checkpoint config metadata.
- [x] CLI command `inference` supports loading `PAMModel` and generating text.
- [x] `configs/train_shakespeare_pam.yaml` contains a small PAM configuration ready for training.
