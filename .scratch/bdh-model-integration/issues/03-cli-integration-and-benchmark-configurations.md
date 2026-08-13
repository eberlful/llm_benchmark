# 03 — CLI Integration & Benchmark Configurations

**What to build:** Integrate BDH model creation into the CLI runner (`src/cli.py`) and provide standard YAML benchmark configs for CPU light training and Shakespeare dataset execution.

**Blocked by:** 01 — Core BDH Model & Config Architecture, 02 — Parameter Optimizer Grouping & Autoregressive Generation

**Status:** completed

- [x] CLI `create_model_from_dict()` and `instantiate_model()` support `type: "bdh"`
- [x] `configs/train_cpu_light_bdh.yaml` created for fast CPU integration tests
- [x] `configs/train_shakespeare_bdh.yaml` created for full Shakespeare training benchmark runs
