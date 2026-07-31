# 04 — CLI Model Integration and Benchmark YAML Configurations

**What to build:**
Register `model.type: "xlstm"` in `src/cli.py` (`create_model_from_dict`) and provide benchmark YAML configs (`configs/train_shakespeare_xlstm.yaml` and `configs/train_cpu_light_xlstm.yaml`).

**Blocked by:** 03 — Autoregressive Generation Step and Recurrent State Update

**Status:** completed

- [x] `create_model_from_dict` handles `type: "xlstm"`, instantiating `xLSTMConfig` and `xLSTMModel`.
- [x] `configs/train_shakespeare_xlstm.yaml` created for full Shakespeare training benchmark.
- [x] `configs/train_cpu_light_xlstm.yaml` created for fast CPU testing.
- [x] `python main.py train configs/train_cpu_light_xlstm.yaml --steps 5` runs without errors.
