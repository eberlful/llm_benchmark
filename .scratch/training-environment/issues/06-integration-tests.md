# 06 — CLI Integration Tests

**What to build:**
A comprehensive test suite verifying the CLI commands (`train`, `eval`, `inference`) and checking for correct exit codes and outputs.

**Blocked by:** 05 — CLI Interface (train, eval, inference)

**Status:** completed

- [x] A test module is created at `tests/test_cli.py`.
- [x] Uses `typer.testing.CliRunner` to invoke `train`, `eval`, and `inference` commands.
- [x] Tests a fast debugging configuration (tiny GPT model, small context, 5 iterations) to assert that:
    - [x] The CLI runs without error (exit code 0).
    - [x] An output directory is created with `out.log`, checkpoints, and TensorBoard events.
    - [x] `eval` can load the saved checkpoint.
    - [x] `inference` can generate text from the saved checkpoint.
- [x] Verification is performed by running the test suite via pytest.
