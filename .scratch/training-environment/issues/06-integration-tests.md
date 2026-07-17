# 06 — CLI Integration Tests

**What to build:**
A comprehensive test suite verifying the CLI commands (`train`, `eval`, `inference`) and checking for correct exit codes and outputs.

**Blocked by:** 05 — CLI Interface (train, eval, inference)

**Status:** ready-for-agent

- [ ] A test module is created at `tests/test_cli.py`.
- [ ] Uses `typer.testing.CliRunner` to invoke `train`, `eval`, and `inference` commands.
- [ ] Tests a fast debugging configuration (tiny GPT model, small context, 5 iterations) to assert that:
    - The CLI runs without error (exit code 0).
    - An output directory is created with `out.log`, checkpoints, and TensorBoard events.
    - `eval` can load the saved checkpoint.
    - `inference` can generate text from the saved checkpoint.
- [ ] Verification is performed by running the test suite via pytest.
