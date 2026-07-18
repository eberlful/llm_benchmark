# 04 — Tests and verification

**What to build:**
Write unit tests in `tests/test_pam.py` and expand `tests/test_cli.py` to cover the new PAM code path.

Tests must verify:
- Complex math operators (`cmul`, `cconj`, `cabs`, `cnormalize`).
- Forward-pass logic with and without targets (verifying shape and loss computation).
- Equivalence between parallel forward pass and recurrent step-by-step state forward pass at logits level.
- Optimizer parameter grouping (proper categorization of decay vs no-decay parameters).
- End-to-end integration via the CLI commands (`train`, `eval`, `inference`) on a dummy PAM configuration.

**Blocked by:** 03 — CLI integration and YAML config

**Status:** resolved
**Triage Role:** resolved

- [x] `tests/test_pam.py` is created and tests all math helpers.
- [x] `tests/test_pam.py` verifies forward passes and parallel vs recurrent logit parity.
- [x] `tests/test_pam.py` verifies optimizer weight decay grouping logic.
- [x] `tests/test_cli.py` is updated to include an end-to-end integration test with a tiny PAM configuration.
- [x] All tests run and pass using `pytest`.
