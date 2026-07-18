# 01 — PAM Model core implementation

**What to build:**
Implement complex math helper functions, activation functions (`ModReLU`, `ModSwish`, `PhaseModulatedActivation`), `ComplexLinear`, `ComplexNorm`, `ComplexGatedUnit` (CGU) channel mixer, and `V11PAMLayer` sequence mixer inside `src/models/pam.py`.

The sequence mixer must support:
- Fused and unfused projection of Q, K, V.
- Complex Rotary Position Embeddings (RoPE) applied to Q and K.
- Gated State Protection (GSP) with learnable protect gate.
- Parallel chunked forward paths (`_forward_chunked_head`, `_forward_chunked_perchannel`, `_forward_delta`, `_forward_multistate`).
- Recurrent step calculations (`_recur_step_additive`, `_recur_step_delta`).

**Blocked by:** None — can start immediately

**Status:** resolved
**Triage Role:** resolved

- [x] Complex math operations (`cmul`, `cconj`, `cabs`, `cnormalize`, `to_real_concat`, `fused_decay_matrix`) are implemented.
- [x] Complex NN components (`ComplexLinear`, `ComplexNorm`, `ComplexEmbed`, `ComplexPosEmbed`) are defined.
- [x] Activation layers (`ModReLU`, `ModSwish`, `PhaseModulatedActivation`) are defined.
- [x] `ComplexGatedUnit` is implemented.
- [x] `V11PAMLayer` is implemented supporting both recurrent step execution and chunked parallel training modes.
