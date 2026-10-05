# Vendored: `nla_inference.py`

| | |
|---|---|
| **Upstream** | https://github.com/kitft/natural_language_autoencoders |
| **File** | `nla_inference.py` (repository root), unmodified |
| **Licence** | Apache License 2.0 — full text in `LICENSE` beside this file |
| **Vendored** | 2026-10-05 |
| **Accompanying report** | Fraser-Taliente, K., Kantamneni, S., Ong, E., et al. (2026). *Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations.* Transformer Circuits Thread. https://transformer-circuits.pub/2026/nla/index.html |

## Why it is vendored rather than installed

Upstream ships this as an explicitly standalone, single-file module ("no `nla`
package deps") intended to be shipped alongside the checkpoints. It is not
published to PyPI and the surrounding repository pulls in `sglang[all]`, Megatron
and FSDP training code that this project neither needs nor can install on
Windows. Vendoring the one file, unmodified and with its licence, is the
lowest-risk way to use the authors' own definitions.

## What this project uses from it

- `normalize_activation` — rescale a vector to a target L2 norm.
- `inject_at_marked_positions` — overwrite embedding rows at the injection
  marker, with left/right neighbour validation.
- `load_nla_config` / `NLAConfig` — read the checkpoint's `nla_meta.yaml`.
- `load_embedding_only`, `resolve_embed_scale` — embedding-table access.
- `NLACritic` — the activation reconstructor, used essentially as shipped.

These are the correctness-critical definitions. Reimplementing them from the
paper would risk exactly the kind of silent convention drift the upstream
docstrings warn about (wrong injection position → the model reads the literal
marker character and emits CJK text).

## What this project replaces

`NLAClient` drives the verbaliser through an **SGLang server** using its
`input_embeds` API. SGLang does not support Windows and would not fit in this
machine's 6.44 GB of VRAM alongside the model. `src/nlaast/nla/verbalizer.py`
substitutes a local `transformers` generation path that reproduces the same
arithmetic step for step — chat template → embedding lookup → `normalize_activation`
to `injection_scale` → `inject_at_marked_positions` → `generate(inputs_embeds=...)`
— calling the vendored functions for the injection math rather than restating it.

The substitution is a backend change, not a method change, and it is recorded as
such in `PROJECT_PLAN.md` §3 and in `docs/DECISIONS.md` D1, which also lists
what `tests/test_inputs_embeds_contract.py` verifies about it.

## Modifications

None. The file is byte-identical to upstream `main` as of the vendoring date.
Project-specific behaviour lives in `src/nlaast/nla/`, which imports from here.
