# NLA-AnswerStableTail

**Natural Language Autoencoders for Redundancy Diagnostics**

*Diagnosing redundant self-verification in LLM reasoning using Natural Language Autoencoders*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-research--in--progress-orange)]()

---

## Overview

Language models frequently enter an **Answer-Stable Tail** during complex reasoning, a phase where the problem is functionally solved, but the model continues generating redundant self-verification tokens, wasting inference compute.

Existing detection methods fall short:
- **Entropy-based signals** capture statistical uncertainty, not *semantic* hesitation.
- **Chain-of-Thought monitoring** is structurally limited, the visible text often omits the model's true internal state.

This project applies **Natural Language Autoencoders (NLAs)**, introduced by Anthropic for safety auditing, to a different problem: explaining *what the model is doing* while it overthinks, not just detecting *that* it does.

## Approach

An NLA translates a model's internal activations into readable text through two jointly-trained modules:

- **Activation Verbalizer (AV)** generates a text explanation from a hidden-state vector
- **Activation Reconstructor (AR)** maps that text back into a reconstructed activation, minimizing MSE

High reconstruction fidelity shows that the explanation retains activation information. It does **not** show that each individual claim inside the explanation is grounded in the activation, so the explanations need an independent faithfulness check.

### A note on RECAP

**RECAP** (Dingeto, 2026) is a *training* intervention: it co-trains the **target model** with linear decodability heads so that pre-specified content stays auditable against a fresh probe. Reproducing RECAP therefore means retraining the target model, which is out of scope here.

This project runs a **RECAP-inspired evaluation** instead. It adopts RECAP's principle of independent verification, using independent probes and verbaliser-only controls against released checkpoints, without reproducing the RECAP training procedure. The distinction matters: RECAP does not validate the explanations produced here, it motivates how they are checked.

## Resources

| | |
|---|---|
| **Target model** | Qwen 2.5 7B Instruct (layer 20) |
| **Other released checkpoints** | Gemma 3 12B (L32), Gemma 3 27B (L41), Llama 3.3 70B (L53) |
| **Activation layer** | roughly two-thirds depth |
| **Reasoning corpora** | GSM8K, MATH |
| **NLA checkpoints** | released alongside the Transformer Circuits report, not trained here |

## Method

1. **Benchmark preparation** - stage GSM8K and MATH as verified question-answer pairs
2. **Trace generation** - produce reasoning traces, chunk them, parse intermediate answers
3. **Tail identification** - define the Answer-Stable Tail independently of any activation description, using answer equivalence under truncation plus resampled continuations checked against the verifier
4. **Readout and baselines** - apply the AV/AR pair at tail windows, alongside answer-convergence stopping, a hidden-state correctness probe and semantic entropy
5. **Comparison** - evaluate at matched compute budgets
6. **Faithfulness controls** - claim deletion, resampling and paraphrase, scored by reconstruction change against independent-probe and verbaliser-only controls
7. **Causal controls** *(extension)* - truncation and tail replacement with matched-position and matched-random-direction comparisons

## Status

Research in progress, at the proposal stage.

| Component | State |
|---|---|
| Benchmark preparation (step 1) | implemented |
| Steps 2 to 6 | proposed |
| Step 7 (causal controls) | proposed, contingent on available compute |

No experimental result has been produced yet. Any number appearing in this repository comes from the cited literature, not from in-house runs.

## Review document

`Project_Review_II.md` and `Project_Review_II.docx` hold the current Introduction and Literature Review, with every citation and figure checked against its primary source. Diagrams live in `figures/` and are regenerated with `python scripts/make_figures.py`; the Word build is `python scripts/build_docx.py`.

Earlier literature working notes are kept under `output/` as the audit trail behind the review.

## Citation

If you use this work, please cite the underlying NLA methodology:

```bibtex
@article{frasertaliente2026nla,
  author  = {Fraser-Taliente, Kit and Kantamneni, Subhash and Ong, Euan and et al.},
  title   = {Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations},
  journal = {Transformer Circuits Thread},
  year    = {2026},
  url     = {https://transformer-circuits.pub/2026/nla/index.html}
}
```

## References

Fraser-Taliente, K., Kantamneni, S., Ong, E., et al. (2026). [Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations](https://transformer-circuits.pub/2026/nla/index.html). *Transformer Circuits Thread.*

Dingeto, H. (2026). Train the Model, Not the Reader: Decodability Supervision for Verifiable Activation Explanations. Preprint, [arXiv:2607.20379](https://arxiv.org/abs/2607.20379).

Li, M., Ceballos Arroyo, A. M., Rogers, G., Saphra, N., & Wallace, B. C. (2026). Do Activation Verbalization Methods Convey Privileged Information? *ICML 2026*, [arXiv:2509.13316](https://arxiv.org/abs/2509.13316).

The full reference list, with publication status marked for every entry, is in the review document.

## License

This project is licensed under the [MIT License](LICENSE).
