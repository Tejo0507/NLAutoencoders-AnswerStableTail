# NLA-AnswerStableTail

**Diagnosing redundant self-verification in LLM reasoning using Natural Language Autoencoders**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-research--in--progress-orange)]()

---

## Overview

Language models frequently enter an **Answer-Stable Tail** during complex reasoning, a phase where the problem is functionally solved, but the model continues generating redundant self-verification tokens, wasting inference compute.

Existing detection methods fall short:
- **Entropy-based signals** capture statistical uncertainty, not *semantic* hesitation.
- **Chain-of-Thought monitoring** is structurally limited, the visible text often omits the model's true internal state.

This project applies **Natural Language Autoencoders (NLAs)**, introduced by Anthropic for safety auditing, to a new problem: explaining *why* models overthink, not just detecting *that* they do.

## Approach

An NLA translates a model's internal activations into readable text through two jointly-trained modules:

- **Activation Verbalizer (AV)** generates a text explanation from a hidden-state vector
- **Activation Reconstructor (AR)** maps that text back into a reconstructed activation, minimizing MSE

**RECAP** (Readable Encodings via Co-trained Auxiliary Predictors) adds independent linear probes that verify the AV's explanations are grounded in the actual activation, rather than confabulated.

## Resources

| | |
|---|---|
| **Target models** | Qwen 2.5 (7B), Gemma 3 (12B / 27B), Llama 3.3 (70B) |
| **Activation layer** | ~2/3 depth |
| **Reasoning corpora** | GSM8K, MATH |
| **NLA training data** | FineWeb (1M-sample slices) |

## Method

1. **Setup & Extraction** - run a target model on reasoning datasets to trigger the Answer-Stable Tail; extract activations during redundant tokens
2. **NLA Integration** - load pretrained AV/AR checkpoints
3. **Probe Training (RECAP)** - train linear probes for faithfulness verification
4. **Diagnostic Generation** - feed extracted vectors into the AV
5. **Analysis** - evaluate for semantic markers of doubt vs. active problem-solving, cross-checked against RECAP

## Status

Research in progress.

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

## License

This project is licensed under the [MIT License](LICENSE).
