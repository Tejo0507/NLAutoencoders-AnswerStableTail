# Notices and attribution

These artifacts (CC-BY-4.0, see `LICENSE`) contain or are derived from the
following third-party material. Each licence was read from the upstream
files on 2026-10-18.

## Data whose text is included

- **GSM8K** — `openai/gsm8k` (Hugging Face), from
  https://github.com/openai/grade-school-math. MIT License,
  Copyright (c) 2021 OpenAI. Full text: `licences/GSM8K_MIT_LICENSE.txt`.
  Cobbe et al. (2021), *Training Verifiers to Solve Math Word Problems*,
  arXiv:2110.14168.
- **MATH** — `EleutherAI/hendrycks_math` (Hugging Face), from
  https://github.com/hendrycks/math. MIT License,
  Copyright (c) 2021 Dan Hendrycks. Full text: `licences/MATH_MIT_LICENSE.txt`.
  Hendrycks et al. (2021), *Measuring Mathematical Problem Solving With the
  MATH Dataset*, NeurIPS.

Problem statements and gold answers appear in `data/` and `traces/` of each
run.

## Models whose outputs and activations are included

- **Qwen2.5-7B-Instruct** — `Qwen/Qwen2.5-7B-Instruct`, Apache License 2.0,
  Copyright 2024 Alibaba Cloud. Full text:
  `licences/Qwen2.5-7B-Instruct_LICENSE.txt`. Generated traces and layer
  activations come from this model (run in 4-bit NF4; bf16 in
  `main_f8_bf16`).
- **NLA autoencoder, Qwen2.5-7B-Instruct layer 20** —
  `kitft/nla-qwen2.5-7b-L20-av` (verbaliser) and `kitft/nla-qwen2.5-7b-L20-ar`
  (reconstructor), Apache License 2.0 (same text). Verbalisations and
  reconstructions come from these models. Fraser-Taliente et al. (2026),
  *Natural Language Autoencoders Produce Unsupervised Explanations of LLM
  Activations*, Transformer Circuits Thread. Their model cards state that the
  fine-tuning data was derived from WildChat-1M (ODC-BY) and Ultra-FineWeb
  (Apache-2.0); none of that data is included here.

No model weights are redistributed.

## Code

The code that produced the artifacts is MIT-licensed, except the vendored
`third_party/nla_inference/` (Apache-2.0), in
https://github.com/Tejo0507/NLAutoencoders-AnswerStableTail.
