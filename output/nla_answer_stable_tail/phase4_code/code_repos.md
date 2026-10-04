# Code and checkpoint landscape

| Resource | Use | Practical finding |
|---|---|---|
| https://github.com/kitft/natural_language_autoencoders | Official NLA library | Released AV/AR checkpoints and extraction/injection metadata; full training is GPU-intensive. |
| https://github.com/EleutherAI/pythia | Open interpretability model suite | Useful controlled target family for smaller causal studies and layer sweeps. |
| https://github.com/TransformerLensOrg/TransformerLens | activation access and interventions | Suitable for hooks, activation caching, logit lens, and patching on supported open models. |
| https://github.com/TransformerLensOrg/sae_lens | SAE baselines | Provides sparse-feature comparison where available. |
| https://github.com/ricyoung/cot-faithfulness-open-models | CoT faithfulness evaluation | Provides an adjacent benchmark and controls for textual explanations. |

## Repository under review

The local NLA-AnswerStableTail repository contains only a downloader, requirements, and raw subsets of GSM8K/MATH. It has no model runner, prompt template, activation extraction, NLA loading, probe training, labels, evaluation scripts, checkpoints, seed control, or experiment configuration. It is a data staging skeleton, not yet an experimental artifact. This is a critical reproducibility limitation rather than a defect in a completed pipeline.
