# Frontier scan — 25 September 2026

## Scope and search record

Primary-source searches covered the NLA paper and code, activation explanations, CoT faithfulness, answer convergence, hidden-state correctness, semantic entropy, self-correction, and dynamic early exit. Sources were searched on Transformer Circuits, ACL Anthology, ICLR/ICML/NeurIPS proceedings, arXiv, Nature, OpenReview, and official code repositories.

## Main frontier findings

1. **NLAs are a real, released method, but not a semantic ground-truth instrument.** Fraser-Taliente et al. (2026) jointly optimize a verbalizer and reconstructor through a language bottleneck. They report informative audit examples, quantitative constructed-ground-truth tasks, and explicit confabulation, layer-sensitivity, and private-code risks.
2. **The proposed tail overlaps materially with established work.** Liu & Wang (EMNLP 2025) study answer convergence for early stopping; Zhang et al. (COLM 2025) probe hidden-state correctness; Caldarella et al. (2026 preprint) formalize post-correctness harmful versus verbose overthinking.
3. **A crucial negative result appeared after NLA.** Dingeto (2026 preprint) shows that reconstruction can preserve gist while individual verbalizer claims are not reconstruction-dependent, and introduces RECAP as *target-model co-training*, not a post hoc probe check.
4. **No source establishes that an NLA sentence reports a model's subjective state.** The strongest current reading is activation-conditioned hypothesis generation whose claims require independent validation.

## Included primary works

- Fraser-Taliente et al. 2026, *Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations* (Transformer Circuits; project report).
- Liu & Wang 2025, *Answer Convergence as a Signal for Early Stopping in Reasoning* (EMNLP).
- Zhang et al. 2025, *Reasoning Models Know When They're Right: Probing Hidden States for Self-Verification* (COLM).
- Huang et al. 2024, *Large Language Models Cannot Self-Correct Reasoning Yet* (ICLR).
- Farquhar et al. 2024, *Detecting Hallucinations in Large Language Models Using Semantic Entropy* (Nature).
- Caldarella et al. 2026, *Thinking Past the Answer* (arXiv preprint).
- Dingeto 2026, *Train the Model, Not the Reader* (arXiv preprint).
