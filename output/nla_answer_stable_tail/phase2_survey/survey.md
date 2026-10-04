# Survey and screening notes

## Inclusion criteria

Included: primary papers or official technical reports that test activation interpretation, explicit-CoT faithfulness, self-correction/verification, adaptive reasoning or early stopping. Preprints are marked as such and are not treated as settled evidence. Excluded: unverified blogs and claims without an inspectable paper or official code.

## Taxonomy

| Theme | Representative sources | What is actually measured |
|---|---|---|
| NLA / activation language | Fraser-Taliente et al. 2026 | activation reconstruction, constructed tasks, audit utility |
| Explanation faithfulness | Turpin et al. 2023; Lanham et al. 2023; Young et al. 2025 | effects of hidden hints, interventions, CoT reports |
| Correctness and confidence | Zhang et al. 2025; Farquhar et al. 2024 | probe decodability; semantic uncertainty |
| Answer convergence / stopping | Liu & Wang 2025; Yang et al. 2026; Min et al. 2026 | prefix answer agreement, stopping trade-off |
| Overthinking after correctness | Caldarella et al. 2026; Wu et al. 2026 | first-correct prefix, final-answer drift |
| Causal interpretability | Meng et al. 2022; Goldowsky-Dill et al. 2023; Marks et al. 2024 | patch/interchange effects and methodological sensitivity |
| Sparse representations | Cunningham et al. 2023; Templeton et al. 2024 | sparse reconstruction and feature interpretability |

## Survey conclusion

The closest existing label is **answer convergence / early stopping in reasoning**, with **verbose overthinking** for harmless surplus tokens and **harmful overthinking** for tokens that later derail a correct trajectory. “Answer-Stable Tail” can remain useful as a precise operational term for a suffix after an independently estimated answer-stability boundary, but it should not name a presumed cognitive mechanism.
