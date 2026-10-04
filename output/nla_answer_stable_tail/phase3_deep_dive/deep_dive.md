# Deep-reading notes

## Fraser-Taliente et al. (2026), NLA — project report
**Method.** An activation verbalizer maps a unit-normalized residual activation to text; an activation reconstructor maps sampled text back to the vector. AV and AR are warm-started using context summaries, then jointly optimized with RL/supervised updates for expected squared reconstruction error plus a KL term. FVE is the reconstruction measure. Released AV/AR pairs support Qwen2.5-7B layer 20, Gemma-3-12B layer 32, Gemma-3-27B layer 41, and Llama-3.3-70B layer 53.

**Evidence.** Constructed prediction tasks improve during training; audit case studies are corroborated by training-data inspection, SAE evidence, attribution, or steering. **Limit.** This does not establish that every sentence is true. The authors explicitly report confabulations, model and layer sensitivity, text inversion/steganography risks, and roughly 500 generated tokens per activation.

## Dingeto (2026), RECAP — preprint
**Method.** Tests whether individual explanation claims alter reconstruction; uses synthetic truth, evaluator swapping, and separate linear heads co-trained with the target model to preserve designated content's decodability. **Evidence.** Reports that reconstruction can be high when few specific claims are reconstruction-dependent; standard recipes form co-adapted codes in controlled runs. **Limit.** RECAP supports only predesignated content and requires target-model training; it does not establish that arbitrary free-form AV prose is true or causally relevant.

## Liu & Wang (2025), Answer Convergence — EMNLP
**Method.** Split CoT into sentence chunks, solicit/interrogate intermediate answers, and stop after agreement; also learn stopping from hidden states. **Evidence.** Across five benchmarks and five open models, reports substantial token reductions with limited average accuracy loss. **Limit.** Agreement is an outcome proxy. It can be stable before the model finds a missed constraint, and does not classify a suffix's semantic role.

## Zhang et al. (2025), hidden-state self-verification — COLM
**Method.** Label intermediate answers by correctness and train probes on answer-position representations. **Evidence.** In-distribution probe AUC exceeds 0.7 with calibrated scores; mathematical-domain transfer is better than cross-domain transfer; probe-guided early exit reduces tokens. **Limit.** A decodable variable is information availability, not proof the model uses it or a verification mechanism causes the answer.

## Caldarella et al. (2026), Thinking Past the Answer — preprint
**Method.** Defines a first correct prefix and tests later behavior. Separates harmless verbose overthinking from harmful answer drift. **Evidence.** Reports early correct prefixes and accuracy improvements from stopping at them in selected tasks. **Limit.** A first correct surface answer can occur by chance; prefix evaluation needs repeated continuation and verifier controls.

## Farquhar et al. (2024), semantic entropy — Nature
**Method.** Samples generations, groups them by semantic equivalence, and estimates entropy over meanings. **Evidence.** Outperforms lexical entropy and several confidence baselines for confabulation detection across tested models and tasks. **Limit.** It measures uncertainty over outcomes, not whether current text is verification or whether compute is redundant.

## Huang et al. (2024), self-correction — ICLR
**Method.** Controlled comparison of repeated self-correction without external feedback. **Evidence.** Finds degradation or no gains under their intrinsic setting; identifies common evaluation confounds. **Limit.** Does not prove all post-answer reasoning is useless, especially for models trained to verify or with external verifiers.

## Goldowsky-Dill et al. (2023), activation patching — preprint
**Method.** Reviews metric, corruption, restoration, and normalization choices in activation patching. **Evidence.** Shows interpretation conclusions can change with choices. **Limit.** Reinforces that a planned NLA-direction intervention needs matched random directions, norm controls, multiple patch sites, and downstream behavior measures.
