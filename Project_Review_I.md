# Project Review – I

**Title.** Beyond Readable Hypotheses: An Activation-Explanation-Independent, Causally Validated Study of the Answer-Stable Tail in Large-Language-Model Reasoning

**Document type.** Project Review – I: Introduction and Literature Review.

---

## CHAPTER 1 – INTRODUCTION

### 1.1 Background of the Study

Large language models (LLMs) are now the dominant computational substrate for multi-step reasoning tasks that span grade-school arithmetic, Olympiad-style mathematics, programming, and expert-level question answering. The interface through which these systems reason is chain-of-thought (CoT) prompting, in which the model is induced to emit an explicit sequence of intermediate steps before producing a final answer [1]. CoT prompting has consistently improved accuracy on multi-step problems [1], and the pattern has been extended by reasoning-specialised models trained with reinforcement learning on verifiable rewards, which further push the frontier of what mathematical and logical tasks are solvable by generation alone.

Two benchmarks define the evaluation substrate used throughout this study. GSM8K [2] contains grade-school word problems with gold-standard final-answer labels and is commonly used as the entry point for reasoning evaluation. MATH [3] contains twelve-and-a-half thousand problems across seven subjects, including algebra, counting and probability, geometry, intermediate algebra, number theory, pre-algebra, and precalculus; it is more challenging and provides subject-level stratification. Both benchmarks admit automatic verification of a parsed final answer, which makes them suitable for the behavioural-outcome measurements this study requires.

With the growth of CoT and reasoning-specialised models has come a well-documented inefficiency. Reasoning models frequently continue generating tokens after the final answer has already stabilised. Three independent lines of recent work have formalised this behaviour. Liu and Wang segmented CoT traces into sentence-level chunks, extracted intermediate answers from each chunk, and showed across five open models and five reasoning benchmarks that stopping at the point of answer agreement preserves most of the final-answer accuracy while cutting generated tokens substantially [4]. Zhang et al. showed that correctness information is linearly decodable from intermediate hidden states of reasoning models, with usable in-distribution AUC values on mathematical tasks, and that probe-guided early exit achieves calibrated token savings [5]. Caldarella et al. introduced a *first-correct-prefix* evaluation that separates harmless surplus text ("verbose overthinking") from later tokens that derail an already-correct trajectory ("harmful overthinking") [6]. The post-answer suffix in reasoning traces is therefore a measurable regularity rather than an anecdotal observation.

What these studies do not resolve is *what the suffix is at a representational level*. One widely entertained possibility is that the model is performing a form of internal self-verification — a redundant check on an answer it has already computed. The hypothesis is attractive because it couples a scientific claim about internal state to a deployment-relevant control: if the "checking" phase can be identified, it can be safely truncated.

The emergence of Natural Language Autoencoders (NLAs) in 2026 appears to make this hypothesis directly testable. Fraser-Taliente et al. jointly train an activation verbaliser (AV) that samples free-form text from a model's residual-stream activation and an activation reconstructor (AR) that maps text back to a vector; the training objective minimises expected squared reconstruction loss between the original activation and the AR output, with a KL regulariser, and is warm-started from context summaries before joint policy optimisation [7]. The method's reconstruction fidelity is reported as fraction of variance explained (FVE), and released AV/AR pairs exist for Qwen2.5-7B-Instruct at layer 20, Gemma-3-12B/27B-IT at layers 32 and 41, and Llama-3.3-70B-Instruct at layer 53 [7]. The exact extraction site, vector normalisation, injection scaling, tokenizer, and chat template must match the released checkpoint, so NLAs are coupled to specific model-and-layer pairs rather than being model-agnostic probes.

Three constraints from the same body of literature prevent a naive application of NLAs to the post-answer suffix.

- Reconstruction fidelity is not claim-level faithfulness. Dingeto introduced the RECAP framework and showed, in controlled settings, that AV outputs can preserve a substantial fraction of activation variance while individual claims within an AV sentence are not themselves reconstruction-dependent [8]. RECAP's constructive proposal is to co-train the target model with linear decodability heads that preserve pre-specified facts, so that designated variables remain auditable against a fresh probe; the method requires target-model training and does not retroactively validate free-form AV prose on released NLA checkpoints.

- Textual reports produced by LLMs are not literal accounts of the underlying computation. Turpin et al. showed that CoT text can be systematically influenced by features of the prompt that the model never acknowledges [9]. Lanham et al. developed a measurement framework in which CoT tokens are deleted, resampled, or paraphrased, and reported that many CoT tokens can be removed without changing the final answer in a range of settings [10]. A surface statement such as "let me check this" in a reasoning tail therefore cannot be taken as a reliable label for an internal verification state.

- Intrinsic self-correction in reasoning LLMs is not reliably useful. Huang et al. showed in controlled experiments that repeated self-correction without external feedback often fails to improve or actively degrades reasoning accuracy, and that many previously reported gains collapse when controls are tightened [11]. A "verifying-looking" suffix cannot therefore be presumed beneficial.

Finally, any causal claim about an internal state operating in the suffix must respect established methodological standards. Goldowsky-Dill et al. documented how choices of metric, corruption, restoration site, and normalisation materially change the interpretation of activation-patching experiments, and recommended matched-random-direction controls, dose–response curves, and multiple-site tests before an intervention is read as mechanistic evidence [12]. Semantic entropy, which clusters multiple sampled answers by meaning and computes entropy over clusters, is the current reference uncertainty signal for free-form generation and outperforms lexical entropy and several confidence baselines on confabulation detection [13]; it is a strong comparator but measures outcome uncertainty rather than suffix type.

The scientific situation at the time of this review is therefore unusual: the necessary components exist in isolation — released NLA checkpoints, strong cheap signals for early stopping, a formal overthinking taxonomy, a methodology for causal interventions on residual states — but they have not yet been combined into a study that could decide whether an NLA-derived readout actually adds safe-stopping value over cheap baselines, nor whether a putative *redundant self-verification tail* has any causal reality distinct from answer stability itself. This absence motivates the study.

### 1.2 Problem Statement

Reasoning LLMs produce suffixes after their final answer has become behaviourally stable, and the empirical literature establishes this as a measurable regularity [4]–[6]. The literature also establishes the main cheap signals that can be used to decide when to stop: answer agreement across sentence-level chunks [4], hidden-state correctness probes [5], and meaning-level semantic entropy [13]. Natural-language readouts of residual activations are now technically possible using released NLA checkpoints [7]. These elements, however, have not been combined into an evaluation that can decide whether such readouts carry safe-stopping information beyond what the cheap signals already supply, and whether any detected tail corresponds to a causally meaningful internal state rather than a textual regularity.

Four concrete limitations persist across the current literature.

- **No activation-explanation-independent operational definition of the post-answer tail.** The first-correct-prefix construction of Caldarella et al. [6] depends on an answer parser and can mark a lucky early surface answer as a "correct" prefix; the answer-agreement criterion of Liu and Wang [4] is a behavioural proxy and can stabilise before a missed constraint is recovered. Neither pairs its stopping decision with K-continuation controls that would test the stability of the parsed answer across resampled continuations from the same prefix.

- **No head-to-head comparison of NLA-derived signals against cheap baselines on safe stopping.** Fraser-Taliente et al. [7] report FVE and audit case studies on activations in general; they do not evaluate NLA against semantic entropy [13], hidden-state correctness probes [5], or answer-convergence stopping [4] on the specific task of safely terminating a reasoning trace at matched compute budgets.

- **No claim-level faithfulness audit of AV sentences on reasoning tails.** The RECAP challenge of Dingeto [8] shows that this audit is methodologically non-trivial; the deletion, resample, and paraphrase template developed by Lanham et al. for CoT text [10] has not been applied to AV sentences in a reasoning-tail setting.

- **No causal validation of the putative verification tail with matched controls.** Early-exit studies report accuracy–token trade-offs [4], [5] but do not include tail-replacement or matched-control activation-patching experiments in the methodology of Goldowsky-Dill et al. [12].

The practical consequence is that early-stopping policies are currently deployed either on heuristic length cutoffs or on behavioural proxies whose relationship to internal state is unknown. The scientific consequence is that strong interpretability claims — that a model is "internally verifying" its answer — are being implied on the basis of textual appearance alone, without faithfulness or causal evidence.

### 1.3 Aim and Objectives

**Aim.** To design and empirically evaluate a pre-registered, activation-explanation-independent, causally validated protocol for the post-answer suffix in reasoning LLMs — hereafter the *Answer-Stable Tail* (AST) — and to compare natural-language readouts of residual-stream activations against established cheap signals on the task of safe early stopping.

**Primary research question.** On GSM8K and MATH, does an NLA-derived readout of residual-stream activations provide incremental, causally supported value for safe stopping over semantic entropy, hidden-state correctness probes, and answer-convergence stopping?

**Supporting research questions.** (RQ-S1) Can the Answer-Stable Tail be identified by a criterion that is independent of NLA output and that passes K-continuation and external-verifier controls? (RQ-S2) Under claim-level audit, what fraction of AV sentences on AST windows are reconstruction-dependent in the sense of Dingeto's RECAP framework [8]? (RQ-S3) Under matched-control activation interventions, do AST-window states carry a direction whose ablation selectively changes verification-like behaviour without changing final answers?

**Objectives.** The study is organised around six objectives. Each objective addresses one of the four problem-statement limitations (L1–L4) in §1.2 or the reporting requirement; each is tied to a method and a planned evaluation.

- **O1.** Define the Answer-Stable Tail operationally without reference to NLA output, using (i) chunk-level answer equivalence under truncation at a candidate boundary, (ii) K independently sampled continuations verified against the external math verifier, and (iii) matched-position and matched-length controls on GSM8K and MATH. *(Addresses L1.)*

- **O2.** Implement a reproducible pipeline that loads the released Qwen2.5-7B-Instruct layer-20 NLA checkpoint [7] and emits AV text together with AR vectors at AST candidate windows, with versioned prompts, decoding configuration, verifier integration, and activation-cache hashes. *(Addresses L2 at the implementation level.)*

- **O3.** Measure the incremental value of the NLA-derived signal over semantic entropy [13], a hidden-state correctness probe [5], and answer-convergence stopping [4] on paired safe-stopping accuracy and tokens saved at matched compute budgets, with paired bootstrap confidence intervals computed at the problem level. *(Addresses L2 at the evaluation level.)*

- **O4.** Audit a stratified sample of AV sentences generated on AST windows using the deletion, resample, and paraphrase template of Lanham et al. [10], scored by AR reconstruction change, to produce claim-level faithfulness statistics for AV outputs on reasoning tails. *(Addresses L3.)*

- **O5.** Run a causal-validation ladder — truncation, tail replacement with neutral filler, matched-random-direction and matched-position controls, and dose–response activation patches — in the methodology of Goldowsky-Dill et al. [12], to test whether removing or altering the AST changes verified behaviour in a direction-specific way. *(Addresses L4.)*

- **O6.** Release the labelled AST corpus and the evaluation harness together with a preregistered analysis plan reporting Krippendorff α on all semantic labels, so that subsequent work on target-model co-trained decodability (RECAP) [8], additional model families, or non-mathematical tasks can be run against a stable baseline. *(Reporting.)*

**Positioning of the contribution.** The study does not propose a new interpretability method and does not claim that natural-language readouts are novel as a technique; AV/AR training and released checkpoints are due to Fraser-Taliente et al. [7]. The contribution is a methodological and evaluation contribution: a composite protocol that integrates an independent tail definition, cheap-signal baselines, blinded semantic annotation, claim-level faithfulness audits, and matched-control causal interventions, together with the first head-to-head safe-stopping comparison of NLA against the cheap signals established in [4], [5] and [13]. The protocol is designed so that a null result — in which the cheap signals match or exceed the NLA-derived signal at matched budget — is a scientifically useful outcome that recasts natural-language activation readouts as a semantic interface over equivalent detectability rather than as a new detector.

---

## CHAPTER 2 – LITERATURE REVIEW

### 2.1 Review of Existing Work

The literature relevant to the Answer-Stable Tail sits at the intersection of two bodies of research that have, until recently, developed separately: *reasoning efficiency and faithfulness*, and *internal interpretability of LLM activations*. This review organises existing work into three thematic strands and then performs a comparative analysis of the ten studies that most directly bear on the inference chain the project investigates. The ten studies and their roles are summarised in Table 2.1 (§2.1.4).

#### 2.1.1 Chain-of-Thought Reasoning and the Limits of Textual Faithfulness

Chain-of-thought prompting established the dominant interface through which LLMs perform multi-step reasoning: the model is induced to emit an explicit sequence of intermediate steps and only then produces a final answer [1]. CoT raises accuracy on multi-step problems, and its text has become both the computational substrate and the candidate interpretability target for subsequent work. Reasoning-specialised models trained with reinforcement learning on verifier-based rewards further rely on long CoT as a working-memory surface.

The question of whether CoT text is a *literal* account of the computation has been examined directly. Turpin et al. perturbed prompt features that would, if detected, change a model's answer, and measured whether the CoT text acknowledged the influence [9]. The reported finding is that models can systematically use hidden prompt features in their answer behaviour without mentioning them in CoT text, which rules out treating CoT as a faithful log of the internal process. Lanham et al. proposed a complementary measurement framework in which CoT tokens are deleted, resampled, or paraphrased, and the sensitivity of the final answer is used as a proxy for faithfulness [10]. Across the models and tasks they considered, a substantial fraction of CoT content can be removed without changing the final answer, while some settings do produce a measurable CoT–answer coupling. Together, [9] and [10] establish that the surface text of a reasoning tail cannot be used as a ground-truth label for whether the model is "verifying" its answer or performing any other specific operation.

A related but distinct issue is whether the behaviour expressed in CoT suffixes is itself productive. Huang et al. ran controlled self-correction experiments on reasoning tasks and reported that *intrinsic* repeated correction — in which the model attempts to improve its own answer without external feedback — often fails to improve or degrades accuracy, and that several previously reported gains collapsed under tighter controls [11]. The implication for the Answer-Stable Tail is a prior of scepticism: a "checking-looking" suffix is not presumptively beneficial, is not presumptively a verification process, and should not be converted into a positive label by appearance alone.

#### 2.1.2 Reasoning Efficiency, Answer Convergence, and Overthinking

A second strand of recent work directly measures the behavioural post-answer regularity and attempts to exploit it for compute savings. Liu and Wang segmented CoT traces into sentence-level chunks and interrogated the model for an intermediate answer at each chunk boundary [4]. Across five open models and five reasoning benchmarks, stopping when consecutive intermediate answers agree yields substantial token reductions with limited average accuracy loss. The same study also trained a learned stopping policy on hidden-state features at chunk boundaries, which further improves the token–accuracy frontier. The method's dependence is behavioural: agreement is a necessary but not sufficient condition for safe stopping, because a stable surface answer can precede the recovery of a missed constraint.

Caldarella et al. developed a complementary operational tail definition on reasoning traces [6]. For each trace, they identified the *first correct prefix* — the earliest point at which the extracted answer matches the verifier label — and classified the remaining tokens as either verbose overthinking (harmless restatement) or harmful overthinking (post-correct drift). The paper reports non-trivial fractions of traces in which an early correct prefix exists and in which later tokens cause the answer to flip, and stopping at the first correct prefix is shown to improve accuracy on selected tasks. Two caveats apply: a first "correct" surface answer may be chance-coincident with the verifier, and the prefix classification does not include activation-level or causal evidence that would separate a semantic verification process from a textual regularity. This paper is currently a preprint.

The broader area of efficient reasoning has moved rapidly: Yang et al. propose a dynamic early-exit mechanism tied to reasoning-specific models [16], Wu et al. examine when longer CoT begins to hurt accuracy rather than help it [17], and the survey by Sui et al. consolidates the taxonomy of efficient-reasoning methods [19]. These works reinforce that the post-answer suffix is empirically real and that stopping-related compute savings are achievable, but they do not themselves settle the representational question.

Representational evidence comes from Zhang et al., who trained probes on reasoning-model activations at intermediate-answer positions [5]. In-distribution AUC for correctness decodability exceeds 0.7 on mathematical tasks (as reported), with weaker cross-domain transfer; probe-guided early exit achieves calibrated token savings against standard confidence baselines. Two methodological caveats apply. First, decodability is an evidence of *information availability* in the activation, not of *causal use* by the model, and converting a probe into a mechanistic claim requires activation-patching evidence with matched controls [12]. Second, the probe is a scalar readout and does not describe what else about the state is predictive of behaviour.

Finally, meaning-level uncertainty must be used as a baseline against any proposed tail-stopping detector. Farquhar et al. introduced semantic entropy, in which K generations are sampled and clustered by semantic equivalence, with entropy computed over meaning clusters rather than over tokens [13]. Across the models and tasks they considered, semantic entropy outperforms lexical entropy and several confidence baselines on confabulation detection. The signal measures uncertainty over outcomes rather than tail type, so it is not itself a verification detector; however, it is the current reference comparator for any new signal that proposes to decide when a reasoning trace can be stopped.

#### 2.1.3 Internal Interpretability and Natural-Language Activation Readouts

A third strand concerns the methods available to read, isolate, and intervene on information in the residual stream. Causal tracing / activation patching is the methodological reference for intervention-based claims about internal state; Meng et al. introduced the pattern for factual-association localisation [14], and Goldowsky-Dill et al. documented in detail how choices of metric, corruption, restoration site, and normalisation materially change the interpretation of patching experiments [12]. The practical recommendation is explicit: any mechanistic claim should include matched-random-direction controls, matched-position controls, and dose–response curves across multiple sites. Sparse autoencoders provide a complementary quantitative method for isolating interpretable directions in the residual stream and have been scaled on production-grade targets, with explicit discussion of monosemantic feature structure in [18].

Into this landscape, Fraser-Taliente et al. introduced Natural Language Autoencoders [7]. The method trains a verbaliser AV that samples text from a residual-stream state and a reconstructor AR that maps text back to a vector; joint training minimises expected squared reconstruction loss with a KL regulariser, with AV warm-started from context-summary pairs and then updated with policy optimisation. AV is implemented by injecting the activation into a special-token embedding, and AR is a truncated backbone followed by a vector head. The authors report that reconstruction fidelity (FVE) is substantial on their constructed-truth tasks and that case studies produce useful audit leads, corroborated by training-data inspection, sparse-autoencoder evidence, attribution, or steering. The same authors are explicit about limits: AV outputs can confabulate; audits are corroborated for *selected* cases rather than for all AV sentences; AV generation is expensive (approximately 500 tokens per activation in their reported setting); the AV/AR pairs are coupled to specific model-and-layer checkpoints; and private-code failure modes exist in which AV and AR agree on an internal protocol that does not correspond to the external language. The method is therefore properly understood as an open-ended hypothesis generator rather than as a semantic ground-truth instrument. This source is a project report rather than a peer-reviewed publication.

The faithfulness of free-form AV sentences has been directly challenged. Dingeto introduced the RECAP framework, which tests whether individual claims in an AV sentence are themselves reconstruction-dependent, and showed in controlled runs that reconstruction fidelity can remain high while most specific claims do not individually constrain the AR output [8]. The constructive proposal is to co-train the target model with linear decodability heads that preserve pre-specified facts, so that designated variables remain auditable against a fresh probe. RECAP therefore strengthens the evidence base for a pre-specified readout but does not retroactively validate arbitrary free-form AV prose on released NLA checkpoints and does not supply a mechanism for the un-RECAPed target models that are currently available. This source is a preprint.

Self-correction frameworks that depend on external feedback occupy an adjacent area. Self-Refine [15] is a representative example of iterative refinement with external critiques, and shows that gains over single-pass reasoning are available when a reliable external signal is present. The relevance to the Answer-Stable Tail is that the post-answer suffix in standard reasoning models does *not* in general have such an external signal, which further limits the amount of productive computation that the suffix is likely to represent.

#### 2.1.4 Comparative Analysis

Ten studies are selected for direct comparison. The selection criterion is that the ten together cover every causal step of the inference chain the study investigates: natural-language readout from activations (what we read), representational evidence (what else is encoded), answer-stability signals (behavioural signal), self-correction and overthinking (what the tail might be), textual-report faithfulness (how far text can be trusted), causal validation methodology (how to prove the signal is used), and the strongest cheap uncertainty baseline (what we must beat). The ten studies are compared in Table 2.1 along ten evidence-supported dimensions.

**Table 2.1 — Comparison of the ten studies most directly relevant to the Answer-Stable Tail.** "PR" denotes project report; "preprint" denotes an unpublished archival manuscript at the time of this review. Reported numeric values appear only where they are stated by the cited authors.

| # | Author(s) and year [cite] | Research problem | Method / approach | Dataset | Experimental setting | Baselines | Evaluation metrics | Main findings | Reported limitations | Relevance to the proposed research |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Fraser-Taliente et al., 2026 [7] (PR, Transformer Circuits) | Produce language-level, unsupervised explanations of LLM residual activations. | Joint training of an activation verbaliser (AV) and activation reconstructor (AR) with squared reconstruction loss and a KL regulariser; AV warm-started from context-summary pairs, then updated by policy optimisation. | Large sampled residual-stream activations from the target models in use. | Released AV/AR pairs for Qwen2.5-7B-Instruct L20, Gemma-3-12B/27B-IT L32/L41, Llama-3.3-70B-Instruct L53. | Context-only summariser; random text. | FVE (fraction of variance explained); constructed-truth prediction tasks; qualitative audits. | AV outputs preserve substantial activation variance; selected audits produce usable leads corroborated by sparse-autoencoder, attribution, or steering evidence. | Confabulation; layer sensitivity; expressive decoder prior; private-code failure modes; approximately 500 generated tokens per activation; strict model/layer coupling. | Supplies the readout mechanism used in the study; its stated limits scope the role of AV output to hypothesis generation rather than detection. |
| 2 | Dingeto, 2026 [8] (preprint) | Does reconstruction fidelity imply claim-level faithfulness of AV sentences? Can faithfulness be supervised? | RECAP: co-train the target model with linear decodability heads that preserve specified facts; test claim-deletion and claim-substitution effects on reconstruction. | Controlled synthetic text; evaluator-swapping tests. | AV/AR with and without RECAP; probes trained post hoc for comparison. | Standard AV/AR without RECAP. | Reconstruction error; dependence of reconstruction on individual claims; decodability of specified variables. | High reconstruction with low claim-dependence is possible under standard recipes; RECAP preserves pre-specified decodability; co-adapted codes form in controlled runs. | Requires retraining the target model, so does not retrofit on released NLA checkpoints; only validates pre-specified content. | Forces the study to treat free-form AV output as unaudited hypothesis and to run claim-level audits; sets the standard any faithfulness claim must meet. |
| 3 | Liu and Wang, 2025 [4] (EMNLP) | Can CoT reasoning be stopped safely once intermediate answers converge? | Split CoT into sentence-level chunks; extract intermediate answers; stop on answer agreement; also train a learned stopping policy on hidden-state features. | Five open reasoning benchmarks. | Five open models at varying scales. | Fixed-length CoT; token-entropy stopping; self-consistency. | Final-answer accuracy; tokens generated; accuracy–token Pareto curves. | Substantial token reductions with limited average accuracy loss across the tested configurations. | Agreement is a behavioural proxy and can be stable before a missed constraint; no semantic-state label is produced. | Provides the chunk-boundary scaffolding for Answer-Stable Tail candidates and the primary cheap behavioural baseline the study must match or beat. |
| 4 | Zhang et al., 2025 [5] (COLM) | Can hidden states of reasoning models indicate whether the model is right? | Train linear/MLP probes on intermediate-answer-position activations; use probe scores to drive calibrated early exit. | Reasoning models on math and mixed tasks; in-distribution and transfer settings. | Multiple reasoning models; held-out task sets. | Confidence / log-probability baselines; semantic-entropy where comparable. | Probe AUROC; stopping accuracy–token trade-off; calibration. | Reported in-distribution AUC exceeds 0.7 on mathematical tasks; probe-guided early exit reduces tokens with calibrated scores; cross-domain transfer is weaker. | Decodability is availability of information in the state, not causal use by the model; cross-domain transfer degrades. | Primary representational baseline; the study must demonstrate that natural-language readouts add information beyond this cheap signal. |
| 5 | Caldarella et al., 2026 [6] (preprint) | How much of the suffix after a model's first correct answer is harmless versus harmful? | Define a first-correct-prefix using an answer parser and verifier; measure whether later tokens flip the final answer; separate verbose from harmful overthinking. | Reasoning models on math-style tasks. | First-correct-prefix stopping vs. full-length generation. | Fixed-length CoT; naive length cutoffs. | Accuracy at first-correct prefix; accuracy at full length; token delta. | Non-trivial fraction of traces have an early correct prefix; stopping at that prefix improves accuracy in selected tasks. | First "correct" surface answer can be chance-coincident; no activation-level or causal evidence; depends on the parser. | Supplies the closest existing operationalisation of the tail; the study extends this with K-continuation and verifier controls. |
| 6 | Farquhar et al., 2024 [13] (Nature) | Can model uncertainty be measured in semantic rather than lexical space? | Sample multiple generations; cluster by semantic equivalence; compute entropy over meaning clusters. | Multiple LLMs; question-answering and generation benchmarks. | Confabulation-detection evaluation across generation tasks. | Lexical entropy; token-probability baselines; several confidence measures. | AUROC for confabulation detection. | Outperforms lexical entropy and several confidence baselines across the tested models and tasks. | Measures outcome uncertainty, not whether the suffix is a verification or restatement step; requires K samples per input. | Primary meaning-level uncertainty baseline; any claimed verification-detector must out-perform this signal at matched budget. |
| 7 | Huang et al., 2024 [11] (ICLR) | Can LLMs intrinsically self-correct reasoning without external feedback? | Controlled self-correction loops; separation of intrinsic from externally-grounded correction. | Multiple reasoning tasks; multiple LLMs. | Zero-shot CoT and self-correction loops. | Zero-shot CoT; self-consistency. | Final-answer accuracy with and without correction loops. | Intrinsic repeated self-correction often degrades or fails to improve accuracy; some previously reported gains collapse under controls. | Does not rule out all post-answer reasoning, especially with external verifiers or specific training. | Establishes a strong prior of scepticism against treating a "verification-looking" suffix as positive label. |
| 8 | Turpin et al., 2023 [9] (preprint) | Is CoT text a literal account of the model's reasoning? | Perturbation experiments on prompt features that would change the answer, with and without CoT acknowledgement. | Multiple LLMs; standard CoT prompts. | CoT with hidden input perturbations. | Standard CoT; counterfactual prompts. | Rate of unacknowledged influence; answer parity. | CoT text can be systematically influenced by hidden features that the model does not mention. | Focuses on specific prompt perturbations; dataset and model dependence. | Rules out using tail surface text as a label for internal state; motivates blinded annotation. |
| 9 | Lanham et al., 2023 [10] (preprint) | How can CoT faithfulness be measured? | Delete, resample, paraphrase, and intervene on CoT tokens; measure the final-answer sensitivity. | Multiple LLMs; multiple reasoning tasks. | Full-CoT, truncated-CoT, and paraphrased-CoT configurations. | Full-CoT baseline. | Answer sensitivity to perturbation. | A substantial fraction of CoT tokens can be deleted or paraphrased without changing the final answer; some settings show measurable CoT–answer coupling. | Task- and model-specific results; coarse faithfulness measure. | Provides the methodological template for claim-level audits of AV sentences on reasoning tails. |
| 10 | Goldowsky-Dill et al., 2023 [12] (preprint) | What design choices make activation-patching conclusions robust? | Controlled study of metric, corruption, restoration, and normalisation choices in activation patching. | Several open models; standard interpretability setups. | Comparison of patching recipes. | Alternative patching recipes. | Interpretation stability across choices. | Interpretation can change materially with these choices; recommends matched-random-direction controls, dose–response curves, and multiple sites. | Standards are not consensus; experiment cost is non-trivial. | Governs every causal step in the proposed study (truncation, tail replacement, direction ablations). |

**Reading of the comparison.** No single row satisfies all three of {activation-reading, free-form language output, causal validation}. The role of the proposed study is therefore to bridge these: entries 1 and 2 supply the readout and its faithfulness constraints; entries 3, 4, and 6 are the cheap signals the readout must be compared against at matched compute; entry 5 supplies the operational tail definition the study extends with K-continuation controls; entry 7 supplies the prior of scepticism; entries 8 and 9 bound the use of surface text as a label; and entry 10 governs every causal step. No entry in the ten, nor any additional reasoning-efficiency work surveyed in [16], [17], [19] or the SAE line in [18], runs the composite evaluation this study proposes.

Three further observations arise from Table 2.1 and §§2.1.1–2.1.3.

- Answer stability and causal redundancy are distinct properties. Both [4] and [6] show that a stable surface answer can be identified behaviourally, but neither pairs the stability signal with an intervention that would show the suffix is causally dispensable or that it corresponds to a specific internal state.

- Decodability and causal use are distinct properties. Zhang et al. [5] show that correctness is linearly decodable from reasoning-model activations, but, as [12] makes explicit, converting a decodable direction into a mechanistic claim requires matched-control activation interventions.

- Reconstruction fidelity and claim-level faithfulness are distinct properties. Fraser-Taliente et al. [7] report substantial FVE; Dingeto [8] shows that, in controlled runs, the same AV sentence can be high-FVE while having individual claims that are not themselves reconstruction-dependent.

### 2.2 Research Gap Identification

#### 2.2.1 Established findings

The following findings are established across the ten comparison studies and the broader literature reviewed above:

- Reasoning models produce suffixes after their final answer has become behaviourally stable, and these suffixes admit both verbose and harmful variants [4], [6].

- Correctness information is linearly decodable from reasoning-model intermediate activations with usable in-distribution AUROC, and probe-guided early exit saves tokens with calibrated scores [5].

- Semantic entropy outperforms lexical entropy and several confidence baselines on confabulation detection and is the current reference meaning-level uncertainty signal [13].

- Intrinsic repeated self-correction without external feedback often fails to improve or degrades reasoning accuracy under controlled conditions [11].

- Chain-of-thought text is not a literal account of the model's internal computation: perturbation experiments [9] and deletion/paraphrase experiments [10] both show substantial decoupling between CoT text and the final answer.

- Natural Language Autoencoders can produce free-form, activation-conditioned descriptions of residual-stream states with substantial reconstruction fidelity, while their authors document confabulation, layer sensitivity, decoder prior, and private-code failure modes [7].

- Reconstruction fidelity does not imply claim-level faithfulness of AV sentences; supervised auditability requires target-model co-training through the RECAP framework [8].

- Any causal claim on residual-stream state requires matched-random-direction controls, dose–response curves, and multiple-site tests [12].

#### 2.2.2 Documented limitations persisting across these findings

- *Operational limitation.* No study in the review defines the Answer-Stable Tail without reference to activation-explanation output, and no study pairs an operational tail definition with K-continuation and external-verifier controls. Caldarella et al. [6] rely on an answer parser and are vulnerable to lucky surface answers; Liu and Wang [4] rely on answer agreement, which is necessary but not sufficient for safe stopping.

- *Comparative limitation.* No study in the review evaluates a natural-language activation readout against the three principal cheap signals — semantic entropy [13], hidden-state correctness probes [5], and answer-convergence stopping [4] — on safe stopping at matched compute budgets. Fraser-Taliente et al. report reconstruction and audit utility [7] but do not evaluate on this task.

- *Faithfulness-audit limitation.* The deletion/resample/paraphrase template of Lanham et al. [10] has not been applied to AV sentences in a reasoning-tail setting. Dingeto's RECAP challenge [8] predicts that an audit of this kind would recover meaningful claim-level dependence only sparsely; this prediction is currently untested on reasoning tails.

- *Causal limitation.* Early-exit studies report accuracy–token trade-offs [4], [5] but do not include tail-replacement or matched-control activation-patching experiments in the methodology of Goldowsky-Dill et al. [12]. The putative verification tail therefore has no causal evidence as a distinct internal event.

- *Annotation limitation.* No study in the review supplies a blinded, multi-rater semantic annotation of suffix type (verification, restatement, derivation, correction) on reasoning traces with reported inter-rater reliability.

#### 2.2.3 Unresolved research issue

Taken together, the limitations in §2.2.2 state an unresolved issue that is specific and testable:

> It is not currently established whether a natural-language readout of residual-stream activations carries safe-stopping information for reasoning LLMs that is not already available from cheap uncertainty, probe, or agreement signals, nor whether any such information corresponds to a causally meaningful internal state as distinct from answer stability itself.

This issue is not merely one of absence: three independent pieces of positive evidence make it a documented gap rather than an unexplored possibility. First, the authors of the readout method themselves document confabulation, layer sensitivity, and decoder prior as unresolved risks of their technique [7]. Second, the RECAP framework [8] *predicts* and demonstrates in controlled runs that free-form AV prose will not in general be claim-level faithful. Third, the two independent lines of CoT-faithfulness evidence [9], [10] and the controlled negative result on intrinsic self-correction [11] rule out the two naive shortcuts — using surface text of a tail as a verification label, or treating a verifying-looking suffix as presumptively useful.

#### 2.2.4 Classification of the gap

The gap is composite rather than monolithic.

- *Methodological gap.* No published protocol integrates an activation-explanation-independent tail definition, blinded multi-rater semantic labels, K-continuation and external-verifier controls, and matched-control activation interventions in the methodology of [12].

- *Evaluation gap.* No head-to-head benchmark exists that compares a natural-language activation readout against the cheap signals of [4], [5], and [13] on safe stopping at matched compute budgets.

- *Causal gap.* No matched-control activation-intervention evidence on verification-like directions exists in the reasoning-tail setting.

- *Faithfulness-audit gap.* The claim-level audit template of [10] has not been applied to AV sentences on reasoning tails, and the RECAP challenge of [8] has not been directly tested in this setting.

The gap is not an implementation gap: running existing NLA checkpoints on GSM8K would not, on its own, address any of the four components above. It is not an application gap: a bare application of NLA to reasoning tasks still omits the comparison, the audit, the causal evidence, and the annotation. It is not a theoretical gap in the sense of a missing formal framework. It is primarily methodological and evaluation, with secondary causal and faithfulness components.

#### 2.2.5 Proposed research direction

The study proposes to address this gap by (i) defining the Answer-Stable Tail operationally and independently of AV output, using chunk-level answer equivalence under truncation and K independently sampled continuations verified against the external math verifier, with matched-position and matched-length controls on GSM8K and MATH; (ii) running released NLA AV/AR on the released Qwen2.5-7B-Instruct layer-20 checkpoint at AST candidate windows alongside semantic entropy [13], a hidden-state correctness probe [5], and answer-convergence stopping [4], on verified GSM8K and MATH; (iii) auditing a stratified sample of AV sentences on AST windows using the deletion/resample/paraphrase template of [10], scored by AR reconstruction change; (iv) running a causal-validation ladder — truncation, tail replacement, matched-random-direction, matched-position, and dose–response controls — in the methodology of [12]; and (v) reporting paired bootstrap confidence intervals by problem, Krippendorff α on all semantic labels, matched-control results on every intervention, and both negative and positive outcomes with equal weight.

The study's expected contribution is methodological and evaluation. It supplies an integrated protocol combining tail definition, cheap-signal comparison, claim-level faithfulness audit, blinded annotation, and causal validation; it supplies the first head-to-head safe-stopping comparison of a natural-language activation readout against the established cheap signals on verified reasoning tasks; and it supplies a labelled corpus together with an evaluation harness suitable for subsequent extensions, including target-model co-trained decodability through RECAP [8], additional model families, and non-mathematical task domains.

#### 2.2.6 Scope and boundaries of the contribution

The contribution is not a new interpretability method. Natural Language Autoencoders are due to Fraser-Taliente et al. [7]; probing hidden states for correctness is due to Zhang et al. [5] and prior probing literature; semantic entropy is due to Farquhar et al. [13]; answer-convergence stopping is due to Liu and Wang [4]; the first-correct-prefix framework is due to Caldarella et al. [6]; the CoT-faithfulness audit template is due to Lanham et al. [10]; and the activation-patching methodology is due to Goldowsky-Dill et al. [12] with foundations in Meng et al. [14]. The study integrates these elements and evaluates them against each other on a task that the literature has not previously resolved. The primary contribution is therefore a methodological framework and a comparative evaluation; the study does not claim a novel representation-learning or interpretability algorithm, and it does not claim that the Answer-Stable Tail construct has been established as a distinct internal state until the proposed causal and faithfulness evidence has been collected.

---

## REFERENCES

Numbered in order of first appearance. Entries marked *preprint* are archival manuscripts without peer-reviewed publication status at the time of this review; the entry marked *project report* is an institutional technical report rather than a peer-reviewed publication.

[1] J. Wei, X. Wang, D. Schuurmans, M. Bosma, B. Ichter, F. Xia, E. Chi, Q. Le, and D. Zhou, "Chain-of-thought prompting elicits reasoning in large language models," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2022.

[2] K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, C. Hesse, and J. Schulman, "Training verifiers to solve math word problems," 2021, preprint, arXiv:2110.14168.

[3] D. Hendrycks, C. Burns, S. Kadavath, A. Arora, S. Basart, E. Tang, D. Song, and J. Steinhardt, "Measuring mathematical problem solving with the MATH dataset," in *Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track*, 2021.

[4] Y. Liu and X. Wang, "Answer convergence as a signal for early stopping in reasoning," in *Proceedings of EMNLP*, 2025.

[5] A. Zhang, Y. Chen, J. Pan, C. Zhao, A. Panda, J. Li, and H. He, "Reasoning models know when they're right: probing hidden states for self-verification," in *Proceedings of the Conference on Language Modeling (COLM)*, 2025.

[6] S. Caldarella, D. Talon, R. Aljundi, E. Ricci, and M. Mancini, "Thinking past the answer: evaluating harmful overthinking in large reasoning models," 2026, preprint.

[7] K. Fraser-Taliente, S. Kantamneni, E. Ong, et al., "Natural language autoencoders produce unsupervised explanations of LLM activations," *Transformer Circuits*, project report, 2026.

[8] H. Dingeto, "Train the model, not the reader: decodability supervision for verifiable activation explanations," 2026, preprint.

[9] M. Turpin, J. Michael, E. Perez, and S. R. Bowman, "Language models don't always say what they think: unfaithful explanations in chain-of-thought prompting," 2023, preprint.

[10] T. Lanham, A. Chen, A. Radhakrishnan, B. Steiner, C. Denison, D. Hernandez, D. Li, E. Durmus, E. Hubinger, J. Kernion, et al., "Measuring faithfulness in chain-of-thought reasoning," 2023, preprint.

[11] J. Huang, X. Chen, S. Mishra, H. S. Zheng, A. W. Yu, X. Song, and D. Zhou, "Large language models cannot self-correct reasoning yet," in *Proceedings of the International Conference on Learning Representations (ICLR)*, 2024.

[12] N. Goldowsky-Dill, C. MacLeod, L. Sato, and A. Arora, "Towards best practices of activation patching in language models: metrics and methods," 2023, preprint.

[13] S. Farquhar, J. Kossen, L. Kuhn, and Y. Gal, "Detecting hallucinations in large language models using semantic entropy," *Nature*, vol. 630, pp. 625–630, 2024, doi: 10.1038/s41586-024-07421-0.

[14] K. Meng, D. Bau, A. Andonian, and Y. Belinkov, "Locating and editing factual associations in GPT," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2022.

[15] A. Madaan, N. Tandon, P. Gupta, S. Hallinan, L. Gao, S. Wiegreffe, U. Alon, N. Dziri, S. Prabhumoye, Y. Yang, S. Welleck, B. P. Majumder, S. Gupta, A. Yazdanbakhsh, and P. Clark, "Self-refine: iterative refinement with self-feedback," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2023.

[16] Z. Yang et al., "Dynamic early exit in reasoning models," in *Proceedings of the International Conference on Learning Representations (ICLR)*, 2026.

[17] Y. Wu et al., "When more is less: understanding chain-of-thought length in LLMs," in *Proceedings of the International Conference on Learning Representations (ICLR)*, 2026.

[18] A. Templeton et al., "Scaling monosemanticity: extracting interpretable features from Claude 3 Sonnet," *Transformer Circuits*, project report, 2024.

[19] R. Sui et al., "Stop overthinking: a survey on efficient reasoning for large language models," 2025, preprint.

---

### Note on citation and verification

Citation style follows IEEE numbered referencing throughout. All entries marked as *preprint* or *project report* are represented as such and are not treated as settled evidence in the body of the review. Entries [1], [3], [4], [5], [11], [13], [14], [15], [16], and [17] correspond to publications in peer-reviewed venues; entries [2], [6], [8], [9], [10], [12], and [19] are preprints; entries [7] and [18] are institutional project reports from Transformer Circuits. Where author lists in preprint entries are abbreviated with *et al.*, this reflects an incomplete internal record and the full list should be verified from the archival source before any public distribution of this document.
