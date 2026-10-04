# 09 · Examiner Questions and Defensible Answers

**Compiled:** 2026-10-03. The questions below are produced by acting as a
skeptical project-review examiner across the dimensions prescribed by the
project brief (literature, gap, novelty, methodology, validity, failure).
Answers are written to be defensible, not flattering. Where an honest
answer is "we do not know", it is stated as such.

---

## A. Literature

**A1. Why these ten papers and not ten others?**
Together they trace every causal step of the AST inference chain: NLA
readout (P1), its faithfulness challenge (P2), closest competing
operational signals (P3 answer convergence, P4 hidden-state probe, P5
first-correct prefix), the strongest cheap uncertainty baseline (P6
semantic entropy), the strongest controlled negative result on
self-correction (P7), the two CoT-faithfulness constraints (P8, P9),
and the causal-interpretability methodology (P10). The screening
record is in `03_literature_candidates.md` §2.

**A2. Why not SAEs / feature circuits (Templeton 2024; Marks 2024)?**
They are the dominant *mechanistic* interpretability method and would be
an alternative method paper. The project does not include an SAE arm
yet, so including them in the comparison table would be
non-informative. They are discussed in Chapter 2.1 §2.1.6 for
completeness.

**A3. Why not Yang 2026 ICLR or Min 2026 (early exit)?**
They overlap with Liu & Wang 2025 on the early-exit baseline. Liu &
Wang 2025 is retained because it is peer-reviewed and because its
chunk framework will be reused.

**A4. Which paper is closest to your work?**
Liu & Wang 2025 (EMNLP). It is the signal the project must beat on
safe stopping. The differentiation is specified in
`05_research_gap.md` §4.1: NLA readout + K-continuation + verifier
+ matched-control activation interventions + blinded semantic labels
+ Lanham-style AV-claim audits.

**A5. What is the strongest competing approach, if NLA fails?**
Liu & Wang 2025 at scale, augmented with semantic entropy
(Farquhar 2024). This is also the project's primary non-NLA baseline.

**A6. Why is Fraser-Taliente 2026 cited as a "project report" and not a
peer-reviewed paper?**
Because it is published at transformer-circuits.pub, an Anthropic project
report. The authors themselves document the limits that constrain the
present project. It is treated with the same care as a preprint.

---

## B. Research gap

**B1. What exactly is your research gap?**
A composite methodological + evaluation + causal + faithfulness gap
(see `05_research_gap.md` §2.5). No published work combines
(i) an NLA-independent operational AST definition with K-continuation
+ verifier controls, (ii) head-to-head NLA-vs-cheap-baseline
comparison on safe stopping at matched compute, (iii) a Lanham-template
faithfulness audit of AV sentences on tails, and (iv) matched-control
activation interventions in the methodology of Goldowsky-Dill 2023.

**B2. Where is the evidence for the gap?**
Three independent pieces of evidence:
Fraser-Taliente 2026 themselves name their method as a hypothesis
generator with explicit confabulation / layer-sensitivity risks; Dingeto
(2026 preprint) directly shows that reconstruction fidelity does not
imply claim-level faithfulness; and no entry in the 36-paper candidate
pool runs the comparison. See the limitation matrix in
`05_research_gap.md` §1.

**B3. Is this a research gap or an implementation gap?**
Research gap, with components in methodology, evaluation, causality, and
faithfulness. An implementation of NLA on GSM8K alone does not resolve
any of these; the gap requires *protocol* design and *comparative /
causal evidence*, not just code.

**B4. Has recent work already solved it?**
To the best of this review (based on the 2026-09-25 phase reports), no.
The three most likely candidates — P1 (NLA source), P2 (RECAP), P5
(Caldarella first-correct prefix) — are each explicitly analysed and
do not supply the composite. If a 2026-Q4 paper does, the project
re-scopes rather than fakes novelty.

---

## C. Novelty

**C1. What exactly is novel?**
A *combination + evaluation* novelty, confidence medium. The project
does not claim a novel method, a novel problem, or a novel benchmark
dataset in isolation. It claims the first head-to-head safe-stopping
comparison of NLA against semantic entropy, hidden-state probes, and
answer-convergence stopping, within a protocol that also includes
blinded semantic labels and matched-control causal interventions.

**C2. What happens if the closest paper already does the same thing?**
Then the project pre-registers a null result and reports it. The
publication strategy explicitly treats null results as scientifically
valuable (see `05_research_gap.md` §6, honest statement of risk).

**C3. What differentiates your contribution if NLA matches but does not
beat the baselines?**
The contribution reduces to *(a)* an evaluation protocol and benchmark,
*(b)* a semantic interface — AV sentences remain a usable human-
readable overlay on a probe/entropy detector — and *(c)* a
negative-result paper on NLA as a detector. The project is designed to
be publishable in that outcome.

**C4. Why not just add RECAP?**
Because RECAP (Dingeto 2026 preprint) requires target-model
co-training, which the released NLA checkpoints have not undergone.
Retraining a 7B target with RECAP is a separate scientific project.
We adopt RECAP's framing as a constraint on claims, not as a method
we deploy.

---

## D. Methodology

**D1. Why Qwen2.5-7B-Instruct L20 as the primary model?**
Because it is the only released NLA checkpoint that is affordable to
run at research scale (the Fraser-Taliente 2026 project report
describes two 8×H100 nodes × 1.5 days for the Gemma-3-27B RL run).
Qwen L20 is also noted by the authors as a historical layer choice,
so it must be used exactly, not approximated.

**D2. Why not train a new NLA?**
Because training is 8×H100-scale and the project's contribution is
not a new NLA. Using the released checkpoint also eliminates one
degree of freedom.

**D3. Why GSM8K and MATH-algebra / counting-and-probability only?**
These are the data already staged in `data/raw/`. The remaining five
MATH subjects, MATH-500 held-out, BBH, GPQA, and HumanEval are
planned for the generalisation phase after Phase 1 evidence is in.

**D4. Why these baselines (Liu & Wang 2025; Zhang 2025; Farquhar 2024)?**
They are the strongest, cheapest signals the project must beat for an
NLA-detector claim: answer-agreement stopping (behavioural), hidden-
state correctness probe (representational), semantic entropy
(uncertainty).

**D5. Why these metrics (verified accuracy, tokens-saved, incremental
AUC, Krippendorff α)?**
Verified accuracy is required by the verifier contract. Tokens-saved
is the deployment-relevant compute metric. Incremental AUC is the
only metric that answers the primary research question. Krippendorff
α is required because any semantic label we use must survive
inter-rater reliability. Paired bootstrap CIs by problem are required
because the unit of inference is the trace, not the token.

**D6. Why these controls (matched-position, matched-random-direction,
dose-response)?**
Because Goldowsky-Dill 2023 preprint shows that interpretation
conclusions change with these choices. Omitting them is the single
most common failure mode in the literature.

---

## E. Validity

**E1. How will you establish improvement?**
A pre-registered pairwise comparison: NLA-aided safe-stopping policy
vs. each baseline at matched compute budget, with the primary endpoint
being the paired change in verified accuracy minus a non-inferiority
margin, bootstrapped at the problem level.

**E2. How will you avoid overfitting?**
Problem-grouped train/dev/test splits; one declared tuning budget;
all seeds and configurations in a manifest; held-out MATH-500 /
cross-subject sets.

**E3. How will you ensure reproducibility?**
Versioned prompt + decoding + answer-parser configs; hash-stamped
activation caches; released evaluation harness; checksummed data;
all baselines reproduced from released code.

**E4. What are the assumptions?**
(a) The released Qwen2.5-7B-L20 NLA checkpoint remains valid; (b) the
verifier is correct on in-domain problems; (c) two trained annotators
+ an adjudicator can produce α ≥ pre-registered threshold on at least
some semantic categories; (d) a sampled K = 8 to 16 continuations
provides enough statistical power for AST candidate verification on
the problem budget we can afford.

**E5. What are the limitations?**
Model/layer coupling to the released checkpoint; AV generation cost
(~500 tokens per activation) limits scale; semantic labels may fail α
for some categories; and the project cannot audit the target model's
training data.

---

## F. Failure

**F1. What if the proposed method performs worse than baselines?**
The pre-registered analysis treats this as the primary scientific
outcome of interest. The paper then becomes an evaluation-protocol +
negative-result contribution, with a clearly stated reason the NLA
signal did not add value.

**F2. What scientific knowledge is still obtained from a negative result?**
A great deal: evidence that NLA is best used as a semantic interface
rather than a detector; a reusable benchmark for future NLA variants
(including RECAP-supervised ones); a lower bound on the value of
cheap signals; and claim-level faithfulness statistics for AV
sentences on reasoning tails.

**F3. What if semantic labels fail inter-rater reliability?**
Semantic claims are then restricted to label categories that pass
α; the project's safe-stopping analysis is unaffected because its
primary outcome is verified accuracy, not semantic label.

**F4. What if the causal-validation ladder produces no selective
effect?**
Then no verification mechanism is identifiable at the released layer,
and the project reports the null with full matched-control results.
This is still a publishable methodological finding.

---

## G. Review-1-specific defensive questions

**G1. The repository has no implementation. How can you defend a Review
1?**
Review 1 asks for an introduction, a literature survey, a comparison
table, and a research gap. The repository has a six-phase prior-
research artefact (`output/nla_answer_stable_tail/phase{1..6}`) and
staged GSM8K + MATH subsets. The current review builds the
department-mandated chapters on this foundation and preregisters the
experiments. Implementation is Review 2's scope.

**G2. The paper database contains preprints with unusual arXiv IDs
(e.g. 2607.*, 2606.*). How do you handle that?**
They are explicitly flagged as preprints throughout, and URLs/DOIs are
labelled "as recorded in the project paper database on 2026-09-25,
not independently re-fetched for Review 1" (see
`03_literature_candidates.md` §4 and `06_chapter_1_introduction.md`
references). Any such citation will be re-verified before Review 2 or
removed.

**G3. Why not just build the NLA pipeline now and present experimental
results at Review 1?**
Because the department rubric is explicit: Review 1 is background,
problem, aim, objectives, literature review, and research gap. A
rushed pipeline without preregistration would weaken the Review 2
story. The current submission commits to a defensible, falsifiable
research plan rather than to premature numbers.

**G4. If NLA simply cannot be made to work on your hardware, what
happens to the project?**
The project retains its evaluation and semantic-labelling backbone
and runs it on probe / entropy / agreement baselines alone. NLA then
becomes an acknowledged scope limit, with a documented pathway for a
future collaborator with access to the released checkpoint's compute
tier.
