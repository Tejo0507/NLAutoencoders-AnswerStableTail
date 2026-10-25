# Declarations

*Prepared 2026-10-25 (D103) from the project's records. Statements marked
**[record]** state what the repository records; the author confirms them.
Statements marked **[author]** can only be supplied by the author and are
left open. Nothing here is author-approved yet.*

**Author.** [record] Tejo Sridhar M V S, the author named in
`CITATION.cff` and `LICENSE`. [author] Affiliation, ORCID, a corresponding
address, and confirmation that the author list is complete.

**Author contributions.** [author] To be supplied, for example in CRediT
terms.

**Funding.** [author] To be supplied. The project's records name no funder;
that is not a statement that the work was unfunded.

**Competing interests.** [author] To be supplied.

**Ethics and consent.** [record] The study uses two public benchmarks,
GSM8K [27] and MATH [28], and openly released models. No human participants
were recruited and no personal data were collected. The planned
two-annotator protocol was not run, and its blinded sheets are withheld.
The one annotation pass that exists (the RQ2 amendment A1, D56–D57) was made
by an AI system, not by people. [author] Whether an institutional ethics
review applies.

**Use of AI assistance.** [record] The project's session records, from
2026-10-16 onwards, show that an AI assistant (Anthropic's Claude, used
through Claude Code):
- wrote and tested analysis code;
- ran the experiments the researcher approved;
- checked numbers and citations against their sources;
- drafted and revised the text of this manuscript.

The researcher's decisions, including every plan freeze and approval, are
recorded in `docs/DECISIONS.md`. [author] The extent of AI use before
2026-10-16, the wording the venue requires, and whether to cite the
assistant's skill packages.

**Data availability.** [record]
- The main run's artifacts are published as the GitHub release
  `data-runs-2026-10-19` (CC BY 4.0) of the code repository.
- The artifacts of the replication, its follow-ups, the rewording null
  and the stopping-policy pilot are prepared as a verified release
  candidate and are not yet published (§7).
- The blinded annotation sheets and their key are withheld until the
  annotation is complete.
- Problem text comes from GSM8K and MATH (MIT). Traces, activations and
  verbalisations come from Qwen2.5-7B-Instruct and the released layer-20
  NLA (Apache 2.0). No model weights are redistributed.

**Code availability.** [record] All code, configurations, frozen plans
and the decision record are at
https://github.com/Tejo0507/NLAutoencoders-AnswerStableTail under the MIT
licence. The vendored `third_party/nla_inference/` is Apache 2.0.

**Compute.** [record] Every run in the record used one local NVIDIA RTX
4050 Laptop GPU, and the records list no cloud compute. The rewording null
took about 101 GPU-minutes and the stopping-policy pilot 9.1 GPU-hours
(D97, D101).
