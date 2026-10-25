# Declarations

*Revised 2026-10-25 (D104) with the author's confirmations. It is laid out
for an IEEE publication, whose policies were checked on 2026-10-25:
- AI-generated content is disclosed in the Acknowledgments;
- a statement on human-subjects oversight is needed only for research
  with human subjects or animals;
- the venue's template decides where funding and conflict-of-interest
  statements go (often the first-page footnote and the Acknowledgments).

Every factual statement below was supplied by the author or is in the
project's records. The text awaits the author's approval.*

**Author.** Tejo Sridhar M V S (sole author), Department of Computational
Intelligence, School of Computing, SRM Institute of Science and Technology,
Kattankulathur 603203, Tamil Nadu, India. ORCID:
https://orcid.org/0009-0007-5023-1092. Corresponding author:
tm1872@srmist.edu.in.

**Funding.** This research received no funding.

**Conflict of interest.** The author declares no competing financial or
professional interests.

**Ethics statement.** This study did not involve human participants or
animals. No surveys, interviews or experiments with people were conducted,
and no private personal data were collected.
- All experiments are computational. They run openly released language
  models on two existing public benchmarks, GSM8K [27] and MATH [28].
- A protocol for human annotation of the explanations was prepared but not
  run.
- The only annotation that exists was produced by an AI system, not by
  people.

Because the work involved no human-subjects research, no institutional
review board review was sought, and no approval or exemption is claimed.
Informed consent was not applicable.

**Author contributions.** The author is the sole author and is
responsible for every part of the work and for the integrity of the
article. In CRediT terms: conceptualization, methodology, software,
validation, formal analysis, investigation, data curation, writing
(original draft, review and editing), visualization and project
administration. The project's decision record (`docs/DECISIONS.md`)
documents the author's decisions, including every plan freeze, approval
and deviation. The AI assistance disclosed below was carried out under
the author's direction and was reviewed by the author.

**Acknowledgments and use of AI-generated content.** The author used
Anthropic's Claude models through the Claude Code agent. The session
records name Claude Opus 4.7, Opus 5 and Opus 5.5, and Claude Sonnet 5.5
and Haiku 4.5. They were used at the following levels:
- *Software and analysis:* writing and testing the analysis, release and
  checking code, and the scripts that generate the figures and tables.
- *Experiments:* running the GPU runs that the author approved, including
  the rewording null and the stopping-policy pilot, and computing the
  analyses from their outputs.
- *Verification:* checking every reported number against the stored
  outputs, and the references against their bibliographic records.
- *Manuscript:* drafting and revising the text of every section (Abstract,
  §1–§7, the captions and these declarations).

The author set the research direction. Designs proposed with AI
assistance were adopted only after the author approved them, as every
plan freeze, approval and deviation in `docs/DECISIONS.md` records. The
author reviewed the AI-generated content and takes full responsibility for
the article's content. A single AI annotation pass of explanation claims (RQ2,
amendment A1) is exploratory and is labelled as not independent.

**Data availability.**
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

**Code availability.** All code, configurations, frozen plans and the
decision record are at
https://github.com/Tejo0507/NLAutoencoders-AnswerStableTail under the MIT
licence. The vendored `third_party/nla_inference/` is Apache 2.0.

**Compute.** Every run in the record used one local NVIDIA RTX 4050
Laptop GPU, and the records list no cloud compute. The rewording null
took about 101 GPU-minutes and the stopping-policy pilot 9.1 GPU-hours
(D97, D101).
