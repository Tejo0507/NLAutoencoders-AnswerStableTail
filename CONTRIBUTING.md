# Contributing

Short guide for changing this repository. The README covers setup, the
pipeline and the tests in more detail; this file is the checklist before you
open a pull request.

## Checks to run

Branch protection requires the CPU suite and a defects-only lint. From a
checkout with the project installed (see *Setup* in the README):

```bash
python -m pytest -m "not slow"
ruff check .
```

`not slow` deselects the GPU tests. On Windows the README's
`.venv/Scripts/python.exe -m pytest ...` form is equivalent. CI runs the same
two commands on every pull request
([`.github/workflows/cpu-tests.yml`](.github/workflows/cpu-tests.yml));
`ruff` is configured in `pyproject.toml`.

## Files that must not change

Some files are pinned by SHA-256 for confirmatory analysis and the results
release. Do not edit them in a normal contribution:

- frozen plans and designs: `docs/*_PLAN.md`, `docs/*_DESIGN.md`,
  `docs/*_PROTOCOL.md`
- `configs/replication.yaml`
- the stopping-policy pilot configs:
  `configs/stopping_pilot_protocol.json`,
  `configs/pilot_cal.yaml`,
  `configs/pilot_eval.yaml`
- `scripts/rq2_claim_probe.py` (code hash recorded with the exploratory
  evaluation in D57)

`.gitattributes` forces LF endings on the plans, protocols and replication
config so a checkout cannot change those hashes by accident. D57 and D61 in
[`docs/DECISIONS.md`](docs/DECISIONS.md) record why those pins exist and how
post-hoc scripts must behave around them.

If a change to a pinned file is genuinely required, stop and open a decision
(see below) before rewriting it.

## Decisions and results

Do not silently rewrite a past conclusion. New results, deviations and
implementation choices go in [`docs/DECISIONS.md`](docs/DECISIONS.md) as a
**new D-number** (next free `D__`), in the same format as the existing
entries. Prefer a new entry over editing an old one in place.

## Pull requests

Use the template at
[`.github/pull_request_template.md`](.github/pull_request_template.md). Fill
in:

- what the change does, and whether it touches the scientific record;
- whether any hash-pinned file was touched (ideally none);
- the CPU suite and `ruff check .` results.

Keep the diff focused. Prefer the plain prose style of the README: short
sentences, British spelling where the project already uses it
(behaviour, normalise, and so on).
