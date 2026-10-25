"""Complete the pre-specified utility estimand (P1 minus P0 net saving) that `evaluate` skipped.

`stopping_pilot.py evaluate` pairs the two policies' walks only when both have a
calibrated threshold. P0 has none (it never stops), so its net saving is 0 on
every problem and the paired estimand is fully defined. This script computes
it with the same module functions, the frozen threshold and the protocol's
bootstrap seed, and writes a new file. It reads only; `evaluation.json` is not
touched.

    python scripts/complete_pilot_c2.py <repo dir>
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

repo = Path(sys.argv[1])
sys.path.insert(0, str(repo / "src"))
from nlaast.analysis.stopping_pilot import (  # noqa: E402
    Walk, build_problems, cp_interval, paired_saving, walk)

runs = repo / "runs"
proto = json.loads((repo / "configs" / "stopping_pilot_protocol.json").read_text(encoding="utf-8"))
cal_path = runs / "pilot_calibration" / "calibration.json"
cal = json.loads(cal_path.read_text(encoding="utf-8"))
assert hashlib.sha256(cal_path.read_bytes()).hexdigest() == \
    (runs / "pilot_calibration" / "calibration.sha256").read_text().strip()
rows = [json.loads(l) for l in open(runs / "pilot_eval" / "pilot_scores.jsonl", encoding="utf-8")]
tau = cal["policies"]["P1_same_label_probe"]["tau"]
probs = build_problems(rows, lambda r: r["P1_same_label_probe"], proto["truncation_rule"])
w1 = walk(probs, tau)
w0 = Walk(np.zeros(len(probs), bool), np.ones(len(probs), bool), np.zeros(len(probs)),
          np.zeros(len(probs), bool))                       # P0: never stops
rng = np.random.default_rng(int(proto["seeds"]["bootstrap"]))
d = paired_saving(w1, w0, 2000, rng)
lo, hi = cp_interval(int(w1.stopped.sum()), len(probs))
by_ds = {}
for ds in ("gsm8k", "math"):
    keep = np.array([next(r["dataset"] for r in rows if r["problem_id"] == p.problem_id) == ds
                     for p in probs])
    st = w1.stopped[keep]
    by_ds[ds] = {"n_problems": int(keep.sum()), "n_stopped": int(st.sum()),
                 "n_unsafe": int((~w1.safe[keep] & st).sum()),
                 "mean_net_saving": float(w1.saving[keep].mean())}
out = {
    "status": ("completes the pre-specified C2 estimand that evaluate skipped because P0 has "
               "no threshold; same functions, frozen threshold, protocol bootstrap seed"),
    "P1_minus_P0_net_saving": d,
    "delta_save": proto["delta_save"],
    "reading": ("mean difference above delta_save with its 95% CI above it" if d["ci95"][0] > proto["delta_save"]
                else "not shown to exceed delta_save"),
    "P1_coverage_ci95": [lo, hi],
    "P1_by_dataset": by_ds,
}
dest = runs / "pilot_evaluation_supplement"
dest.mkdir(exist_ok=False)
(dest / "c2_completed.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1))
