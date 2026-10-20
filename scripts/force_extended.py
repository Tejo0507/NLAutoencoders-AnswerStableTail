"""Extended-budget forced answers at cut-off menu boundaries (D82). GPU.

Implements the generation of ``docs/EXTENDED_FORCING_PLAN.md`` §2. The forcing
call is the registered one (``force_all_boundaries.py``): the same prompt,
suffix, greedy decoding, per-problem blocks of ``block_size(batch_size)`` and
seed derivation. Only the budget differs, at 256 new tokens. It forces:

* every design-A menu boundary whose registered 48-token completion was cut
  off (``purpose: cut_off``), and
* a seeded sample of 50 menu boundaries whose completion finished
  (``purpose: check``), the plan's instrument check.

Each row records the answer, whether generation finished, the exact number
of generated tokens, and whether the new text begins with the stored 48-token
completion.

    python scripts/force_extended.py --config main --run-id main_nla240 \\
        --forced-dir runs/main_forced --dry-run
    python scripts/force_extended.py --config replication --forced-dir runs/rep1_forced

Writes ``runs/<run_id>_extforced/forced_ext.jsonl`` (resumable) and
``provenance.json``. Never touches ``runs/<run_id>/`` or the registered
completion.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from _stage import base_parser, load_config

from nlaast.data.answers import PERMISSIVE, equivalent, extract_answer
from nlaast.logging_utils import JsonlWriter, read_jsonl, setup as setup_logging, write_json
from nlaast.provenance import git_commit, utcnow
from nlaast.seeding import derive
from nlaast.trace.chunking import Chunk, prefix_text

from force_all_boundaries import block_size, blocks

BUDGET = 256
N_CHECK = 50
CHECK_SEED = 20261020
REAL_KINDS = ("tail", "matched_position", "matched_length")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def registered_completions(traces: dict, forced_dir: Path) -> dict[tuple[str, int], dict]:
    """``(pid, i) -> {finished, text, answer}`` of the registered 48-token
    completion used as the outcome: stored where the scan evaluated the
    boundary, else the D69 completion's (as ``forced_answer_sequence``)."""
    done = {(r["problem_id"], int(r["index"])): r for r in read_jsonl(forced_dir / "forced.jsonl")
            if r["purpose"] == "missing"}
    out = {}
    for pid, t in traces.items():
        for b in t["boundaries"]:
            key = (pid, int(b["index"]))
            if b.get("evaluated"):
                out[key] = {"finished": b.get("forced_finished"), "text": b.get("forced_text"),
                            "answer": b.get("forced_answer")}
            elif key in done:
                r = done[key]
                out[key] = {"finished": r["forced_finished"], "text": r["forced_text"],
                            "answer": r["forced_answer"]}
    return out


def menu_of(cfg) -> dict[str, set[int]]:
    menu: dict[str, set[int]] = {}
    for r in read_jsonl(cfg.dir / "nla" / "reconstructions.jsonl"):
        if r["window_kind"] in REAL_KINDS and r.get("integrity_ok"):
            menu.setdefault(r["problem_id"], set()).add(int(r["chunk_index"]))
    return menu


def select(menu: dict[str, set[int]], reg: dict, n_check: int, seed: int
           ) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    """``(cut_off, check)`` boundary indices per problem (plan §2)."""
    cut, fin = {}, []
    for pid in sorted(menu):
        for i in sorted(menu[pid]):
            r = reg.get((pid, i))
            if r is None or r["finished"] is None:
                raise SystemExit(f"{pid}|{i}: menu boundary without a registered completion")
            if r["finished"]:
                fin.append((pid, i))
            else:
                cut.setdefault(pid, []).append(i)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(fin), size=min(n_check, len(fin)), replace=False)
    check: dict[str, list[int]] = {}
    for k in sorted(int(x) for x in pick):
        pid, i = fin[k]
        check.setdefault(pid, []).append(i)
    return cut, check


def generated_part(forced_text: str | None, suffix: str) -> str:
    t = forced_text or ""
    return t[len(suffix):] if t.startswith(suffix) else t


def main() -> int:
    p = base_parser(__doc__)
    p.add_argument("--forced-dir", required=True, help="the registered D69 completion")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()
    cfg = load_config(args)
    out_dir = Path(args.out_dir) if args.out_dir else cfg.dir.parent / f"{cfg.run_id}_extforced"
    forced_dir = Path(args.forced_dir)
    traces_path = cfg.dir / "traces" / "traces.jsonl"
    traces = {t["problem_id"]: t for t in read_jsonl(traces_path) if t.get("ok")}
    fprov = json.loads((forced_dir / "provenance.json").read_text(encoding="utf-8"))
    if fprov.get("traces_sha256") != sha256(traces_path):
        print(f"{forced_dir} was made from other traces", file=sys.stderr)
        return 1
    reg = registered_completions(traces, forced_dir)
    cut, check = select(menu_of(cfg), reg, N_CHECK, CHECK_SEED)
    n_cut = sum(map(len, cut.values()))
    n_chk = sum(map(len, check.values()))
    blk = block_size(cfg.generation.batch_size)
    print(f"{len(traces)} traces; {n_cut} cut-off menu boundaries in {len(cut)} problems; "
          f"{n_chk} finished ones to check; blocks of {blk}; at most "
          f"{(n_cut + n_chk) * BUDGET} new tokens")
    if args.dry_run:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(out_dir)
    gcfg = cfg.generation
    prov_path = out_dir / "provenance.json"
    prov = {
        "status": "extended-budget forcing (D82); never replaces the registered outcome",
        "plan_sha256": sha256(Path(__file__).resolve().parents[1] / "docs" / "EXTENDED_FORCING_PLAN.md"),
        "run_id": cfg.run_id, "config_hash": cfg.hash(), "traces_sha256": sha256(traces_path),
        "registered_forced_sha256": sha256(forced_dir / "forced.jsonl"),
        "reconstructions_sha256": sha256(cfg.dir / "nla" / "reconstructions.jsonl"),
        "force_answer_suffix": gcfg.force_answer_suffix, "registered_budget":
            gcfg.force_answer_max_new_tokens, "budget": BUDGET, "block_size": blk,
        "decoding": "greedy (temperature 0, top_p 1, do_sample False)",
        "n_cut_off": n_cut, "n_check": n_chk, "check_seed": CHECK_SEED,
    }
    if prov_path.exists():
        old = json.loads(prov_path.read_text(encoding="utf-8"))
        for k in ("config_hash", "traces_sha256", "registered_forced_sha256", "budget",
                  "block_size", "force_answer_suffix", "reconstructions_sha256"):
            if old.get(k) != prov[k]:
                print(f"{prov_path} records {k}={old.get(k)!r}, now {prov[k]!r}; refusing to mix",
                      file=sys.stderr)
                return 1
        prov = {**old, **{k: v for k, v in prov.items() if k not in old}}
    else:
        prov["started"] = utcnow()
        prov["git"] = git_commit()
        write_json(prov_path, prov)

    from nlaast.models.target import TargetModel

    chunks_of = {pid: [Chunk(**{k: c[k] for k in ("index", "text", "char_start", "char_end",
                                                  "token_end", "prefix_tokens")})
                       for c in t["chunks"]] for pid, t in traces.items()}
    model = TargetModel(cfg)
    tok = model.tokenizer

    def prompt(pid, i):
        t = traces[pid]
        return model.build_prompt(t["question"], prefix_text(chunks_of[pid], i, t["trace"])
                                  + gcfg.force_answer_suffix)

    work = [[(pid, i, purpose) for i in b]
            for purpose, sel in (("cut_off", cut), ("check", check))
            for pid in sorted(sel) for b in blocks(sel[pid], blk)]
    writer = JsonlWriter(out_dir / "forced_ext.jsonl", key="id")
    started, done, n_gen = time.monotonic(), 0, 0
    try:
        with writer:
            for batch in work:
                todo = [it for it in batch if not writer.has(f"{it[0]}|{it[1]}|{it[2]}")]
                if not todo:
                    continue
                gens = model.generate([prompt(pid, i) for pid, i, _ in todo],
                                      max_new_tokens=BUDGET, temperature=0.0, top_p=1.0,
                                      do_sample=False,
                                      seed=derive(cfg.seed, "force", todo[0][0], todo[0][1]),
                                      batch_size=len(todo))
                n_gen += len(todo)
                for (pid, i, purpose), g in zip(todo, gens):
                    t = traces[pid]
                    r48 = reg[(pid, i)]
                    old_gen = generated_part(r48["text"], gcfg.force_answer_suffix)
                    text = gcfg.force_answer_suffix + g.text
                    ans = extract_answer(text, PERMISSIVE)
                    writer.write({
                        "id": f"{pid}|{i}|{purpose}", "problem_id": pid, "index": i,
                        "purpose": purpose, "answer": ans, "text": text,
                        "finished": bool(g.finished), "n_tokens": int(g.n_tokens),
                        "matches_final": equivalent(ans, t.get("final_answer")),
                        "registered_answer": r48["answer"],
                        "registered_finished": r48["finished"],
                        "registered_retokenised_tokens": len(
                            tok(old_gen, add_special_tokens=False)["input_ids"]),
                        "begins_with_registered": g.text.startswith(old_gen.rstrip("�")),
                        "agrees_with_registered": equivalent(ans, r48["answer"])
                        or (ans is None and r48["answer"] is None),
                        "prefix_tokens": chunks_of[pid][i].prefix_tokens,
                        "block": [[q, j] for q, j, _ in todo]})
                done += 1
                if done % 20 == 0:
                    log.info("%d/%d blocks, %.1fs per block", done, len(work),
                             (time.monotonic() - started) / done)
    finally:
        model.close()

    rows = read_jsonl(out_dir / "forced_ext.jsonl")
    chk = [r for r in rows if r["purpose"] == "check"]
    prov.update({
        "finished": utcnow(),
        "this_session": {"generations": n_gen, "seconds": time.monotonic() - started},
        "n_rows": len(rows),
        "check_answer_agreement": (sum(r["agrees_with_registered"] for r in chk) / len(chk)
                                   if chk else None),
        "check_token_count_max_abs_diff": (max(abs(r["n_tokens"] - r["registered_retokenised_tokens"])
                                               for r in chk) if chk else None),
        "cut_off_still_cut_off_at_budget": (
            sum(not r["finished"] for r in rows if r["purpose"] == "cut_off")
            / max(1, sum(r["purpose"] == "cut_off" for r in rows))),
        "forced_ext_sha256": sha256(out_dir / "forced_ext.jsonl"),
    })
    write_json(prov_path, prov)
    print(json.dumps(prov, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
