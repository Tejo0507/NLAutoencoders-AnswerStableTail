"""Forced answers at the boundaries the AST scan never reached (D69). GPU.

PROJECT_PLAN §5 scores a stop at boundary ``s`` on the answer **forced** from
the prefix ``P_s``. The traces stage generated forced answers only at the
boundaries its backward scan evaluated (D4, D5), so the registered outcome
cannot be scored where a rule stops earlier. This script fills those gaps with
the same instrument the traces stage used (``boundary_evidence_batch``): the
same chat prompt, the same suffix, the same ``force_answer_max_new_tokens``,
greedy decoding, blocks of at most four boundaries of one problem per batch,
and the same seed derivation. Nothing else is generated.

It also re-forces a seeded random sample of boundaries that **already** have a
stored forced answer (``purpose: repro``). Greedy decoding in 4-bit with
batch padding is not guaranteed bit-stable, so the agreement between stored
and regenerated answers is measured rather than assumed. The completed answers
are used only if that agreement is reported beside them.

Writes ``runs/<run_id>_forced/forced.jsonl`` (resumable, one row per boundary)
and ``provenance.json``. Never touches ``runs/<run_id>/``.

    python scripts/force_all_boundaries.py --config main --dry-run
    python scripts/force_all_boundaries.py --config main
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

#: Stored boundaries re-forced to measure the instrument's reproducibility.
N_REPRO = 200
REPRO_SEED = 20261019
#: The traces stage's block size: ``max(1, min(4, generation.batch_size))``.
MAX_BLOCK = 4


def select_boundaries(traces: list[dict], n_repro: int, seed: int
                      ) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    """``(missing, repro)``: boundary indices per problem.

    ``missing`` is every boundary without a stored forced answer, i.e. not
    ``evaluated``. ``repro`` is a uniform random sample, without replacement,
    of ``n_repro`` boundaries that *are* evaluated, drawn with ``seed`` over a
    sorted list so the draw does not depend on file order.
    """
    missing: dict[str, list[int]] = {}
    stored: list[tuple[str, int]] = []
    for t in traces:
        if not t.get("ok"):
            continue
        for b in t.get("boundaries", []):
            if b.get("evaluated"):
                stored.append((t["problem_id"], int(b["index"])))
            else:
                missing.setdefault(t["problem_id"], []).append(int(b["index"]))
    stored.sort()
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(stored), size=min(n_repro, len(stored)), replace=False)
    repro: dict[str, list[int]] = {}
    for k in sorted(int(i) for i in pick):
        pid, idx = stored[k]
        repro.setdefault(pid, []).append(idx)
    return ({p: sorted(v) for p, v in missing.items()},
            {p: sorted(v) for p, v in repro.items()})


def blocks(indices: list[int], size: int = MAX_BLOCK) -> list[list[int]]:
    """Indices in descending blocks of at most ``size``, as the backward scan
    batches them."""
    desc = sorted(set(indices), reverse=True)
    return [desc[i:i + size] for i in range(0, len(desc), size)]


def cross_problem_batches(items: list[tuple[str, int, str]], lengths: dict, size: int
                          ) -> list[list[tuple[str, int, str]]]:
    """Batches of ``size`` boundaries from any problems, sorted by prompt length.

    The traces stage batches at most four boundaries of one problem at a time.
    Long, nearly equal-length prompts from different problems pad far less
    and fill the GPU better (D75). Sorting by length, then by id, makes the
    batches deterministic.
    """
    order = sorted(items, key=lambda it: (lengths[(it[0], it[1])], it[0], it[1], it[2]))
    return [order[i:i + size] for i in range(0, len(order), size)]


def main() -> int:
    p = base_parser(__doc__)
    p.add_argument("--dry-run", action="store_true",
                   help="count the work and exit without loading a model")
    p.add_argument("--batch", type=int, default=0,
                   help="batch across problems, sorted by length (D75); 0 = the "
                        "traces stage's per-problem blocks of four")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--repro-only", type=int, default=0,
                   help="benchmark: re-force only this many stored boundaries")
    p.add_argument("--menu-only", action="store_true",
                   help="force only the verbalised window chunks (design A; D75)")
    p.add_argument("--n-repro", type=int, default=N_REPRO,
                   help="stored boundaries re-forced to measure reproducibility")
    args = p.parse_args()
    cfg = load_config(args)
    out_dir = Path(args.out_dir) if args.out_dir else cfg.dir.parent / f"{cfg.run_id}_forced"
    traces = [t for t in read_jsonl(cfg.dir / "traces" / "traces.jsonl") if t.get("ok")]
    missing, repro = select_boundaries(traces, args.repro_only or args.n_repro, REPRO_SEED)
    if args.repro_only:
        missing = {}
    if args.menu_only:
        # Design A can stop only at chunks the verbaliser scored, so only
        # those need a measured outcome (D75). Design B is then not computable.
        menu: dict[str, set[int]] = {}
        for r in read_jsonl(cfg.dir / "nla" / "reconstructions.jsonl"):
            if r["window_kind"] != "gaussian_control" and r.get("integrity_ok"):
                menu.setdefault(r["problem_id"], set()).add(int(r["chunk_index"]))
        missing = {p: [i for i in v if i in menu.get(p, set())] for p, v in missing.items()}
        missing = {p: v for p, v in missing.items() if v}
    n_missing = sum(len(v) for v in missing.values())
    n_repro = sum(len(v) for v in repro.values())
    budget = cfg.generation.force_answer_max_new_tokens
    print(f"{len(traces)} traces; {n_missing} boundaries without a forced answer "
          f"in {len(missing)} problems; {n_repro} stored boundaries to re-force; "
          f"at most {(n_missing + n_repro) * budget} new tokens")
    if args.dry_run:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(out_dir)
    prov_path = out_dir / "provenance.json"
    traces_sha = hashlib.sha256(
        (cfg.dir / "traces" / "traces.jsonl").read_bytes()).hexdigest()
    prov = {
        "status": "post hoc completion of the registered O3 instrument (D69)",
        "run_id": cfg.run_id, "config_hash": cfg.hash(),
        "traces_sha256": traces_sha,
        "force_answer_suffix": cfg.generation.force_answer_suffix,
        "force_answer_max_new_tokens": budget,
        "decoding": "greedy (temperature 0, top_p 1, do_sample False)",
        "block_size": MAX_BLOCK, "n_missing": n_missing, "n_repro": n_repro,
        "repro_seed": REPRO_SEED,
        "menu_only": bool(args.menu_only), "repro_only": int(args.repro_only),
    }
    if prov_path.exists():
        old = json.loads(prov_path.read_text(encoding="utf-8"))
        for k in ("config_hash", "traces_sha256", "force_answer_suffix",
                  "force_answer_max_new_tokens", "repro_seed"):
            if old.get(k) != prov[k]:
                print(f"{prov_path} records {k}={old.get(k)!r}, now {prov[k]!r}; "
                      f"refusing to mix two instruments in one file", file=sys.stderr)
                return 1
        prov = {**old, **{k: v for k, v in prov.items() if k not in old}}
    else:
        prov["started"] = utcnow()
        prov["git"] = git_commit()
        write_json(prov_path, prov)

    from nlaast.models.target import TargetModel

    by_pid = {t["problem_id"]: t for t in traces}
    gcfg = cfg.generation
    writer = JsonlWriter(out_dir / "forced.jsonl", key="id")
    chunks_of = {pid: [Chunk(**{k: c[k] for k in ("index", "text", "char_start", "char_end",
                                                  "token_end", "prefix_tokens")})
                       for c in t["chunks"]] for pid, t in by_pid.items()}
    model = TargetModel(cfg)

    def prompt(pid, i):
        t = by_pid[pid]
        return model.build_prompt(t["question"], prefix_text(chunks_of[pid], i, t["trace"])
                                  + gcfg.force_answer_suffix)

    if args.batch:
        items = [(pid, i, purpose) for purpose, sel in (("missing", missing), ("repro", repro))
                 for pid in sorted(sel) for i in sel[pid]]
        lengths = {(pid, i): len(model.tokenizer(prompt(pid, i), add_special_tokens=False)["input_ids"])
                   for pid, i, _ in items}
        work = cross_problem_batches(items, lengths, args.batch)
    else:
        work = [[(pid, i, purpose) for i in blk]
                for purpose, sel in (("missing", missing), ("repro", repro))
                for pid in sorted(sel) for blk in blocks(sel[pid])]
    prov["batching"] = f"cross-problem, size {args.batch}" if args.batch else "per-problem blocks of 4"
    started, done, n_gen = time.monotonic(), 0, 0
    try:
        with writer:
            for batch in work:
                todo = [it for it in batch if not writer.has(f"{it[0]}|{it[1]}|{it[2]}")]
                if not todo:
                    continue
                gens = model.generate([prompt(pid, i) for pid, i, _ in todo],
                                      max_new_tokens=budget, temperature=0.0,
                                      top_p=1.0, do_sample=False,
                                      seed=derive(cfg.seed, "force", todo[0][0], todo[0][1]),
                                      batch_size=len(todo))
                n_gen += len(todo)
                for (pid, i, purpose), g in zip(todo, gens):
                    t = by_pid[pid]
                    text = gcfg.force_answer_suffix + g.text
                    ans = extract_answer(text, PERMISSIVE)
                    row = {"id": f"{pid}|{i}|{purpose}", "problem_id": pid, "index": i,
                           "purpose": purpose, "forced_answer": ans, "forced_text": text,
                           "forced_finished": bool(g.finished),
                           "forced_matches_final": equivalent(ans, t.get("final_answer")),
                           "prefix_tokens": chunks_of[pid][i].prefix_tokens,
                           "block": [[p, j] for p, j, _ in todo]}
                    if purpose == "repro":
                        b = next(x for x in t["boundaries"] if x["index"] == i)
                        row["stored_forced_answer"] = b.get("forced_answer")
                        row["agrees_with_stored"] = equivalent(ans, b.get("forced_answer")) \
                            or (ans is None and b.get("forced_answer") is None)
                    writer.write(row)
                done += 1
                if done % 20 == 0:
                    rate = (time.monotonic() - started) / done
                    log.info("%d/%d blocks, %.1fs per block", done, len(work), rate)
    finally:
        model.close()

    elapsed = time.monotonic() - started
    rows = read_jsonl(out_dir / "forced.jsonl")
    rep = [r for r in rows if r["purpose"] == "repro"]
    prov.update({
        "finished": utcnow(),
        "this_session": {"generations": n_gen, "seconds": elapsed,
                         "seconds_per_generation": elapsed / n_gen if n_gen else None},
        "n_rows": len(rows),
        "n_missing_written": sum(r["purpose"] == "missing" for r in rows),
        "n_repro_written": len(rep),
        "repro_agreement": (sum(bool(r["agrees_with_stored"]) for r in rep) / len(rep)
                            if rep else None),
        "forced_sha256": hashlib.sha256(
            (out_dir / "forced.jsonl").read_bytes()).hexdigest(),
    })
    write_json(prov_path, prov)
    print(json.dumps(prov, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
