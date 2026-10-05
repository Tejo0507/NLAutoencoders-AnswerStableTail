"""Re-derive every parsed answer in an existing trace file.

Answer extraction is the one component every downstream number depends on, and
it is the component most likely to need a fix after seeing real output. Without
this utility a parser change would mean regenerating hours of traces; with it,
the stored text is re-parsed in seconds and the `ast` stage onwards can simply re-run.

    python scripts/reparse_traces.py --config pilot
    python scripts/reparse_traces.py --config pilot --dry-run

Traces written before the continuation-text retention change do not carry the
continuation bodies, so their continuation answers cannot be re-derived. Those
are reported as unrepairable rather than silently left stale.
"""

from __future__ import annotations

import sys

from _stage import base_parser, setup

from nlaast.data.answers import PERMISSIVE, equivalent, extract_answer
from nlaast.logging_utils import read_jsonl, write_jsonl
from nlaast.trace.chunking import Chunk, prefix_text


def main() -> int:
    p = base_parser(__doc__)
    p.add_argument("--dry-run", action="store_true",
                   help="report what would change without writing")
    args = p.parse_args()
    cfg, manifest, log = setup(args)

    path = cfg.dir / "traces" / "traces.jsonl"
    rows = read_jsonl(path)
    if not rows:
        log.error("no traces at %s", path)
        return 1

    changed = {"final": 0, "parsed": 0, "forced": 0, "continuations": 0}
    unrepairable = 0

    for row in rows:
        if not row.get("ok"):
            continue
        chunks = [Chunk(**{k: c[k] for k in ("index", "text", "char_start",
                                             "char_end", "token_end", "prefix_tokens")})
                  for c in row.get("chunks", [])]

        new_final = extract_answer(row["trace"], PERMISSIVE)
        if new_final != row.get("final_answer"):
            changed["final"] += 1
            row["final_answer"] = new_final
        row["final_correct"] = equivalent(new_final, row.get("gold"))

        for b in row.get("boundaries", []):
            idx = b["index"]
            if idx < len(chunks):
                np_ = extract_answer(prefix_text(chunks, idx, row["trace"]), PERMISSIVE)
                if np_ != b.get("parsed_answer"):
                    changed["parsed"] += 1
                b["parsed_answer"] = np_

            if b.get("forced_text"):
                nf = extract_answer(b["forced_text"], PERMISSIVE)
                if nf != b.get("forced_answer"):
                    changed["forced"] += 1
                b["forced_answer"] = nf
            b["forced_matches_final"] = equivalent(b.get("forced_answer"), new_final)

            tails = b.get("continuation_tails")
            if tails:
                prefix = prefix_text(chunks, idx, row["trace"]) if idx < len(chunks) else ""
                new_answers = [extract_answer(prefix + t, PERMISSIVE) for t in tails]
                if new_answers != b.get("continuation_answers"):
                    changed["continuations"] += 1
                b["continuation_answers"] = new_answers
            elif b.get("n_continuations"):
                # Pre-retention trace: the bodies are gone, so these answers
                # keep whatever the old parser produced. Said plainly rather
                # than left to look re-derived.
                unrepairable += 1
            b["continuations_matching"] = sum(
                1 for a in b.get("continuation_answers", []) if equivalent(a, new_final)
            )

        for s in row.get("entropy_samples", []):
            if s.get("text"):
                s["answer"] = extract_answer(s["text"], PERMISSIVE)

    log.info("changes: %s", changed)
    if unrepairable:
        log.warning("%d boundaries have continuation answers that cannot be "
                    "re-derived (traces predate continuation-text retention). "
                    "Regenerate to refresh them.", unrepairable)
    if args.dry_run:
        log.info("dry run - nothing written")
        return 0

    backup = path.with_suffix(".jsonl.bak")
    if not backup.exists():
        backup.write_bytes(path.read_bytes())
        log.info("original kept at %s", backup.name)
    write_jsonl(path, rows)
    log.info("rewrote %s (%d traces). Re-run stages 05 onwards.", path.name, len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
