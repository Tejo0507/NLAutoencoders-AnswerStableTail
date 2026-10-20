"""Engineering benchmark: decode throughput by batch size on this GPU (D78). Not science.

Times AST-style continuations: real prefixes from the main run, 200 sampled
tokens at T = 0.8, the same call the traces stage makes. Each batch size runs
the same prompts. Writes ``runs/bench_decode_batch/results.json``.

    python scripts/bench_decode_batch.py --config main
"""

from __future__ import annotations

import json
import sys
import time

from _stage import base_parser, load_config

from nlaast.logging_utils import read_jsonl, write_json
from nlaast.trace.chunking import Chunk, prefix_text

N_PROMPTS = 8


def main() -> int:
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args)
    from nlaast.models.target import TargetModel

    traces = [t for t in read_jsonl(cfg.dir / "traces" / "traces.jsonl") if t.get("ok")][:N_PROMPTS]
    model = TargetModel(cfg)
    prompts = []
    for t in traces:
        chunks = [Chunk(**{k: c[k] for k in ("index", "text", "char_start", "char_end",
                                             "token_end", "prefix_tokens")}) for c in t["chunks"]]
        i = len(chunks) // 2
        prompts.append(model.build_prompt(t["question"], prefix_text(chunks, i, t["trace"])))
    out = {}
    try:
        model.generate(prompts[:1], max_new_tokens=8, temperature=0.8, top_p=0.95,
                       do_sample=True, seed=1, batch_size=1)   # warm-up
        for bs in (1, 2, 4):
            t0 = time.monotonic()
            gens = model.generate(prompts, max_new_tokens=cfg.ast.continuation_max_new_tokens,
                                  temperature=cfg.ast.continuation_temperature,
                                  top_p=cfg.generation.top_p, do_sample=True, seed=7,
                                  batch_size=bs)
            dt = time.monotonic() - t0
            tok = sum(g.n_tokens for g in gens)
            out[bs] = {"seconds": dt, "tokens": tok, "tokens_per_second": tok / dt}
            print(bs, out[bs], flush=True)
    finally:
        model.close()
    path = cfg.dir.parent / "bench_decode_batch" / "results.json"
    write_json(path, {"prompts": N_PROMPTS, "results": out})
    return 0


if __name__ == "__main__":
    sys.exit(main())
