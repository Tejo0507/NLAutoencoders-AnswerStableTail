"""Check the manuscript's key numbers against the stored outputs they come from. CPU, read only.

`paper/v2/evidence/key_numbers.json` binds each key number in `paper/v2/` to
a field of a stored output. For every fact this script

1. reads the field (or derives the quantity, for the few listed derivations)
   from the output under `--runs` or `--d76`;
2. formats it the way the manuscript writes numbers (U+2212 for minus, the
   fact's decimals) and compares it with the fact's `written` form;
3. checks that `written` appears in every file the fact lists, and that each
   variant phrasing appears in its files with the same decimals.

It writes nothing. Exit code 0 when every fact checks, 1 otherwise, and 2 when
no fact could be checked (for example, no runs directory).

    python scripts/verify_manuscript_numbers.py --runs D:/elsewhere/runs --d76 D:/exp/d76/repo/runs
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "paper" / "v2"
REGISTRY = V2 / "evidence" / "key_numbers.json"
MINUS = "\u2212"


def _signed(x: float, nd: int) -> str:
    return f"{x:+.{nd}f}".replace("-", MINUS)


def _plain(x: float, nd: int) -> str:
    return f"{x:.{nd}f}".replace("-", MINUS)


def fmt_value(x: float, fmt: str) -> str:
    kind = fmt.rstrip("0123456789")
    nd = fmt[len(kind):]                    # "signed3" -> ("signed", "3")
    if kind == "signed":
        return _signed(x, int(nd))
    if kind == "plain":
        return _plain(x, int(nd))
    if kind == "pct":
        return f"{100 * x:.{int(nd)}f}%"
    if fmt == "int":
        return str(int(round(x)))
    if fmt == "intc":
        return f"{int(round(x)):,}"
    if fmt == "raw":
        return f"{x:g}"
    raise ValueError(f"unknown format {fmt!r}")


def fmt_ci(lo: float, hi: float, fmt: str) -> str:
    nd = int(fmt[len(fmt.rstrip("0123456789")):] or 3)
    return f"[{_plain(lo, nd)}, {_plain(hi, nd)}]"


def dig(node: Any, path: list) -> Any:
    for k in path:
        if isinstance(node, list) and isinstance(k, int) and k < len(node):
            node = node[k]
        elif isinstance(node, dict) and k in node:
            node = node[k]
        else:
            raise KeyError(f"no {k!r} in path {path}")
    return node


def _jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def verbaliser_mean_tokens(path: Path) -> float:
    """Mean generated tokens per verbaliser sample that passed the integrity gate."""
    return statistics.mean(s["n_tokens"] for r in _jsonl(path) for s in r["samples"] if s.get("ok"))


def trace_mean_tokens(path: Path) -> float:
    return statistics.mean(t["n_tokens"] for t in _jsonl(path) if t.get("ok"))


def derive(fact: dict, root: Path) -> float:
    d = fact["derive"]
    if d == "verbaliser_mean_tokens":
        return verbaliser_mean_tokens(root / fact["file"])
    if d == "trace_mean_tokens":
        return trace_mean_tokens(root / fact["file"])
    if d == "nla_check_cost":
        # one check = two verbaliser samples, against one mean trace
        return 2 * verbaliser_mean_tokens(root / fact["file"]) / trace_mean_tokens(root / fact["file2"])
    raise ValueError(f"unknown derivation {d!r}")


def expected(fact: dict, root: Path) -> str:
    if "derive" in fact:
        return fmt_value(derive(fact, root), fact["fmt"])
    data = json.loads((root / fact["file"]).read_text(encoding="utf-8"))
    v = float(dig(data, fact["value"]))
    if "times" in fact:
        v *= float(dig(data, fact["times"]))
    if fact.get("neg"):
        v = -v
    s = fmt_value(v, fact["fmt"])
    ci = None
    if "ci" in fact:
        ci = [float(x) for x in dig(data, fact["ci"])]
    elif "ci2" in fact:
        ci = [float(dig(data, p)) for p in fact["ci2"]]
    if ci is not None:
        lo, hi = ci
        if fact.get("neg"):
            lo, hi = -hi, -lo
        s += " " + fmt_ci(lo, hi, fact.get("ci_fmt", fact["fmt"]))
    return s


def contains_token(haystack: str, needle: str) -> bool:
    """`needle` occurs in `haystack`, not as part of a longer number."""
    return re.search(rf"(?<![\d.]){re.escape(needle)}(?![\d])", haystack) is not None


def decimals(text: str) -> list[str]:
    """Absolute decimal numbers in order, as written (sign and percent dropped)."""
    return re.findall(r"\d+\.\d+", text)


def check(registry: dict, roots: dict[str, Path], v2: Path = V2) -> tuple[list[str], int]:
    failures: list[str] = []
    checked = 0
    cache: dict[str, str] = {}

    def text_of(name: str) -> str:
        if name not in cache:
            cache[name] = re.sub(r"\s+", " ", (v2 / name).read_text(encoding="utf-8"))
        return cache[name]

    for fact in registry["facts"]:
        fid = fact["id"]
        root = roots.get(fact["root"])
        if root is None or not (root / fact["file"]).exists():
            failures.append(f"{fid}: source {fact['root']}:{fact['file']} not available")
            continue
        try:
            exp = expected(fact, root)
        except (KeyError, ValueError, TypeError) as e:
            failures.append(f"{fid}: cannot read the source: {e}")
            continue
        checked += 1
        if not contains_token(fact["written"], exp):
            failures.append(f"{fid}: source gives {exp!r}, manuscript registry says {fact['written']!r}")
        for name in fact["files"]:
            if not contains_token(text_of(name), fact["written"]):
                failures.append(f"{fid}: {fact['written']!r} not found in {name}")
        for var in fact.get("variants", []):
            if decimals(var["text"]) != decimals(exp):
                failures.append(f"{fid}: variant {var['text']!r} disagrees with {exp!r}")
            for name in var["files"]:
                if not contains_token(text_of(name), var["text"]):
                    failures.append(f"{fid}: variant {var['text']!r} not found in {name}")
    return failures, checked


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=ROOT / "runs", help="stored run outputs (read only)")
    ap.add_argument("--d76", type=Path, default=None,
                    help="the D76 experiment's runs directory (read only)")
    ap.add_argument("--registry", type=Path, default=REGISTRY)
    a = ap.parse_args()
    registry = json.loads(a.registry.read_text(encoding="utf-8"))
    roots = {"runs": a.runs}
    if a.d76 is not None:
        roots["d76"] = a.d76
    failures, checked = check(registry, roots)
    n = len(registry["facts"])
    for f in failures:
        print("FAIL", f)
    print(f"{checked} of {n} facts read from their sources; {len(failures)} failure(s)")
    if checked == 0:
        return 2
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
