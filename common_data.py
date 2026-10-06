import glob
import json
import re

import pandas as pd


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_traces():
    seen = {}
    for path in sorted(glob.glob("runs/*/traces/traces.jsonl")):
        for r in read_jsonl(path):
            if r.get("ok", True) and r["id"] not in seen:
                seen[r["id"]] = r
    return list(seen.values())


def load_ast():
    seen = {}
    for path in sorted(glob.glob("runs/*/ast/ast.jsonl")):
        for r in read_jsonl(path):
            seen[r["id"]] = r
    return seen


def text_features(row):
    trace = row["trace"]
    question = row["question"]
    words = re.findall(r"[A-Za-z']+", trace)
    return {
        "q_words": len(question.split()),
        "q_numbers": len(re.findall(r"\d+(?:\.\d+)?", question)),
        "trace_chars": len(trace),
        "trace_words": len(words),
        "n_tokens": row.get("n_tokens", len(trace) // 4),
        "n_chunks": row.get("n_chunks", trace.count("\n\n") + 1),
        "n_equals": trace.count("="),
        "n_latex": trace.count("\\"),
        "n_numbers": len(re.findall(r"\d+(?:\.\d+)?", trace)),
        "n_lines": trace.count("\n") + 1,
        "cue_verify": len(re.findall(r"\b(check|verify|confirm|double-check|wait|actually)\b", trace, re.I)),
        "cue_conclude": len(re.findall(r"\b(therefore|thus|hence|so)\b", trace, re.I)),
        "cue_answer": len(re.findall(r"answer is", trace, re.I)),
        "is_math": int(row.get("dataset") == "math"),
        "truncated": int(bool(row.get("truncated", False))),
    }


def build_table(with_ast=False):
    ast = load_ast() if with_ast else {}
    records = []
    for row in load_traces():
        rec = {"id": row["id"], "dataset": row.get("dataset")}
        rec.update(text_features(row))
        rec["final_correct"] = int(bool(row.get("final_correct")))
        if with_ast:
            a = ast.get(row["id"])
            if a is None or a["status"] != "ok":
                continue
            rec["tail_fraction"] = a["tail_fraction"]
        records.append(rec)
    return pd.DataFrame(records)
