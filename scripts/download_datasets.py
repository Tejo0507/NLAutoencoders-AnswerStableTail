"""Download GSM8K and MATH datasets into data/raw/ as JSONL files.

Usage:
    python scripts/download_datasets.py
"""

from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"

MATH_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def download_gsm8k() -> None:
    out_dir = RAW_DIR / "gsm8k"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("openai/gsm8k", "main")
    for split, data in ds.items():
        out_path = out_dir / f"{split}.jsonl"
        data.to_json(out_path, orient="records", lines=True)
        print(f"gsm8k/{split}: {len(data)} rows -> {out_path}")


def download_math() -> None:
    out_dir = RAW_DIR / "math"
    out_dir.mkdir(parents=True, exist_ok=True)

    for subject in MATH_SUBJECTS:
        ds = load_dataset("EleutherAI/hendrycks_math", subject)
        for split, data in ds.items():
            split_dir = out_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            out_path = split_dir / f"{subject}.jsonl"
            data.to_json(out_path, orient="records", lines=True)
            print(f"math/{split}/{subject}: {len(data)} rows -> {out_path}")


if __name__ == "__main__":
    download_gsm8k()
    download_math()
