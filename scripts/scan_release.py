"""Scan a release package, a directory or a ``.tar.gz``, for data that should not ship.

Looks for:

* Windows absolute paths of two or more segments (``abs_windows_path``) and
  bare roots of one segment (``abs_windows_root``, only as a whole JSON
  string or a delimited token, and in JSON only with an escaped separator,
  so that model text such as ``s:\\n`` or ``s:\\\\frac`` is not reported);
* home-directory paths, e-mail addresses, Hugging Face, GitHub, ``sk-`` and
  AWS key formats, private-key blocks, bearer headers and secret assignments;
* the local identities this machine knows: user name, host name, git user
  name and e-mail, plus any ``--identity``.

Coverage: text files are scanned whole (decoded as UTF-8, invalid bytes
replaced); ``.npz`` files only through their object arrays, which hold the
JSON metadata, not their numeric arrays; PNGs are skipped. UNC paths
(``\\\\server\\share``), relative paths and secrets in formats not listed
above are not looked for.

Prints file, category and count. Path findings show their drive and first
directory only; nothing else that matched is printed, so a found secret does
not reach a terminal or a log. Exit 0 when nothing is found, 1 otherwise.
A clean result is the absence of these patterns, not proof that nothing
sensitive remains.

    python scripts/scan_release.py <package>/nlaast-runs
    python scripts/scan_release.py <package>/<archive>.tar.gz
"""

from __future__ import annotations

import argparse
import getpass
import io
import os
import re
import socket
import subprocess
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

SKIP_EXT = {".png"}
#: Directories that hold user homes on Linux and macOS.
HOME_ROOTS = ("home", "Users", "root")

PATTERNS = {
    # A drive letter, a separator, then two path segments; the "\n" escapes in
    # JSON-encoded model text ("s:\n...") are excluded.
    "abs_windows_path": re.compile(
        r"(?<![A-Za-z0-9])[A-Za-z]:(?:\\\\|\\|/)(?!n[^A-Za-z0-9_])[A-Za-z0-9_.~$ -]{1,64}"
        r"(?:\\\\|\\|/)[A-Za-z0-9_.~$-]"),
    "home_path": re.compile("/(?:" + "|".join(HOME_ROOTS) + r")/[A-Za-z0-9_.-]+"),
    # "abs_windows_root" (a drive and one directory) is added per file type in
    # scan(), because only the file type tells a JSON escape from a separator.
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9.-]*[A-Za-z]{2,}"),
    "hf_token": re.compile(r"\bhf_[A-Za-z0-9]{30,}"),
    "github_token": re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}"),
    "api_key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}|\bAKIA[0-9A-Z]{16}\b"),
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "bearer": re.compile(r"(?i)authorization['\"]?\s*[:=]\s*['\"]?bearer\s+\S+"),
    "secret_assignment": re.compile(
        r"(?i)\b(?:password|passwd|api[_-]?key|secret[_-]?key|access[_-]?token|hf_token)\b"
        r"['\"]?\s*[:=]\s*['\"][^'\"]{6,}"),
}


_ROOT_COMP = r"[A-Za-z0-9_.~$ -]+"
#: A bare root only where it is a whole value: in JSON a complete string
#: written with an escaped separator; elsewhere a token delimited by quotes,
#: whitespace or brackets. Model text such as "s:\\frac" or "s:\n" is not one.
ROOT_PATTERNS = {
    "json": re.compile(rf'(?<=")[A-Za-z]:(?:\\\\|/){_ROOT_COMP}(?=")'),
    "text": re.compile(rf"(?<=[\"'\s=(])[A-Za-z]:(?:\\\\|\\|/){_ROOT_COMP}(?=[\"'\s),]|$)", re.M),
}


def local_identities(extra: list[str]) -> set[str]:
    found = {getpass.getuser(), socket.gethostname(), os.environ.get("USERNAME", ""),
             os.environ.get("COMPUTERNAME", ""), *extra}
    for key in ("user.name", "user.email"):
        try:
            found.add(subprocess.run(["git", "config", key], capture_output=True,
                                     text=True).stdout.strip())
        except OSError:
            pass
    return {x for x in found if len(x) >= 3}


def iter_files(target: Path):
    if target.is_dir():
        for p in sorted(target.rglob("*")):
            if p.is_file():
                yield p.relative_to(target).as_posix(), p.read_bytes()
    else:
        with tarfile.open(target, "r:gz") as tf:
            for m in tf.getmembers():
                if m.isfile():
                    yield m.name, tf.extractfile(m).read()


def text_of(name: str, data: bytes) -> str | None:
    suffix = Path(name).suffix.lower()
    if suffix in SKIP_EXT:
        return None
    if suffix == ".npz":
        import numpy as np
        z = np.load(io.BytesIO(data), allow_pickle=True)
        # The strings themselves, one per line: a repr() of the array would
        # double every backslash and hide an escaped path from the patterns.
        return "\n".join(str(x) for k in z.files if z[k].dtype == object or z[k].dtype.kind in "US"
                         for x in z[k].ravel().tolist())
    return data.decode("utf-8", errors="replace")


def redact(category: str, s: str) -> str:
    if category in {"abs_windows_path", "abs_windows_root"}:
        parts = re.split(r"\\\\|\\|/", s)
        return f"{parts[0]}\\{parts[1][:24] if len(parts) > 1 else ''}\\..."
    return "<redacted>"


def scan(target: Path, identities: set[str]) -> tuple[int, dict]:
    patterns = dict(PATTERNS)
    for i, ident in enumerate(sorted(identities)):
        patterns[f"identity_{i}"] = re.compile(re.escape(ident), re.I)
    findings: dict[str, dict[str, list]] = defaultdict(dict)
    n = 0
    for name, data in iter_files(target):
        n += 1
        text = text_of(name, data)
        if not text:
            continue
        kind = "json" if Path(name).suffix.lower() in {".json", ".jsonl"} else "text"
        for cat, rx in {**patterns, "abs_windows_root": ROOT_PATTERNS[kind]}.items():
            hits = [m.group(0) for m in rx.finditer(text)]
            if hits:
                findings[name][cat] = [len(hits), sorted({redact(cat, h) for h in hits})[:3]]
    return n, findings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", type=Path)
    ap.add_argument("--identity", action="append", default=[],
                    help="another string that must not appear; repeatable")
    args = ap.parse_args()
    identities = local_identities(args.identity)
    n, findings = scan(args.target, identities)
    print(f"scanned {n} files; {len(identities)} local identities checked")
    for name in sorted(findings):
        for cat, (count, sample) in sorted(findings[name].items()):
            print(f"{name}\t{cat}\t{count}\t{sample}")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
