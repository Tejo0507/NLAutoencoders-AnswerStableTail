"""Activation storage, hashed and resumable.

Objective O2 requires activation-cache hashes in the reproducibility record, so
the store is content-addressed: each problem's activation block is written once
with its SHA-256, and the index records shape, dtype, layer and digest. A later
stage that reads a vector can prove which bytes it read.

Layout, one directory per run stage:

    acts/
      index.jsonl            one record per (problem, layer)
      blocks/<problem>.npz   float32 [n_positions, d_model] + position metadata

``.npz`` rather than a single parquet because the stages are resumable and
interruptible: a per-problem file can be written atomically and a partial run
leaves every completed problem intact.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from ..logging_utils import read_jsonl


def digest_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


@dataclass(frozen=True)
class ActivationRecord:
    problem_id: str
    layer: int
    positions: tuple[int, ...]
    #: What each row is: ``chunk_boundary`` | ``tail`` | ``matched_position`` | ...
    kinds: tuple[str, ...]
    shape: tuple[int, int]
    sha256: str
    path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": f"{self.problem_id}|L{self.layer}",
            "problem_id": self.problem_id,
            "layer": self.layer,
            "positions": list(self.positions),
            "kinds": list(self.kinds),
            "shape": list(self.shape),
            "sha256": self.sha256,
            "path": self.path,
        }


class ActivationStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.blocks = self.root / "blocks"
        self.blocks.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.jsonl"
        self._index: dict[str, dict[str, Any]] = {
            r["id"]: r for r in read_jsonl(self.index_path)
        }

    # -- queries ----------------------------------------------------------

    def key(self, problem_id: str, layer: int) -> str:
        return f"{problem_id}|L{layer}"

    def has(self, problem_id: str, layer: int) -> bool:
        k = self.key(problem_id, layer)
        if k not in self._index:
            return False
        return (self.root / self._index[k]["path"]).exists()

    def __iter__(self) -> Iterator[dict[str, Any]]:
        yield from self._index.values()

    def __len__(self) -> int:
        return len(self._index)

    # -- io ---------------------------------------------------------------

    def put(
        self,
        problem_id: str,
        layer: int,
        array: np.ndarray,
        positions: list[int],
        kinds: list[str],
        extra: dict[str, Any] | None = None,
    ) -> ActivationRecord:
        arr = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
        if arr.ndim != 2:
            raise ValueError(f"expected [n, d], got shape {arr.shape}")
        if len(positions) != arr.shape[0] or len(kinds) != arr.shape[0]:
            raise ValueError(
                f"positions ({len(positions)}) and kinds ({len(kinds)}) must "
                f"both match the {arr.shape[0]} rows"
            )
        rel = f"blocks/{problem_id}__L{layer}.npz"
        out = self.root / rel
        # Written through a file handle, not a path: ``np.savez_compressed``
        # appends ".npz" to any path that does not already end in it, so a
        # ".npz.tmp" target would be written as ".npz.tmp.npz" and the atomic
        # rename would then fail on a missing file.
        tmp = out.with_name(out.name + ".tmp")
        with open(tmp, "wb") as fh:
            np.savez_compressed(
                fh,
                activations=arr,
                positions=np.asarray(positions, dtype=np.int32),
                kinds=np.asarray(kinds, dtype=object),
                meta=np.asarray([json.dumps(extra or {})], dtype=object),
            )
        tmp.replace(out)

        rec = ActivationRecord(
            problem_id=problem_id,
            layer=layer,
            positions=tuple(int(p) for p in positions),
            kinds=tuple(kinds),
            shape=(int(arr.shape[0]), int(arr.shape[1])),
            sha256=digest_array(arr),
            path=rel,
        )
        d = rec.to_dict()
        self._index[d["id"]] = d
        with open(self.index_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(d) + "\n")
        return rec

    def get(self, problem_id: str, layer: int, verify: bool = True) -> tuple[np.ndarray, dict]:
        k = self.key(problem_id, layer)
        if k not in self._index:
            raise KeyError(f"no activations stored for {k}")
        rec = self._index[k]
        with np.load(self.root / rec["path"], allow_pickle=True) as z:
            arr = z["activations"]
            meta = {
                "positions": z["positions"].tolist(),
                "kinds": [str(x) for x in z["kinds"].tolist()],
                **json.loads(str(z["meta"][0])),
            }
        if verify and digest_array(arr) != rec["sha256"]:
            raise ValueError(
                f"activation block for {k} fails its recorded SHA-256 - the "
                f"cache is corrupt, delete it and re-extract"
            )
        return arr, meta

    def get_rows(self, problem_id: str, layer: int, kind: str) -> np.ndarray:
        arr, meta = self.get(problem_id, layer)
        sel = [i for i, k in enumerate(meta["kinds"]) if k == kind]
        return arr[sel] if sel else np.empty((0, arr.shape[1]), dtype=np.float32)

    def summary(self) -> dict[str, Any]:
        layers: dict[int, int] = {}
        rows = 0
        for r in self._index.values():
            layers[r["layer"]] = layers.get(r["layer"], 0) + 1
            rows += r["shape"][0]
        return {
            "n_blocks": len(self._index),
            "n_vectors": rows,
            "blocks_per_layer": layers,
            "root": str(self.root),
        }
