"""Writing a converted checkpoint to disk, on a volume that is nearly full.

`save_pretrained` defaults `max_shard_size` to 50 GB in transformers 5.x, so a
7B model is written as a single file and safetensors preallocates it. On this
machine that failed with `os error 112` (disk full) against about 10 GB of free
space - after the 15 GB download and the quantisation pass had both already
been spent.

`_save_sharded` writes in 2 GB shards and, before the first byte, checks there
is room for the whole thing with headroom for the index and the quantisation
state bitsandbytes stores alongside each weight. A conversion that cannot
finish should say so immediately, naming the volume and the shortfall, rather
than fail halfway through.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from nlaast.models import loading  # noqa: E402


class _Model(torch.nn.Module):
    """Records what `_save_sharded` asks `save_pretrained` for."""

    def __init__(self, n_params: int = 1024):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(n_params, dtype=torch.float32))
        self.register_buffer("scale", torch.zeros(16, dtype=torch.float32))
        self.calls: list[dict] = []

    def save_pretrained(self, out, **kwargs):
        self.calls.append({"out": str(out), **kwargs})


def test_it_shards_rather_than_writing_one_preallocated_file(tmp_path):
    m = _Model()
    loading._save_sharded(m, tmp_path)
    assert m.calls[0]["max_shard_size"] == loading.SHARD_SIZE
    assert m.calls[0]["safe_serialization"] is True


def test_the_shard_size_is_well_under_a_whole_7b_checkpoint():
    """A 50 GB shard means "one file"; the point is that it does not."""
    assert loading.SHARD_SIZE.endswith("GB")
    assert int(loading.SHARD_SIZE.removesuffix("GB")) <= 5


def test_it_refuses_before_writing_when_there_is_no_room(tmp_path, monkeypatch):
    import shutil as _shutil

    m = _Model(n_params=1_000_000)

    class _Usage:
        total = 10**12
        used = 10**12
        free = 1024  # one kilobyte

    monkeypatch.setattr(_shutil, "disk_usage", lambda _p: _Usage)
    monkeypatch.setattr(loading.shutil, "disk_usage", lambda _p: _Usage)
    with pytest.raises(OSError) as err:
        loading._save_sharded(m, tmp_path)
    assert "not enough space" in str(err.value)
    assert str(tmp_path) in str(err.value)
    assert "NLAAST_QUANT_DIR" in str(err.value)
    # Nothing was attempted.
    assert m.calls == []


def test_the_budget_includes_headroom_above_the_raw_parameter_bytes(tmp_path,
                                                                    monkeypatch):
    """Index files and bitsandbytes' quantisation state are not free."""
    m = _Model(n_params=1_000_000)          # 4 MB of float32 parameters
    raw = sum(p.numel() * p.element_size() for p in m.parameters())
    raw += sum(b.numel() * b.element_size() for b in m.buffers())

    class _JustRaw:
        total = 10**12
        used = 0
        free = raw + 1  # enough for the weights alone, nothing more

    monkeypatch.setattr(loading.shutil, "disk_usage", lambda _p: _JustRaw)
    with pytest.raises(OSError):
        loading._save_sharded(m, tmp_path)

    class _WithHeadroom:
        total = 10**12
        used = 0
        free = int(raw * loading._SAVE_HEADROOM) + 1

    monkeypatch.setattr(loading.shutil, "disk_usage", lambda _p: _WithHeadroom)
    loading._save_sharded(m, tmp_path)
    assert m.calls


def test_a_roomy_volume_just_saves(tmp_path):
    m = _Model()
    loading._save_sharded(m, tmp_path)
    assert Path(m.calls[0]["out"]) == tmp_path
