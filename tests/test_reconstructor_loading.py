"""Where the reconstructor's backbone is placed when it loads.

Upstream's `NLACritic` loads its backbone with no `device_map` and then calls
`.to(device)`. That materialises the whole checkpoint in host memory first -
for the released AR, 21 quantised layers plus a bf16 embedding table and an
`lm_head` it discards immediately, about 4.9 GB - and this machine routinely
has 2-3 GB free, so the load fails before any reconstruction happens.

`_load_straight_to` gives `from_pretrained` a `device_map` for the duration of
the construction only. It changes where the weights are put, not what is
computed: the backbone, the LayerNorm removal, the trained value head and the
MSE convention stay upstream's.

These tests pin the two properties that make that safe - the patch is scoped
and reversible, and it does nothing at all on CPU - without needing the 11 GB
checkpoint.
"""

from __future__ import annotations

import logging

import pytest

from nlaast.nla.reconstructor import _load_straight_to

LOG = logging.getLogger("test")


class _FakeAuto:
    """Stands in for the vendored module's AutoModelForCausalLM."""

    calls: list[dict] = []

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.calls.append(dict(kwargs))
        return "model"


class _FakeUpstream:
    def __init__(self):
        self.AutoModelForCausalLM = _FakeAuto


@pytest.fixture(autouse=True)
def _reset():
    _FakeAuto.calls = []


def test_the_patch_adds_a_device_map_while_it_is_active(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    up = _FakeUpstream()
    with _load_straight_to(up, "cuda", LOG):
        up.AutoModelForCausalLM.from_pretrained("ckpt", torch_dtype="bf16")
    assert _FakeAuto.calls == [{"torch_dtype": "bf16", "device_map": {"": 0},
                                "low_cpu_mem_usage": True}]


def test_the_original_is_restored_afterwards(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    up = _FakeUpstream()
    with _load_straight_to(up, "cuda", LOG):
        pass
    assert up.AutoModelForCausalLM is _FakeAuto


def test_the_original_is_restored_even_if_loading_raises(monkeypatch):
    """A failed AR load must not leave the vendored module patched - the next
    attempt, or the verbaliser, would then load under a map it never asked for.
    """
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    up = _FakeUpstream()
    with pytest.raises(RuntimeError):
        with _load_straight_to(up, "cuda", LOG):
            raise RuntimeError("out of memory")
    assert up.AutoModelForCausalLM is _FakeAuto


def test_an_explicit_device_map_from_upstream_is_not_overridden(monkeypatch):
    """setdefault, not assignment: if upstream ever starts passing its own
    placement, that wins."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    up = _FakeUpstream()
    with _load_straight_to(up, "cuda", LOG):
        up.AutoModelForCausalLM.from_pretrained("ckpt", device_map="auto")
    assert _FakeAuto.calls[0]["device_map"] == "auto"


def test_on_cpu_nothing_is_patched(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    up = _FakeUpstream()
    with _load_straight_to(up, "cpu", LOG):
        up.AutoModelForCausalLM.from_pretrained("ckpt")
    assert _FakeAuto.calls == [{}]


def test_without_cuda_nothing_is_patched(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    up = _FakeUpstream()
    with _load_straight_to(up, "cuda", LOG):
        up.AutoModelForCausalLM.from_pretrained("ckpt")
    assert _FakeAuto.calls == [{}]


def test_an_upstream_without_the_symbol_is_tolerated(monkeypatch):
    """The patch is best-effort: a vendored copy that imports differently must
    still load, just without the placement hint."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)

    class _Bare:
        pass

    bare = _Bare()
    with _load_straight_to(bare, "cuda", LOG):
        pass
    assert not hasattr(bare, "AutoModelForCausalLM")
