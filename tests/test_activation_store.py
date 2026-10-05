"""Activation storage.

Objective O2 requires activation-cache hashes in the reproducibility record,
so the store is content-addressed and a corrupt block must be detected rather
than silently used.
"""

import numpy as np
import pytest

from nlaast.activations.store import ActivationStore, digest_array


@pytest.fixture
def store(tmp_path):
    return ActivationStore(tmp_path / "acts")


def block(n=4, d=8, seed=0):
    return np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)


class TestRoundTrip:
    def test_put_then_get(self, store):
        arr = block()
        store.put("p1", 20, arr, positions=[1, 2, 3, 4],
                  kinds=["chunk_boundary"] * 4, extra={"chunk_indices": [0, 1, 2, 3]})
        got, meta = store.get("p1", 20)
        np.testing.assert_allclose(got, arr)
        assert meta["positions"] == [1, 2, 3, 4]
        assert meta["chunk_indices"] == [0, 1, 2, 3]

    def test_has(self, store):
        assert not store.has("p1", 20)
        store.put("p1", 20, block(), [0, 1, 2, 3], ["chunk_boundary"] * 4)
        assert store.has("p1", 20)
        assert not store.has("p1", 14)

    def test_layers_are_separate(self, store):
        a, b = block(seed=1), block(seed=2)
        store.put("p1", 20, a, [0, 1, 2, 3], ["chunk_boundary"] * 4)
        store.put("p1", 14, b, [0, 1, 2, 3], ["chunk_boundary"] * 4)
        np.testing.assert_allclose(store.get("p1", 20)[0], a)
        np.testing.assert_allclose(store.get("p1", 14)[0], b)

    def test_missing_key_raises(self, store):
        with pytest.raises(KeyError):
            store.get("nope", 20)

    def test_get_rows_by_kind(self, store):
        store.put("p1", 20, block(), [0, 1, 2, 3],
                  ["tail", "tail", "matched_position", "matched_length"])
        assert store.get_rows("p1", 20, "tail").shape[0] == 2
        assert store.get_rows("p1", 20, "absent").shape == (0, 8)


class TestValidation:
    def test_rejects_wrong_dimensionality(self, store):
        with pytest.raises(ValueError):
            store.put("p1", 20, np.zeros(8, dtype=np.float32), [0], ["x"])

    def test_rejects_mismatched_metadata(self, store):
        with pytest.raises(ValueError):
            store.put("p1", 20, block(), positions=[0, 1], kinds=["x"] * 4)

    def test_detects_corruption(self, store, tmp_path):
        arr = block()
        rec = store.put("p1", 20, arr, [0, 1, 2, 3], ["chunk_boundary"] * 4)
        path = store.root / rec.path
        tampered = arr.copy()
        tampered[0, 0] += 1.0
        np.savez_compressed(
            path, activations=tampered,
            positions=np.asarray([0, 1, 2, 3], dtype=np.int32),
            kinds=np.asarray(["chunk_boundary"] * 4, dtype=object),
            meta=np.asarray(["{}"], dtype=object),
        )
        with pytest.raises(ValueError, match="SHA-256"):
            store.get("p1", 20)
        # The caller can still opt out when it only wants the bytes.
        assert store.get("p1", 20, verify=False)[0].shape == (4, 8)


class TestPersistence:
    def test_index_survives_reopen(self, tmp_path):
        s1 = ActivationStore(tmp_path / "acts")
        s1.put("p1", 20, block(), [0, 1, 2, 3], ["chunk_boundary"] * 4)
        s2 = ActivationStore(tmp_path / "acts")
        assert s2.has("p1", 20)
        assert len(s2) == 1

    def test_summary(self, store):
        store.put("p1", 20, block(), [0, 1, 2, 3], ["chunk_boundary"] * 4)
        store.put("p2", 20, block(n=6), list(range(6)), ["chunk_boundary"] * 6)
        s = store.summary()
        assert s["n_blocks"] == 2 and s["n_vectors"] == 10
        assert s["blocks_per_layer"] == {20: 2}


class TestDigest:
    def test_stable_and_sensitive(self):
        a = block()
        assert digest_array(a) == digest_array(a.copy())
        b = a.copy()
        b[0, 0] += 1e-3
        assert digest_array(a) != digest_array(b)
