"""Unit tests for frame.core.encode — the CachingEncoder wrapper.

Only the pure-Python caching layer is unit-tested here; SiglipEncoder needs
torch/transformers (the `encode` extra) and a model download, so it is out of
scope for the light CI unit suite.
"""

from __future__ import annotations

import numpy as np

from frame.core.encode import MAX_LENGTH, MODEL_NAME, CachingEncoder


class CountingEncoder:
    """Inner encoder that records how many times it actually computed."""

    def __init__(self):
        self.compute_count = 0

    def encode(self, text: str) -> np.ndarray:
        self.compute_count += 1
        return np.array([float(len(text))], dtype=np.float32)


def test_cache_returns_same_vector():
    inner = CountingEncoder()
    enc = CachingEncoder(inner)
    v1 = enc.encode("hello")
    v2 = enc.encode("hello")
    assert np.array_equal(v1, v2)


def test_cache_avoids_recomputation():
    inner = CountingEncoder()
    enc = CachingEncoder(inner)
    enc.encode("a")
    enc.encode("a")
    enc.encode("a")
    assert inner.compute_count == 1  # computed once, served from cache twice


def test_cache_distinguishes_texts():
    inner = CountingEncoder()
    enc = CachingEncoder(inner)
    enc.encode("a")
    enc.encode("bb")
    assert inner.compute_count == 2
    assert enc.encode("bb")[0] == 2.0


def test_fairness_constants_pinned():
    # these are the "do not change" fairness invariants from the module docstring
    assert MAX_LENGTH == 64
    assert MODEL_NAME == "google/siglip-base-patch16-224"
