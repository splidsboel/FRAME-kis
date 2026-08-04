"""Shared fixtures + fakes for the FRAME unit tests.

The suite covers the pure-logic harness core (schema, analyzer, runner, profiler,
adapter contract, caching encoder, queryset build validation). Anything that needs
a live system (pgvector adapter → Postgres, SiglipEncoder → torch, oracle build_gt
→ HPC/GPU) is deliberately out of scope here; those are integration concerns, not
unit-testable without external infrastructure.
"""

from __future__ import annotations

import copy
from typing import Sequence

import numpy as np
import pytest

from frame.core.adapter import VectorDBAdapter
from frame.core.schema import Predicate


# ─── sample authored / GT-enriched items ────────────────────────────────────

@pytest.fixture
def authored_item() -> dict:
    """A minimal but complete authored item, as it lives in queryset/queries/*.json
    (no `computed` block yet — that is stubbed by build.py, filled by the oracle)."""
    return {
        "query_id": "q0001",
        "status": "verified",
        "source": {"origin": "vbs2019", "team": "test"},
        "raw_query_text": "a red car driving on a highway at night",
        "decomposition": {
            "vector_query": "a red car driving on a highway",
            "filters": [
                {
                    "filter_type": "scene",
                    "attribute": "scene_label",
                    "op": "in",
                    "value": ["night"],
                    "vocab": "places365",
                    "verified": True,
                }
            ],
        },
        "target": {"video_id": "00123", "start_s": 10.0, "end_s": 12.0},
        "notes": "a note",
    }


@pytest.fixture
def enriched_item(authored_item) -> dict:
    """The same item after oracle GT enrichment — a scorable item (filtered GT
    present AND the target survives its own filter)."""
    item = copy.deepcopy(authored_item)
    item["computed"] = {
        "target_keyframe_ids": ["kf_target"],
        "target_passes_filter": True,
        "filter_selectivity": [0.1],
        "geometric_gt_filtered": ["kf_target", "kf_a", "kf_b", "kf_c"],
        "geometric_gt_nofilter": ["kf_x", "kf_y", "kf_z"],
        "geometric_gt_vec_nofilter": ["kf_target", "kf_p", "kf_q", "kf_r"],
    }
    return item


# ─── fakes for the Runner / Profiler collaborators ──────────────────────────

class FakeEncoder:
    """Deterministic stand-in for the shared text encoder. Records every text it
    was asked to encode; returns a stable unit vector per distinct text."""

    def __init__(self, dim: int = 4):
        self.dim = dim
        self.calls: list[str] = []

    def encode(self, text: str) -> np.ndarray:
        self.calls.append(text)
        h = abs(hash(text)) % 997
        v = np.full(self.dim, float(h) + 1.0, dtype=np.float32)
        return v / np.linalg.norm(v)


class FakeAdapter(VectorDBAdapter):
    """Minimal concrete adapter: no DB, returns canned ranked ids and records the
    (filters, k) it was called with. Distinguishes filtered vs no-filter calls."""

    name = "fake"

    def __init__(self, filtered_ids=None, unfiltered_ids=None):
        self._filtered = filtered_ids or ["kf_target", "kf_a", "kf_z"]
        self._unfiltered = unfiltered_ids or ["kf_x", "kf_target", "kf_y"]
        self.setup_calls = 0
        self.teardown_calls = 0
        self.load_calls: list = []
        self.search_calls: list[tuple[int, int]] = []  # (n_filters, k)

    def load_data(self, dataset) -> None:
        self.load_calls.append(dataset)

    def setup(self) -> None:
        self.setup_calls += 1

    def teardown(self) -> None:
        self.teardown_calls += 1

    def search(self, query_vector, filters: Sequence[Predicate], k: int) -> list[str]:
        self.search_calls.append((len(filters), k))
        return list(self._filtered if filters else self._unfiltered)


@pytest.fixture
def fake_encoder() -> FakeEncoder:
    return FakeEncoder()


@pytest.fixture
def fake_adapter() -> FakeAdapter:
    return FakeAdapter()
