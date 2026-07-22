"""Unit tests for the frame.core.adapter contract (the ABC's shared behaviour)."""

from __future__ import annotations

import numpy as np
import pytest

from frame.core.adapter import VectorDBAdapter


def test_cannot_instantiate_abstract_base():
    with pytest.raises(TypeError):
        VectorDBAdapter()  # type: ignore[abstract]


def test_context_manager_calls_setup_and_teardown(fake_adapter):
    with fake_adapter as a:
        assert a is fake_adapter
        assert fake_adapter.setup_calls == 1
        assert fake_adapter.teardown_calls == 0
    assert fake_adapter.teardown_calls == 1


def test_default_teardown_is_a_noop():
    class Bare(VectorDBAdapter):
        name = "bare"

        def setup(self) -> None: ...

        def search(self, query_vector, filters, k):
            return []

    # teardown not overridden — should not raise
    Bare().teardown()


def test_search_receives_filters_and_k(fake_adapter):
    ids = fake_adapter.search(np.zeros(4, dtype=np.float32), [], k=3)
    assert ids == ["kf_x", "kf_target", "kf_y"]
    assert fake_adapter.search_calls[-1] == (0, 3)
