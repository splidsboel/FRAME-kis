"""Unit tests for frame.core.dataset (the Tier-2 canonical shard handle).

Pure-logic only: path resolution, manifest parsing, validation. Actually reading
parquet/h5 needs pyarrow + h5py, which are ingest-side deps the harness core does
not carry — those paths are exercised on the HPC by scripts/load_dataset.py.
"""

from __future__ import annotations

import json

import pytest

from frame.core.dataset import TABLES, Dataset


def _shard(tmp_path, tables=("videos", "keyframes"), manifest=None, embeddings=True):
    d = tmp_path / "v3ctest"
    d.mkdir()
    for t in tables:
        (d / f"{t}.parquet").write_bytes(b"")
    if embeddings:
        (d / "keyframes_embeddings.h5").write_bytes(b"")
    if manifest is not None:
        (d / "MANIFEST.json").write_text(json.dumps(manifest))
    return d


def test_name_falls_back_to_folder_when_no_manifest(tmp_path):
    ds = Dataset(_shard(tmp_path))
    assert ds.name == "v3ctest"
    assert ds.manifest == {}


def test_name_and_counts_come_from_manifest(tmp_path):
    path = _shard(tmp_path, manifest={
        "dataset": "v3c1",
        "tables": {"videos": 7475, "keyframes": 1082566, "shots": None},
        "embeddings": {"count": 1082566},
    })
    ds = Dataset(path)
    assert ds.name == "v3c1"
    assert ds.expected_rows("videos") == 7475
    assert ds.expected_embeddings == 1082566
    # a table the manifest records as absent, and one it never mentions
    assert ds.expected_rows("shots") is None
    assert ds.expected_rows("scene_labels") is None


def test_corrupt_manifest_is_tolerated(tmp_path):
    """A broken manifest must not make the shard unusable — it only carries
    optimisation hints (row counts), never data."""
    path = _shard(tmp_path)
    (path / "MANIFEST.json").write_text("{not json")
    ds = Dataset(path)
    assert ds.manifest == {}
    assert ds.name == "v3ctest"
    ds.validate()


def test_tables_are_listed_in_fk_safe_order(tmp_path):
    present = ("object_detections", "keyframes", "videos", "shots")
    ds = Dataset(_shard(tmp_path, tables=present))
    listed = ds.tables()
    assert listed == [t for t in TABLES if t in present]
    # parents strictly before children
    assert listed.index("videos") < listed.index("shots") < listed.index("keyframes")
    assert listed.index("keyframes") < listed.index("object_detections")


def test_optional_tables_are_not_required(tmp_path):
    """The prep pipeline emits no object_detection_done; that must be fine."""
    ds = Dataset(_shard(tmp_path, tables=("videos", "keyframes", "scene_labels")))
    ds.validate()
    assert not ds.has_table("object_detection_done")
    assert "object_detection_done" not in ds.tables()


def test_validate_rejects_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        Dataset(tmp_path / "nope").validate()


def test_validate_rejects_missing_required_table(tmp_path):
    ds = Dataset(_shard(tmp_path, tables=("videos",)))
    with pytest.raises(FileNotFoundError, match="keyframes"):
        ds.validate()


def test_validate_rejects_shard_without_vectors(tmp_path):
    ds = Dataset(_shard(tmp_path, embeddings=False))
    with pytest.raises(FileNotFoundError, match="keyframes_embeddings"):
        ds.validate()
