#!/usr/bin/env python3
"""
build.py — compile the hand-authored query items into data/benchmark.jsonl.

Source of truth is queryset/queries/*.json (one authored item each). This
validates them and emits data/benchmark.jsonl for the eval harness. The DB-backed
`computed` fields (target keyframes, selectivity, exact k-NN, target_passes_filter)
need the V3C database on the HPC; they are stubbed null here and filled in place by
oracle/build_gt.py (enrich-in-place — GT lives inside each item's `computed` block).

Usage:  python3 queryset/build.py        (or: uv run python queryset/build.py)
Exit code is non-zero if any item is invalid or has a duplicate id.
"""

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
QDIR = os.path.join(HERE, "queries")
OUT = os.path.join(HERE, "..", "data", "benchmark.jsonl")

REQUIRED = ["query_id", "status", "source", "raw_query_text", "decomposition", "target"]


def validate(item):
    errs = []
    for k in REQUIRED:
        if k not in item:
            errs.append(f"missing '{k}'")
    d = item.get("decomposition", {})
    if not d.get("vector_query"):
        errs.append("decomposition.vector_query empty")
    for i, f in enumerate(d.get("filters", [])):
        for k in ("filter_type", "attribute", "op", "value"):
            if k not in f:
                errs.append(f"filters[{i}] missing '{k}'")
        if f.get("filter_type") == "scene" and not f.get("value"):
            errs.append(f"filters[{i}] scene filter has empty value list")
    t = item.get("target", {})
    for k in ("video_id", "start_s", "end_s"):
        if t.get(k) in (None, ""):
            errs.append(f"target.{k} empty")
    return errs


def computed_stub(item):
    # Field names mirror frame.core.schema.GroundTruth (the harness reads these).
    n = len(item["decomposition"].get("filters", []))
    return {
        "target_keyframe_ids": None,       # join shots/keyframes on target time range
        "target_passes_filter": None,      # target must satisfy the filter or item is broken
        "filter_selectivity": [None] * n,  # corpus fraction each filter keeps
        "geometric_gt_filtered": None,     # exact filtered k-NN keyframe ids
        "geometric_gt_nofilter": None,     # exact unfiltered k-NN (no-filter condition)
        "_pending": [
            "target_keyframe_ids", "target_passes_filter", "filter_selectivity",
            "geometric_gt_filtered", "geometric_gt_nofilter",
        ],
    }


def main():
    paths = sorted(
        p for p in glob.glob(os.path.join(QDIR, "*.json"))
        if not os.path.basename(p).startswith("_")
    )
    items, seen, bad = [], set(), 0
    for p in paths:
        name = os.path.basename(p)
        try:
            item = json.load(open(p))
        except json.JSONDecodeError as e:
            print(f"[INVALID] {name}: bad JSON — {e}")
            bad += 1
            continue
        errs = validate(item)
        if errs:
            print(f"[INVALID] {name}: {'; '.join(errs)}")
            bad += 1
            continue
        qid = item["query_id"]
        if qid in seen:
            print(f"[DUP] {name}: query_id '{qid}' already used")
            bad += 1
            continue
        seen.add(qid)
        item.setdefault("computed", computed_stub(item))
        items.append(item)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")

    n_verified = sum(1 for it in items if it["status"] == "verified")
    print(f"[done] {len(items)} items -> {os.path.relpath(OUT, os.path.join(HERE, '..'))}  "
          f"({n_verified} verified, {len(items) - n_verified} draft, {bad} rejected)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
