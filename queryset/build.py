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

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frame.core.version import BenchmarkVersion, query_digest  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
QDIR = os.path.join(HERE, "queries")
OUT = os.path.join(HERE, "..", "data", "benchmark.jsonl")
IDENTITY = os.path.join(HERE, "queryset.json")

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
        # One exact answer per cell of the 2x2 condition matrix
        # (frame.core.schema.CONDITIONS) — no cell is scored against a stand-in.
        "geometric_gt_filtered": None,      # vector_query + filter    (semantic+filter)
        "geometric_gt_nofilter": None,      # raw_query_text, no filter (raw+nofilter)
        "geometric_gt_raw_filtered": None,  # raw_query_text + filter   (raw+filter)
        "geometric_gt_vec_nofilter": None,  # vector_query, no filter   (semantic+nofilter)
        "_pending": [
            "target_keyframe_ids", "target_passes_filter", "filter_selectivity",
            "geometric_gt_filtered", "geometric_gt_nofilter",
            "geometric_gt_raw_filtered", "geometric_gt_vec_nofilter",
        ],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default=None,
                    help="override the corpus label from queryset.json (e.g. "
                         "'v3c1+2+3' for the union). The corpus is part of the "
                         "benchmark identity, so results built against different "
                         "corpora are kept incomparable — see frame/core/version.py.")
    # argv defaults to sys.argv[1:] for the CLI; callers (tests) pass an explicit
    # list so this never swallows an outer process's arguments.
    args = ap.parse_args(argv)

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
        items.append(item)

    # Carry ground truth forward. GT costs an HPC job, and it stays valid exactly
    # when the query it was computed for is unchanged — which is what the query
    # digest tells us. An edited query correctly loses its GT (re-run the oracle);
    # everything else survives a rebuild untouched.
    prior = read_items(OUT)
    carried = dropped = 0
    for item in items:
        old = prior.get(item["query_id"])
        if old and old.get("computed") and query_digest(old) == query_digest(item):
            item["computed"] = old["computed"]
            carried += 1
        else:
            if old and old.get("computed") and _has_gt(old):
                dropped += 1
            item["computed"] = computed_stub(item)

    identity = json.load(open(IDENTITY))
    corpus = args.corpus or identity["corpus"]
    # GT is stubbed here and filled later by the oracle, which rewrites this header
    # with the real gt_params. Carry forward whatever the previous build recorded so
    # a rebuild of unchanged queries does not look like a GT parameter change.
    previous = read_header(OUT)
    gt_params = previous.gt_params if previous else {}
    version = BenchmarkVersion.compute(
        version=identity["version"], corpus=corpus,
        items=items, gt_params=gt_params)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write(json.dumps(version.to_dict()) + "\n")
        for it in items:
            f.write(json.dumps(it) + "\n")

    n_verified = sum(1 for it in items if it["status"] == "verified")
    print(f"[done] {len(items)} items -> {os.path.relpath(OUT, os.path.join(HERE, '..'))}  "
          f"({n_verified} verified, {len(items) - n_verified} draft, {bad} rejected)")
    print(f"[version] {version.label}  ({version.n_with_gt}/{version.n_items} with GT)")
    if carried or dropped:
        print(f"[gt] carried forward for {carried} unchanged item(s)"
              + (f"; DROPPED for {dropped} edited item(s) — re-run oracle/build_gt.py"
                 if dropped else ""))
    if version.n_with_gt and not version.gt_params:
        print("[note] ground truth present but its parameters (oracle k, thresholds, "
              "encoder) are unrecorded — it predates versioning. The next "
              "oracle/build_gt.py run will record them.")

    # The drift catcher: a hand-set semver WILL be forgotten eventually, so say so
    # loudly rather than let two different query sets share one version number.
    if previous and previous.digest != version.digest \
            and previous.version == version.version:
        from frame.core.version import compare
        c = compare(previous, version)
        print(f"\n[WARN] contents changed but the version is still "
              f"{version.version} — bump 'version' in queryset/queryset.json")
        print(f"[WARN] what changed: {c.reason or 'ground truth / parameters'}")
        print(f"[WARN] guidance: {'MINOR' if c.status == 'additive' else 'MAJOR'} "
              f"(see queryset.json '_bumping')")

    sys.exit(1 if bad else 0)


def read_header(path):
    """The BenchmarkVersion on an existing benchmark.jsonl, or None."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        first = f.readline().strip()
    if not first:
        return None
    d = json.loads(first)
    return None if "query_id" in d else BenchmarkVersion.from_dict(d)


def read_items(path):
    """{query_id: item} from an existing benchmark.jsonl (header skipped)."""
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "query_id" in d:
                out[d["query_id"]] = d
    return out


def _has_gt(item):
    c = item.get("computed") or {}
    return any(c.get(k) is not None for k in
               ("target_keyframe_ids", "geometric_gt_filtered", "geometric_gt_nofilter"))


if __name__ == "__main__":
    main()
