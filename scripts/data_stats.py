#!/usr/bin/env python3
"""data_stats.py — corpus / metadata characterization of the V3C oracle DB.

The dataset-characterization statistics Omar asked for (meeting 2026-07-20): not
the query-driven selectivity (that lives in author_probe.py / frame.core.profile)
but descriptive statistics over the DATA and METADATA the filters are built on —
what the thesis Methods/characterization section reports about V3C.

Everything runs as GROUP BY aggregations against the in-job postgres (pure SQL,
psycopg2 only), using the SAME filter semantics as author_probe.py / build_gt.py:
  object filter = object_detections, conf >= OBJ_T (0.30)   [pinned fairness t]
  scene  filter = scene_labels,      conf >= SCN_T (0.10)

Blocks (each computed RAW = all detections and PINNED = above the fairness
threshold — Omar wanted both: raw characterizes the model output, pinned is the
universe the benchmark filters over):

  1. coverage        — corpus totals; keyframes with >=1 scene / object / OCR;
                       object_detection_done fraction (the object denominator).
  2. per-label dist  — keyframe count + selectivity per scene label (Places365)
                       and per object label, full vocab -> CSV. Reuses the
                       author_probe global_selectivity shape.
  3. per-keyframe     — histograms of #distinct scene labels/keyframe,
     histograms         #distinct object labels/keyframe, and #object INSTANCES/
                       keyframe (Omar's "max 8 objects; max 2-3 scenes").
  4. confidence       — width_bucket(0,1,20) histogram + percentiles of the
                       scene and object detection confidence scores.
  5. co-occurrence    — scene x object keyframe co-occurrence over the query-set
                       labels UNION the top-N most frequent labels (pinned): which
                       conjunctions actually exist, and how selective they are.
  6. video-metadata   — distribution of curated video-level categories / tags
                       (Omar's "extra metadata to filter on"), + per-video counts.

Writes data/data_stats.json (the machine-readable report) plus wide CSVs for the
full-vocab distributions and the co-occurrence matrix, and prints a summary.
Plot with scripts/plot_data_stats.py. Run inside the in-job postgres via
scripts/data_stats.sh.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import sys

import psycopg2

# scripts/ lives one level below the repo root.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUERIES = os.path.join(REPO, "queryset", "queries")
DATA = os.path.join(REPO, "data")

# Pinned fairness thresholds (CLAUDE.md / build_gt.py): scene 0.10, object 0.30.
THRESH = {"object": 0.30, "scene": 0.10}
TABLE = {"object": "object_detections", "scene": "scene_labels"}

# How many top-frequency labels (per kind) to fold into the co-occurrence matrix
# on top of the query-set's own labels.
DEFAULT_TOP_N = 30
CONF_BINS = 20  # width_bucket resolution for the confidence histograms


def connect():
    host = os.environ.get("PGHOST")
    if not host:
        sys.exit("PGHOST not set — run from data_stats.sh (in-job postgres socket).")
    conn = psycopg2.connect(host=host, port=5432, user="postgres", dbname="postgres")
    conn.autocommit = True
    return conn


def scalar(cur, sql, params=None):
    cur.execute(sql, params or ())
    return cur.fetchone()[0]


# ── query-set label inventory (drives the co-occurrence label sets) ──
def queryset_labels():
    """{'scene': {...}, 'object': {...}} — the label values used by authored filters."""
    labels = {"scene": set(), "object": set()}
    for path in sorted(glob.glob(os.path.join(QUERIES, "q*.json"))):
        d = json.load(open(path))
        for f in (d.get("decomposition") or {}).get("filters", []) or []:
            ft = f.get("filter_type")
            if ft in labels:
                v = f.get("value")
                labels[ft].update([v] if isinstance(v, str) else (v or []))
    return {k: sorted(v) for k, v in labels.items()}


# ── block 1: coverage ──
def coverage(cur):
    total = scalar(cur, "SELECT count(*) FROM keyframes")
    rep = {
        "keyframes": total,
        "videos": scalar(cur, "SELECT count(*) FROM videos"),
        "shots": scalar(cur, "SELECT count(*) FROM shots"),
        "object_detection_done": scalar(cur, "SELECT count(*) FROM object_detection_done"),
        "keyframes_with_ocr": scalar(
            cur, "SELECT count(DISTINCT keyframe_id) FROM keyframe_ocr"),
    }
    for kind, tbl in TABLE.items():
        t = THRESH[kind]
        rep[f"keyframes_with_{kind}_raw"] = scalar(
            cur, f"SELECT count(DISTINCT keyframe_id) FROM {tbl}")
        rep[f"keyframes_with_{kind}_pinned"] = scalar(
            cur, f"SELECT count(DISTINCT keyframe_id) FROM {tbl} WHERE confidence >= %s", (t,))
    return rep


# ── block 2: per-label distribution (full vocab) ──
def label_distribution(cur, kind, total, pinned):
    tbl = TABLE[kind]
    where = "WHERE confidence >= %s" if pinned else ""
    params = (THRESH[kind],) if pinned else ()
    cur.execute(
        f"SELECT label, count(DISTINCT keyframe_id) AS kf FROM {tbl} "
        f"{where} GROUP BY label ORDER BY kf DESC",
        params,
    )
    return [
        {"label": lbl, "kf": kf, "selectivity": kf / total if total else None}
        for lbl, kf in cur.fetchall()
    ]


# ── block 3: per-keyframe count histograms ──
def per_keyframe_hist(cur, inner_count, tbl, total, pinned, processed=None):
    """Histogram: #keyframes grouped by their `inner_count` value.

    inner_count is a SQL count expression over `tbl` grouped by keyframe_id
    (e.g. count(DISTINCT label) or count(*)). Keyframes with 0 rows never appear
    in `tbl`, so the 0-bucket is added explicitly from the corpus/processed total.
    """
    where = "WHERE confidence >= %s" if pinned else ""
    params = (THRESH[kind_of(tbl)],) if pinned else ()
    cur.execute(
        f"SELECT n, count(*) AS keyframes FROM "
        f"(SELECT keyframe_id, {inner_count} AS n FROM {tbl} {where} "
        f"GROUP BY keyframe_id) s GROUP BY n ORDER BY n",
        params,
    )
    hist = [{"n": n, "keyframes": kf} for n, kf in cur.fetchall()]
    denom = processed if processed is not None else total
    with_any = sum(r["keyframes"] for r in hist)
    zero = max(denom - with_any, 0)
    if zero:
        hist.insert(0, {"n": 0, "keyframes": zero})
    return {"histogram": hist, "denominator": denom, "max_n": hist[-1]["n"] if hist else 0}


def kind_of(tbl):
    return next(k for k, v in TABLE.items() if v == tbl)


# ── block 4: confidence histogram + percentiles ──
def confidence_stats(cur, kind):
    tbl = TABLE[kind]
    cur.execute(
        f"SELECT width_bucket(confidence, 0, 1, %s) AS bin, count(*) "
        f"FROM {tbl} GROUP BY bin ORDER BY bin",
        (CONF_BINS,),
    )
    hist = [{"bin": b, "lo": (b - 1) / CONF_BINS, "hi": b / CONF_BINS, "count": c}
            for b, c in cur.fetchall()]
    cur.execute(
        f"SELECT min(confidence), avg(confidence), max(confidence), "
        f"percentile_cont(0.5) WITHIN GROUP (ORDER BY confidence), "
        f"percentile_cont(0.9) WITHIN GROUP (ORDER BY confidence) FROM {tbl}"
    )
    mn, avg, mx, p50, p90 = cur.fetchone()
    return {
        "threshold": THRESH[kind],
        "histogram": hist,
        "min": _f(mn), "avg": _f(avg), "max": _f(mx),
        "p50": _f(p50), "p90": _f(p90),
    }


# ── block 5: scene x object co-occurrence (pinned) ──
def cooccurrence(cur, scene_labels, object_labels, total):
    """Keyframe co-occurrence count for every (scene,object) pair in the given
    label sets, at pinned thresholds. Pairs absent from the result have 0 co-occurrence."""
    cur.execute(
        "SELECT s.label AS scene, o.label AS object, count(*) AS kf FROM "
        "(SELECT DISTINCT keyframe_id, label FROM scene_labels "
        "  WHERE label = ANY(%s) AND confidence >= %s) s "
        "JOIN "
        "(SELECT DISTINCT keyframe_id, label FROM object_detections "
        "  WHERE label = ANY(%s) AND confidence >= %s) o "
        "USING (keyframe_id) "
        "GROUP BY s.label, o.label ORDER BY kf DESC",
        (scene_labels, THRESH["scene"], object_labels, THRESH["object"]),
    )
    pairs = [
        {"scene": s, "object": o, "kf": kf, "selectivity": kf / total if total else None}
        for s, o, kf in cur.fetchall()
    ]
    present = {(p["scene"], p["object"]) for p in pairs}
    n_possible = len(scene_labels) * len(object_labels)
    return {
        "scene_labels": scene_labels,
        "object_labels": object_labels,
        "pairs": pairs,
        "n_present": len(present),
        "n_possible": n_possible,
        "n_empty": n_possible - len(present),
    }


# ── block 6: video-level metadata ──
def video_metadata(cur):
    rep = {}
    for col in ("categories", "tags"):
        cur.execute(
            f"SELECT val, count(*) AS n FROM videos, unnest({col}) AS val "
            f"GROUP BY val ORDER BY n DESC"
        )
        rep[col] = [{"value": v, "videos": n} for v, n in cur.fetchall()]
        cur.execute(
            f"SELECT coalesce(array_length({col}, 1), 0) AS n, count(*) "
            f"FROM videos GROUP BY n ORDER BY n"
        )
        rep[f"{col}_per_video"] = [{"n": n, "videos": v} for n, v in cur.fetchall()]
    return rep


def _f(x):
    return float(x) if x is not None else None


# ── top-N by frequency, for the co-occurrence label sets ──
def top_labels(dist, n):
    return [r["label"] for r in dist[:n]]


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=DEFAULT_TOP_N,
                    help="top-frequency labels per kind folded into co-occurrence")
    args = ap.parse_args()

    os.makedirs(DATA, exist_ok=True)
    conn = connect()
    cur = conn.cursor()

    cov = coverage(cur)
    total = cov["keyframes"]
    processed = cov["object_detection_done"]
    qs = queryset_labels()
    print(f"[info] corpus={total} keyframes, {cov['videos']} videos; "
          f"object_detection_done={processed} ({processed/total*100:.1f}%); "
          f"thresholds={THRESH}; query-set labels: "
          f"{len(qs['scene'])} scene / {len(qs['object'])} object")

    report = {
        "corpus": total,
        "thresholds": THRESH,
        "coverage": cov,
        "queryset_labels": qs,
        "distributions": {},
        "per_keyframe": {},
        "confidence": {},
        "video_metadata": video_metadata(cur),
    }

    # block 2 — per-label distributions (raw + pinned), full vocab -> CSV
    dist_pinned = {}
    for kind in ("scene", "object"):
        for mode, pinned in (("raw", False), ("pinned", True)):
            d = label_distribution(cur, kind, total, pinned)
            report["distributions"][f"{kind}_{mode}"] = {"n_labels": len(d), "labels": d}
            write_csv(os.path.join(DATA, f"data_stats.dist.{kind}.{mode}.csv"),
                      d, ["label", "kf", "selectivity"])
            if pinned:
                dist_pinned[kind] = d
        print(f"[dist] {kind}: {report['distributions'][f'{kind}_raw']['n_labels']} labels "
              f"(raw) / {report['distributions'][f'{kind}_pinned']['n_labels']} (pinned)")

    # block 3 — per-keyframe histograms (pinned = the passing universe)
    report["per_keyframe"]["scene_labels_per_kf"] = per_keyframe_hist(
        cur, "count(DISTINCT label)", TABLE["scene"], total, pinned=True)
    report["per_keyframe"]["object_labels_per_kf"] = per_keyframe_hist(
        cur, "count(DISTINCT label)", TABLE["object"], total, pinned=True, processed=processed)
    report["per_keyframe"]["object_instances_per_kf"] = per_keyframe_hist(
        cur, "count(*)", TABLE["object"], total, pinned=True, processed=processed)
    print(f"[hist] max scenes/kf={report['per_keyframe']['scene_labels_per_kf']['max_n']}, "
          f"max objects/kf={report['per_keyframe']['object_instances_per_kf']['max_n']} (pinned)")

    # block 4 — confidence distributions
    for kind in ("scene", "object"):
        report["confidence"][kind] = confidence_stats(cur, kind)

    # block 5 — co-occurrence (query-set labels UNION top-N frequent)
    scene_set = sorted(set(qs["scene"]) | set(top_labels(dist_pinned["scene"], args.top_n)))
    object_set = sorted(set(qs["object"]) | set(top_labels(dist_pinned["object"], args.top_n)))
    co = cooccurrence(cur, scene_set, object_set, total)
    report["cooccurrence"] = co
    write_csv(os.path.join(DATA, "data_stats.cooccurrence.csv"),
              co["pairs"], ["scene", "object", "kf", "selectivity"])
    print(f"[cooc] {len(scene_set)}x{len(object_set)} label grid: "
          f"{co['n_present']}/{co['n_possible']} pairs co-occur, {co['n_empty']} empty")

    out = os.path.join(DATA, "data_stats.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[done] wrote {os.path.relpath(out, REPO)} "
          f"(+ data_stats.dist.*.csv, data_stats.cooccurrence.csv)")


if __name__ == "__main__":
    main()
