#!/usr/bin/env python3
"""
build_gt.py — enrich benchmark.jsonl with DB-computed ground truth. HPC ONLY:
needs the V3C postgres (pgvector) up and a GPU for the SigLIP text encoder.

Self-contained (does not depend on the old ~/jaeg experiment code, which is
retired). Reads/writes benchmark.jsonl next to this file.

────────────────────────────────────────────────────────────────────────────
GROUND TRUTH IS SYSTEM-AGNOSTIC
────────────────────────────────────────────────────────────────────────────
An item's filter is an ABSTRACT predicate (filter_type/attribute/op/value), not
SQL. This script uses postgres only as an ORACLE to compute the exact correct
answer to that predicate — it is NOT the pgvector-under-test filter. pgvector,
Chroma and Milvus are each graded against the SAME ground truth produced here,
and each realizes the abstract predicate in its own API (see predicate_to_sql()
for the oracle side; per-system runners do the translation). Where a system
cannot express a predicate, that gap is a result, not a bug.

────────────────────────────────────────────────────────────────────────────
ESTABLISHED  (confirmed from the loader scripts; re-check only if you re-embed
or re-load the DB):
 * Encoder = google/siglip-base-patch16-224; text = text_model(...).pooler_output,
   L2-normalised, 768-d, same space as the stored image vectors.
 * Distance = cosine (<=>); vectors are stored normalised.
 * Exact k-NN via SET enable_indexscan/bitmapscan = off (sequential scan).
 * scene_labels holds CANONICAL Places365 labels (church/indoor, mountain_snowy)
   only AFTER the places365 --reset re-run; before it the labels are the old
   collapsed form and filter values will not match.

DESIGN DECISIONS  (the benchmark's definitions, not facts to discover):
 * Scene filter  = multi-label set-containment on scene_labels      at confidence >= t_scene.
 * Object filter = multi-label set-containment on object_detections at confidence >= t_object.
   Same shape; independent thresholds (Places365 softmax vs YOLO-World confidence
   are not comparable). Multiple filters on an item are AND-ed (a scene+object
   conjunction is two filter entries).
 * target_passes_filter = at least one target keyframe satisfies the FULL
   (conjunctive) predicate, each filter at its own threshold.
 * Two run conditions per item: filtered (vector_query + filter) and
   no-filter (raw_query_text).

STILL TO VERIFY ON THE HPC  (the one real unknown):
 * A3 — target time-range -> keyframes via (k.shot_id = s.shot_id) OVERLAP.
   keyframes.shot_id was overwritten to equal keyframe_id, so this join is valid
   only if keyframe numbering aligns with shot indexing. The script WARNS on any
   target mapping to 0 keyframes — sanity-check the seed items before trusting GT.
 * That the chosen label strings exist: SELECT DISTINCT label FROM scene_labels.

THRESHOLDS are deliberately NOT hardcoded (one per side-table filter type).
  * default (no threshold flags): DIAGNOSTICS ONLY. Fills the t-independent
    fields (target keyframes, per-filter target best-match confidence, a
    selectivity-vs-threshold curve, unfiltered GT + target rank). Use these to
    CHOOSE each t empirically — look at where the target's real match confidence
    sits vs. where corpus selectivity explodes.
  * --scene-threshold / --object-threshold T: additionally fills the t-DEPENDENT
    fields (filtered GT, filtered target rank, target_passes_filter). An item's
    filtered block is computed only once EVERY filter type it uses has a threshold
    (so a scene+object item needs both flags).

Usage:
    python3 build_gt.py                          # diagnostics pass
    python3 build_gt.py --scene-threshold 0.10   # + filtered GT for scene items
    python3 build_gt.py --scene-threshold 0.10 --object-threshold 0.25  # + object items
    python3 build_gt.py --k 100 --limit 5
"""

import argparse
import json
import os
import sys
import warnings

import psycopg2
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "..", "data", "benchmark.jsonl")
MODEL_NAME = "google/siglip-base-patch16-224"

# Repo root on the path so `frame.core.version` imports when this runs as a script
# from a SLURM job (same pattern as scripts/prep_*.py).
sys.path.insert(0, os.path.dirname(HERE))

# thresholds sampled for the selectivity-vs-t diagnostic curve
CURVE_THRESHOLDS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# Filter types backed by a (keyframe_id, label, confidence) side table. Both scene
# and object filtering are the SAME shape — multi-label set-containment at a
# confidence floor — differing only in the source table (Places365 softmax scores
# vs YOLO-World detection confidences) and thus in the threshold you choose. The
# alias is used verbatim in the EXISTS subquery. Table/alias come from this dict,
# never from item data, so they are safe to interpolate into SQL.
FILTER_SOURCES = {
    "scene":  ("scene_labels", "sl"),
    "object": ("object_detections", "od"),
}

# Text/pattern filtering is a DIFFERENT shape from the side-table label filters:
# a case-insensitive substring match over OCR spans, with NO confidence floor
# (keyframe_ocr is loaded at --min-conf 0.0 — every span kept, thresholded at
# query time; we don't threshold it here). It therefore bypasses the
# thresholds machinery entirely.
PATTERN_TYPE = "pattern-match"

# Video-level metadata filtering is a THIRD shape: set-overlap against a curated
# ARRAY column on the parent `videos` row (joined by video_id), with NO confidence
# floor — V3C ships these per-video `categories`/`tags` as dataset facets, not model
# outputs, so there is nothing to threshold. Broad by nature (a category can span
# 1%→30% of the corpus), which is exactly why they exercise the planner's
# selectivity estimate. Column comes from this dict, never from item data.
VIDEO_META_SOURCES = {
    "video-category": "categories",
    "video-tag": "tags",
}


def _meta_values(f):
    """Normalize a video-metadata filter's `value` to a non-empty list of strings.
    A single string or a list is accepted; op must be set-overlap ('in'/'contains')."""
    if f.get("op") not in ("in", "contains"):
        raise ValueError(
            f"{f['filter_type']} filter op '{f.get('op')}' not supported (use 'in'/'contains')")
    val = f["value"]
    vals = [val] if isinstance(val, str) else list(val)
    vals = [v for v in vals if v]
    if not vals:
        raise ValueError(f"{f['filter_type']} filter has empty value")
    return vals


def video_meta_diagnostics(cur, f, target_kf_ids, total):
    """t-independent diagnostics for a video-level category/tag filter.

    Like OCR pattern-match (and unlike the side-table label filters) there is no
    confidence-threshold curve — set-overlap against a curated per-video array is a
    single selectivity number plus whether the target video carries the facet.
    """
    col = VIDEO_META_SOURCES[f["filter_type"]]
    vals = _meta_values(f)
    cur.execute(
        f"SELECT count(*) FROM keyframes k WHERE EXISTS "
        f"(SELECT 1 FROM videos v WHERE v.video_id = k.video_id AND v.{col} && %s::text[])",
        [vals])
    kept = cur.fetchone()[0]
    target_has_match = None
    if target_kf_ids:
        cur.execute(
            f"SELECT 1 FROM keyframes k JOIN videos v ON v.video_id = k.video_id "
            f"WHERE k.keyframe_id = ANY(%s) AND v.{col} && %s::text[] LIMIT 1",
            [target_kf_ids, vals])
        target_has_match = cur.fetchone() is not None
    return {
        "filter_type": f["filter_type"],
        "values": f["value"],
        "target_has_match": target_has_match,
        "selectivity": kept / total if total else None,
    }


def _pattern_likes(f):
    """Normalize a pattern-match filter's `value` to LIKE patterns (`%substr%`).

    `value` may be a single string or a list of strings; multiple values are
    OR-ed (match ANY). Patterns are lower-cased so the query can use
    `lower(text) LIKE …` and hit the keyframe_ocr_text_trgm GIN index.
    """
    if f.get("op") != "contains":
        raise ValueError(
            f"pattern-match filter op '{f.get('op')}' not supported (use 'contains')")
    val = f["value"]
    vals = [val] if isinstance(val, str) else list(val)
    vals = [v for v in vals if v]
    if not vals:
        raise ValueError("pattern-match filter has empty value")
    return ["%" + v.lower() + "%" for v in vals]


def ocr_pattern_diagnostics(cur, f, target_kf_ids, total):
    """t-independent diagnostics for an OCR pattern-match filter.

    Mirrors the side-table diagnostics (target match + corpus selectivity) but
    there is no confidence-threshold curve — OCR matching is a single ILIKE, so
    selectivity is one number, not a curve.
    """
    likes = _pattern_likes(f)
    ors = " OR ".join(["lower(text) LIKE %s"] * len(likes))
    cur.execute(f"SELECT count(DISTINCT keyframe_id) FROM keyframe_ocr WHERE {ors}", likes)
    kept = cur.fetchone()[0]
    target_has_match = None
    if target_kf_ids:
        cur.execute(
            f"SELECT 1 FROM keyframe_ocr WHERE keyframe_id = ANY(%s) AND ({ors}) LIMIT 1",
            [target_kf_ids] + likes)
        target_has_match = cur.fetchone() is not None
    return {
        "filter_type": PATTERN_TYPE,
        "patterns": f["value"],
        "target_has_match": target_has_match,
        "selectivity": kept / total if total else None,
    }


def filter_ready(f, thresholds):
    """Can this filter's t-dependent GT be computed now? Side-table filters need
    their confidence threshold supplied; pattern-match needs nothing."""
    ft = f["filter_type"]
    if ft in FILTER_SOURCES:
        return thresholds.get(ft) is not None
    if ft == PATTERN_TYPE or ft in VIDEO_META_SOURCES:
        return True
    return False


# ── DB ────────────────────────────────────────────────────────────────────────

def connect():
    host = os.environ.get("PGHOST")
    if not host:
        sys.exit("PGHOST not set — run from build_gt.sh (in-job postgres socket).")
    conn = psycopg2.connect(host=host, port=5432, user="postgres", dbname="postgres")
    conn.autocommit = True
    return conn


def corpus_size(cur):
    cur.execute("SELECT count(*) FROM keyframes")
    return cur.fetchone()[0]


def target_keyframes(cur, video_id, start_s, end_s):
    """Keyframes whose shot overlaps the target time range (A3)."""
    cur.execute(
        "SELECT k.keyframe_id "
        "FROM keyframes k JOIN shots s ON k.shot_id = s.shot_id "
        "WHERE k.video_id = %s AND s.start_time_s <= %s AND s.end_time_s >= %s",
        (video_id, end_s, start_s),
    )
    return [r[0] for r in cur.fetchall()]


def target_best_match(cur, table, target_kf_ids, labels):
    """Highest confidence at which any target keyframe carries a filter label (A4/A5).

    `table` is the filter's source table (scene_labels or object_detections); both
    share the (keyframe_id, label, confidence) shape. Returns
    {max_confidence, label, keyframe_id} or nulls if no match at any conf. Lets
    target_passes_filter(t) be derived later as (max_confidence >= t).
    """
    if not target_kf_ids or not labels:
        return {"max_confidence": None, "label": None, "keyframe_id": None}
    cur.execute(
        f"SELECT keyframe_id, label, confidence FROM {table} "
        "WHERE keyframe_id = ANY(%s) AND label = ANY(%s) "
        "ORDER BY confidence DESC LIMIT 1",
        (target_kf_ids, labels),
    )
    row = cur.fetchone()
    if not row:
        return {"max_confidence": 0.0, "label": None, "keyframe_id": None}
    return {"max_confidence": float(row[2]), "label": row[1], "keyframe_id": row[0]}


def selectivity_curve(cur, table, labels, total):
    """Corpus fraction kept by `label IN labels at conf >= t`, for each sampled t.

    `table` is the filter's source table (scene_labels or object_detections).
    """
    curve = {}
    for t in CURVE_THRESHOLDS:
        cur.execute(
            f"SELECT count(DISTINCT keyframe_id) FROM {table} "
            "WHERE label = ANY(%s) AND confidence >= %s",
            (labels, t),
        )
        curve[f"{t:.2f}"] = cur.fetchone()[0] / total if total else None
    return curve


def label_selectivity(cur, table, labels, total, t):
    """Corpus fraction kept by `label IN labels at conf >= t` at ONE threshold.

    The single-number summary behind `filter_selectivity` for a side-table filter
    (the curve above is the diagnostic; this is the value at the pinned threshold).
    Returns None when no threshold is set for this filter type yet.
    """
    if t is None:
        return None
    cur.execute(
        f"SELECT count(DISTINCT keyframe_id) FROM {table} "
        "WHERE label = ANY(%s) AND confidence >= %s",
        (labels, t),
    )
    return cur.fetchone()[0] / total if total else None


def target_passes(cur, target_kf_ids, filters, thresholds):
    """Does at least one target keyframe satisfy the FULL (conjunctive) predicate?

    Generalizes the old scene-only `max_confidence >= t` check to any mix of
    filter types (e.g. a scene+object conjunction): a target passes only if some
    target keyframe satisfies every filter at its own threshold.
    """
    if not target_kf_ids:
        return False
    where, wp = predicate_to_sql(filters, thresholds)
    cur.execute(
        f"SELECT 1 FROM keyframes k WHERE k.keyframe_id = ANY(%s) AND {where} LIMIT 1",
        [target_kf_ids] + wp,
    )
    return cur.fetchone() is not None


def predicate_to_sql(filters, thresholds):
    """Realize an ABSTRACT predicate list as an oracle WHERE fragment over keyframes `k`.

    This is the single seam where abstract filters (filter_type/attribute/op/value)
    become concrete SQL — and it is the ORACLE side only (computes the shared,
    system-agnostic ground truth). Each system-under-test has its OWN translator of
    the same abstract predicate. Returns (where_sql, params); filters are AND-ed
    (conjunction). Empty -> ("TRUE", []).

    `thresholds` is a {filter_type: confidence_floor} mapping (scene/object have
    independent floors, since Places365 softmax and YOLO-World confidences are not
    comparable). A side-table filter whose threshold is None raises — the caller
    must supply one (this is the diagnostics-vs-filtered-GT gate).

    Filter types:
      scene / object -> EXISTS over the label side-table at confidence >= t
      pattern-match  -> EXISTS over keyframe_ocr with a case-insensitive
                        substring match (no confidence floor)
    """
    thresholds = thresholds or {}
    clauses, params = [], []
    for f in filters or []:
        ft = f["filter_type"]
        if ft in FILTER_SOURCES:
            # both ops mean set-containment (the keyframe's label set contains any
            # of `value`) — scene items author it as "in", object items as
            # "contains"; the oracle SQL (label = ANY(...)) is identical either way.
            if f.get("op") not in ("in", "contains"):
                raise ValueError(f"{ft} filter op '{f.get('op')}' not supported yet")
            t = thresholds.get(ft)
            if t is None:
                raise ValueError(f"{ft} filter needs a confidence threshold; none provided")
            table, alias = FILTER_SOURCES[ft]
            clauses.append(
                f"EXISTS (SELECT 1 FROM {table} {alias} WHERE {alias}.keyframe_id = k.keyframe_id "
                f"AND {alias}.label = ANY(%s) AND {alias}.confidence >= %s)")
            params += [f["value"], t]
        elif ft == PATTERN_TYPE:
            likes = _pattern_likes(f)
            ors = " OR ".join(["lower(o.text) LIKE %s"] * len(likes))
            clauses.append(
                f"EXISTS (SELECT 1 FROM keyframe_ocr o WHERE o.keyframe_id = k.keyframe_id "
                f"AND ({ors}))")
            params += likes
        elif ft in VIDEO_META_SOURCES:
            # set-overlap against the parent video's curated array; no confidence floor.
            col = VIDEO_META_SOURCES[ft]
            clauses.append(
                f"EXISTS (SELECT 1 FROM videos v WHERE v.video_id = k.video_id "
                f"AND v.{col} && %s::text[])")
            params.append(_meta_values(f))
        else:
            raise ValueError(f"no oracle SQL for filter_type '{ft}' yet")
    return (" AND ".join(clauses) if clauses else "TRUE"), params


def brute_knn(cur, emb, k, filters=None, thresholds=None):
    """Exact (index-off) top-k keyframe ids satisfying the abstract predicate."""
    where, wp = predicate_to_sql(filters, thresholds)
    cur.execute("SET enable_indexscan = off")
    cur.execute("SET enable_bitmapscan = off")
    cur.execute(
        f"SELECT k.keyframe_id FROM keyframes k WHERE {where} "
        "ORDER BY k.embedding <=> %s LIMIT %s",
        wp + [emb, k],
    )
    ids = [r[0] for r in cur.fetchall()]
    cur.execute("RESET enable_indexscan")
    cur.execute("RESET enable_bitmapscan")
    return ids


def target_rank(cur, emb, target_kf_ids, filters=None, thresholds=None):
    """Rank (1-based) of the *closest* target keyframe in the (filtered) ordering.

    None if no target keyframe survives the filter. Task-success signal: 'at what
    rank does the answer first appear'. Uses the same abstract predicate as the GT.
    """
    if not target_kf_ids:
        return None
    where, wp = predicate_to_sql(filters, thresholds)
    cur.execute("SET enable_indexscan = off")
    cur.execute("SET enable_bitmapscan = off")
    best = None
    for kf in target_kf_ids:
        # distance of this target keyframe to the query
        cur.execute("SELECT k.embedding <=> %s FROM keyframes k WHERE k.keyframe_id = %s",
                    (emb, kf))
        row = cur.fetchone()
        if row is None:
            continue
        d = row[0]
        # the target keyframe must itself satisfy the filter to have a rank at all
        cur.execute(f"SELECT 1 FROM keyframes k WHERE k.keyframe_id = %s AND {where} LIMIT 1",
                    [kf] + wp)
        if cur.fetchone() is None:
            continue
        # count filter-passing keyframes strictly closer than the target keyframe
        cur.execute(f"SELECT count(*) FROM keyframes k WHERE (k.embedding <=> %s) < %s AND {where}",
                    [emb, d] + wp)
        rank = cur.fetchone()[0] + 1
        best = rank if best is None else min(best, rank)
    cur.execute("RESET enable_indexscan")
    cur.execute("RESET enable_bitmapscan")
    return best


# ── Encoder ─────────────────────────────────────────────────────────────────

def load_encoder():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        warnings.warn("CUDA not available — SigLIP on CPU will be slow")
    print(f"[info] loading {MODEL_NAME} on {device}")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    return processor, model, device


def embed_texts(processor, model, device, texts):
    """Return pgvector literal strings for a list of texts (matches image space).

    SigLIP MUST be tokenised with padding='max_length' (fixed 64 tokens) — that's
    how it was trained; padding=True degrades the text embedding and destroys
    text-image alignment (targets rank ~randomly). This is the SigLIP gotcha.
    """
    inputs = processor(text=texts, return_tensors="pt",
                       padding="max_length", max_length=64, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.text_model(input_ids=inputs["input_ids"],
                               attention_mask=inputs.get("attention_mask"))
        embs = F.normalize(out.pooler_output, dim=-1).cpu()
    return ["[" + ",".join(f"{v:.8f}" for v in e.tolist()) + "]" for e in embs]


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-threshold", type=float, default=None,
                    help="confidence floor for scene filters; if set, also compute "
                         "filtered GT + target_passes_filter (else diagnostics only)")
    ap.add_argument("--object-threshold", type=float, default=None,
                    help="confidence floor for object filters (YOLO-World detections); "
                         "analogue of --scene-threshold for object_detections")
    ap.add_argument("--k", type=int, default=100, help="k for geometric GT (default 100)")
    ap.add_argument("--limit", type=int, default=None, help="process only first N items")
    args = ap.parse_args()

    thresholds = {"scene": args.scene_threshold, "object": args.object_threshold}

    if not os.path.exists(BENCH):
        sys.exit(f"{BENCH} not found — run build.py first")
    rows = [json.loads(l) for l in open(BENCH) if l.strip()]
    # First line is the version header (build.py writes it); keep it so it can be
    # rewritten below with the GT parameters this run used.
    header = rows[0] if rows and "query_id" not in rows[0] else None
    items = rows[1:] if header else rows
    if args.limit:
        items = items[:args.limit]
        print(f"[warn] --limit {args.limit}: writing a PARTIAL benchmark file; the "
              f"version digest will reflect only these items")

    conn = connect()
    cur = conn.cursor()
    total = corpus_size(cur)
    tstr = ", ".join(f"{ft}={v}" for ft, v in thresholds.items() if v is not None) or "diagnostics-only"
    print(f"[info] corpus = {total} keyframes; k={args.k}; thresholds: {tstr}")

    processor, model, device = load_encoder()

    for i, item in enumerate(items):
        qid = item["query_id"]
        tgt = item["target"]
        vq = item["decomposition"]["vector_query"]
        rawq = item["raw_query_text"]
        filters = item["decomposition"].get("filters", [])

        emb_vec, emb_raw = embed_texts(processor, model, device, [vq, rawq])

        tkfs = target_keyframes(cur, tgt["video_id"], tgt["start_s"], tgt["end_s"])
        flag = "" if len(tkfs) else "  <-- WARNING: 0 target keyframes (A3)"
        print(f"[{i+1}/{len(items)}] {qid}: {len(tkfs)} target keyframes{flag}")

        c = item.setdefault("computed", {})
        c["target_keyframe_ids"] = tkfs
        c.pop("_pending", None)

        # per-filter diagnostics (t-independent) + the single-number summary
        # selectivity per filter (filter_selectivity, one entry per filter, in
        # filter order): pattern/video-meta are threshold-independent so it's
        # always the corpus fraction they keep; scene/object are evaluated at the
        # pinned threshold (None until that filter type's threshold flag is set).
        diags = []
        sels = []
        for f in filters:
            ft = f["filter_type"]
            if ft == PATTERN_TYPE:
                d = ocr_pattern_diagnostics(cur, f, tkfs, total)
                diags.append(d)
                sels.append(d["selectivity"])
                continue
            if ft in VIDEO_META_SOURCES:
                d = video_meta_diagnostics(cur, f, tkfs, total)
                diags.append(d)
                sels.append(d["selectivity"])
                continue
            if ft not in FILTER_SOURCES:
                diags.append({"filter_type": ft, "note": "no GT support yet"})
                sels.append(None)
                continue
            table = FILTER_SOURCES[ft][0]
            labels = f["value"]
            diags.append({
                "filter_type": ft,
                "labels": labels,
                "target_best_match": target_best_match(cur, table, tkfs, labels),
                "selectivity_vs_threshold": selectivity_curve(cur, table, labels, total),
            })
            sels.append(label_selectivity(cur, table, labels, total, thresholds.get(ft)))
        c["filter_diagnostics"] = diags
        c["filter_selectivity"] = sels

        # unfiltered GT (t-independent). Two no-filter baselines so the filter's
        # effect can be isolated: *_nofilter uses the raw query text (naive
        # baseline), *_vec_nofilter uses the decomposed vector_query (same query
        # as the filtered condition, minus the filter). Compare vec_nofilter ->
        # filtered to attribute a rank change to the FILTER alone; nofilter is the
        # separate "does decompose+filter beat naive raw search" baseline.
        c["geometric_gt_nofilter"] = brute_knn(cur, emb_raw, args.k)
        c["target_rank_nofilter"] = target_rank(cur, emb_raw, tkfs)
        c["geometric_gt_vec_nofilter"] = brute_knn(cur, emb_vec, args.k)
        c["target_rank_vec_nofilter"] = target_rank(cur, emb_vec, tkfs)

        # t-dependent block. Together with the two no-filter baselines above this
        # forms the full 2x2 {raw, vec} x {no-filter, filter}, so each factor is
        # attributable:
        #   *_nofilter        = raw query, no filter   (naive baseline)
        #   *_vec_nofilter    = vec query, no filter    (decomposition effect vs raw)
        #   *_filtered        = vec query, + filter     (filter effect vs vec_nofilter)
        #   *_raw_filtered    = raw query, + filter      (filter effect on the full text)
        ready = bool(filters) and all(filter_ready(f, thresholds) for f in filters)
        if ready:
            # record the threshold used for each side-table filter type this item
            # uses (pattern-match has no threshold, so it records none)
            for ft in {f["filter_type"] for f in filters}:
                if ft in FILTER_SOURCES:
                    c[f"{ft}_threshold"] = thresholds[ft]
            c["target_passes_filter"] = target_passes(cur, tkfs, filters, thresholds)
            c["geometric_gt_filtered"] = brute_knn(cur, emb_vec, args.k, filters, thresholds)
            c["target_rank_filtered"] = target_rank(cur, emb_vec, tkfs, filters, thresholds)
            c["geometric_gt_raw_filtered"] = brute_knn(cur, emb_raw, args.k, filters, thresholds)
            c["target_rank_raw_filtered"] = target_rank(cur, emb_raw, tkfs, filters, thresholds)

    # Refresh the version header: the GT just changed, and the parameters it was
    # computed under are part of what makes two runs comparable. Recording them here
    # means a later run against different thresholds or a different k is REFUSED by
    # the Analyzer rather than silently compared. See frame/core/version.py.
    if header is None:
        print("[warn] no version header on benchmark.jsonl (built before versioning) "
              "— re-run queryset/build.py to add one")
    else:
        from frame.core.version import BenchmarkVersion

        gt_params = {
            "oracle_k": args.k,
            "scene_threshold": args.scene_threshold,
            "object_threshold": args.object_threshold,
            "encoder": MODEL_NAME,
        }
        version = BenchmarkVersion.compute(
            version=header["version"], corpus=header["corpus"],
            items=items, gt_params=gt_params)
        header = version.to_dict()
        print(f"[version] {version.label}  "
              f"({version.n_with_gt}/{version.n_items} with GT)")

    with open(BENCH, "w") as f:
        if header is not None:
            f.write(json.dumps(header) + "\n")
        for it in items:
            f.write(json.dumps(it) + "\n")
    print(f"[done] wrote enriched GT to {os.path.relpath(BENCH, HERE)}")


if __name__ == "__main__":
    main()
