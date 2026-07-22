#!/usr/bin/env python3
"""author_probe.py — DB inputs for authoring disjunctive KIS queries.

Step 1 of the cutover work (see the thesis diary): the only realistic way a KIS
filter reaches the pgvector planner's exact->approximate cutover (~5-15% here) is
a DISJUNCTION — a describer unsure of the exact label hedges between candidates
and their UNION lands in that band while the target still passes via >=1 member.

This probe dumps the two inputs needed to author such hedges, straight from the
V3C oracle DB, using the SAME filter semantics as oracle/build_gt.py:
  object filter = object_detections, conf >= OBJ_T (0.30)   [pinned fairness t]
  scene  filter = scene_labels,      conf >= SCN_T (0.10)
  selectivity   = count(DISTINCT keyframe_id) / corpus_keyframes

Two modes:

  (default) SURVEY
    (1) global per-label selectivity for object & scene (full vocab, sorted).
    (2) per authored target: object/scene labels its keyframes carry, with the
        max confidence and whether that clears threshold — i.e. what the target
        can legitimately "pass" a filter on.
    Cross-reference (2)'s carried labels + real user text + (1)'s selectivities
    offline to design hedges whose union should sit ~5-15%.

  --unions FILE   VERIFY
    FILE is a JSON list of candidates
        [{"qid": "...", "filter_type": "object"|"scene", "labels": [...]}]
    For each, compute the EXACT union selectivity (distinct-keyframe / corpus at
    the type's threshold) and whether the target's keyframes pass — the check that
    closes out "Author + verify selectivity" before the step-2 sweep.

Targets come from queryset/queries/*.json (video_id/start_s/end_s), so the probe
is self-contained and does not need GT enrichment. Run inside the in-job postgres
via author_probe.sh.
"""

import glob
import json
import os
import sys

import psycopg2

# scripts/ lives one level below the repo root.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUERIES = os.path.join(REPO, "queryset", "queries")
OUT = os.path.join(REPO, "data", "author_probe.json")

# Pinned fairness thresholds (CLAUDE.md / build_gt.py): scene 0.10, object 0.30.
THRESH = {"object": 0.30, "scene": 0.10}
TABLE = {"object": "object_detections", "scene": "scene_labels"}


def connect():
    host = os.environ.get("PGHOST")
    if not host:
        sys.exit("PGHOST not set — run from author_probe.sh (in-job postgres socket).")
    conn = psycopg2.connect(host=host, port=5432, user="postgres", dbname="postgres")
    conn.autocommit = True
    return conn


def load_targets():
    """{qid: {video_id, start_s, end_s}} from the authored query set."""
    out = {}
    for path in sorted(glob.glob(os.path.join(QUERIES, "q*.json"))):
        d = json.load(open(path))
        t = d.get("target") or {}
        if t.get("video_id") is not None:
            out[d["query_id"]] = {
                "video_id": t["video_id"],
                "start_s": t.get("start_s", 0.0),
                "end_s": t.get("end_s", 0.0),
            }
    return out


def corpus_size(cur):
    cur.execute("SELECT count(*) FROM keyframes")
    return cur.fetchone()[0]


def target_keyframes(cur, video_id, start_s, end_s):
    """Keyframes whose shot overlaps the target window (build_gt A3 semantics)."""
    cur.execute(
        "SELECT k.keyframe_id FROM keyframes k JOIN shots s ON k.shot_id = s.shot_id "
        "WHERE k.video_id = %s AND s.start_time_s <= %s AND s.end_time_s >= %s",
        (video_id, end_s, start_s),
    )
    return [r[0] for r in cur.fetchall()]


def global_selectivity(cur, kind, total):
    """Full-vocab per-label selectivity for one filter kind at its pinned threshold."""
    tbl, t = TABLE[kind], THRESH[kind]
    cur.execute(
        f"SELECT label, count(DISTINCT keyframe_id) AS kf FROM {tbl} "
        "WHERE confidence >= %s GROUP BY label ORDER BY kf DESC",
        (t,),
    )
    return [
        {"label": lbl, "kf": kf, "sel": kf / total if total else None}
        for lbl, kf in cur.fetchall()
    ]


def target_labels(cur, kind, tkfs):
    """Labels the target's keyframes carry (max conf), flagged vs threshold.

    Shows sub-threshold detections too (they explain why a target fails / needs a
    relaxed hedge), sorted by confidence.
    """
    if not tkfs:
        return []
    tbl, t = TABLE[kind], THRESH[kind]
    cur.execute(
        f"SELECT label, max(confidence) AS c FROM {tbl} "
        "WHERE keyframe_id = ANY(%s) GROUP BY label ORDER BY c DESC",
        (tkfs,),
    )
    return [
        {"label": lbl, "max_conf": float(c), "passes": float(c) >= t}
        for lbl, c in cur.fetchall()
    ]


def union_selectivity(cur, kind, labels, total):
    tbl, t = TABLE[kind], THRESH[kind]
    cur.execute(
        f"SELECT count(DISTINCT keyframe_id) FROM {tbl} "
        "WHERE label = ANY(%s) AND confidence >= %s",
        (labels, t),
    )
    kf = cur.fetchone()[0]
    return kf, (kf / total if total else None)


def target_passes(cur, kind, labels, tkfs):
    if not tkfs:
        return False
    tbl, t = TABLE[kind], THRESH[kind]
    cur.execute(
        f"SELECT 1 FROM {tbl} WHERE keyframe_id = ANY(%s) AND label = ANY(%s) "
        "AND confidence >= %s LIMIT 1",
        (tkfs, labels, t),
    )
    return cur.fetchone() is not None


def run_survey(cur, total, targets):
    report = {"corpus": total, "thresholds": THRESH, "global": {}, "targets": {}}
    for kind in ("object", "scene"):
        g = global_selectivity(cur, kind, total)
        report["global"][kind] = g
        print(f"\n===== GLOBAL {kind.upper()} selectivity "
              f"(conf>={THRESH[kind]}, {len(g)} labels) — top 40 =====")
        for r in g[:40]:
            print(f"  {r['sel']*100:6.2f}%  {r['kf']:>7}  {r['label']}")

    for qid, t in targets.items():
        tkfs = target_keyframes(cur, t["video_id"], t["start_s"], t["end_s"])
        entry = {"target": t, "n_keyframes": len(tkfs), "object": [], "scene": []}
        for kind in ("object", "scene"):
            entry[kind] = target_labels(cur, kind, tkfs)
        report["targets"][qid] = entry
        warn = "  <-- 0 keyframes" if not tkfs else ""
        print(f"\n----- {qid}  video={t['video_id']} "
              f"[{t['start_s']}-{t['end_s']}s]  {len(tkfs)} kf{warn} -----")
        for kind in ("object", "scene"):
            passing = [r for r in entry[kind] if r["passes"]]
            shown = passing[:12] if passing else entry[kind][:6]
            tag = "" if passing else " (none clear threshold; showing top sub-threshold)"
            print(f"  {kind}{tag}:")
            for r in shown:
                mark = "*" if r["passes"] else " "
                print(f"    {mark} {r['max_conf']:.3f}  {r['label']}")
    return report


def run_unions(cur, total, targets, path):
    cands = json.load(open(path))
    report = {"corpus": total, "thresholds": THRESH, "candidates": []}
    print(f"\n===== UNION VERIFY ({len(cands)} candidates) =====")
    print(f"{'qid':8} {'kind':7} {'sel':>7}  {'pass':4}  labels")
    for c in cands:
        qid, kind, labels = c["qid"], c["filter_type"], c["labels"]
        kf, sel = union_selectivity(cur, kind, labels, total)
        tkfs = target_keyframes(cur, targets[qid]["video_id"],
                                targets[qid]["start_s"], targets[qid]["end_s"]) if qid in targets else []
        passes = target_passes(cur, kind, labels, tkfs)
        band = 0.05 <= (sel or 0) <= 0.15
        rec = {"qid": qid, "filter_type": kind, "labels": labels,
               "kf": kf, "selectivity": sel, "target_passes": passes,
               "in_band_5_15": band}
        report["candidates"].append(rec)
        flag = "OK" if (band and passes) else ("!band" if not band else "!pass")
        print(f"{qid:8} {kind:7} {sel*100:6.2f}%  {str(passes):5} {flag:6} {labels}")
    return report


def main():
    args = sys.argv[1:]
    unions_path = None
    if "--unions" in args:
        unions_path = args[args.index("--unions") + 1]

    conn = connect()
    cur = conn.cursor()
    total = corpus_size(cur)
    targets = load_targets()
    print(f"[info] corpus={total} keyframes; {len(targets)} authored targets; "
          f"thresholds={THRESH}")

    report = (run_unions(cur, total, targets, unions_path) if unions_path
              else run_survey(cur, total, targets))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out = OUT if not unions_path else OUT.replace(".json", ".unions.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[done] wrote {os.path.relpath(out, REPO)}")


if __name__ == "__main__":
    main()
