#!/usr/bin/env python3
"""plot_task_success.py — Experiment C (task success) figures + tables.

    python3 scripts/plot_task_success.py --in data/metrics.pgvector.jsonl --out results/figures/pgvector

The 2x2 recall figures (plot_metrics.py) measure geometric correctness on ONE
canonical phrasing per task. This script measures TASK SUCCESS over the real human
phrasings each VBS team actually typed: for every graded task, the ~dozens of user
wordings scored against the no-filter top-k. Two facets, both Omar's:

  1. two-version MRR       — MRR over ALL phrasings vs the SUCCEEDING-only subset
                             (phrasings whose no-filter rank <= SUCCEEDING_CAP). The
                             split separates "the wording was too weak to find the
                             target at all" from a system/filter effect. Printed as a
                             table AND drawn as paired bars.
  2. per_task_mrr_box      — BOXPLOT of per-phrasing reciprocal rank, one box per
                             task, ordered by median. This is the honest view behind
                             the pooled MRR: a single pooled number hides that a few
                             well-worded tasks carry it while most sit near zero — the
                             recall<->MRR gap that is the headline.

Reuses the real Metrics/QueryMetrics/VariantMetric dataclasses (rebuilt from the
metrics jsonl) so every number matches Analyzer.variant_summary exactly.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from frame.core.schema import (
    Metrics, QueryMetrics, VariantMetric, SUCCEEDING_CAP, VARIANT_CONDS,
)

# ── palette (matches plot_metrics.py) ────────────────────────────────────────
BLUE, ORANGE, GREEN, MAGENTA, GRAY = "#2a78d6", "#eb6834", "#008300", "#e87ba4", "#898781"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

mpl.rcParams.update({
    "figure.dpi": 120,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
})

CAP = SUCCEEDING_CAP  # 100 — the depth-100 success gate


def graded_under(m: Metrics, cond: str):
    """Variants ACTUALLY graded under `cond`. A variant carries a rank key only for
    the conditions its item ran: an unfiltered item (no predicate) has no `filter`
    key, and must be excluded from the filter aggregate rather than counted as a
    filter miss (rr=0) — otherwise the 7 unfiltered tasks dilute every filter number
    and clutter the filter boxplot with all-zero boxes."""
    return [v for v in m.variant_rows() if cond in v.target_rank]


def rr(v, cond: str, cap: int | None = CAP) -> float:
    r = v.target_rank.get(cond)
    return 0.0 if (r is None or (cap is not None and r > cap)) else 1.0 / r


def mean(xs) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  {name}.pdf / .png")


def load_metrics(path: str) -> Metrics:
    """Rebuild a Metrics from what Analyzer.write_jsonl emitted: a header line, then
    one QueryMetrics record per query (variants embedded)."""
    with open(path) as f:
        lines = [json.loads(l) for l in f if l.strip()]
    header, rows = lines[0], lines[1:]
    per_query = []
    for r in rows:
        variants = []
        for v in r.get("variants", []):
            variants.append(VariantMetric(
                index=v["index"], team=v.get("team", ""), action=v.get("action", ""),
                text=v.get("text", ""), target_rank=v.get("target_rank", {}),
            ))
        per_query.append(QueryMetrics(
            query_id=r["query_id"], scorable=r.get("scorable", {}),
            recall={}, target_rank=r.get("target_rank", {}), latency_ms={},
            selectivity=r.get("selectivity"), harm_exemplar=r.get("harm_exemplar", False),
            variants=variants,
        ))
    return Metrics(system=header["system"], ks=header.get("ks", []),
                   per_query=per_query, retrieval_k=header.get("retrieval_k", 0))


def print_two_version(m: Metrics) -> dict:
    """Table: MRR@cap over ALL phrasings vs SUCCEEDING-only, per variant condition."""
    conds = [c for c in VARIANT_CONDS if graded_under(m, c)]
    n_tasks = sum(1 for q in m.per_query if q.variants)
    # The succeeding gate is the SAME phrasings for every cond (no-filter rank <= CAP),
    # but restricted to the ones graded under that cond.
    print(f"    {'cond':<10} | {'n phr':>6} | {'MRR all':>8} | {'n succ':>6} | {'MRR succ-only':>13}")
    print(f"    {'-'*10}-+-{'-'*6}-+-{'-'*8}-+-{'-'*6}-+-{'-'*13}")
    table = {}
    for c in conds:
        vs = graded_under(m, c)
        succ = [v for v in vs if v.succeeds(CAP)]
        all_mrr = mean([rr(v, c) for v in vs])
        succ_mrr = mean([rr(v, c) for v in succ])
        table[c] = {"n": len(vs), "n_succ": len(succ), "all": all_mrr, "succ": succ_mrr}
        print(f"    {c:<10} | {len(vs):>6} | {all_mrr:>8.3f} | {len(succ):>6} | {succ_mrr:>13.3f}")
    print(f"  ({m.system}: {n_tasks} tasks with variants)")
    return {"conds": conds, "n_tasks": n_tasks, "table": table}


def plot_two_version(m: Metrics, summary: dict, out: str):
    conds = summary["conds"]
    t = summary["table"]
    x = np.arange(len(conds))
    w = 0.38
    all_v = [t[c]["all"] for c in conds]
    succ_v = [t[c]["succ"] for c in conds]
    fig, ax = plt.subplots(figsize=(1.6 + 1.4 * len(conds), 4.2))
    ax.bar(x - w/2, all_v, w, color=GRAY, label="all phrasings")
    ax.bar(x + w/2, succ_v, w, color=BLUE, label="succeeding-only (no-filter rank ≤ cap)")
    for xi, (a, s) in enumerate(zip(all_v, succ_v)):
        ax.annotate(f"{a:.3f}", (xi - w/2, a), ha="center", va="bottom", fontsize=8, color=INK2)
        ax.annotate(f"{s:.3f}", (xi + w/2, s), ha="center", va="bottom", fontsize=8, color=BLUE)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\nn={t[c]['n']} (succ {t[c]['n_succ']})" for c in conds])
    ax.set_ylabel(f"MRR@{CAP} over human phrasings")
    ax.set_title(f"Task success — two-version MRR ({m.system})", color=INK)
    ax.set_ylim(0, max(all_v + succ_v) * 1.18)
    ax.legend(frameon=False, fontsize=9)
    save(fig, out, "task_success_mrr")


def plot_per_task_box(m: Metrics, cond: str, out: str):
    # id -> [rr per phrasing], only over tasks actually graded under this cond
    task_rr = {q.query_id: [rr(v, cond) for v in q.variants if cond in v.target_rank]
               for q in m.per_query}
    task_rr = {q: xs for q, xs in task_rr.items() if xs}
    if not task_rr:
        return
    # order tasks by median rr (ascending) so the long tail of ~0 tasks reads first
    order = sorted(task_rr, key=lambda q: (float(np.median(task_rr[q])), np.mean(task_rr[q])))
    data = [task_rr[q] for q in order]
    ncounts = [len(task_rr[q]) for q in order]
    fig, ax = plt.subplots(figsize=(max(7, 0.34 * len(order)), 4.6))
    ax.boxplot(data, positions=np.arange(len(order)), widths=0.6,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, linewidth=1.4),
                    boxprops=dict(facecolor="#dbe7f6", edgecolor=INK2, linewidth=0.8),
                    whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
    # jittered per-phrasing points
    rng = np.random.default_rng(0)
    for i, ys in enumerate(data):
        xs = i + (rng.random(len(ys)) - 0.5) * 0.4
        ax.scatter(xs, ys, s=8, color=BLUE, alpha=0.35, linewidths=0, zorder=3)
    pooled = mean([rr(v, cond) for v in graded_under(m, cond)])
    ax.axhline(pooled, color=ORANGE, linewidth=1.3, linestyle="--",
               label=f"pooled MRR@{CAP} = {pooled:.3f}")
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([f"{q}\n(n={c})" for q, c in zip(order, ncounts)],
                       rotation=90, fontsize=6.5)
    ax.set_ylabel(f"reciprocal rank ({cond}, cap {CAP})")
    ax.set_title(f"Per-task success over human phrasings — {m.system} ({cond})", color=INK)
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    save(fig, out, f"per_task_mrr_box_{cond}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()

    m = load_metrics(args.inp)
    if not m.has_variants():
        print(f"[skip] {args.inp}: no variant grading in this run")
        return
    print(f"[task-success] {args.inp} -> {args.out}")
    summary = print_two_version(m)
    plot_two_version(m, summary, args.out)
    for cond in summary["conds"]:
        plot_per_task_box(m, cond, args.out)
    print("[done]")


if __name__ == "__main__":
    main()
