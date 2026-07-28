#!/usr/bin/env python3
"""plot_query_selectivity.py — thesis figures for the query-set filter selectivity.

    python3 scripts/plot_query_selectivity.py --in data/profile.pgvector.jsonl --out results/figures

Standalone (json + matplotlib only, no `frame` import) so it runs anywhere with the
viz extra — locally after pulling data/profile.pgvector.jsonl, or in-job from
profile_queryset.sh. Reads the selectivity profile profile.py writes (one JSON
record per filtered query; the first line is a {system,kind} header) and emits
figures (PDF + PNG each), light-mode / print, from the validated data-viz palette.

  1. query_selectivity   — per-query conjunction selectivity (all filtered queries,
                           sorted), coloured by the plan pgvector picks (exact
                           seqscan vs approximate HNSW). Where the workload lands.
  2. conjunction_parts   — for each multi-filter query, each filter part ALONE vs
                           the AND-ed conjunction, with the tightening factor
                           (Omar's "selektivitet på konjunktioner", meeting 2026-07-20).
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ── validated palette (dataviz skill, light surface) — matches plot_cutover.py ──
BLUE, ORANGE, GREEN, MAGENTA, GRAY = "#2a78d6", "#eb6834", "#008300", "#e87ba4", "#898781"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
PLAN_COLOR = {"seqscan": BLUE, "hnsw": ORANGE, "other": GRAY}
PLAN_LABEL = {"seqscan": "exact seqscan", "hnsw": "approximate HNSW", "other": "other"}
# filter parts vs the whole, in the conjunction figure
TYPE_COLOR = {"scene": BLUE, "object": ORANGE, "pattern-match": MAGENTA}

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
    "grid.linewidth": 0.6,
})


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  {name}.pdf / .png")


def load_profiles(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    # drop the {system, kind} header line
    return [r for r in rows if "query_id" in r]


def _part_type(summary: str) -> str:
    return summary.split(":", 1)[0]


def plot_query_selectivity(profs, out):
    rows = sorted(profs, key=lambda p: p["global_selectivity"])
    y = np.arange(len(rows))
    sel = [max(p["global_selectivity"] * 100, 1e-4) for p in rows]
    colors = [PLAN_COLOR.get(p["plan"], GRAY) for p in rows]
    fig, ax = plt.subplots(figsize=(7, max(4, len(rows) * 0.26)))
    ax.barh(y, sel, color=colors, height=0.78)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{p['query_id']}  {p['filter_summary'][:34]}" for p in rows],
                       fontsize=6.5)
    ax.set_xscale("log")
    ax.set_xlabel("conjunction selectivity (%, log)")
    ax.set_title(f"Per-query filter selectivity ({len(rows)} filtered queries), by plan")
    ax.grid(axis="y", visible=False)
    plans = [pl for pl in ("seqscan", "hnsw", "other") if any(p["plan"] == pl for p in rows)]
    ax.legend(handles=[Patch(facecolor=PLAN_COLOR[pl], label=PLAN_LABEL[pl]) for pl in plans],
              frameon=False, loc="lower right")
    save(fig, out, "query_selectivity")


def plot_conjunction_parts(profs, out):
    conj = [p for p in profs if p["n_filters"] > 1]
    if not conj:
        print("  (no multi-filter conjunction queries — skipping conjunction_parts)")
        return
    fig, axes = plt.subplots(1, len(conj), figsize=(4.2 * len(conj), 4.2), squeeze=False)
    for ax, p in zip(axes[0], conj):
        labels = [f"{f['summary'][:22]}\n(part)" for f in p["per_filter"]] + ["AND\n(conjunction)"]
        vals = [f["selectivity"] * 100 for f in p["per_filter"]] + [p["global_selectivity"] * 100]
        colors = [TYPE_COLOR.get(_part_type(f["summary"]), GRAY) for f in p["per_filter"]] + [GREEN]
        x = np.arange(len(vals))
        ax.bar(x, vals, color=colors, width=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("selectivity (%)")
        tight = f"{p['tightening']:.1f}x" if p.get("tightening") is not None else "n/a"
        ax.set_title(f"{p['query_id']} — conjunction tightens {tight}", fontsize=10)
        ax.grid(axis="x", visible=False)
        for xi, v in zip(x, vals):
            ax.annotate(f"{v:.2f}%", (xi, v), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=7, color=INK2)
    fig.suptitle("Conjunction queries: each filter part alone vs the AND-ed whole (pinned)")
    fig.tight_layout()
    save(fig, out, "conjunction_parts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/profile.pgvector.jsonl")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()

    profs = load_profiles(args.inp)
    os.makedirs(args.out, exist_ok=True)
    print(f"[plot] {args.inp} -> {args.out} ({len(profs)} filtered queries)")
    plot_query_selectivity(profs, args.out)
    plot_conjunction_parts(profs, args.out)
    print("[done]")


if __name__ == "__main__":
    main()
