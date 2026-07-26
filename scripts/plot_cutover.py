#!/usr/bin/env python3
"""
plot_cutover.py — thesis figures for the exact↔approximate cutover experiments.

    python3 scripts/plot_cutover.py --in <dir-with-jsonl> --out <figures-dir>

Standalone (json + matplotlib only, no `frame` import) so it runs anywhere with the
viz extra. Reads the two artifacts produced by the sweep + k-profile:
  * sweep.pgvector*.jsonl        (recall@k across ef_search × iterative_scan × plan)
  * profile_vs_k.pgvector*.jsonl (plan choice at k ∈ {50,100,250,1000})

Emits four figures (PDF + PNG each), light-mode / print, from the validated data-viz
palette: blue = exact seqscan, orange = approximate HNSW.
  1. plan_vs_depth      — heatmap: item (by selectivity) × k, coloured by plan
  2. recall_vs_selectivity — scatter: recall@100 vs selectivity, coloured by plan
  3. recall_vs_ef       — mean recall@100 vs ef_search, per iterative_scan mode
  4. cutover_vs_k       — HNSW-path item count vs retrieval depth k
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ── validated palette (dataviz skill, light surface) ──
BLUE, ORANGE, GREEN, MAGENTA, GRAY = "#2a78d6", "#eb6834", "#008300", "#e87ba4", "#898781"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
PLAN_COLOR = {"seqscan": BLUE, "hnsw": ORANGE, "other": GRAY}
PLAN_LABEL = {"seqscan": "exact seqscan", "hnsw": "approximate HNSW", "other": "other"}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "savefig.bbox": "tight",
})


def _read(pattern: str, indir: str) -> list[dict]:
    hits = sorted(glob.glob(os.path.join(indir, pattern)))
    if not hits:
        raise SystemExit(f"no artifact matching {pattern!r} in {indir}")
    with open(hits[-1]) as f:
        return [json.loads(l) for l in f if l.strip()][1:]  # drop header line


def _save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), dpi=200)
    plt.close(fig)
    print(f"  {name}.pdf / .png")


# ── figure 1: plan vs retrieval depth (heatmap staircase) ──
def fig_plan_vs_depth(pvk, outdir):
    ks = sorted({r["k"] for r in pvk})
    by_qid: dict[str, dict] = {}
    for r in pvk:
        by_qid.setdefault(r["query_id"], {})[r["k"]] = r["plan"]
    sel = {r["query_id"]: r["global_selectivity"] for r in pvk}
    qids = sorted(by_qid, key=lambda q: sel[q])           # low selectivity at bottom
    order = ["seqscan", "hnsw", "other"]
    code = {p: i for i, p in enumerate(order)}
    grid = [[code[by_qid[q][k]] for k in ks] for q in qids]

    fig, ax = plt.subplots(figsize=(5.6, 8.2))
    cmap = mcolors.ListedColormap([PLAN_COLOR[p] for p in order])
    ax.pcolormesh(grid, cmap=cmap, vmin=0, vmax=len(order),
                  edgecolors="white", linewidth=2)          # 2px surface gap
    ax.set_xticks([i + 0.5 for i in range(len(ks))], [str(k) for k in ks])
    ax.set_yticks([i + 0.5 for i in range(len(qids))],
                  [f"{q}  {sel[q]*100:4.1f}%" for q in qids], fontsize=7)
    ax.set_xlabel("retrieval depth  k")
    ax.set_title("Plan choice slides with retrieval depth\n"
                 "(each row an item, ordered by filter selectivity)", fontsize=11, pad=12)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    ax.legend(handles=[Patch(facecolor=PLAN_COLOR[p], label=PLAN_LABEL[p]) for p in order],
              loc="upper center", bbox_to_anchor=(0.5, -0.035), ncol=3, frameon=False,
              fontsize=9, handlelength=1.2)
    _save(fig, outdir, "cutover_plan_vs_depth")


# ── figure 2: recall@100 vs selectivity, coloured by plan (auto, relaxed, ef=1000) ──
def fig_recall_vs_selectivity(sweep, outdir, ef=1000):
    rows = [r for r in sweep if r["plan_mode"] == "auto"
            and r["iterative_scan"] == "relaxed_order" and r["ef_search"] == ef]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.axhline(1.0, color=GRID, lw=1, zorder=0)
    seen = set()
    marker = {"seqscan": "o", "hnsw": "s", "other": "D"}
    for r in rows:
        p = r["plan"]
        ax.scatter(r["global_selectivity"] * 100, r["recall"]["100"],
                   c=PLAN_COLOR[p], marker=marker[p], s=42, zorder=3,
                   edgecolors="white", linewidth=0.8,
                   label=PLAN_LABEL[p] if p not in seen else None)
        seen.add(p)
    # direct-label the worst-recall HNSW items (kept sparse to avoid label collisions)
    for r in rows:
        if r["plan"] == "hnsw" and r["recall"]["100"] < 0.90:
            ax.annotate(r["query_id"], (r["global_selectivity"] * 100, r["recall"]["100"]),
                        xytext=(6, -1), textcoords="offset points", fontsize=8, color=INK2)
    ax.set_xscale("log")
    ax.set_xlabel("filter selectivity  (% of corpus passing, log scale)")
    ax.set_ylabel("recall@100  vs exact filtered k-NN")
    ax.set_title(f"Even at max ef_search={ef}, selective filters on the HNSW path lose recall\n"
                 "(k=100, iterative_scan=relaxed_order)", fontsize=10.5)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    _save(fig, outdir, "cutover_recall_vs_selectivity")


# ── figure 3: mean recall@100 vs ef_search, per iterative_scan mode (HNSW items) ──
def fig_recall_vs_ef(sweep, outdir):
    modes = ["relaxed_order", "strict_order", "off"]
    style = {"relaxed_order": (BLUE, "-", "o"), "strict_order": (GREEN, "--", "^"),
             "off": (MAGENTA, ":", "s")}
    label = {"relaxed_order": "relaxed_order", "strict_order": "strict_order", "off": "off"}
    efs = sorted({r["ef_search"] for r in sweep})
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for m in modes:
        ys = []
        for ef in efs:
            vals = [r["recall"]["100"] for r in sweep if r["plan_mode"] == "auto"
                    and r["plan"] == "hnsw" and r["iterative_scan"] == m and r["ef_search"] == ef]
            ys.append(sum(vals) / len(vals) if vals else float("nan"))
        c, ls, mk = style[m]
        ax.plot(efs, ys, color=c, ls=ls, marker=mk, lw=2.2, ms=6, zorder=3)
    ax.set_xlabel("hnsw.ef_search")
    ax.set_ylabel("mean recall@100  (HNSW-path items)")
    ax.set_title("ef_search is a recall dial; iterative_scan=off truncates the result set\n"
                 "(k=100, plan=auto)", fontsize=10.5)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlim(efs[0] - 30, efs[-1] + 160)
    ax.legend(handles=[Line2D([0], [0], color=style[m][0], ls=style[m][1], marker=style[m][2],
                              lw=2.2, label=label[m]) for m in modes],
              loc="center right", frameon=False, fontsize=9)
    _save(fig, outdir, "cutover_recall_vs_ef")


# ── figure 4: HNSW-path item count vs retrieval depth k ──
def fig_cutover_vs_k(pvk, outdir):
    ks = sorted({r["k"] for r in pvk})
    n_total = len({r["query_id"] for r in pvk})
    counts = [sum(1 for r in pvk if r["k"] == k and r["plan"] == "hnsw") for k in ks]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(ks, counts, color=BLUE, marker="o", lw=2.2, ms=7, zorder=3)
    for k, c in zip(ks, counts):
        ax.annotate(str(c), (k, c), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.set_xlabel("retrieval depth  k  (log scale)")
    ax.set_ylabel(f"filters on the approximate\nHNSW path  (of {n_total})")
    ax.set_title("Shrinking k pulls more filters onto the approximate path", fontsize=11)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_ylim(0, n_total)
    _save(fig, outdir, "cutover_hnsw_count_vs_k")


def main():
    ap = argparse.ArgumentParser()
    HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--in", dest="indir", default=os.path.join(HERE, "data"))
    ap.add_argument("--out", dest="outdir", default=os.path.join(HERE, "data", "figures"))
    args = ap.parse_args()

    sweep = _read("sweep.pgvector*.jsonl", args.indir)
    pvk = _read("profile_vs_k.pgvector*.jsonl", args.indir)
    print(f"sweep rows: {len(sweep)}   profile_vs_k rows: {len(pvk)}")
    print(f"writing figures to {args.outdir}")
    fig_plan_vs_depth(pvk, args.outdir)
    fig_recall_vs_selectivity(sweep, args.outdir)
    fig_recall_vs_ef(sweep, args.outdir)
    fig_cutover_vs_k(pvk, args.outdir)


if __name__ == "__main__":
    main()
