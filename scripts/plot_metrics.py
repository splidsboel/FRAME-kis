#!/usr/bin/env python3
"""plot_metrics.py — thesis figures for a benchmark run's scored metrics.

    python3 scripts/plot_metrics.py --in data/metrics.pgvector.jsonl --out results/figures

Standalone (json + matplotlib only, no `frame` import) so it runs anywhere with the
viz extra — locally after pulling data/metrics.<system>.jsonl, or in-job. Reads what
Analyzer.write_jsonl emits (a {system,ks,retrieval_k} header line, then one JSON
record per query) and writes figures (PDF + PNG each), light-mode / print, from the
same validated palette as plot_query_selectivity.py / plot_cutover.py.

  1. rank_distribution — BOXPLOT of the target's rank per condition, log scale, with
                         every query overlaid as a jittered point. The mean/median
                         rank gap in the 2x2 run is large enough that a single number
                         per condition misleads; the distribution is the honest view.
                         Targets never found are NOT silently dropped — they cannot
                         go on a rank axis, so each box is annotated with how many
                         of the scorable queries it actually contains.
  2. mrr_caps          — MRR at rank caps {1000,100,50,10}, filtered vs no-filter.
                         Recomputed here from the per-query ranks (the jsonl carries
                         uncapped rr only), so it matches Metrics.mrr_filtered(cap).
                         Caps at or beyond the run's retrieval depth are hatched:
                         no rank beyond k exists, so those bars are the uncapped MRR.
  3. recall_at_k       — mean Recall@k vs the oracle's exact k-NN, per condition.

Conditions are built in one place (`conditions()`), so when the Runner grows the
full 2x2 the figures follow by extending that function alone.
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

MRR_CAPS = (1000, 100, 50, 10)          # mirrors analyzer.DEFAULT_MRR_CAPS

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


def load_metrics(path):
    """-> (header dict, list of per-query rows)."""
    rows = [json.loads(l) for l in open(path) if l.strip()]
    header = rows[0] if rows and "query_id" not in rows[0] else {}
    return header, [r for r in rows if "query_id" in r]


def conditions():
    """The conditions to compare, in plot order.

    ONE place defines what a condition is: a label, its colour, and how to pull that
    condition's rank / recall out of a per-query row. The Runner currently produces
    the two diagonal cells of the 2x2 (semantic+filter, raw+no-filter); when it
    produces all four, add them here and every figure below picks them up.
    """
    return [
        ("filtered\n(semantic + predicate)", ORANGE,
         "target_rank_filtered", "recall_filtered"),
        ("no-filter\n(raw query)", BLUE,
         "target_rank_unfiltered", "recall_unfiltered"),
    ]


def scorable(rows):
    """Only items with self-consistent GT carry meaningful ranks — the Analyzer
    leaves rank=None on the rest, which is NOT the same as 'not found'."""
    return [r for r in rows if r.get("scorable")]


def _mrr(rows, rank_key, cap):
    """Mean reciprocal rank over scorable items, target beyond `cap` counting as a
    miss. Mirrors schema._reciprocal_rank so figures and printed summary agree."""
    if not rows:
        return 0.0
    total = 0.0
    for r in rows:
        rank = r.get(rank_key)
        if rank and (cap is None or rank <= cap):
            total += 1.0 / rank
    return total / len(rows)


# ─── 1. rank distribution ───────────────────────────────────────────────────

def plot_rank_distribution(rows, out, system):
    rows = scorable(rows)
    if not rows:
        print("  (no scorable items — skipping rank_distribution)")
        return

    conds = conditions()
    data, colors, labels = [], [], []
    for label, color, rank_key, _ in conds:
        ranks = [r[rank_key] for r in rows if r.get(rank_key)]
        misses = len(rows) - len(ranks)
        data.append(ranks or [np.nan])
        colors.append(color)
        # A box drawn over 12 of 15 queries is a different claim from one drawn over
        # all 15. Put the count in the tick label itself so it cannot be cropped or
        # read as belonging to the neighbouring box.
        note = f"n={len(ranks)}/{len(rows)}" + (f", {misses} not found" if misses else "")
        labels.append(f"{label}\n{note}")

    fig, ax = plt.subplots(figsize=(1.9 * len(conds) + 3.0, 5.0))
    x = np.arange(1, len(conds) + 1)
    bp = ax.boxplot(data, positions=x, widths=0.5, showfliers=False,
                    medianprops={"color": INK, "linewidth": 1.6},
                    whiskerprops={"color": INK2}, capprops={"color": INK2},
                    boxprops={"color": INK2}, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.30)

    # Every query as a point: with ~30 items the individual ranks are readable and
    # the boxplot alone would hide how few points shape each quartile.
    rng = np.random.default_rng(0)
    for xi, ranks, color in zip(x, data, colors):
        pts = [v for v in ranks if v == v]          # drop the NaN placeholder
        if not pts:
            continue
        ax.scatter(xi + rng.uniform(-0.13, 0.13, len(pts)), pts,
                   s=22, color=color, alpha=0.85, zorder=3,
                   edgecolors="white", linewidths=0.5)

    ax.set_yscale("log")
    ax.set_ylabel("rank of the target keyframe (log, lower is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(bottom=0.8)
    ax.grid(axis="x", visible=False)
    ax.axhline(1, color=GREEN, linewidth=0.9, linestyle=":", zorder=1)
    ax.annotate("rank 1", (0.02, 1), xycoords=("axes fraction", "data"),
                xytext=(0, 4), textcoords="offset points", fontsize=7, color=GREEN)
    ax.set_title(f"Target-rank distribution by condition — {system}")
    save(fig, out, "rank_distribution")


# ─── 2. MRR at rank caps ────────────────────────────────────────────────────

def plot_mrr_caps(rows, out, system, retrieval_k):
    rows = scorable(rows)
    if not rows:
        print("  (no scorable items — skipping mrr_caps)")
        return

    conds = conditions()
    caps = list(MRR_CAPS)
    x = np.arange(len(caps))
    width = 0.8 / len(conds)

    # Caps that cannot bite: nothing was retrieved past k, so MRR@cap there is just
    # the uncapped MRR and must not read as an independent measurement.
    inert = [bool(retrieval_k) and c >= retrieval_k for c in caps]

    fig, ax = plt.subplots(figsize=(1.4 * len(caps) + 3.0, 4.4))
    top = 0.0
    for i, (_label, color, rank_key, _) in enumerate(conds):
        vals = [_mrr(rows, rank_key, c) for c in caps]
        top = max(top, *vals)
        offs = x + (i - (len(conds) - 1) / 2) * width
        bars = ax.bar(offs, vals, width=width * 0.92, color=color)
        for bar, is_inert in zip(bars, inert):
            if is_inert:
                bar.set_hatch("//")
                bar.set_edgecolor("white")
        for xi, v in zip(offs, vals):
            ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=7, color=INK2)

    ax.set_xticks(x)
    ax.set_xticklabels([f"@{c}" for c in caps])
    ax.set_xlabel("rank cap (a target found deeper than the cap counts as a miss)")
    ax.set_ylabel("MRR")
    # Scale to the data, not to 1.0: KIS MRRs sit well under 0.5 and a fixed 0..1
    # axis flattens the differences between caps, which is the whole point here.
    ax.set_ylim(0, max(0.1, top * 1.25))
    ax.grid(axis="x", visible=False)

    # Legend built from plain swatches — letting bar containers supply the handles
    # copies the hatch of whichever bar came first and makes every condition look
    # inert.
    handles = [Patch(facecolor=c, label=lbl.replace("\n", " "))
               for lbl, c, _, _ in conds]
    if any(inert):
        handles.append(Patch(facecolor="white", edgecolor=MUTED, hatch="//",
                             label=f"cap ≥ retrieved k={retrieval_k} (= uncapped)"))
    ax.legend(handles=handles, frameon=False, loc="upper right", fontsize=8.5)
    ax.set_title(f"MRR at rank caps — {system}")
    save(fig, out, "mrr_caps")


# ─── 3. Recall@k ────────────────────────────────────────────────────────────

def plot_recall_at_k(rows, out, system, ks):
    rows = scorable(rows)
    if not rows or not ks:
        print("  (no scorable items / no ks — skipping recall_at_k)")
        return

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for label, color, _, recall_key in conditions():
        means = [float(np.mean([r[recall_key].get(str(k), 0.0) for r in rows]))
                 for k in ks]
        ax.plot(ks, means, marker="o", color=color, linewidth=1.8,
                markersize=5, label=label.replace("\n", " "))

    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("k")
    ax.set_ylabel("mean Recall@k vs the oracle's exact k-NN")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title(f"Geometric correctness — {system} ({len(rows)} scorable queries)")
    save(fig, out, "recall_at_k")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/metrics.pgvector.jsonl")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()

    header, rows = load_metrics(args.inp)
    system = header.get("system", "unknown")
    ks = header.get("ks") or []
    retrieval_k = header.get("retrieval_k") or 0
    os.makedirs(args.out, exist_ok=True)
    print(f"[plot] {args.inp} -> {args.out} "
          f"({len(rows)} queries, {len(scorable(rows))} scorable, system={system})")
    if not retrieval_k:
        print("  [note] no retrieval_k in the header (pre-2026-08 metrics file); "
              "MRR caps cannot be checked against the run's retrieval depth")

    plot_rank_distribution(rows, args.out, system)
    plot_mrr_caps(rows, args.out, system, retrieval_k)
    plot_recall_at_k(rows, args.out, system, ks)
    print("[done]")


if __name__ == "__main__":
    main()
