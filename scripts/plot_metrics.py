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
                         rank gap across conditions is large enough that a single
                         number per condition misleads; the distribution is the
                         honest view. Targets never found are NOT silently dropped —
                         they cannot go on a rank axis, so each box is annotated with
                         how many of the items it actually contains.
  2. mrr_caps          — MRR at rank caps {1000,100,50,10}, per condition.
                         Recomputed here from the per-query ranks (the jsonl carries
                         uncapped rr only), so it matches Metrics.mrr(cond, cap).
                         Caps at or beyond the run's retrieval depth are hatched:
                         no rank beyond k exists, so those bars are the uncapped MRR.
  3. recall_at_k       — mean Recall@k, each condition against ITS OWN exact k-NN.
  4. condition_grid    — the 2x2 read as a matrix, with both margins: the filter
                         delta and the raw-vs-semantic delta. Only drawn when all
                         four cells ran.

EVERY figure is computed over the items scorable in ALL of the run's conditions
(`comparable`). Averaging each condition over its own scorable set would compare
different query subsets — the no-filter cells are scorable for items that have no
predicate at all — and the difference would read as a condition effect.
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


CANONICAL_ORDER = ("raw+nofilter", "raw+filter", "semantic+nofilter", "semantic+filter")
COND_COLOR = {"raw+nofilter": BLUE, "raw+filter": ORANGE,
              "semantic+nofilter": GREEN, "semantic+filter": MAGENTA}
# Two lines so a four-condition axis stays readable.
COND_LABEL = {"raw+nofilter": "raw\nno-filter", "raw+filter": "raw\n+ filter",
              "semantic+nofilter": "semantic\nno-filter",
              "semantic+filter": "semantic\n+ filter"}


def load_metrics(path):
    """-> (header dict, list of per-query rows), normalising legacy files.

    Pre-2026-08-04 runs stored two flat conditions (`recall_filtered`,
    `target_rank_filtered`, ...). Those meant vector_query+predicate and
    raw_query_text alone, so they are re-keyed onto the two matching 2x2 cells
    rather than being unplottable — such a run simply has two of the four.
    """
    rows = [json.loads(l) for l in open(path) if l.strip()]
    header = rows[0] if rows and "query_id" not in rows[0] else {}
    rows = [r for r in rows if "query_id" in r]

    if rows and "recall" not in rows[0]:
        for r in rows:
            ok = bool(r.get("scorable"))
            r["scorable"] = {"semantic+filter": ok, "raw+nofilter": ok}
            r["recall"] = {"semantic+filter": r.pop("recall_filtered", {}),
                           "raw+nofilter": r.pop("recall_unfiltered", {})}
            r["target_rank"] = {"semantic+filter": r.pop("target_rank_filtered", None),
                                "raw+nofilter": r.pop("target_rank_unfiltered", None)}
            r["latency_ms"] = {"semantic+filter": r.pop("latency_filtered_ms", 0.0),
                               "raw+nofilter": r.pop("latency_unfiltered_ms", 0.0)}
        header.setdefault("conditions", ["raw+nofilter", "semantic+filter"])
        print("  [note] legacy two-condition metrics file — re-keyed onto the "
              "matching 2x2 cells")
    return header, rows


def conditions(header, rows):
    """Conditions present in this run, in canonical 2x2 order."""
    named = header.get("conditions")
    if not named:
        named = sorted({c for r in rows for c in (r.get("target_rank") or {})})
    return [c for c in CANONICAL_ORDER if c in named] + \
           [c for c in named if c not in CANONICAL_ORDER]


def comparable(rows, conds):
    """Items scorable in EVERY condition — the one common subset a cross-condition
    comparison may use. Averaging each condition over its own scorable set would
    compare different query subsets and read the difference as a condition effect
    (mirrors Metrics.comparable)."""
    return [r for r in rows
            if all((r.get("scorable") or {}).get(c) for c in conds)]


def _mrr(rows, cond, cap):
    """Mean reciprocal rank in one condition, target beyond `cap` counting as a
    miss. Mirrors schema._reciprocal_rank so figures and printed summary agree."""
    if not rows:
        return 0.0
    total = 0.0
    for r in rows:
        rank = (r.get("target_rank") or {}).get(cond)
        if rank and (cap is None or rank <= cap):
            total += 1.0 / rank
    return total / len(rows)


# ─── 1. rank distribution ───────────────────────────────────────────────────

def plot_rank_distribution(rows, out, system, conds):
    rows = comparable(rows, conds)
    if not rows:
        print("  (no comparable items — skipping rank_distribution)")
        return

    data, colors, labels = [], [], []
    for cond in conds:
        ranks = [r["target_rank"][cond] for r in rows if r["target_rank"].get(cond)]
        misses = len(rows) - len(ranks)
        data.append(ranks or [np.nan])
        colors.append(COND_COLOR.get(cond, GRAY))
        # A box drawn over 12 of 15 queries is a different claim from one drawn over
        # all 15. Put the count in the tick label itself so it cannot be cropped or
        # read as belonging to the neighbouring box.
        note = f"n={len(ranks)}/{len(rows)}" + (f", {misses} not found" if misses else "")
        labels.append(f"{COND_LABEL.get(cond, cond)}\n{note}")

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
    ax.set_title(f"Target-rank distribution by condition — {system}\n"
                 f"({len(rows)} items scorable in all {len(conds)} conditions)",
                 fontsize=10)
    save(fig, out, "rank_distribution")


# ─── 2. MRR at rank caps ────────────────────────────────────────────────────

def plot_mrr_caps(rows, out, system, retrieval_k, conds):
    rows = comparable(rows, conds)
    if not rows:
        print("  (no comparable items — skipping mrr_caps)")
        return

    caps = list(MRR_CAPS)
    x = np.arange(len(caps))
    width = 0.8 / len(conds)

    # Caps that cannot bite: nothing was retrieved past k, so MRR@cap there is just
    # the uncapped MRR and must not read as an independent measurement.
    inert = [bool(retrieval_k) and c >= retrieval_k for c in caps]

    fig, ax = plt.subplots(figsize=(1.8 * len(caps) + 3.5, 4.4))
    top = 0.0
    for i, cond in enumerate(conds):
        vals = [_mrr(rows, cond, c) for c in caps]
        top = max(top, *vals)
        offs = x + (i - (len(conds) - 1) / 2) * width
        bars = ax.bar(offs, vals, width=width * 0.92, color=COND_COLOR.get(cond, GRAY))
        for bar, is_inert in zip(bars, inert):
            if is_inert:
                bar.set_hatch("//")
                bar.set_edgecolor("white")
        for xi, v in zip(offs, vals):
            ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=6.5, color=INK2,
                        rotation=90 if len(conds) > 2 else 0)

    ax.set_xticks(x)
    ax.set_xticklabels([f"@{c}" for c in caps])
    ax.set_xlabel("rank cap (a target found deeper than the cap counts as a miss)")
    ax.set_ylabel("MRR")
    # Scale to the data, not to 1.0: KIS MRRs sit well under 0.5 and a fixed 0..1
    # axis flattens the differences between caps, which is the whole point here.
    ax.set_ylim(0, max(0.1, top * 1.35))
    ax.grid(axis="x", visible=False)

    # Legend built from plain swatches — letting bar containers supply the handles
    # copies the hatch of whichever bar came first and makes every condition look
    # inert.
    handles = [Patch(facecolor=COND_COLOR.get(c, GRAY), label=c) for c in conds]
    if any(inert):
        handles.append(Patch(facecolor="white", edgecolor=MUTED, hatch="//",
                             label=f"cap ≥ retrieved k={retrieval_k} (= uncapped)"))
    ax.legend(handles=handles, frameon=False, loc="upper right", fontsize=8,
              ncol=2 if len(handles) > 3 else 1)
    ax.set_title(f"MRR at rank caps — {system} ({len(rows)} comparable items)")
    save(fig, out, "mrr_caps")


# ─── 3. Recall@k ────────────────────────────────────────────────────────────

def plot_recall_at_k(rows, out, system, ks, conds):
    rows = comparable(rows, conds)
    if not rows or not ks:
        print("  (no comparable items / no ks — skipping recall_at_k)")
        return

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for cond in conds:
        means = [float(np.mean([(r["recall"].get(cond) or {}).get(str(k), 0.0)
                                for r in rows])) for k in ks]
        ax.plot(ks, means, marker="o", color=COND_COLOR.get(cond, GRAY),
                linewidth=1.8, markersize=5, label=cond)

    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("k")
    # Each condition is scored against ITS OWN oracle answer, so this compares how
    # well the index reproduces four different exact rankings — not the conditions
    # against one shared truth.
    ax.set_ylabel("mean Recall@k vs that condition's exact k-NN")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    ax.set_title(f"Geometric correctness — {system} ({len(rows)} comparable items)")
    save(fig, out, "recall_at_k")


# ─── 4. the 2x2 grid ────────────────────────────────────────────────────────

def plot_condition_grid(rows, out, system, retrieval_k, conds):
    """The matrix read as a matrix: filter delta down one axis, raw-vs-semantic
    across the other. Only drawn when all four cells ran — a partial matrix has no
    interpretable margins."""
    if not all(c in conds for c in CANONICAL_ORDER):
        print("  (partial condition matrix — skipping condition_grid)")
        return
    rows = comparable(rows, conds)
    if not rows:
        print("  (no comparable items — skipping condition_grid)")
        return

    cap = next((c for c in sorted(MRR_CAPS, reverse=True)
                if not retrieval_k or c < retrieval_k), min(MRR_CAPS))
    grid = np.array([[_mrr(rows, f"{t}+{p}", cap) for p in ("nofilter", "filter")]
                     for t in ("raw", "semantic")])

    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    ax.grid(False)          # the global grid rcParam draws lines through the cells
    im = ax.imshow(grid, cmap="YlGnBu", vmin=0, vmax=max(grid.max() * 1.1, 0.05))
    ax.set_xticks([0, 1], ["no filter", "+ filter"])
    ax.set_yticks([0, 1], ["raw\nquery text", "semantic\nremainder"])
    for i in range(2):
        for j in range(2):
            # Contrast against the colormap rather than a fixed ink colour.
            colour = "white" if grid[i, j] > grid.max() * 0.6 else INK
            ax.annotate(f"{grid[i, j]:.3f}", (j, i), ha="center", va="center",
                        fontsize=15, color=colour)
    ax.set_title(f"The 2×2 — MRR@{cap} — {system}\n"
                 f"({len(rows)} items scorable in all four cells)", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.75, label=f"MRR@{cap}")
    # The margins are the two deltas the matrix exists to separate.
    d_filter = grid[:, 1] - grid[:, 0]
    d_sem = grid[1, :] - grid[0, :]
    ax.set_xlabel(f"Δ filter: raw {d_filter[0]:+.3f}, semantic {d_filter[1]:+.3f}\n"
                  f"Δ semantic: no-filter {d_sem[0]:+.3f}, filter {d_sem[1]:+.3f}",
                  fontsize=8.5)
    save(fig, out, "condition_grid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/metrics.pgvector.jsonl")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"[plot] {args.inp} -> {args.out}")
    header, rows = load_metrics(args.inp)
    system = header.get("system", "unknown")
    ks = header.get("ks") or []
    retrieval_k = header.get("retrieval_k") or 0
    conds = conditions(header, rows)
    print(f"  {len(rows)} queries, {len(comparable(rows, conds))} comparable, "
          f"system={system}")
    print(f"  conditions ({len(conds)}/{len(CANONICAL_ORDER)}): {', '.join(conds)}")
    if not retrieval_k:
        print("  [note] no retrieval_k in the header (pre-2026-08 metrics file); "
              "MRR caps cannot be checked against the run's retrieval depth")

    plot_rank_distribution(rows, args.out, system, conds)
    plot_mrr_caps(rows, args.out, system, retrieval_k, conds)
    plot_recall_at_k(rows, args.out, system, ks, conds)
    plot_condition_grid(rows, args.out, system, retrieval_k, conds)
    print("[done]")


if __name__ == "__main__":
    main()
