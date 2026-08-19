#!/usr/bin/env python3
"""plot_kef_sweep.py — frontier figures + headline reducer for the k×ef sweep.

    python3 scripts/plot_kef_sweep.py --data data --out results/figures

Reads every cell the sweep wrote — data/metrics.<system>.k<k>.ef<ef>.jsonl (see
run_benchmark.py --k-grid/--ef-grid) — and turns the grid into:

  * a tidy table (printed + kef_sweep.tidy.csv): one row per (system, k, ef) with
    mean Recall@k and MRR for the two swept conditions and the filtered p50/p95.
  * the HEADLINE point per system: at k=50, the SMALLEST ef whose filtered
    Recall@k >= 0.9 (the realistic KIS operating point — see the vault note
    "FRAME k×ef sweep — plan and code spec"). Written to headline_point.json.
  * figures (PDF+PNG): recall-vs-ef (the index-quality dial), latency-vs-recall
    (the QPS/Recall frontier — the headline figure), MRR-vs-k and latency-vs-k at
    each system's headline ef.

Standalone but reuses plot_metrics.py's parsing + palette (sibling in scripts/), so
the recall/MRR math is byte-for-byte the same as the per-run figures. Aggregates are
over the COMPARABLE subset (items scorable in every swept condition), exactly as
plot_metrics does, so a cell's number is not an average over a shifting item set.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Sibling script — same palette + the exact recall/MRR/comparable logic the per-run
# figures use. scripts/ is on sys.path[0] when this file is run directly.
from plot_metrics import (BLUE, GRID, INK2, MAGENTA, MUTED, comparable,
                          conditions, load_metrics, save, _mean_recall, _mrr)

FILT, NOFILT = "semantic+filter", "semantic+nofilter"
HEADLINE_K = 50            # the KIS operating depth (Omar: top-25..50)
RECALL_TARGET = 0.90       # realistic operating point, read on FILTERED recall
KIS_CAP = 10               # MRR cap for the task headline (KIS ≈ top-10)

# One filename: metrics.<system>.k<k>.ef<ef>.jsonl
_CELL_RE = re.compile(r"metrics\.([^.]+)\.k(\d+)\.ef(\d+)\.jsonl$")


def collect(data_dir):
    """-> list of cell dicts, one per metrics.<system>.k<k>.ef<ef>.jsonl."""
    cells = []
    for path in sorted(glob.glob(os.path.join(data_dir, "metrics.*.k*.ef*.jsonl"))):
        m = _CELL_RE.search(os.path.basename(path))
        if not m:
            continue
        system, k, ef = m.group(1), int(m.group(2)), int(m.group(3))
        header, rows = load_metrics(path)
        # ef in the header is authoritative; fall back to the filename.
        ef = header.get("ef_search", ef)
        k = header.get("retrieval_k", k)
        conds = conditions(header, rows)
        comp = comparable(rows, [c for c in (FILT, NOFILT) if c in conds])
        cells.append({
            "system": system, "k": k, "ef": ef, "n": len(comp),
            "recall_filt": _mean_recall(comp, FILT, k),
            "recall_nofilt": _mean_recall(comp, NOFILT, k),
            "mrr_filt": _mrr(comp, FILT, KIS_CAP),
            "p50": _percentile([r["latency_ms"].get(FILT, 0.0) for r in comp], 50),
            "p95": _percentile([r["latency_ms"].get(FILT, 0.0) for r in comp], 95),
        })
    return cells


def _percentile(xs, p):
    xs = [x for x in xs if x is not None]
    return float(np.percentile(xs, p)) if xs else 0.0


def systems(cells):
    return sorted({c["system"] for c in cells})


def _for(cells, system):
    return sorted((c for c in cells if c["system"] == system),
                  key=lambda c: (c["k"], c["ef"]))


def print_tidy(cells, data_dir):
    """Print the grid and write kef_sweep.tidy.csv."""
    cols = ["system", "k", "ef", "n", "recall_filt", "recall_nofilt",
            "mrr_filt", "p50", "p95"]
    print(f"\n{'system':>9} {'k':>4} {'ef':>5} {'n':>3} "
          f"{'R@k filt':>9} {'R@k nof':>8} {'MRR@10':>7} {'p50 ms':>9} {'p95 ms':>9}")
    print("-" * 74)
    for c in cells:
        print(f"{c['system']:>9} {c['k']:>4} {c['ef']:>5} {c['n']:>3} "
              f"{c['recall_filt']:>9.3f} {c['recall_nofilt']:>8.3f} "
              f"{c['mrr_filt']:>7.3f} {c['p50']:>9.1f} {c['p95']:>9.1f}")
    csv_path = os.path.join(data_dir, "kef_sweep.tidy.csv")
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for c in cells:
            f.write(",".join(str(c[k]) for k in cols) + "\n")
    print(f"\ntidy -> {csv_path}")


def headline(cells, data_dir):
    """Per system: at k=HEADLINE_K, the SMALLEST ef with filtered Recall@k >= 0.90.
    Falls back to the max-recall cell if none reaches the target."""
    out = {}
    for system in systems(cells):
        depth = [c for c in _for(cells, system) if c["k"] == HEADLINE_K]
        if not depth:
            continue
        hits = [c for c in sorted(depth, key=lambda c: c["ef"])
                if c["recall_filt"] >= RECALL_TARGET]
        pick = hits[0] if hits else max(depth, key=lambda c: c["recall_filt"])
        out[system] = {
            "k": HEADLINE_K, "ef": pick["ef"],
            "reached_target": bool(hits),
            "recall_filt": round(pick["recall_filt"], 4),
            "mrr_filt_at10": round(pick["mrr_filt"], 4),
            "p50_ms": round(pick["p50"], 1), "p95_ms": round(pick["p95"], 1),
        }
    print(f"\n=== headline @ k={HEADLINE_K}, filtered Recall >= {RECALL_TARGET} ===")
    for system, h in out.items():
        flag = "" if h["reached_target"] else "  [!] target not reached — max-recall cell"
        print(f"  {system:>9}: ef={h['ef']}  R={h['recall_filt']}  "
              f"MRR@10={h['mrr_filt_at10']}  p50={h['p50_ms']}ms  "
              f"p95={h['p95_ms']}ms{flag}")
    path = os.path.join(data_dir, "headline_point.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"headline -> {path}")
    return out


# ─── figures ─────────────────────────────────────────────────────────────────
def _kcolors(ks):
    cmap = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(ks)))
    return {k: cmap[i] for i, k in enumerate(sorted(ks))}


def fig_recall_vs_ef(cells, out, stamp=None):
    """Recall@k vs ef, per k — filtered (solid) and unfiltered (dashed). One panel
    per system. The 0.90 target is a horizontal guide."""
    syss = systems(cells)
    fig, axes = plt.subplots(1, len(syss), figsize=(5.2 * len(syss), 4.0),
                             squeeze=False, sharey=True)
    for ax, system in zip(axes[0], syss):
        cs = _for(cells, system)
        ks = sorted({c["k"] for c in cs})
        col = _kcolors(ks)
        for k in ks:
            row = [c for c in cs if c["k"] == k]
            ef = [c["ef"] for c in row]
            ax.plot(ef, [c["recall_filt"] for c in row], "-o", color=col[k],
                    ms=4, label=f"k={k}")
            ax.plot(ef, [c["recall_nofilt"] for c in row], "--", color=col[k],
                    alpha=0.6)
        ax.axhline(RECALL_TARGET, color=MUTED, lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("ef_search")
        ax.set_title(system)
        ax.grid(True, color=GRID, lw=0.6)
    axes[0][0].set_ylabel("mean Recall@k")
    axes[0][-1].legend(frameon=False, fontsize=8, title="solid=filter\ndash=no-filter")
    fig.suptitle("Recall@k vs ef_search (the index-quality dial)")
    save(fig, out, "kef_recall_vs_ef", stamp)


def fig_latency_vs_recall(cells, out, stamp=None):
    """The QPS/Recall frontier: filtered p50 latency vs filtered Recall@k, one line
    per k (walking ef along it). The headline figure."""
    syss = systems(cells)
    fig, axes = plt.subplots(1, len(syss), figsize=(5.2 * len(syss), 4.0),
                             squeeze=False, sharey=True)
    for ax, system in zip(axes[0], syss):
        cs = _for(cells, system)
        ks = sorted({c["k"] for c in cs})
        col = _kcolors(ks)
        for k in ks:
            row = sorted((c for c in cs if c["k"] == k), key=lambda c: c["recall_filt"])
            ax.plot([c["recall_filt"] for c in row], [c["p50"] for c in row],
                    "-o", color=col[k], ms=4, label=f"k={k}")
        ax.axvline(RECALL_TARGET, color=MUTED, lw=0.8, ls=":")
        ax.set_yscale("log")
        ax.set_xlabel("filtered Recall@k")
        ax.set_title(system)
        ax.grid(True, color=GRID, lw=0.6)
    axes[0][0].set_ylabel("filtered p50 latency (ms)")
    axes[0][-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Latency vs Recall frontier (filtered, semantic query)")
    save(fig, out, "kef_latency_vs_recall", stamp)


def fig_vs_k_at_headline(cells, head, out, stamp=None):
    """MRR@10 and p50/p95 vs k, each system pinned at its headline ef."""
    syss = [s for s in systems(cells) if s in head]
    fig, (axm, axl) = plt.subplots(1, 2, figsize=(9.5, 4.0))
    colr = {s: c for s, c in zip(syss, (BLUE, MAGENTA, INK2))}
    for system in syss:
        ef = head[system]["ef"]
        row = sorted((c for c in cells if c["system"] == system and c["ef"] == ef),
                     key=lambda c: c["k"])
        ks = [c["k"] for c in row]
        axm.plot(ks, [c["mrr_filt"] for c in row], "-o", color=colr[system],
                 ms=4, label=f"{system} (ef={ef})")
        axl.plot(ks, [c["p50"] for c in row], "-o", color=colr[system], ms=4,
                 label=f"{system} p50")
        axl.plot(ks, [c["p95"] for c in row], "--s", color=colr[system], ms=4,
                 alpha=0.6, label=f"{system} p95")
    for ax in (axm, axl):
        ax.set_xlabel("retrieval depth k")
        ax.grid(True, color=GRID, lw=0.6)
        ax.legend(frameon=False, fontsize=8)
    axm.set_ylabel("filtered MRR@10")
    axm.set_title("Task success vs depth")
    axl.set_ylabel("filtered latency (ms)")
    axl.set_yscale("log")
    axl.set_title("Latency vs depth")
    save(fig, out, "kef_vs_k_at_headline", stamp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="dir with metrics.*.k*.ef*.jsonl")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()

    cells = collect(args.data)
    if not cells:
        raise SystemExit(f"no k×ef cells found in {args.data} "
                         "(expected metrics.<system>.k<k>.ef<ef>.jsonl)")
    os.makedirs(args.out, exist_ok=True)
    print(f"loaded {len(cells)} cell(s) across systems: {', '.join(systems(cells))}")
    print_tidy(cells, args.data)
    head = headline(cells, args.data)

    print("\nfigures:")
    fig_recall_vs_ef(cells, args.out)
    fig_latency_vs_recall(cells, args.out)
    if head:
        fig_vs_k_at_headline(cells, head, args.out)


if __name__ == "__main__":
    main()
