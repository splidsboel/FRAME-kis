#!/usr/bin/env python3
"""plot_data_stats.py — thesis figures for the V3C corpus/metadata characterization.

    python3 scripts/plot_data_stats.py --in data/data_stats.json --out results/figures

Standalone (json + matplotlib only, no `frame` import) so it runs anywhere with the
viz extra — locally after pulling data/data_stats.json, or in-job from data_stats.sh.
Reads the report data_stats.py writes and emits figures (PDF + PNG each), light-mode
/ print, from the validated data-viz palette.

  1. scene_label_dist    — top-40 scene labels by keyframe selectivity (pinned)
  2. object_label_dist   — top-40 object labels by keyframe selectivity (pinned)
  3. labels_per_keyframe  — histograms: #scene labels/kf, #object labels/kf,
                           #object instances/kf (Omar's "max 8 objects, 2-3 scenes")
  4. confidence_dist     — scene & object detection confidence histograms + threshold
  5. cooccurrence        — scene x object co-occurrence selectivity heatmap
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

# ── validated palette (dataviz skill, light surface) — matches plot_cutover.py ──
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
    "grid.linewidth": 0.6,
})


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  {name}.pdf / .png")


def _hbar_dist(rep, kind, color, out, name, top=40):
    labels = rep["distributions"][f"{kind}_pinned"]["labels"][:top]
    y = np.arange(len(labels))[::-1]
    sel = [r["selectivity"] * 100 for r in labels]
    fig, ax = plt.subplots(figsize=(6.5, 8))
    ax.barh(y, sel, color=color, height=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in labels], fontsize=7)
    ax.set_xlabel("keyframe selectivity (%)")
    ax.set_title(f"Top {top} {kind} labels by selectivity (pinned t={rep['thresholds'][kind]})")
    ax.grid(axis="y", visible=False)
    save(fig, out, name)


def plot_label_dists(rep, out):
    _hbar_dist(rep, "scene", BLUE, out, "scene_label_dist")
    _hbar_dist(rep, "object", ORANGE, out, "object_label_dist")


def plot_labels_per_keyframe(rep, out):
    pk = rep["per_keyframe"]
    specs = [
        ("scene_labels_per_kf", "# scene labels / keyframe", BLUE),
        ("object_labels_per_kf", "# object labels / keyframe", ORANGE),
        ("object_instances_per_kf", "# object detections / keyframe", GREEN),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (key, xlabel, color) in zip(axes, specs):
        hist = pk[key]["histogram"]
        ns = [h["n"] for h in hist]
        kf = [h["keyframes"] for h in hist]
        ax.bar(ns, kf, color=color, width=0.85)
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("keyframes (log)")
        ax.set_title(f"max={pk[key]['max_n']}")
        ax.grid(axis="x", visible=False)
    fig.suptitle("Per-keyframe tag-count distributions (pinned thresholds)")
    fig.tight_layout()
    save(fig, out, "labels_per_keyframe")


def plot_confidence(rep, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (kind, color) in zip(axes, (("scene", BLUE), ("object", ORANGE))):
        c = rep["confidence"][kind]
        centers = [(h["lo"] + h["hi"]) / 2 for h in c["histogram"]]
        counts = [h["count"] for h in c["histogram"]]
        width = (c["histogram"][0]["hi"] - c["histogram"][0]["lo"]) if c["histogram"] else 0.05
        ax.bar(centers, counts, width=width * 0.95, color=color)
        ax.axvline(c["threshold"], color=INK, ls="--", lw=1.2,
                   label=f"pinned t={c['threshold']}")
        ax.set_yscale("log")
        ax.set_xlabel(f"{kind} detection confidence")
        ax.set_ylabel("detections (log)")
        ax.set_title(f"{kind}: p50={c['p50']:.2f}, p90={c['p90']:.2f}")
        ax.legend(frameon=False)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Detection confidence distributions")
    fig.tight_layout()
    save(fig, out, "confidence_dist")


def plot_cooccurrence(rep, out):
    co = rep.get("cooccurrence")
    if not co:
        return
    scenes, objects = co["scene_labels"], co["object_labels"]
    si = {s: i for i, s in enumerate(scenes)}
    oi = {o: i for i, o in enumerate(objects)}
    M = np.zeros((len(scenes), len(objects)))
    for p in co["pairs"]:
        M[si[p["scene"]], oi[p["object"]]] = p["kf"]
    fig, ax = plt.subplots(figsize=(max(6, len(objects) * 0.32),
                                     max(5, len(scenes) * 0.30)))
    masked = np.ma.masked_where(M == 0, M)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#f2f1ea")
    im = ax.imshow(masked, aspect="auto", cmap=cmap,
                   norm=LogNorm(vmin=max(masked.min(), 1), vmax=masked.max()))
    ax.set_xticks(np.arange(len(objects)))
    ax.set_xticklabels(objects, rotation=90, fontsize=6)
    ax.set_yticks(np.arange(len(scenes)))
    ax.set_yticklabels(scenes, fontsize=6)
    ax.set_xlabel("object label")
    ax.set_ylabel("scene label")
    ax.set_title(f"scene x object co-occurrence (keyframes, pinned) — "
                 f"{co['n_present']}/{co['n_possible']} pairs exist")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="co-occurring keyframes (log)", shrink=0.7)
    fig.tight_layout()
    save(fig, out, "cooccurrence")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/data_stats.json")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()

    rep = json.load(open(args.inp))
    os.makedirs(args.out, exist_ok=True)
    print(f"[plot] {args.inp} -> {args.out}")
    plot_label_dists(rep, args.out)
    plot_labels_per_keyframe(rep, args.out)
    plot_confidence(rep, args.out)
    plot_cooccurrence(rep, args.out)
    print("[done]")


if __name__ == "__main__":
    main()
