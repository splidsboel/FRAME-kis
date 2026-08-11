"""
Versioning — what a run was measured against, and whether two runs may be compared.

THE PROBLEM. A results file used to record nothing about its inputs. Edit one
query's filter, re-run pgvector, compare against last month's Milvus numbers, and
part of the difference is the edit — with nothing on disk to say so. The same trap
caught the 2026-08-04 harness change (two conditions -> four): identical queries,
incomparable results.

TWO MARKERS, because two independent things move:

  * BenchmarkVersion — the query set AND its ground truth. One marker, because the
    oracle enriches GT in place into each item's `computed` block: they are
    physically the same artifact, and there is nothing to keep in sync.
  * HARNESS_CONTRACT — what a results file MEANS (which conditions exist, what the
    metrics are, the comparable-subset rule). This lives in the code, not the data,
    and changes without benchmark.jsonl being touched.

A hand-set semver alone would drift the first time someone edits a query and forgets
to bump it, so the version carries a DIGEST of the file's result-affecting contents.
The semver is the human-readable claim; the digest is what is enforced.

BUMPING (semantics are about comparability, not API):
  MAJOR  a query's meaning changed, items removed, or GT recomputed under different
         parameters  -> prior results are void.
  MINOR  items added, nothing else  -> prior results still valid on the shared items.
  PATCH  nothing result-affecting (notes, status, `verified` flags).

Digests deliberately cover ONLY what changes results (see _query_core / _gt_core):
an allow-list, so a new authored field cannot silently enter the digest, and
flipping `status: draft -> verified` or fixing a typo in `notes` does not invalidate
a run.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

# Bump when the MEANING of a results file changes: conditions added/removed/renamed,
# a metric redefined, the comparable-subset rule changed. Not for bug fixes that
# leave the shape and semantics intact.
#   1  two conditions (filtered / unfiltered), item-level scorability
#   2  full 2x2 matrix, per-condition scorability, common-subset aggregates
HARNESS_CONTRACT = 2

DIGEST_LEN = 12

# The `computed` fields the harness actually scores against. filter_diagnostics and
# _pending are bookkeeping and deliberately excluded.
GT_FIELDS = (
    "target_keyframe_ids", "target_passes_filter", "filter_selectivity",
    "filter_selectivity_conjunction",
    "geometric_gt_filtered", "geometric_gt_nofilter",
    "geometric_gt_raw_filtered", "geometric_gt_vec_nofilter",
    "scene_threshold", "object_threshold",
)
# Fields whose presence means "the oracle has run on this item" — see _gt_core.
# The selectivity summaries are excluded: filter_selectivity is stubbed to
# [None]*n (a non-None value) and the conjunction is only filled in the
# t-dependent block, so neither is a reliable "oracle ran" signal.
GT_PRESENCE_FIELDS = tuple(
    f for f in GT_FIELDS
    if f not in ("filter_selectivity", "filter_selectivity_conjunction"))


def _sha(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:DIGEST_LEN]


def _query_core(item: dict) -> dict:
    """The authored half that determines results. An ALLOW-LIST: `status`, `source`,
    `notes`, and a filter's `vocab`/`mapping_source`/`verified` are provenance, not
    inputs — changing them must not invalidate anyone's run."""
    d = item.get("decomposition") or {}
    t = item.get("target") or {}
    return {
        "query_id": item["query_id"],
        "raw_query_text": item.get("raw_query_text", ""),
        "vector_query": d.get("vector_query", ""),
        "filters": [
            {"filter_type": f.get("filter_type"), "attribute": f.get("attribute"),
             "op": f.get("op"), "value": f.get("value")}
            for f in (d.get("filters") or [])
        ],
        "target": {k: t.get(k) for k in ("video_id", "start_s", "end_s")},
    }


def _gt_core(item: dict) -> dict | None:
    """The oracle half, or None while the item is still awaiting GT.

    Presence is decided from the substantive fields only: build.py's stub sets
    `filter_selectivity` to `[None] * n_filters`, which is a non-None VALUE and
    would otherwise make an unenriched item look like it had ground truth.
    """
    c = item.get("computed") or {}
    if not any(c.get(k) is not None for k in GT_PRESENCE_FIELDS):
        return None
    return {k: c.get(k) for k in GT_FIELDS}


def query_digest(item: dict) -> str:
    """Digest of the authored half alone. Equal digests mean any ground truth
    computed for the old item is still valid for the new one."""
    return _sha(_query_core(item))


def item_digest(item: dict) -> str:
    """`<query>:<gt>` — one string per item, splittable so a mismatch can say WHICH
    half moved. The gt half is empty while the item has no ground truth."""
    gt = _gt_core(item)
    return f"{query_digest(item)}:{_sha(gt) if gt is not None else ''}"


@dataclass(frozen=True)
class BenchmarkVersion:
    """The identity of one benchmark.jsonl: its queries and their ground truth."""

    version: str                                # hand-set semver, e.g. "1.0.0"
    corpus: str                                 # "v3c1" — queries are corpus-specific
    digest: str                                 # over everything below
    items: dict[str, str] = field(default_factory=dict)   # query_id -> item_digest
    gt_params: dict = field(default_factory=dict)  # oracle k, thresholds, encoder
    n_items: int = 0
    n_with_gt: int = 0

    @classmethod
    def compute(cls, version: str, corpus: str, items: list[dict],
                gt_params: dict | None = None) -> "BenchmarkVersion":
        digests = {it["query_id"]: item_digest(it) for it in items}
        params = dict(gt_params or {})
        return cls(
            version=version,
            corpus=corpus,
            digest=_sha({"corpus": corpus, "items": digests, "gt_params": params}),
            items=digests,
            gt_params=params,
            n_items=len(digests),
            n_with_gt=sum(1 for d in digests.values() if d.split(":", 1)[1]),
        )

    @property
    def label(self) -> str:
        """Short human tag for filenames and figure captions: `v3c1/1.0.0+ab12cd34`."""
        return f"{self.corpus}/{self.version}+{self.digest}"

    def to_dict(self) -> dict:
        return {
            "version": self.version, "corpus": self.corpus, "digest": self.digest,
            "n_items": self.n_items, "n_with_gt": self.n_with_gt,
            "gt_params": self.gt_params, "items": self.items,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BenchmarkVersion":
        return cls(
            version=d["version"], corpus=d["corpus"], digest=d["digest"],
            items=d.get("items", {}), gt_params=d.get("gt_params", {}),
            n_items=d.get("n_items", 0), n_with_gt=d.get("n_with_gt", 0),
        )


@dataclass(frozen=True)
class Compatibility:
    status: str                 # "identical" | "additive" | "incompatible"
    reason: str
    shared: list[str] = field(default_factory=list)   # ids valid in both

    @property
    def ok(self) -> bool:
        return self.status != "incompatible"


def compare(ran: BenchmarkVersion | None,
            current: BenchmarkVersion | None) -> Compatibility:
    """Can results produced under `ran` be scored against `current`?

    Decided from the per-item digests rather than the semver, so a forgotten bump
    cannot wave through a changed query set. `additive` means the sets differ but
    every SHARED item is byte-identical — the run is still valid on those, which is
    what makes the yearly query-set update non-destructive.
    """
    if ran is None or current is None:
        missing = "results file" if ran is None else "query set"
        return Compatibility("incompatible",
                             f"no version marker on the {missing} — it predates "
                             f"versioning, so what it was run against is unknown")

    if ran.corpus != current.corpus:
        return Compatibility("incompatible",
                             f"different corpus: run={ran.corpus} vs "
                             f"query set={current.corpus}")

    if ran.gt_params != current.gt_params:
        diffs = sorted(set(ran.gt_params) | set(current.gt_params))
        detail = ", ".join(f"{k}: {ran.gt_params.get(k)!r} -> {current.gt_params.get(k)!r}"
                           for k in diffs
                           if ran.gt_params.get(k) != current.gt_params.get(k))
        return Compatibility("incompatible",
                             f"ground truth computed under different parameters "
                             f"({detail}) — every recall number changes")

    shared = sorted(set(ran.items) & set(current.items))
    changed = [q for q in shared if ran.items[q] != current.items[q]]
    if changed:
        q_changed = [q for q in changed
                     if ran.items[q].split(":")[0] != current.items[q].split(":")[0]]
        gt_changed = [q for q in changed if q not in q_changed]
        bits = []
        if q_changed:
            bits.append(f"{len(q_changed)} query/queries edited "
                        f"({', '.join(q_changed[:5])}{'…' if len(q_changed) > 5 else ''})")
        if gt_changed:
            bits.append(f"{len(gt_changed)} ground truth changed "
                        f"({', '.join(gt_changed[:5])}{'…' if len(gt_changed) > 5 else ''})")
        return Compatibility("incompatible", "; ".join(bits), shared)

    if set(ran.items) == set(current.items):
        # Same contents. A differing semver here is only a labelling slip.
        if ran.version != current.version:
            return Compatibility(
                "identical",
                f"contents identical but versions differ "
                f"({ran.version} vs {current.version}) — a version was mis-set",
                shared)
        return Compatibility("identical", "", shared)

    added = sorted(set(current.items) - set(ran.items))
    removed = sorted(set(ran.items) - set(current.items))
    bits = []
    if added:
        bits.append(f"{len(added)} item(s) added since the run")
    if removed:
        bits.append(f"{len(removed)} item(s) the run covered are gone")
    return Compatibility(
        "additive",
        f"{'; '.join(bits)} — every shared item is unchanged, so scoring is "
        f"restricted to the {len(shared)} shared item(s)",
        shared)
