"""
profile.py — selectivity / plan profiler for the authored query set.

Standalone diagnostic (like the Analyzer): given the loaded query items and a LIVE
pgvector connection, report, per filtered item, where its REAL VBS filter lands on
pgvector's exact↔approximate cutover:

  * global selectivity — TRUE count of keyframes satisfying the full (AND-ed)
    predicate at the pinned thresholds, / corpus. The real number, not the
    planner's estimate, and (unlike the oracle's per-filter selectivity curve) the
    conjunctive count.
  * plan + planner_est_rows — which strategy the planner picks (HNSW index scan =
    approximate post-filter, vs keyframes seqscan = exact filter-then-scan) and the
    estimated passing set that drove the choice (from EXPLAIN, no execution).
  * near_query_pass_rate — fraction of the query's top-N EXACT unfiltered
    neighbours (of vector_query) that satisfy the filter: the LOCAL pass-rate the
    HNSW walk actually experiences.
  * divergence = global_selectivity − near_query_pass_rate — the signal that
    predicts silent recall loss on the HNSW path: positive means the planner's
    global estimate OVERSTATES the near-query pass-rate (thinks "broad/safe" while
    locally the filter is sparse). See the vault note "FRAME — pgvector planner split".

This answers the first cutover-characterization question — where the real filters
land and how far global can misprice local — over the authored set. It needs the
DB, so it runs on the HPC (like build_gt / run_benchmark).

Plan detection is pgvector-specific (EXPLAIN); the selectivity + pass-rate parts
are system-agnostic. Kept out of the adapter ABC: a diagnostic, not the run contract.
Requires an adapter exposing corpus_size / count_passing / count_members /
plan_choice (pgvector has them).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence

from .encode import Encoder
from .schema import Predicate, QueryItem

DEFAULT_NEAR_QUERY_N = 100


@dataclass
class SelectivityProfile:
    query_id: str
    filter_summary: str                 # e.g. "object:{fish,hat}"
    n_filters: int
    corpus_size: int
    global_count: int                   # true passing keyframes (full conjunction)
    global_selectivity: float           # global_count / corpus_size
    plan: str                           # "hnsw" | "seqscan" | "other"
    planner_est_rows: int | None        # planner's estimated passing set (EXPLAIN)
    near_query_n: int                   # neighbours actually examined (0 if GT absent)
    near_query_passes: int              # of those that satisfy the filter
    near_query_pass_rate: float | None  # None when no vec-nofilter GT is available

    @property
    def divergence(self) -> float | None:
        """global_selectivity − near_query_pass_rate; positive ⇒ global overstates
        the local pass-rate (the silent-recall-loss-on-HNSW risk case)."""
        if self.near_query_pass_rate is None:
            return None
        return self.global_selectivity - self.near_query_pass_rate

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "filter_summary": self.filter_summary,
            "n_filters": self.n_filters,
            "corpus_size": self.corpus_size,
            "global_count": self.global_count,
            "global_selectivity": self.global_selectivity,
            "plan": self.plan,
            "planner_est_rows": self.planner_est_rows,
            "near_query_n": self.near_query_n,
            "near_query_passes": self.near_query_passes,
            "near_query_pass_rate": self.near_query_pass_rate,
            "divergence": self.divergence,
        }


class Profiler:
    """Profiles the selectivity + plan choice of each filtered item's predicate.

    `adapter` must expose the profiling diagnostics (corpus_size, count_passing,
    count_members, plan_choice). `encoder` is the shared text encoder. `k` is the
    retrieval depth the plan is priced at (match the run's k so plan choice agrees
    with what the benchmark sees).
    """

    def __init__(
        self,
        adapter,
        encoder: Encoder,
        k: int = 1000,
        near_query_n: int = DEFAULT_NEAR_QUERY_N,
    ):
        self.adapter = adapter
        self.encoder = encoder
        self.k = k
        self.near_query_n = near_query_n

    def profile(self, items: Sequence[QueryItem]) -> list[SelectivityProfile]:
        out: list[SelectivityProfile] = []
        with self.adapter:
            corpus = self.adapter.corpus_size()
            for item in items:
                if not item.filters:
                    continue  # no-filter items have no selectivity to profile
                out.append(self._profile_one(item, corpus))
        return out

    def _profile_one(self, item: QueryItem, corpus: int) -> SelectivityProfile:
        vec = self.encoder.encode(item.vector_query)
        global_count = self.adapter.count_passing(item.filters)
        plan, est_rows = self.adapter.plan_choice(vec, item.filters, self.k)

        # near-query pass-rate over the exact unfiltered neighbours of vector_query
        # (the list the filtered HNSW walk traverses). Absent until an --all-baselines
        # GT run has populated geometric_gt_vec_nofilter.
        gt = item.ground_truth
        neighbours = list((gt.gt_vec_nofilter or []) if gt else [])[: self.near_query_n]
        if neighbours:
            passes = self.adapter.count_members(neighbours, item.filters)
            rate: float | None = passes / len(neighbours)
        else:
            passes, rate = 0, None

        return SelectivityProfile(
            query_id=item.query_id,
            filter_summary=_summarize_filters(item.filters),
            n_filters=len(item.filters),
            corpus_size=corpus,
            global_count=global_count,
            global_selectivity=(global_count / corpus) if corpus else 0.0,
            plan=plan,
            planner_est_rows=est_rows,
            near_query_n=len(neighbours),
            near_query_passes=passes,
            near_query_pass_rate=rate,
        )

    def summary(self, profiles: Sequence[SelectivityProfile]) -> str:
        if not profiles:
            return "no filtered items to profile"
        rows = sorted(profiles, key=lambda p: p.global_selectivity)
        lines = [
            f"{'qid':>6} | {'filter':<26} | {'glob.sel':>9} | {'plan':>8} | "
            f"{'est.rows':>9} | {'near-pass':>9} | {'diverg':>7}",
            "-" * 92,
        ]
        for p in rows:
            npass = f"{p.near_query_pass_rate:.3f}" if p.near_query_pass_rate is not None else "  n/a"
            div = f"{p.divergence:+.3f}" if p.divergence is not None else "  n/a"
            est = str(p.planner_est_rows) if p.planner_est_rows is not None else "n/a"
            lines.append(
                f"{p.query_id:>6} | {p.filter_summary[:26]:<26} | "
                f"{p.global_selectivity:>9.4f} | {p.plan:>8} | {est:>9} | "
                f"{npass:>9} | {div:>7}"
            )
        # distribution roll-up — the actual "where does the workload land" answer
        by_plan: dict[str, int] = {}
        for p in profiles:
            by_plan[p.plan] = by_plan.get(p.plan, 0) + 1
        scored = [p for p in profiles if p.divergence is not None]
        risk = [p for p in scored
                if p.plan == "hnsw" and (p.divergence or 0) > 0]
        lines += [
            "-" * 92,
            f"filtered items: {len(profiles)}   "
            + "  ".join(f"{plan}={n}" for plan, n in sorted(by_plan.items())),
            f"HNSW-path items where global overstates near-query pass-rate "
            f"(silent-recall-loss risk): {len(risk)}"
            + (f"  [{', '.join(p.query_id for p in risk)}]" if risk else ""),
        ]
        if len(scored) < len(profiles):
            lines.append(
                f"note: {len(profiles) - len(scored)} item(s) lack vec-nofilter GT "
                f"→ near-query pass-rate n/a (run build_gt with vec-nofilter baseline)"
            )
        return "\n".join(lines)


def write_jsonl(profiles: Sequence[SelectivityProfile], path: str, system: str) -> None:
    with open(path, "w") as f:
        f.write(json.dumps({"system": system, "kind": "selectivity_profile"}) + "\n")
        for p in profiles:
            f.write(json.dumps(p.to_dict()) + "\n")


def _summarize_filters(filters: Sequence[Predicate]) -> str:
    parts = []
    for f in filters:
        vals = [f.value] if isinstance(f.value, str) else list(f.value)
        parts.append(f"{f.filter_type}:{{{','.join(str(v) for v in vals)}}}")
    return " AND ".join(parts)
