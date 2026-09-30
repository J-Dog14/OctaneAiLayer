"""
The nightly cohort job.

Group-level statistics belong here, not inside a per-athlete report. The
stratum velocity-correlate table is identical for every PRO pitcher, so
recomputing it per athlete both burned a warehouse pull per report and let
two reports generated a day apart disagree with nothing on record to explain
why. Computed once, stamped with a config hash, stored, and joined to.

What it computes
----------------
  1. metric reliability — the measurement-noise floor, per metric, plus a
     per-stratum estimate where the stratum is big enough to support one
  2. within-athlete velocity correlates, per stratum, with bootstrap intervals
     and a confidence tier
  3. for the strongest of those, the assessment correlates (the "chain"),
     FDR-corrected within each stratum

Run it on a schedule:

    python -m src.main research cohort-refresh

Everything is upserted, so a re-run is cheap and safe. Nothing here writes to
the App DB.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from src.research import findings_store, loaders
from src.research.config import CONFIG, ResearchConfig
from src.research.kinematic_drivers import correlate_kinematic_to_assessments
from src.research.pitching_deep import (
    correlate_velocity_within_athlete,
    metric_columns_pitching,
)
from src.research.reliability import (
    compute_reliability,
    estimate_between_session,
    merge_reliability,
)
from src.research.stats_support import correlation_tier, spearman_with_ci

ANALYSIS_VELOCITY = "velocity_within_athlete"
ANALYSIS_CHAIN = "kin_to_assessment"
STRATUM_ALL = "ALL"

DEFAULT_STRATA = ["PRO", "COLLEGE", "HIGH SCHOOL", "YOUTH"]


@dataclass
class CohortJobResult:
    started_at: datetime
    finished_at: datetime
    config_hash: str
    reliability_rows: int
    velocity_rows: dict[str, int]
    chain_rows: dict[str, int]
    skipped: dict[str, str]
    notes: list[str]

    def describe(self) -> str:
        secs = (self.finished_at - self.started_at).total_seconds()
        lines = [
            f"cohort refresh finished in {secs:.1f}s (config {self.config_hash})",
            f"  reliability          {self.reliability_rows} metrics",
        ]
        for s, n in sorted(self.velocity_rows.items()):
            lines.append(f"  velocity correlates  {s:<14} {n}")
        for s, n in sorted(self.chain_rows.items()):
            lines.append(f"  assessment chain     {s:<14} {n}")
        for s, why in sorted(self.skipped.items()):
            lines.append(f"  skipped              {s:<14} {why}")
        for n in self.notes:
            lines.append(f"  note: {n}")
        return "\n".join(lines)


def refresh_cohort_findings(
    *,
    config: ResearchConfig = CONFIG,
    strata: list[str] | None = None,
    top_k_chain: int = 5,
    persist: bool = True,
    with_ci: bool = True,
) -> CohortJobResult:
    started = datetime.now()
    cfg_hash = config.fingerprint()
    strata = strata if strata is not None else list(DEFAULT_STRATA)
    velocity_rows: dict[str, int] = {}
    chain_rows: dict[str, int] = {}
    skipped: dict[str, str] = {}
    notes: list[str] = []

    loaders.clear_cache()
    all_trials = loaders.trials(copy=False)
    if all_trials.empty:
        return CohortJobResult(started, datetime.now(), cfg_hash, 0, {}, {},
                               {"ALL": "no trial data"}, notes)

    # ── 1. Reliability ────────────────────────────────────────────────────
    metrics = metric_columns_pitching(
        all_trials, processed_only=True, exclude_symptomatic=False,
        role="pitcher")
    if "velocity_mph" in all_trials.columns:
        metrics = metrics + ["velocity_mph"]

    within = compute_reliability(all_trials, metrics, config=config)
    between = estimate_between_session(all_trials, metrics, config=config)
    rel = merge_reliability(within, between)
    n_rel = 0
    if persist and not rel.empty:
        n_rel = findings_store.save_reliability(
            rel, stratum=STRATUM_ALL, config_hash=cfg_hash,
            analysis_version=config.analysis_version)
    else:
        n_rel = len(rel)
    if not between.empty:
        notes.append(
            f"{len(between)} metric(s) have a true test-retest estimate from "
            f"repeat captures; the rest fall back to within-session scatter, "
            f"which understates real measurement error.")
    low_icc = rel[rel["icc"].notna() & (rel["icc"] < 0.5)] if not rel.empty else pd.DataFrame()
    if not low_icc.empty:
        notes.append(
            f"{len(low_icc)} metric(s) have ICC below 0.5 — most of their "
            f"spread is measurement scatter rather than real difference "
            f"between athletes. They are excluded from coach-facing findings.")

    # ── 2. Velocity correlates, per stratum ───────────────────────────────
    counts = loaders.athlete_counts_by_stratum()
    for stratum in strata + [STRATUM_ALL]:
        if stratum == STRATUM_ALL:
            sub = all_trials
            n_ath = int(sub["athlete_uuid"].nunique())
        else:
            n_ath = int(counts.get(stratum, 0))
            if n_ath < config.min_athletes_for_stratum:
                skipped[stratum] = (f"{n_ath} athletes, needs "
                                    f"{config.min_athletes_for_stratum}")
                continue
            sub = loaders.trials(age_group=stratum, copy=False)
        if sub.empty:
            skipped[stratum] = "no trials"
            continue

        try:
            res = correlate_velocity_within_athlete(
                sub,
                min_trials_per_athlete=config.min_trials_per_athlete,
                min_n=config.min_n_within_athlete,
                fdr_alpha=config.fdr_alpha,
                processed_only=True, exclude_symptomatic=True,
            )
        except Exception as e:
            skipped[stratum] = f"velocity analysis failed: {e}"
            continue
        if res.empty:
            skipped[stratum] = "no correlate met the n floor"
            continue

        res = res.copy()
        res["n"] = res.get("n_trials", 0)
        if with_ci:
            res = _add_intervals(sub, res, config=config)
        res["tier"] = [
            correlation_tier(
                _as_result(r), fdr_significant=bool(r.get("fdr_significant")),
                min_n=config.min_n_within_athlete, config=config)
            for _, r in res.iterrows()
        ]
        res["n_athletes"] = res.get("n_athletes", n_ath)

        if persist:
            velocity_rows[stratum] = findings_store.save_cohort_findings(
                res, analysis=ANALYSIS_VELOCITY, stratum=stratum,
                target="velocity_mph", config_hash=cfg_hash,
                analysis_version=config.analysis_version)
        else:
            velocity_rows[stratum] = len(res)

        # ── 3. Chain: top correlates → assessments ────────────────────────
        top = res.head(top_k_chain)
        n_chain = 0
        for _, row in top.iterrows():
            m = row["metric"]
            try:
                chain = correlate_kinematic_to_assessments(
                    m, role="pitcher",
                    age_group=None if stratum == STRATUM_ALL else stratum,
                    min_trials=config.min_trials_per_athlete,
                    aggregation="mean",
                    min_n=config.min_n_cross_sectional,
                    fdr_alpha=config.fdr_alpha,
                )
            except Exception:
                continue
            corr = chain.get("correlations", pd.DataFrame())
            if corr.empty:
                continue
            corr = corr.rename(columns={"assessment_metric": "metric"}).copy()
            corr["n_athletes"] = chain.get("n_athletes", 0)
            corr["tier"] = [
                correlation_tier(_as_result(r),
                                 fdr_significant=bool(r.get("fdr_significant")),
                                 min_n=config.min_n_cross_sectional,
                                 config=config)
                for _, r in corr.iterrows()
            ]
            if persist:
                n_chain += findings_store.save_cohort_findings(
                    corr, analysis=ANALYSIS_CHAIN, stratum=stratum,
                    target=m, config_hash=cfg_hash,
                    analysis_version=config.analysis_version)
            else:
                n_chain += len(corr)
        chain_rows[stratum] = n_chain

    return CohortJobResult(started, datetime.now(), cfg_hash, n_rel,
                           velocity_rows, chain_rows, skipped, notes)


def _add_intervals(trials: pd.DataFrame, res: pd.DataFrame, *,
                   config: ResearchConfig, limit: int = 40) -> pd.DataFrame:
    """Bootstrap intervals for the top correlates.

    Only the top `limit` get an interval: bootstrapping 300 metrics would
    dominate the job's runtime, and the tail is never shown to anyone.
    Everything below the cut keeps a null interval, which `correlation_tier`
    correctly reads as 'suggestive at best'.
    """
    counts = trials.groupby("athlete_uuid").size()
    keep = counts[counts >= config.min_trials_per_athlete].index
    sub = trials[trials["athlete_uuid"].isin(keep)]
    if sub.empty:
        res["ci_low"] = None
        res["ci_high"] = None
        return res

    cols = [c for c in res["metric"].head(limit) if c in sub.columns]
    numeric = cols + ["velocity_mph"]
    means = sub.groupby("athlete_uuid")[numeric].transform("mean")
    resid = sub[numeric] - means

    lows: dict[str, float] = {}
    highs: dict[str, float] = {}
    for c in cols:
        r = spearman_with_ci(resid[c], resid["velocity_mph"], config=config)
        lows[c] = r.ci_low
        highs[c] = r.ci_high
    res["ci_low"] = res["metric"].map(lows)
    res["ci_high"] = res["metric"].map(highs)
    return res


def _as_result(row) -> Any:
    from src.research.stats_support import CorrelationResult
    return CorrelationResult(
        r=_f(row.get("r")), p_value=_f(row.get("p_value")),
        n=int(row.get("n") or row.get("n_trials") or 0),
        ci_low=_f(row.get("ci_low")), ci_high=_f(row.get("ci_high")),
    )


def _f(v) -> float | None:
    try:
        f = float(v)
        return None if f != f else f
    except (TypeError, ValueError):
        return None


__all__ = ["refresh_cohort_findings", "CohortJobResult",
           "ANALYSIS_VELOCITY", "ANALYSIS_CHAIN", "STRATUM_ALL"]
