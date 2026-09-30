"""
Kinematic-driver analysis — chain from pitching mechanics back to physical
assessments.

The pipeline:
  1. Take a target kinematic metric (e.g., Pelvis_Ang_Vel@Release.Z)
  2. Aggregate to athlete-level (mean across all trials for that athlete)
  3. Correlate athlete-level values against ai_layer.athlete_profiles Z-scores
     (mobility, athletic screen, proteus, force plate, ...)

Answers questions like:
  - What mobility metric predicts a good pelvis-brake mechanic?
  - Do stronger single-leg jumpers block their lead leg better?
  - Which strength assessments correlate with better hip-shoulder separation?

Interpretation notes:
  - The trial→athlete aggregation is intentional: profile Z-scores are
    athlete-level (from d_athletes lifetime data), so this is the correct
    join level.
  - Correlations here are ACROSS-athlete (like the profile matrix work),
    NOT within-athlete. That's the trade-off: we've moved from causal-adjacent
    within-athlete velocity → cross-sectional across-athlete assessment.
  - A finding here means "athletes who have MORE of assessment X tend to
    display MORE of kinematic Y" — hypothesis-generating, not proof.
"""
from __future__ import annotations

import pandas as pd

from src.research.correlations import _bh_q_values, _pairwise_corr
from src.research.pitching_deep import (
    load_pitching_trials_wide,
    metric_columns_pitching,
)
from src.research.profile_matrix import (
    _classify_metric,
    load_matrix,
    metric_columns,
)


def athlete_level_kinematic(
    trial_df: pd.DataFrame,
    kin_metric: str,
    *,
    min_trials: int = 3,
    aggregation: str = "mean",   # 'mean' | 'median' | 'max'
) -> pd.DataFrame:
    """Aggregate a trial-level kinematic to per-athlete values.

    Returns DataFrame: athlete_uuid, name, <kin_metric>, n_trials
    """
    if kin_metric not in trial_df.columns:
        raise ValueError(f"{kin_metric!r} not in trial DataFrame.")
    counts = trial_df.groupby("athlete_uuid").size()
    keep = counts[counts >= min_trials].index
    df = trial_df[trial_df["athlete_uuid"].isin(keep)]
    grouped = df.groupby(["athlete_uuid", "name"])[kin_metric]
    if aggregation == "median":
        val = grouped.median()
    elif aggregation == "max":
        val = grouped.max()
    else:
        val = grouped.mean()
    n_tr = grouped.size().rename("n_trials")
    out = pd.DataFrame({kin_metric: val, "n_trials": n_tr}).reset_index()
    return out


def correlate_kinematic_to_assessments(
    kin_metric: str,
    *,
    role: str | None = "pitcher",
    age_group: str | None = None,
    min_trials: int = 3,
    aggregation: str = "mean",
    min_n: int = 8,
    fdr_alpha: float = 0.10,
    min_as_of_date: str | None = None,
    exclude_mobility: bool = False,
) -> dict:
    """Full pipeline for one kinematic metric.

    Returns dict with:
      kin_metric, aggregation, n_athletes,
      correlations (DataFrame ranked by |r|),
      athlete_level (DataFrame with per-athlete kinematic value)
    """
    # 1. Load trial data (all pitchers by default — kinematic is athlete-level)
    trial_df = load_pitching_trials_wide(age_group=age_group)
    if trial_df.empty:
        raise ValueError("No trial data available.")
    if kin_metric not in trial_df.columns:
        # Try prefix normalization
        if not kin_metric.startswith("kin_") and not kin_metric.startswith("fm_"):
            for prefix in ("kin_", "fm_"):
                if f"{prefix}{kin_metric}" in trial_df.columns:
                    kin_metric = f"{prefix}{kin_metric}"
                    break
        if kin_metric not in trial_df.columns:
            raise ValueError(f"{kin_metric!r} not present in trial data.")

    ath_level = athlete_level_kinematic(
        trial_df, kin_metric, min_trials=min_trials, aggregation=aggregation,
    )

    # 2. Load athlete profile matrix
    prof = load_matrix(role=role, age_group=age_group, latest_only=True,
                        min_as_of_date=min_as_of_date,
                        exclude_mobility=exclude_mobility)
    if prof.empty:
        raise ValueError("No athlete profiles loaded.")

    # 3. Merge on athlete_uuid
    merged = ath_level.merge(prof, on="athlete_uuid", how="inner",
                              suffixes=("", "_prof"))
    if merged.empty:
        raise ValueError("No overlap between trial cohort and profile cohort.")

    # 4. Correlate the kinematic against every profile metric
    #    Role leakage filter: for a pitcher analysis, drop hit_/bat_ prefixed
    #    profile metrics (hitting_3d z-scores). For a hitter analysis, drop
    #    pitch_* z-scores. Prevents nonsense hits like 'bat_angle_at_contact'
    #    showing up as a driver of a pitching mechanic.
    def _profile_role_ok(col: str) -> bool:
        if role == "pitcher":
            return not (col.startswith("hit_") or col.startswith("bat_"))
        if role == "hitter":
            return not col.startswith("pitch_")
        return True

    rows: list[dict] = []
    target = merged[kin_metric]
    for col in metric_columns(prof):
        if col not in merged.columns:
            continue
        if not _profile_role_ok(col):
            continue
        r, p, n = _pairwise_corr(target, merged[col], "spearman")
        if r is None or n < min_n:
            continue
        rows.append({
            "assessment_metric": col,
            "domain": _classify_metric(col),
            "r": r,
            "p_value": p,
            "n": n,
        })
    if not rows:
        return {
            "kin_metric": kin_metric,
            "aggregation": aggregation,
            "n_athletes": len(merged),
            "correlations": pd.DataFrame(),
            "athlete_level": ath_level,
        }
    out = pd.DataFrame(rows)
    out["q_value"] = _bh_q_values(out["p_value"])
    out["fdr_significant"] = out["q_value"] <= fdr_alpha
    out["abs_r"] = out["r"].abs()
    out = (out.sort_values("abs_r", ascending=False)
              .drop(columns=["abs_r"])
              .reset_index(drop=True))

    return {
        "kin_metric": kin_metric,
        "aggregation": aggregation,
        "n_athletes": len(merged),
        "correlations": out,
        "athlete_level": ath_level,
    }


def stratified_chain(
    *,
    strata: list[str] | None = None,   # None → ['HIGH SCHOOL', 'COLLEGE', 'PRO', 'YOUTH']
    top_k_per_stratum: int = 5,
    min_trials_per_athlete: int = 3,
    min_n_within: int = 30,
    min_trials_agg: int = 3,
    min_n_assessment: int = 8,
    aggregation: str = "mean",
    fdr_alpha: float = 0.10,
    min_as_of_date: str | None = None,
    exclude_mobility: bool = False,
    processed_only: bool = True,
    exclude_symptomatic: bool = True,
) -> dict[str, dict]:
    """The full triangulation, per age group.

    For EACH stratum:
      1. Load trial data for that stratum only
      2. Find top-K within-athlete velocity correlates in that stratum
      3. For each of those top-K, correlate to assessments RESTRICTED to that
         same stratum's cohort

    Returns dict[stratum → dict] with:
      top_velocity_correlates (DataFrame), n_athletes_trials, n_trials,
      chain (dict[kin_metric → correlate_kinematic_to_assessments result]),
      n_athletes_profile (int)
    """
    from src.research.pitching_deep import (
        correlate_velocity_within_athlete,
    )

    if strata is None:
        strata = ["HIGH SCHOOL", "COLLEGE", "PRO", "YOUTH"]

    out: dict[str, dict] = {}
    for stratum in strata:
        # 1. Load trial data for this age group only
        trial_df = load_pitching_trials_wide(age_group=stratum)
        if trial_df.empty:
            out[stratum] = {"error": f"No trial data for {stratum!r}"}
            continue
        n_trials = len(trial_df)
        n_ath_tr = int(trial_df["athlete_uuid"].nunique())

        # 2. Top-K within-athlete velocity correlates in this stratum
        within = correlate_velocity_within_athlete(
            trial_df,
            min_trials_per_athlete=min_trials_per_athlete,
            min_n=min_n_within,
            fdr_alpha=fdr_alpha,
            processed_only=processed_only,
            exclude_symptomatic=exclude_symptomatic,
        )
        if within.empty:
            out[stratum] = {
                "n_trials": n_trials,
                "n_athletes_trials": n_ath_tr,
                "top_velocity_correlates": within,
                "chain": {},
                "n_athletes_profile": 0,
                "error": "No within-athlete velocity correlates.",
            }
            continue
        top = within.head(top_k_per_stratum)
        top_metrics = top["metric"].tolist()

        # 3. For each top-K, correlate against profile assessments in this stratum
        chain: dict[str, dict] = {}
        for m in top_metrics:
            try:
                res = correlate_kinematic_to_assessments(
                    m, role="pitcher", age_group=stratum,
                    min_trials=min_trials_agg, aggregation=aggregation,
                    min_n=min_n_assessment, fdr_alpha=fdr_alpha,
                    min_as_of_date=min_as_of_date,
                    exclude_mobility=exclude_mobility,
                )
                chain[m] = res
            except ValueError as e:
                chain[m] = {"error": str(e)}

        n_ath_profile = 0
        for m, r in chain.items():
            if "error" not in r:
                n_ath_profile = max(n_ath_profile, r.get("n_athletes", 0))

        out[stratum] = {
            "n_trials": n_trials,
            "n_athletes_trials": n_ath_tr,
            "top_velocity_correlates": top,
            "chain": chain,
            "n_athletes_profile": n_ath_profile,
        }
    return out


def batch_kinematic_drivers(
    kin_metrics: list[str],
    *,
    role: str | None = "pitcher",
    age_group: str | None = None,
    min_trials: int = 3,
    aggregation: str = "mean",
    min_n: int = 8,
    fdr_alpha: float = 0.10,
    min_as_of_date: str | None = None,
    exclude_mobility: bool = False,
) -> dict[str, dict]:
    """Run correlate_kinematic_to_assessments for a list of kinematic metrics.
    Returns dict[kin_metric → result dict]."""
    out: dict[str, dict] = {}
    for m in kin_metrics:
        try:
            out[m] = correlate_kinematic_to_assessments(
                m, role=role, age_group=age_group,
                min_trials=min_trials, aggregation=aggregation,
                min_n=min_n, fdr_alpha=fdr_alpha,
                min_as_of_date=min_as_of_date,
                exclude_mobility=exclude_mobility,
            )
        except ValueError as e:
            out[m] = {"error": str(e)}
    return out
