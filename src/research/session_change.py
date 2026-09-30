"""
Session-to-session change analysis.

For athletes with ≥2 pitching-3D sessions AND ≥2 profile assessments over
time, compute the deltas in:

  - Velocity (mean session mph)
  - Any pitching kinematic / force metric (mean session values)
  - Every profile Z-score (from the profile closest in time to each session)

Then across athletes, correlate assessment deltas with kinematic/velocity
deltas.

This answers the user's key question: "when an athlete's front-leg GRF
metrics improved between two sessions, what physical assessment metric
changed with it?" That's the closest thing to a natural experiment we have
without randomized intervention data.

Interpretation:
  - Positive delta_r means: when the athlete's assessment X went UP between
    sessions, their kinematic Y also went UP. Consistent direction across
    athletes → the two adaptations tend to co-occur.
  - Cannot prove causation (both could be driven by a hidden third variable
    like training block), but a strong consistent delta correlation is much
    stronger evidence than cross-sectional correlation.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.db import backend_conn, query
from src.research.correlations import _bh_q_values
from src.research.pitching_deep import (
    load_pitching_trials_wide,
    metric_columns_pitching,
)


def load_profiles_all_dates() -> pd.DataFrame:
    """Every profile row, wide-format Z-scores. Multiple rows per athlete OK."""
    sql = """
        SELECT p.id, p.athlete_uuid, p.as_of_date,
               p.age_group, p.z_scores,
               d.name
        FROM ai_layer.athlete_profiles p
        JOIN analytics.d_athletes d USING (athlete_uuid)
        ORDER BY p.athlete_uuid, p.as_of_date
    """
    with backend_conn() as conn:
        rows = query(conn, sql)
    if not rows:
        return pd.DataFrame()
    records = []
    for r in rows:
        z = r["z_scores"] or {}
        rec = {
            "profile_id": r["id"],
            "athlete_uuid": r["athlete_uuid"],
            "name": r["name"],
            "as_of_date": r["as_of_date"],
            "age_group": r["age_group"],
        }
        for k, v in z.items():
            try:
                rec[k] = float(v) if v is not None else None
            except (TypeError, ValueError):
                rec[k] = None
        records.append(rec)
    return pd.DataFrame(records)


def _closest_profile_per_session(
    profile_df: pd.DataFrame,
    athlete_uuid: str,
    session_date,
) -> pd.Series | None:
    """Return the profile row closest to a given session date for one athlete."""
    subset = profile_df[profile_df["athlete_uuid"] == athlete_uuid]
    if subset.empty:
        return None
    subset = subset.copy()
    subset["day_diff"] = (pd.to_datetime(subset["as_of_date"])
                          - pd.to_datetime(session_date)).dt.days.abs()
    return subset.sort_values("day_diff").iloc[0]


def build_session_delta_frame(
    *,
    min_span_days: int = 30,
    max_span_days: int = 730,
    age_group: str | None = None,
    aggregation: str = "mean",
    metric_prefixes: tuple[str, ...] = ("kin_PROCESSED.", "fm_"),
    max_profile_gap_days: int = 90,
) -> pd.DataFrame:
    """One row per athlete with first-to-last session deltas + profile deltas.

    Columns:
      athlete_uuid, name, age_group,
      session1_date, session2_date, span_days,
      delta_velocity_mph,
      <kin_or_fm_metric>_delta   (one per included metric),
      <profile_metric>_delta      (one per profile metric).
    """
    trial_df = load_pitching_trials_wide(age_group=age_group)
    if trial_df.empty:
        return pd.DataFrame()

    # Restrict to trial metrics we care about (PROCESSED kinematics + force_metrics)
    trial_metric_cols = [c for c in trial_df.columns
                         if any(c.startswith(p) for p in metric_prefixes)]

    # Session-mean per athlete-session
    numeric_agg_cols = ["velocity_mph"] + trial_metric_cols
    if aggregation == "median":
        sess = trial_df.groupby(["athlete_uuid", "name", "session_date"])[numeric_agg_cols].median().reset_index()
    else:
        sess = trial_df.groupby(["athlete_uuid", "name", "session_date"])[numeric_agg_cols].mean().reset_index()

    # Athletes with ≥2 sessions
    counts = sess.groupby("athlete_uuid").size()
    keep = counts[counts >= 2].index
    sess = sess[sess["athlete_uuid"].isin(keep)].sort_values(["athlete_uuid", "session_date"])
    if sess.empty:
        return pd.DataFrame()

    # For each athlete, take FIRST and LAST session within the span window
    profile_df = load_profiles_all_dates()

    rows: list[dict] = []
    for uuid, g in sess.groupby("athlete_uuid"):
        first = g.iloc[0]
        last = g.iloc[-1]
        span = (pd.to_datetime(last["session_date"])
                 - pd.to_datetime(first["session_date"])).days
        if span < min_span_days or span > max_span_days:
            continue

        rec: dict[str, Any] = {
            "athlete_uuid": uuid,
            "name": first["name"],
            "session1_date": first["session_date"],
            "session2_date": last["session_date"],
            "span_days": span,
        }
        # Kinematic + FM deltas
        rec["delta_velocity_mph"] = float(last["velocity_mph"] - first["velocity_mph"]) \
            if pd.notna(last["velocity_mph"]) and pd.notna(first["velocity_mph"]) else None
        for c in trial_metric_cols:
            v1 = first[c]
            v2 = last[c]
            if pd.notna(v1) and pd.notna(v2):
                rec[f"{c}_delta"] = float(v2 - v1)
            else:
                rec[f"{c}_delta"] = None

        # Profile deltas + baselines
        #
        # DELTA (two-profile) case: requires two DISTINCT profile snapshots,
        # each within max_profile_gap_days of one of the two sessions.
        # Writes <metric>_delta cols. Rare in practice — most athletes have
        # only one profile.
        #
        # BASELINE (one-profile) case: uses the profile closest to session1
        # (or session2 if session1 has none in range). Writes <metric>_baseline
        # cols. Answers 'athletes who came in with X gained more velocity'.
        # Works even when the athlete has only ever been profiled once.
        _has_delta_profile = False
        _has_baseline_profile = False
        if not profile_df.empty:
            p1 = _closest_profile_per_session(profile_df, uuid, first["session_date"])
            p2 = _closest_profile_per_session(profile_df, uuid, last["session_date"])
            _META = {"profile_id", "athlete_uuid", "name",
                     "as_of_date", "age_group", "day_diff"}

            # Baseline: prefer p1 within range; fall back to p2 within range
            baseline_p = None
            if p1 is not None and p1["day_diff"] <= max_profile_gap_days:
                baseline_p = p1
            elif p2 is not None and p2["day_diff"] <= max_profile_gap_days:
                baseline_p = p2
            if baseline_p is not None:
                _has_baseline_profile = True
                rec["age_group"] = baseline_p.get("age_group") or rec.get("age_group") or first.get("age_group")
                for k in set(baseline_p.index) - _META:
                    v = baseline_p.get(k)
                    if pd.notna(v):
                        try:
                            rec[f"{k}_baseline"] = float(v)
                        except (TypeError, ValueError):
                            rec[f"{k}_baseline"] = None
                    else:
                        rec[f"{k}_baseline"] = None

            # Delta (two distinct profiles within range)
            if (p1 is not None and p2 is not None
                and p1["day_diff"] <= max_profile_gap_days
                and p2["day_diff"] <= max_profile_gap_days
                and p1["profile_id"] != p2["profile_id"]
            ):
                _has_delta_profile = True
                rec["age_group"] = p2.get("age_group") or rec.get("age_group") or first.get("age_group")
                p1_keys = set(p1.index) - _META
                p2_keys = set(p2.index) - _META
                for k in p1_keys | p2_keys:
                    v1 = p1.get(k)
                    v2 = p2.get(k)
                    if pd.notna(v1) and pd.notna(v2):
                        try:
                            rec[f"{k}_delta"] = float(v2) - float(v1)
                        except (TypeError, ValueError):
                            rec[f"{k}_delta"] = None
                    else:
                        rec[f"{k}_delta"] = None
        rec["_has_delta_profile"] = _has_delta_profile
        rec["_has_baseline_profile"] = _has_baseline_profile
        rows.append(rec)

    return pd.DataFrame(rows)


_ASSESSMENT_PREFIXES = ("mob_", "screen_", "proteus_", "fp_", "rs_", "arm_", "cb_")
_KINEMATIC_PREFIXES = ("kin_", "fm_", "pitch_", "hit_")

# Populated by correlate_deltas as a side-channel diagnostic so the CLI can
# report how sparsity filtered the predictor pool.
_last_predictor_density_drop: dict = {"before": 0, "after": 0, "min_n": 0}


def _is_assessment_delta(col: str) -> bool:
    """A delta column is an assessment predictor if its base name has an
    assessment prefix (mob_/screen_/proteus_/etc.)"""
    if not col.endswith("_delta"):
        return False
    base = col[:-len("_delta")]
    return any(base.startswith(p) for p in _ASSESSMENT_PREFIXES)


def _is_assessment_baseline(col: str) -> bool:
    """A baseline column is an assessment predictor if its base name has an
    assessment prefix and it ends with _baseline."""
    if not col.endswith("_baseline"):
        return False
    base = col[:-len("_baseline")]
    return any(base.startswith(p) for p in _ASSESSMENT_PREFIXES)


def correlate_deltas(
    delta_df: pd.DataFrame,
    *,
    targets: list[str] | None = None,   # None → derive from delta_df
    min_n: int = 6,
    fdr_alpha: float = 0.10,
    predictor_domain: str = "all",
    # 'all' | 'assessment_delta' | 'assessment_baseline' | 'kinematic_only'
    # NOTE: 'assessment_only' is a deprecated alias for 'assessment_delta'.
) -> dict[str, pd.DataFrame]:
    """For each target (a *_delta column), correlate against a chosen predictor
    pool. Returns dict[target → results DataFrame].

    Predictor pools:
      - 'all'                → every *_delta col except the target (kin+fm+
                               assessment deltas)
      - 'kinematic_only'     → *_delta cols that are NOT assessment
      - 'assessment_delta'   → only assessment *_delta cols (needs athletes
                               with 2+ profiles)
      - 'assessment_baseline'→ only assessment *_baseline cols (needs 1+
                               profile per athlete — usually the winning mode)
    """
    all_delta_cols = [c for c in delta_df.columns if c.endswith("_delta")]
    all_baseline_cols = [c for c in delta_df.columns if c.endswith("_baseline")]

    # Filter predictor pool based on domain choice
    if predictor_domain in ("assessment_only", "assessment_delta"):
        delta_cols = [c for c in all_delta_cols if _is_assessment_delta(c)]
    elif predictor_domain == "assessment_baseline":
        delta_cols = [c for c in all_baseline_cols if _is_assessment_baseline(c)]
    elif predictor_domain == "kinematic_only":
        delta_cols = [c for c in all_delta_cols if not _is_assessment_delta(c)]
    else:
        delta_cols = all_delta_cols

    # DENSITY FILTER — drop predictor cols that don't have min_n non-null values.
    # Athlete profiles are wildly sparse across the union of z_score keys
    # (a HS athlete's mob_* keys don't overlap a pro's), so most predictor cols
    # never had enough valid rows to correlate. Reporting the drop count makes
    # this visible instead of returning silent zeros.
    _n_before = len(delta_cols)
    delta_cols = [c for c in delta_cols if int(delta_df[c].notna().sum()) >= min_n]
    _n_after = len(delta_cols)
    if _n_before and (_n_before - _n_after) > 0:
        # attach diag info via a module-level sink readable from the CLI
        _last_predictor_density_drop["before"] = _n_before
        _last_predictor_density_drop["after"] = _n_after
        _last_predictor_density_drop["min_n"] = min_n

    if targets is None:
        # Default targets = velocity + top GRF metrics
        # Note: derive targets from ALL delta cols regardless of predictor filter
        default = ["delta_velocity_mph"]
        for c in all_delta_cols:
            n = c.lower()
            if "grf" in n or "vertical" in n or "impulse" in n or "rfd" in n:
                default.append(c)
        targets = list(dict.fromkeys(default))  # dedupe preserving order

    out: dict[str, pd.DataFrame] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=stats.ConstantInputWarning)
        for tgt in targets:
            if tgt not in delta_df.columns:
                out[tgt] = pd.DataFrame()
                continue
            rows = []
            y = delta_df[tgt]
            for c in delta_cols:
                if c == tgt:
                    continue
                mask = delta_df[c].notna() & y.notna()
                n = int(mask.sum())
                if n < min_n:
                    continue
                x = delta_df.loc[mask, c].astype(float).values
                yv = y[mask].astype(float).values
                if float(np.std(x)) == 0.0 or float(np.std(yv)) == 0.0:
                    continue
                try:
                    r, p = stats.spearmanr(x, yv)
                except Exception:
                    continue
                if r != r:
                    continue
                rows.append({
                    "delta_metric": c,
                    "r": float(r),
                    "p_value": float(p),
                    "n": n,
                })
            if not rows:
                out[tgt] = pd.DataFrame()
                continue
            res = pd.DataFrame(rows)
            res["q_value"] = _bh_q_values(res["p_value"])
            res["fdr_significant"] = res["q_value"] <= fdr_alpha
            res["abs_r"] = res["r"].abs()
            out[tgt] = (res.sort_values("abs_r", ascending=False)
                          .drop(columns=["abs_r"])
                          .reset_index(drop=True))
    return out


def cross_target_predictor_summary(
    per_target_results: dict[str, pd.DataFrame],
    *,
    fdr_alpha: float = 0.10,
    min_hits: int = 2,
) -> pd.DataFrame:
    """For each unique predictor, tally how many force-metric targets it hits
    with FDR significance and consistent direction.

    Answers: 'across ALL the force-metric targets we care about, which single
    assessment/kinematic delta shows up consistently as a driver?'
    """
    hits: dict[str, dict] = {}
    for tgt, res in per_target_results.items():
        if res.empty:
            continue
        sig = res[res["fdr_significant"] == True]
        for _, row in sig.iterrows():
            pred = row["delta_metric"]
            if pred not in hits:
                hits[pred] = {
                    "predictor": pred,
                    "n_targets_hit": 0,
                    "n_positive": 0,
                    "n_negative": 0,
                    "rs": [],
                    "targets_hit": [],
                }
            hits[pred]["n_targets_hit"] += 1
            r = float(row["r"])
            hits[pred]["rs"].append(r)
            hits[pred]["targets_hit"].append(tgt)
            if r > 0:
                hits[pred]["n_positive"] += 1
            else:
                hits[pred]["n_negative"] += 1
    rows = []
    for pred, info in hits.items():
        if info["n_targets_hit"] < min_hits:
            continue
        rs = info["rs"]
        if _is_assessment_baseline(pred):
            ptype = "assessment_baseline"
        elif _is_assessment_delta(pred):
            ptype = "assessment_delta"
        else:
            ptype = "kinematic/force"
        rows.append({
            "predictor": pred,
            "predictor_type": ptype,
            "n_targets_hit": info["n_targets_hit"],
            "direction_consistency": max(info["n_positive"], info["n_negative"]) / info["n_targets_hit"],
            "mean_r": round(sum(rs) / len(rs), 3),
            "min_r": round(min(rs), 3),
            "max_r": round(max(rs), 3),
            "example_targets": ", ".join(info["targets_hit"][:3]),
        })
    if not rows:
        return pd.DataFrame()
    return (pd.DataFrame(rows)
              .sort_values(["n_targets_hit", "direction_consistency"],
                            ascending=[False, False])
              .reset_index(drop=True))
