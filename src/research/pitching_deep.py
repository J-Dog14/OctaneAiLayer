"""
Trial-level pitching analysis. Bypasses the ai_layer.athlete_profiles
aggregation and pulls one row per throw directly from:

  public.f_pitching_trials        — JSONB `metrics` + velocity_mph + trial_index
  public.f_pitching_force_metrics — flat force-plate columns per trial

Enables four levels of velocity-correlation analysis:

  1. POOLED     — every trial across every athlete. Baseline (confounded by age/size).
  2. STRATIFIED — same, split by age group. Controls for training-age effect.
  3. WITHIN-ATHLETE — subtract each athlete's mean from every trial and correlate
                       the deltas. Answers "when THIS athlete throws harder than
                       his own average, what changes?" Controls for all fixed
                       athlete characteristics (size, mobility, strength).
  4. SESSION-LEVEL — for athletes with 2+ sessions, correlate session-mean velocity
                     with session-mean kinematics. Answers "when this athlete gained
                     mph over N months, what mechanical adaptations came with it?"

The within-athlete (fixed effects) is the most causal-adjacent because it
removes all between-athlete variance. It's what identifies mechanics that
CAUSE velocity swings within an individual pitcher.
"""
from __future__ import annotations

import json
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.db import backend_conn, query


# ──────────────────────────────────────────────────────────────────────────
# Trial-level data loader
# ──────────────────────────────────────────────────────────────────────────

def load_pitching_trials_wide(
    *,
    age_group: str | None = None,
    min_velocity: float | None = None,
    max_velocity: float | None = None,
    include_force_metrics: bool = True,
    min_as_of_date: str | None = None,
) -> pd.DataFrame:
    """Return one row per (athlete_uuid, session_date, trial_index).

    Columns:
      athlete_uuid, name, session_date, trial_index, age_at_collection,
      age_group, height, weight, handedness, velocity_mph, score,
      <all keys from f_pitching_trials.metrics JSONB, prefixed 'kin_'>,
      <all f_pitching_force_metrics columns, prefixed 'fm_'>  (if included)
    """
    where = ["1=1"]
    params: list[Any] = []
    if age_group:
        where.append("TRIM(pt.age_group) = %s")
        params.append(age_group)
    if min_velocity is not None:
        where.append("pt.velocity_mph >= %s")
        params.append(min_velocity)
    if max_velocity is not None:
        where.append("pt.velocity_mph <= %s")
        params.append(max_velocity)
    if min_as_of_date is not None:
        where.append("pt.session_date >= %s")
        params.append(min_as_of_date)

    sql = f"""
        SELECT pt.athlete_uuid,
               pt.name,
               pt.session_date,
               pt.trial_index,
               pt.age_at_collection,
               pt.age_group,
               pt.height,
               pt.weight,
               pt.handedness,
               pt.velocity_mph,
               pt.score,
               pt.metrics
        FROM public.f_pitching_trials pt
        WHERE {' AND '.join(where)}
        ORDER BY pt.athlete_uuid, pt.session_date, pt.trial_index
    """
    with backend_conn() as conn:
        trial_rows = query(conn, sql, params)

    if not trial_rows:
        return pd.DataFrame()

    # Flatten the JSONB metrics column into per-metric columns
    records: list[dict] = []
    for r in trial_rows:
        rec: dict[str, Any] = {
            "athlete_uuid": r["athlete_uuid"],
            "name": r["name"],
            "session_date": r["session_date"],
            "trial_index": r["trial_index"],
            "age_at_collection": _coerce_float(r["age_at_collection"]),
            "age_group": (r["age_group"] or "").strip() or None,
            "height": _coerce_float(r["height"]),
            "weight": _coerce_float(r["weight"]),
            "handedness": r["handedness"],
            "velocity_mph": _coerce_float(r["velocity_mph"]),
            "score": _coerce_float(r["score"]),
        }
        metrics = r["metrics"]
        if isinstance(metrics, str):
            try:
                metrics = json.loads(metrics)
            except json.JSONDecodeError:
                metrics = {}
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                rec[f"kin_{k}"] = _coerce_float(v)
        records.append(rec)
    df = pd.DataFrame(records)

    if include_force_metrics:
        fm_where = ["1=1"]
        fm_params: list[Any] = []
        if min_as_of_date is not None:
            fm_where.append("session_date >= %s")
            fm_params.append(min_as_of_date)
        with backend_conn() as conn:
            fm_rows = query(conn, f"""
                SELECT * FROM public.f_pitching_force_metrics
                WHERE {' AND '.join(fm_where)}
            """, fm_params)
        if fm_rows:
            fm_df = pd.DataFrame(fm_rows)
            # Drop columns we already have or don't need
            drop_cols = {"id", "source_system", "source_athlete_id", "owner_filename",
                          "handedness", "created_at", "processor_version",
                          "qa_flags", "qa_warnings_json"}
            fm_df = fm_df.drop(columns=[c for c in drop_cols if c in fm_df.columns])
            # Prefix all non-key columns with fm_
            key_cols = {"athlete_uuid", "session_date", "trial_index"}
            rename_map = {c: f"fm_{c}" for c in fm_df.columns if c not in key_cols}
            fm_df = fm_df.rename(columns=rename_map)
            # Coerce numeric columns
            for c in [c for c in fm_df.columns if c.startswith("fm_")]:
                fm_df[c] = pd.to_numeric(fm_df[c], errors="coerce")
            df = df.merge(fm_df, on=list(key_cols), how="left")

    # Age-group cleanup — some rows have empty strings
    df["age_group"] = df["age_group"].fillna("UNKNOWN")
    df["age_group"] = df["age_group"].apply(lambda s: s.strip() if isinstance(s, str) else s)
    return df.reset_index(drop=True)


# Metric values that are not bare numbers. Two pipeline versions wrote the
# metrics blob: the current one calls jsonlite::toJSON(auto_unbox = TRUE) and
# stores 1.23, an older one stored the same scalar as the one-element vector R
# handed it — [1.23], or {"value": 1.23}. Both are the same measurement. An
# athlete's 2025 captures came back with 7,710 values of type "other" and ZERO
# numbers, which read downstream as two sessions with no kinematics at all.
# Unwrapping them here costs nothing and means no reprocessing.
_VALUE_KEYS = ("value", "val", "v", "mean", "x")


def _unwrap(v):
    """Reduce a wrapped scalar to the scalar. Anything genuinely multi-valued
    is left alone and fails the float() below, as it should."""
    seen = 0
    while seen < 4:
        if isinstance(v, (list, tuple)):
            if len(v) != 1:
                return v
            v = v[0]
        elif isinstance(v, dict):
            if len(v) == 1:
                v = next(iter(v.values()))
            else:
                for k in _VALUE_KEYS:
                    if k in v:
                        v = v[k]
                        break
                else:
                    return v
        else:
            return v
        seen += 1
    return v


def _coerce_float(v):
    if v is None:
        return None
    v = _unwrap(v)
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
        return None if f != f else f
    except (TypeError, ValueError):
        return None


def metric_columns_pitching(df: pd.DataFrame,
                             *,
                             processed_only: bool = False,
                             exclude_symptomatic: bool = False,
                             role: str | None = "pitcher") -> list[str]:
    """Non-metadata numeric columns eligible for correlation.

    processed_only: drop kin_INCREMENT.* columns (timepoint snapshots) and keep
        only kin_PROCESSED.* summary metrics. Slashes noise dramatically.
    exclude_symptomatic: drop metrics that are OUTPUTS of throwing hard rather
        than causes — arm-decel torques, distraction forces, elbow-varus torque.
        These correlate with velocity because higher velo requires higher force,
        so their inclusion contaminates the "what causes velocity" question.
    role: 'pitcher' (default) drops hit_ / bat_ prefixed cols that leak into the
        pitching trial JSONB. 'hitter' drops pitch_ prefixed ones. None keeps
        everything.
    """
    meta = {"athlete_uuid", "name", "session_date", "trial_index",
            "age_at_collection", "age_group", "height", "weight",
            "handedness"}
    cols = [c for c in df.columns
            if c not in meta and c != "velocity_mph"
            and pd.api.types.is_numeric_dtype(df[c])]

    # ── Universal noise filter ─────────────────────────────────────────────
    # Frame indices (event counters), boolean flags, and sign indicators are
    # not physiological signals and shouldn't enter correlations.
    def _is_noise(name: str) -> bool:
        n = name.lower()
        if n.endswith("_frame") or "_frame_" in n:  # fm_fc_frame etc.
            return True
        if n.endswith("_flag") or n.endswith("_sign"):  # binary/direction
            return True
        return False
    cols = [c for c in cols if not _is_noise(c)]

    # ── Role leakage filter ────────────────────────────────────────────────
    if role == "pitcher":
        cols = [c for c in cols
                if not c.startswith("hit_")
                and not c.startswith("bat_")
                and not c.startswith("kin_HITTING_")]
    elif role == "hitter":
        cols = [c for c in cols
                if not c.startswith("pitch_")
                and not c.startswith("fm_")]  # pitching force plate

    if processed_only:
        cols = [c for c in cols
                if not c.startswith("kin_INCREMENT.")
                and not c.startswith("kin_TIME_SERIES.")]
    if exclude_symptomatic:
        symptomatic_keywords = [
            "elbow_torque", "elbow_force", "elbow_varus",
            "shoulder_torque", "shoulder_dist_force", "shoulder_force",
            "humerus_ang_acc",  # arm angular ACC (decel is passive consequence)
            "distraction",
        ]
        # Also drop the velocity-derived outcomes themselves — they either ARE
        # velocity or contain it as a linear term. Score is 50% velo + 50%
        # subjective grade; pitch_ball_release_speed IS velocity.
        derived_outcomes = {"score", "kin_score",
                             "pitch_ball_release_speed",
                             "fm_score"}
        cols = [c for c in cols if c not in derived_outcomes]
        def _is_symptom(name: str) -> bool:
            n = name.lower()
            return any(kw in n for kw in symptomatic_keywords)
        cols = [c for c in cols if not _is_symptom(c)]
    return cols


def list_trial_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Coverage per trial-metric column. Lists ALL metrics — no filters."""
    cols = metric_columns_pitching(df, processed_only=False,
                                    exclude_symptomatic=False)
    n = len(df)
    rows = []
    for c in cols:
        non_null = int(df[c].notna().sum())
        rows.append({
            "metric": c,
            "family": "kinematics" if c.startswith("kin_") else "force_metrics",
            "n_trials_non_null": non_null,
            "coverage_pct": non_null / n if n else 0.0,
            "mean": float(df[c].mean()) if non_null else None,
            "std":  float(df[c].std())  if non_null > 1 else None,
        })
    return pd.DataFrame(rows).sort_values(
        ["family", "coverage_pct"], ascending=[True, False]
    )


# ──────────────────────────────────────────────────────────────────────────
# Correlation levels — pooled / stratified / within-athlete / session-level
# ──────────────────────────────────────────────────────────────────────────

def _corr_column_vs_velocity(df: pd.DataFrame, target: str = "velocity_mph",
                              min_n: int = 20,
                              processed_only: bool = False,
                              exclude_symptomatic: bool = False) -> pd.DataFrame:
    """Spearman correlation of every numeric column against velocity_mph.
    Suppresses ConstantInputWarning."""
    cols = metric_columns_pitching(df, processed_only=processed_only,
                                    exclude_symptomatic=exclude_symptomatic)
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=stats.ConstantInputWarning)
        for c in cols:
            mask = df[c].notna() & df[target].notna()
            n = int(mask.sum())
            if n < min_n:
                continue
            x = df.loc[mask, c].astype(float).values
            y = df.loc[mask, target].astype(float).values
            if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
                continue
            try:
                r, p = stats.spearmanr(x, y)
            except Exception:
                continue
            if r != r:
                continue
            rows.append({
                "metric": c,
                "family": "kinematics" if c.startswith("kin_") else "force_metrics",
                "r": float(r),
                "p_value": float(p),
                "n": n,
            })
    return pd.DataFrame(rows)


def correlate_velocity_pooled(df: pd.DataFrame, *, min_n: int = 20,
                                fdr_alpha: float = 0.10,
                                processed_only: bool = False,
                                exclude_symptomatic: bool = False) -> pd.DataFrame:
    """Every trial across every athlete. Baseline analysis, no controls."""
    from src.research.correlations import _bh_q_values
    out = _corr_column_vs_velocity(df, min_n=min_n,
                                    processed_only=processed_only,
                                    exclude_symptomatic=exclude_symptomatic)
    if out.empty:
        return out
    out["q_value"] = _bh_q_values(out["p_value"])
    out["fdr_significant"] = out["q_value"] <= fdr_alpha
    out["abs_r"] = out["r"].abs()
    return (out.sort_values("abs_r", ascending=False)
              .drop(columns=["abs_r"])
              .reset_index(drop=True))


def correlate_velocity_stratified(
    df: pd.DataFrame, *, min_n: int = 15, fdr_alpha: float = 0.10,
    processed_only: bool = False,
    exclude_symptomatic: bool = False,
) -> dict[str, pd.DataFrame]:
    """Same as pooled but split by age_group. Returns dict[age_group → results]."""
    from src.research.correlations import _bh_q_values
    out: dict[str, pd.DataFrame] = {}
    for ag in sorted(df["age_group"].dropna().unique()):
        sub = df[df["age_group"] == ag]
        if len(sub) < min_n:
            out[ag] = pd.DataFrame()
            continue
        r = _corr_column_vs_velocity(sub, min_n=min_n,
                                       processed_only=processed_only,
                                       exclude_symptomatic=exclude_symptomatic)
        if r.empty:
            out[ag] = r
            continue
        r["q_value"] = _bh_q_values(r["p_value"])
        r["fdr_significant"] = r["q_value"] <= fdr_alpha
        r["abs_r"] = r["r"].abs()
        out[ag] = (r.sort_values("abs_r", ascending=False)
                     .drop(columns=["abs_r"])
                     .reset_index(drop=True))
    return out


def correlate_velocity_within_athlete(
    df: pd.DataFrame, *, min_trials_per_athlete: int = 3,
    min_n: int = 30, fdr_alpha: float = 0.10,
    processed_only: bool = False,
    exclude_symptomatic: bool = False,
) -> pd.DataFrame:
    """Fixed-effects (within-athlete) analysis.

    For each athlete, subtract that athlete's mean from every trial's metrics
    AND from velocity. Then correlate the residuals across ALL athletes.

    Interpretation: 'When an athlete throws harder than his own average, what
    kinematic changes are associated with that throw?'

    This removes ALL between-athlete variance — the strongest control we can
    apply with cross-sectional trial data. Requires ≥ min_trials_per_athlete
    to include an athlete (otherwise mean is meaningless).
    """
    from src.research.correlations import _bh_q_values

    # Restrict to athletes with enough trials
    counts = df.groupby("athlete_uuid").size()
    keep = counts[counts >= min_trials_per_athlete].index
    df = df[df["athlete_uuid"].isin(keep)].copy()
    if df.empty:
        return pd.DataFrame()

    # Compute athlete means and subtract
    numeric_cols = metric_columns_pitching(df,
        processed_only=processed_only,
        exclude_symptomatic=exclude_symptomatic) + ["velocity_mph"]
    means = df.groupby("athlete_uuid")[numeric_cols].transform("mean")
    residuals = df[numeric_cols] - means
    residuals["athlete_uuid"] = df["athlete_uuid"].values
    # Now correlate residuals of each metric against velocity residual
    rows = []
    y = residuals["velocity_mph"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=stats.ConstantInputWarning)
        for c in metric_columns_pitching(df,
                processed_only=processed_only,
                exclude_symptomatic=exclude_symptomatic):
            mask = residuals[c].notna() & y.notna()
            n = int(mask.sum())
            if n < min_n:
                continue
            x = residuals.loc[mask, c].astype(float).values
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
                "metric": c,
                "family": "kinematics" if c.startswith("kin_") else "force_metrics",
                "r": float(r),
                "p_value": float(p),
                "n_trials": n,
                "n_athletes": int(df["athlete_uuid"].nunique()),
            })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["q_value"] = _bh_q_values(out["p_value"])
    out["fdr_significant"] = out["q_value"] <= fdr_alpha
    out["abs_r"] = out["r"].abs()
    return (out.sort_values("abs_r", ascending=False)
              .drop(columns=["abs_r"])
              .reset_index(drop=True))


def correlate_velocity_session_level(
    df: pd.DataFrame, *, min_sessions_per_athlete: int = 2,
    min_n: int = 20, fdr_alpha: float = 0.10,
    processed_only: bool = False,
    exclude_symptomatic: bool = False,
) -> pd.DataFrame:
    """Session-to-session within-athlete change analysis.

    For each (athlete, session), compute mean velocity + mean of every metric.
    Then within-athlete-across-sessions residualization: subtract athlete's
    average session-mean from each session-mean, correlate.

    Answers: 'When this athlete's session-average velocity changes across time,
    what kinematic session-averages change with it?'

    Requires athletes with ≥2 sessions to contribute anything meaningful.
    """
    from src.research.correlations import _bh_q_values
    numeric = metric_columns_pitching(df) + ["velocity_mph"]

    # Session-level aggregation
    sess = df.groupby(["athlete_uuid", "session_date"])[numeric].mean().reset_index()
    counts = sess.groupby("athlete_uuid").size()
    keep = counts[counts >= min_sessions_per_athlete].index
    sess = sess[sess["athlete_uuid"].isin(keep)]
    if sess.empty:
        return pd.DataFrame()

    means = sess.groupby("athlete_uuid")[numeric].transform("mean")
    residuals = sess[numeric] - means
    residuals["athlete_uuid"] = sess["athlete_uuid"].values

    rows = []
    y = residuals["velocity_mph"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=stats.ConstantInputWarning)
        for c in metric_columns_pitching(df,
                processed_only=processed_only,
                exclude_symptomatic=exclude_symptomatic):
            mask = residuals[c].notna() & y.notna()
            n = int(mask.sum())
            if n < min_n:
                continue
            x = residuals.loc[mask, c].astype(float).values
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
                "metric": c,
                "family": "kinematics" if c.startswith("kin_") else "force_metrics",
                "r": float(r),
                "p_value": float(p),
                "n_session_pairs": n,
                "n_athletes": int(sess["athlete_uuid"].nunique()),
            })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["q_value"] = _bh_q_values(out["p_value"])
    out["fdr_significant"] = out["q_value"] <= fdr_alpha
    out["abs_r"] = out["r"].abs()
    return (out.sort_values("abs_r", ascending=False)
              .drop(columns=["abs_r"])
              .reset_index(drop=True))


# ──────────────────────────────────────────────────────────────────────────
# Backwards chain: for the top velocity correlates, find what correlates
# with THEM (within-athlete)
# ──────────────────────────────────────────────────────────────────────────

def backwards_chain(
    df: pd.DataFrame,
    top_metrics: list[str],
    *,
    min_trials_per_athlete: int = 3,
    min_n: int = 30,
    fdr_alpha: float = 0.10,
    top_k_per_target: int = 15,
    processed_only: bool = False,
    exclude_symptomatic: bool = False,
) -> dict[str, pd.DataFrame]:
    """For each metric in top_metrics, run within-athlete correlate against
    every OTHER metric in the trial data. Returns dict[metric → results].

    Use this after velocity-search: 'trunk velocity correlates with velocity
    at r=0.75 within-athlete. What correlates with trunk velocity?' →
    backwards_chain(['kin_trunk_ang_vel_max']) → top-k drivers of trunk velo.
    """
    from src.research.correlations import _bh_q_values

    counts = df.groupby("athlete_uuid").size()
    keep = counts[counts >= min_trials_per_athlete].index
    df = df[df["athlete_uuid"].isin(keep)].copy()

    all_cols = metric_columns_pitching(df,
        processed_only=processed_only,
        exclude_symptomatic=exclude_symptomatic)
    numeric = all_cols + ["velocity_mph"]
    means = df.groupby("athlete_uuid")[numeric].transform("mean")
    residuals = df[numeric] - means

    out: dict[str, pd.DataFrame] = {}
    for target in top_metrics:
        if target not in residuals.columns:
            out[target] = pd.DataFrame()
            continue
        rows = []
        y = residuals[target]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=stats.ConstantInputWarning)
            for c in all_cols:
                if c == target:
                    continue
                mask = residuals[c].notna() & y.notna()
                n = int(mask.sum())
                if n < min_n:
                    continue
                x = residuals.loc[mask, c].astype(float).values
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
                    "metric": c,
                    "family": "kinematics" if c.startswith("kin_") else "force_metrics",
                    "r": float(r),
                    "p_value": float(p),
                    "n_trials": n,
                })
        if not rows:
            out[target] = pd.DataFrame()
            continue
        results = pd.DataFrame(rows)
        results["q_value"] = _bh_q_values(results["p_value"])
        results["fdr_significant"] = results["q_value"] <= fdr_alpha
        results["abs_r"] = results["r"].abs()
        out[target] = (results.sort_values("abs_r", ascending=False)
                                 .drop(columns=["abs_r"])
                                 .head(top_k_per_target)
                                 .reset_index(drop=True))
    return out
