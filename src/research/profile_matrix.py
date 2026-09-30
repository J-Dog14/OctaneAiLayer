"""
Extract the wide-format athlete × metric matrix from ai_layer.athlete_profiles.

This is the base data structure every other research module consumes. Each row
is one athlete (optionally one athlete × as_of_date if serial mode), columns are:

  - Metadata: athlete_uuid, name, age_group, role, as_of_date, session dates per modality
  - Z-scores: one column per metric key, prefixed by domain (e.g., 'mob_...', 'pitch_...')
  - Raw values: same metrics but un-normalized (optional, when the analyst wants raw)

Typical use:

    from src.research.profile_matrix import load_matrix
    df = load_matrix(role="pitcher", latest_only=True)
    # df has ~40 rows × ~80 columns for pitchers with latest profile

Filters supported: role, age_group, min_metrics (drop athletes with too much
missing data), require_modalities (only include athletes with data in every
listed modality).
"""
from __future__ import annotations

import json
from datetime import date
from typing import Any

import pandas as pd

from src.db import backend_conn, query


# ──────────────────────────────────────────────────────────────────────────
# Metric domain prefix map — matches how metrics_spec.py names things
# ORDER MATTERS: longest prefix wins (checked first). E.g. `proteus_pitcher_`
# must be checked before `proteus_`. Dict iteration order preserves insertion
# order in Python 3.7+.
# ──────────────────────────────────────────────────────────────────────────
_DOMAIN_PREFIXES: dict[str, str] = {
    "proteus_pitcher_":  "proteus_pitcher",
    "proteus_hitter_":   "proteus_hitter",
    "screen_dj_":        "athletic_screen_dj",
    "screen_cmj_":       "athletic_screen_cmj",
    "screen_ppu_":       "athletic_screen_ppu",
    "screen_slv_":       "athletic_screen_slv",
    "screen_":           "athletic_screen",   # catch-all
    "mob_":              "mobility",
    "pitch_":            "pitching_3d",
    "hit_":              "hitting_3d",
    "fp_":               "force_plate",
    "rs_":               "readiness_screen",
    "arm_":              "arm_action",
    "cb_":               "curveball_test",
}


def _classify_metric(key: str) -> str:
    """Map a metric key to its assessment-domain label."""
    for prefix, domain in _DOMAIN_PREFIXES.items():
        if key.startswith(prefix):
            return domain
    return "other"


def load_matrix(
    *,
    role: str | None = None,             # 'pitcher' | 'hitter' | 'both' | None (all)
    age_group: str | None = None,        # 'YOUTH' | 'HIGH SCHOOL' | 'COLLEGE' | 'PRO'
    latest_only: bool = True,            # one row per athlete (most recent profile)
    require_modalities: list[str] | None = None,  # only athletes with data in ALL listed domains
    min_non_null_metrics: int = 5,       # drop athletes with fewer non-null Z-scores
    include_raw: bool = False,           # add un-normalized values as _raw suffix cols
    min_as_of_date: str | date | None = None,  # e.g. '2024-01-01' to exclude old mobility-scale data
    exclude_mobility: bool = False,      # dropped all mob_* columns (scale changes make them unreliable across time)
) -> pd.DataFrame:
    """Return a wide-format DataFrame — athletes × metrics.

    Column layout:
      Metadata (always):
        athlete_uuid, name, age_group, role, as_of_date,
        has_pitching_data, has_hitting_data, source_dates (dict)
      Z-score columns:
        one per metric key present in any athlete_profiles row
      Raw columns (only when include_raw=True):
        <metric_key>_raw
    """
    where = ["1=1"]
    params: list[Any] = []
    if age_group:
        where.append("TRIM(d.age_group) = %s")
        params.append(age_group)
    if min_as_of_date is not None:
        where.append("p.as_of_date >= %s")
        params.append(min_as_of_date)

    order_slice = (
        "ORDER BY p.athlete_uuid, p.as_of_date DESC"
        if latest_only
        else "ORDER BY p.athlete_uuid, p.as_of_date"
    )
    distinct_prefix = "DISTINCT ON (p.athlete_uuid)" if latest_only else ""

    sql = f"""
        SELECT {distinct_prefix}
               p.id                AS profile_id,
               p.athlete_uuid,
               p.as_of_date,
               p.age_group,
               p.raw_values,
               p.z_scores,
               p.source_dates,
               d.name,
               d.has_pitching_data,
               d.has_hitting_data,
               d.email
        FROM ai_layer.athlete_profiles p
        JOIN analytics.d_athletes d USING (athlete_uuid)
        WHERE {' AND '.join(where)}
        {order_slice}
    """

    with backend_conn() as conn:
        rows = query(conn, sql, params)

    if not rows:
        return pd.DataFrame()

    # ── Build wide format
    records: list[dict[str, Any]] = []
    all_metric_keys: set[str] = set()

    for r in rows:
        z = r["z_scores"] or {}
        raw = r["raw_values"] or {}
        # JSONB → dict already via psycopg2 RealDictCursor
        rec: dict[str, Any] = {
            "profile_id": r["profile_id"],
            "athlete_uuid": r["athlete_uuid"],
            "name": r["name"],
            "age_group": r["age_group"],
            "as_of_date": r["as_of_date"],
            "has_pitching_data": bool(r["has_pitching_data"]),
            "has_hitting_data": bool(r["has_hitting_data"]),
            "source_dates": r["source_dates"] or {},
        }
        # Derive role for role-filter
        if rec["has_pitching_data"] and rec["has_hitting_data"]:
            rec["role"] = "both"
        elif rec["has_pitching_data"]:
            rec["role"] = "pitcher"
        elif rec["has_hitting_data"]:
            rec["role"] = "hitter"
        else:
            rec["role"] = "unknown"

        for k, v in z.items():
            rec[k] = _coerce_float(v)
            all_metric_keys.add(k)
        if include_raw:
            for k, v in raw.items():
                rec[f"{k}_raw"] = _coerce_float(v)
        records.append(rec)

    df = pd.DataFrame.from_records(records)

    # ── Role filter (after we've derived role for each row)
    if role and role != "all":
        if role == "both":
            df = df[df["role"] == "both"]
        else:
            df = df[df["role"].isin([role, "both"])]

    # ── require_modalities: only athletes with any data in each listed domain
    if require_modalities:
        for domain in require_modalities:
            cols_in_domain = [c for c in df.columns
                              if c in all_metric_keys and _classify_metric(c) == domain]
            if not cols_in_domain:
                # No columns in this domain exist at all — return empty
                return df.iloc[0:0]
            has_any = df[cols_in_domain].notna().any(axis=1)
            df = df[has_any]

    # ── Optional: drop mobility columns entirely
    # Mobility protocol changed over the years (1-3 scale → real measurements,
    # tests added/removed) so pooled Z-scores across time are noisy.
    if exclude_mobility:
        mob_cols = [c for c in df.columns
                     if c in all_metric_keys and _classify_metric(c) == "mobility"]
        df = df.drop(columns=mob_cols, errors="ignore")
        all_metric_keys = {k for k in all_metric_keys if _classify_metric(k) != "mobility"}

    # ── min_non_null_metrics guard
    metric_cols = [c for c in df.columns if c in all_metric_keys]
    if metric_cols:
        n_non_null = df[metric_cols].notna().sum(axis=1)
        df = df[n_non_null >= min_non_null_metrics]

    return df.reset_index(drop=True)


def _coerce_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        # Guard against NaN → keep as None for cleaner pandas handling
        if f != f:  # NaN check
            return None
        return f
    except (TypeError, ValueError):
        return None


def metric_columns(df: pd.DataFrame) -> list[str]:
    """Metric columns only (drop metadata)."""
    meta = {"profile_id", "athlete_uuid", "name", "age_group", "as_of_date",
            "has_pitching_data", "has_hitting_data", "source_dates", "role"}
    return [c for c in df.columns if c not in meta and not c.endswith("_raw")]


def columns_by_domain(df: pd.DataFrame) -> dict[str, list[str]]:
    """Group metric columns by their assessment domain."""
    out: dict[str, list[str]] = {}
    for c in metric_columns(df):
        d = _classify_metric(c)
        out.setdefault(d, []).append(c)
    return out


def coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-metric coverage — non-null count and % across the athlete cohort."""
    metric_cols = metric_columns(df)
    n_total = len(df)
    rows = []
    for c in metric_cols:
        n_non_null = int(df[c].notna().sum())
        rows.append({
            "metric": c,
            "domain": _classify_metric(c),
            "n_non_null": n_non_null,
            "coverage_pct": (n_non_null / n_total) if n_total else 0.0,
            "mean_z": float(df[c].mean()) if n_non_null else None,
            "std_z":  float(df[c].std())  if n_non_null > 1 else None,
        })
    return pd.DataFrame(rows).sort_values(["domain", "coverage_pct"],
                                          ascending=[True, False])
