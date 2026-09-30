"""
Longitudinal analysis — for athletes with ≥2 assessment profiles over time.

Pre/post metric change ("delta") computation, and correlation of those deltas
with the *program content* prescribed between the two dates.

Answers questions like:
    - Which prep exercise families correlate with T-spine mobility improvement?
    - Do athletes who got heavier lift templates gain more Proteus power?
    - Does prescribing more waterbag work improve pelvic-trunk dissociation?

Notes:
    - This is CORRELATIONAL, not causal. With n=5-15 athletes in most delta
      buckets, we can only surface plausible hypotheses, not prove them.
    - Coach selection bias is huge: athletes who *needed* T-spine work got
      Bretzel 2.0. Their T-spine improved. That's regression to the mean +
      selection — not necessarily "Bretzel 2.0 works." Interpret carefully.
    - Time-window matters: if delta is measured over 2 months and program is
      12 months, we're only crediting ~1/6 of the prescribed volume.
"""
from __future__ import annotations

import pandas as pd

from src.db import backend_conn, query
from src.research.profile_matrix import _classify_metric


# ──────────────────────────────────────────────────────────────────────────
# Serial-profile detection
# ──────────────────────────────────────────────────────────────────────────

def athletes_with_serial_profiles(min_profiles: int = 2) -> pd.DataFrame:
    """List athletes with at least N distinct athlete_profile rows.

    Returns DataFrame with columns: athlete_uuid, name, n_profiles, first_date,
    last_date, span_days.
    """
    with backend_conn() as conn:
        rows = query(conn, """
            SELECT p.athlete_uuid,
                   d.name,
                   COUNT(*)::int         AS n_profiles,
                   MIN(p.as_of_date)     AS first_date,
                   MAX(p.as_of_date)     AS last_date
            FROM ai_layer.athlete_profiles p
            JOIN analytics.d_athletes d USING (athlete_uuid)
            GROUP BY p.athlete_uuid, d.name
            HAVING COUNT(*) >= %s
            ORDER BY n_profiles DESC, name
        """, [min_profiles])
    if not rows:
        return pd.DataFrame(columns=["athlete_uuid", "name", "n_profiles",
                                     "first_date", "last_date", "span_days"])
    df = pd.DataFrame(rows)
    df["span_days"] = (pd.to_datetime(df["last_date"])
                        - pd.to_datetime(df["first_date"])).dt.days
    return df


def load_deltas(
    *,
    role: str | None = None,
    min_span_days: int = 30,
    max_span_days: int = 365,
) -> pd.DataFrame:
    """Compute first-to-last Z-score deltas per athlete.

    Returns wide-format DataFrame:
      athlete_uuid, name, first_date, last_date, span_days,
      <metric>_first, <metric>_last, <metric>_delta   (one triple per metric)

    Only includes athletes with ≥2 profiles and span within [min_span_days,
    max_span_days] (to filter re-profile-same-day noise + very old data).
    """
    with backend_conn() as conn:
        rows = query(conn, """
            SELECT p.athlete_uuid,
                   d.name,
                   d.has_pitching_data,
                   d.has_hitting_data,
                   p.as_of_date,
                   p.z_scores
            FROM ai_layer.athlete_profiles p
            JOIN analytics.d_athletes d USING (athlete_uuid)
            ORDER BY p.athlete_uuid, p.as_of_date
        """)
    if not rows:
        return pd.DataFrame()

    # Group by athlete, pick first + last profile
    from collections import defaultdict
    by_ath: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_ath[r["athlete_uuid"]].append(r)

    records: list[dict] = []
    for uuid, profs in by_ath.items():
        if len(profs) < 2:
            continue
        first = profs[0]
        last  = profs[-1]
        span = (pd.to_datetime(last["as_of_date"])
                 - pd.to_datetime(first["as_of_date"])).days
        if span < min_span_days or span > max_span_days:
            continue

        # Role filter
        if role and role != "all":
            has_p = bool(first["has_pitching_data"])
            has_h = bool(first["has_hitting_data"])
            if role == "pitcher" and not has_p:
                continue
            if role == "hitter" and not has_h:
                continue
            if role == "both" and not (has_p and has_h):
                continue

        z_first = first["z_scores"] or {}
        z_last  = last["z_scores"]  or {}
        rec: dict = {
            "athlete_uuid": uuid,
            "name": first["name"],
            "first_date": first["as_of_date"],
            "last_date":  last["as_of_date"],
            "span_days": span,
        }
        # Union of metrics
        all_metrics = set(z_first) | set(z_last)
        for m in all_metrics:
            f = z_first.get(m)
            l = z_last.get(m)
            rec[f"{m}_first"] = _coerce_float(f)
            rec[f"{m}_last"]  = _coerce_float(l)
            if f is not None and l is not None:
                try:
                    rec[f"{m}_delta"] = float(l) - float(f)
                except (TypeError, ValueError):
                    rec[f"{m}_delta"] = None
            else:
                rec[f"{m}_delta"] = None
        records.append(rec)

    return pd.DataFrame(records).reset_index(drop=True)


def _coerce_float(v):
    if v is None:
        return None
    try:
        f = float(v)
        if f != f:
            return None
        return f
    except (TypeError, ValueError):
        return None


def delta_summary(delta_df: pd.DataFrame) -> pd.DataFrame:
    """Per-metric summary of pre/post changes: n, mean delta, direction.

    Positive delta_mean means the population improved on that metric between
    the two dates (Z-score went up = better than population). Watch for
    regression-to-the-mean effects when interpreting.
    """
    delta_cols = [c for c in delta_df.columns if c.endswith("_delta")]
    rows = []
    for c in delta_cols:
        metric = c[:-len("_delta")]
        s = delta_df[c].dropna()
        if len(s) < 3:
            continue
        rows.append({
            "metric": metric,
            "domain": _classify_metric(metric),
            "n": len(s),
            "delta_mean": float(s.mean()),
            "delta_median": float(s.median()),
            "delta_std": float(s.std()) if len(s) > 1 else None,
            "pct_improved": float((s > 0).mean()),
        })
    return (pd.DataFrame(rows)
              .sort_values(["domain", "delta_mean"], ascending=[True, False])
              .reset_index(drop=True))
