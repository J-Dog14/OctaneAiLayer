"""
Program-response analysis — cross-reference prescribed exercises with
subsequent metric changes.

Pipeline:
  1. Compute per-athlete metric deltas over a time window (longitudinal.load_deltas)
  2. For each athlete, aggregate the exercise prescriptions they received
     between first_date and last_date
  3. Bucket exercises by movement pattern (reuse eval_deep_report's classifier)
  4. For each (pattern, metric) pair: correlate volume-of-pattern with
     metric-delta across athletes

Read as: "athletes who got more X saw more Y-improvement." NOT causal —
coaches select exercises based on the same deficits we're measuring, so
selection bias inflates apparent effect. Interpret as hypothesis-generating.

Example finding this might surface:
    - Waterbag drill volume ↔ pelvic-trunk dissociation improvement (r=0.52, n=12)
    - PUM tool prescriptions ↔ shoulder scap ROM improvement (r=0.41, n=15)
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy import stats

from src.db import backend_conn, query
from src.eval_deep_report import _MOVEMENT_PATTERNS, _classify  # reuse existing classifier
from src.research.profile_matrix import _classify_metric


def load_prescriptions_between(athlete_uuid: str,
                                start_date, end_date) -> pd.DataFrame:
    """Every exercise prescription an athlete received via coach programs whose
    duration overlaps the [start_date, end_date] window."""
    with backend_conn() as conn:
        rows = query(conn, """
            SELECT pep.category,
                   pep.exercise_name,
                   pep.exercise_type,
                   pep.n_sets,
                   pep.plyo_intensity,
                   ps.program_id,
                   ps.program_name,
                   ps.created_at_app,
                   ps.end_date_app
            FROM ai_layer.program_exercise_prescriptions pep
            JOIN ai_layer.program_summaries ps ON ps.program_id = pep.program_id
            WHERE pep.athlete_uuid = %s
              AND pep.exercise_name IS NOT NULL
              AND COALESCE(ps.created_at_app::date, %s) <= %s
              AND COALESCE(ps.end_date_app::date,   %s) >= %s
        """, [athlete_uuid, end_date, end_date, start_date, start_date])
    return pd.DataFrame(rows)


def pattern_volume_by_athlete(delta_df: pd.DataFrame) -> pd.DataFrame:
    """For each athlete in delta_df, count how many exercises of each movement
    pattern they received in the window. Returns DataFrame:

      athlete_uuid, pattern, category, n_exercises

    (long-format so it's easy to pivot)
    """
    rows: list[dict] = []
    for _, ath in delta_df.iterrows():
        pres = load_prescriptions_between(
            ath["athlete_uuid"], ath["first_date"], ath["last_date"]
        )
        if pres.empty:
            continue
        counter: Counter = Counter()
        for _, ex in pres.iterrows():
            pattern = _classify(ex["exercise_name"])
            if pattern == "unclassified":
                continue
            key = (pattern, ex["category"])
            counter[key] += 1
        for (pattern, category), n in counter.items():
            rows.append({
                "athlete_uuid": ath["athlete_uuid"],
                "name": ath["name"],
                "pattern": pattern,
                "category": category,
                "n_exercises": n,
            })
    return pd.DataFrame(rows)


def correlate_pattern_volume_to_delta(
    delta_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    *,
    min_n: int = 6,
    method: str = "spearman",
) -> pd.DataFrame:
    """For each (pattern × metric-delta) pair, compute correlation across
    athletes. Returns ranked findings.

    Result columns:
      pattern, category, metric, domain, r, p, n
    """
    delta_cols = [c for c in delta_df.columns if c.endswith("_delta")]
    if volume_df.empty or not delta_cols:
        return pd.DataFrame()

    # Pivot volumes to wide: rows = athlete, columns = (category::pattern)
    volume_wide = (volume_df.assign(key=volume_df["category"] + "::" + volume_df["pattern"])
                            .pivot_table(index="athlete_uuid", columns="key",
                                          values="n_exercises", fill_value=0))

    findings: list[dict] = []
    for pattern_key in volume_wide.columns:
        cat, pat = pattern_key.split("::", 1)
        vol_series = volume_wide[pattern_key]
        for dcol in delta_cols:
            metric = dcol[:-len("_delta")]
            # Merge on athlete_uuid
            merged = pd.DataFrame({
                "vol": vol_series,
            }).join(delta_df.set_index("athlete_uuid")[[dcol]], how="inner")
            valid = merged.dropna()
            n = len(valid)
            if n < min_n:
                continue
            x = valid["vol"].astype(float).values
            y = valid[dcol].astype(float).values
            if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
                continue
            try:
                if method == "spearman":
                    r, p = stats.spearmanr(x, y)
                else:
                    r, p = stats.pearsonr(x, y)
            except Exception:
                continue
            if r != r:
                continue
            findings.append({
                "category": cat,
                "pattern": pat,
                "metric": metric,
                "domain": _classify_metric(metric),
                "r": float(r),
                "p_value": float(p),
                "n": n,
            })

    if not findings:
        return pd.DataFrame()
    out = pd.DataFrame(findings)
    # BH FDR correction — with thousands of pattern×metric pairs, raw p-values
    # are essentially useless without correction.
    from src.research.correlations import _bh_q_values
    out["q_value"] = _bh_q_values(out["p_value"])
    out["abs_r"] = out["r"].abs()
    return (out.sort_values("abs_r", ascending=False)
              .drop(columns=["abs_r"])
              .reset_index(drop=True))
