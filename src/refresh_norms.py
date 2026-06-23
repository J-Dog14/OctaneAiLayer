"""
Compute per-age-group population norms for every metric in metrics_spec.METRICS
and upsert them into ai_layer.assessment_norms.

For each metric, the workflow is:
  1. Build a "per-session" SQL subquery that yields (athlete_uuid, session_date, value).
  2. Join with analytics.d_athletes to attach age_group.
  3. GROUP BY age_group; compute n, mean, std, percentiles.
  4. UPSERT into ai_layer.assessment_norms keyed by (modality, metric_key, age_group).

Run periodically (monthly is fine):
    python -m src.refresh_norms
    python -m src.refresh_norms screen_dj_rsi proteus_pitcher_shotput_power_mean
"""
from __future__ import annotations

import sys
from typing import Any

from src.db import backend_conn, exec_sql, query, returning_id
from src.metrics_spec import METRICS


# Groups with fewer than this many observations are skipped (norms unreliable).
MIN_OBSERVATIONS = 5


def _per_session_sql(extract: dict) -> tuple[str, list[Any]]:
    """Build the per-session subquery for one metric.

    Returns (sql, params). `sql` produces rows of (athlete_uuid, session_date, value).
    """
    t = extract["type"]

    # ── JSONB extraction from f_pitching_trials.metrics / f_hitting_trials.metrics ──
    # Keys are FLAT dotted strings (e.g. "PROCESSED.Pelvis_Angle@Footstrike.Z").
    # Values are inconsistent:
    #   - PROCESSED.* metrics are wrapped in a single-element JSON array: [41.69]
    #   - PLANE.* metrics are stored as scalars:                          5.2
    # The CASE handles both: take array[0] if array, else cast scalar.
    _NUM_FROM_KEY = (
        "CASE "
        "  WHEN jsonb_typeof(metrics->%s) = 'array'                 THEN (metrics->%s->>0)::numeric "
        "  WHEN jsonb_typeof(metrics->%s) IN ('number', 'string')   THEN (metrics->>%s)::numeric "
        "  ELSE NULL "
        "END"
    )

    def _wrap(expr: str) -> str:
        """Apply optional abs() and scale to a numeric SQL expression."""
        if extract.get("abs"):
            expr = f"ABS({expr})"
        if "scale" in extract:
            expr = f"({expr}) * {float(extract['scale'])}"
        return expr

    if t in ("pt_json", "ht_json"):
        table = "f_pitching_trials" if t == "pt_json" else "f_hitting_trials"
        key = extract["key"]
        axis = extract.get("axis")
        full_key = f"{key}.{axis}" if axis else key
        value_expr = _wrap(_NUM_FROM_KEY)
        return (
            f"SELECT athlete_uuid, session_date, "
            f"  AVG({value_expr})::numeric AS value "
            f"FROM public.{table} "
            f"WHERE metrics ? %s "
            f"GROUP BY athlete_uuid, session_date",
            [full_key, full_key, full_key, full_key, full_key],
        )

    if t in ("pt_json_diff", "ht_json_diff"):
        table = "f_pitching_trials" if t == "pt_json_diff" else "f_hitting_trials"
        m_key = extract["minuend_key"]
        s_key = extract["subtrahend_key"]
        axis = extract.get("axis")
        m_full = f"{m_key}.{axis}" if axis else m_key
        s_full = f"{s_key}.{axis}" if axis else s_key
        diff_expr = _wrap(f"({_NUM_FROM_KEY}) - ({_NUM_FROM_KEY})")
        return (
            f"SELECT athlete_uuid, session_date, "
            f"  AVG({diff_expr})::numeric AS value "
            f"FROM public.{table} "
            f"WHERE metrics ? %s AND metrics ? %s "
            f"GROUP BY athlete_uuid, session_date",
            [m_full, m_full, m_full, m_full,
             s_full, s_full, s_full, s_full,
             m_full, s_full],
        )

    if t == "mob_col":
        col = extract["column"]
        return (
            f'SELECT athlete_uuid, session_date, "{col}"::numeric AS value '
            f'FROM public.f_mobility WHERE "{col}" IS NOT NULL',
            [],
        )

    if t in ("pt_col", "ht_col"):
        # Default tables: f_pitching_trials / f_hitting_trials. Pass `table` in the
        # extract spec to point at a different table (e.g. f_pitching_force_metrics).
        # Pass `bool_to_int: True` if the column is boolean.
        table = extract.get("table") or (
            "f_pitching_trials" if t == "pt_col" else "f_hitting_trials")
        col = extract["column"]
        col_expr = f'"{col}"::int' if extract.get("bool_to_int") else f'"{col}"'
        return (
            f'SELECT athlete_uuid, session_date, AVG({col_expr})::numeric AS value '
            f'FROM public.{table} WHERE "{col}" IS NOT NULL '
            f'GROUP BY athlete_uuid, session_date',
            [],
        )

    if t == "proteus_movement":
        col = extract["value_col"]
        params: list[Any] = []
        filters = [f'"{col}" IS NOT NULL']
        if "movement_eq" in extract:
            filters.append("movement = %s")
            params.append(extract["movement_eq"])
        if "movement_like" in extract:
            filters.append("movement ILIKE %s")
            params.append(f"%{extract['movement_like']}%")
        if "position_like" in extract:
            filters.append("position ILIKE %s")
            params.append(f"{extract['position_like']}%")
        if "position_not_like" in extract:
            filters.append("(position IS NULL OR position NOT ILIKE %s)")
            params.append(f"{extract['position_not_like']}%")
        where = " AND ".join(filters)
        return (
            f'SELECT athlete_uuid, session_date, AVG("{col}")::numeric AS value '
            f"FROM public.f_proteus WHERE {where} "
            f"GROUP BY athlete_uuid, session_date",
            params,
        )

    if t == "screen_col":
        col = extract["column"]
        tbl = extract["table"]
        return (
            f'SELECT athlete_uuid, session_date, AVG("{col}")::numeric AS value '
            f'FROM public."{tbl}" WHERE "{col}" IS NOT NULL '
            f"GROUP BY athlete_uuid, session_date",
            [],
        )

    if t == "screen_col_side":
        col = extract["column"]
        tbl = extract["table"]
        side_char = extract["side"][0].lower()  # 'l' or 'r'
        return (
            f'SELECT athlete_uuid, session_date, AVG("{col}")::numeric AS value '
            f'FROM public."{tbl}" '
            f'WHERE "{col}" IS NOT NULL AND lower(substr(side, 1, 1)) = %s '
            f"GROUP BY athlete_uuid, session_date",
            [side_char],
        )

    raise ValueError(f"Unknown extract type: {t}")


def _norms_sql(per_session_sql: str) -> str:
    """Wrap a per-session subquery in the population-stats SELECT.

    age_group normalization happens in the `joined` CTE so the outer
    aggregation only sees the resolved value: TRIM the stored age_group first
    (collapses whitespace variants like 'HIGH SCHOOL '), fall back to deriving
    from d.age using standard baseball cutoffs, and only land on 'unknown' if
    both fields are NULL.
    """
    return f"""
        WITH per_session AS ({per_session_sql}),
        joined AS (
          SELECT
            p.value,
            COALESCE(
              NULLIF(TRIM(d.age_group), ''),
              CASE
                WHEN d.age <  14 THEN 'YOUTH'
                WHEN d.age <  19 THEN 'HIGH SCHOOL'
                WHEN d.age <  23 THEN 'COLLEGE'
                WHEN d.age >= 23 THEN 'PRO'
                ELSE 'unknown'
              END
            ) AS age_group
          FROM per_session p
          JOIN analytics.d_athletes d ON d.athlete_uuid = p.athlete_uuid
          WHERE p.value IS NOT NULL
        )
        SELECT
          age_group,
          COUNT(*)::int                         AS n_observations,
          AVG(value)::numeric                   AS mean_value,
          STDDEV_SAMP(value)::numeric           AS std_value,
          percentile_cont(0.05) WITHIN GROUP (ORDER BY value) AS p05,
          percentile_cont(0.25) WITHIN GROUP (ORDER BY value) AS p25,
          percentile_cont(0.50) WITHIN GROUP (ORDER BY value) AS p50,
          percentile_cont(0.75) WITHIN GROUP (ORDER BY value) AS p75,
          percentile_cont(0.95) WITHIN GROUP (ORDER BY value) AS p95
        FROM joined
        GROUP BY age_group
        HAVING COUNT(*) >= {MIN_OBSERVATIONS}
        ORDER BY age_group
    """


def refresh_one_metric(metric: dict, log_id: int | None = None) -> int:
    """Compute and upsert norms for one metric. Returns count of age-group rows written."""
    key = metric["key"]
    modality = metric["modality"]
    inner_sql, params = _per_session_sql(metric["extract"])
    sql = _norms_sql(inner_sql)

    rows_written = 0
    with backend_conn() as conn:
        rows = query(conn, sql, params)
        # Wipe and replace this metric's norms (small table, this is fine).
        exec_sql(conn, """
            DELETE FROM ai_layer.assessment_norms
            WHERE modality = %s AND metric_key = %s
        """, [modality, key])
        for r in rows:
            exec_sql(conn, """
                INSERT INTO ai_layer.assessment_norms
                  (modality, metric_key, age_group, n_observations,
                   mean_value, std_value, p05, p25, p50, p75, p95)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (modality, metric_key, age_group) DO UPDATE
                  SET n_observations = EXCLUDED.n_observations,
                      mean_value     = EXCLUDED.mean_value,
                      std_value      = EXCLUDED.std_value,
                      p05            = EXCLUDED.p05,
                      p25            = EXCLUDED.p25,
                      p50            = EXCLUDED.p50,
                      p75            = EXCLUDED.p75,
                      p95            = EXCLUDED.p95,
                      computed_at    = now()
            """, [modality, key, r["age_group"], r["n_observations"],
                  r["mean_value"], r["std_value"],
                  r["p05"], r["p25"], r["p50"], r["p75"], r["p95"]])
            rows_written += 1

    return rows_written


def refresh_all(only_keys: list[str] | None = None) -> None:
    targets = [m for m in METRICS if (only_keys is None or m["key"] in only_keys)]
    print(f"[norms] refreshing {len(targets)} metric(s)")

    with backend_conn() as conn:
        log_id = returning_id(conn, """
            INSERT INTO ai_layer.sync_log (sync_type, table_name, status)
            VALUES ('embedding_refresh', 'assessment_norms', 'running')
            RETURNING id
        """, [])

    total_rows = 0
    failures: list[tuple[str, str]] = []
    for m in targets:
        try:
            n = refresh_one_metric(m)
            total_rows += n
            print(f"[norms] {m['key']:<50} {n:>3} age-group rows", flush=True)
        except Exception as e:
            err = str(e)[:300]
            print(f"[norms] {m['key']}: FAILED - {err}", flush=True)
            failures.append((m["key"], err))

    status = "success" if not failures else "partial"
    err_summary = "; ".join(f"{k}: {e}" for k, e in failures)[:480] if failures else None

    with backend_conn() as conn:
        exec_sql(conn, """
            UPDATE ai_layer.sync_log
            SET rows_synced = %s, status = %s, completed_at = now(),
                error_message = %s
            WHERE id = %s
        """, [total_rows, status, err_summary, log_id])

    print(f"[norms] done. {total_rows} rows written. failures: {len(failures)}")


if __name__ == "__main__":
    args = sys.argv[1:]
    refresh_all(args if args else None)
