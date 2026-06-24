"""
Build a deficit profile for one athlete at one as-of date and write it to
ai_layer.athlete_profiles.

Workflow:
  1. Figure out the athlete's role (pitcher / hitter / both) from d_athletes flags.
  2. For each modality the athlete has data for, find the latest session_date
     at-or-before as_of_date.
  3. For each metric that applies to the role, extract the raw value from the
     corresponding modality's latest session.
  4. Look up the norm for that metric × the athlete's age_group, compute Z-score.
  5. Upsert one row into ai_layer.athlete_profiles.

CLI:
    python -m src.profiler <athlete_uuid> <YYYY-MM-DD>
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from src.db import backend_conn, exec_sql, query
from src.metrics_spec import METRICS, MODALITY_SESSION_TABLE, get_metrics_for_role


def _json_default(o: Any):
    if isinstance(o, Decimal):
        return float(o)
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    raise TypeError(f"Not JSON serializable: {type(o)}")


def _determine_role(conn, athlete_uuid: str) -> tuple[str, str | None]:
    """Return (role, age_group). Role in {'pitcher','hitter','both','unknown'}.

    age_group is normalized the same way refresh_norms does: trim whitespace,
    fall back to deriving from age if the stored value is NULL/empty.
    """
    rows = query(conn, """
        SELECT has_pitching_data, has_hitting_data, name,
               COALESCE(
                 NULLIF(TRIM(age_group), ''),
                 CASE
                   WHEN age <  14 THEN 'YOUTH'
                   WHEN age <  19 THEN 'HIGH SCHOOL'
                   WHEN age <  23 THEN 'COLLEGE'
                   WHEN age >= 23 THEN 'PRO'
                   ELSE NULL
                 END
               ) AS age_group
        FROM analytics.d_athletes
        WHERE athlete_uuid = %s
    """, [athlete_uuid])
    if not rows:
        raise ValueError(f"Athlete {athlete_uuid} not found in d_athletes.")
    r = rows[0]
    if r["has_pitching_data"] and r["has_hitting_data"]:
        role = "both"
    elif r["has_pitching_data"]:
        role = "pitcher"
    elif r["has_hitting_data"]:
        role = "hitter"
    else:
        role = "unknown"
    return role, r["age_group"]


def _modality_session_dates(conn, athlete_uuid: str, as_of_date: str) -> dict[str, date | None]:
    """Latest session_date at-or-before as_of_date, per modality."""
    out: dict[str, date | None] = {}
    for modality, recipe in MODALITY_SESSION_TABLE.items():
        rows = query(conn, recipe["sql"], [athlete_uuid, as_of_date])
        out[modality] = rows[0]["max"] if rows and rows[0]["max"] else None
    return out


def _extract_value(conn, athlete_uuid: str, session_date: date,
                   extract: dict) -> float | None:
    """Pull one numeric value for a metric, scoped to a specific (athlete, session_date)."""
    t = extract["type"]

    # ── JSONB extraction from f_pitching_trials.metrics / f_hitting_trials.metrics ──
    # Values can be a single-element array ([41.69]) OR a scalar (5.2). CASE handles both.
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
        rows = query(conn, f"""
            SELECT AVG({value_expr})::float AS v
            FROM public.{table}
            WHERE athlete_uuid = %s AND session_date = %s
              AND metrics ? %s
        """, [full_key, full_key, full_key, full_key,
              athlete_uuid, session_date, full_key])
        return rows[0]["v"]

    if t in ("pt_json_diff", "ht_json_diff"):
        table = "f_pitching_trials" if t == "pt_json_diff" else "f_hitting_trials"
        m_key = extract["minuend_key"]
        s_key = extract["subtrahend_key"]
        axis = extract.get("axis")
        m_full = f"{m_key}.{axis}" if axis else m_key
        s_full = f"{s_key}.{axis}" if axis else s_key
        diff_expr = _wrap(f"({_NUM_FROM_KEY}) - ({_NUM_FROM_KEY})")
        rows = query(conn, f"""
            SELECT AVG({diff_expr})::float AS v
            FROM public.{table}
            WHERE athlete_uuid = %s AND session_date = %s
              AND metrics ? %s AND metrics ? %s
        """, [m_full, m_full, m_full, m_full,
              s_full, s_full, s_full, s_full,
              athlete_uuid, session_date, m_full, s_full])
        return rows[0]["v"] if rows else None

    if t == "mob_col":
        col = extract["column"]
        midpoints = extract.get("midpoints") or {}
        if midpoints:
            when_clauses = " ".join(
                f'WHEN "{col}"::numeric = {int(g)} THEN {float(v)}'
                for g, v in sorted(midpoints.items())
            )
            value_expr = (
                f'(CASE WHEN "{col}" IS NULL THEN NULL '
                f'{when_clauses} '
                f'ELSE "{col}"::float END)'
            )
        else:
            value_expr = f'"{col}"::float'
        rows = query(conn, f"""
            SELECT {value_expr} AS v FROM public.f_mobility
            WHERE athlete_uuid = %s AND session_date = %s
              AND "{col}" IS NOT NULL
            LIMIT 1
        """, [athlete_uuid, session_date])
        return rows[0]["v"] if rows else None

    if t in ("pt_col", "ht_col"):
        table = extract.get("table") or (
            "f_pitching_trials" if t == "pt_col" else "f_hitting_trials")
        col = extract["column"]
        col_expr = f'"{col}"::int' if extract.get("bool_to_int") else f'"{col}"'
        rows = query(conn, f"""
            SELECT AVG({col_expr})::float AS v FROM public.{table}
            WHERE athlete_uuid = %s AND session_date = %s
              AND "{col}" IS NOT NULL
        """, [athlete_uuid, session_date])
        return rows[0]["v"]

    if t == "proteus_movement":
        col = extract["value_col"]
        filters = [f'"{col}" IS NOT NULL', "athlete_uuid = %s", "session_date = %s"]
        params: list[Any] = [athlete_uuid, session_date]
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
        rows = query(conn, f'SELECT AVG("{col}")::float AS v FROM public.f_proteus '
                           f'WHERE {where}', params)
        return rows[0]["v"]

    if t == "screen_col":
        col = extract["column"]
        tbl = extract["table"]
        rows = query(conn, f"""
            SELECT AVG("{col}")::float AS v FROM public."{tbl}"
            WHERE athlete_uuid = %s AND session_date = %s
              AND "{col}" IS NOT NULL
        """, [athlete_uuid, session_date])
        return rows[0]["v"]

    if t == "screen_col_side":
        col = extract["column"]
        tbl = extract["table"]
        side_char = extract["side"][0].lower()
        rows = query(conn, f"""
            SELECT AVG("{col}")::float AS v FROM public."{tbl}"
            WHERE athlete_uuid = %s AND session_date = %s
              AND "{col}" IS NOT NULL
              AND lower(substr(side, 1, 1)) = %s
        """, [athlete_uuid, session_date, side_char])
        return rows[0]["v"]

    raise ValueError(f"Unknown extract type: {t}")


def _load_norms(conn, age_group: str | None) -> dict[tuple[str, str], dict]:
    """Index norms for fast lookup: (modality, metric_key) -> norm row."""
    if age_group is None:
        return {}
    rows = query(conn, """
        SELECT modality, metric_key, n_observations, mean_value, std_value,
               p05, p25, p50, p75, p95
        FROM ai_layer.assessment_norms
        WHERE age_group = %s
    """, [age_group])
    return {(r["modality"], r["metric_key"]): r for r in rows}


def _zscore(value: float | None, norm: dict | None) -> float | None:
    if value is None or norm is None:
        return None
    mean = norm["mean_value"]
    std = norm["std_value"]
    if mean is None or std is None or float(std) == 0.0:
        return None
    return (float(value) - float(mean)) / float(std)


def build_profile(athlete_uuid: str, as_of_date: str) -> dict:
    """Build the profile dict (without persisting). Returns:
        {
          "athlete_uuid": ...,
          "as_of_date": "YYYY-MM-DD",
          "role": "pitcher" | "hitter" | "both" | "unknown",
          "age_group": "...",
          "source_dates": { modality: 'YYYY-MM-DD' | None, ... },
          "raw_values": { metric_key: float | None, ... },
          "z_scores":   { metric_key: float | None, ... },
        }
    """
    with backend_conn() as conn:
        role, age_group = _determine_role(conn, athlete_uuid)
        mod_dates = _modality_session_dates(conn, athlete_uuid, as_of_date)
        norms = _load_norms(conn, age_group)

        metrics = get_metrics_for_role(role if role != "unknown" else "both")
        raw: dict[str, float | None] = {}
        zs:  dict[str, float | None] = {}

        for m in metrics:
            mod = m["modality"]
            sd = mod_dates.get(mod)
            if sd is None:
                raw[m["key"]] = None
                zs[m["key"]] = None
                continue
            try:
                v = _extract_value(conn, athlete_uuid, sd, m["extract"])
            except Exception as e:
                # Bad column name / metric_name typo / etc. — record None, keep going.
                v = None
            raw[m["key"]] = v
            zs[m["key"]] = _zscore(v, norms.get((mod, m["key"])))

    return {
        "athlete_uuid": athlete_uuid,
        "as_of_date":   as_of_date,
        "role":         role,
        "age_group":    age_group,
        "source_dates": {k: (v.isoformat() if v else None) for k, v in mod_dates.items()},
        "raw_values":   raw,
        "z_scores":     zs,
    }


def save_profile(profile: dict) -> int:
    """Upsert the profile into ai_layer.athlete_profiles. Returns the row id."""
    with backend_conn() as conn:
        # Use UPSERT to be safely re-runnable
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO ai_layer.athlete_profiles
                  (athlete_uuid, as_of_date, age_group, raw_values, z_scores, source_dates, notes)
                VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s)
                ON CONFLICT (athlete_uuid, as_of_date) DO UPDATE
                  SET age_group    = EXCLUDED.age_group,
                      raw_values   = EXCLUDED.raw_values,
                      z_scores     = EXCLUDED.z_scores,
                      source_dates = EXCLUDED.source_dates,
                      notes        = EXCLUDED.notes,
                      created_at   = now()
                RETURNING id
            """, [
                profile["athlete_uuid"],
                profile["as_of_date"],
                profile["age_group"],
                json.dumps(profile["raw_values"], default=_json_default),
                json.dumps(profile["z_scores"],   default=_json_default),
                json.dumps(profile["source_dates"], default=_json_default),
                f"role={profile['role']}",
            ])
            return cur.fetchone()[0]


def build_and_save(athlete_uuid: str, as_of_date: str) -> tuple[int, dict]:
    p = build_profile(athlete_uuid, as_of_date)
    pid = save_profile(p)
    return pid, p


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m src.profiler <athlete_uuid> <YYYY-MM-DD>")
        sys.exit(2)
    pid, profile = build_and_save(sys.argv[1], sys.argv[2])
    print(f"Saved profile id: {pid}")
    print(json.dumps(profile, default=_json_default, indent=2))
