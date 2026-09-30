"""
Persistence for research output.

The rule this enforces: a rendered HTML file is a VIEW of a finding, never the
finding itself. Findings live in `ai_layer.research_findings` as structured
rows, so you can ask questions the filesystem cannot answer — what did we tell
this coach last month, which athletes have a limiter on front-leg braking,
what changed between this report and the previous one, which config produced
a given claim.

Everything here degrades gracefully. If the migration has not been applied,
the store logs once and becomes a no-op rather than taking down a report run:
the analysis is still worth having when the audit trail is not available yet.
Apply it with:

    psql "$BACKEND_DB_URL" -f sql/002_research_findings.sql
"""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.db import backend_conn, query

_MISSING_TABLE_WARNED: set[str] = set()


def _json_default(o: Any):
    if isinstance(o, (Decimal, np.floating)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    if is_dataclass(o):
        return asdict(o)
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)


def _dumps(v: Any) -> str:
    return json.dumps(v, default=_json_default)


def _clean(v: Any) -> Any:
    """NaN/inf → None, numpy scalars → python. psycopg2 will not take a NaN
    into a NUMERIC column, and silently dropping the row instead is worse."""
    if v is None:
        return None
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if not np.isfinite(f) else f
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime()
    return v


class _Unavailable(Exception):
    pass


def _guard(conn, table: str) -> None:
    rows = query(conn, "SELECT to_regclass(%s) AS t", [table])
    if not rows or rows[0]["t"] is None:
        if table not in _MISSING_TABLE_WARNED:
            _MISSING_TABLE_WARNED.add(table)
            print(f"[findings_store] {table} does not exist — skipping "
                  f"persistence. Apply sql/002_research_findings.sql to enable "
                  f"the research audit trail.")
        raise _Unavailable(table)


# ──────────────────────────────────────────────────────────────────────────
# Reliability
# ──────────────────────────────────────────────────────────────────────────

def save_reliability(table_df: pd.DataFrame, *, stratum: str = "ALL",
                     config_hash: str | None = None,
                     analysis_version: str | None = None) -> int:
    if table_df is None or table_df.empty:
        return 0
    sql = """
        INSERT INTO ai_layer.metric_reliability
          (metric, stratum, sem, cv_pct, icc, mdc95_single, sd_between,
           n_sessions, n_trials, n_athletes, source, config_hash,
           analysis_version, computed_at)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, now())
        ON CONFLICT (metric, stratum) DO UPDATE SET
          sem = EXCLUDED.sem, cv_pct = EXCLUDED.cv_pct, icc = EXCLUDED.icc,
          mdc95_single = EXCLUDED.mdc95_single, sd_between = EXCLUDED.sd_between,
          n_sessions = EXCLUDED.n_sessions, n_trials = EXCLUDED.n_trials,
          n_athletes = EXCLUDED.n_athletes, source = EXCLUDED.source,
          config_hash = EXCLUDED.config_hash,
          analysis_version = EXCLUDED.analysis_version,
          computed_at = now()
    """
    n = 0
    try:
        with backend_conn() as conn:
            _guard(conn, "ai_layer.metric_reliability")
            with conn.cursor() as cur:
                for r in table_df.to_dict("records"):
                    cur.execute(sql, [
                        r["metric"], stratum, _clean(r.get("sem")),
                        _clean(r.get("cv_pct")), _clean(r.get("icc")),
                        _clean(r.get("mdc95_single")), _clean(r.get("sd_between")),
                        int(r.get("n_sessions") or 0), int(r.get("n_trials") or 0),
                        int(r.get("n_athletes") or 0), r.get("source"),
                        config_hash, analysis_version,
                    ])
                    n += 1
    except _Unavailable:
        return 0
    return n


def load_reliability(*, stratum: str = "ALL") -> pd.DataFrame:
    try:
        with backend_conn() as conn:
            _guard(conn, "ai_layer.metric_reliability")
            rows = query(conn, """
                SELECT metric, sem, cv_pct, icc, mdc95_single, sd_between,
                       n_sessions, n_trials, n_athletes, source, computed_at
                FROM ai_layer.metric_reliability WHERE stratum = %s
            """, [stratum])
    except _Unavailable:
        return pd.DataFrame()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for c in ("sem", "cv_pct", "icc", "mdc95_single", "sd_between"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ──────────────────────────────────────────────────────────────────────────
# Cohort findings
# ──────────────────────────────────────────────────────────────────────────

def save_cohort_findings(df: pd.DataFrame, *, analysis: str, stratum: str,
                         target: str, config_hash: str | None = None,
                         analysis_version: str | None = None) -> int:
    if df is None or df.empty:
        return 0
    sql = """
        INSERT INTO ai_layer.cohort_findings
          (analysis, stratum, target, metric, r, ci_low, ci_high, p_value,
           q_value, fdr_significant, n, n_athletes, tier, extra,
           config_hash, analysis_version, computed_at)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, now())
        ON CONFLICT (analysis, stratum, target, metric) DO UPDATE SET
          r = EXCLUDED.r, ci_low = EXCLUDED.ci_low, ci_high = EXCLUDED.ci_high,
          p_value = EXCLUDED.p_value, q_value = EXCLUDED.q_value,
          fdr_significant = EXCLUDED.fdr_significant, n = EXCLUDED.n,
          n_athletes = EXCLUDED.n_athletes, tier = EXCLUDED.tier,
          extra = EXCLUDED.extra, config_hash = EXCLUDED.config_hash,
          analysis_version = EXCLUDED.analysis_version, computed_at = now()
    """
    known = {"metric", "r", "ci_low", "ci_high", "p_value", "q_value",
             "fdr_significant", "n", "n_trials", "n_athletes", "tier"}
    n = 0
    try:
        with backend_conn() as conn:
            _guard(conn, "ai_layer.cohort_findings")
            with conn.cursor() as cur:
                for r in df.to_dict("records"):
                    extra = {k: _clean(v) for k, v in r.items() if k not in known}
                    cur.execute(sql, [
                        analysis, stratum, target, r["metric"],
                        _clean(r.get("r")), _clean(r.get("ci_low")),
                        _clean(r.get("ci_high")), _clean(r.get("p_value")),
                        _clean(r.get("q_value")),
                        bool(r.get("fdr_significant", False)),
                        int(r.get("n") or r.get("n_trials") or 0),
                        int(r.get("n_athletes") or 0),
                        r.get("tier"), _dumps(extra),
                        config_hash, analysis_version,
                    ])
                    n += 1
    except _Unavailable:
        return 0
    return n


def load_cohort_findings(*, analysis: str, stratum: str,
                         target: str | None = None,
                         only_significant: bool = False) -> pd.DataFrame:
    where = ["analysis = %s", "stratum = %s"]
    params: list[Any] = [analysis, stratum]
    if target:
        where.append("target = %s")
        params.append(target)
    if only_significant:
        where.append("fdr_significant")
    try:
        with backend_conn() as conn:
            _guard(conn, "ai_layer.cohort_findings")
            rows = query(conn, f"""
                SELECT metric, r, ci_low, ci_high, p_value, q_value,
                       fdr_significant, n, n_athletes, tier, extra, computed_at
                FROM ai_layer.cohort_findings
                WHERE {' AND '.join(where)}
                ORDER BY abs(r) DESC NULLS LAST
            """, params)
    except _Unavailable:
        return pd.DataFrame()
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ──────────────────────────────────────────────────────────────────────────
# Per-athlete findings
# ──────────────────────────────────────────────────────────────────────────

def save_deep_dive(report, *, output_paths: dict[str, str] | None = None
                   ) -> int | None:
    """Persist a DeepDiveReport's structured findings. Returns the row id."""
    findings = [f.to_dict() for f in (report.headlines or [])]
    summary = dict(report.summary or {})
    ath = report.athlete
    try:
        with backend_conn() as conn:
            _guard(conn, "ai_layer.research_findings")
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO ai_layer.research_findings
                      (athlete_uuid, report_type, focus_session_date,
                       generated_at, config_hash, analysis_version, n_findings,
                       velocity_delta, velocity_verdict, findings, summary,
                       warnings, output_paths)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    RETURNING id
                """, [
                    ath.uuid, "athlete_deep_dive",
                    _clean(ath.focus_session_date), report.generated_at,
                    report.config.fingerprint(), report.config.analysis_version,
                    len(findings), _clean(summary.get("velocity_delta")),
                    summary.get("velocity_verdict"),
                    _dumps(findings), _dumps(summary),
                    _dumps(report.warnings or []),
                    _dumps(output_paths or {}),
                ])
                return int(cur.fetchone()[0])
    except _Unavailable:
        return None


def load_previous_deep_dive(athlete_uuid: str, *, before: datetime | None = None
                            ) -> dict | None:
    """The most recent stored report for an athlete, so a new run can diff
    against it instead of the coach having to remember."""
    where = ["athlete_uuid = %s", "report_type = 'athlete_deep_dive'"]
    params: list[Any] = [athlete_uuid]
    if before is not None:
        where.append("generated_at < %s")
        params.append(before)
    try:
        with backend_conn() as conn:
            _guard(conn, "ai_layer.research_findings")
            rows = query(conn, f"""
                SELECT id, generated_at, focus_session_date, config_hash,
                       analysis_version, findings, summary, velocity_delta,
                       velocity_verdict
                FROM ai_layer.research_findings
                WHERE {' AND '.join(where)}
                ORDER BY generated_at DESC LIMIT 1
            """, params)
    except _Unavailable:
        return None
    return rows[0] if rows else None


def diff_against_previous(report) -> dict:
    """What is new, resolved, or persisting since the last stored report.

    This is the thing a coach actually asks at the second meeting — 'is the
    thing you told me about last time fixed?' — and it was unanswerable while
    reports were only files on disk.
    """
    prev = load_previous_deep_dive(report.athlete.uuid, before=report.generated_at)
    if not prev:
        return {"has_previous": False}
    prev_findings = prev.get("findings") or []
    if isinstance(prev_findings, str):
        prev_findings = json.loads(prev_findings)
    prev_keys = {(f.get("kind"), f.get("metric")) for f in prev_findings}
    now_keys = {(f.kind, f.metric) for f in (report.headlines or [])}
    by_key = {(f.get("kind"), f.get("metric")): f for f in prev_findings}
    return {
        "has_previous": True,
        "previous_id": prev["id"],
        "previous_generated_at": prev["generated_at"],
        "new": sorted(k for k in now_keys - prev_keys),
        "resolved": sorted(k for k in prev_keys - now_keys),
        "persisting": sorted(k for k in now_keys & prev_keys),
        "previous_by_key": by_key,
        "previous_velocity_delta": prev.get("velocity_delta"),
    }


def recent_reports(limit: int = 50, report_type: str = "athlete_deep_dive"
                   ) -> pd.DataFrame:
    try:
        with backend_conn() as conn:
            _guard(conn, "ai_layer.research_findings")
            rows = query(conn, """
                SELECT r.id, r.athlete_uuid, d.name, r.focus_session_date,
                       r.generated_at, r.n_findings, r.velocity_delta,
                       r.velocity_verdict, r.config_hash
                FROM ai_layer.research_findings r
                LEFT JOIN analytics.d_athletes d USING (athlete_uuid)
                WHERE r.report_type = %s
                ORDER BY r.generated_at DESC
                LIMIT %s
            """, [report_type, limit])
    except _Unavailable:
        return pd.DataFrame()
    return pd.DataFrame(rows) if rows else pd.DataFrame()


__all__ = [
    "save_reliability", "load_reliability",
    "save_cohort_findings", "load_cohort_findings",
    "save_deep_dive", "load_previous_deep_dive", "diff_against_previous",
    "recent_reports",
]
