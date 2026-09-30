"""
"Why isn't this athlete's newest session showing up?"

This repo READS the warehouse. It never writes `public.f_*`. Those tables are
populated by the UAIS runners in OctaneBiomechBackend (`uais/R/pitching/main.R`
for pitching, via /dashboard/uais-maintenance). So when a capture you know
happened is missing from a report, no command here will conjure it — the
useful question is *where* the chain broke, because there are four common
causes and each has a different fix:

  1. WRONG DATABASE / BRANCH — `.env` BACKEND_DB_URL points at a different Neon
     branch than the one the biomech backend writes to. The whole warehouse
     looks stale, not just one athlete.
  2. DUPLICATE ATHLETE RECORD — the new session landed under a second
     `athlete_uuid` for the same person (the exact failure
     `find_and_merge_similar_athletes.py` exists to fix). The athlete looks
     stale; the data is there under another id.
  3. NOT INGESTED — the capture exists as raw files but the pitching runner
     was never run, or it failed. Nothing is in the warehouse at all.
  4. PARTIALLY INGESTED — kinematics landed but force plate did not, or the
     session_date is null, so it exists but drops out of the analysis.

`research diagnose <athlete>` distinguishes all four in one pass and says
which one you are looking at.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import pandas as pd

from src.db import backend_conn, query

# Every table that carries a per-athlete, per-session record. Name → table.
SOURCE_TABLES: dict[str, str] = {
    "pitching_trials": "public.f_pitching_trials",
    "pitching_force": "public.f_pitching_force_metrics",
    "kinematics_pitching": "public.f_kinematics_pitching",
    "hitting_trials": "public.f_hitting_trials",
    "mobility": "public.f_mobility",
    "proteus": "public.f_proteus",
    "screen_cmj": "public.f_athletic_screen_cmj",
    "screen_dj": "public.f_athletic_screen_dj",
    "screen_ppu": "public.f_athletic_screen_ppu",
    "screen_slv": "public.f_athletic_screen_slv",
    # NMT is discontinued — kept out of the round sources; diagnose still
    # lists it so a stray row in the table is visible.
    "screen_nmt (discontinued)": "public.f_athletic_screen_nmt",
    "arm_action": "public.f_arm_action",
    "curveball": "public.f_curveball_test",
    "pro_sup": "public.f_pro_sup",
}

# The one this repo's pitching analysis is actually driven by.
PRIMARY_TABLE = "pitching_trials"


@dataclass
class AthleteRecord:
    athlete_uuid: str
    name: str
    email: str | None
    age_group: str | None
    app_db_uuid: str | None
    has_pitching: bool | None
    pitching_session_count: int | None
    per_table: dict[str, dict] = field(default_factory=dict)

    @property
    def latest_pitching(self) -> Any | None:
        return (self.per_table.get(PRIMARY_TABLE) or {}).get("max_date")

    @property
    def n_pitching_sessions(self) -> int:
        return int((self.per_table.get(PRIMARY_TABLE) or {}).get("n_sessions") or 0)


@dataclass
class Diagnosis:
    query_str: str
    db_host: str
    db_name: str
    db_branch_hint: str | None
    matches: list[AthleteRecord]
    warehouse_latest: dict[str, Any]
    would_pick: str | None
    verdict: str
    advice: list[str] = field(default_factory=list)


def _table_exists(conn, table: str) -> bool:
    rows = query(conn, "SELECT to_regclass(%s) AS t", [table])
    return bool(rows and rows[0]["t"] is not None)


def _connection_identity(conn) -> tuple[str, str, str | None]:
    """Which database are we actually talking to?

    Neon serves branches on distinct hostnames, so the host is the fastest way
    to spot 'my .env points at a different branch than the app writes to'.
    """
    rows = query(conn, """
        SELECT current_database() AS db,
               inet_server_addr()::text AS addr,
               current_setting('neon.endpoint_id', true) AS endpoint
    """)
    r = rows[0] if rows else {}
    host = r.get("addr") or "(local socket)"
    return host, r.get("db") or "?", r.get("endpoint")


def _dsn_host(url: str | None) -> str:
    """Host portion of the DSN, without leaking credentials."""
    if not url:
        return "(BACKEND_DB_URL unset)"
    try:
        after_at = url.split("@", 1)[1]
        return after_at.split("/", 1)[0].split("?", 1)[0]
    except Exception:
        return "(unparseable)"


def diagnose_athlete(query_str: str) -> Diagnosis:
    """Everything needed to tell which of the four failure modes this is."""
    import os

    with backend_conn() as conn:
        host, dbname, endpoint = _connection_identity(conn)

        # ── Every athlete record that could plausibly be this person ──────
        rows = query(conn, """
            SELECT athlete_uuid, name, email, TRIM(age_group) AS age_group,
                   app_db_uuid::text AS app_db_uuid,
                   has_pitching_data, pitching_session_count
            FROM analytics.d_athletes
            WHERE LOWER(name) = LOWER(%s)
               OR name ILIKE %s
               OR LOWER(COALESCE(email, '')) = LOWER(%s)
               OR athlete_uuid::text = %s
            ORDER BY name
        """, [query_str, f"%{query_str}%", query_str, query_str])

        matches: list[AthleteRecord] = []
        for r in rows:
            rec = AthleteRecord(
                athlete_uuid=str(r["athlete_uuid"]),
                name=r["name"],
                email=r.get("email"),
                age_group=r.get("age_group"),
                app_db_uuid=r.get("app_db_uuid"),
                has_pitching=r.get("has_pitching_data"),
                pitching_session_count=r.get("pitching_session_count"),
            )
            for key, table in SOURCE_TABLES.items():
                if not _table_exists(conn, table):
                    continue
                try:
                    stat = query(conn, f"""
                        SELECT COUNT(*)::int                AS n_rows,
                               COUNT(DISTINCT session_date)::int AS n_sessions,
                               MIN(session_date)            AS min_date,
                               MAX(session_date)            AS max_date,
                               COUNT(*) FILTER (WHERE session_date IS NULL)::int
                                                            AS n_null_dates
                        FROM {table} WHERE athlete_uuid = %s
                    """, [rec.athlete_uuid])
                except Exception as e:
                    rec.per_table[key] = {"error": str(e).split("\n")[0]}
                    continue
                s = stat[0] if stat else {}
                if (s.get("n_rows") or 0) > 0:
                    rec.per_table[key] = dict(s)
            matches.append(rec)

        # ── How fresh is the warehouse as a whole? ────────────────────────
        warehouse_latest: dict[str, Any] = {}
        for key, table in SOURCE_TABLES.items():
            if not _table_exists(conn, table):
                warehouse_latest[key] = {"missing_table": True}
                continue
            try:
                s = query(conn, f"""
                    SELECT MAX(session_date) AS max_date,
                           COUNT(DISTINCT athlete_uuid)::int AS n_athletes
                    FROM {table}
                """)
                warehouse_latest[key] = dict(s[0]) if s else {}
            except Exception as e:
                warehouse_latest[key] = {"error": str(e).split("\n")[0]}

    # ── Which record would the report have used? ──────────────────────────
    exact = [m for m in matches if m.name.strip().lower() == query_str.strip().lower()]
    pool = exact or matches
    would_pick = None
    if pool:
        with_data = [m for m in pool if m.n_pitching_sessions > 0]
        would_pick = (max(with_data, key=lambda m: (m.latest_pitching or date.min)).athlete_uuid
                      if with_data else pool[0].athlete_uuid)

    verdict, advice = _interpret(query_str, matches, exact, warehouse_latest,
                                 host, _dsn_host(os.getenv("BACKEND_DB_URL")))

    return Diagnosis(
        query_str=query_str,
        db_host=_dsn_host(os.getenv("BACKEND_DB_URL")),
        db_name=dbname,
        db_branch_hint=endpoint,
        matches=matches,
        warehouse_latest=warehouse_latest,
        would_pick=would_pick,
        verdict=verdict,
        advice=advice,
    )


def _interpret(query_str, matches, exact, warehouse_latest, server_host,
               dsn_host) -> tuple[str, list[str]]:
    """Turn the raw counts into 'this is what happened, here is the fix'."""
    advice: list[str] = []

    if not matches:
        return ("NO ATHLETE RECORD — nothing in analytics.d_athletes matches "
                f"{query_str!r}.", [
            "The athlete has never been created in the warehouse, or the name "
            "is spelled differently there. Search the Athletes page in "
            "OctaneBiomechBackend, or query d_athletes with a looser pattern.",
        ])

    with_pitching = [m for m in matches if m.n_pitching_sessions > 0]

    # ── Duplicates ────────────────────────────────────────────────────────
    if len(exact) > 1:
        dates = {m.athlete_uuid: m.latest_pitching for m in exact}
        newest = max((d for d in dates.values() if d), default=None)
        advice.append(
            f"{len(exact)} separate athlete records share this exact name. "
            f"The newest pitching session across them is {newest}. If that is "
            f"the capture you expected and the report showed an older one, the "
            f"new session landed on a DIFFERENT athlete_uuid.")
        advice.append(
            "Fix in OctaneBiomechBackend: run "
            "`uais/python/scripts/find_and_merge_similar_athletes.py` to merge "
            "them, then re-run the report here. Until they are merged, you can "
            "point at the right one directly: "
            "`research coach-report <athlete_uuid>`.")
        return (f"DUPLICATE ATHLETE RECORDS — {len(exact)} rows in d_athletes "
                f"with this name.", advice)

    # ── Nothing ingested ──────────────────────────────────────────────────
    if not with_pitching:
        advice.append(
            "The athlete record exists but has no rows in f_pitching_trials. "
            "The pitching runner has not written this athlete's capture to the "
            "warehouse.")
        advice.append(
            "Fix in OctaneBiomechBackend: /dashboard/uais-maintenance → select "
            "the `pitching` runner → choose this existing athlete → upload the "
            "capture files → Run. Watch the stream for errors; the runner has "
            "to exit 0 for the rows to land.")
        return ("NOT INGESTED — no pitching trials in the warehouse for this "
                "athlete.", advice)

    # ── Whole-warehouse staleness ─────────────────────────────────────────
    wh_latest = (warehouse_latest.get(PRIMARY_TABLE) or {}).get("max_date")
    ath_latest = with_pitching[0].latest_pitching
    if wh_latest and ath_latest and wh_latest == ath_latest:
        advice.append(
            f"This athlete's latest capture ({ath_latest}) is ALSO the newest "
            f"pitching session in the entire warehouse. That usually means the "
            f"whole database is behind, not this one athlete.")
        advice.append(
            f"Check that .env BACKEND_DB_URL points at the same Neon branch the "
            f"biomech backend writes to. This session is connected to "
            f"`{dsn_host}`. Compare it against WAREHOUSE_DATABASE_URL / the "
            f"`uais_warehouse_db_url` setting in OctaneBiomechBackend — Neon "
            f"serves each branch on its own hostname, so if those differ you "
            f"are reading a different copy of the data.")
        return ("POSSIBLY WRONG BRANCH, OR WAREHOUSE-WIDE STALE — this "
                "athlete's newest capture is the newest capture anywhere.",
                advice)

    # ── Partial ingest ────────────────────────────────────────────────────
    m = with_pitching[0]
    pt = m.per_table.get(PRIMARY_TABLE, {})
    if pt.get("n_null_dates"):
        advice.append(
            f"{pt['n_null_dates']} pitching trial row(s) have a NULL "
            f"session_date. Those are invisible to every date-based analysis "
            f"here, including 'latest session'.")
        return ("NULL SESSION DATES — rows exist but cannot be placed in time.",
                advice)

    force = m.per_table.get("pitching_force", {})
    if force and force.get("max_date") and pt.get("max_date") \
            and force["max_date"] < pt["max_date"]:
        advice.append(
            f"Kinematics for {pt['max_date']} are in the warehouse but the "
            f"force-plate rows stop at {force['max_date']}. The capture is "
            f"half-ingested — most force findings will be missing for the "
            f"newest session.")

    advice.append(
        f"The newest pitching session in the warehouse for this athlete is "
        f"{m.latest_pitching}. If the capture you are thinking of is newer than "
        f"that, it has not been written to f_pitching_trials yet — run the "
        f"`pitching` runner for it in OctaneBiomechBackend.")
    advice.append(
        "Nothing in OctaneAiLayer can pull it in: this repo only reads "
        "public.f_*. Once the runner has written the rows, re-run "
        "`research coach-report` and it will pick them up (no cache to clear — "
        "each CLI run queries fresh).")
    return (f"WAREHOUSE HAS DATA UP TO {m.latest_pitching} FOR THIS ATHLETE.",
            advice)


def format_diagnosis(d: Diagnosis) -> str:
    """Terminal-friendly rendering."""
    out: list[str] = []
    out.append(f"Connected to : {d.db_host}  (database {d.db_name}"
               + (f", neon endpoint {d.db_branch_hint}" if d.db_branch_hint else "")
               + ")")
    out.append("")
    out.append(f"Records matching {d.query_str!r}: {len(d.matches)}")
    for m in d.matches:
        star = " <-- the report uses this one" if m.athlete_uuid == d.would_pick else ""
        out.append("")
        out.append(f"  {m.name}  [{m.age_group or '?'}]{star}")
        out.append(f"    uuid            {m.athlete_uuid}")
        out.append(f"    email           {m.email or '(none)'}")
        out.append(f"    linked to app   {m.app_db_uuid or '(not linked)'}")
        out.append(f"    d_athletes says {m.pitching_session_count} pitching session(s), "
                   f"has_pitching={m.has_pitching}")
        if not m.per_table:
            out.append("    NO ROWS in any assessment table")
            continue
        out.append(f"    {'source':<22s} {'rows':>6s} {'sess':>5s}  "
                   f"{'first':<12s} {'latest':<12s}")
        for key in SOURCE_TABLES:
            s = m.per_table.get(key)
            if not s:
                continue
            if "error" in s:
                out.append(f"    {key:<22s}  ERROR: {s['error']}")
                continue
            nul = f"  ({s['n_null_dates']} null dates)" if s.get("n_null_dates") else ""
            out.append(f"    {key:<22s} {s['n_rows']:>6d} {s['n_sessions']:>5d}  "
                       f"{str(s['min_date']):<12s} {str(s['max_date']):<12s}{nul}")

    out.append("")
    out.append("Newest session anywhere in the warehouse:")
    for key in SOURCE_TABLES:
        s = d.warehouse_latest.get(key) or {}
        if s.get("missing_table"):
            out.append(f"  {key:<22s} (table not present)")
        elif s.get("error"):
            out.append(f"  {key:<22s} ERROR: {s['error']}")
        else:
            out.append(f"  {key:<22s} {str(s.get('max_date')):<12s} "
                       f"({s.get('n_athletes', 0)} athletes)")

    out.append("")
    out.append(f"VERDICT: {d.verdict}")
    for a in d.advice:
        out.append("")
        out.append(f"  - {a}")
    return "\n".join(out)


__all__ = ["diagnose_athlete", "format_diagnosis", "Diagnosis", "AthleteRecord",
           "SOURCE_TABLES"]
