"""
Bulk-generate ai_layer.athlete_profiles rows.

Two modes:
  --latest-only : one profile per athlete using their most recent assessment date.
                  Fast first pass, gives one snapshot per athlete (~457 rows).
  default       : one profile per (athlete, session_date) across every assessment
                  table — gives longitudinal coverage but is slower and larger.

Run:
    python -m src.main backfill                # full longitudinal
    python -m src.main backfill --latest-only  # latest snapshot per athlete
"""
from __future__ import annotations

import sys
import time

from src.db import backend_conn, exec_sql, query, returning_id
from src.profiler import build_and_save


# Every fact table that contributes (athlete, session_date) pairs to the timeline.
ASSESSMENT_TABLES = [
    "f_pitching_trials",
    "f_hitting_trials",
    "f_athletic_screen_cmj",
    "f_athletic_screen_dj",
    "f_athletic_screen_ppu",
    "f_athletic_screen_slv",
    "f_mobility",
    "f_proteus",
    "f_pro_sup",
]


def _all_pairs() -> list[tuple[str, str]]:
    """Distinct (athlete_uuid, session_date) pairs across all assessment tables."""
    union = " UNION ".join(
        f"SELECT athlete_uuid, session_date FROM public.{t} "
        f"WHERE athlete_uuid IS NOT NULL AND session_date IS NOT NULL"
        for t in ASSESSMENT_TABLES
    )
    sql = (
        f"SELECT athlete_uuid, session_date::text AS session_date "
        f"FROM ({union}) s ORDER BY athlete_uuid, session_date"
    )
    with backend_conn() as conn:
        return [(r["athlete_uuid"], r["session_date"]) for r in query(conn, sql)]


def _latest_per_athlete() -> list[tuple[str, str]]:
    """One (athlete_uuid, session_date) per athlete: their most recent assessment date."""
    union = " UNION ALL ".join(
        f"SELECT athlete_uuid, session_date FROM public.{t} "
        f"WHERE athlete_uuid IS NOT NULL AND session_date IS NOT NULL"
        for t in ASSESSMENT_TABLES
    )
    sql = (
        f"SELECT athlete_uuid, MAX(session_date)::text AS session_date "
        f"FROM ({union}) s GROUP BY athlete_uuid ORDER BY athlete_uuid"
    )
    with backend_conn() as conn:
        return [(r["athlete_uuid"], r["session_date"]) for r in query(conn, sql)]


def run_backfill(latest_only: bool = False) -> None:
    pairs = _latest_per_athlete() if latest_only else _all_pairs()
    total = len(pairs)
    mode = "latest-only" if latest_only else "full longitudinal"
    print(f"[backfill] {total} athlete-session pair(s) to process ({mode})")

    with backend_conn() as conn:
        log_id = returning_id(conn, """
            INSERT INTO ai_layer.sync_log (sync_type, table_name, status)
            VALUES ('embedding_refresh', 'athlete_profiles', 'running')
            RETURNING id
        """, [])

    started = time.time()
    successes = 0
    failures: list[tuple[str, str, str]] = []

    for i, (uuid, sd) in enumerate(pairs, 1):
        try:
            build_and_save(uuid, sd)
            successes += 1
        except Exception as e:
            failures.append((uuid, sd, str(e)[:200]))

        if i % 25 == 0 or i == total:
            elapsed = time.time() - started
            rate = i / elapsed if elapsed > 0 else 0
            eta_s = int((total - i) / rate) if rate > 0 else 0
            print(f"[backfill] {i}/{total}  ok={successes}  fail={len(failures)}  "
                  f"rate={rate:.1f}/s  eta={eta_s}s", flush=True)

    duration_ms = int((time.time() - started) * 1000)
    status = "success" if not failures else "partial"
    err_msg = (
        "; ".join(f"{u}@{d}: {e}" for u, d, e in failures[:5])[:480]
        if failures else None
    )

    with backend_conn() as conn:
        exec_sql(conn, """
            UPDATE ai_layer.sync_log
            SET rows_synced = %s, duration_ms = %s, status = %s,
                completed_at = now(), error_message = %s
            WHERE id = %s
        """, [successes, duration_ms, status, err_msg, log_id])

    print(f"[backfill] done. {successes} profiles written. failures: {len(failures)}")
    if failures:
        print("[backfill] sample failures:")
        for u, d, e in failures[:5]:
            print(f"  {u} @ {d}: {e}")


if __name__ == "__main__":
    run_backfill(latest_only=("--latest-only" in sys.argv))
