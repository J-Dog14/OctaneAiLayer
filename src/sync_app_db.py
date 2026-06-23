"""
Snapshot a curated set of App DB tables into the backend's app_db_snapshot schema.

Transfer mechanism: native Postgres COPY (CSV format). This is dramatically
faster than INSERT batches because the data streams as a single statement on
each side instead of one round-trip per row.

Flow per table:
  1. Read App DB column definitions from information_schema.
  2. CREATE TABLE app_db_snapshot._staging_<table> with permissive types
     (enums and varchars become TEXT so we don't have to re-create enum types).
  3. On the App DB:   COPY (SELECT ... FROM table) TO STDOUT WITH CSV
     On the backend:  COPY ..._staging_<table> FROM STDIN WITH CSV
     Bytes are piped through a SpooledTemporaryFile (in-RAM up to 100MB,
     auto-spills to disk for huge tables).
  4. Atomically swap the live table for the staging one.

Run:
    python -m src.sync_app_db                # full sync
    python -m src.sync_app_db User Program   # just these tables
"""
from __future__ import annotations

import sys
import tempfile
import time

from src.db import app_db_conn, backend_conn, exec_sql, query, returning_id

# Tables to snapshot. Order matters only loosely (no FKs are enforced in snapshot).
TABLES: list[str] = [
    "User",
    "Org",
    "AthleteMetrics", "WeightLog",
    "Program", "ProgramDay",
    "LiftToProgram", "LiftToProgramDay", "ExerciseToLift", "Exercise",
    "PlyoToProgram", "Plyo", "ThrowingExerciseToPlyo",
    "PrepToProgram", "ExerciseToPrep",
    "BulletProofingToProgram", "ExerciseToBulletProofing",
    "HittingToProgram", "HittingToProgramDay", "ExerciseToHitting",
    "MovementEnhancementToProgram", "ExerciseToMovementEnhancement",
    "ThrowingActivity",
    # Weekly templates — the "coach intent" layer behind programs.
    "WeeklyTemplate", "WeeklyTemplateDay", "WeeklyTemplateApplication",
    "WeeklyTemplateMetadata",
    "ActivityTemplate",
    "ExerciseToPrepTemplate", "ExerciseToBulletProofingTemplate",
    "ExerciseToHittingTemplate", "ExerciseToMovementEnhancementTemplate",
    "PlyoToTemplate", "ThrowingExerciseToPlyoTemplate",
]

# Map information_schema data_type -> a permissive Postgres type for the snapshot.
# Enums collapse to TEXT so we don't need to recreate enum types on the backend side.
_TYPE_MAP = {
    "character varying": "TEXT",
    "text": "TEXT",
    "uuid": "UUID",
    "integer": "INTEGER",
    "bigint": "BIGINT",
    "smallint": "SMALLINT",
    "boolean": "BOOLEAN",
    "numeric": "NUMERIC",
    "real": "REAL",
    "double precision": "DOUBLE PRECISION",
    "date": "DATE",
    "timestamp without time zone": "TIMESTAMP",
    "timestamp with time zone": "TIMESTAMPTZ",
    "json": "JSONB",
    "jsonb": "JSONB",
    "bytea": "BYTEA",
}


def _columns_of(conn, schema: str, table: str) -> list[tuple[str, str, str]]:
    rows = query(conn, """
        SELECT column_name, data_type, udt_name
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position
    """, [schema, table])
    return [(r["column_name"], r["data_type"], r["udt_name"]) for r in rows]


def _pg_type(data_type: str, udt_name: str) -> str:
    if data_type == "USER-DEFINED":
        return "TEXT"  # enums
    if data_type == "ARRAY":
        # _text -> TEXT[], _int4 -> INTEGER[], etc.
        base = udt_name.lstrip("_")
        return f"{_TYPE_MAP.get(base, 'TEXT')}[]"
    return _TYPE_MAP.get(data_type, "TEXT")


def _build_create_table(schema_dest: str, table: str, cols: list[tuple[str, str, str]]) -> str:
    col_defs = [f'"{name}" {_pg_type(dtype, udt)}' for name, dtype, udt in cols]
    return (
        f'CREATE TABLE "{schema_dest}"."{table}" (\n  '
        + ",\n  ".join(col_defs)
        + "\n)"
    )


def snapshot_one_table(table: str, mem_cap_bytes: int = 100 * 1024 * 1024) -> dict:
    """Atomically refresh app_db_snapshot.<table> from the App DB via COPY."""
    started = time.time()

    with app_db_conn() as src, backend_conn() as dst:
        cols = _columns_of(src, "public", table)
        if not cols:
            raise RuntimeError(f'Table public."{table}" not found in App DB.')
        col_names = [c[0] for c in cols]
        col_list_quoted = ", ".join(f'"{c}"' for c in col_names)

        staging = f"_staging_{table}"

        exec_sql(dst, f'DROP TABLE IF EXISTS "app_db_snapshot"."{staging}"')
        exec_sql(dst, _build_create_table("app_db_snapshot", staging, cols))

        # Buffer that lives in RAM until it exceeds mem_cap_bytes, then spills to disk.
        # This bounds memory use even for very large tables.
        with tempfile.SpooledTemporaryFile(max_size=mem_cap_bytes, mode="w+b") as buf:
            # ---- Source: COPY out of App DB as CSV ----
            src_copy_sql = (
                f'COPY (SELECT {col_list_quoted} FROM public."{table}") '
                f'TO STDOUT WITH (FORMAT CSV, NULL \'\\N\', FORCE_QUOTE *)'
            )
            with src.cursor() as scur:
                scur.copy_expert(src_copy_sql, buf)

            # ---- Destination: COPY into staging as CSV ----
            buf.seek(0)
            dst_copy_sql = (
                f'COPY "app_db_snapshot"."{staging}" ({col_list_quoted}) '
                f'FROM STDIN WITH (FORMAT CSV, NULL \'\\N\')'
            )
            with dst.cursor() as dcur:
                dcur.copy_expert(dst_copy_sql, buf)

        # Count what landed (cheap, and useful for the log)
        rows = query(dst, f'SELECT count(*)::int AS n FROM "app_db_snapshot"."{staging}"')
        rows_synced = rows[0]["n"]

        # Atomic swap
        exec_sql(dst, f'DROP TABLE IF EXISTS "app_db_snapshot"."{table}"')
        exec_sql(dst, f'ALTER TABLE "app_db_snapshot"."{staging}" RENAME TO "{table}"')

    duration_ms = int((time.time() - started) * 1000)
    return {"rows_synced": rows_synced, "duration_ms": duration_ms}


def run_full_sync(tables: list[str] | None = None) -> None:
    tables = tables or TABLES
    print(f"[sync] starting; {len(tables)} table(s)")
    for t in tables:
        log_id: int | None = None
        try:
            with backend_conn() as conn:
                log_id = returning_id(conn, """
                    INSERT INTO ai_layer.sync_log (sync_type, table_name, status)
                    VALUES ('app_db_snapshot', %s, 'running')
                    RETURNING id
                """, [t])

            print(f"[sync] {t} ...", flush=True)
            result = snapshot_one_table(t)

            with backend_conn() as conn:
                exec_sql(conn, """
                    UPDATE ai_layer.sync_log
                    SET rows_synced = %s, duration_ms = %s,
                        status = 'success', completed_at = now()
                    WHERE id = %s
                """, [result["rows_synced"], result["duration_ms"], log_id])
            print(f"[sync] {t}: {result['rows_synced']} rows in {result['duration_ms']} ms")

        except Exception as e:
            err = str(e)[:500]
            print(f"[sync] {t}: FAILED - {err}", flush=True)
            if log_id is not None:
                with backend_conn() as conn:
                    exec_sql(conn, """
                        UPDATE ai_layer.sync_log
                        SET status = 'error', error_message = %s, completed_at = now()
                        WHERE id = %s
                    """, [err, log_id])
            # Continue with the next table rather than aborting the whole run
            continue

    print("[sync] done.")


if __name__ == "__main__":
    args = sys.argv[1:]
    run_full_sync(args if args else None)
