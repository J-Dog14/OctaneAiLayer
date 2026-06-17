"""
Snapshot a curated set of App DB tables into the backend's app_db_snapshot schema.

Design:
- For each table, infer columns from information_schema in the App DB.
- Build a CREATE TABLE in app_db_snapshot._staging_<table> with permissive types
  (enums and varchars become TEXT to avoid recreating enum types in the backend).
- Stream rows in batches via a named (server-side) cursor.
- After load, atomically DROP the live table and RENAME the staging table.

This means: while the sync is running, the previous "live" snapshot is still
queryable. There is never a moment where the live table has half the rows.

Run:
    python -m src.sync_app_db                # full sync
    python -m src.sync_app_db User Program   # just these tables
"""
from __future__ import annotations

import sys
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


def snapshot_one_table(table: str, batch_size: int = 1000) -> dict:
    """Atomically refresh app_db_snapshot.<table> from the App DB."""
    started = time.time()
    rows_synced = 0

    with app_db_conn() as src, backend_conn() as dst:
        cols = _columns_of(src, "public", table)
        if not cols:
            raise RuntimeError(f'Table public."{table}" not found in App DB.')
        col_names = [c[0] for c in cols]
        col_list_quoted = ", ".join(f'"{c}"' for c in col_names)

        staging = f"_staging_{table}"

        exec_sql(dst, f'DROP TABLE IF EXISTS "app_db_snapshot"."{staging}"')
        exec_sql(dst, _build_create_table("app_db_snapshot", staging, cols))

        placeholders = ", ".join(["%s"] * len(col_names))
        insert_sql = (
            f'INSERT INTO "app_db_snapshot"."{staging}" '
            f"({col_list_quoted}) VALUES ({placeholders})"
        )

        # Server-side cursor in source DB streams without loading whole table to memory.
        with src.cursor(name=f"sync_{table}") as scur:
            scur.itersize = batch_size
            scur.execute(f'SELECT {col_list_quoted} FROM public."{table}"')
            with dst.cursor() as dcur:
                while True:
                    batch = scur.fetchmany(batch_size)
                    if not batch:
                        break
                    dcur.executemany(insert_sql, batch)
                    rows_synced += len(batch)

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
