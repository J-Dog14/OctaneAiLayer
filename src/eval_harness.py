"""
Blind-eval harness for the recommender.

For a named athlete:
  1. Look up their athlete_uuid (fuzzy name match against analytics.d_athletes)
  2. Find their most recent coach-prescribed program via ai_layer.program_summaries
  3. Infer the focus the coach was working under (from goal_power/strength/etc.)
  4. Run the recommender on the athlete WITHOUT showing it the coach's program
  5. Diff the two programs component-by-component and persist the eval

Use this to find systematic drift: exercises we always pick that coaches don't,
exercises coaches always pick that we miss, plyo-cadence misses, dose drift,
lift-template-family misses. The aggregate `eval-summary` rolls these up.

CLI:
    python -m src.main eval-athlete "Connor Chicoli"
    python -m src.main eval-batch path/to/names.txt
    python -m src.main eval-summary
"""
from __future__ import annotations

import json
import re
from datetime import date, datetime
from decimal import Decimal
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from src.db import backend_conn, query
from src.payload_builder import _normalize_loose
from src.profiler import build_and_save, _json_default
from src.program_summarizer import (
    _extract_exercise_prescriptions,
    _save_prescriptions,
    save_summary,
    summarize_one,
)
from src.recommender import recommend_lift_program, save_markdown


# ----------------------------------------------------------------------------
# Schema bootstrap
# ----------------------------------------------------------------------------

_EVAL_RUNS_DDL = """
CREATE TABLE IF NOT EXISTS ai_layer.eval_runs (
    id                       SERIAL PRIMARY KEY,
    athlete_uuid             UUID,
    athlete_name             TEXT,
    age_group                TEXT,
    focus_used               TEXT,
    focus_source             TEXT,            -- 'coach_program' | 'manual'
    coach_program_id         INTEGER,         -- App DB Program.id
    coach_program_name       TEXT,
    coach_program_created_at TIMESTAMP,
    recommended_program_id   INTEGER REFERENCES ai_layer.recommended_programs(id),
    overall_overlap_score    NUMERIC,
    comparison               JSONB,
    created_at               TIMESTAMP DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_eval_runs_athlete
    ON ai_layer.eval_runs (athlete_uuid);
CREATE INDEX IF NOT EXISTS idx_eval_runs_focus
    ON ai_layer.eval_runs (focus_used);
"""


def ensure_eval_table() -> None:
    with backend_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(_EVAL_RUNS_DDL)
        conn.commit()


# ----------------------------------------------------------------------------
# Athlete + program lookup
# ----------------------------------------------------------------------------

def lookup_athlete_by_name(name: str) -> dict | None:
    """Fuzzy-match an athlete by name. Returns the best match or None.

    Tries: exact (case-insensitive) → contains → token-overlap. Ties are
    broken by recency of any assessment data we have.
    """
    target = name.strip().lower()
    if not target:
        return None
    with backend_conn() as conn:
        rows = query(conn, """
            SELECT athlete_uuid, name, email, age_group,
                   has_pitching_data, has_hitting_data
            FROM analytics.d_athletes
            WHERE name IS NOT NULL
        """)
    if not rows:
        return None
    # Exact match first
    for r in rows:
        if (r["name"] or "").strip().lower() == target:
            return r
    # Substring match
    contains = [r for r in rows if target in (r["name"] or "").lower()]
    if len(contains) == 1:
        return contains[0]
    if len(contains) > 1:
        # Pick the longest-matching shortest-name to avoid "Connor C" matching
        # both "Connor Chicoli" and "Connor Charles". Tie-break alphabetically.
        contains.sort(key=lambda r: (len(r["name"] or ""), r["name"] or ""))
        return contains[0]
    # Fuzzy fallback
    scored = []
    for r in rows:
        score = SequenceMatcher(None, target, (r["name"] or "").lower()).ratio()
        scored.append((score, r))
    scored.sort(reverse=True, key=lambda x: x[0])
    if scored and scored[0][0] >= 0.7:
        return scored[0][1]
    return None


def _user_name_columns(conn) -> list[str]:
    """Return any columns on app_db_snapshot.User that look name-related."""
    rows = query(conn, """
        SELECT column_name FROM information_schema.columns
        WHERE table_schema='app_db_snapshot' AND table_name='User'
          AND data_type IN ('text','character varying','varchar')
        ORDER BY ordinal_position
    """)
    cols = [r["column_name"] for r in rows]
    return [c for c in cols if "name" in c.lower()]


def find_user_in_snapshot_by_name(name: str) -> list[dict]:
    """Search app_db_snapshot.User for users matching the given name.

    Introspects which name-like columns exist on User (e.g. firstName/lastName,
    or fullName, or just name) and searches across them. Returns a list of
    candidates with id + the matched name field(s).
    """
    target = (name or "").strip().lower()
    if not target:
        return []
    with backend_conn() as conn:
        name_cols = _user_name_columns(conn)
        if not name_cols:
            raise RuntimeError(
                "app_db_snapshot.User has no name-like columns; can't search by name. "
                "Re-sync with `python -m src.main sync --tables User` first."
            )
        # Build a SELECT pulling id + every name column, and a WHERE that
        # matches case-insensitive against any of them or against a concatenation.
        select_cols = ['"id"::text AS id'] + [f'"{c}"' for c in name_cols]
        # Concatenation for "FirstName LastName" style match
        concat_expr = " || ' ' || ".join(f'COALESCE("{c}", \'\')' for c in name_cols)
        where_parts = [f'lower(COALESCE("{c}", \'\')) LIKE %s' for c in name_cols]
        where_parts.append(f"lower({concat_expr}) LIKE %s")
        like = f"%{target}%"
        params = [like] * len(where_parts)
        sql = (
            f'SELECT {", ".join(select_cols)} '
            f'FROM app_db_snapshot."User" '
            f'WHERE {" OR ".join(where_parts)}'
        )
        rows = query(conn, sql, params)
    return rows


def find_programs_for_user(user_id: str) -> list[dict]:
    """List non-archived App DB programs for a user, most recent first."""
    with backend_conn() as conn:
        rows = query(conn, """
            SELECT "id"::int AS id, "name", "createdAt", "endDate",
                   "programType", "isArchived"
            FROM app_db_snapshot."Program"
            WHERE "userId"::text = %s
              AND NOT COALESCE("isArchived", false)
            ORDER BY "createdAt" DESC NULLS LAST
        """, [user_id])
    return rows


def link_coach_program_by_name(athlete_name: str,
                                program_id_override: int | None = None,
                                ) -> dict:
    """Link a coach program from app_db_snapshot into ai_layer.program_summaries
    for an athlete whose email-based User link is broken.

    Workflow:
      1. Find the athlete in analytics.d_athletes by name → warehouse uuid
      2. Find matching user(s) in app_db_snapshot.User by name
      3. Find their most recent non-archived Program (or use override)
      4. Run the existing summarize_one, but override the athlete_uuid that
         gets saved to both ai_layer.program_summaries and
         ai_layer.program_exercise_prescriptions
    """
    # Step 1: warehouse athlete
    ath = lookup_athlete_by_name(athlete_name)
    if not ath:
        raise ValueError(f"No athlete named {athlete_name!r} in analytics.d_athletes")
    athlete_uuid = ath["athlete_uuid"]

    # Step 2: App DB user(s)
    candidates = find_user_in_snapshot_by_name(athlete_name)
    if not candidates:
        raise ValueError(
            f"No user found in app_db_snapshot.User matching {athlete_name!r}. "
            f"Either the User snapshot is stale (re-run sync) or the name "
            f"differs between the warehouse and the App DB."
        )
    if len(candidates) > 1:
        # Try to pick the one whose name field is closest to the target name
        target = athlete_name.strip().lower()
        scored = []
        for c in candidates:
            # All non-id columns are name parts; score by best match against any
            best = 0.0
            for k, v in c.items():
                if k == "id" or v is None:
                    continue
                best = max(best, SequenceMatcher(None, target, str(v).lower()).ratio())
            scored.append((best, c))
        scored.sort(reverse=True, key=lambda x: x[0])
        # If top score is ambiguous (multiple within 0.05), surface them
        top = scored[0][0]
        tight = [c for s, c in scored if top - s < 0.05]
        if len(tight) > 1:
            preview = [
                {k: v for k, v in c.items() if v}
                for c in tight[:5]
            ]
            raise ValueError(
                f"Multiple users matched {athlete_name!r} in app_db_snapshot.User. "
                f"Re-run with --user-id <id> to disambiguate. Candidates: {preview}"
            )
        user = scored[0][1]
    else:
        user = candidates[0]
    user_id = user["id"]
    print(f"[link] matched app db user id={user_id} from {athlete_name!r}")

    # Step 3: pick a program
    if program_id_override is not None:
        program_id = program_id_override
    else:
        programs = find_programs_for_user(user_id)
        if not programs:
            raise ValueError(
                f"User {user_id} has no non-archived Program rows in "
                f"app_db_snapshot. (Sync the Program table or check isArchived.)"
            )
        program_id = programs[0]["id"]
        print(f"[link] picked latest program id={program_id} \"{programs[0]['name']}\"")
        if len(programs) > 1:
            print(f"[link] (athlete has {len(programs)} non-archived programs; "
                  f"override with --program-id to pick a different one)")

    # Step 4: summarize with athlete_uuid override
    with backend_conn() as conn:
        s = summarize_one(conn, program_id)
        # Email-link probably came back NULL; force the warehouse uuid
        original_uuid = s.get("athlete_uuid")
        s["athlete_uuid"] = athlete_uuid
        sid = save_summary(conn, s)
        prescriptions = _extract_exercise_prescriptions(
            conn, program_id, athlete_uuid)
        n_pres = _save_prescriptions(conn, program_id, prescriptions)
        conn.commit()

    print(f"[link] saved program_summary id={sid} "
          f"({n_pres} exercise prescription rows)")
    print(f"[link]   warehouse athlete_uuid: {athlete_uuid}")
    print(f"[link]   email-derived uuid was: {original_uuid or '(none — confirms link was broken)'}")
    return {
        "athlete_uuid": athlete_uuid,
        "athlete_name": ath["name"],
        "user_id": user_id,
        "program_id": program_id,
        "summary_id": sid,
        "n_prescriptions": n_pres,
    }


def find_latest_coach_program(athlete_uuid: str) -> dict | None:
    """Return the athlete's most recent non-archived program summary.

    Reads from ai_layer.program_summaries (already built by `summarize-all`).
    """
    with backend_conn() as conn:
        rows = query(conn, """
            SELECT
                ps.program_id,
                ps.program_name,
                ps.created_at_app   AS created_at,
                ps.end_date_app     AS end_date,
                ps.program_type,
                ps.goal_power, ps.goal_strength, ps.goal_speed,
                ps.goal_size, ps.goal_mobility, ps.goal_coordination,
                ps.goal_durability,
                ps.is_archived
            FROM ai_layer.program_summaries ps
            WHERE ps.athlete_uuid = %s
              AND NOT COALESCE(ps.is_archived, false)
            ORDER BY ps.created_at_app DESC NULLS LAST
            LIMIT 1
        """, [athlete_uuid])
    return rows[0] if rows else None


def detect_focus_from_program(program: dict) -> str:
    """Infer one of our 5 focuses from a program's goals + programType.

    Priority: explicit programType → most-prominent athleteGoal* → default Power.
    """
    pt = (program.get("program_type") or "").upper()
    if "IN_SEASON" in pt or "IN-SEASON" in pt:
        return "In-Season"
    if "HYPERTROPHY" in pt or "SIZE" in pt:
        return "Hypertrophy"
    # Goal-driven fallback. The athleteGoals* booleans aren't always mutually
    # exclusive — pick the strongest signal that matches one of our focuses.
    if program.get("goal_power"):
        return "Power"
    if program.get("goal_speed"):
        return "Speed"
    if program.get("goal_strength"):
        return "Strength"
    if program.get("goal_size"):
        return "Hypertrophy"
    # Default fallback when no signal is present
    return "Power"


def _get_lift_bucket_lookup(conn, program_id: int) -> dict[int, str]:
    """Map exercise_id -> bucket using LiftToProgram metadata.

    There is no separate `Lift` table in the Octane App DB — LiftToProgram is
    the parent entity. We introspect what columns are on it (name/type/etc.)
    and try them in order. If nothing useful is there, returns an empty dict
    and the caller falls back to Exercise.type bucketing.

    Why this matters: ai_layer.program_exercise_prescriptions stores
    Exercise.type, but for accessories/core/cuff exercises inside a Lift
    workout (e.g. 'CORE', 'ACCESSORY'), that classification doesn't reveal
    the workout-day bucket. The LiftToProgram row tells us "this is the Lower
    Body workout", so all its exercises share that bucket label regardless of
    their individual Exercise.type.
    """
    out: dict[int, str] = {}
    try:
        # Introspect what columns LiftToProgram actually has
        col_rows = query(conn, """
            SELECT column_name FROM information_schema.columns
            WHERE table_schema='app_db_snapshot' AND table_name='LiftToProgram'
        """)
    except Exception as e:
        print(f"[eval] LiftToProgram column introspection failed: {type(e).__name__}")
        conn.rollback()
        return out
    cols = {c["column_name"] for c in col_rows}

    # Pick whichever bucket-signal column exists, in priority order
    candidate_cols = [c for c in ("type", "category", "name", "label", "title")
                      if c in cols]
    if not candidate_cols:
        print("[eval] LiftToProgram has no name/type/category column; "
              "falling back to Exercise.type for lift bucketing")
        return out
    primary_col = candidate_cols[0]

    select_extras = ", ".join(f'l2p."{c}"::text AS "{c}"' for c in candidate_cols)
    try:
        rows = query(conn, f"""
            SELECT b."exerciseId"::int AS exercise_id,
                   {select_extras}
            FROM app_db_snapshot."LiftToProgram" l2p
            JOIN app_db_snapshot."ExerciseToLift" b
              ON b."liftToProgramId" = l2p."id"
            WHERE l2p."programId" = %s
        """, [program_id])
    except Exception as e:
        print(f"[eval] lift bucket lookup failed: {type(e).__name__}: "
              f"{str(e)[:120]}")
        conn.rollback()
        return out
    for r in rows:
        ex_id = r.get("exercise_id")
        if ex_id is None or ex_id in out:
            continue
        # Try each candidate column in priority order
        for col in candidate_cols:
            raw = r.get(col)
            if raw:
                bucket = _normalize_lift_bucket(raw)
                if bucket != "Other":
                    out[ex_id] = bucket
                    break
        else:
            # All candidate cols gave "Other" — store the primary col raw value
            # in a way that still lets fallback to Exercise.type take over
            pass
    return out


def extract_coach_program_components(program_id: int) -> dict[str, list[dict]]:
    """Pull every exercise prescription for a coach program, grouped by category.

    Source: ai_layer.program_exercise_prescriptions (built by program_summarizer).
    Returns a dict: {'lift': [...], 'plyo': [...], 'prep': [...], 'bp': [...],
                     'hit': [...], 'me': [...]}.
    """
    with backend_conn() as conn:
        rows = query(conn, """
            SELECT category, exercise_id, exercise_name, exercise_type,
                   n_sets, reps_array, avg_reps,
                   plyo_intensity, plyo_ball_weight,
                   weight_unit, order_in_program
            FROM ai_layer.program_exercise_prescriptions
            WHERE program_id = %s
              AND exercise_name IS NOT NULL
            ORDER BY category, order_in_program NULLS LAST
        """, [program_id])
        # Pull the lift-bucket lookup (exercise_id → bucket via Lift.type).
        # Cheaper than re-querying per row.
        lift_bucket_by_exercise = _get_lift_bucket_lookup(conn, program_id)
    grouped: dict[str, list[dict]] = {
        "lift": [], "plyo": [], "prep": [], "bp": [], "hit": [], "me": [],
    }
    for r in rows:
        # Reconstruct a "n x reps" display string from n_sets + reps_array
        n_sets = r.get("n_sets")
        reps_arr = r.get("reps_array") or []
        if n_sets and reps_arr:
            # Use the first rep value if all same, else "varied"
            first = reps_arr[0]
            if all(rv == first for rv in reps_arr):
                r["raw_sets_x_reps"] = f"{n_sets} x {first}"
            else:
                r["raw_sets_x_reps"] = f"{n_sets} x {'/'.join(str(x) for x in reps_arr)}"
        elif n_sets and r.get("avg_reps"):
            r["raw_sets_x_reps"] = f"{n_sets} x {int(r['avg_reps'])}"
        else:
            r["raw_sets_x_reps"] = None
        cat = r.get("category")
        # Grouping keys for strict matching:
        #   lifts use bucket from PARENT Lift.type (not Exercise.type — that
        #     misses accessories/core inside a Lower-Body lift). Falls back to
        #     Exercise.type if the Lift-lookup didn't find a mapping.
        #   plyos use plyo_level (from plyo_intensity int)
        if cat == "lift":
            ex_id = r.get("exercise_id")
            bucket = lift_bucket_by_exercise.get(ex_id) if ex_id else None
            if not bucket or bucket == "Other":
                # Fallback: try Exercise.type normalization
                exc_type_bucket = _normalize_lift_bucket(r.get("exercise_type"))
                if exc_type_bucket != "Other":
                    bucket = exc_type_bucket
            r["bucket"] = bucket or "Other"
        elif cat == "plyo":
            r["plyo_level"] = _plyo_intensity_to_level(r.get("plyo_intensity"))
        if cat in grouped:
            grouped[cat].append(r)
    return grouped


def _count_coach_plyo_prescriptions(program_id: int) -> int:
    """How many plyo drill rows did the coach assign?"""
    with backend_conn() as conn:
        rows = query(conn, """
            SELECT COUNT(*) AS n
            FROM ai_layer.program_exercise_prescriptions
            WHERE program_id = %s AND category = 'plyo'
        """, [program_id])
    return int(rows[0]["n"]) if rows else 0


def _count_coach_hit_prescriptions(program_id: int) -> int:
    """How many hitting drill rows did the coach assign?"""
    with backend_conn() as conn:
        rows = query(conn, """
            SELECT COUNT(*) AS n
            FROM ai_layer.program_exercise_prescriptions
            WHERE program_id = %s AND category = 'hit'
        """, [program_id])
    return int(rows[0]["n"]) if rows else 0


# Day-of-week mapping from postgres EXTRACT(DOW): 0=Sunday … 6=Saturday
_DOW_TO_NAME = {
    0: "SUNDAY",    1: "MONDAY",   2: "TUESDAY", 3: "WEDNESDAY",
    4: "THURSDAY",  5: "FRIDAY",   6: "SATURDAY",
}


_PLYO_CADENCE_SQL_CANDIDATES = [
    # Variant 0: ProgramDay.appliedProgramPlyoId → PlyoToProgram → Plyo.
    # Plyo.intensity is an int (0/1/2/3) that we map to 'P0'/'P1'/'P2'/'P3'.
    # Day-of-week is derived from ProgramDay.programDayDate.
    ("""
        SELECT EXTRACT(DOW FROM pd."programDayDate")::int AS dow,
               ('P' || p."intensity"::text)            AS plyo_level
        FROM app_db_snapshot."ProgramDay" pd
        JOIN app_db_snapshot."PlyoToProgram" ptp ON ptp."id" = pd."appliedProgramPlyoId"
        JOIN app_db_snapshot."Plyo" p ON p."id" = ptp."plyoId"
        WHERE pd."programId" = %s
          AND pd."appliedProgramPlyoId" IS NOT NULL
          AND pd."programDayDate" IS NOT NULL
          AND NOT COALESCE(pd."isArchived", false)
        ORDER BY pd."programDayDate"
    """),
    # Variant 1: same path but appliedProgramPlyoId points directly at Plyo
    # (skipping the junction). Fallback in case the schema differs.
    ("""
        SELECT EXTRACT(DOW FROM pd."programDayDate")::int AS dow,
               ('P' || p."intensity"::text)            AS plyo_level
        FROM app_db_snapshot."ProgramDay" pd
        JOIN app_db_snapshot."Plyo" p ON p."id" = pd."appliedProgramPlyoId"
        WHERE pd."programId" = %s
          AND pd."appliedProgramPlyoId" IS NOT NULL
          AND pd."programDayDate" IS NOT NULL
          AND NOT COALESCE(pd."isArchived", false)
        ORDER BY pd."programDayDate"
    """),
]


_PLYO_DIAGNOSTIC_PRINTED = False


def _print_plyo_schema_diagnostic(conn) -> None:
    """Print every plyo-related table + columns from the snapshot, so we can
    see what the actual schema looks like. Defensive — any one query failure
    rolls back and continues with the rest. Runs once per process."""
    global _PLYO_DIAGNOSTIC_PRINTED
    if _PLYO_DIAGNOSTIC_PRINTED:
        return
    _PLYO_DIAGNOSTIC_PRINTED = True

    def _safe_query(sql: str, params: list | None = None) -> list[dict]:
        try:
            return query(conn, sql, params or [])
        except Exception as e:
            try:
                conn.rollback()
            except Exception:
                pass
            print(f"[eval][diag]   ⚠ query failed: {type(e).__name__}: {str(e)[:120]}")
            return []

    print("[eval][diag] introspecting app_db_snapshot for plyo-related schema...")

    # 1. List every plyo-related table.
    # NOTE: percent signs in literal LIKE patterns must be DOUBLED when going
    # through psycopg2's parameter substitution — otherwise '%p' is parsed as
    # a placeholder and raises IndexError when no params are provided.
    tables = _safe_query("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'app_db_snapshot'
          AND lower(table_name) LIKE %s
        ORDER BY table_name
    """, ["%plyo%"])
    print(f"[eval][diag] {len(tables)} plyo-related table(s):")

    for t in tables:
        tname = t.get("table_name")
        if not tname:
            continue
        cols = _safe_query("""
            SELECT column_name, data_type FROM information_schema.columns
            WHERE table_schema='app_db_snapshot' AND table_name=%s
            ORDER BY ordinal_position
        """, [tname])
        col_str = ", ".join(
            f"{c.get('column_name')}:{c.get('data_type')}" for c in cols if c.get("column_name")
        ) or "(no columns)"
        print(f"[eval][diag]   {tname}({col_str})")

    # 2. ProgramDay columns — does it directly carry a plyo_level / plyoId?
    pd_cols = _safe_query("""
        SELECT column_name, data_type FROM information_schema.columns
        WHERE table_schema='app_db_snapshot' AND table_name='ProgramDay'
        ORDER BY ordinal_position
    """)
    if pd_cols:
        print(f"[eval][diag]   ProgramDay("
              + ", ".join(f"{c.get('column_name')}:{c.get('data_type')}"
                          for c in pd_cols if c.get("column_name"))
              + ")")
    else:
        print("[eval][diag]   ProgramDay: (table not found in snapshot)")

    # 3. Look for ANY table in the snapshot that has a plyo-FK-style column —
    # auto-discovers the cadence-mapping table no matter what it's named.
    candidates = _safe_query("""
        SELECT table_name
        FROM information_schema.columns
        WHERE table_schema='app_db_snapshot'
          AND lower(column_name) LIKE %s
        GROUP BY table_name
    """, ["%plyo%id%"])
    if candidates:
        print(f"[eval][diag] tables with a *plyoId-like* column:")
        for c in candidates:
            cname = c.get("table_name")
            cols = _safe_query("""
                SELECT column_name FROM information_schema.columns
                WHERE table_schema='app_db_snapshot' AND table_name=%s
                ORDER BY ordinal_position
            """, [cname])
            col_names = [r.get("column_name") for r in cols if r.get("column_name")]
            print(f"[eval][diag]   {cname}: {col_names}")


def extract_coach_plyo_cadence(program_id: int) -> dict[str, str | None]:
    """Pull the per-day plyo level for a coach program, if available.

    The exact junction table varies across App DB schema versions, so we try a
    few known variants in order and return whichever one yields data. When all
    variants fail, we print a schema diagnostic (once per process) so we can
    see what the actual snapshot looks like and write the right query.

    Returns {day_of_week: 'P0'|'P1'|'P2'|'P3'|None}.
    """
    rows: list[dict] = []
    errors: list[str] = []
    with backend_conn() as conn:
        for i, sql in enumerate(_PLYO_CADENCE_SQL_CANDIDATES):
            try:
                rows = query(conn, sql, [program_id])
                if rows:
                    break  # found it
            except Exception as e:
                errors.append(f"variant {i}: {type(e).__name__}: {str(e)[:120]}")
                conn.rollback()
                continue
        if not rows:
            # All variants either errored or returned no rows. Print the
            # snapshot's plyo-related schema once so Joey can read off the
            # right table/column names.
            _print_plyo_schema_diagnostic(conn)
            if errors:
                print(f"[eval] plyo cadence query — none of {len(errors)} "
                      f"variants matched. Errors: {'; '.join(errors)}")
            else:
                print(f"[eval] plyo cadence query — no error but no rows "
                      f"returned for program {program_id}")
    cadence: dict[str, str | None] = {
        d: None for d in
        ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY",
         "FRIDAY", "SATURDAY", "SUNDAY"]
    }
    for r in rows:
        # Two possible shapes depending on which variant matched:
        #   - 'dow' (int 0-6 from EXTRACT(DOW FROM date))
        #   - 'day_of_week' (string from older PlyoToProgramDay-style queries)
        day_name: str | None = None
        if r.get("dow") is not None:
            day_name = _DOW_TO_NAME.get(int(r["dow"]))
        else:
            raw = r.get("day_of_week")
            if raw:
                day_name = str(raw).upper()
        if day_name and day_name in cadence and cadence[day_name] is None:
            cadence[day_name] = r.get("plyo_level")
    return cadence


# ----------------------------------------------------------------------------
# Comparison logic
# ----------------------------------------------------------------------------

def _norm_name(name: str | None) -> str:
    """Match comparison uses the same loose normalization the payload builder uses."""
    return _normalize_loose(name or "")


def _parse_sets_reps(sxr: str | None) -> tuple[int | None, int | None]:
    """'3 x 5' -> (3, 5). '1 x 60s' -> (1, 60)."""
    if not sxr:
        return None, None
    m = re.match(r"^\s*(\d+)\s*[x×]\s*(\d+)", str(sxr).lower())
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def _dose_matches(coach_sxr: str | None, rec_sxr: str | None,
                  rec_sets: int | None, rec_reps: int | None) -> bool:
    """Are the doses equivalent? Tolerates 1-rep / 0-set difference."""
    c_sets, c_reps = _parse_sets_reps(coach_sxr)
    if rec_sxr is not None:
        r_sets, r_reps = _parse_sets_reps(rec_sxr)
    else:
        r_sets, r_reps = rec_sets, rec_reps
    if c_sets is None or r_sets is None:
        return False
    return (abs(c_sets - r_sets) <= 0 and
            abs((c_reps or 0) - (r_reps or 0)) <= 1)


def _rec_component_exercises(rec_payload: dict, component: str) -> list[dict]:
    """Normalize the recommender's component output into a flat list.

    Returns rows with name, sets, reps, dose, and grouping keys (bucket for
    lifts, plyo_level for plyo).
    """
    if component == "lift":
        out = []
        lp = rec_payload.get("lift_program") or {}
        for tpl in lp.get("templates") or []:
            for ex in tpl.get("exercises") or []:
                out.append({
                    "name": ex.get("name"),
                    "dose": ex.get("sets_x_reps"),
                    "sets": None, "reps": None,
                    "bucket": _normalize_lift_bucket(tpl.get("movement_bucket")),
                })
        # Pitcher mobility cap (Dead Hang / Shoulder CAR / Wrist ISO / etc.) —
        # these run alongside every lift day. They don't belong to a specific
        # bucket, so we tag them "Other" to align with where coach's similar
        # exercises typically land (their Exercise.type doesn't fit our 5
        # bucket map either).
        for ex in (lp.get("mobility_cap") or {}).get("exercises") or []:
            out.append({
                "name": ex.get("name"),
                "dose": ex.get("sets_x_reps"),
                "sets": None, "reps": None,
                "bucket": "Other",
            })
        return out
    if component == "plyo":
        # Aggregate across all 3 plyo sub-sessions
        out = []
        plyo = rec_payload.get("plyo_program") or {}
        for sess in plyo.get("cycle") or []:
            for d in sess.get("drills") or []:
                out.append({
                    "name": d.get("exercise_name"),
                    "dose": None,
                    "sets": d.get("sets"), "reps": d.get("reps"),
                    "plyo_level": (sess.get("plyo_level") or "").upper() or None,
                    "ball_weight": d.get("ball_weight"),
                })
        return out
    # prep / bp / hit / me
    program_key = {
        "prep": "prep_program",
        "bp": "bulletproofing_program",
        "hit": "hitting_program",
        "me": "movement_enhancement_program",
    }.get(component)
    block = rec_payload.get(program_key) or {}
    out = []
    for ex in block.get("exercises") or []:
        out.append({
            "name": ex.get("exercise_name"),
            "dose": None,
            "sets": ex.get("sets"), "reps": ex.get("reps"),
        })
    return out


# Threshold at which a fuzzy match counts as the same exercise.
# Slightly stricter than the payload-builder's 0.88 since we'd rather under-
# count an overlap than over-count a false positive.
_EVAL_FUZZY_THRESHOLD = 0.90


# Canonical lift bucket names — what we use for grouping. The recommender
# outputs Legs/Upper/Total Body/Sprint/Jump. Coach-side `exercise_type` values
# (from app_db_snapshot.Exercise.type) vary in casing/wording; this map
# normalizes them. Unknown values fall into "Other" and get reported but
# don't contribute to per-bucket matches.
_LIFT_BUCKETS = ("Legs", "Upper", "Total Body", "Sprint", "Jump", "Other")


def _normalize_lift_bucket(raw: str | None) -> str:
    """Map any exercise_type / bucket label string to one of our 5 canonical
    lift buckets. Defaults to 'Other' if unmappable so the row is still
    countable but doesn't false-match against the wrong bucket."""
    if not raw:
        return "Other"
    s = str(raw).strip().lower()
    # Lower body family
    if any(k in s for k in ("leg", "lower")):
        return "Legs"
    # Upper body family
    if "upper" in s or s == "ub":
        return "Upper"
    # Total body
    if "total" in s or "full body" in s or s == "tb":
        return "Total Body"
    # Sprint / speed
    if any(k in s for k in ("sprint", "speed", "run")):
        return "Sprint"
    # Jump / plyometric (the LIFT category, not the throwing plyo)
    if any(k in s for k in ("jump", "plyo")):
        return "Jump"
    return "Other"


def _plyo_intensity_to_level(intensity: int | None) -> str | None:
    """Map plyo_intensity int (0/1/2/3) to 'P0'..'P3'. None → None."""
    if intensity is None:
        return None
    try:
        return f"P{int(intensity)}"
    except (TypeError, ValueError):
        return None


def _fuzzy_pair(coach_keys: list[str], rec_keys: list[str]
                ) -> list[tuple[str, str, float]]:
    """For remaining unmatched keys, find best 1:1 fuzzy pairs ≥ threshold.

    Returns (coach_key, rec_key, score) triples. Uses greedy best-first matching
    so each key on either side is paired with at most one counterpart.
    """
    if not coach_keys or not rec_keys:
        return []
    scored: list[tuple[float, str, str]] = []
    for ck in coach_keys:
        for rk in rec_keys:
            s = SequenceMatcher(None, ck, rk).ratio()
            if s >= _EVAL_FUZZY_THRESHOLD:
                scored.append((s, ck, rk))
    scored.sort(reverse=True, key=lambda x: x[0])
    used_c: set[str] = set()
    used_r: set[str] = set()
    pairs: list[tuple[str, str, float]] = []
    for s, ck, rk in scored:
        if ck in used_c or rk in used_r:
            continue
        used_c.add(ck)
        used_r.add(rk)
        pairs.append((ck, rk, s))
    return pairs


def _compare_component(coach_rows: list[dict], rec_rows: list[dict]) -> dict:
    """Diff one component (lift/plyo/prep/bp/hit/me).

    Match logic:
      1. Loose-normalized exact equality (handles case/punctuation/curly-quote drift)
      2. Fuzzy fallback (difflib ≥ 0.90) for remaining names — catches plural
         drift like 'Pullthrough' vs 'Pullthroughs'

    Returns:
      n_coach, n_rec, n_intersection
      intersection: [{name, coach_dose, rec_dose, dose_match, match_method}]
      rec_added: [name]   # in rec, not in coach
      rec_missed: [name]  # in coach, not in rec
      precision, recall, f1, jaccard, dose_alignment (all 0–1 or None)
    """
    coach_idx: dict[str, dict] = {}
    for r in coach_rows:
        nk = _norm_name(r.get("exercise_name") or r.get("name"))
        if nk:
            coach_idx[nk] = {
                "display": r.get("exercise_name") or r.get("name"),
                "dose": r.get("raw_sets_x_reps") or r.get("dose"),
            }
    rec_idx: dict[str, dict] = {}
    for r in rec_rows:
        nk = _norm_name(r.get("name"))
        if nk:
            rec_idx[nk] = {
                "display": r.get("name"),
                "dose": r.get("dose"),
                "sets": r.get("sets"),
                "reps": r.get("reps"),
            }

    coach_keys = set(coach_idx.keys())
    rec_keys = set(rec_idx.keys())

    # Stage 1: exact normalized intersection
    exact_inter = coach_keys & rec_keys
    # Stage 2: fuzzy fallback on the leftovers
    leftover_coach = coach_keys - exact_inter
    leftover_rec = rec_keys - exact_inter
    fuzzy_pairs = _fuzzy_pair(sorted(leftover_coach), sorted(leftover_rec))

    intersection = []
    dose_matches_n = 0
    matched_coach: set[str] = set()
    matched_rec: set[str] = set()
    for k in sorted(exact_inter):
        c = coach_idx[k]
        r = rec_idx[k]
        dm = _dose_matches(c["dose"], r["dose"], r.get("sets"), r.get("reps"))
        if dm:
            dose_matches_n += 1
        intersection.append({
            "name": c["display"],
            "coach_dose": c["dose"],
            "rec_dose": r["dose"] or (
                f"{r['sets']} x {r['reps']}" if r.get("sets") and r.get("reps") else None),
            "dose_match": dm,
            "match_method": "exact",
        })
        matched_coach.add(k)
        matched_rec.add(k)
    for ck, rk, score in fuzzy_pairs:
        c = coach_idx[ck]
        r = rec_idx[rk]
        dm = _dose_matches(c["dose"], r["dose"], r.get("sets"), r.get("reps"))
        if dm:
            dose_matches_n += 1
        intersection.append({
            "name": c["display"],
            "rec_name": r["display"],
            "coach_dose": c["dose"],
            "rec_dose": r["dose"] or (
                f"{r['sets']} x {r['reps']}" if r.get("sets") and r.get("reps") else None),
            "dose_match": dm,
            "match_method": f"fuzzy:{score:.2f}",
        })
        matched_coach.add(ck)
        matched_rec.add(rk)

    rec_added = sorted([rec_idx[k]["display"] for k in rec_keys - matched_rec])
    rec_missed = sorted([coach_idx[k]["display"] for k in coach_keys - matched_coach])

    n_inter = len(intersection)
    n_c = len(coach_keys)
    n_r = len(rec_keys)
    # F1 = 2 * P * R / (P + R)
    precision = (n_inter / n_r) if n_r else None
    recall = (n_inter / n_c) if n_c else None
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    union_n = n_c + n_r - n_inter
    jaccard = (n_inter / union_n) if union_n else None
    dose_alignment = (dose_matches_n / n_inter) if n_inter else None

    return {
        "n_coach": n_c,
        "n_rec": n_r,
        "n_intersection": n_inter,
        "intersection": intersection,
        "rec_added": rec_added,
        "rec_missed": rec_missed,
        "precision": round(precision, 3) if precision is not None else None,
        "recall": round(recall, 3) if recall is not None else None,
        "f1": round(f1, 3) if f1 is not None else None,
        "jaccard": round(jaccard, 3) if jaccard is not None else None,
        "dose_alignment": round(dose_alignment, 3) if dose_alignment is not None else None,
    }


def _compare_component_grouped(coach_rows: list[dict], rec_rows: list[dict],
                                group_key: str) -> dict:
    """Strict group-aware comparison. Same as _compare_component but exercises
    only match when they share BOTH the same (normalized) name AND the same
    group key (e.g., lift bucket or plyo level).

    Returns the same shape as _compare_component, plus:
      per_group: {group_value: <full _compare_component dict>}
    Per-group precision/recall/f1 lets us see which buckets/levels are weak.
    """
    coach_by_group: dict[str, list[dict]] = {}
    rec_by_group: dict[str, list[dict]] = {}
    for r in coach_rows:
        g = r.get(group_key) or "Other"
        coach_by_group.setdefault(g, []).append(r)
    for r in rec_rows:
        g = r.get(group_key) or "Other"
        rec_by_group.setdefault(g, []).append(r)

    all_groups = sorted(set(coach_by_group) | set(rec_by_group))
    per_group: dict[str, dict] = {}
    n_c = n_r = n_inter = dose_n = 0
    intersection_all: list[dict] = []
    rec_added_all: list[str] = []
    rec_missed_all: list[str] = []
    for g in all_groups:
        sub = _compare_component(coach_by_group.get(g, []), rec_by_group.get(g, []))
        sub["group"] = g
        per_group[g] = sub
        n_c += sub["n_coach"]
        n_r += sub["n_rec"]
        n_inter += sub["n_intersection"]
        if sub.get("dose_alignment") is not None:
            dose_n += int(round(sub["dose_alignment"] * sub["n_intersection"]))
        for ex in sub["intersection"]:
            ex["group"] = g
            intersection_all.append(ex)
        rec_added_all.extend(f"[{g}] {n}" for n in sub["rec_added"])
        rec_missed_all.extend(f"[{g}] {n}" for n in sub["rec_missed"])

    precision = (n_inter / n_r) if n_r else None
    recall = (n_inter / n_c) if n_c else None
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    union_n = n_c + n_r - n_inter
    jaccard = (n_inter / union_n) if union_n else None
    dose_alignment = (dose_n / n_inter) if n_inter else None

    return {
        "n_coach": n_c, "n_rec": n_r, "n_intersection": n_inter,
        "intersection": intersection_all,
        "rec_added": sorted(rec_added_all),
        "rec_missed": sorted(rec_missed_all),
        "precision": round(precision, 3) if precision is not None else None,
        "recall": round(recall, 3) if recall is not None else None,
        "f1": round(f1, 3) if f1 is not None else None,
        "jaccard": round(jaccard, 3) if jaccard is not None else None,
        "dose_alignment": round(dose_alignment, 3) if dose_alignment is not None else None,
        "per_group": per_group,
        "group_key": group_key,
    }


def _compare_plyo_cadence(coach_cadence: dict[str, str | None],
                          rec_layout: dict[str, str] | None) -> dict:
    rec_layout = rec_layout or {}
    days = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY",
            "FRIDAY", "SATURDAY", "SUNDAY"]
    matches = 0
    rows = []
    for d in days:
        coach_v = (coach_cadence.get(d) or "").upper() or None
        # rec_layout values are full strings like "P2 - Mound Day" — extract token
        rv = (rec_layout.get(d) or "").upper()
        rec_v = None
        for token in ("P0", "P1", "P2", "P3"):
            if token in rv:
                rec_v = token
                break
        match = (coach_v == rec_v) and (coach_v is not None)
        if match:
            matches += 1
        rows.append({"day": d, "coach": coach_v, "rec": rec_v, "match": match})
    coach_days_set = sum(1 for d in days if coach_cadence.get(d))
    return {
        "days": rows,
        "n_days_with_coach_data": coach_days_set,
        "n_matches": matches,
        "match_rate": round(matches / coach_days_set, 3) if coach_days_set else None,
    }


def _compare_lift_template_family(rec_payload: dict,
                                  coach_lift_exercises: list[dict]
                                  ) -> dict:
    """Did the recommender land in the same lift template family the coach used?

    Hard to compare directly because the coach program doesn't carry our
    spreadsheet's template_id. As a proxy, check whether the recommender's
    selected family is consistent with the coach's general exercise picks.
    Mostly informational — populated when we can derive it.
    """
    rec_template_ids = rec_payload.get("selected_template_ids") or []
    # Recommender stores the picked family ids; we just surface them. The
    # "did we match coach's family" question is open until coach programs
    # carry an explicit template_id. For now, return the rec's IDs + the
    # families they fall under so coach can eyeball it.
    families = sorted({str(t).split("-")[0] for t in rec_template_ids if t})
    return {
        "rec_template_ids": rec_template_ids,
        "rec_families": families,
        "coach_template_id": None,  # not stored on coach programs today
        "family_match": None,
    }


def _overall_overlap_score(component_diffs: dict) -> float:
    """Weighted average RECALL (% of coach picks we also picked) across components.

    Why recall, not F1: coaches START from the full spreadsheet template and
    then TRIM down to their final program. The recommender outputs the full
    template (~55 lift exercises). So having more picks than the coach is by
    design, not over-prescription — Recall is the honest measure of "did we
    cover what the coach kept".

    Weights: lift 0.30, plyo 0.30, prep 0.30, bp 0.10.
    """
    weights = {"lift": 0.30, "plyo": 0.30, "prep": 0.30, "bp": 0.10, "hit": 0.0}
    total_w = 0.0
    weighted = 0.0
    for comp, d in component_diffs.items():
        if comp == "me":
            continue  # merged into prep before comparison
        r = d.get("recall")
        if r is None:
            continue
        w = weights.get(comp, 0.0)
        weighted += r * w
        total_w += w
    return round(weighted / total_w, 3) if total_w else 0.0


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------

def run_eval_for_athlete(
    athlete_name: str,
    focus_override: str | None = None,
    as_of_date: str | None = None,
    athlete_role: str = "Starter",
    skip_profile: bool = False,
) -> dict:
    """Run one full eval against an athlete by name. Returns the eval dict."""
    ensure_eval_table()

    # 1. Find athlete
    athlete = lookup_athlete_by_name(athlete_name)
    if not athlete:
        raise ValueError(f"No athlete found matching name: {athlete_name!r}")
    print(f"[eval] athlete: {athlete['name']}  ({athlete['athlete_uuid']})")

    # 2. Find their most recent coach program
    coach_program = find_latest_coach_program(athlete["athlete_uuid"])
    if not coach_program:
        raise ValueError(
            f"No coach-prescribed program found for {athlete['name']!r}. "
            f"This athlete must have at least one App DB program (synced into "
            f"ai_layer.program_summaries) to be eval'd."
        )
    print(f"[eval] coach program: id={coach_program['program_id']} "
          f"\"{coach_program['program_name']}\"")

    # 3. Detect focus (unless overridden)
    if focus_override:
        focus = focus_override
        focus_source = "manual"
    else:
        focus = detect_focus_from_program(coach_program)
        focus_source = "coach_program"
    print(f"[eval] focus: {focus} (source: {focus_source})")

    # 4. Make sure the athlete has a profile (rebuild if not skipped)
    as_of = as_of_date or date.today().isoformat()
    if not skip_profile:
        print(f"[eval] rebuilding profile as of {as_of}...")
        build_and_save(athlete["athlete_uuid"], as_of)

    # 5. Run the recommender (blind — we don't show it the coach's program)
    # Derive role from data-presence flags, with a fallback: if the coach's
    # actual program has plyo prescriptions, the athlete IS a pitcher regardless
    # of what d_athletes.has_pitching_data says (the flag is sometimes stale
    # for athletes inserted directly into the warehouse).
    has_p = athlete.get("has_pitching_data")
    has_h = athlete.get("has_hitting_data")
    coach_plyo_n = _count_coach_plyo_prescriptions(coach_program["program_id"])
    coach_hit_n = _count_coach_hit_prescriptions(coach_program["program_id"])
    if not has_p and coach_plyo_n > 0:
        print(f"[eval] d_athletes.has_pitching_data=False but coach assigned "
              f"{coach_plyo_n} plyo drills → treating as pitcher")
        has_p = True
    if not has_h and coach_hit_n > 0:
        print(f"[eval] d_athletes.has_hitting_data=False but coach assigned "
              f"{coach_hit_n} hitting drills → treating as hitter")
        has_h = True
    if has_p and has_h:
        role_hint = "both"
    elif has_h and not has_p:
        role_hint = "hitter"
    elif has_p:
        role_hint = "pitcher"
    else:
        role_hint = None
    print(f"[eval] running recommender (this is the LLM cost)... role={role_hint}")
    # In eval mode, we KNOW from the coach's program whether this athlete is
    # actually being trained as a pitcher / hitter. Patch the profile *before*
    # recommendation so the plyo / hitting gates see the truth, even when both
    # d_athletes flags AND the fact-table fallback say otherwise (which can
    # happen for athletes whose data was inserted via a non-standard path).
    from src.recommender import load_athlete_profile  # late import; avoid cycle
    import src.recommender as _rec_mod
    _orig_loader = _rec_mod.load_athlete_profile

    def _patched_loader(uuid: str) -> dict:
        p = _orig_loader(uuid)
        if uuid == athlete["athlete_uuid"]:
            if coach_plyo_n > 0 and not p.get("has_pitching_data"):
                print(f"[eval] forcing has_pitching_data=True on profile "
                      f"(coach assigned {coach_plyo_n} plyo drills)")
                p["has_pitching_data"] = True
            if coach_hit_n > 0 and not p.get("has_hitting_data"):
                print(f"[eval] forcing has_hitting_data=True on profile "
                      f"(coach assigned {coach_hit_n} hitting drills)")
                p["has_hitting_data"] = True
        return p

    _rec_mod.load_athlete_profile = _patched_loader
    try:
        rec = recommend_lift_program(
            athlete_uuid=athlete["athlete_uuid"],
            focus=focus,
            role=role_hint,
            athlete_role=athlete_role,
        )
    finally:
        _rec_mod.load_athlete_profile = _orig_loader
    save_markdown(rec)

    # 6. Diff coach vs recommended
    print("[eval] comparing programs...")
    coach_components = extract_coach_program_components(coach_program["program_id"])
    coach_cadence = extract_coach_plyo_cadence(coach_program["program_id"])
    rec_weekly_layout = (rec.get("plyo_program") or {}).get("weekly_layout") or {}

    # Merge coach's ME exercises into Prep — we explicitly folded ME into Prep
    # on the recommender side, so this is the apples-to-apples comparison.
    coach_prep_plus_me = (coach_components["prep"] or []) + (coach_components["me"] or [])
    component_diffs = {
        # Lift uses STRICT bucket-aware matching: an exercise only counts as
        # matched if it's in the same bucket (Legs/Upper/Total Body/Sprint/Jump)
        # on both sides. Reflects coach intent — "did we get a Squat into the
        # right movement-day bucket?"
        "lift": _compare_component_grouped(
            coach_components["lift"], _rec_component_exercises(rec, "lift"),
            group_key="bucket",
        ),
        # Plyo uses STRICT level-aware matching: a drill only counts if it's
        # in the same plyo level on both sides. P2 OG Pivots != P0 OG Pivots.
        "plyo": _compare_component_grouped(
            coach_components["plyo"], _rec_component_exercises(rec, "plyo"),
            group_key="plyo_level",
        ),
        "prep": _compare_component(coach_prep_plus_me,        _rec_component_exercises(rec, "prep")),
        "bp":   _compare_component(coach_components["bp"],    _rec_component_exercises(rec, "bp")),
        "hit":  _compare_component(coach_components["hit"],   _rec_component_exercises(rec, "hit")),
        # ME is informational only — we don't generate it as its own block
        "me":   _compare_component(coach_components["me"],    _rec_component_exercises(rec, "me")),
    }
    plyo_cadence_diff = _compare_plyo_cadence(coach_cadence, rec_weekly_layout)
    template_family_diff = _compare_lift_template_family(rec, coach_components["lift"])

    overall = _overall_overlap_score(component_diffs)

    comparison = {
        "by_component": component_diffs,
        "plyo_cadence": plyo_cadence_diff,
        "lift_template_family": template_family_diff,
    }

    eval_dict = {
        "athlete_uuid": athlete["athlete_uuid"],
        "athlete_name": athlete["name"],
        "age_group": athlete.get("age_group"),
        "focus_used": focus,
        "focus_source": focus_source,
        "coach_program_id": coach_program["program_id"],
        "coach_program_name": coach_program["program_name"],
        "coach_program_created_at": coach_program.get("created_at"),
        "recommended_program_id": rec["recommended_program_id"],
        "overall_overlap_score": overall,
        "comparison": comparison,
    }

    # 7. Persist
    eval_id = _save_eval(eval_dict)
    eval_dict["eval_id"] = eval_id

    # 8. Write the side-by-side markdown
    md_path = save_eval_markdown(eval_dict)
    eval_dict["markdown_path"] = str(md_path)

    return eval_dict


def _save_eval(eval_dict: dict) -> int:
    with backend_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO ai_layer.eval_runs (
                    athlete_uuid, athlete_name, age_group, focus_used, focus_source,
                    coach_program_id, coach_program_name, coach_program_created_at,
                    recommended_program_id, overall_overlap_score, comparison
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                RETURNING id
            """, [
                eval_dict["athlete_uuid"], eval_dict["athlete_name"],
                eval_dict["age_group"], eval_dict["focus_used"],
                eval_dict["focus_source"],
                eval_dict["coach_program_id"], eval_dict["coach_program_name"],
                eval_dict.get("coach_program_created_at"),
                eval_dict["recommended_program_id"],
                eval_dict["overall_overlap_score"],
                json.dumps(eval_dict["comparison"], default=_json_default),
            ])
            new_id = cur.fetchone()[0]
        conn.commit()
    return int(new_id)


# ----------------------------------------------------------------------------
# Markdown rendering
# ----------------------------------------------------------------------------

_COMPONENT_TITLES = {
    "lift": "Lift", "plyo": "Plyo", "prep": "Prep",
    "bp": "Bulletproofing", "hit": "Hitting", "me": "Movement Enhancement",
}


def save_eval_markdown(eval_dict: dict) -> Path:
    """Render the eval as a side-by-side markdown report."""
    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    safe_name = re.sub(r"\W+", "_", eval_dict["athlete_name"] or "athlete")
    path = out_dir / f"eval_{eval_dict.get('eval_id', 'NA')}_{safe_name}_{ts}.md"

    md = []
    md.append(f"# Eval — {eval_dict['athlete_name']}")
    md.append("")
    md.append(f"- **Age group:** {eval_dict.get('age_group')}")
    md.append(f"- **Focus used:** {eval_dict['focus_used']}  *(source: {eval_dict['focus_source']})*")
    md.append(f"- **Coach program:** id={eval_dict['coach_program_id']} — "
              f"\"{eval_dict['coach_program_name']}\"")
    md.append(f"- **Recommended program:** id={eval_dict['recommended_program_id']}")
    md.append(f"- **Overall recall (coverage of coach's picks):** "
              f"**{eval_dict['overall_overlap_score']:.1%}**")
    md.append("")

    # Headline metric table — Recall first as the key metric
    md.append("## At a glance")
    md.append("")
    md.append("| Component | Coach | Rec | Both | **Recall** | Precision | F1 | Dose |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for comp in ["lift", "plyo", "prep", "bp", "hit", "me"]:
        d = eval_dict["comparison"]["by_component"][comp]
        def _pct(v):
            return f"{v:.1%}" if v is not None else "—"
        note = " *(in Prep)*" if comp == "me" else ""
        recall_str = f"**{_pct(d.get('recall'))}**"
        md.append(
            f"| {_COMPONENT_TITLES[comp]}{note} | {d['n_coach']} | {d['n_rec']} | "
            f"{d['n_intersection']} | {recall_str} | "
            f"{_pct(d.get('precision'))} | {_pct(d.get('f1'))} | "
            f"{_pct(d.get('dose_alignment'))} |"
        )
    md.append("")
    md.append("*Recall = % of coach's picks that we also picked. "
              "**This is the headline metric** — coaches start from the full "
              "spreadsheet template and trim it down, while we output the full "
              "template, so it's expected our exercise count is higher than the "
              "coach's final program. Precision and F1 are shown for context but "
              "are de-prioritized given the design. Prep's coach count includes "
              "Movement Enhancement exercises since the recommender places them "
              "at the end of Prep.*")

    # Plyo cadence
    cadence = eval_dict["comparison"]["plyo_cadence"]
    md.append("")
    md.append("## Plyo cadence (by day)")
    md.append("")
    md.append(f"Coach data available for {cadence['n_days_with_coach_data']} day(s). "
              f"Match: **{cadence['n_matches']} / {cadence['n_days_with_coach_data']}**.")
    md.append("")
    md.append("| Day | Coach | Rec | Match |")
    md.append("|---|---|---|:---:|")
    for r in cadence["days"]:
        check = "✓" if r["match"] else ("·" if r["coach"] is None else "✗")
        md.append(f"| {r['day'][:3].title()} | {r['coach'] or '—'} | "
                  f"{r['rec'] or '—'} | {check} |")

    # Lift template family
    tf = eval_dict["comparison"]["lift_template_family"]
    md.append("")
    md.append("## Lift template families picked by recommender")
    md.append("")
    md.append(f"- Rec template IDs: `{tf['rec_template_ids']}`")
    md.append(f"- Rec families:     `{tf['rec_families']}`")
    md.append("")
    md.append("*(Coach programs don't carry template IDs today, so this is informational.)*")
    md.append("")

    # Per-component side-by-side, with group breakdown for lift + plyo
    for comp in ["lift", "plyo", "prep", "bp", "hit", "me"]:
        d = eval_dict["comparison"]["by_component"][comp]
        # If this component has a per-group breakdown, render it
        per_group = d.get("per_group") if isinstance(d, dict) else None
        if per_group:
            group_label = "Bucket" if comp == "lift" else "Plyo level"
            md.append(f"## {_COMPONENT_TITLES[comp]} — by {group_label.lower()}")
            md.append("")
            md.append(f"| {group_label} | Coach | Rec | Both | Recall | Precision |")
            md.append("|---|---:|---:|---:|---:|---:|")
            # Order: enforce a canonical order if known
            if comp == "lift":
                ordered = ["Legs", "Upper", "Total Body", "Sprint", "Jump", "Other"]
            else:
                ordered = ["P0", "P1", "P2", "P3"]
            for g in ordered:
                if g not in per_group:
                    # Still show row if either side has it; skip if neither
                    continue
                gd = per_group[g]
                if gd["n_coach"] == 0 and gd["n_rec"] == 0:
                    continue
                def _p(v):
                    return f"{v:.1%}" if v is not None else "—"
                md.append(f"| {g} | {gd['n_coach']} | {gd['n_rec']} | "
                          f"{gd['n_intersection']} | {_p(gd.get('recall'))} | "
                          f"{_p(gd.get('precision'))} |")
            md.append("")

    for comp in ["lift", "plyo", "prep", "bp", "hit", "me"]:
        d = eval_dict["comparison"]["by_component"][comp]
        if d["n_coach"] == 0 and d["n_rec"] == 0:
            continue
        title = _COMPONENT_TITLES[comp]
        md.append(f"## {title}")
        md.append("")
        line = (f"Coach: **{d['n_coach']}**   Rec: **{d['n_rec']}**   "
                f"Overlap: **{d['n_intersection']}**")
        parts: list[str] = []
        if d.get("recall") is not None:
            parts.append(f"**Recall: {d['recall']:.1%}**")
        if d.get("precision") is not None:
            parts.append(f"Precision: {d['precision']:.1%}")
        if d.get("f1") is not None:
            parts.append(f"F1: {d['f1']:.1%}")
        if parts:
            line += "   " + "   ".join(parts)
        md.append(line)
        md.append("")

        if d["intersection"]:
            md.append("### Both picked")
            md.append("")
            md.append("| Coach exercise | Rec exercise | Coach dose | Rec dose | Dose match |")
            md.append("|---|---|---|---|:---:|")
            for ex in d["intersection"]:
                check = "✓" if ex["dose_match"] else "✗"
                coach_name = ex["name"]
                rec_name = ex.get("rec_name") or ex["name"]
                method = ex.get("match_method") or "exact"
                if method.startswith("fuzzy"):
                    rec_name = f"{rec_name}  *({method})*"
                md.append(f"| {coach_name} | {rec_name} | {ex['coach_dose'] or '—'} | "
                          f"{ex['rec_dose'] or '—'} | {check} |")
            md.append("")

        if d["rec_added"]:
            md.append(f"### Recommender added ({len(d['rec_added'])}) — coach didn't pick")
            md.append("")
            for n in d["rec_added"]:
                md.append(f"- {n}")
            md.append("")

        if d["rec_missed"]:
            md.append(f"### Recommender missed ({len(d['rec_missed'])}) — coach picked, we didn't")
            md.append("")
            for n in d["rec_missed"]:
                md.append(f"- {n}")
            md.append("")

    path.write_text("\n".join(md), encoding="utf-8")
    return path


# ----------------------------------------------------------------------------
# Aggregate summary across many eval runs
# ----------------------------------------------------------------------------

def aggregate_eval_summary(focus: str | None = None) -> dict:
    """Roll up every eval_runs row into systematic-bias insights.

    Returns a dict with:
      n_evals, avg_overall_overlap, per_component_avg_jaccard,
      most_overprescribed (rec_added counts), most_missed (rec_missed counts),
      lift_template_family_distribution.
    """
    with backend_conn() as conn:
        where = ""
        params: list[Any] = []
        if focus:
            where = "WHERE focus_used = %s"
            params = [focus]
        rows = query(conn, f"""
            SELECT athlete_name, focus_used, overall_overlap_score, comparison
            FROM ai_layer.eval_runs
            {where}
            ORDER BY created_at DESC
        """, params)
    if not rows:
        return {"n_evals": 0}

    comp_f1: dict[str, list[float]] = {}
    comp_recall: dict[str, list[float]] = {}
    comp_precision: dict[str, list[float]] = {}
    comp_dose_aligns: dict[str, list[float]] = {}
    overprescribed: dict[str, int] = {}      # exercises rec adds that coach doesn't
    missed: dict[str, int] = {}              # exercises coach picks that rec misses
    template_family_counter: dict[str, int] = {}
    plyo_match_rates: list[float] = []

    for r in rows:
        comp = r["comparison"]["by_component"]
        for c_name, c_data in comp.items():
            if c_data.get("f1") is not None:
                comp_f1.setdefault(c_name, []).append(c_data["f1"])
            if c_data.get("recall") is not None:
                comp_recall.setdefault(c_name, []).append(c_data["recall"])
            if c_data.get("precision") is not None:
                comp_precision.setdefault(c_name, []).append(c_data["precision"])
            if c_data.get("dose_alignment") is not None:
                comp_dose_aligns.setdefault(c_name, []).append(c_data["dose_alignment"])
            for name in c_data.get("rec_added") or []:
                key = f"{c_name}::{name}"
                overprescribed[key] = overprescribed.get(key, 0) + 1
            for name in c_data.get("rec_missed") or []:
                key = f"{c_name}::{name}"
                missed[key] = missed.get(key, 0) + 1
        for fam in (r["comparison"]["lift_template_family"].get("rec_families") or []):
            template_family_counter[fam] = template_family_counter.get(fam, 0) + 1
        mr = r["comparison"]["plyo_cadence"].get("match_rate")
        if mr is not None:
            plyo_match_rates.append(mr)

    avg = lambda xs: round(sum(xs) / len(xs), 3) if xs else None

    return {
        "n_evals": len(rows),
        "avg_overall_overlap": avg([float(r["overall_overlap_score"] or 0) for r in rows]),
        "avg_plyo_cadence_match": avg(plyo_match_rates),
        "per_component_avg_f1":        {c: avg(v) for c, v in comp_f1.items()},
        "per_component_avg_recall":    {c: avg(v) for c, v in comp_recall.items()},
        "per_component_avg_precision": {c: avg(v) for c, v in comp_precision.items()},
        "per_component_avg_dose_alignment": {c: avg(v) for c, v in comp_dose_aligns.items()},
        "most_overprescribed":
            sorted(overprescribed.items(), key=lambda x: x[1], reverse=True)[:20],
        "most_missed":
            sorted(missed.items(), key=lambda x: x[1], reverse=True)[:20],
        "lift_template_family_distribution":
            sorted(template_family_counter.items(), key=lambda x: x[1], reverse=True),
    }
