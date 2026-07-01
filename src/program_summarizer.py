"""
Summarize each program in app_db_snapshot."Program" into ai_layer.program_summaries.

For each program we produce one row with:
  - Program metadata (denormalized from the App DB)
  - Athlete link (via User.email → analytics.d_athletes.email)
  - Goal scores (athleteGoals* fields)
  - Counts of prescriptions across lifts / plyos / prep / bulletproofing / hitting / movement-enhancement
  - JSONB breakdowns (e.g. {Exercise.type: count}) for richer downstream analysis
  - Linked weekly templates (when the program was instantiated from one)

The point: pair every athlete's deficit profile with a structured description of
what their coach actually prescribed. This pair is the training corpus for the
program recommender.

ASSUMPTIONS about join columns (Prisma camelCase conventions):
  Program.id              ←→ {Lift,Plyo,Prep,BP,Hit,ME,ProgramDay}ToProgram."programId"
  LiftToProgram."liftId"  ←→ ExerciseToLift."liftId"
  ExerciseToLift."exerciseId" ←→ Exercise."id"
  Same shape for plyo / prep / bp / hitting / movement-enhancement
If any column name is different in your data, the SQL will throw — fix the column
name in this file and rerun.

Run:
    python -m src.main summarize-program <program_id>            # one program
    python -m src.main summarize-all                              # everything
"""
from __future__ import annotations

import json
import sys
import time
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from src.db import backend_conn, exec_sql, query, returning_id


def _json_default(o: Any):
    if isinstance(o, Decimal):     return float(o)
    if isinstance(o, (date, datetime)): return o.isoformat()
    raise TypeError(f"Not JSON serializable: {type(o)}")


def _category_breakdown(
    conn, program_id: int,
    *,
    link_table: str,
    bridge_table: str | None,
    bridge_to_link_fk: str | None,   # column on bridge pointing to link.id
    target_table: str,
    target_fk_col: str,              # column on bridge (or link, if no bridge) pointing to target.id
    group_col: str,
) -> list[dict]:
    """Walk link -> [bridge ->] target and group-count by target.<group_col>.

    Three-step (lifts):
        link_table='LiftToProgram'
        bridge_table='ExerciseToLift'  bridge_to_link_fk='liftToProgramId'
        target_table='Exercise'        target_fk_col='exerciseId'  group_col='type'

    Two-step (plyos, no bridge):
        link_table='PlyoToProgram'
        bridge_table=None
        target_table='Plyo'            target_fk_col='plyoId'      group_col='name'
    """
    if bridge_table is None:
        sql = (
            f'SELECT COALESCE(t."{group_col}"::text, \'unknown\') AS category, '
            f'  COUNT(*)::int AS n, '
            f'  COUNT(DISTINCT t."id")::int AS unique_count '
            f'FROM app_db_snapshot."{link_table}" l '
            f'LEFT JOIN app_db_snapshot."{target_table}" t '
            f'  ON t."id" = l."{target_fk_col}" '
            f'WHERE l."programId" = %s '
            f'GROUP BY category ORDER BY n DESC'
        )
        return query(conn, sql, [program_id])

    sql = (
        f'SELECT COALESCE(e."{group_col}"::text, \'unknown\') AS category, '
        f'  COUNT(*)::int AS n, '
        f'  COUNT(DISTINCT e."id")::int AS unique_count '
        f'FROM app_db_snapshot."{link_table}" l '
        f'LEFT JOIN app_db_snapshot."{bridge_table}" b '
        f'  ON b."{bridge_to_link_fk}" = l."id" '
        f'LEFT JOIN app_db_snapshot."{target_table}" e '
        f'  ON e."id" = b."{target_fk_col}" '
        f'WHERE l."programId" = %s '
        f'GROUP BY category ORDER BY n DESC'
    )
    return query(conn, sql, [program_id])


def _count(conn, table: str, program_id: int) -> int:
    rows = query(conn, f'SELECT COUNT(*)::int AS n FROM app_db_snapshot."{table}" '
                       f'WHERE "programId" = %s', [program_id])
    return rows[0]["n"] if rows else 0


def _weekly_templates_for_program(conn, program_id: int) -> list[dict]:
    """Find weekly templates linked to this program via WeeklyTemplateApplication.

    Conservative version: pull only the columns we're confident exist (id + name).
    If the join fails we rollback so the outer transaction stays usable, and
    return an empty list. Metadata enrichment (tags/goals/notes) is added later
    once we know the exact WeeklyTemplateMetadata column names.
    """
    try:
        return query(conn, """
            SELECT wt."id", wt."name"
            FROM app_db_snapshot."WeeklyTemplateApplication" wta
            JOIN app_db_snapshot."WeeklyTemplate" wt
              ON wt."id" = wta."weeklyTemplateId"
            WHERE wta."programId" = %s
        """, [program_id])
    except Exception:
        # CRITICAL: rollback resets the aborted transaction so subsequent
        # statements on this connection can still run.
        conn.rollback()
        return []


def summarize_one(conn, program_id: int) -> dict:
    """Build the summary dict for one program (returned, not yet persisted)."""
    p_rows = query(conn, """
        SELECT
            p."id",         p."name",          p."description",
            p."throwerType",                   p."skillLevel",
            p."programType",                   p."enablePitching", p."enableHitting",
            p."isArchived", p."createdAt",     p."endDate",
            p."athleteGoalsCoordination" AS goal_coordination,
            p."athleteGoalsDurability"   AS goal_durability,
            p."athleteGoalsMobility"     AS goal_mobility,
            p."athleteGoalsPower"        AS goal_power,
            p."athleteGoalsSize"         AS goal_size,
            p."athleteGoalsSpeed"        AS goal_speed,
            p."athleteGoalsStrength"     AS goal_strength,
            u."email" AS app_user_email,
            d.athlete_uuid
        FROM app_db_snapshot."Program" p
        LEFT JOIN app_db_snapshot."User" u
          ON u."id" = p."userId"
        LEFT JOIN analytics.d_athletes d
          ON lower(d.email) = lower(u."email")
        WHERE p."id" = %s
    """, [program_id])
    if not p_rows:
        raise ValueError(f"Program {program_id} not found in app_db_snapshot.")
    p = p_rows[0]

    duration_days = None
    if p["createdAt"] and p["endDate"]:
        duration_days = (p["endDate"] - p["createdAt"]).days

    n_program_days = _count(conn, "ProgramDay", program_id)

    # Lifts: LiftToProgram → ExerciseToLift → Exercise (group by Exercise.type)
    lift_break = _category_breakdown(
        conn, program_id,
        link_table="LiftToProgram",
        bridge_table="ExerciseToLift",
        bridge_to_link_fk="liftToProgramId",
        target_table="Exercise",
        target_fk_col="exerciseId",
        group_col="type",
    )
    n_lift = sum(r["n"] for r in lift_break)
    n_unique_lifts = sum(r.get("unique_count", 0) or 0 for r in lift_break)

    # Plyos: PlyoToProgram → Plyo (direct, no bridge — group by Plyo.name)
    plyo_break = _category_breakdown(
        conn, program_id,
        link_table="PlyoToProgram",
        bridge_table=None, bridge_to_link_fk=None,
        target_table="Plyo",
        target_fk_col="plyoId",
        group_col="name",
    )
    n_plyo = sum(r["n"] for r in plyo_break)

    # Prep: PrepToProgram → ExerciseToPrep → Exercise
    prep_break = _category_breakdown(
        conn, program_id,
        link_table="PrepToProgram",
        bridge_table="ExerciseToPrep",
        bridge_to_link_fk="prepToProgramId",
        target_table="Exercise",
        target_fk_col="exerciseId",
        group_col="type",
    )
    n_prep = sum(r["n"] for r in prep_break)

    # Bulletproofing — note the FK column has a CAPITAL B (mixed-case quirk in App DB)
    bp_break = _category_breakdown(
        conn, program_id,
        link_table="BulletProofingToProgram",
        bridge_table="ExerciseToBulletProofing",
        bridge_to_link_fk="BulletProofingToProgramId",
        target_table="Exercise",
        target_fk_col="exerciseId",
        group_col="type",
    )
    n_bp = sum(r["n"] for r in bp_break)

    # Hitting
    hit_break = _category_breakdown(
        conn, program_id,
        link_table="HittingToProgram",
        bridge_table="ExerciseToHitting",
        bridge_to_link_fk="hittingToProgramId",
        target_table="Exercise",
        target_fk_col="exerciseId",
        group_col="type",
    )
    n_hit = sum(r["n"] for r in hit_break)

    # Movement Enhancement
    me_break = _category_breakdown(
        conn, program_id,
        link_table="MovementEnhancementToProgram",
        bridge_table="ExerciseToMovementEnhancement",
        bridge_to_link_fk="movementEnhancementToProgramId",
        target_table="Exercise",
        target_fk_col="exerciseId",
        group_col="type",
    )
    n_me = sum(r["n"] for r in me_break)

    weekly = _weekly_templates_for_program(conn, program_id)

    return {
        "program_id":       p["id"],
        "athlete_uuid":     p["athlete_uuid"],
        "app_user_email":   p["app_user_email"],
        "program_name":     p["name"],
        "program_type":     p["programType"],
        "thrower_type":     p["throwerType"],
        "skill_level":      p["skillLevel"],
        "enable_pitching":  p["enablePitching"],
        "enable_hitting":   p["enableHitting"],
        "is_archived":      p["isArchived"],
        "created_at_app":   p["createdAt"],
        "end_date_app":     p["endDate"],
        "duration_days":    duration_days,
        "goal_coordination": p["goal_coordination"],
        "goal_durability":   p["goal_durability"],
        "goal_mobility":     p["goal_mobility"],
        "goal_power":        p["goal_power"],
        "goal_size":         p["goal_size"],
        "goal_speed":        p["goal_speed"],
        "goal_strength":     p["goal_strength"],
        "n_program_days":       n_program_days,
        "n_lift_prescriptions": n_lift,
        "n_unique_lifts":       n_unique_lifts,
        "n_plyo_prescriptions": n_plyo,
        "n_prep_prescriptions": n_prep,
        "n_bp_prescriptions":   n_bp,
        "n_hit_prescriptions":  n_hit,
        "n_me_prescriptions":   n_me,
        "lift_breakdown": lift_break,
        "plyo_breakdown": plyo_break,
        "prep_breakdown": prep_break,
        "bp_breakdown":   bp_break,
        "hit_breakdown":  hit_break,
        "me_breakdown":   me_break,
        "weekly_templates": weekly,
    }


def save_summary(conn, s: dict) -> int:
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO ai_layer.program_summaries (
                program_id, athlete_uuid, app_user_email,
                program_name, program_type, thrower_type, skill_level,
                enable_pitching, enable_hitting, is_archived,
                created_at_app, end_date_app, duration_days,
                goal_coordination, goal_durability, goal_mobility,
                goal_power, goal_size, goal_speed, goal_strength,
                n_program_days, n_lift_prescriptions, n_unique_lifts,
                n_plyo_prescriptions, n_prep_prescriptions, n_bp_prescriptions,
                n_hit_prescriptions, n_me_prescriptions,
                lift_breakdown, plyo_breakdown, prep_breakdown,
                bp_breakdown, hit_breakdown, me_breakdown,
                weekly_templates
            ) VALUES (
                %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s,
                %s::jsonb, %s::jsonb, %s::jsonb,
                %s::jsonb, %s::jsonb, %s::jsonb,
                %s::jsonb
            )
            ON CONFLICT (program_id) DO UPDATE SET
                athlete_uuid    = EXCLUDED.athlete_uuid,
                app_user_email  = EXCLUDED.app_user_email,
                program_name    = EXCLUDED.program_name,
                program_type    = EXCLUDED.program_type,
                thrower_type    = EXCLUDED.thrower_type,
                skill_level     = EXCLUDED.skill_level,
                enable_pitching = EXCLUDED.enable_pitching,
                enable_hitting  = EXCLUDED.enable_hitting,
                is_archived     = EXCLUDED.is_archived,
                created_at_app  = EXCLUDED.created_at_app,
                end_date_app    = EXCLUDED.end_date_app,
                duration_days   = EXCLUDED.duration_days,
                goal_coordination = EXCLUDED.goal_coordination,
                goal_durability   = EXCLUDED.goal_durability,
                goal_mobility     = EXCLUDED.goal_mobility,
                goal_power        = EXCLUDED.goal_power,
                goal_size         = EXCLUDED.goal_size,
                goal_speed        = EXCLUDED.goal_speed,
                goal_strength     = EXCLUDED.goal_strength,
                n_program_days       = EXCLUDED.n_program_days,
                n_lift_prescriptions = EXCLUDED.n_lift_prescriptions,
                n_unique_lifts       = EXCLUDED.n_unique_lifts,
                n_plyo_prescriptions = EXCLUDED.n_plyo_prescriptions,
                n_prep_prescriptions = EXCLUDED.n_prep_prescriptions,
                n_bp_prescriptions   = EXCLUDED.n_bp_prescriptions,
                n_hit_prescriptions  = EXCLUDED.n_hit_prescriptions,
                n_me_prescriptions   = EXCLUDED.n_me_prescriptions,
                lift_breakdown   = EXCLUDED.lift_breakdown,
                plyo_breakdown   = EXCLUDED.plyo_breakdown,
                prep_breakdown   = EXCLUDED.prep_breakdown,
                bp_breakdown     = EXCLUDED.bp_breakdown,
                hit_breakdown    = EXCLUDED.hit_breakdown,
                me_breakdown     = EXCLUDED.me_breakdown,
                weekly_templates = EXCLUDED.weekly_templates,
                computed_at      = now()
            RETURNING id
        """, [
            s["program_id"], s["athlete_uuid"], s["app_user_email"],
            s["program_name"], s["program_type"], s["thrower_type"], s["skill_level"],
            s["enable_pitching"], s["enable_hitting"], s["is_archived"],
            s["created_at_app"], s["end_date_app"], s["duration_days"],
            s["goal_coordination"], s["goal_durability"], s["goal_mobility"],
            s["goal_power"], s["goal_size"], s["goal_speed"], s["goal_strength"],
            s["n_program_days"], s["n_lift_prescriptions"], s["n_unique_lifts"],
            s["n_plyo_prescriptions"], s["n_prep_prescriptions"], s["n_bp_prescriptions"],
            s["n_hit_prescriptions"], s["n_me_prescriptions"],
            json.dumps(s["lift_breakdown"], default=_json_default),
            json.dumps(s["plyo_breakdown"], default=_json_default),
            json.dumps(s["prep_breakdown"], default=_json_default),
            json.dumps(s["bp_breakdown"],   default=_json_default),
            json.dumps(s["hit_breakdown"],  default=_json_default),
            json.dumps(s["me_breakdown"],   default=_json_default),
            json.dumps(s["weekly_templates"], default=_json_default),
        ])
        return cur.fetchone()[0]


# ─────────────────────── Per-exercise prescriptions ────────────────────────
# These populate ai_layer.program_exercise_prescriptions with one row per
# individual exercise prescribed in a program, including sets/reps/weight
# arrays. This is the data the recommender will use to actually compose
# programs — it captures what coaches did at the exercise dose level.

# Categories that go through the standard Exercise table:
#   (category_key, link_table, bridge_table, bridge_to_link_fk, has_weight,
#    notes_on_bridge)
# notes_on_bridge=False means the bridge table has no notes column (e.g.
# ExerciseToLift). We fall back to the parent link table's notes in that case.
_PRESCRIPTION_CATEGORIES = [
    ("lift", "LiftToProgram",                "ExerciseToLift",
        "liftToProgramId",                  True,  False),
    ("prep", "PrepToProgram",                "ExerciseToPrep",
        "prepToProgramId",                  True,  True),
    ("bp",   "BulletProofingToProgram",      "ExerciseToBulletProofing",
        "BulletProofingToProgramId",        True,  True),
    ("hit",  "HittingToProgram",             "ExerciseToHitting",
        "hittingToProgramId",               True,  True),
    ("me",   "MovementEnhancementToProgram", "ExerciseToMovementEnhancement",
        "movementEnhancementToProgramId",   True,  True),
]


def _extract_exercise_prescriptions(conn, program_id: int, athlete_uuid: str | None
                                    ) -> list[dict]:
    """Pull one row per exercise prescription across all categories of a program.

    Returns dicts ready to insert into ai_layer.program_exercise_prescriptions.
    """
    prescriptions: list[dict] = []

    for (category, link_table, bridge_table, fk_col,
         has_weight, notes_on_bridge) in _PRESCRIPTION_CATEGORIES:
        weight_select = 'b."weight"' if has_weight else 'NULL'
        weight_unit_select = (
            'b."weightUnit"' if (has_weight and category == "lift") else 'NULL'
        )
        notes_select = 'b."notes"' if notes_on_bridge else 'l."notes"'
        rows = query(conn, f"""
            SELECT
                b."exerciseId"::int   AS exercise_id,
                e."name"              AS exercise_name,
                e."type"::text        AS exercise_type,
                b."reps"              AS reps_array,
                {weight_select}       AS weight_array,
                b."repsUnitCount"::int AS reps_unit_count,
                {weight_unit_select}  AS weight_unit,
                b."order"::int        AS order_in_program,
                {notes_select}        AS notes
            FROM app_db_snapshot."{link_table}" l
            JOIN app_db_snapshot."{bridge_table}" b
              ON b."{fk_col}" = l."id"
            LEFT JOIN app_db_snapshot."Exercise" e ON e."id" = b."exerciseId"
            WHERE l."programId" = %s
            ORDER BY l."id", b."order"
        """, [program_id])
        for r in rows:
            r["category"] = category
        prescriptions.extend(rows)

    # Plyos: PlyoToProgram → Plyo (parent umbrella) + ThrowingExerciseToPlyo
    # (per-drill sets/reps/ball-weight) + Exercise (the actual drill name).
    # The "throwingExerciseId" on the bridge is a FK to Exercise.id — the drill
    # names ARE in the standard Exercise catalog, not in a separate table.
    plyos = query(conn, """
        SELECT
            p2p."plyoId"::int     AS plyo_id,
            pl."name"             AS plyo_name,
            pl."intensity"::int   AS plyo_intensity,
            tep."throwingExerciseId"::int AS exercise_id,
            e."name"              AS exercise_name,
            e."type"::text        AS exercise_type,
            tep."reps"            AS reps_array,
            tep."plyoBallWeight"  AS plyo_ball_weight,
            tep."order"::int      AS order_in_program,
            tep."notes"           AS notes
        FROM app_db_snapshot."PlyoToProgram" p2p
        LEFT JOIN app_db_snapshot."Plyo" pl ON pl."id" = p2p."plyoId"
        LEFT JOIN app_db_snapshot."ThrowingExerciseToPlyo" tep
          ON tep."plyoToProgramId" = p2p."id"
        LEFT JOIN app_db_snapshot."Exercise" e
          ON e."id" = tep."throwingExerciseId"
        WHERE p2p."programId" = %s
        ORDER BY p2p."id", tep."order"
    """, [program_id])
    for r in plyos:
        r["category"] = "plyo"
        # If the join failed (orphan drill ID), fall back to the umbrella name
        # so we don't lose the row entirely.
        if not r.get("exercise_name"):
            r["exercise_name"] = r.get("plyo_name") or "Unknown plyo drill"
            r["exercise_type"] = "plyo"
    prescriptions.extend(plyos)

    # Derive numeric summaries from the array fields. The raw arrays may come
    # through as text[] (because Prisma's Int[]/Float[] sources get stored
    # permissively in the snapshot), and entries can include non-numeric
    # tokens like "AMRAP" or "30s" — those get dropped here.
    def _to_int(v):
        try:
            return int(float(v)) if v is not None and str(v).strip() != "" else None
        except (TypeError, ValueError):
            return None

    def _to_float(v):
        try:
            return float(v) if v is not None and str(v).strip() != "" else None
        except (TypeError, ValueError):
            return None

    for p in prescriptions:
        p["athlete_uuid"] = athlete_uuid

        raw_reps = p.get("reps_array") or []
        raw_weights = p.get("weight_array") or []

        reps    = [r for r in (_to_int(x)   for x in raw_reps)    if r is not None]
        weights = [w for w in (_to_float(x) for x in raw_weights) if w is not None]

        # IMPORTANT: write None instead of [] for empty arrays so psycopg2
        # serializes them as SQL NULL (not '{}', which would be text-array typed).
        p["reps_array"]   = reps    or None
        p["weight_array"] = weights or None

        p["n_sets"]     = len(reps)             if reps else None
        p["total_reps"] = sum(reps)             if reps else None
        p["avg_reps"]   = sum(reps) / len(reps) if reps else None
        p["max_reps"]   = max(reps)             if reps else None
        p["min_reps"]   = min(reps)             if reps else None
        p["avg_weight"] = sum(weights) / len(weights) if weights else None
        p["max_weight"] = max(weights)                if weights else None

    return prescriptions


def _save_prescriptions(conn, program_id: int, prescriptions: list[dict]) -> int:
    """Wipe and replace prescription rows for a program. Returns count written."""
    exec_sql(conn,
        "DELETE FROM ai_layer.program_exercise_prescriptions WHERE program_id = %s",
        [program_id])
    n = 0
    for p in prescriptions:
        exec_sql(conn, """
            INSERT INTO ai_layer.program_exercise_prescriptions (
                program_id, athlete_uuid, category,
                exercise_id, exercise_name, exercise_type,
                plyo_id, plyo_name, plyo_intensity, plyo_ball_weight,
                n_sets, reps_array, weight_array,
                total_reps, avg_reps, max_reps, min_reps,
                avg_weight, max_weight,
                reps_unit_count, weight_unit,
                order_in_program, notes
            ) VALUES (
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s,
                %s, %s,
                %s, %s
            )
        """, [
            program_id, p.get("athlete_uuid"), p["category"],
            p.get("exercise_id"), p.get("exercise_name"), p.get("exercise_type"),
            p.get("plyo_id"), p.get("plyo_name"),
            p.get("plyo_intensity"), p.get("plyo_ball_weight"),
            p.get("n_sets"), p.get("reps_array"), p.get("weight_array"),
            p.get("total_reps"), p.get("avg_reps"),
            p.get("max_reps"), p.get("min_reps"),
            p.get("avg_weight"), p.get("max_weight"),
            p.get("reps_unit_count"), p.get("weight_unit"),
            p.get("order_in_program"), p.get("notes"),
        ])
        n += 1
    return n


def summarize_one_and_save(program_id: int) -> tuple[int, dict]:
    with backend_conn() as conn:
        s = summarize_one(conn, program_id)
        sid = save_summary(conn, s)
        prescriptions = _extract_exercise_prescriptions(conn, program_id, s["athlete_uuid"])
        n_pres = _save_prescriptions(conn, program_id, prescriptions)
        s["_prescriptions_saved"] = n_pres
    return sid, s


def summarize_all() -> None:
    """Walk every program in the snapshot and persist a summary row for each."""
    with backend_conn() as conn:
        program_ids = [r["id"] for r in query(conn,
            'SELECT "id" FROM app_db_snapshot."Program" ORDER BY "id"')]

    total = len(program_ids)
    print(f"[summary] {total} program(s) to summarize")

    with backend_conn() as conn:
        log_id = returning_id(conn, """
            INSERT INTO ai_layer.sync_log (sync_type, table_name, status)
            VALUES ('embedding_refresh', 'program_summaries', 'running') RETURNING id
        """, [])

    started = time.time()
    successes = 0
    failures: list[tuple[int, str]] = []

    for i, pid in enumerate(program_ids, 1):
        try:
            with backend_conn() as conn:
                s = summarize_one(conn, pid)
                save_summary(conn, s)
                prescriptions = _extract_exercise_prescriptions(
                    conn, pid, s["athlete_uuid"])
                _save_prescriptions(conn, pid, prescriptions)
            successes += 1
        except Exception as e:
            failures.append((pid, str(e)[:200]))

        if i % 25 == 0 or i == total:
            elapsed = time.time() - started
            rate = i / elapsed if elapsed > 0 else 0
            eta = int((total - i) / rate) if rate > 0 else 0
            print(f"[summary] {i}/{total}  ok={successes}  fail={len(failures)}  "
                  f"rate={rate:.1f}/s  eta={eta}s", flush=True)

    duration_ms = int((time.time() - started) * 1000)
    status = "success" if not failures else "partial"
    err = "; ".join(f"#{pid}: {e}" for pid, e in failures[:5])[:480] if failures else None

    with backend_conn() as conn:
        exec_sql(conn, """
            UPDATE ai_layer.sync_log
            SET rows_synced = %s, duration_ms = %s, status = %s,
                completed_at = now(), error_message = %s
            WHERE id = %s
        """, [successes, duration_ms, status, err, log_id])

    print(f"[summary] done. {successes} written. failures: {len(failures)}")
    if failures:
        print("[summary] sample failures:")
        for pid, e in failures[:5]:
            print(f"  program {pid}: {e}")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "all":
        summarize_all()
    elif len(sys.argv) >= 2:
        sid, s = summarize_one_and_save(int(sys.argv[1]))
        print(f"Saved summary id: {sid}")
        print(json.dumps(s, default=_json_default, indent=2)[:4000])
    else:
        print("Usage: python -m src.program_summarizer <program_id>")
        print("       python -m src.program_summarizer all")
        sys.exit(2)
