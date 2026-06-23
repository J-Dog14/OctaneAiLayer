"""
Athletic screen pipeline: for one athlete + one session_date,

1. Pull CMJ / DJ / PPU / SLV rows from the backend warehouse.
2. Pull recent program context from app_db_snapshot (joined by email).
3. Send the lot to Gemini using the `athletic-screen-analysis` skill.
4. Persist the report to ai_layer.generated_reports.

Run:
    python -m src.pipelines.athletic_screen <athlete_uuid> <YYYY-MM-DD>
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from src.db import backend_conn, exec_sql, query, returning_id
from src.gemini_client import generate
from src.prompt_loader import load_and_register

SKILL_NAME = "athletic-screen-analysis"
SOURCE_TYPE = "athletic_screen"


def _json_default(o: Any):
    if isinstance(o, Decimal):
        return float(o)
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    raise TypeError(f"Not JSON serializable: {type(o)}")


def _table_exists(conn, schema: str, table: str) -> bool:
    rows = query(conn, """
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = %s AND table_name = %s
    """, [schema, table])
    return bool(rows)


def _pull(athlete_uuid: str, session_date: str) -> dict:
    """Gather everything needed for the prompt as a plain dict."""
    with backend_conn() as conn:
        athletes = query(conn, """
            SELECT athlete_uuid, name, age_group, age_at_collection,
                   gender, height, weight, email, app_db_uuid
            FROM analytics.d_athletes
            WHERE athlete_uuid = %s
        """, [athlete_uuid])
        if not athletes:
            raise ValueError(f"Athlete {athlete_uuid} not found in analytics.d_athletes.")
        athlete = athletes[0]

        cmj = query(conn, """
            SELECT * FROM public.f_athletic_screen_cmj
            WHERE athlete_uuid = %s AND session_date = %s
        """, [athlete_uuid, session_date])
        dj = query(conn, """
            SELECT * FROM public.f_athletic_screen_dj
            WHERE athlete_uuid = %s AND session_date = %s
        """, [athlete_uuid, session_date])
        ppu = query(conn, """
            SELECT * FROM public.f_athletic_screen_ppu
            WHERE athlete_uuid = %s AND session_date = %s
        """, [athlete_uuid, session_date])
        slv = query(conn, """
            SELECT * FROM public.f_athletic_screen_slv
            WHERE athlete_uuid = %s AND session_date = %s
        """, [athlete_uuid, session_date])

        # Program context. Best-effort: skip cleanly if snapshot tables aren't there yet.
        # Join: Program.userId (uuid) = User.id (uuid). Athlete linked via email.
        recent_programs: list[dict] = []
        if athlete.get("email") and _table_exists(conn, "app_db_snapshot", "User") \
                                 and _table_exists(conn, "app_db_snapshot", "Program"):
            try:
                recent_programs = query(conn, """
                    SELECT p."id", p."name", p."description", p."programType",
                           p."throwerType", p."skillLevel", p."enablePitching",
                           p."enableHitting", p."createdAt", p."endDate",
                           p."athleteGoalsCoordination", p."athleteGoalsDurability",
                           p."athleteGoalsMobility",     p."athleteGoalsPower",
                           p."athleteGoalsSize",         p."athleteGoalsSpeed",
                           p."athleteGoalsStrength"
                    FROM app_db_snapshot."Program" p
                    JOIN app_db_snapshot."User" u ON u."id" = p."userId"
                    WHERE lower(u."email") = lower(%s)
                      AND p."isArchived" = false
                    ORDER BY p."createdAt" DESC NULLS LAST
                    LIMIT 3
                """, [athlete["email"]])
            except Exception:
                recent_programs = []

    return {
        "athlete": athlete,
        "session_date": str(session_date),
        "cmj": cmj,
        "dj": dj,
        "ppu": ppu,
        "slv": slv,
        "recent_programs": recent_programs,
    }


def run(athlete_uuid: str, session_date: str) -> int:
    """Generate one athletic-screen report. Returns the new generated_reports.id."""
    payload = _pull(athlete_uuid, session_date)
    payload_json = json.dumps(payload, default=_json_default, indent=2)

    system_prompt, prompt_version_id = load_and_register(SKILL_NAME)

    user_msg = (
        "Below is the athlete's athletic-screen data for one session, plus recent "
        "programming context pulled from the App DB snapshot.\n\n"
        "Apply the framework defined above and produce the standard breakdown for "
        "this athlete.\n\n"
        f"```json\n{payload_json}\n```"
    )

    # Insert a stub row first so the LLM call log can FK back to it,
    # even if the call later errors out.
    with backend_conn() as conn:
        report_id = returning_id(conn, """
            INSERT INTO ai_layer.generated_reports
              (athlete_uuid, session_date, source_type, skill_name,
               prompt_version_id, model_name, input_payload, output_text)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, [
            athlete_uuid, payload["session_date"], SOURCE_TYPE, SKILL_NAME,
            prompt_version_id, "pending", payload_json, "",
        ])

    result = generate(system_prompt, user_msg, generated_report_id=report_id)

    with backend_conn() as conn:
        exec_sql(conn, """
            UPDATE ai_layer.generated_reports
            SET output_text = %s, model_name = %s
            WHERE id = %s
        """, [result["text"], result["model"], report_id])

    return report_id


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m src.pipelines.athletic_screen <athlete_uuid> <YYYY-MM-DD>")
        sys.exit(2)
    rid = run(sys.argv[1], sys.argv[2])
    print(f"Generated report id: {rid}")
