"""
Pitching biomechanics pipeline. STUB — fill in once you've validated the
athletic_screen pipeline end-to-end.

The shape mirrors athletic_screen.py: pull data, format, call Gemini with the
biomech-pitching-breakdown skill, persist.

Run:
    python -m src.pipelines.biomech_pitching <athlete_uuid> <YYYY-MM-DD>
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

SKILL_NAME = "biomech-pitching-breakdown"
SOURCE_TYPE = "pitching_biomech"


def _json_default(o: Any):
    if isinstance(o, Decimal):
        return float(o)
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    raise TypeError(f"Not JSON serializable: {type(o)}")


def _pull(athlete_uuid: str, session_date: str) -> dict:
    """Gather the pitching session for one athlete + session_date.

    TODO: tighten this once you have an example session. For now we grab the
    headline kinematics, trial rows, and force metrics if present.
    """
    with backend_conn() as conn:
        athletes = query(conn, """
            SELECT athlete_uuid, name, age_group, age_at_collection,
                   gender, height, weight, email, app_db_uuid
            FROM analytics.d_athletes
            WHERE athlete_uuid = %s
        """, [athlete_uuid])
        if not athletes:
            raise ValueError(f"Athlete {athlete_uuid} not found.")
        athlete = athletes[0]

        # Pull kinematics rows for this session. The f_kinematics_pitching table is
        # long-form (metric_name, frame, value), which can get big — limit to
        # one row per metric_name aggregated at key frames first when you refine
        # this. For now, leave it long-form and let the prompt summarize.
        kinematics = query(conn, """
            SELECT metric_name, frame, value, velocity_mph, score
            FROM public.f_kinematics_pitching
            WHERE athlete_uuid = %s AND session_date = %s
            ORDER BY metric_name, frame
        """, [athlete_uuid, session_date])

        trials = query(conn, """
            SELECT * FROM public.f_pitching_trials
            WHERE athlete_uuid = %s AND session_date = %s
        """, [athlete_uuid, session_date])

        force_metrics = query(conn, """
            SELECT * FROM public.f_pitching_force_metrics
            WHERE athlete_uuid = %s AND session_date = %s
        """, [athlete_uuid, session_date])

    return {
        "athlete": athlete,
        "session_date": str(session_date),
        "kinematics_count": len(kinematics),
        "kinematics": kinematics,
        "trials": trials,
        "force_metrics": force_metrics,
    }


def run(athlete_uuid: str, session_date: str) -> int:
    payload = _pull(athlete_uuid, session_date)
    payload_json = json.dumps(payload, default=_json_default, indent=2)

    system_prompt, prompt_version_id = load_and_register(SKILL_NAME)

    user_msg = (
        "Below is the pitching biomechanics data for one session. Apply the "
        "8-cylinder framework and produce the standard breakdown.\n\n"
        f"```json\n{payload_json}\n```"
    )

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
        print("Usage: python -m src.pipelines.biomech_pitching <athlete_uuid> <YYYY-MM-DD>")
        sys.exit(2)
    rid = run(sys.argv[1], sys.argv[2])
    print(f"Generated report id: {rid}")
