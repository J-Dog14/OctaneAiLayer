"""
Send a compiled athlete profile to the Octane app.

POSTs one row from ai_layer.athlete_profiles (the full ~217-metric raw-value +
z-score document built by src.profiler) to Octane's `POST /api/biomech/profile`
endpoint, where it lands in the reports DB (athlete_assessment_profiles) and
feeds the AI template-generation feature (app/admin/generate-templates).

Identity resolution mirrors payload_builder.py: warehouse athlete_uuid →
analytics.d_athletes (name, email, age_group) → app_db_snapshot."User" by
email (octane user uuid). Octane re-verifies uuid+email on its side before
storing anything.

Env (add to .env):
    OCTANE_API_URL           Base URL, no trailing slash.
                             e.g. https://<octane-host> or http://localhost:3000
    OCTANE_REPORTS_API_KEY   Bearer token; must match Octane's REPORTS_API_KEY.

Run:
    python -m src.main send-profile <athlete_uuid> [--as-of YYYY-MM-DD] [--dry-run]
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from src.db import backend_conn, query


def _json_default(o: Any):
    if isinstance(o, Decimal):
        return float(o)
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    raise TypeError(f"Not JSON serializable: {type(o)}")


def _require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing required env var: {name}. Check your .env file.")
    return val


def _load_profile(conn, athlete_uuid: str, as_of: str | None) -> dict:
    if as_of:
        rows = query(conn, """
            SELECT * FROM ai_layer.athlete_profiles
            WHERE athlete_uuid = %s AND as_of_date = %s
        """, [athlete_uuid, as_of])
    else:
        rows = query(conn, """
            SELECT * FROM ai_layer.athlete_profiles
            WHERE athlete_uuid = %s
            ORDER BY as_of_date DESC
            LIMIT 1
        """, [athlete_uuid])
    if not rows:
        raise ValueError(
            f"No profile found for athlete {athlete_uuid}"
            + (f" at {as_of}" if as_of else "")
            + ". Run the profiler first."
        )
    return rows[0]


def _resolve_identity(conn, athlete_uuid: str) -> dict:
    """Warehouse athlete → (name, email, age_group, octane user uuid)."""
    ath = query(conn, """
        SELECT name, email, age_group
        FROM analytics.d_athletes
        WHERE athlete_uuid = %s
    """, [athlete_uuid])
    if not ath:
        raise ValueError(f"Athlete {athlete_uuid} not found in analytics.d_athletes")
    row = ath[0]
    if not row.get("email"):
        raise ValueError(
            f"Athlete {athlete_uuid} ({row.get('name')}) has no email in "
            "d_athletes — cannot link to an Octane user."
        )

    users = query(conn, """
        SELECT "uuid"::text AS octane_user_uuid, "email"
        FROM app_db_snapshot."User"
        WHERE lower("email") = lower(%s)
        LIMIT 1
    """, [row["email"]])
    if not users:
        raise ValueError(
            f"No app_db_snapshot.User with email {row['email']!r}. "
            "Refresh the snapshot (sync-app-db) or fix the email link."
        )

    return {
        "name": row.get("name") or "Unknown Athlete",
        "email": row["email"],
        "age_group": row.get("age_group"),
        "octane_user_uuid": users[0]["octane_user_uuid"],
    }


def _derive_role(conn, athlete_uuid: str) -> str | None:
    """Role from warehouse fact tables (matches Octane's expectation):
    f_pitching_trials rows -> pitcher, f_hitting_trials rows -> hitter,
    both -> both, neither -> None."""
    has_p = bool(query(conn, "SELECT 1 FROM public.f_pitching_trials WHERE athlete_uuid = %s LIMIT 1", [athlete_uuid]))
    has_h = bool(query(conn, "SELECT 1 FROM public.f_hitting_trials WHERE athlete_uuid = %s LIMIT 1", [athlete_uuid]))
    if has_p and has_h:
        return "both"
    if has_p:
        return "pitcher"
    if has_h:
        return "hitter"
    return None


def _clean_metric_map(raw: Any) -> dict[str, float | None]:
    """JSONB map → {metric_key: float|None}, dropping anything non-conforming."""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float | None] = {}
    for key, val in raw.items():
        if val is None:
            out[key] = None
        elif isinstance(val, (int, float, Decimal)):
            out[key] = float(val)
        # Non-numeric values are silently dropped — the Octane schema only
        # accepts number|null and a stray string would reject the whole payload.
    return out


def build_profile_payload(athlete_uuid: str, as_of: str | None = None) -> dict:
    """Assemble the JSON body for POST /api/biomech/profile."""
    with backend_conn() as conn:
        profile = _load_profile(conn, athlete_uuid, as_of)
        identity = _resolve_identity(conn, athlete_uuid)
        role = _derive_role(conn, athlete_uuid)

    if role is None:
        raise ValueError(
            f"Athlete {athlete_uuid} has no rows in f_pitching_trials or "
            "f_hitting_trials - cannot determine role (pitcher/hitter/both)."
        )

    as_of_date = profile["as_of_date"]
    if isinstance(as_of_date, (datetime, date)):
        as_of_date = as_of_date.isoformat()[:10]

    source_dates = profile.get("source_dates")
    if not isinstance(source_dates, dict):
        source_dates = None

    payload = {
        "octaneUserUuid": identity["octane_user_uuid"],
        "athleteEmail": identity["email"],
        "athleteName": identity["name"],
        "asOfDate": as_of_date,
        "role": role,
        "ageGroup": profile.get("age_group") or identity.get("age_group"),
        "rawValues": _clean_metric_map(profile.get("raw_values")),
        "zScores": _clean_metric_map(profile.get("z_scores")),
        "sourceDates": source_dates,
        "source": f"ai_layer.athlete_profiles:{profile['id']}",
    }
    if not payload["zScores"]:
        raise ValueError(
            f"Profile {profile['id']} has an empty z_scores map — nothing to send."
        )
    return payload


def send_profile(athlete_uuid: str, as_of: str | None = None,
                 dry_run: bool = False) -> dict:
    """Build and POST the profile. Returns the parsed API response."""
    payload = build_profile_payload(athlete_uuid, as_of)

    if dry_run:
        return {
            "dry_run": True,
            "athlete": payload["athleteName"],
            "asOfDate": payload["asOfDate"],
            "role": payload["role"],
            "metrics": len(payload["zScores"]),
        }

    base_url = _require_env("OCTANE_API_URL").rstrip("/")
    api_key = _require_env("OCTANE_REPORTS_API_KEY")

    body = json.dumps(payload, default=_json_default).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/api/biomech/profile",
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Octane rejected the profile ({e.code}): {detail}"
        ) from e

# ─────────────────────────── Corpus sync ────────────────────────────────────
# Pushes the reference population to Octane: every warehouse athlete's latest
# deficit profile + their historical coach prescriptions. Octane's AI program
# generation uses this corpus for similarity search and candidate pools.
# One POST per athlete → /api/biomech/corpus (idempotent per athlete).


def _corpus_athletes(conn, athlete_uuid: str | None = None,
                     limit: int | None = None) -> list[dict]:
    """Athletes that have at least one profile, with latest profile attached."""
    clauses = ["1=1"]
    params: list[Any] = []
    if athlete_uuid:
        clauses.append("p.athlete_uuid = %s")
        params.append(athlete_uuid)
    sql = f"""
        SELECT DISTINCT ON (p.athlete_uuid)
               p.athlete_uuid, p.as_of_date, p.z_scores, p.age_group,
               d.name, d.has_pitching_data, d.has_hitting_data
        FROM ai_layer.athlete_profiles p
        JOIN analytics.d_athletes d USING (athlete_uuid)
        WHERE {' AND '.join(clauses)}
        ORDER BY p.athlete_uuid, p.as_of_date DESC
    """
    rows = query(conn, sql, params)
    return rows[:limit] if limit else rows


def _role_flags_with_fallback(conn, athlete_uuid: str,
                              has_pitching: bool, has_hitting: bool) -> tuple[bool, bool]:
    """d_athletes role flags lag ingestion sometimes — fall back to fact tables
    (same logic as recommender.load_athlete_profile)."""
    if not has_pitching:
        if query(conn, "SELECT 1 FROM public.f_pitching_trials WHERE athlete_uuid = %s LIMIT 1",
                 [athlete_uuid]):
            has_pitching = True
    if not has_hitting:
        if query(conn, "SELECT 1 FROM public.f_hitting_trials WHERE athlete_uuid = %s LIMIT 1",
                 [athlete_uuid]):
            has_hitting = True
    return has_pitching, has_hitting


def _athlete_prescriptions(conn, athlete_uuid: str) -> list[dict]:
    rows = query(conn, """
        SELECT category, exercise_name, exercise_id::text AS exercise_id,
               exercise_type, n_sets, avg_reps, max_reps, avg_weight,
               plyo_intensity, plyo_ball_weight, plyo_name
        FROM ai_layer.program_exercise_prescriptions
        WHERE athlete_uuid = %s
          AND exercise_name IS NOT NULL
          AND category IN ('lift', 'plyo', 'prep', 'bp', 'hit', 'me')
    """, [athlete_uuid])
    out = []
    for r in rows:
        out.append({
            "category": r["category"],
            "exerciseName": r["exercise_name"],
            "exerciseId": r.get("exercise_id"),
            "exerciseType": r.get("exercise_type"),
            "nSets": float(r["n_sets"]) if r.get("n_sets") is not None else None,
            "avgReps": float(r["avg_reps"]) if r.get("avg_reps") is not None else None,
            "maxReps": float(r["max_reps"]) if r.get("max_reps") is not None else None,
            "avgWeight": float(r["avg_weight"]) if r.get("avg_weight") is not None else None,
            "plyoIntensity": int(r["plyo_intensity"]) if r.get("plyo_intensity") is not None else None,
            "plyoBallWeight": r.get("plyo_ball_weight"),
            "plyoName": r.get("plyo_name"),
        })
    return out


def _post_json(path: str, payload: dict) -> dict:
    base_url = _require_env("OCTANE_API_URL").rstrip("/")
    api_key = _require_env("OCTANE_REPORTS_API_KEY")
    body = json.dumps(payload, default=_json_default).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}{path}",
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Octane rejected {path} ({e.code}): {detail}") from e


def send_corpus(athlete_uuid: str | None = None, limit: int | None = None,
                dry_run: bool = False) -> dict:
    """Send corpus (all athletes with profiles, or one) to Octane."""
    with backend_conn() as conn:
        athletes = _corpus_athletes(conn, athlete_uuid, limit)
        prepared = []
        for a in athletes:
            z = _clean_metric_map(a.get("z_scores"))
            if not z:
                continue
            has_p, has_h = _role_flags_with_fallback(
                conn, a["athlete_uuid"],
                bool(a.get("has_pitching_data")), bool(a.get("has_hitting_data")),
            )
            as_of = a["as_of_date"]
            if isinstance(as_of, (datetime, date)):
                as_of = as_of.isoformat()[:10]
            prepared.append({
                "athleteUuid": a["athlete_uuid"],
                "name": a.get("name") or "Unknown Athlete",
                "ageGroup": (a.get("age_group") or "").strip().upper() or None,
                "hasPitchingData": has_p,
                "hasHittingData": has_h,
                "asOfDate": as_of,
                "zScores": z,
                "prescriptions": _athlete_prescriptions(conn, a["athlete_uuid"]),
            })

    if dry_run:
        n_rx = sum(len(p["prescriptions"]) for p in prepared)
        return {"dry_run": True, "athletes": len(prepared), "prescriptions": n_rx}

    sent = 0
    failed: list[str] = []
    for p in prepared:
        try:
            _post_json("/api/biomech/corpus", p)
            sent += 1
            print(f"[send-corpus] {sent}/{len(prepared)} {p['name']} "
                  f"({len(p['prescriptions'])} rx)")
        except Exception as e:
            failed.append(f"{p['name']}: {e}")
            print(f"[send-corpus] FAILED {p['name']}: {e}")
    return {"athletes": len(prepared), "sent": sent, "failed": failed}

