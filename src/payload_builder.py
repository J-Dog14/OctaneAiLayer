"""
Compile a row from ai_layer.recommended_programs into a JSON payload that
mirrors the Octane App DB schema for weekly templates.

The payload is shaped so that a future API endpoint (which you'll build) can
deserialize it into actual App DB inserts:

  - one ActivityTemplate row per component (prep, bulletproofing, plyo, plus
    N lift templates — one per movement bucket)
  - the corresponding child rows (ExerciseToPrepTemplate, ...ToBulletProofingTemplate,
    PlyoToTemplate + ThrowingExerciseToPlyoTemplate, ExerciseToLift with
    activityTemplateId)
  - one WeeklyTemplate row referencing those ActivityTemplate ids
  - seven WeeklyTemplateDay rows mapping each day to its lift template + plyo
    level + throwing activity
  - (optional) a WeeklyTemplateApplication row linking the athlete

Exercise name → Exercise.id resolution happens here against app_db_snapshot.
Unresolved names are flagged in the payload's `unmatched_exercises` list for
manual review before insertion.

Run:
    python -m src.main compile-payload <recommended_program_id>
"""
from __future__ import annotations

import json
import re
import unicodedata
from datetime import date, datetime
from decimal import Decimal
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from src.db import backend_conn, query

# Fuzzy match threshold for the difflib fallback. 0.88 means the names must
# share ~88% similarity. Anything below 0.88 is left unmatched. Tighten if you
# get false positives, loosen if real matches keep being missed.
_FUZZY_THRESHOLD = 0.88

# Ball-weight enum mapping (display string -> DB enum). Inverts _format_ball_weight.
_BALL_WEIGHT_BACK = {
    "32 oz": "OZ_32", "16 oz": "OZ_16", "9 oz": "OZ_9",
    "7 oz": "OZ_7",   "5 oz":  "OZ_5",  "3 oz": "OZ_3",
    "1 lb": "LB_1",   "2 lb":  "LB_2",
}


def _format_ball_weight_back(display: str | None) -> str | None:
    """'5 oz' -> 'OZ_5' (DB enum). Returns None if can't parse."""
    if not display:
        return None
    s = str(display).strip().lower()
    if s in _BALL_WEIGHT_BACK:
        return _BALL_WEIGHT_BACK[s]
    # Loose parse
    m = re.match(r"^(\d+)\s*(oz|lb)$", s)
    if m:
        return f"{m.group(2).upper()}_{m.group(1)}"
    return None


def _json_default(o):
    if isinstance(o, Decimal):
        return float(o)
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    if isinstance(o, set):
        return list(o)
    raise TypeError(f"Not JSON serializable: {type(o)}")


class ExerciseLookup:
    """Stateful holder for Exercise.id resolution. Built once per compile."""

    def __init__(self, conn) -> None:
        rows = query(conn, """
            SELECT "id", "name"
            FROM app_db_snapshot."Exercise"
            WHERE NOT "isArchived" AND "name" IS NOT NULL
        """)
        self.exact_idx: dict[str, int] = {}
        self.loose_idx: dict[str, int] = {}
        self.candidates: list[tuple[str, int]] = []
        self.name_by_id: dict[int, str] = {}
        for r in rows:
            raw_name = r["name"] or ""
            ex_id = int(r["id"])
            self.name_by_id[ex_id] = raw_name
            ek = raw_name.strip().lower()
            if ek and ek not in self.exact_idx:
                self.exact_idx[ek] = ex_id
            lk = _normalize_loose(raw_name)
            if lk and lk not in self.loose_idx:
                self.loose_idx[lk] = ex_id
                self.candidates.append((lk, ex_id))

    def resolve(self, name: str) -> tuple[int | None, str | None, str | None]:
        return _resolve_exercise(
            name, self.exact_idx, self.loose_idx,
            self.candidates, self.name_by_id,
        )

    def nearest(self, name: str, top_n: int = 3) -> list[dict]:
        return _nearest_candidates(name, self.candidates, self.name_by_id, top_n)


def _normalize_loose(name: str) -> str:
    """Aggressive normalization for fuzzy-equality matching.

    - Lowercase + strip
    - Unicode-normalize (NFKD) and drop diacritics & smart-quote variants
    - Replace common abbreviations expansions are NOT done here (we don't want
      "DB" → "Dumbbell" because the App DB may use either). Instead we strip
      punctuation and collapse whitespace so superficial drift goes away.
    - Treat "w/" and "w " as the same. Same for "&" vs "and".
    - Collapse internal whitespace, drop trailing punctuation
    """
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().strip()
    # Normalize curly quotes / dashes
    s = s.replace("’", "'").replace("‘", "'")
    s = s.replace("“", '"').replace("”", '"')
    s = s.replace("–", "-").replace("—", "-")
    # Standardize "with" forms
    s = re.sub(r"\bw\s*/\s*", "w ", s)         # "w/" -> "w "
    s = re.sub(r"\bwith\b", "w", s)
    # Standardize "and"
    s = s.replace("&", "and")
    # Strip all punctuation except slashes (5/10/5 matters)
    s = re.sub(r"[^a-z0-9/ ]+", " ", s)
    # Collapse runs of whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _build_exercise_index(conn) -> tuple[dict[str, int], dict[str, int], list[tuple[str, int]]]:
    """Build two lookup dicts + a fuzzy candidate list against
    app_db_snapshot.Exercise.

    Returns:
        exact_idx: lowercased+stripped name -> Exercise.id
        loose_idx: aggressively-normalized name -> Exercise.id
        candidates: list[(loose_name, id)] for the difflib fallback.
    """
    rows = query(conn, """
        SELECT "id", "name"
        FROM app_db_snapshot."Exercise"
        WHERE NOT "isArchived" AND "name" IS NOT NULL
    """)
    exact_idx: dict[str, int] = {}
    loose_idx: dict[str, int] = {}
    candidates: list[tuple[str, int]] = []
    for r in rows:
        raw_name = r["name"] or ""
        ex_id = int(r["id"])
        ek = raw_name.strip().lower()
        if ek and ek not in exact_idx:
            exact_idx[ek] = ex_id
        lk = _normalize_loose(raw_name)
        if lk and lk not in loose_idx:
            loose_idx[lk] = ex_id
            candidates.append((lk, ex_id))
    return exact_idx, loose_idx, candidates


def _resolve_exercise(
    name: str,
    exact_idx: dict[str, int],
    loose_idx: dict[str, int],
    candidates: list[tuple[str, int]],
    name_by_id: dict[int, str],
) -> tuple[int | None, str | None, str | None]:
    """Resolve a name to an Exercise.id. Returns (id, match_method, matched_name).

    match_method ∈ {'exact', 'loose', 'fuzzy', None}
        - 'exact' — straight case-insensitive equality (safe, no review)
        - 'loose' — equality after aggressive normalization (safe, no review)
        - 'fuzzy' — difflib similarity ≥ threshold (review-required)
        - None    — no match; caller should flag in unmatched_exercises
    """
    if not name:
        return None, None, None
    # 1. Exact lowercased match
    k = name.strip().lower()
    if k in exact_idx:
        ex_id = exact_idx[k]
        return ex_id, "exact", name_by_id.get(ex_id)
    # 2. Loose normalized match
    lk = _normalize_loose(name)
    if lk and lk in loose_idx:
        ex_id = loose_idx[lk]
        return ex_id, "loose", name_by_id.get(ex_id)
    # 3. Fuzzy match via difflib — only if threshold met. Compare against the
    # already-normalized candidate list so superficial drift doesn't penalize.
    if lk and candidates:
        best_score = 0.0
        best_id = None
        best_name = None
        # Use a fast prefilter — only compare against candidates that share
        # the first token (rough word-overlap heuristic). Falls back to full
        # scan if prefilter empties out.
        first_token = lk.split(" ", 1)[0] if lk else ""
        prefiltered = [c for c in candidates if c[0].startswith(first_token)]
        scan_pool = prefiltered if prefiltered else candidates
        for cand_norm, cand_id in scan_pool:
            score = SequenceMatcher(None, lk, cand_norm).ratio()
            if score > best_score:
                best_score = score
                best_id = cand_id
                best_name = cand_norm
            if score == 1.0:
                break
        if best_id is not None and best_score >= _FUZZY_THRESHOLD:
            return best_id, f"fuzzy:{best_score:.2f}", name_by_id.get(best_id)
    return None, None, None


def _nearest_candidates(name: str, candidates: list[tuple[str, int]],
                        name_by_id: dict[int, str], top_n: int = 3
                        ) -> list[dict]:
    """Return the top-N nearest Exercise rows for an unmatched name. Used to
    populate `unmatched_exercises[].closest_candidates` so the user can audit."""
    if not name or not candidates:
        return []
    lk = _normalize_loose(name)
    if not lk:
        return []
    scored = []
    for cand_norm, cand_id in candidates:
        score = SequenceMatcher(None, lk, cand_norm).ratio()
        scored.append((score, cand_id, cand_norm))
    scored.sort(reverse=True)
    return [
        {"exercise_id": cid, "exercise_name": name_by_id.get(cid),
         "similarity": round(s, 3)}
        for (s, cid, _norm) in scored[:top_n]
    ]


def _parse_sets_reps(sxr: str | None) -> tuple[int | None, list[int]]:
    """'3 x 5' -> (3, [5,5,5]); '1 x 60s' -> (1, []); '3x8' -> (3, [8,8,8]).
    Returns (n_sets, reps_array)."""
    if not sxr:
        return None, []
    s = str(sxr).strip().lower()
    m = re.match(r"^(\d+)\s*[x×]\s*(\d+)\s*(s|sec)?\s*$", s)
    if not m:
        return None, []
    sets = int(m.group(1))
    rep_value = int(m.group(2))
    is_seconds = m.group(3) is not None
    if is_seconds:
        # Time-based work — leave reps_array empty; consumer treats as duration
        return sets, []
    return sets, [rep_value] * sets


def _build_lift_template(lift_block: dict, ex_lookup: "ExerciseLookup",
                         athlete_name: str, unmatched: list[dict]
                         ) -> dict:
    """Convert one lift block (one of the 5 movement buckets) into an
    ActivityTemplate payload with ExerciseToLift children."""
    bucket = lift_block.get("movement_bucket", "Lift")
    exercises = []
    for ex in lift_block.get("exercises") or []:
        name = ex.get("name") or ""
        ex_id, method, matched_name = ex_lookup.resolve(name)
        sxr = ex.get("sets_x_reps")
        n_sets, reps_arr = _parse_sets_reps(sxr)
        if ex_id is None:
            unmatched.append({
                "category": "lift", "bucket": bucket,
                "name": name, "sets_x_reps": sxr,
                "closest_candidates": ex_lookup.nearest(name),
            })
        exercises.append({
            "exercise_id": ex_id,                    # nullable when unmatched
            "exercise_name": name,
            "matched_exercise_name": matched_name,    # what we matched to (for audit)
            "match_method": method,                   # 'exact'/'loose'/'fuzzy:0.92'/None
            "sets_x_reps_raw": sxr,
            "sets": n_sets,
            "reps_array": reps_arr,
            "weight_array": [],                       # placeholder; coach fills
            "order": int(ex.get("order") or 0),
        })
    return {
        "role": "lift",
        "name": f"Lift {bucket} — {athlete_name}",
        "description": lift_block.get("description") or f"AI-generated lift template ({bucket})",
        "source_template_id": lift_block.get("template_id"),  # e.g. "5227-11" for audit
        "movement_bucket": bucket,
        "exercises": exercises,
    }


def _build_simple_exercise_template(
    component: dict | None, role: str, athlete_name: str,
    ex_lookup: "ExerciseLookup", unmatched: list[dict],
) -> dict | None:
    """Build a prep/bp/me ActivityTemplate payload from a recommender component."""
    if not component:
        return None
    exercises = []
    for i, ex in enumerate(component.get("exercises") or []):
        name = ex.get("exercise_name") or ""
        ex_id, method, matched_name = ex_lookup.resolve(name)
        if ex_id is None:
            unmatched.append({
                "category": role, "name": name,
                "sets": ex.get("sets"), "reps": ex.get("reps"),
                "closest_candidates": ex_lookup.nearest(name),
            })
        sets = ex.get("sets")
        reps = ex.get("reps")
        duration_seconds = ex.get("duration_seconds")
        reps_array = [int(reps)] * int(sets or 1) if (reps is not None and sets) else []
        exercises.append({
            "exercise_id": ex_id,
            "exercise_name": name,
            "matched_exercise_name": matched_name,
            "match_method": method,
            "sets": sets,
            "reps": reps,
            "duration_seconds": duration_seconds,
            "reps_array": reps_array,
            "weight_array": [],
            "order": i,
            "slot": ex.get("slot"),
            "notes": ex.get("rationale"),
        })
    return {
        "role": role,
        "name": f"{role.title()} — {athlete_name}",
        "description": component.get("reasoning") or f"AI-generated {role} template",
        "exercises": exercises,
    }


def _build_plyo_template(plyo_program: dict | None, athlete_name: str,
                          ex_lookup: "ExerciseLookup", unmatched: list[dict]
                          ) -> dict | None:
    """Build a plyo ActivityTemplate payload. Has 3 sub-sessions (P0/P1/P2),
    each maps to a PlyoToTemplate row with throwing exercises as children."""
    if not plyo_program:
        return None
    plyo_id_by_level = {"P0": 1, "P1": 2, "P2": 3, "P3": 4}
    sessions = []
    for sess in plyo_program.get("cycle") or []:
        level = sess.get("plyo_level")
        plyo_id = plyo_id_by_level.get(level)
        drills = []
        for d in sess.get("drills") or []:
            name = d.get("exercise_name") or ""
            ex_id, method, matched_name = ex_lookup.resolve(name)
            if ex_id is None:
                unmatched.append({
                    "category": "plyo", "plyo_level": level,
                    "name": name, "ball_weight": d.get("ball_weight"),
                    "closest_candidates": ex_lookup.nearest(name),
                })
            ball_enum = _format_ball_weight_back(d.get("ball_weight"))
            reps_val = d.get("reps")
            sets_val = d.get("sets") or 1
            reps_array = [int(reps_val)] * int(sets_val) if reps_val is not None else []
            drills.append({
                "throwing_exercise_id": ex_id,             # FK to Exercise.id
                "throwing_exercise_name": name,
                "matched_exercise_name": matched_name,
                "match_method": method,
                "ball_weight": ball_enum,                  # 'OZ_32', etc.
                "ball_weight_display": d.get("ball_weight"),
                "reps_array": reps_array,
                "sets": sets_val,
                "reps": reps_val,
                "order": int(d.get("order") or 0),
                "notes": d.get("rationale"),
            })
        sessions.append({
            "plyo_id": plyo_id,                            # FK to Plyo catalog
            "plyo_level": level,
            "label": sess.get("label"),
            "session_intent": sess.get("session_intent"),
            "drills": drills,
            "order": {"P0": 0, "P1": 1, "P2": 2, "P3": 3}.get(level, 99),
        })
    return {
        "role": "plyo",
        "name": f"Plyo — {athlete_name}",
        "description": plyo_program.get("reasoning") or "AI-generated plyo template",
        "sessions": sessions,
    }


_DAY_TO_LIFT_BUCKET_DEFAULT = {
    "MONDAY":    "Legs",
    "TUESDAY":   "Upper",
    "WEDNESDAY": "Total Body",
    "THURSDAY":  "Sprint",
    "FRIDAY":    "Jump",
    "SATURDAY":  None,
    "SUNDAY":    None,
}


def _build_weekly_layout(lift_templates: list[dict], plyo_program: dict | None
                         ) -> list[dict]:
    """Build the 7-day layout. Uses the LLM-suggested plyo weekly_layout when
    present; defaults the lift→day mapping to Mon=Legs/Tue=Upper/etc."""
    bucket_to_index = {t["movement_bucket"]: i for i, t in enumerate(lift_templates)}
    plyo_layout = (plyo_program or {}).get("weekly_layout") or {}
    plyo_id_by_level = {"P0": 1, "P1": 2, "P2": 3, "P3": 4}
    days = []
    for day_name, default_bucket in _DAY_TO_LIFT_BUCKET_DEFAULT.items():
        bucket = default_bucket
        lift_index = bucket_to_index.get(bucket) if bucket else None
        # Plyo level for the day — extract first "Pn" token
        plyo_text = (plyo_layout.get(day_name) or "").upper()
        plyo_level = None
        for token in ("P0", "P1", "P2", "P3"):
            if token in plyo_text:
                plyo_level = token
                break
        plyo_id = plyo_id_by_level.get(plyo_level) if plyo_level else None
        is_off_day = bool(re.search(r"\b(rest|off)\b", plyo_text, re.IGNORECASE))
        days.append({
            "day_of_week": day_name,
            "lift_template_index": lift_index,
            "lift_template_bucket": bucket,
            "plyo_level": plyo_level,
            "plyo_id": plyo_id,
            "throwing_activity_id": None,             # filled by coach in app
            "hitting_template_id": None,
            "is_off_day": is_off_day,
            "notes": None,
        })
    return days


def compile_payload(recommended_program_id: int) -> dict:
    """Compile a recommended_programs row into an App DB-compatible payload."""
    with backend_conn() as conn:
        rows = query(conn, """
            SELECT * FROM ai_layer.recommended_programs WHERE id = %s
        """, [recommended_program_id])
        if not rows:
            raise ValueError(f"Recommended program {recommended_program_id} not found")
        rec = rows[0]

        # Look up the App DB user from the athlete's email (for WeeklyTemplateApplication)
        app_user = None
        app_org_id = None
        if rec.get("athlete_uuid"):
            ath = query(conn, """
                SELECT email FROM analytics.d_athletes WHERE athlete_uuid = %s
            """, [rec["athlete_uuid"]])
            if ath and ath[0].get("email"):
                # User table does NOT carry orgId — only pull the id here.
                u = query(conn, """
                    SELECT "id"::text AS app_user_id
                    FROM app_db_snapshot."User"
                    WHERE lower("email") = lower(%s) LIMIT 1
                """, [ath[0]["email"]])
                if u:
                    app_user = u[0]["app_user_id"]
                # Org affiliation comes from any existing Program row for this user.
                # Wrapped defensively in case Program's org column is named differently
                # in the current snapshot (orgId vs organizationId vs no org column).
                if app_user:
                    try:
                        org = query(conn, """
                            SELECT "orgId"::text AS org_id
                            FROM app_db_snapshot."Program"
                            WHERE "userId"::text = %s
                              AND "orgId" IS NOT NULL
                            ORDER BY "createdAt" DESC NULLS LAST
                            LIMIT 1
                        """, [app_user])
                        if org:
                            app_org_id = org[0]["org_id"]
                    except Exception as e:
                        # Don't tank payload compilation just because we can't
                        # resolve the org — leave it null and let the API endpoint
                        # decide later.
                        print(f"[payload_builder] org lookup skipped: {e}")
                        conn.rollback()

        ex_lookup = ExerciseLookup(conn)

    unmatched: list[dict] = []
    athlete_name = rec.get("athlete_name") or "Athlete"
    timestamp = datetime.now().strftime("%Y-%m-%d")

    # Build lift activity templates — one per movement bucket
    lift_templates: list[dict] = []
    for lift_block in (rec.get("lift_program") or {}).get("templates") or []:
        lift_templates.append(
            _build_lift_template(lift_block, ex_lookup, athlete_name, unmatched)
        )

    # Build prep / bp / plyo / me templates
    prep_template = _build_simple_exercise_template(
        rec.get("prep_program"), "prep", athlete_name, ex_lookup, unmatched)
    bp_template = _build_simple_exercise_template(
        rec.get("bulletproofing_program"), "bulletproofing", athlete_name, ex_lookup, unmatched)
    me_template = _build_simple_exercise_template(
        rec.get("movement_enhancement_program"), "movement_enhancement",
        athlete_name, ex_lookup, unmatched)
    plyo_template = _build_plyo_template(
        rec.get("plyo_program"), athlete_name, ex_lookup, unmatched)
    hitting_template = _build_simple_exercise_template(
        rec.get("hitting_program"), "hitting", athlete_name, ex_lookup, unmatched)

    # Weekly layout
    weekly_days = _build_weekly_layout(lift_templates, rec.get("plyo_program"))

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(),
        "source": {
            "recommended_program_id": recommended_program_id,
            "athlete_uuid": rec.get("athlete_uuid"),
            "athlete_name": athlete_name,
            "age_group": rec.get("age_group"),
            "focus": rec.get("focus"),
            "role": rec.get("role"),
            "model_name": rec.get("model_name"),
        },
        "app_db_links": {
            "app_user_id": app_user,
            "org_id": app_org_id,
        },

        # Top-level WeeklyTemplate row
        "weekly_template": {
            "name": f"AI Recommended — {athlete_name} — {rec.get('focus')} — {timestamp}",
            "description": (
                f"AI-generated weekly program. "
                f"Focus: {rec.get('focus')}. "
                f"Source: ai_layer.recommended_programs id={recommended_program_id}."
            ),
            "is_generated": True,
            "is_internal": True,
            "is_archived": False,
            # FKs (filled by the inserter once activity_templates land):
            "prep_template_role":              "prep" if prep_template else None,
            "bulletproofing_template_role":    "bulletproofing" if bp_template else None,
            "plyo_template_role":              "plyo" if plyo_template else None,
            "movement_enhancement_template_role": "movement_enhancement" if me_template else None,
        },

        # All ActivityTemplate rows to insert (each becomes one ActivityTemplate +
        # its child exercise/drill rows in the relevant ExerciseToXTemplate or
        # PlyoToTemplate/ThrowingExerciseToPlyoTemplate table).
        "activity_templates": list(filter(None, [
            prep_template,
            bp_template,
            me_template,
            plyo_template,
            hitting_template,
            *lift_templates,
        ])),

        # WeeklyTemplateDay rows. lift_template_index points into
        # activity_templates[] where role=='lift' for that bucket.
        "days": weekly_days,

        # WeeklyTemplateApplication: tie the new template to the athlete
        "application": {
            "user_app_id": app_user,                          # nullable if no match
            "athlete_uuid": rec.get("athlete_uuid"),
        } if app_user else None,

        # Names that didn't resolve to an Exercise.id — these need manual
        # reconciliation before the insertion can succeed (either create the
        # exercise in the App DB Exercise table, or correct the name).
        "unmatched_exercises": unmatched,
    }
    return payload


def save_payload(payload: dict, recommended_program_id: int) -> Path:
    """Write the payload JSON to outputs/."""
    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    name = (payload.get("source", {}).get("athlete_name") or "athlete").replace(" ", "_")
    path = out_dir / f"payload_{recommended_program_id}_{name}_{ts}.json"
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return path
