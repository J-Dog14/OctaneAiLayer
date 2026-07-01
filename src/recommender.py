"""
Lift program recommender.

Given an athlete + a chosen training focus, produces a draft weekly lift
program by:
  1. Loading the athlete's deficit profile (from ai_layer.athlete_profiles).
  2. Validating the requested focus is available for their age_group (only
     focuses that actually have loaded templates count as available).
  3. Finding "named-athlete neighbors" — historical athletes the strength
     coach explicitly tagged in the template Index sheet (their profile is
     the closest match by Z-score Euclidean distance).
  4. Pulling candidate template families that match age_group + focus.
  5. Calling Gemini with the lift-programming skill loaded into the system
     prompt; asking it to pick the right templates + levels.
  6. Looking up the actual exercises for each selected template_id from
     ai_layer.lift_template_exercises (so the output is exact catalog
     data, not LLM-generated exercise names).
  7. Persisting the draft to ai_layer.recommended_programs with full audit
     metadata.

Run:
    python -m src.main recommend <athlete_uuid> --focus power
    python -m src.main available-focuses <athlete_uuid>
"""
from __future__ import annotations

import json
import math
import time
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from src.db import backend_conn, exec_sql, query, returning_id
from src.gemini_client import generate
from src.schemas import (
    ExerciseComponentOutput,
    LiftTemplateOutput,
    PlyoOutput,
)


def _salvage_truncated_json(text: str) -> str | None:
    """Try to salvage a JSON object that got cut off mid-string.

    Strategy: walk forward tracking brace/bracket depth and string state, and
    when the response was truncated find the last position where depth==0 was
    reachable. Conservative — returns None if salvage isn't possible.
    """
    if not text:
        return None
    s = text.lstrip()
    if not s.startswith("{") and not s.startswith("["):
        return None
    depth = 0
    in_string = False
    escape = False
    last_safe = -1
    for i, ch in enumerate(s):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                last_safe = i
    if last_safe < 0:
        return None
    return s[: last_safe + 1]
from src.prompt_loader import load_and_register

SKILL_NAME = "lift-programming"

# Skill folder roots. Each component's skill lives in its own subfolder.
SKILLS_ROOT = Path(__file__).resolve().parents[1] / "skills"
SKILL_DIR = SKILLS_ROOT / "lift-programming"
PREP_SKILL_DIR = SKILLS_ROOT / "prep-programming"
BP_SKILL_DIR   = SKILLS_ROOT / "bulletproofing-programming"
ME_SKILL_DIR   = SKILLS_ROOT / "movement-enhancement-programming"
HIT_SKILL_DIR  = SKILLS_ROOT / "hitting-programming"
PLYO_SKILL_DIR = SKILLS_ROOT / "plyo-programming"


# ─────────────────────────── Age group ↔ template level mapping ─────────────
# analytics.d_athletes.age_group uses upper-case canonical values ('COLLEGE',
# 'HIGH SCHOOL', 'PRO', 'YOUTH'). ai_layer.lift_templates.athlete_level uses
# the human-readable spreadsheet-decoded values ('College', 'High School',
# 'High School (Beginner)', 'Pro', 'Before Baseball Club (Beginner)', etc.).
# A single backend age_group may map to MULTIPLE template levels (e.g. a
# "HIGH SCHOOL" athlete could fit either the mainstream HS or HS-Beginner
# template families). This is the bridge.
_AGE_GROUP_TO_TEMPLATE_LEVELS: dict[str, list[str]] = {
    "COLLEGE":       ["College"],
    "HIGH SCHOOL":   ["High School", "High School (Beginner)"],
    "PRO":           ["Pro"],
    "YOUTH":         ["Before Baseball Club (Beginner)", "Baseball Club (Advanced)"],
    "MIDDLE SCHOOL": ["Before Baseball Club (Beginner)", "Baseball Club (Advanced)"],
    # Softball isn't in d_athletes.age_group typically — softball athletes are
    # tagged by training level. If you start using a 'SOFTBALL' age_group later,
    # add it here.
}


def _normalize_age_group(ag: str | None) -> str:
    """Strip + uppercase the backend age_group so it matches our keys."""
    return (ag or "").strip().upper()


def template_levels_for(age_group: str | None) -> list[str]:
    """Map a backend age_group to the list of template athlete_level values
    that should be searched. Empty list = no templates match."""
    return _AGE_GROUP_TO_TEMPLATE_LEVELS.get(_normalize_age_group(age_group), [])


# ─────────────────────────── Input validation ───────────────────────────────

def get_available_focuses(age_group: str) -> list[str]:
    """Return the list of focuses that actually have loaded templates for this
    athlete level. Dynamic — adapts as templates are added/removed."""
    levels = template_levels_for(age_group)
    if not levels:
        return []
    with backend_conn() as conn:
        rows = query(conn, """
            SELECT focus, COUNT(*) AS n
            FROM ai_layer.lift_templates
            WHERE athlete_level = ANY(%s::text[]) AND focus IS NOT NULL
            GROUP BY focus
            HAVING COUNT(*) >= 3
            ORDER BY n DESC
        """, [levels])
    return [r["focus"] for r in rows]


def load_athlete_profile(athlete_uuid: str) -> dict:
    """Load the athlete's latest profile + demographic context.

    Includes a fallback for has_pitching_data / has_hitting_data: if the
    d_athletes flag is False but the warehouse fact tables (f_pitching_trials,
    f_hitting_trials) have rows for the athlete, override the flag. The flags
    are derived nightly and sometimes lag behind raw data ingestion, especially
    for athletes inserted directly into the warehouse.
    """
    with backend_conn() as conn:
        rows = query(conn, """
            SELECT p.athlete_uuid, p.age_group, p.as_of_date,
                   p.raw_values, p.z_scores, p.source_dates,
                   d.name, d.has_pitching_data, d.has_hitting_data, d.email
            FROM ai_layer.athlete_profiles p
            JOIN analytics.d_athletes d USING (athlete_uuid)
            WHERE p.athlete_uuid = %s
            ORDER BY p.as_of_date DESC
            LIMIT 1
        """, [athlete_uuid])
        if not rows:
            raise ValueError(
                f"No profile found for athlete {athlete_uuid}. "
                f"Run profile/backfill first."
            )
        profile = rows[0]

        # Secondary signal from fact tables — source of truth when the flag
        # is stale.
        if not profile.get("has_pitching_data"):
            pt = query(conn, """
                SELECT 1 FROM public.f_pitching_trials
                WHERE athlete_uuid = %s LIMIT 1
            """, [athlete_uuid])
            if pt:
                profile["has_pitching_data"] = True
                print(f"[recommender] {profile.get('name')}: "
                      f"d_athletes.has_pitching_data=False but f_pitching_trials "
                      f"has rows → treating as pitcher")
        if not profile.get("has_hitting_data"):
            ht = query(conn, """
                SELECT 1 FROM public.f_hitting_trials
                WHERE athlete_uuid = %s LIMIT 1
            """, [athlete_uuid])
            if ht:
                profile["has_hitting_data"] = True
                print(f"[recommender] {profile.get('name')}: "
                      f"d_athletes.has_hitting_data=False but f_hitting_trials "
                      f"has rows → treating as hitter")
    return profile


# ────────────────────────── Similar-athlete search ──────────────────────────

def _profile_distance(z1: dict | None, z2: dict | None) -> float:
    """Euclidean distance over the metrics both profiles have (NULLs treated as 0)."""
    if not z1 or not z2:
        return float("inf")
    common = set(z1) & set(z2)
    if len(common) < 5:
        return float("inf")
    total = 0.0
    n = 0
    for k in common:
        v1 = z1.get(k)
        v2 = z2.get(k)
        if v1 is None or v2 is None:
            continue
        total += (float(v1) - float(v2)) ** 2
        n += 1
    if n < 5:
        return float("inf")
    return math.sqrt(total / n)   # mean-squared (so different-coverage athletes compare fairly)


def find_similar_athletes(
    target_uuid: str, *,
    k: int = 10,
    age_group_filter: str | None = None,
    role_filter: str | None = None,
) -> list[dict]:
    """Find the K athletes whose deficit profile is closest to the target's,
    by Z-score Euclidean distance. Reusable across all program components
    (prep / bp / me / plyo / hitting / lift).

    Filters:
      age_group_filter: limit to athletes in this age_group (canonical form,
          e.g. 'COLLEGE'). None = no filter.
      role_filter: 'pitcher' / 'hitter' / None = no filter.
    """
    with backend_conn() as conn:
        target = query(conn, """
            SELECT z_scores FROM ai_layer.athlete_profiles
            WHERE athlete_uuid = %s
            ORDER BY as_of_date DESC LIMIT 1
        """, [target_uuid])
        if not target or not target[0]["z_scores"]:
            return []
        target_z = target[0]["z_scores"]

        clauses = ["p.athlete_uuid <> %s"]
        params: list[Any] = [target_uuid]
        if age_group_filter:
            clauses.append("TRIM(d.age_group) = %s")
            params.append(age_group_filter)
        if role_filter == "pitcher":
            clauses.append("d.has_pitching_data = TRUE")
        elif role_filter == "hitter":
            clauses.append("d.has_hitting_data = TRUE")

        rows = query(conn, f"""
            SELECT DISTINCT ON (p.athlete_uuid)
                   p.athlete_uuid, d.name, p.z_scores
            FROM ai_layer.athlete_profiles p
            JOIN analytics.d_athletes d USING (athlete_uuid)
            WHERE {' AND '.join(clauses)}
            ORDER BY p.athlete_uuid, p.as_of_date DESC
        """, params)

    scored: list[dict] = []
    for r in rows:
        dist = _profile_distance(target_z, r["z_scores"])
        if dist == float("inf"):
            continue
        scored.append({
            "athlete_uuid": r["athlete_uuid"],
            "name": r["name"],
            "distance": round(dist, 3),
        })
    scored.sort(key=lambda x: x["distance"])
    return scored[:k]


def corpus_staple_exercises(
    category: str | list[str],
    *,
    min_athletes: int = 10,
    top_n: int = 6,
    age_group: str | None = None,
    role: str | None = None,
) -> list[dict]:
    """Top-N most-prescribed exercises in a category across the whole corpus.

    Use this as a floor for the candidate pool when the similar-athlete history
    aggregation surfaces too few options. These are the staples coaches use
    everywhere — they belong in the pool regardless of athlete-level matching.

    Returns the same shape as aggregate_historical_prescriptions so the two can
    be merged directly.
    """
    categories = [category] if isinstance(category, str) else list(category)
    clauses = [
        "pep.category = ANY(%s)",
        "pep.exercise_name IS NOT NULL",
    ]
    params: list[Any] = [categories]
    if age_group:
        clauses.append("TRIM(d.age_group) = %s")
        params.append(age_group)
    if role == "pitcher":
        clauses.append("d.has_pitching_data = TRUE")
    elif role == "hitter":
        clauses.append("d.has_hitting_data = TRUE")
    with backend_conn() as conn:
        rows = query(conn, f"""
            SELECT pep.exercise_name,
                   MAX(pep.exercise_id::text) AS exercise_id,
                   COUNT(DISTINCT pep.athlete_uuid) AS athletes_with_it,
                   ROUND(AVG(pep.n_sets)::numeric, 1)   AS avg_sets,
                   ROUND(AVG(pep.avg_reps)::numeric, 1) AS avg_reps_per_set
            FROM ai_layer.program_exercise_prescriptions pep
            JOIN analytics.d_athletes d USING (athlete_uuid)
            WHERE {' AND '.join(clauses)}
            GROUP BY pep.exercise_name
            HAVING COUNT(DISTINCT pep.athlete_uuid) >= %s
            ORDER BY athletes_with_it DESC, pep.exercise_name
            LIMIT %s
        """, params + [min_athletes, top_n])
    return rows


def _merge_candidate_pools(history_pool: list[dict],
                           staples_pool: list[dict]) -> list[dict]:
    """Merge staples into history pool. Staples come first (prepended) so they
    show up at the top of the candidate prompt, with a 'staple' marker."""
    seen = {(c.get("exercise_name") or "").strip().lower() for c in history_pool}
    merged = list(history_pool)
    added = 0
    for s in staples_pool:
        k = (s.get("exercise_name") or "").strip().lower()
        if k and k not in seen:
            s = dict(s)
            s["_is_staple"] = True
            merged.insert(added, s)  # prepend in order
            added += 1
            seen.add(k)
    return merged


def aggregate_historical_prescriptions(
    athlete_uuids: list[str],
    category: str | list[str],
    min_athletes: int = 2,
) -> list[dict]:
    """For a set of similar athletes, return commonly-prescribed exercises in
    one or more categories with frequency + typical dose.

    category can be a single string ('lift','plyo','prep','bp','hit','me') or a
    list to combine multiple categories into one pool (e.g. ['prep','me'] for
    a unified pre-session block).

    Exercises that appear across multiple categories get their counts summed —
    if the same exercise was prescribed as prep for some athletes and me for
    others, it shows up once with the combined prevalence.

    Returns exercises prescribed to >= min_athletes distinct athletes,
    ordered by prevalence (athletes_with_it DESC).
    """
    if not athlete_uuids:
        return []
    categories = [category] if isinstance(category, str) else list(category)
    with backend_conn() as conn:
        rows = query(conn, """
            SELECT exercise_name,
                   MAX(exercise_id::text) AS exercise_id,
                   MAX(exercise_type)     AS exercise_type,
                   COUNT(*)               AS times_prescribed,
                   COUNT(DISTINCT athlete_uuid) AS athletes_with_it,
                   ROUND(AVG(n_sets)::numeric, 1)        AS avg_sets,
                   ROUND(AVG(avg_reps)::numeric, 1)      AS avg_reps_per_set,
                   ROUND(AVG(max_reps)::numeric, 1)      AS avg_top_set_reps,
                   ROUND(AVG(avg_weight)::numeric, 0)    AS avg_weight
            FROM ai_layer.program_exercise_prescriptions
            WHERE athlete_uuid = ANY(%s::varchar[])
              AND category    = ANY(%s::text[])
              AND exercise_name IS NOT NULL
            GROUP BY exercise_name
            HAVING COUNT(DISTINCT athlete_uuid) >= %s
            ORDER BY athletes_with_it DESC, times_prescribed DESC
        """, [athlete_uuids, categories, min_athletes])
    return rows


def find_named_athlete_neighbors(
    target_uuid: str, age_group: str, focus: str, k: int = 5
) -> list[dict]:
    """Find the K templates whose original "made_for_athlete" has a profile
    closest to the target athlete's profile. The named athlete is a gold
    coaching signal — if our target looks like Ollie Swartz on the screen,
    Ollie's template (4222) is a strong candidate.
    """
    with backend_conn() as conn:
        target = query(conn, """
            SELECT z_scores FROM ai_layer.athlete_profiles
            WHERE athlete_uuid = %s
            ORDER BY as_of_date DESC LIMIT 1
        """, [target_uuid])
        if not target or not target[0]["z_scores"]:
            return []
        target_z = target[0]["z_scores"]

        # Templates that have named athletes attached, filtered to relevant level+focus
        levels = template_levels_for(age_group)
        if not levels:
            return []
        rows = query(conn, """
            SELECT DISTINCT t.family_id, t.family_description, t.made_for_athletes
            FROM ai_layer.lift_templates t
            WHERE t.athlete_level = ANY(%s::text[]) AND t.focus = %s
              AND t.made_for_athletes IS NOT NULL
              AND jsonb_array_length(t.made_for_athletes) > 0
        """, [levels, focus])

        candidates: list[dict] = []
        for t in rows:
            for name in (t["made_for_athletes"] or []):
                # Find that named athlete in d_athletes
                a = query(conn, """
                    SELECT athlete_uuid, name FROM analytics.d_athletes
                    WHERE name ILIKE %s OR normalized_name = lower(replace(%s, ' ', '_'))
                    LIMIT 1
                """, [f"%{name}%", name])
                if not a:
                    continue
                # Get that athlete's profile
                p = query(conn, """
                    SELECT z_scores FROM ai_layer.athlete_profiles
                    WHERE athlete_uuid = %s
                    ORDER BY as_of_date DESC LIMIT 1
                """, [a[0]["athlete_uuid"]])
                if not p:
                    continue
                dist = _profile_distance(target_z, p[0]["z_scores"])
                if dist == float("inf"):
                    continue
                candidates.append({
                    "family_id": t["family_id"],
                    "description": t["family_description"],
                    "named_athlete_uuid": a[0]["athlete_uuid"],
                    "named_athlete": name,
                    "distance": round(dist, 3),
                })

        # Dedupe by family_id (keep closest)
        by_family: dict[str, dict] = {}
        for c in candidates:
            existing = by_family.get(c["family_id"])
            if not existing or c["distance"] < existing["distance"]:
                by_family[c["family_id"]] = c

        return sorted(by_family.values(), key=lambda c: c["distance"])[:k]


def find_candidate_families(age_group: str, focus: str) -> list[dict]:
    """All families matching age_group + focus, with their movement-bucket coverage."""
    levels = template_levels_for(age_group)
    if not levels:
        return []
    with backend_conn() as conn:
        rows = query(conn, """
            SELECT family_id,
                   MAX(family_description) AS description,
                   MAX(made_for_athletes::text)::jsonb AS made_for_athletes,
                   array_agg(DISTINCT movement_bucket) FILTER (WHERE movement_bucket IS NOT NULL)
                       AS movement_buckets,
                   array_agg(DISTINCT template_id ORDER BY template_id) AS template_ids
            FROM ai_layer.lift_templates
            WHERE athlete_level = ANY(%s::text[]) AND focus = %s
              AND family_id IS NOT NULL
            GROUP BY family_id
            ORDER BY family_id
        """, [levels, focus])
    return rows


def get_template_exercises(template_id: str) -> list[dict]:
    """Pull exact exercises for a given template_id."""
    with backend_conn() as conn:
        rows = query(conn, """
            SELECT e.exercise_order, e.exercise_name, e.sets_x_reps,
                   e.parsed_sets, e.parsed_reps,
                   t.movement_bucket, t.mesocycle_number, t.family_description
            FROM ai_layer.lift_template_exercises e
            JOIN ai_layer.lift_templates t USING (template_id)
            WHERE template_id = %s
            ORDER BY e.exercise_order
        """, [template_id])
    return rows


# ────────────────────────── Skill + LLM ─────────────────────────────────────

def _load_skill_prompt() -> str:
    """Combine the lift-programming SKILL.md + reference .md files into one
    system prompt. Skip the JSON catalog — that's queried from DB, not stuffed
    in context."""
    parts: list[str] = []
    skill_md = SKILL_DIR / "SKILL.md"
    if skill_md.exists():
        parts.append("# Skill: lift-programming\n\n" + skill_md.read_text(encoding="utf-8"))
    refs_dir = SKILL_DIR / "references"
    if refs_dir.exists():
        for md in sorted(refs_dir.glob("*.md")):
            parts.append(f"\n\n## Reference: {md.stem}\n\n{md.read_text(encoding='utf-8')}")
    return "".join(parts)


_DOMAIN_PREFIXES = [
    ("Pitching mechanics (3D)",         "pitch_"),
    ("Hitting mechanics (3D)",          "hit_"),
    ("Mobility / soft tissue",          "mob_"),
    ("Proteus (active power)",          "proteus_"),
    ("Athletic screen (force plate)",   "screen_"),
]
_PER_DOMAIN_STRENGTHS = 3
_PER_DOMAIN_DEFICITS = 3


def _summarize_profile_by_domain(z: dict) -> dict[str, dict]:
    """Group metrics by domain, return top N strengths + bottom N deficits each.

    Doing it per-domain (instead of globally) ensures every assessment family
    contributes signal to the LLM. Without this, a domain with many metrics
    (e.g. pitching 3D, ~50 metrics) crowds out smaller domains (proteus has 12)
    in the top-8 view — which biases the recommender to reason mostly about
    whichever assessment was the most heavily measured, not whichever was most
    informative.
    """
    out: dict[str, dict] = {}
    for label, prefix in _DOMAIN_PREFIXES:
        rows = [(k, float(v)) for k, v in z.items()
                if v is not None and k.startswith(prefix)]
        if not rows:
            continue
        sorted_asc = sorted(rows, key=lambda kv: kv[1])
        out[label] = {
            "strengths": list(reversed(sorted_asc[-_PER_DOMAIN_STRENGTHS:])),
            "deficits":  sorted_asc[:_PER_DOMAIN_DEFICITS],
        }
    return out


def _build_user_message(profile: dict, focus: str,
                       neighbors: list[dict],
                       candidates: list[dict]) -> str:
    """Build the user content that asks the LLM to pick templates."""
    z = profile.get("z_scores") or {}

    # Group metrics by domain so every assessment family is represented.
    per_domain = _summarize_profile_by_domain(z)

    neighbor_lines = []
    for n in neighbors:
        neighbor_lines.append(
            f"  - family {n['family_id']} \"{n['description']}\" "
            f"originally built for {n['named_athlete']} (distance: {n['distance']})"
        )

    candidate_lines = []
    for c in candidates:
        buckets = ", ".join(sorted(c.get("movement_buckets") or []))
        candidate_lines.append(
            f"  - family {c['family_id']} \"{c.get('description') or '(no description)'}\" "
            f"— available buckets: [{buckets}] "
            f"— templates: {', '.join(c.get('template_ids') or [])}"
        )

    # Per-domain profile summary. This guarantees every assessment family
    # (pitching, hitting, mobility, proteus, athletic screen) contributes
    # signal to the LLM, instead of one domain crowding out the others.
    domain_section: list[str] = []
    for label, _prefix in _DOMAIN_PREFIXES:
        d = per_domain.get(label)
        if not d:
            continue
        domain_section.append(f"\n  [{label}]")
        if d["strengths"]:
            domain_section.append("    Top strengths:")
            for k, v in d["strengths"]:
                domain_section.append(f"      + {k}: Z={v:+.2f}")
        if d["deficits"]:
            domain_section.append("    Top deficits:")
            for k, v in d["deficits"]:
                domain_section.append(f"      - {k}: Z={v:+.2f}")

    return f"""ATHLETE TO PROGRAM FOR
======================
Name:       {profile.get('name')}
Age group:  {profile.get('age_group')}
Focus:      {focus}

DEFICIT PROFILE — broken out per assessment domain so you see signal from each.
Z-scores are deviations from the age-group population (negative = deficit).
Each assessment family informs different parts of programming:

  · Pitching 3D    → drives drill / plyo / med-ball selection (mechanical work)
  · Hitting 3D     → drives hitting drill / bat work
  · Mobility       → drives prep + bulletproofing prescriptions
  · Proteus        → drives med-ball / rotational power + UPPER-body power work
  · Athletic screen → drives lift programming volume + intensity
{chr(10).join(domain_section)}

CANDIDATE TEMPLATE FAMILIES (filtered to {profile.get('age_group')} × {focus}):
{chr(10).join(candidate_lines)}

NAMED-ATHLETE NEIGHBORS (templates built for historical athletes whose deficit
profile is closest to this athlete; treat these as the strongest candidates):
{chr(10).join(neighbor_lines) if neighbor_lines else "  (none found — no overlap between named athletes and this profile yet)"}

TASK
====
Pick a complete weekly program by selecting specific template_ids from the
candidate families above. A complete week typically includes the movement
buckets Legs, Upper, Total Body, Sprint, Jump — pick one template_id per
bucket. Stay within mesocycle 1 (template_ids ending in -1X) for a brand new
athlete unless their profile suggests they're already at a higher level.

Output STRICT JSON with this exact shape:

{{
  "reasoning": "1-3 sentences explaining the template family chosen and why this athlete's profile suggested it",
  "primary_family_id": "4222",
  "primary_family_rationale": "1-2 sentences explaining the family choice",
  "selected_templates": [
    {{
      "template_id": "4222-11",
      "movement_bucket": "Legs",
      "level": "L1",
      "rationale": "1-2 sentences explaining why this template for this athlete"
    }}
  ]
}}

Pick template_ids ONLY from the candidate families listed above. Do not invent
template_ids. If a needed bucket isn't available in your chosen family, you may
choose that bucket's template from a different candidate family — explain why
in its rationale.
"""


# ────────────────────────── Output builders ─────────────────────────────────

# Pitcher-specific lift accessories that appear in 7-9/13 pitcher coach
# programs but aren't in the spreadsheet templates. These are pitcher-care
# add-ons coaches embed in every lift day. Surfaced as a "lift_mobility_cap"
# block alongside the regular templates so the consumer can render them as
# a shared mobility tail across all lift days.
_PITCHER_LIFT_MOBILITY_CAP = [
    {"name": "Dead Hang",            "sets_x_reps": "1 x 30s"},
    {"name": "Shoulder CAR",         "sets_x_reps": "1 x 5 (each side)"},
    {"name": "Glute Bridge ISO",     "sets_x_reps": "1 x 30s"},
    {"name": "Hip Flexor ISO",       "sets_x_reps": "1 x 30s (each side)"},
    {"name": "Wrist Extension ISO",  "sets_x_reps": "1 x 20"},
    {"name": "Wrist Flexion Iso",    "sets_x_reps": "1 x 20"},
    {"name": "Bretzel 2.0",          "sets_x_reps": "1 x 5 (each side)"},
    {"name": "90/90 Hip Shifts",     "sets_x_reps": "2 x 8 (each side)"},
]


def _build_lift_program_json(selected: list[dict],
                              include_pitcher_mobility_cap: bool = False) -> dict:
    """Expand selected template_ids into the full lift_program JSONB payload.

    When include_pitcher_mobility_cap=True (athlete has pitching data), append
    a 'mobility_cap' block of pitcher-specific accessory work that lives on
    every lift day. Coaches universally prescribe these for pitchers but the
    spreadsheet templates skip them.
    """
    templates_with_exercises = []
    for sel in selected:
        tid = sel["template_id"]
        exercises = get_template_exercises(tid)
        if not exercises:
            continue
        templates_with_exercises.append({
            "template_id": tid,
            "movement_bucket": sel.get("movement_bucket"),
            "level": sel.get("level"),
            "rationale": sel.get("rationale"),
            "description": exercises[0].get("family_description") if exercises else None,
            "exercises": [{
                "order": int(e["exercise_order"] or 0),
                "name": e["exercise_name"],
                "sets_x_reps": e["sets_x_reps"],
                "parsed_sets": e.get("parsed_sets"),
                "parsed_reps": e.get("parsed_reps"),
            } for e in exercises],
        })
    result: dict[str, Any] = {
        "category": "lift",
        "n_templates": len(templates_with_exercises),
        "templates": templates_with_exercises,
    }
    if include_pitcher_mobility_cap:
        result["mobility_cap"] = {
            "description": "Pitcher-specific mobility / care exercises to run "
                           "alongside every lift day. Spinal decompression, "
                           "shoulder mobility, forearm care, glute activation, "
                           "and hip mobility flow.",
            "exercises": _PITCHER_LIFT_MOBILITY_CAP,
        }
    return result


# ────────────────────────── Main entry point ────────────────────────────────

def _json_default(o):
    if isinstance(o, Decimal): return float(o)
    if isinstance(o, (date, datetime)): return o.isoformat()
    raise TypeError(f"Not JSON serializable: {type(o)}")


def recommend_lift_program(athlete_uuid: str, focus: str,
                           role: str | None = None,
                           *,
                           plyo_day: str | None = None,
                           annual_phase: str | None = None,
                           athlete_role: str = "Starter",
                           game_day: str = "SATURDAY") -> dict:
    """Generate a draft lift program for one athlete + focus.

    Plyo-specific extras:
      plyo_day: 'P0'|'P1'|'P2'|'P3' or None to infer from focus
      annual_phase: phase name or None to infer from focus
      athlete_role: 'Starter' | 'Reliever' — affects cadence and volume
      game_day: day of the week the athlete pitches games — anchors the plyo
                weekly cadence. Default SATURDAY (summer-ball / fall scrimmages).

    Returns a dict with the full structured program + metadata. Also saves to
    ai_layer.recommended_programs.
    """
    t_start = time.time()

    # 1. Load profile + validate inputs
    profile = load_athlete_profile(athlete_uuid)
    age_group = profile.get("age_group")
    if not age_group:
        raise ValueError(f"Athlete {athlete_uuid} has no age_group.")

    available = get_available_focuses(age_group)
    if focus not in available:
        raise ValueError(
            f"Focus '{focus}' is not available for age group '{age_group}'. "
            f"Available focuses: {available}"
        )

    if role is None:
        if profile.get("has_pitching_data") and profile.get("has_hitting_data"):
            role = "both"
        elif profile.get("has_pitching_data"):
            role = "pitcher"
        elif profile.get("has_hitting_data"):
            role = "hitter"
        else:
            role = "unknown"

    # 2. Find named-athlete neighbors + general candidates
    print(f"[recommender] finding similar named athletes...", flush=True)
    neighbors = find_named_athlete_neighbors(athlete_uuid, age_group, focus, k=5)

    print(f"[recommender] gathering candidate families...", flush=True)
    candidates = find_candidate_families(age_group, focus)
    if not candidates:
        raise ValueError(
            f"No template families found for {age_group} × {focus}. "
            f"Have you run `load-templates`?"
        )

    # 3. Call Gemini with the lift-programming skill
    print(f"[recommender] calling Gemini ({len(candidates)} candidate families, "
          f"{len(neighbors)} named-athlete matches)...", flush=True)
    skill_prompt = _load_skill_prompt()
    user_msg = _build_user_message(profile, focus, neighbors, candidates)

    # Register the skill version so this generation is reproducible
    _, prompt_version_id = load_and_register(SKILL_NAME)

    result = generate(
        system_prompt=skill_prompt,
        user_content=user_msg,
        response_mime_type="application/json",
        response_schema=LiftTemplateOutput,
        # Lift template selection: 5 templates × ~200 tokens rationale each
        # + ~150 tokens overall reasoning + JSON overhead. 16k is comfortable.
        max_output_tokens=16384,
    )

    raw_text = result["text"]
    try:
        selection = json.loads(raw_text)
    except json.JSONDecodeError as e:
        # Try a forgiving re-parse: strip everything after the last closing brace.
        cleaned = _salvage_truncated_json(raw_text)
        if cleaned is not None:
            try:
                selection = json.loads(cleaned)
                print("[recommender] WARNING: lift template JSON was truncated, "
                      "salvaged via last-brace trim")
            except json.JSONDecodeError:
                raise RuntimeError(
                    f"LLM returned non-JSON response. First 500 chars:\n{raw_text[:500]}"
                ) from e
        else:
            raise RuntimeError(
                f"LLM returned non-JSON response. First 500 chars:\n{raw_text[:500]}"
            ) from e

    # 4. Validate template_ids exist + assemble exercise lists
    selected = selection.get("selected_templates") or []
    valid_template_ids = {c_t for c in candidates for c_t in (c.get("template_ids") or [])}
    invalid = [s for s in selected if s.get("template_id") not in valid_template_ids]
    if invalid:
        print(f"[recommender] WARNING — LLM picked {len(invalid)} invalid template_ids: "
              f"{[s.get('template_id') for s in invalid]}. They will be dropped.")
        selected = [s for s in selected if s.get("template_id") in valid_template_ids]

    lift_program = _build_lift_program_json(
        selected,
        # Pitcher-specific lift-day mobility cap (Dead Hang / Shoulder CAR /
        # Wrist Ext-Flex ISO / Bretzel 2.0 / etc.) — coaches embed these in
        # every lift workout for pitchers but the spreadsheet templates skip
        # them. Surface them as a separate cap block so they're available to
        # every lift day in the output.
        include_pitcher_mobility_cap=bool(profile.get("has_pitching_data")),
    )

    # 5. Find general similar athletes (not just named ones) for prep / future components
    print(f"[recommender] finding similar athletes for prep + future components...",
          flush=True)
    similar_athletes = find_similar_athletes(
        athlete_uuid, k=8,
        age_group_filter=_normalize_age_group(age_group),
        role_filter=role if role in ("pitcher", "hitter") else None,
    )

    # 6. Generate the history-driven components, in the order they appear in a session.
    #    Each generator gates itself by role/data availability and returns None when
    #    it can't produce output (no candidates / wrong role).
    components: dict[str, dict | None] = {}
    component_metas: dict[str, dict] = {}

    # NOTE: ME is folded into Prep (coaches treat them as one pre-session block).
    # We keep the movement_enhancement_program column in the schema but no longer
    # generate it. The prep generator pulls from BOTH prep and me historical
    # categories to assemble a unified prep block.

    # Standard generators (no per-component extras)
    for label, generator in [
        ("prep_program",            generate_prep_component),
        ("bulletproofing_program",  generate_bp_component),
        ("hitting_program",         generate_hitting_component),
    ]:
        print(f"[recommender] generating {label}...", flush=True)
        component, meta = generator(athlete_uuid, profile, focus, similar_athletes)
        components[label] = component
        component_metas[label] = meta

    # Plyo has extra inputs (plyo day level, annual phase, starter/reliever role)
    print(f"[recommender] generating plyo_program "
          f"(plyo_day={plyo_day or 'auto'}, phase={annual_phase or 'auto'}, "
          f"athlete_role={athlete_role})...", flush=True)
    components["plyo_program"], component_metas["plyo_program"] = generate_plyo_component(
        athlete_uuid, profile, focus, similar_athletes,
        plyo_day=plyo_day, annual_phase=annual_phase, athlete_role=athlete_role,
        game_day=game_day,
    )

    # 7. Aggregate cost/latency across all LLM calls (lift + every component)
    total_cost = (result.get("cost_usd") or 0.0) + sum(
        (m.get("cost_usd") or 0.0) for m in component_metas.values()
    )
    total_latency = (result.get("latency_ms") or 0) + sum(
        (m.get("latency_ms") or 0) for m in component_metas.values()
    )

    # Concatenate all raw LLM responses for debugging
    raw_combined = raw_text
    for label, meta in component_metas.items():
        if meta.get("raw_response"):
            raw_combined += f"\n\n[{label.upper()} RAW]\n{meta['raw_response']}"

    # 8. Save to DB
    payload = {
        "athlete_uuid": athlete_uuid,
        "athlete_name": profile.get("name"),
        "age_group": age_group,
        "focus": focus,
        "role": role,
        "model_name": result["model"],
        "prompt_version_id": prompt_version_id,
        "selected_template_ids": [s.get("template_id") for s in selected],
        "similar_athletes": (neighbors + similar_athletes),
        "reasoning_text": selection.get("reasoning"),
        "raw_llm_response": raw_combined,
        "lift_program": lift_program,
        "prep_program":                   components.get("prep_program"),
        "bulletproofing_program":         components.get("bulletproofing_program"),
        "movement_enhancement_program":   None,   # folded into Prep (see orchestrator note)
        "plyo_program":                   components.get("plyo_program"),
        "hitting_program":                components.get("hitting_program"),
        "generation_cost_usd": total_cost,
        "generation_latency_ms": total_latency,
    }

    with backend_conn() as conn:
        new_id = returning_id(conn, """
            INSERT INTO ai_layer.recommended_programs
              (athlete_uuid, athlete_name, age_group, focus, role,
               model_name, prompt_version_id,
               selected_template_ids, similar_athletes, reasoning_text,
               raw_llm_response,
               lift_program, prep_program, bulletproofing_program,
               movement_enhancement_program, plyo_program, hitting_program,
               generation_cost_usd, generation_latency_ms)
            VALUES (%s, %s, %s, %s, %s,
                    %s, %s,
                    %s::jsonb, %s::jsonb, %s,
                    %s,
                    %s::jsonb, %s::jsonb, %s::jsonb,
                    %s::jsonb, %s::jsonb, %s::jsonb,
                    %s, %s)
            RETURNING id
        """, [
            payload["athlete_uuid"], payload["athlete_name"], payload["age_group"],
            payload["focus"], payload["role"],
            payload["model_name"], payload["prompt_version_id"],
            json.dumps(payload["selected_template_ids"], default=_json_default),
            json.dumps(payload["similar_athletes"], default=_json_default),
            payload["reasoning_text"], payload["raw_llm_response"],
            json.dumps(payload["lift_program"], default=_json_default),
            json.dumps(payload["prep_program"], default=_json_default) if payload["prep_program"] else None,
            json.dumps(payload["bulletproofing_program"], default=_json_default) if payload["bulletproofing_program"] else None,
            json.dumps(payload["movement_enhancement_program"], default=_json_default) if payload["movement_enhancement_program"] else None,
            json.dumps(payload["plyo_program"], default=_json_default) if payload["plyo_program"] else None,
            json.dumps(payload["hitting_program"], default=_json_default) if payload["hitting_program"] else None,
            payload["generation_cost_usd"], payload["generation_latency_ms"],
        ])
    payload["recommended_program_id"] = new_id
    payload["total_elapsed_ms"] = int((time.time() - t_start) * 1000)
    return payload


# ════════════════════════ HISTORY-DRIVEN COMPONENTS ═════════════════════════
# Prep, Bulletproofing, Movement Enhancement, Hitting, and Plyo all share a
# common pattern:
#   1. Find similar athletes (general profile distance)
#   2. Aggregate what those athletes were prescribed in this category
#   3. Hand the candidate pool + deficit profile to Gemini with the component's
#      skill prose
#   4. Validate output against the candidate pool, drop hallucinations
#   5. Return component JSONB + cost/latency metadata
#
# Variants:
#   - Hitting + plyo are role-gated (hitter-only / pitcher-only)
#   - Plyo includes plyo_ball_weight in its candidate strings
#   - Each component sets its own expected exercise count + slot taxonomy
#     via its skill content


def _load_skill_prose(skill_dir: Path) -> str:
    """Load a skill's SKILL.md + any markdown references into one string."""
    parts: list[str] = []
    skill_md = skill_dir / "SKILL.md"
    if skill_md.exists():
        parts.append(skill_md.read_text(encoding="utf-8"))
    refs_dir = skill_dir / "references"
    if refs_dir.exists():
        for md in sorted(refs_dir.glob("*.md")):
            parts.append(f"\n\n## Reference: {md.stem}\n\n{md.read_text(encoding='utf-8')}")
    return "".join(parts)


def _format_domain_section(z: dict) -> str:
    """Per-domain profile breakdown for use in any history-driven prompt."""
    per_domain = _summarize_profile_by_domain(z)
    lines: list[str] = []
    for label, _prefix in _DOMAIN_PREFIXES:
        d = per_domain.get(label)
        if not d:
            continue
        lines.append(f"\n  [{label}]")
        if d["strengths"]:
            lines.append("    Top strengths:")
            for k, v in d["strengths"]:
                lines.append(f"      + {k}: Z={v:+.2f}")
        if d["deficits"]:
            lines.append("    Top deficits:")
            for k, v in d["deficits"]:
                lines.append(f"      - {k}: Z={v:+.2f}")
    return "\n".join(lines)


def _format_candidate_pool(candidates: list[dict], extra_field: str | None = None,
                           cap: int = 40) -> list[str]:
    """Format the candidate exercises into prompt lines.

    Tags:
      [STAPLE] — corpus-wide common exercise (turn on only when caller asks)
      [ME]     — exercise lives in the Movement Enhancement category; in prep
                 these must be placed in the 'movement enhancement' slot at the
                 END of the prep block.
    """
    lines = []
    for c in candidates[:cap]:
        avg_sets = c.get("avg_sets") or "?"
        avg_reps = c.get("avg_reps_per_set") or "?"
        extra = f" ({c.get(extra_field)})" if extra_field and c.get(extra_field) else ""
        tags = []
        if c.get("_is_pitcher_staple"):
            tags.append("[PITCHER STAPLE]")
        if c.get("_is_staple"):
            tags.append("[STAPLE]")
        if c.get("_is_me"):
            tags.append("[ME]")
        tag_str = (" " + " ".join(tags)) if tags else ""
        if c.get("_is_pitcher_staple"):
            n_label = "pitchers (corpus)"
        elif c.get("_is_staple"):
            n_label = "athletes (corpus)"
        else:
            n_label = "similar athletes"
        lines.append(
            f"  - \"{c['exercise_name']}\"{extra}{tag_str} — used by "
            f"{c['athletes_with_it']} {n_label} "
            f"(typical dose: {avg_sets} × {avg_reps})"
        )
    return lines


def _generate_history_based_component(
    *,
    component_label: str,             # 'prep', 'bulletproofing', etc.
    category: str | list[str],        # 'prep' OR ['prep','me'] etc.
    skill_dir: Path,
    profile: dict,
    focus: str,
    similar_athletes: list[dict],
    role_gate: str | None = None,     # if set, skip if profile role doesn't match
    target_count_text: str = "6-10",  # what to ask the LLM for
    extra_candidate_field: str | None = None,  # e.g. plyo_ball_weight for plyo
    include_staples_floor: bool = False,  # add corpus-wide staples to the pool
    staples_top_n: int = 6,
    include_pitcher_floor: bool = False,  # role-gated pitcher staples (BP)
) -> tuple[dict | None, dict]:
    """Run the standard history-based recommendation pattern.

    Returns (component_dict | None, metadata). Returns None for component_dict
    when role-gated and athlete doesn't match, or when candidate pool is empty.
    """
    metadata: dict[str, Any] = {
        "n_candidates": 0,
        "raw_response": None,
        "cost_usd": 0.0,
        "latency_ms": 0,
    }

    # Role gating
    if role_gate and not profile.get(f"has_{role_gate}_data"):
        print(f"[{component_label}] athlete is not a {role_gate}, skipping")
        return None, metadata

    similar_uuids = [s["athlete_uuid"] for s in similar_athletes]
    candidates = aggregate_historical_prescriptions(
        similar_uuids, category=category, min_athletes=2,
    )
    # Tag ME-origin exercises so the LLM can place them in the ME slot at the
    # end of the prep block. The aggregate function pools ['prep','me'] for
    # prep, so without this tag the LLM can't distinguish ME from other prep.
    if isinstance(category, list) and "me" in category:
        with backend_conn() as conn:
            me_rows = query(conn, """
                SELECT DISTINCT exercise_name
                FROM ai_layer.program_exercise_prescriptions
                WHERE category = 'me' AND exercise_name IS NOT NULL
            """)
            me_names_lc = {(r["exercise_name"] or "").strip().lower()
                           for r in me_rows}
        for c in candidates:
            if (c.get("exercise_name") or "").strip().lower() in me_names_lc:
                c["_is_me"] = True
    history_n = len(candidates)

    if include_pitcher_floor and profile.get("has_pitching_data"):
        # Role-gated: pitcher-specific BP staples. Eval data shows coaches
        # prescribe ~5-7 arm-care exercises to virtually every pitcher
        # independent of Z-score deficit signal. NOT a corpus-wide floor —
        # only fires for pitchers.
        with backend_conn() as _conn:
            pitcher_staples = _build_pitcher_staples_pool(_conn)
        existing = {(c.get("exercise_name") or "").strip().lower()
                    for c in candidates}
        if not pitcher_staples:
            print(f"[{component_label}] pitcher staples query returned 0 rows — "
                  f"check whether {len(_PITCHER_BP_STAPLES)} curated staple names "
                  f"exist in ai_layer.program_exercise_prescriptions "
                  f"(category='bp') with exact spelling")
        else:
            new_staples = [s for s in pitcher_staples
                           if (s.get("exercise_name") or "").strip().lower() not in existing]
            already_present = [s.get("exercise_name") for s in pitcher_staples
                               if (s.get("exercise_name") or "").strip().lower() in existing]
            for s in new_staples:
                s["_is_pitcher_staple"] = True
            # Also tag any history-pool staples that match the curated list,
            # so the LLM treats them as pitcher staples in the prompt
            staple_names_lc = {(s.get("exercise_name") or "").strip().lower()
                               for s in pitcher_staples}
            for c in candidates:
                if (c.get("exercise_name") or "").strip().lower() in staple_names_lc:
                    c["_is_pitcher_staple"] = True
            candidates = new_staples + candidates
            print(f"[{component_label}] added {len(new_staples)} new pitcher staple(s) "
                  f"to candidate pool (already in history: {len(already_present)})")
            if new_staples:
                print(f"[{component_label}]   added: "
                      f"{[s.get('exercise_name') for s in new_staples]}")
            if already_present:
                print(f"[{component_label}]   already in history (tagged as staples): "
                      f"{already_present}")

    if include_staples_floor:
        # Force-include the top-N corpus-wide staples for this category.
        # Useful for components where the similar-athlete pool tends to be too
        # narrow (e.g. bulletproofing — coaches use a stable rotation of
        # Sleeper Stretch / Pail-Rail / Young Stretch / Scorpions for everyone).
        staples = corpus_staple_exercises(
            category, top_n=staples_top_n,
            age_group=profile.get("age_group"),
            role="pitcher" if profile.get("has_pitching_data") else (
                  "hitter" if profile.get("has_hitting_data") else None),
        )
        if staples:
            candidates = _merge_candidate_pools(candidates, staples)
            staples_added = len(candidates) - history_n
            print(f"[{component_label}] added {staples_added} corpus staple(s) to "
                  f"the {history_n}-entry history pool "
                  f"(staples: {[s['exercise_name'] for s in staples[:staples_added]]})")

    metadata["n_candidates"] = len(candidates)
    if not candidates:
        print(f"[{component_label}] no candidate exercises found in similar-athlete history — skipping")
        return None, metadata

    print(f"[{component_label}] {len(candidates)} candidate exercises (history + staples)")

    skill_prose = _load_skill_prose(skill_dir)
    if not skill_prose:
        print(f"[{component_label}] WARNING: skill prose not found at {skill_dir}")
        skill_prose = (
            f"You are selecting exercises for the {component_label} component. "
            f"Pick from the candidate pool only, with sensible dose."
        )

    similar_line = ", ".join(f"{s['name']} (d={s['distance']})" for s in similar_athletes[:5])
    domain_section = _format_domain_section(profile.get("z_scores") or {})
    candidate_lines = _format_candidate_pool(candidates, extra_field=extra_candidate_field)

    user_msg = f"""ATHLETE TO PROGRAM {component_label.upper()} FOR
============================
Name:       {profile.get('name')}
Age group:  {profile.get('age_group')}
Focus:      {focus}

5 MOST SIMILAR ATHLETES (the candidate pool below is aggregated from their
actual coach-prescribed {component_label} work):
  {similar_line or "(no similar athletes found)"}

DEFICIT PROFILE — per assessment domain (use the deficits relevant to {component_label}):
{domain_section}

CANDIDATE EXERCISES (what coaches actually prescribed to similar athletes,
ordered by prevalence). PICK ONLY FROM THIS LIST:

  Entries marked [PITCHER STAPLE] are arm-care exercises coaches prescribe to
  nearly every pitcher regardless of specific deficits — Sleeper Stretch,
  Young Stretch, Horizontal Abduction Scorpions, Bretzel 2.0, Dead Hang Series,
  etc.

  **For PITCHERS, your BP output MUST include at least 4 of the available
  [PITCHER STAPLE] entries as the foundation of the block.** These are
  non-negotiable arm-care work. After picking the staples, add 2-4 more
  deficit-specific exercises from the rest of the candidate pool to address
  this athlete's specific Z-score deficits. A pitcher's BP block should be
  6-10 exercises total — 4-6 staples + 2-4 deficit-specific.

  Entries marked [STAPLE] are corpus-wide defaults — exercises that >10 athletes
  across the population receive in this category. Coaches use them as a stable
  base layer. Strongly consider including 2-4 staples unless they clearly
  conflict with this athlete's deficit profile.

  Entries marked [ME] are Movement Enhancement exercises (med balls, waterbags,
  PVC, Indian clubs, etc.). These live INSIDE the prep block but in a dedicated
  slot at the END — AFTER general warmup, mobility, activation, and cns prep.
  In your JSON output, put each [ME] exercise in the "movement enhancement"
  slot, and order them last so order_in_program > all non-ME exercises.

{chr(10).join(candidate_lines)}

TASK
====
Follow the slot structure + selection logic in your skill description.
Select {target_count_text} exercises addressing this athlete's deficits.

Output STRICT JSON with this exact shape:
{{
  "reasoning": "1-3 sentences summarizing the athlete's dominant deficits and how the exercise selection addresses them",
  "exercises": [
    {{
      "exercise_name": "exact name from candidate pool",
      "slot": "see your skill's slot taxonomy",
      "sets": 2,
      "reps": 10,
      "duration_seconds": null,
      "rationale": "1-2 sentences tying this pick to the athlete's specific deficit"
    }}
  ]
}}

Pick exercise_name strings ONLY from the candidate pool above.
"""

    result = generate(
        system_prompt=skill_prose,
        user_content=user_msg,
        response_mime_type="application/json",
        # Structured output — model is constrained to conform to this schema.
        response_schema=ExerciseComponentOutput,
        # Prep/BP/hitting/ME: up to 15 exercises × ~80 tokens each (rationale +
        # dose fields) + reasoning overhead. 16k gives plenty of room for
        # 1-3 sentence reasoning across the full exercise list.
        max_output_tokens=16384,
    )
    metadata["raw_response"] = result["text"]
    metadata["cost_usd"] = result.get("cost_usd", 0.0)
    metadata["latency_ms"] = result.get("latency_ms", 0)

    raw_text = result["text"]
    try:
        selection = json.loads(raw_text)
    except json.JSONDecodeError:
        cleaned = _salvage_truncated_json(raw_text)
        if cleaned:
            try:
                selection = json.loads(cleaned)
                print(f"[{component_label}] WARNING: JSON was truncated, "
                      f"salvaged via last-brace trim")
            except json.JSONDecodeError:
                print(f"[{component_label}] WARNING: LLM returned non-JSON, skipping. "
                      f"First 300 chars:\n{raw_text[:300]}")
                return None, metadata
        else:
            print(f"[{component_label}] WARNING: LLM returned non-JSON, skipping. "
                  f"First 300 chars:\n{raw_text[:300]}")
            return None, metadata

    valid_names = {c["exercise_name"] for c in candidates}
    exercises = selection.get("exercises") or []
    filtered = [e for e in exercises if e.get("exercise_name") in valid_names]
    if len(filtered) < len(exercises):
        dropped = [e.get("exercise_name") for e in exercises
                   if e.get("exercise_name") not in valid_names]
        print(f"[{component_label}] dropped {len(dropped)} hallucinated exercise(s): {dropped}")

    return {
        "category": category,
        "n_exercises": len(filtered),
        "reasoning": selection.get("reasoning"),
        "exercises": filtered,
    }, metadata


# ─── Per-component thin wrappers ───────────────────────────────────────────

def generate_prep_component(athlete_uuid, profile, focus, similar_athletes):
    """Prep includes Movement Enhancement (med balls, waterbags, PVC, Indian
    clubs, PUM tools) as the final slot of the block. Coaches treat ME as part
    of prep but separate enough that it always sits at the end — after general
    warmup → mobility → activation → cns prep → movement enhancement.

    The candidate pool aggregates ['prep', 'me'] together; the [ME] tag in
    the prompt tells the LLM which entries belong in the ME slot.

    Target bumped to 10-15. Eval data shows coach prep blocks for pitchers
    average 17-27 exercises (we were under-prescribing at 7-12). Variety
    across slots matters more than minimizing volume.
    """
    return _generate_history_based_component(
        component_label="prep", category=["prep", "me"], skill_dir=PREP_SKILL_DIR,
        profile=profile, focus=focus, similar_athletes=similar_athletes,
        target_count_text="10-15",
    )


def generate_bp_component(athlete_uuid, profile, focus, similar_athletes):
    """BP generation. Z-score driven for the deficit-specific picks, with a
    *role-gated* floor of pitcher arm-care staples that coaches prescribe to
    every pitcher regardless of Z-score signal.

    Target count bumped to 6-10 because coach BP blocks for pitchers average
    ~9 exercises per program (we were under-prescribing at 4-8).
    """
    return _generate_history_based_component(
        component_label="bulletproofing", category="bp", skill_dir=BP_SKILL_DIR,
        profile=profile, focus=focus, similar_athletes=similar_athletes,
        target_count_text="6-10",
        include_pitcher_floor=True,
    )


# Curated pitcher-specific BP staples. Eval feedback (batch 59-71) refined
# the list: dropped "Wall Pec Stretch w/ ER" (we over-prescribed 12× — coaches
# rarely use it), dropped "Dead Hang Series" (we over-prescribed 4× vs coach's
# "Dead Hang" which was missed 4×), added "Hip Flexor Stretch" (coaches use
# it as a BP staple, missed 5× in the last batch) and "Dead Hang" (coach
# variant vs Dead Hang Series). Only fires when athlete has_pitching_data=True.
_PITCHER_BP_STAPLES = [
    "Horizontal Abduction Scorpions",
    "Young Stretch",
    "Bretzel 2.0",
    "Dead Hang",
    "Sleeper Stretch",
    "Sleeper Pail/Rail",
    "Hip Flexor Stretch",
]


def _build_pitcher_staples_pool(conn) -> list[dict]:
    """Resolve the curated staples list into candidate-pool format by joining
    against ai_layer.program_exercise_prescriptions for prevalence data."""
    if not _PITCHER_BP_STAPLES:
        return []
    rows = query(conn, """
        SELECT pep.exercise_name,
               MAX(pep.exercise_id::text) AS exercise_id,
               COUNT(DISTINCT pep.athlete_uuid) AS athletes_with_it,
               ROUND(AVG(pep.n_sets)::numeric, 1) AS avg_sets,
               ROUND(AVG(pep.avg_reps)::numeric, 1) AS avg_reps_per_set
        FROM ai_layer.program_exercise_prescriptions pep
        WHERE pep.category = 'bp'
          AND pep.exercise_name = ANY(%s)
        GROUP BY pep.exercise_name
        ORDER BY athletes_with_it DESC
    """, [_PITCHER_BP_STAPLES])
    out = []
    for r in rows:
        r["_is_pitcher_staple"] = True
        out.append(r)
    return out


def generate_hitting_component(athlete_uuid, profile, focus, similar_athletes):
    return _generate_history_based_component(
        component_label="hitting", category="hit", skill_dir=HIT_SKILL_DIR,
        profile=profile, focus=focus, similar_athletes=similar_athletes,
        role_gate="hitting",
        target_count_text="5-10",
    )


def _format_ball_weight(enum_str: str | None) -> str:
    """Convert DB enum 'OZ_32' / 'LB_1' to display strings '32 oz' / '1 lb'."""
    if not enum_str:
        return ""
    s = str(enum_str).upper()
    if s.startswith("OZ_"):
        return f"{s[3:]} oz"
    if s.startswith("LB_"):
        return f"{s[3:]} lb"
    if s.endswith("_OZ"):
        return f"{s[:-3]} oz"
    if s.endswith("_LB"):
        return f"{s[:-3]} lb"
    return s.lower().replace("_", " ")


def aggregate_plyo_drills_by_intensity(
    athlete_uuids: list[str],
    intensity: int,
    min_athletes: int = 2,
) -> list[dict]:
    """For a set of similar athletes, return the most-commonly-prescribed plyo
    drills at one intensity level, with the ball weights coaches actually used.

    Returns one row per (drill_name, ball_weight) pair so the LLM can see how
    the same drill was prescribed at different weights. Drill prevalence is
    measured by distinct athletes who received that drill at that intensity.
    """
    if not athlete_uuids:
        return []
    with backend_conn() as conn:
        rows = query(conn, """
            SELECT exercise_name,
                   plyo_ball_weight,
                   MAX(plyo_intensity) AS intensity,
                   MAX(plyo_name)      AS umbrella_name,
                   COUNT(*)            AS times_prescribed,
                   COUNT(DISTINCT athlete_uuid) AS athletes_with_it,
                   ROUND(AVG(n_sets)::numeric, 1)   AS avg_sets,
                   ROUND(AVG(avg_reps)::numeric, 1) AS avg_reps_per_set
            FROM ai_layer.program_exercise_prescriptions
            WHERE athlete_uuid = ANY(%s::varchar[])
              AND category = 'plyo'
              AND plyo_intensity = %s
              AND exercise_name IS NOT NULL
              AND exercise_name NOT LIKE 'Plyo +%%'   -- exclude umbrella-only orphans
            GROUP BY exercise_name, plyo_ball_weight
            HAVING COUNT(DISTINCT athlete_uuid) >= %s
            ORDER BY athletes_with_it DESC, times_prescribed DESC
        """, [athlete_uuids, intensity, min_athletes])
    return rows


def _infer_plyo_day(focus: str | None) -> str:
    """Default plyo day level when not explicitly provided. The CLI / caller
    can override. These defaults are conservative — coaches will adjust."""
    if focus == "In-Season":
        return "P1"
    if focus in ("Power", "Speed"):
        return "P2"
    return "P1"


def _infer_annual_phase(focus: str | None) -> str:
    """Default annual phase mapped from training focus. The CLI / caller can
    override. These are rough fits — see references/annual-throwing-system.md
    in the skill for the true seasonal calendar."""
    return {
        "Strength":    "Rebuild Capacity",
        "Power":       "Velocity Phase",
        "Speed":       "Workload Ramp",
        "In-Season":   "In-Season Maintenance",
        "Hypertrophy": "Recovery",
    }.get(focus, "Workload Ramp")


def generate_plyo_component(athlete_uuid, profile, focus, similar_athletes,
                            *,
                            plyo_day: str | None = None,    # kept for back-compat; ignored in multi-day mode
                            annual_phase: str | None = None,
                            athlete_role: str = "Starter",
                            game_day: str = "SATURDAY"):
    """Plyo recommender — generates a 3-day plyo CYCLE (P0/P1/P2) the coach
    rotates across the week.

    For each plyo level the LLM picks drills + ball weights + dose from the
    pool of drills similar pitchers were actually prescribed at that intensity.
    It also outputs a 7-day weekly layout suggesting which days run which level
    based on the annual phase.

    P3 isn't generated by default — your coaches almost never prescribe it (the
    historical data confirms this), and it requires a coach judgment call that
    only the human can make.
    """
    metadata: dict[str, Any] = {
        "n_candidates": 0, "raw_response": None,
        "cost_usd": 0.0, "latency_ms": 0,
    }

    # Role gate — only pitchers get plyo
    if not profile.get("has_pitching_data"):
        print("[plyo] athlete is not a pitcher, skipping")
        return None, metadata

    annual_phase = annual_phase or _infer_annual_phase(focus)
    similar_uuids = [s["athlete_uuid"] for s in similar_athletes]

    # Pull candidate drill pools per intensity (P0=0, P1=1, P2=2). P3 is omitted.
    intensities = {0: "P0", 1: "P1", 2: "P2"}
    candidate_pools: dict[str, list[dict]] = {}
    for intensity_int, label in intensities.items():
        candidate_pools[label] = aggregate_plyo_drills_by_intensity(
            similar_uuids, intensity=intensity_int, min_athletes=2,
        )

    total_candidates = sum(len(p) for p in candidate_pools.values())
    metadata["n_candidates"] = total_candidates
    if total_candidates == 0:
        print("[plyo] no candidate drills at any intensity — skipping")
        return None, metadata
    for label, pool in candidate_pools.items():
        print(f"[plyo]   {label}: {len(pool)} drill candidates")
    print(f"[plyo] phase={annual_phase}, role={athlete_role}")

    # Format each pool for the prompt — include drill name + ball weight + dose
    def _format_pool(pool: list[dict], cap: int = 25) -> list[str]:
        lines = []
        for c in pool[:cap]:
            weight = _format_ball_weight(c.get("plyo_ball_weight")) or "—"
            sets = c.get("avg_sets") or "?"
            reps = c.get("avg_reps_per_set") or "?"
            lines.append(
                f"  - \"{c['exercise_name']}\" @ {weight} — "
                f"prescribed to {c['athletes_with_it']} similar pitchers "
                f"(typical dose: {sets} × {reps})"
            )
        return lines

    skill_prose = _load_skill_prose(PLYO_SKILL_DIR)
    similar_line = ", ".join(f"{s['name']} (d={s['distance']})" for s in similar_athletes[:5])
    domain_section = _format_domain_section(profile.get("z_scores") or {})

    user_msg = f"""ATHLETE TO PROGRAM PLYO FOR
============================
Name:         {profile.get('name')}
Age group:    {profile.get('age_group')}
Focus:        {focus}
Role:         {athlete_role}
Annual phase: {annual_phase}
Game day:     {game_day}   (the day of the week this athlete pitches games)

5 MOST SIMILAR PITCHERS (their historical plyo prescriptions seed the candidate pools):
  {similar_line or "(none)"}

DEFICIT PROFILE — per assessment domain (pitching 3D + Proteus matter most for plyo selection):
{domain_section}

CANDIDATE DRILLS BY PLYO LEVEL
==============================
These are real drills coaches prescribed to similar pitchers, separated by intensity.
Each line is one (drill, ball weight) pair with prevalence + typical dose.

▼ P0 — Recovery day (day after outing/pen; volume target 4-5 drills, 1-2 × 6 reps,
       weights 32-7 oz; emphasis: feel + tissue restoration):
{chr(10).join(_format_pool(candidate_pools["P0"])) or "  (no candidates)"}

▼ P1 — Hybrid day (between recovery and work; volume target 5-6 drills, 1-2 × 6-8 reps,
       weights 32-5 oz; emphasis: balanced feel + light intent):
{chr(10).join(_format_pool(candidate_pools["P1"])) or "  (no candidates)"}

▼ P2 — Work day (pen day or higher-intent throwing day; volume target 5-7 drills,
       1-2 × 8-10 reps, weights 32-5 oz including some 3 oz; emphasis: max intent
       expression + inefficiency correction):
{chr(10).join(_format_pool(candidate_pools["P2"])) or "  (no candidates)"}

TASK
====
Following your plyo-programming skill, build a THREE-DAY CYCLE the coach can
rotate through the week. Output ONE complete JSON object containing P0, P1, P2
sessions plus a 7-day weekly layout suggesting which days run which level
(considering the annual phase + role).

For each session pick drills ONLY from that level's candidate pool above.
Address the athlete's biggest pitching-3D and Proteus deficits where the level
permits (lighter focus at P0, heavier at P2).

STRICT JSON SHAPE:

{{
  "reasoning": "2-4 sentences. Identify the dominant deficit you targeted, the feel goal, any phase-based downgrades, and how the cycle addresses the athlete's needs across the three intensities.",
  "annual_phase_used": "{annual_phase}",
  "cycle": [
    {{
      "plyo_level": "P0",
      "label": "Recovery day",
      "session_intent": "1-2 sentences summarizing the day's goal",
      "drills": [
        {{
          "exercise_name": "exact name from P0 candidate pool",
          "ball_weight": "32 oz",
          "sets": 1,
          "reps": 6,
          "order": 0,
          "rationale": "1-2 sentences tying this drill to the athlete's inefficiency / feel goal"
        }}
      ]
    }},
    {{ "plyo_level": "P1", "label": "Hybrid day", "session_intent": "...", "drills": [...] }},
    {{ "plyo_level": "P2", "label": "Work day",  "session_intent": "...", "drills": [...] }}
  ],
  "weekly_layout": {{
    "MONDAY":    "P0",
    "TUESDAY":   "P1",
    "WEDNESDAY": "P1",
    "THURSDAY":  "P0",
    "FRIDAY":    "P1",
    "SATURDAY":  "P1",
    "SUNDAY":    "P0"
  }}
}}

============================
WEEKLY CADENCE — IRON-CLAD RULES
============================
The cadence must respect these non-negotiable rules:

1. **NEVER P2 the day after ANY game, pen, or P2 plyo session.** Day after
   any high-intent throwing = P0. No exceptions.
2. **The GAME itself IS the P2 event.** For in-season athletes, the plyo
   block on game day is LIGHT (P1) — you don't stack a P2 plyo block on top
   of an actual game. The game supersedes.
3. **ONLY ONE high-intent day per week.** For in-season: the game (P1 plyo
   block since the game IS P2). For off-season: one dedicated pen day (that
   day gets P2 plyo, day after gets P0).
4. **Thursday defaults to P0** (mid-week structural rest / hard reset before
   Friday's pre-game ramp).

**Cadence selection — check {annual_phase}:**

If the annual_phase contains "Off Season" / "Off-Season" / "Offseason" or
similar → use the OFF-SEASON cadence with a mid-week P2 pen. Otherwise use
the IN-SEASON cadence (no P2 in the week — the game is the P2 event).

**IN-SEASON cadence (game_day = {game_day}):**

  Sun P0 | Mon P0 | Tue P1 | Wed P1 | Thu **P0** | Fri P1 | Sat **P1**

Note Saturday is P1 (light plyo — game is the P2). Note Thursday is P0.

Shifted for non-Saturday game days:
  - game_day        → P1   (LIGHT — game is the P2 event)
  - (game_day + 1)  → P0   MANDATORY post-game recovery
  - (game_day + 2)  → P0   continued recovery
  - (game_day + 3)  → P1   begin rebuild
  - (game_day + 4)  → P1
  - (game_day + 5)  → P0   mid-week structural rest
  - (game_day + 6)  → P1   pre-game ramp

**OFF-SEASON cadence (mid-week pen, no game):**

  Sun P0 | Mon P0 | Tue P1 | Wed **P2** (pen) | Thu **P0** | Fri P1 | Sat P1

The Wed=P2 is the pen day; Thu=P0 enforces post-throw recovery.

For relievers, anchor to the most recent appearance — same post-throw P0
rule applies. Relievers may have multiple appearances per week, each is
its own high-intent event followed by P0.

Pick exercise_name strings ONLY from the candidate pool of that specific level.
Don't reuse a drill across levels unless it appears in both pools.
"""

    result = generate(
        system_prompt=skill_prose,
        user_content=user_msg,
        response_mime_type="application/json",
        response_schema=PlyoOutput,
        # Plyo: 3 sessions × ~7 drills each × ~80 tokens per drill + weekly
        # layout + per-session intent + overall reasoning. 16k is comfortable.
        max_output_tokens=16384,
    )
    metadata["raw_response"] = result["text"]
    metadata["cost_usd"] = result.get("cost_usd", 0.0)
    metadata["latency_ms"] = result.get("latency_ms", 0)

    raw_text = result["text"]
    try:
        selection = json.loads(raw_text)
    except json.JSONDecodeError:
        cleaned = _salvage_truncated_json(raw_text)
        if cleaned:
            try:
                selection = json.loads(cleaned)
                print("[plyo] WARNING: JSON was truncated, salvaged via last-brace trim")
            except json.JSONDecodeError:
                print(f"[plyo] WARNING: LLM returned non-JSON, skipping. "
                      f"First 300 chars:\n{raw_text[:300]}")
                return None, metadata
        else:
            print(f"[plyo] WARNING: LLM returned non-JSON, skipping. "
                  f"First 300 chars:\n{raw_text[:300]}")
            return None, metadata

    # Validate each session's drills against its candidate pool, dedupe reverse
    # throws, and force reverse throws to order=0.
    def _is_reverse_throw(name: str | None) -> bool:
        if not name:
            return False
        n = name.lower()
        return "reverse throw" in n  # catches "Reverse Throw" and "Throw to Reverse Throw"

    def _post_process_drills(drills: list[dict]) -> list[dict]:
        """Keep only one reverse-throw drill per session (coaches sometimes use
        both 'Reverse Throw' AND 'Throw to Reverse Throw' across the cycle, but
        within a single session only one belongs). Use weight-rank as tiebreak
        — heaviest ball first since the reverse throw opens the session and
        the heavy variant has a stronger priming effect. Place at order=0,
        renumber rest."""
        if not drills:
            return drills
        rev = [d for d in drills if _is_reverse_throw(d.get("exercise_name"))]
        non_rev = [d for d in drills if not _is_reverse_throw(d.get("exercise_name"))]
        def _weight_rank(d):
            w = (d.get("ball_weight") or "").lower().split()
            try:
                return int(w[0])
            except (IndexError, ValueError):
                return 0
        chosen_rev = max(rev, key=_weight_rank) if rev else None
        ordered = ([chosen_rev] if chosen_rev else []) + non_rev
        for i, d in enumerate(ordered):
            d["order"] = i
        return ordered

    cycle_in = selection.get("cycle") or []
    cycle_out: list[dict] = []
    total_dropped = 0
    for session in cycle_in:
        level = session.get("plyo_level")
        pool = candidate_pools.get(level, [])
        valid_names = {c["exercise_name"] for c in pool}
        drills = session.get("drills") or []
        kept = [d for d in drills if d.get("exercise_name") in valid_names]
        dropped = len(drills) - len(kept)
        if dropped:
            total_dropped += dropped
            print(f"[plyo] {level}: dropped {dropped} hallucinated drill(s)")
        # Apply ordering rules (reverse throws first + dedupe)
        n_rev_before = sum(1 for d in kept if _is_reverse_throw(d.get("exercise_name")))
        kept = _post_process_drills(kept)
        n_rev_after = sum(1 for d in kept if _is_reverse_throw(d.get("exercise_name")))
        if n_rev_before > n_rev_after:
            print(f"[plyo] {level}: deduped {n_rev_before - n_rev_after} duplicate reverse throw(s)")
        cycle_out.append({
            "plyo_level": level,
            "label": session.get("label"),
            "session_intent": session.get("session_intent"),
            "n_drills": len(kept),
            "drills": kept,
        })

    if total_dropped:
        print(f"[plyo] total {total_dropped} drills dropped (not in candidate pool)")

    return {
        "category": "plyo",
        "annual_phase": selection.get("annual_phase_used") or annual_phase,
        "athlete_role": athlete_role,
        "reasoning": selection.get("reasoning"),
        "cycle": cycle_out,
        "weekly_layout": selection.get("weekly_layout") or {},
    }, metadata


# ────────────────────────── Markdown summary (for review) ───────────────────

def _build_weekly_structure_section(payload: dict) -> list[str]:
    """Explicit "how to apply this program across a week" block so the lift
    template days + the program-level routines (prep, bp, me) + the plyo
    session all combine into a clear weekly framework."""
    out: list[str] = []
    out.append("## Weekly Structure — how to apply this program\n")
    out.append("This recommendation is a **weekly framework**. It can be applied for "
               "1-2 mesocycles (4-8 weeks) before advancing to the next progression.\n")
    out.append("**Daily routines** (do every training day, in this order):")
    daily = []
    if payload.get("prep_program"):           daily.append("Prep")  # includes ME-style movement work
    if payload.get("bulletproofing_program"): daily.append("Bulletproofing")
    out.append(f"  {' → '.join(daily) if daily else '(none generated)'}\n")
    lp = payload.get("lift_program") or {}
    templates = lp.get("templates") or []
    if templates:
        out.append("**Day-specific work** (one of these lift templates per day, "
                   "+ a plyo session on 2-4 days/week for pitchers):\n")
        out.append("| Day | Lift template | Movement focus | Plyo? |")
        out.append("|---|---|---|---|")
        plyo_days = {"Legs", "Sprint", "Jump", "Total Body"}  # heuristic: throw on lower-half days
        has_plyo = bool(payload.get("plyo_program"))
        for tpl in templates:
            mb = tpl.get("movement_bucket", "?")
            plyo_cell = "✓" if (has_plyo and mb in plyo_days) else "—"
            out.append(f"| _next available_ | `{tpl.get('template_id')}` | {mb} | {plyo_cell} |")
        out.append("")
    if payload.get("hitting_program"):
        out.append("**Hitting** is the on-field work — performed after lift on hitting-prescribed days.\n")
    out.append("**Progression**: when the athlete completes this mesocycle, advance the "
               "lift template's mesocycle digit (e.g. `5227-11` → `5227-21`). "
               "Prep/BP/ME stay roughly the same with small dose progressions. "
               "Plyo session structure stays the same with ball-weight progression.\n")
    return out


def render_markdown(payload: dict) -> str:
    """Render the recommendation as a coach-readable markdown."""
    out: list[str] = []
    out.append(f"# Weekly program — {payload.get('athlete_name')}")
    out.append("")
    out.append(f"- Recommended program id: `{payload.get('recommended_program_id')}`")
    out.append(f"- Age group: **{payload.get('age_group')}**")
    out.append(f"- Focus: **{payload.get('focus')}**")
    out.append(f"- Role: **{payload.get('role')}**")
    out.append(f"- Generation cost: ${payload.get('generation_cost_usd', 0):.4f} "
               f"(latency {payload.get('generation_latency_ms', 0)} ms)")
    out.append("")
    out.extend(_build_weekly_structure_section(payload))
    out.append(f"## Reasoning\n\n{payload.get('reasoning_text') or '(none)'}\n")
    if payload.get("similar_athletes"):
        # The similar_athletes list mixes two shapes:
        # 1. Named-athlete neighbors (have family_id + named_athlete): used for lift template matching
        # 2. General similar athletes (have athlete_uuid + name): used for prep / future components
        named = [n for n in payload["similar_athletes"] if n.get("named_athlete")]
        general = [n for n in payload["similar_athletes"] if not n.get("named_athlete")]

        if named:
            out.append("## Named-athlete neighbors (drove lift template selection)\n")
            for n in named:
                out.append(
                    f"- **{n['named_athlete']}** — family {n.get('family_id', '?')} "
                    f"\"{n.get('description', '')}\" (distance: {n.get('distance')})"
                )
            out.append("")

        if general:
            out.append("## General similar athletes (drove prep + future components)\n")
            for n in general[:8]:
                out.append(
                    f"- **{n.get('name', n.get('athlete_uuid', '?'))}** "
                    f"(distance: {n.get('distance')})"
                )
            out.append("")
    # ── History-driven blocks rendered in session order ──
    def _render_history_block(title: str, key: str) -> None:
        comp = payload.get(key)
        if not comp:
            return
        out.append(f"## {title}\n")
        if comp.get("reasoning"):
            out.append(f"> {comp['reasoning']}\n")
        out.append("| # | Slot | Exercise | Dose | Rationale |")
        out.append("|---|---|---|---|---|")
        for i, ex in enumerate(comp.get("exercises") or [], 1):
            sets = ex.get("sets")
            reps = ex.get("reps")
            dur = ex.get("duration_seconds")
            if dur:
                dose = f"{sets or 1} × {dur}s"
            else:
                dose = f"{sets or '?'} × {reps or '?'}"
            out.append(
                f"| {i} | {ex.get('slot', '?')} | {ex.get('exercise_name')} | "
                f"{dose} | {ex.get('rationale', '')} |"
            )
        out.append("")

    # Session order: prep (includes movement enhancement) → bulletproofing → plyo → lift → hitting
    _render_history_block("Prep block",                "prep_program")
    _render_history_block("Bulletproofing",            "bulletproofing_program")

    # Plyo: 3-day cycle (P0/P1/P2) plus a weekly layout showing which day runs which level.
    plyo = payload.get("plyo_program")
    if plyo:
        phase = plyo.get("annual_phase", "?")
        role = plyo.get("athlete_role", "?")
        out.append(f"## Plyo / Throwing — 3-day cycle · {role} · phase: {phase}\n")
        if plyo.get("reasoning"):
            out.append(f"> {plyo['reasoning']}\n")

        # Weekly layout table — which days use which level
        layout = plyo.get("weekly_layout") or {}
        if layout:
            out.append("### Suggested weekly layout\n")
            out.append("| Day | Plyo level |")
            out.append("|---|---|")
            for day in ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY",
                         "FRIDAY", "SATURDAY", "SUNDAY"]:
                out.append(f"| {day.title()} | {layout.get(day, '—')} |")
            out.append("")

        # Each plyo session in the cycle
        for session in plyo.get("cycle") or []:
            level = session.get("plyo_level", "?")
            label = session.get("label", "")
            out.append(f"### {level} — {label}\n")
            if session.get("session_intent"):
                out.append(f"*Session intent:* {session['session_intent']}\n")
            drills = session.get("drills") or []
            if not drills:
                out.append("_(no drills selected — candidate pool was empty)_\n")
                continue
            out.append("| # | Drill | Ball weight | Sets × Reps | Rationale |")
            out.append("|---|---|---|---|---|")
            for d in drills:
                order = d.get("order")
                idx = (order + 1) if isinstance(order, int) else "—"
                sets = d.get("sets") or "?"
                reps = d.get("reps") or "?"
                weight = d.get("ball_weight") or "—"
                out.append(
                    f"| {idx} | {d.get('exercise_name')} | {weight} | "
                    f"{sets} × {reps} | {d.get('rationale', '')} |"
                )
            out.append("")

    out.append("## Lift program\n")
    lp = payload.get("lift_program") or {}
    for tpl in (lp.get("templates") or []):
        out.append(f"### {tpl.get('movement_bucket')} — `{tpl['template_id']}`")
        if tpl.get("description"):
            out.append(f"*{tpl['description']}*")
        if tpl.get("rationale"):
            out.append(f"\n> {tpl['rationale']}\n")
        out.append("")
        out.append("| # | Exercise | Sets × Reps |")
        out.append("|---|---|---|")
        for ex in (tpl.get("exercises") or []):
            out.append(f"| {ex['order'] + 1} | {ex['name']} | {ex['sets_x_reps']} |")
        out.append("")

    # Pitcher-specific mobility cap (Dead Hang / Shoulder CAR / Wrist Ext-Flex /
    # Bretzel 2.0 / etc.) — runs alongside every lift day for pitchers.
    mob_cap = lp.get("mobility_cap")
    if mob_cap:
        out.append(f"### Lift mobility cap (every lift day) — pitcher-specific\n")
        if mob_cap.get("description"):
            out.append(f"*{mob_cap['description']}*\n")
        out.append("| # | Exercise | Sets × Reps |")
        out.append("|---|---|---|")
        for i, ex in enumerate(mob_cap.get("exercises") or []):
            out.append(f"| {i + 1} | {ex['name']} | {ex['sets_x_reps']} |")
        out.append("")

    # Hitting goes at the end (it's typically the on-field work that follows lift/plyo)
    _render_history_block("Hitting", "hitting_program")

    return "\n".join(out)


def save_markdown(payload: dict) -> Path:
    """Save the markdown summary to outputs/."""
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    name_slug = (payload.get("athlete_name") or "athlete").replace(" ", "_")
    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"program_{name_slug}_{payload['focus']}_{ts}.md"
    path.write_text(render_markdown(payload), encoding="utf-8")
    return path
