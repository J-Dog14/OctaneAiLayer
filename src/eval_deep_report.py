"""
Deep eval report — qualitative cross-eval analysis.

Reads ai_layer.eval_runs rows in a given id range and synthesizes a single
markdown report that goes beyond per-athlete numbers. The headline addition
over `eval-summary` is **movement-pattern-clustered intent recall**: when our
'Trapbar Deadlift' meets the coach's 'Hex Bar Deadlift', they both classify as
'hip-hinge / posterior chain' and count as a match. That answers the question
'how aligned is the *intent* of our programs?' which name-only Recall misses.

Other sections:
  - Per-component pattern overlap (lift/plyo/prep/bp)
  - Systematic gaps (what we always miss, ranked by prevalence)
  - Systematic over-prescription (what we always add)
  - Day-by-day plyo cadence aggregate (where coaches differ from our defaults)
  - Lift template family selection summary
  - Concrete recommendations ranked by impact

CLI:
    python -m src.main eval-deep-report                  # all evals
    python -m src.main eval-deep-report --from 31 --to 43  # specific range
    python -m src.main eval-deep-report --last 13         # last N
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from src.db import backend_conn, query


# ──────────────────────────────────────────────────────────────────────────
# Movement-pattern classifier
# ──────────────────────────────────────────────────────────────────────────
# Each pattern is a list of substring keywords. Classification is first-match
# on a lowercased, normalized name. Order matters — narrower patterns first.
# This is the heart of "intent overlap" — Trapbar Deadlift and Hex Bar
# Deadlift both classify as hip-hinge / posterior chain even though their
# names don't match.

_MOVEMENT_PATTERNS: list[tuple[str, list[str]]] = [
    # ── Plyo-throwing patterns (more specific first)
    ("reverse throw",                ["reverse throw"]),
    ("pivot drill",                  ["og pivot", "pivot v", "hip rotation pivot"]),
    ("saddle drill",                 ["saddle"]),
    ("stomp drill",                  ["stomp throw", "2 stomp", "two stomp"]),
    ("walk-in / windup",             ["walk in", "walkin", "windup", "stride out", "walking in"]),
    ("step away / step back",        ["step away", "stepaway", "step back", "stepback", "rhythm step"]),
    ("drop step",                    ["drop step", "drop in step"]),
    ("happy feet / rhythm",          ["happy feet", "tippy", "ten toes", "rhythm"]),
    ("janitor",                      ["janitor"]),
    ("figure 8",                     ["figure 8", "figure eight"]),
    ("ball-weight throw",            ["plyo +", "ctp throw", "darvish", "p90", "ride the slope"]),
    ("turn & burn / drop",           ["turn and burn", "turn & burn", "drop step", "drop ins"]),
    ("toss / partner",               ["partner", "toss up"]),

    # ── Lift / strength patterns
    ("hip-hinge / posterior chain",  ["rdl", "deadlift", "good morning", "hip thrust", "kb swing",
                                       "glute bridge", "hip hinge", "back extension", "ghd back"]),
    ("knee-dominant / squat",        ["squat", "lunge", "step up", "split stance squat",
                                       "split squat", "rfe", "bulgarian"]),
    ("horizontal push",              ["bench press", "push up", "pushup", "chest press",
                                       "db press", "landmine press"]),
    ("vertical push / overhead",     ["overhead press", "shoulder press", "military press",
                                       "rotational press"]),
    ("horizontal pull / row",        ["row", "cable row", "db row", "barbell row", "3 point"]),
    ("vertical pull",                ["pull up", "pullup", "chin up", "lat pulldown", "pulldown"]),
    ("med ball / rotational power",  ["med ball", " mb ", "mb chest", "mb stomp", "scoop toss",
                                       "scoop", "shotput", "rotational shotput", "stomp & twist",
                                       "stomp and twist", "windmills", "windmill"]),
    ("waterbag / unpredictable load",["waterbag", "waterbag skenes", "waterbag rotation",
                                       "wb stomp", "wb skenes", "wb saddle", "wb rhythm",
                                       "wb stationary"]),
    ("pvc / indian club / pum",      ["pvc", "indian club", "pum "]),
    ("anti-rotation core",           ["pallof", "anti-rot", "anti rot", "farmers carry",
                                       "single arm carry", "with head turn", "head turn"]),
    ("anti-extension core",          ["plank", "deadbug", "dead bug", "hollow", "ab roll",
                                       "copenhagen"]),
    ("spinal flexion / hip flexion", ["sit up", "crunch", "hanging leg raise", "leg raise",
                                       "v up", "hip flexion series"]),
    ("hip mobility / activation",    ["90/90", "hip airplane", "hip shift", "hip flexor stretch",
                                       "hip flexor iso", "spanish squat", "couch stretch",
                                       "elephant walk", "active pigeon", "band pigeon",
                                       "adductor", "lateral lunge"]),
    ("glute activation",             ["lateral band", "monster walk", "x band",
                                       "kneeling side plank", "single leg hip thrust",
                                       "glute bridge"]),
    ("shoulder mobility",            ["sleeper stretch", "sleeper pail", "sleeper rail",
                                       "t spine", "t-spine", "wall slide", "young stretch",
                                       "thread the needle", "open book", "scorpion",
                                       "horizontal abduction"]),
    ("shoulder cuff / care",         ["er off", "external rotation", "internal rotation",
                                       "incline y", "incline t", "blackburn", "i", "y t w",
                                       "rear delt fly", "rear delt flys", "pec scorpion",
                                       "wall pec stretch"]),
    ("nerve glides / wrist care",    ["radial nerve", "ulnar nerve", "wrist curl", "wrist ext",
                                       "wrist flex", "pro sup", "pronation", "supination",
                                       "wrist flexion", "wrist extension"]),
    ("breathing / postural",         ["alligator breath", "pelvic tilt", "deep squat breathing",
                                       "trx deep squat", "no money", "breathing",
                                       "back to wall"]),

    # ── Speed / Plyometric ground-based
    ("sprint / accel running",       ["sprint", "flying 10", "flying ten", "sled march",
                                       "sled sprint", "build up", "buildup",
                                       "5/10/5", "10 yard", "shuttle", "wall sprint"]),
    ("acceleration jump",            ["broad jump", "vertical jump", "rapid vertical",
                                       "box jump", "reactive box", "drop to broad",
                                       "lateral heiden", "split stance jump"]),
    ("plyometric coordination",      ["skip", "high knee", "butt kick", "carioca",
                                       "shuffle", "pop pop", "hop", "pogo", "scissor"]),

    # ── Lower-leg / forearm small muscle
    ("calf / shin / ankle",          ["calf", "tib raise", "ankle 2", "ankle 3"]),
    ("forearm / grip",               ["dead hang", "deadhang", "wrist roller", "forearm"]),

    # ── Mobility / recovery / soft tissue
    ("soft tissue / recovery",       ["foam roll", "soft tissue", "trap stretch",
                                       "treadmill walk", "phone sit", "body blade"]),
    ("J-band / arm care",            ["j band", "j-band"]),
    ("bretzel / hip mobility flow",  ["bretzel", "stick hip flexor"]),
]


def _classify(name: str | None) -> str:
    """Return the movement-pattern label for an exercise name."""
    if not name:
        return "unclassified"
    n = name.lower().strip()
    # Collapse spacing + punctuation for stable substring checks
    n = re.sub(r"\s+", " ", n)
    for label, keywords in _MOVEMENT_PATTERNS:
        for kw in keywords:
            if kw in n:
                return label
    return "unclassified"


# ──────────────────────────────────────────────────────────────────────────
# Eval loading
# ──────────────────────────────────────────────────────────────────────────

def load_evals(from_id: int | None = None, to_id: int | None = None,
               last_n: int | None = None) -> list[dict]:
    """Pull eval_runs rows matching the filter. comparison is decoded JSONB."""
    with backend_conn() as conn:
        if last_n is not None:
            rows = query(conn, """
                SELECT * FROM ai_layer.eval_runs
                ORDER BY id DESC LIMIT %s
            """, [last_n])
            rows.reverse()
            return rows
        clauses = []
        params: list[Any] = []
        if from_id is not None:
            clauses.append("id >= %s")
            params.append(from_id)
        if to_id is not None:
            clauses.append("id <= %s")
            params.append(to_id)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM ai_layer.eval_runs {where} ORDER BY id"
        return query(conn, sql, params)


# ──────────────────────────────────────────────────────────────────────────
# Analysis primitives
# ──────────────────────────────────────────────────────────────────────────

def _iter_component_exercises(comparison: dict, component: str,
                              side: str) -> list[str]:
    """Pull every coach or rec exercise name for a component from the
    comparison JSONB. side ∈ {'coach', 'rec'}."""
    out: list[str] = []
    if component not in (comparison.get("by_component") or {}):
        return out
    d = comparison["by_component"][component]
    # When a group_key was used (lift bucket / plyo level), per_group exists
    sources = []
    if d.get("per_group"):
        sources = list(d["per_group"].values())
    else:
        sources = [d]
    for sub in sources:
        if side == "coach":
            # Coach names come from both `intersection` (the coach side) and
            # `rec_missed` (which are coach-picked we didn't pick)
            for ex in sub.get("intersection") or []:
                if ex.get("name"):
                    out.append(ex["name"])
            for name in sub.get("rec_missed") or []:
                # rec_missed entries may be prefixed with "[bucket] " when grouped
                clean = re.sub(r"^\[[^\]]+\]\s*", "", name)
                out.append(clean)
        elif side == "rec":
            # Rec names from `intersection` + `rec_added`
            for ex in sub.get("intersection") or []:
                rn = ex.get("rec_name") or ex.get("name")
                if rn:
                    out.append(rn)
            for name in sub.get("rec_added") or []:
                clean = re.sub(r"^\[[^\]]+\]\s*", "", name)
                out.append(clean)
    return out


def _pattern_recall_per_eval(evals: list[dict], component: str
                              ) -> tuple[float | None, list[float]]:
    """For each eval, compute pattern-clustered recall for one component.

    Returns (avg, per_eval_list). None when no signal across evals.
    """
    rates: list[float] = []
    for e in evals:
        comparison = e["comparison"]
        coach_names = _iter_component_exercises(comparison, component, "coach")
        rec_names = _iter_component_exercises(comparison, component, "rec")
        coach_patterns = Counter(_classify(n) for n in coach_names)
        # Strip "unclassified" — too noisy to count toward recall
        coach_patterns.pop("unclassified", None)
        rec_patterns = set(_classify(n) for n in rec_names)
        if not coach_patterns:
            continue
        # Recall = % of *unique patterns* coach used that we also touched
        hit = sum(1 for p in coach_patterns if p in rec_patterns)
        rate = hit / len(coach_patterns)
        rates.append(rate)
    if not rates:
        return None, []
    return sum(rates) / len(rates), rates


def _aggregate_missed_by_pattern(evals: list[dict], component: str
                                  ) -> list[tuple[str, int]]:
    """Top patterns coaches use that we miss, across all evals.

    A pattern "miss" happens when a coach picks an exercise of that pattern
    and we don't pick ANY exercise of that pattern.
    """
    misses: Counter = Counter()
    for e in evals:
        coach_names = _iter_component_exercises(e["comparison"], component, "coach")
        rec_names = _iter_component_exercises(e["comparison"], component, "rec")
        rec_patterns = set(_classify(n) for n in rec_names)
        for cn in coach_names:
            p = _classify(cn)
            if p == "unclassified":
                continue
            if p not in rec_patterns:
                misses[p] += 1
    return sorted(misses.items(), key=lambda x: x[1], reverse=True)


def _aggregate_overprescribed_by_pattern(evals: list[dict], component: str
                                          ) -> list[tuple[str, int]]:
    """Top patterns we use that coaches DON'T use, across all evals."""
    over: Counter = Counter()
    for e in evals:
        coach_names = _iter_component_exercises(e["comparison"], component, "coach")
        rec_names = _iter_component_exercises(e["comparison"], component, "rec")
        coach_patterns = set(_classify(n) for n in coach_names)
        for rn in rec_names:
            p = _classify(rn)
            if p == "unclassified":
                continue
            if p not in coach_patterns:
                over[p] += 1
    return sorted(over.items(), key=lambda x: x[1], reverse=True)


def _top_missed_exercises(evals: list[dict], component: str,
                          limit: int = 15) -> list[tuple[str, int]]:
    """Specific exercise names coaches pick that we don't, ranked by count."""
    counts: Counter = Counter()
    for e in evals:
        comparison = e["comparison"]
        if component not in (comparison.get("by_component") or {}):
            continue
        d = comparison["by_component"][component]
        sources = list(d.get("per_group", {}).values()) if d.get("per_group") else [d]
        for sub in sources:
            for name in sub.get("rec_missed") or []:
                clean = re.sub(r"^\[[^\]]+\]\s*", "", name)
                counts[clean] += 1
    return counts.most_common(limit)


def _top_added_exercises(evals: list[dict], component: str,
                         limit: int = 15) -> list[tuple[str, int]]:
    counts: Counter = Counter()
    for e in evals:
        comparison = e["comparison"]
        if component not in (comparison.get("by_component") or {}):
            continue
        d = comparison["by_component"][component]
        sources = list(d.get("per_group", {}).values()) if d.get("per_group") else [d]
        for sub in sources:
            for name in sub.get("rec_added") or []:
                clean = re.sub(r"^\[[^\]]+\]\s*", "", name)
                counts[clean] += 1
    return counts.most_common(limit)


def _plyo_cadence_aggregate(evals: list[dict]) -> dict[str, dict[str, Counter]]:
    """Aggregate plyo cadence: per day, what plyo level did coach assign vs rec?

    Returns:
      {
        'MONDAY':   {'coach': Counter({'P0': 3, 'P2': 1, ...}),
                     'rec':   Counter({'P2': 4, 'P0': 0})},
        'TUESDAY':  {...},
        ...
      }
    """
    days = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY",
            "FRIDAY", "SATURDAY", "SUNDAY"]
    agg = {d: {"coach": Counter(), "rec": Counter()} for d in days}
    for e in evals:
        cadence = (e["comparison"].get("plyo_cadence") or {}).get("days") or []
        for row in cadence:
            d = row.get("day")
            if d not in agg:
                continue
            c = row.get("coach")
            r = row.get("rec")
            if c:
                agg[d]["coach"][c] += 1
            if r:
                agg[d]["rec"][r] += 1
    return agg


def _family_selection(evals: list[dict]) -> Counter:
    fams: Counter = Counter()
    for e in evals:
        ff = (e["comparison"].get("lift_template_family") or {}).get("rec_families") or []
        for f in ff:
            fams[f] += 1
    return fams


# ──────────────────────────────────────────────────────────────────────────
# Markdown rendering
# ──────────────────────────────────────────────────────────────────────────

def _pct(v: float | None) -> str:
    return f"{v:.1%}" if v is not None else "—"


def _section(title: str) -> str:
    return f"\n## {title}\n\n"


def _component_breakdown(comp: str, evals: list[dict]) -> str:
    """Build a per-component breakdown block."""
    title_map = {"lift": "Lift", "plyo": "Plyo", "prep": "Prep", "bp": "Bulletproofing"}
    out = []
    out.append(_section(f"{title_map.get(comp, comp.title())} — pattern analysis"))

    # Per-eval name-based recall + pattern recall
    name_recalls = []
    pattern_recall_avg, pattern_per_eval = _pattern_recall_per_eval(evals, comp)
    for e in evals:
        d = e["comparison"]["by_component"].get(comp) or {}
        if d.get("recall") is not None:
            name_recalls.append(float(d["recall"]))
    avg_name = sum(name_recalls) / len(name_recalls) if name_recalls else None

    out.append(f"- **Name-based recall** (avg): {_pct(avg_name)}\n")
    out.append(f"- **Pattern-clustered intent recall** (avg): "
               f"**{_pct(pattern_recall_avg)}**")
    if pattern_recall_avg is not None and avg_name is not None:
        lift = pattern_recall_avg - avg_name
        out.append(f" *(+{lift:.1%} over name-based — intent overlap exceeds name overlap)*")
    out.append("\n\n")

    # Patterns we miss
    missed = _aggregate_missed_by_pattern(evals, comp)[:10]
    if missed:
        out.append(f"### Movement patterns coaches use that we miss ({comp})\n\n")
        out.append("| Pattern | # of evals missed in |\n|---|---:|\n")
        for p, n in missed:
            out.append(f"| {p} | {n} |\n")
        out.append("\n")

    # Patterns we over-prescribe
    over = _aggregate_overprescribed_by_pattern(evals, comp)[:10]
    if over:
        out.append(f"### Movement patterns we prescribe that coaches didn't ({comp})\n\n")
        out.append("| Pattern | # of evals over-prescribed in |\n|---|---:|\n")
        for p, n in over:
            out.append(f"| {p} | {n} |\n")
        out.append("\n")

    # Specific exercise names
    missed_ex = _top_missed_exercises(evals, comp, 10)
    if missed_ex:
        out.append(f"### Specific exercises coaches use most-often that we miss\n\n")
        out.append("| Exercise | # of evals |\n|---|---:|\n")
        for name, n in missed_ex:
            out.append(f"| {name} | {n} |\n")
        out.append("\n")

    added_ex = _top_added_exercises(evals, comp, 10)
    if added_ex:
        out.append(f"### Specific exercises we add most-often that coaches don't pick\n\n")
        out.append("| Exercise | # of evals |\n|---|---:|\n")
        for name, n in added_ex:
            out.append(f"| {name} | {n} |\n")
        out.append("\n")

    return "".join(out)


def _plyo_cadence_section(evals: list[dict]) -> str:
    agg = _plyo_cadence_aggregate(evals)
    out = []
    out.append(_section("Plyo cadence — day-by-day across all evals"))
    out.append("Each cell shows the distribution of plyo levels coaches assigned "
               "on that day vs what we assigned. Wide divergence → our default "
               "weekly layout is misaligned with coach intent.\n\n")
    out.append("| Day | Coach distribution | Rec distribution |\n|---|---|---|\n")
    for day in ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY",
                "FRIDAY", "SATURDAY", "SUNDAY"]:
        coach_str = ", ".join(f"{lvl}: {n}" for lvl, n in
                              sorted(agg[day]["coach"].items())) or "—"
        rec_str = ", ".join(f"{lvl}: {n}" for lvl, n in
                            sorted(agg[day]["rec"].items())) or "—"
        out.append(f"| {day[:3].title()} | {coach_str} | {rec_str} |\n")
    out.append("\n")

    # Identify the most-common coach work-day
    work_day_by_coach: Counter = Counter()
    for day in agg:
        # Work day = P2 assignments
        work_day_by_coach[day] += agg[day]["coach"].get("P2", 0)
    if work_day_by_coach:
        top_coach_work_days = work_day_by_coach.most_common(3)
        out.append("**Coach work-day (P2) preference** — top 3 days coaches assign P2:\n\n")
        for d, n in top_coach_work_days:
            out.append(f"- {d.title()}: {n} P2 assignments\n")
        out.append("\n")
        most_common_work_day = top_coach_work_days[0][0]
        if most_common_work_day != "MONDAY":
            out.append(f"> Our default plyo layout assigns P2 on Monday. Coaches "
                       f"most commonly put P2 on **{most_common_work_day.title()}** "
                       f"— likely because the pen / mound day sits mid-week. "
                       f"This is the single biggest cadence-skill improvement to make.\n\n")

    return "".join(out)


# ──────────────────────────────────────────────────────────────────────────
# Main report generator
# ──────────────────────────────────────────────────────────────────────────

def generate_deep_report(from_id: int | None = None, to_id: int | None = None,
                          last_n: int | None = None) -> Path:
    evals = load_evals(from_id=from_id, to_id=to_id, last_n=last_n)
    if not evals:
        raise ValueError("No eval_runs rows matched the filter. "
                         "Run `eval-batch` first.")

    ids = [e["id"] for e in evals]
    out = []
    out.append(f"# Deep Eval Report — {len(evals)} evals (id {min(ids)}–{max(ids)})\n\n")
    out.append(f"_Generated: {datetime.now().isoformat(timespec='seconds')}_\n\n")
    out.append(f"_Athletes: {', '.join(e['athlete_name'] for e in evals)}_\n\n")

    # ── Executive summary
    out.append(_section("Executive summary"))
    overall = [float(e["overall_overlap_score"] or 0) for e in evals]
    avg_overall = sum(overall) / len(overall)
    out.append(f"- **{len(evals)} athletes**, focuses: "
               f"{', '.join(sorted(set(e['focus_used'] for e in evals)))}\n")
    out.append(f"- **Headline recall (name-based)**: {avg_overall:.1%}\n")

    # Pattern recall per component
    out.append("- **Pattern-clustered intent recall** "
               "(when same movement pattern counts as a match):\n")
    for comp in ["lift", "plyo", "prep", "bp"]:
        avg, _ = _pattern_recall_per_eval(evals, comp)
        out.append(f"  - {comp}: {_pct(avg)}\n")

    # Plyo cadence aggregate
    cadence_matches = []
    for e in evals:
        mr = (e["comparison"].get("plyo_cadence") or {}).get("match_rate")
        if mr is not None:
            cadence_matches.append(float(mr))
    if cadence_matches:
        out.append(f"- **Plyo cadence avg match**: "
                   f"{sum(cadence_matches)/len(cadence_matches):.1%}\n")

    # Family selection
    fams = _family_selection(evals)
    if fams:
        out.append("- **Lift template families chosen**: ")
        out.append(", ".join(f"{f}×{n}" for f, n in fams.most_common(5)))
        out.append("\n")
    out.append("\n")

    # ── Per-component deep dive
    for comp in ["lift", "plyo", "prep", "bp"]:
        out.append(_component_breakdown(comp, evals))

    # ── Plyo cadence
    out.append(_plyo_cadence_section(evals))

    # ── Family selection detail
    out.append(_section("Lift template family selection"))
    out.append("| Family | # of evals | % |\n|---|---:|---:|\n")
    total = sum(fams.values())
    for f, n in fams.most_common():
        out.append(f"| {f} | {n} | {n/total:.0%} |\n")
    out.append("\n")

    # ── Recommendations
    out.append(_section("Concrete recommendations (ranked by signal strength)"))

    # 1. Plyo cadence
    work_day_by_coach: Counter = Counter()
    for day, sides in _plyo_cadence_aggregate(evals).items():
        work_day_by_coach[day] += sides["coach"].get("P2", 0)
    if work_day_by_coach:
        top = work_day_by_coach.most_common(1)[0]
        out.append(f"### 1. Move default plyo work day off Monday\n\n"
                   f"Coaches assign P2 most often on **{top[0].title()}** "
                   f"({top[1]} instances across these evals). Our skill defaults "
                   f"to Mon=P2, which is misaligned with the typical pitcher "
                   f"cadence (pen day mid-week). Fix in "
                   f"`skills/plyo-programming/SKILL.md` weekly layout guidance.\n\n")

    # 2. BP staples for pitchers
    bp_missed = _top_missed_exercises(evals, "bp", 5)
    if bp_missed:
        out.append(f"### 2. Pitcher BP staples are systematically missed\n\n"
                   f"Top 5 missed BP exercises across evals: ")
        out.append(", ".join(f"**{name}** ({n}×)" for name, n in bp_missed[:5]))
        out.append(f". These are arm-care staples coaches prescribe to nearly "
                   f"every pitcher regardless of Z-score deficit signal. "
                   f"Consider a *role-gated* (`has_pitching_data=True`) BP "
                   f"floor rather than a corpus-wide one.\n\n")

    # 3. Top missed prep patterns
    prep_missed_patterns = _aggregate_missed_by_pattern(evals, "prep")[:3]
    if prep_missed_patterns:
        out.append(f"### 3. Prep patterns systematically missed\n\n"
                   f"Top movement patterns we never touch in prep: ")
        out.append(", ".join(f"**{p}** ({n}×)" for p, n in prep_missed_patterns))
        out.append(". Worth checking whether the prep candidate pool contains "
                   "any exercises matching these patterns at all, or whether "
                   "the skill prompt needs explicit slot guidance for them.\n\n")

    # 4. Over-prescribed sprint/jump in lift
    lift_over_patterns = _aggregate_overprescribed_by_pattern(evals, "lift")[:3]
    if lift_over_patterns:
        out.append(f"### 4. Lift patterns we over-prescribe vs coaches\n\n"
                   f"Top patterns we add that coaches don't include: ")
        out.append(", ".join(f"**{p}** ({n}×)" for p, n in lift_over_patterns))
        out.append(". If sprint or jump patterns dominate this list, our default "
                   "of outputting all 5 lift templates (Legs/Upper/Total Body/"
                   "Sprint/Jump) is too broad. Coaches at the position-player "
                   "level often skip Sprint or Jump days entirely.\n\n")

    # 5. Family selection — are we collapsing?
    if fams and len(fams) <= 3:
        out.append(f"### 5. Lift template family selection might be too narrow\n\n"
                   f"We picked only {len(fams)} distinct families across "
                   f"{len(evals)} athletes. This could indicate the candidate-"
                   f"family filter is too aggressive — worth checking whether "
                   f"thinner candidate pools are constraining selection.\n\n")

    # ── Per-eval roll-up table
    out.append(_section("Per-eval roll-up"))
    out.append("| Eval | Athlete | Focus | Name recall | Lift name recall | Plyo recall |\n")
    out.append("|---|---|---|---:|---:|---:|\n")
    for e in evals:
        comp = e["comparison"]["by_component"]
        ovr = float(e["overall_overlap_score"] or 0)
        lr = (comp.get("lift") or {}).get("recall")
        pr = (comp.get("plyo") or {}).get("recall")
        out.append(f"| {e['id']} | {e['athlete_name']} | {e['focus_used']} | "
                   f"{_pct(ovr)} | {_pct(lr)} | {_pct(pr)} |\n")
    out.append("\n")

    # Write to disk
    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    path = out_dir / f"eval_deep_report_{min(ids)}-{max(ids)}_{ts}.md"
    path.write_text("".join(out), encoding="utf-8")
    return path
