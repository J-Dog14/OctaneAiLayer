---
name: lift-programming
description: Apply the 8ctane strength coach's lift programming system — a curated catalog of ~70 named templates with 4 levels of progression per template, encoded under a 4-7 digit ID scheme. Use this skill whenever building, adapting, or recommending a strength/power/speed lifting program for a baseball or softball athlete, when an athlete's profile (age group, sport-focus, training level, mobility/power archetype) needs to be matched to a coach-built template, when an athlete needs to be progressed to a harder variant of an existing template, when composing a multi-template program (e.g. layering Lower + Upper + Sprint + Core templates into a complete week), or when interpreting a template ID like "4222-15" or "5113-11". Trigger on any mention of programming, templates, progressions, lift selection, sets and reps, or any of the ID prefixes from the catalog (1111, 2121, 3111, 4xxx, 5xxx, 6xxx, 7xxx).
---

# Lift Programming Skill (8ctane)

This skill encodes the 8ctane strength coach's complete lift programming system. The system is built around **named template families** — curated, hand-built progressions that a real coach has assigned to real athletes over years of work. The skill teaches a model how to read the ID code, pick the right templates for a given athlete, choose the right progression level, and layer multiple templates into a complete program.

The skill assumes you (Claude) have access to four sibling files:

- **`references/id_legend.md`** — the full decode table for the 4-7 digit ID scheme. Read this first whenever you encounter an unfamiliar ID.
- **`references/template_selection.md`** — heuristics for matching an athlete's profile (age, focus, archetype, deficits) to template IDs.
- **`references/progression_rules.md`** — when and how to move an athlete from Level 1 → 2 → 3 → 4 within a template family.
- **`references/template_catalog.json`** — the full structured catalog of 340 template blocks across 71 ID families, with every exercise + sets×reps + the athletes each template has historically been assigned to.

## When to invoke this skill

Invoke yourself any time the conversation involves:

- Building, modifying, or critiquing a lift program for an athlete
- Translating an athlete's deficit profile or archetype into a concrete set of templates
- Deciding whether an athlete should advance to the next progression level
- Reading or interpreting a template ID (`4222-11`, `5113-21`, etc.)
- Comparing two athletes' programs against each other through a coaching lens

If the user is asking about general fitness, hypertrophy outside the baseball context, or non-programming topics, this skill probably isn't the right one.

## Conceptual model in 90 seconds

### 1. Every template has a numeric ID that *is* its taxonomy.

A code like `4222-11` decodes as:

| Digit | Meaning | Value here |
|---|---|---|
| 4 | Athlete level | High School |
| 2 | Focus | Power |
| 2 | Sprint days/week | 2 |
| 2 | Identifier (variant) | 2nd version |
| — | (mesocycle separator) | |
| 1 | Mesocycle number | 1 |
| 1 | Movement bucket | Legs |

Full decode tables are in `references/id_legend.md`. Always consult that file rather than guessing.

### 2. Every template comes in 4 progression levels.

The same template ID has Level 1 (base), Level 2, Level 3, Level 4. The *movements within a level* hit the same patterns (push, pull, hinge, squat, rotation, carry, jump) but each level uses a harder or more specific variant. For example, in the 4111-11 (HS Strength Lower) template:

| Level | Hinge slot | Squat slot |
|---|---|---|
| L1 | Single Leg RDL 3×4 | Heels-Elevated Goblet Squat 3×4 |
| L2 | Kickstand RDL 3×4 | Barbell Squat 3×4 |
| L3 | Barbell RDL 3×5 | SSB Box Squat 3×5 |
| L4 | Single Leg Barbell RDL 3×5 | Forward Lunge 3×5 |

The athlete progresses through levels as they demonstrate competency. `references/progression_rules.md` defines the criteria.

### 3. A complete program is a *layered* set of templates.

A real program for a high school power athlete might use:

- `4221-11` Lower (mesocycle 1, legs)
- `4221-12` Upper (mesocycle 1, upper)
- `4221-13` Total Body (mesocycle 1)
- `4221-14` Sprint (mesocycle 1, sprint focus)
- `4221-15` Jump (mesocycle 1, jump focus)

The two-digit suffix `-1X` indicates the movement bucket within the same mesocycle. When the athlete advances mesocycles, the suffix changes to `-2X`, `-3X`, etc.

### 4. The Index records who each template was built for.

The `index` block in `template_catalog.json` includes the original athlete name a template was hand-built for (e.g., `4222` → "Ollie Swartz"). These are gold examples — they tell you what kind of athlete this template was *literally designed for*. Use them as anchors when matching new athletes.

## The catalog file (`references/template_catalog.json`)

Structure:

```jsonc
{
  "n_blocks_parsed": 340,
  "n_unique_template_ids": 71,
  "index": {
    "4222": {
      "template_id_numeric": "4222",
      "description": "Highschool - Power - Lower Body Velo & Rotational",
      "made_for_athletes": ["Ollie Swartz"],
      "decoded": { "level": "High School", "focus": "Power", ... }
    },
    ...
  },
  "blocks": [
    {
      "levels": {
        "L1": {
          "header": { "raw": "4111-11 - ...", "id_numeric": "4111", "mesocycle_id": "11", "description": "..." },
          "decoded_id": { "level": "High School", "focus": "Strength", ... },
          "exercises": [
            { "exercise": "Single Leg Broad to 2 Leg Landing", "sets_x_reps": "3 x 3" },
            ...
          ]
        },
        "L2": { ... }, "L3": { ... }, "L4": { ... }
      },
      "index_metadata": { ... }       // Lookup in `index` if present
    },
    ...
  ],
  "templates_by_id": { "4111": [block, block, ...], ... }
}
```

**How to use it:**

- To find a template by ID: `catalog["templates_by_id"]["4222"]` → list of all the `-11`, `-12`, etc. blocks for that family.
- To look up who a template was built for: `catalog["index"]["4222"]["made_for_athletes"]`.
- To get the actual prescribed exercises at level N: `block["levels"]["L<N>"]["exercises"]` — each entry has `exercise` (name) and `sets_x_reps` (e.g. `"3 x 5"` or `"1 x 60s"`).

## Workflow: how to build a program for a new athlete

When asked to design a lift program for an athlete:

**Step 1 — Decode the athlete's situation.**

Identify:
- *Level digit*: their training-level category (HS Beginner, HS, College, Pro, Softball, etc.)
- *Focus digit*: what the block is for (Strength, Power, Speed, In-Season, Hypertrophy)
- *Sprint days/week*: how much sprint volume they're tolerating
- Any specific deficits or archetypes (rotational power deficit, mobility issue, etc.)

These three digits alone narrow the search to a small handful of template families. Use `references/id_legend.md` to convert verbal context ("he's a high school junior on a power phase, 2 sprint days") into the prefix `422`.

**Step 2 — Find candidate template families.**

Filter `templates_by_id` for IDs starting with the matching prefix. Then read each family's `description` and `made_for_athletes` from the index to pick the most-aligned one. `references/template_selection.md` covers the matching heuristics in detail.

**Step 3 — Pick the right progression level.**

For a brand new athlete, default to L1 (Base). For a returning athlete who's already crushed an earlier level, use the progression rules in `references/progression_rules.md` to decide whether to keep them at the current level, move them up, or down.

**Step 4 — Layer templates to form a complete week.**

A typical week is 5 templates: Lower (`-X1`), Upper (`-X2`), Total Body (`-X3`), Sprint (`-X4`), Jump (`-X5`). All five share the same first 4 digits — that's the family — and the same mesocycle digit (the digit immediately after the hyphen).

If the athlete's focus implies fewer days (e.g., in-season), drop the Sprint or Jump layer and keep the lower-volume ones.

**Step 5 — Output the program.**

Present the prescription as a structured list. For each template, include the family ID, the level you chose, the list of exercises with sets×reps. The recommender that calls this skill will pass that structured output forward.

## Important framing for the LLM

The templates in this catalog are **what a real coach actually built**, not what a textbook would say. Trust the catalog. If the coach prescribed Trampoline Split Squat Pulses for a particular template, don't substitute Barbell Back Squats just because the textbook would. The catalog *is* the coach's expertise.

When you need to deviate (e.g., the athlete can't do a movement due to injury), explain why and suggest a substitution from another template that hits the same slot — but flag clearly that you're substituting.

The `description` field on the index entry is the coach's own one-line summary of what each template is *for*. That's usually the most informative single string for picking a template.

## What this skill *does not* do

- It doesn't prescribe plyos, prep work, bulletproofing, or hitting drills. Those have their own catalogs (plyo prescriptions sit on `PlyoToProgram` + `ThrowingExerciseToPlyo`).
- It doesn't compute athlete deficit scores. That's the deficit-profile pipeline's job (`ai_layer.athlete_profiles`).
- It doesn't choose between templates of equal goodness — that's a judgment call to surface up to the human coach.

When in doubt, **pick the template whose `made_for_athletes` description matches the new athlete most closely**, present your reasoning, and let the human reviewer confirm.
