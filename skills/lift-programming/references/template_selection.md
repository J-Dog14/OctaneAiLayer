# Template Selection — Profile → Template Matching

This file is the playbook for going from "here's an athlete" to "here are the
specific template IDs they should be on." Read this whenever the recommender
or coach asks for a program recommendation.

## Step 1 — Build the 3-digit prefix

Translate the athlete's situation into the first three digits of the template
ID. This narrows ~70 families to usually 2-8 candidates.

**1st digit (athlete level)** — Match age + competitive level:

- Youth/middle school first-timer → 1 (Before Baseball Club)
- Club ball, has trained for ≥1 year → 2 (Baseball Club Advanced)
- HS freshman or first-time-in-weight-room HS athlete → 3 (HS Beginner)
- HS sophomore-senior, has training base → 4 (High School)
- College → 5
- Pro / affiliated → 6
- Softball at any level → 7

**2nd digit (focus)** — Match the current training phase:

- Coming off the season, needs to rebuild base → 1 (Strength)
- Has strength foundation, building velocity → 2 (Power)
- Pre-season, sprint/agility-heavy phase → 3 (Speed)
- During competitive season → 4 (In-Season)
- Hypertrophy / mass-gain block → 5 (Hypertrophy)

**3rd digit (sprint days)** — Match training load tolerance and schedule:

- 1 day/week (default for in-season, return-from-injury, or low-volume)
- 2 days/week (standard off-season for most athletes)
- 3 days/week (advanced athletes or speed-emphasis blocks)

A High School athlete in a Power block doing 2 sprint days/week is `422...`,
which narrows the search to families `4221, 4222, 4223, 4224, 4225, 4226, 4227`.

## Step 2 — Distinguish among variants using the index

Once you have a 3-digit prefix, several 4th-digit variants exist. Use these
disambiguators:

### Read the `index` description verbatim.

Each template family has a one-line `description` like:

- `4221` → "Highschool - Power - Sprint, Jump, Lower, Upper, ..."
- `4222` → "Highschool - Power - Lower Body Velo & Rotational"
- `4224` → "Highschool - Power - Rotational and Ground Force"
- `4225` → "Highschool - Power - General"

These are the coach's own one-line summaries of what makes each variant
distinct. **Pick the one whose description matches the athlete's deficit
profile most closely.**

### Use the "made_for_athletes" anchor.

Each template has a `made_for_athletes` list — the original athlete it was
hand-built for. If the new athlete is similar to that original athlete
(same level, similar archetype), the template is a strong candidate.

For example: `4222` was built for Ollie Swartz. If the new athlete's profile
looks similar to Ollie's (or your archetype clustering puts them in the same
cluster as Ollie), `4222` is the natural pick.

### When variants tie, use these defaults

- "General" focus templates are the safest default for a new athlete you
  don't know well.
- "Heavy Compounds" variants are for athletes with proven lifting form and
  no significant mobility limits.
- "Low Volume" or "Poor Mover" variants are for athletes with mobility
  restrictions, return-from-injury, or limited training history.
- "Return from UE Injury" variants only when the index explicitly flags it.

## Step 3 — Match deficit profile to focus modifier

If you have a deficit profile (from `ai_layer.athlete_profiles`), use it to
pick *between* templates within the same family. Examples:

| Athlete deficit (Z-score < -1)       | Bias toward                              |
|--------------------------------------|------------------------------------------|
| Low rotational power (proteus)       | Rotational/Trunk-emphasis variants       |
| Low CMJ / DJ power                   | Power/Jump-emphasis variants             |
| Tight T-spine                        | Mobility-mixed variants                  |
| Tight hip flexors / Thomas test      | Posterior-chain focus templates          |
| Weak shoulder stability MMT          | Templates with bulletproofing companions |
| Low pelvis-trunk separation (pitch)  | Rotational power templates               |

Use the deficit signal **in addition to** the 1st-3rd digit selection, not as
a replacement. The first three digits set the *category*; the deficit informs
the *variant* (4th digit).

## Step 4 — Layer movement-bucket templates into a week

A complete weekly program is a *set* of templates that share the first 4
digits and (usually) the mesocycle digit but differ in the movement bucket.
The 6th digit is the bucket:

```
4222-11  (Legs)
4222-12  (Upper)
4222-13  (Total Body)
4222-14  (Sprint)
4222-15  (Jump)
```

For a complete program, select **5 templates** with matching prefixes that
cover Legs, Upper, Total Body, Sprint, and Jump (per the `references/id_legend.md`
movement-bucket table).

Adjustments by focus:

- **Strength focus** — usually 4 days: Legs, Upper, Total Body, Sprint. Drop Jump.
- **Power focus** — 5 days: full set including Jump.
- **Speed focus** — Same 5 days, but the Sprint and Jump templates carry more weight.
- **In-Season** — 2 or 3 days: usually Total Body + a short Lower or Upper.
- **Hypertrophy** — 4 days: Legs, Upper, Push, Pull (using the 6 and 7 movement buckets).

## Step 5 — Pick the level

For the chosen template family, decide which of Levels 1-4 to start the
athlete at. Default rule:

- New to this template family → **L1 (Base)**
- Has trained the previous mesocycle's template successfully → progress one
  level (see `references/progression_rules.md`)
- Returning from layoff or injury → drop one level from where they ended

## Edge cases

**No template matches exactly.** Pick the closest 3-digit prefix, then within
that, pick the variant whose `made_for_athletes` looks most similar to your
new athlete. Note the imperfect match in your output.

**Hybrid focus (e.g., "Strength-to-Power" block).** Look for templates with
"Strength/Power" in the description (the coach used a hybrid label for these
blocks, e.g., `4115` = "Highschool - Strength/Power - General").

**Special populations.** For "Misc" (level 8) templates, treat them as
one-off cases — they exist for specific rehab or special situations and
should be matched by description only, not by general pattern.

**Multiple equally-good candidates.** When 2-3 templates look equally
appropriate, surface all of them to the human coach with a note explaining
the differences. Don't pick arbitrarily; this is where coach judgment matters
most.
