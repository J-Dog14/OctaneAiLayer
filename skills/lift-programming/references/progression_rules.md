# Progression Rules — L1 → L2 → L3 → L4

Every template family has 4 progression levels. The skill of programming is
deciding **when** to move an athlete up a level, **when** to hold them at the
current level, and **when** to drop them back. This file is the playbook.

## The principle

Each level uses the same movement *pattern* (push, pull, squat, hinge,
rotation, jump, carry) but a more advanced *variant*. Progressing isn't
adding load — it's changing the exercise to one that demands more skill,
stability, or motor control while continuing to develop the same quality.

For example, a hinge slot might progress:

| Level | Exercise | What's getting harder |
|---|---|---|
| L1 | Single-Leg RDL | Unilateral balance under bodyweight |
| L2 | Kickstand RDL | Slight bilateral, but still mostly single-leg loaded |
| L3 | Barbell RDL | Bilateral with full bar load |
| L4 | Single-Leg Barbell RDL | Unilateral *plus* bar load — combines both |

The progression isn't always linear in difficulty — sometimes L3 is a
strength-dense version and L4 returns to a unilateral skill variant. Read
the catalog, don't assume.

## Default cadence

The default rhythm is **one mesocycle per level**. A mesocycle is 4-6 weeks.

- Week 1-2 of a mesocycle: athlete is learning / acclimating to the new variant
- Week 3-4: athlete is performing at the level with consistent form
- Week 5-6: optional consolidation; or move on

After completing a mesocycle, advance one level *if all criteria below are met*.

## Advancement criteria (L_N → L_N+1)

Move the athlete up only when **all of these are true**:

1. **Form consistency.** They can execute every exercise at the current level
   with clean form for all prescribed sets. No technical regressions across
   the last 2 weeks.

2. **Set/rep completion.** They've completed the prescribed sets×reps for at
   least 80% of sessions at the current level. Missing one set out of every
   five is fine; missing whole sessions is not.

3. **Subjective readiness.** The athlete reports the current level feels
   "manageable" or "easy" — not still challenging at the end of a 4-6 week
   block.

4. **No active mobility/recovery issues.** Significant soreness or stiffness
   that's lasting >48 hours, or any pain flag, means hold or drop. Don't
   advance over a pain signal.

5. **Coach sign-off.** The recommender flags eligibility, but a human coach
   confirms the move. Never auto-advance someone in a draft program without
   surfacing it for review.

## Hold criteria (stay at L_N)

Hold the athlete at the current level when:

- They've completed the mesocycle but one or two specific exercises still
  show form issues. Hold to lock in those exercises.
- They're in-season and you want to minimize new motor-learning load.
- They're coming back from a layoff and need to re-establish the level
  before progressing.
- Mobility scores are blocking specific patterns (e.g., poor ankle mobility
  → hold at L2 squat variants longer).

## Demotion criteria (L_N → L_N-1)

Drop down a level when:

- Athlete returns from injury, layoff > 4 weeks, or surgical recovery
- Athlete fails to complete 50%+ of sessions at the current level
- Any movement pattern shows a new pain signal at the current variant
- The athlete recently changed their training context substantially (e.g.,
  moved from off-season to in-season; transferred from JUCO to D1)

When demoting, drop **one** level — not all the way to L1 — unless the
recovery context is severe.

## Special case: progressing within a season

In-season templates (focus digit = 4) typically *don't* progress through the
4-level system. They cycle in mesocycle-1 of the in-season family
(e.g. `4421-11`, `4421-12`, ...) and hold there. The progression system in
this skill primarily applies to off-season blocks.

## Programmatic progression check

Pseudocode for the recommender to decide a level:

```
def choose_level(athlete, template_family):
    history = athlete.lift_history_for(template_family)
    if history.last_level is None:
        return 1                          # First time on this family
    if history.last_completed_pct < 0.5:
        return max(1, history.last_level - 1)  # Demote
    if history.last_completed_pct >= 0.8 and athlete.weeks_on_level >= 4:
        if athlete.has_any_pain_flag():
            return history.last_level     # Hold
        if athlete.coach_confirmed_ready:
            return min(4, history.last_level + 1)  # Advance
    return history.last_level             # Hold by default
```

This is intentionally conservative — the system biases toward holding rather
than over-promoting. A coach who wants faster advancement will override; a
coach who wants more conservative pacing won't.

## When you don't have a level history

If you can't see an athlete's previous level (e.g., new athlete, no
`ai_layer.program_exercise_prescriptions` history), default to **L1** and
note the unknown-history flag in the program output. This lets the human
coach decide whether to bump them up based on observed performance during
the first session.
