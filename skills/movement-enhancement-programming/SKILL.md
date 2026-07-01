---
name: movement-enhancement-programming
description: Select "movement enhancement" exercises — the low-volume mobility/movement-quality work programmed alongside the main lift block to address an athlete's specific movement restrictions. Use this skill when an athlete shows mobility deficits beyond what prep addresses, when programming corrective movement work, when picking exercises to lock in a movement pattern (e.g., hip mobility while squatting), or when an athlete needs targeted movement quality drills. Trigger on any mention of movement enhancement, ME, corrective work, mobility drills, movement quality work, or pattern-specific mobility.
---

# Movement Enhancement Programming Skill (8ctane)

Movement Enhancement (ME) is the **mobility-with-purpose block** that addresses specific movement restrictions the athlete walks in with. Unlike prep (transient warmup) or bulletproofing (structural joint health), ME is *corrective* — it's the work that fixes the things that limit performance and increase injury risk.

Typical structure: 3-6 exercises, 1-3 sets each. Programmed 3-5 days/week.

## What you'll receive

- Athlete profile per assessment domain, especially:
  - Mobility ROM measurements (degrees-based, post-2026 measurements)
  - 1-3 scale legacy mobility scores (passive ROM tests, hip mobility, T-spine)
  - Hip flexor / Thomas test scores
  - T-spine rotation, hip IR/ER, ankle dorsiflexion
- Candidate exercises with prevalence + typical dose

## Slot structure

Movement enhancement typically covers these in priority order based on the athlete's deficits:

1. **Hip mobility / dissociation** (0-2 exercises) — hip airplanes, 90/90 transitions, world's greatest stretch variants
2. **T-spine / thoracic** (0-1 exercises) — open books, t-spine rotations, foam roller work
3. **Ankle / lower-leg mobility** (0-1 exercises) — ankle rocking, soleus stretches
4. **Postural / position-specific work** (1-2 exercises) — quadruped breathing, pelvic-tilt work, anti-rotation holds
5. **Athletic position / patterns** (0-1 exercises) — split-stance reaches, banded patterns

Lower volume than prep — pick 3-6 total, weighted toward the athlete's actual deficits.

## Selection logic

- **Top deficit pattern dictates priority**: a tight Thomas test → hip flexor priority; poor T-spine rotation → T-spine work first
- **Bilateral asymmetries** (left vs. right ROM gaps) → unilateral exercises on the weaker side
- **Pair with the day's main lift**: if it's a leg-focused day, weight ME toward hip mobility; upper-focused day → shoulder/T-spine
- **Low ankle dorsiflexion** flags a chronic issue; include ankle work consistently

## Output format

Same JSON shape used across components:

```json
{
  "reasoning": "1-3 sentences",
  "exercises": [
    {
      "exercise_name": "exact name from candidate pool",
      "slot": "hip | t-spine | ankle | postural | athletic pattern",
      "sets": 2,
      "reps": 8,
      "duration_seconds": null,
      "rationale": "1 sentence"
    }
  ]
}
```

For holds, use `reps: null` and `duration_seconds: 30`.

## Coaching framing

ME is the place to be selective — fewer exercises chosen well beats a long list. Three exercises that genuinely address an athlete's biggest movement limitations are more valuable than six generic mobility drills. When deficits are minor (Z-scores between -0.5 and 0), keep the ME block small (3 exercises). When deficits are significant (Z < -1 on multiple metrics), use up to 6.
