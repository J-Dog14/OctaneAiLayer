---
name: bulletproofing-programming
description: Select shoulder and arm "bulletproofing" exercises for a baseball or softball athlete — the low-volume joint-health and stability work that's done daily or near-daily to protect the throwing shoulder, elbow, and surrounding musculature. Use this skill when an athlete profile shows shoulder mobility deficits (low IR/ER ROM, scap stability issues), low shoulder stability MMTs, asymmetries between dominant and non-dominant arms, or when programming arm-care work for pitchers and high-volume hitters. Trigger on any mention of bulletproofing, shoulder health, arm care, prehab, scap work, or rotator cuff work.
---

# Bulletproofing Programming Skill (8ctane)

Bulletproofing is the **joint-health block** that protects the throwing shoulder and elbow. Unlike prep (which is a transient pre-session warmup), bulletproofing is structural: it builds resilience over weeks and months. Typical structure: 4-8 exercises, 1-3 sets each, low-to-moderate volume per exercise. Programmed daily or 4-6 days/week.

## What you'll receive

- Athlete profile per assessment domain, especially:
  - Shoulder ROM (IR / ER / horizontal abduction / total arc)
  - Shoulder stability MMTs (low scores → more bulletproofing needed)
  - Scapular function (mid-trap, low-trap MMT)
  - Pro-sup data if available (forearm endurance / fatigue resistance)
  - Pitching mechanics — high-stress mechanics need more bulletproofing
- Candidate exercises with prevalence + typical dose

## Slot structure

A good bulletproofing block covers these in roughly this priority:

1. **Cuff strength / isolation** (1-2 exercises) — ER off knee, Y/T/W variations, rear delts
2. **Scapular control + stability** (1-2 exercises) — pushup plus, serratus slides, scap pulls
3. **Mid/lower trap activation** (1 exercise) — Incline Y's, prone T's, banded versions
4. **Joint-position holds / isos** (1-2 exercises) — Pushup iso, shoulder iso holds
5. **Forearm / wrist** (optional, 0-1 exercises) — wrist curls, pronator work

Most athletes get 4-6 of these slots, not all. Pick based on deficit signal.

## Selection logic

- **Low rotator cuff metrics** (low IR/ER or MMT scores below average) → prioritize cuff isolation
- **Asymmetry between dom/non-dom shoulder** → include unilateral cuff work
- **Low mid-trap or low-trap MMTs** → add scap activation work
- **Pitchers in Power/Speed phases** → bias volume slightly higher (these phases stress the arm most)
- **Pitchers in In-Season** → keep volume conservative; don't fatigue the arm

## Output format

Same JSON shape used across components:

```json
{
  "reasoning": "1-3 sentences identifying the joint-health priorities for this athlete",
  "exercises": [
    {
      "exercise_name": "exact name from candidate pool",
      "slot": "cuff | scap stability | mid-low trap | iso | forearm",
      "sets": 3,
      "reps": 10,
      "duration_seconds": null,
      "rationale": "1 sentence"
    }
  ]
}
```

For iso/hold exercises, use `reps: null` and `duration_seconds: 30` (or appropriate seconds).

## Coaching framing

Bulletproofing should look *boring* — same handful of exercises repeated week to week, with the dose slowly progressing. The point isn't variety, it's consistency. Resist the urge to make it interesting. The shoulder gets healthy from *thousands* of well-executed cuff reps, not from clever new exercises. Trust the historical pool: if 8 of 10 similar athletes got Incline Y's, that's signal.
