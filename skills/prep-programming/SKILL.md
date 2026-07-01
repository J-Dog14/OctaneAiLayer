---
name: prep-programming
description: Select dynamic warmup / activation / movement-prep exercises for a baseball or softball athlete's daily training session. Use this skill when prescribing pre-lift / pre-throwing prep work, when selecting exercises to address an athlete's mobility deficits, when picking activation drills based on athletic-screen asymmetries, when designing a 6-10 exercise warmup that primes the body for the main session, or when an athlete profile flags specific tightness (Thomas test, T-spine, shoulder ROM, ankle dorsiflexion, etc.) that needs targeted prep. Trigger on any mention of prep, warmup, mobility work, activation, movement prep, or pre-training.
---

# Prep Programming Skill (8ctane)

This skill prescribes the **prep block** that opens an athlete's daily session — typically 6-10 exercises lasting 8-15 minutes, completed before the main lift and throwing/hitting work. The goal isn't conditioning; it's *priming*: get the body ready by addressing the specific mobility, stability, and activation deficits the athlete walked in with.

## Inputs you'll see

When this skill is invoked, you'll receive:

- **The athlete's deficit profile** broken out by assessment domain. Pay closest attention to:
  - **Mobility metrics** (Z-scores below -0.5 are real deficits worth addressing in prep)
  - **Athletic screen asymmetries** (SLV left vs. right power, force-plate asymmetries)
  - **Shoulder stability MMTs** (especially relevant for prep when low)
- **A pool of candidate exercises** that coaches have historically prescribed to athletes with similar profiles. Each candidate includes:
  - Exercise name
  - How many similar athletes received it
  - Typical sets × reps when prescribed
- **Athlete demographics** (age group, focus, role).

## What "good prep" looks like

A complete prep block has roughly these slots, in this order:

1. **General warmup** (1-2 exercises) — raise body temperature, light dynamic movement
2. **Mobility-targeted work** (2-3 exercises) — address the athlete's specific tight areas
3. **Activation** (2-3 exercises) — wake up under-active muscles (glutes, scaps, deep core)
4. **CNS prep / movement integration** (1-2 exercises) — pattern-specific work that primes the upcoming session
5. **Movement Enhancement** (1-3 exercises) — **always last in the prep block.** This is where med balls, waterbags, and other rotational/CNS-prep tools go. ME is part of prep but lives in its own slot at the tail end — after the body is warmed and the rest of prep is done, ME bridges into the session intent (power, rotation, etc.).

Total: **10-15 exercises**. Coach prep blocks for pitchers consistently include
12-20+ exercises because the slots above each need 2-3 entries to actually do
the work. Skimping (e.g., 1 mobility, 1 activation) leaves real deficits
unaddressed. Each at low volume — typically **1-3 sets, 8-15 reps or 20-30s
holds**. Prep should leave the athlete *warmer and looser*, not fatigued.

**Variety across slots matters.** A common failure mode is over-loading one
slot (e.g., 5 mobility exercises) while skipping another (e.g., zero CNS prep).
Aim for: 1-2 warmup + 3-4 mobility + 2-3 activation + 1-2 cns prep + 2-3 ME =
10-15 total.

## Selection logic

For each athlete:

1. **Identify the athlete's top 3-5 mobility/activation deficits** from their Z-scores. These are the slots prep should address with the highest priority.
   - Example: low Z on `mob_thomas_test_hip_flexor_r` → include hip flexor work
   - Example: low Z on `mob_horizontal_abduction` → include T-spine + shoulder mobility
   - Example: SLV asymmetry > 10% → include single-leg activation
2. **Pull from the candidate pool first.** Coaches have prescribed these exercises to similar athletes; that's the strongest signal. Only deviate when the deficit signal is specific and the historical pool doesn't address it.
3. **Cover the slot structure.** Don't pick 8 mobility exercises with no activation. Each slot in the structure above should have at least one exercise.
4. **Match dose to focus.** Speed/Power blocks → more activation, less mobility holds. Strength blocks → more mobility, slightly higher volume on activation.

## Output format

Return strict JSON:

```json
{
  "reasoning": "1-3 sentences explaining the deficits prioritized and the structure of the prep block",
  "exercises": [
    {
      "exercise_name": "exact name from the candidate pool",
      "slot": "general warmup | mobility | activation | cns prep | movement enhancement",
      "sets": 2,
      "reps": 10,
      "duration_seconds": null,
      "rationale": "1 sentence — why for this athlete"
    },
    ...
  ]
}
```

For time-based exercises (planks, holds, breathing drills), set `reps: null` and `duration_seconds: 30`.

**Pick exercise names ONLY from the candidate pool provided.** If a needed slot isn't covered by the pool, omit that slot rather than inventing exercises — note the gap in `reasoning`.

## What this skill does NOT do

- It doesn't prescribe lifts, plyos, bulletproofing, or movement enhancement. Those have their own pipelines.
- It doesn't run a full assessment workup; it consumes the deficit profile produced upstream.
- It doesn't decide whether the athlete is ready to train (no readiness assessment); coach judgment owns that.

## Movement Enhancement — the final slot of prep

Movement Enhancement is the tail end of the prep block. It includes med ball
drills (MB Chest Pass, MB Stomp, MB Scoop Toss, etc.), waterbag drills
(Waterbag Skenes, Waterbag Rotations, Waterbag Saddle Rotation), PVC tools,
Indian clubs, and similar rotational/intent-bridging exercises. **ME exercises
must always be placed AFTER the general warmup / mobility / activation / cns
prep slots — they're the closer of the prep block.**

In the candidate pool, ME exercises will appear alongside other prep exercises
(the underlying data treats them as a single category umbrella). It's your job
to recognize them by name and put them in the `movement enhancement` slot at
the end of the output.

Selection guidance:

- **Power / Speed athletes:** include **2-3 ME exercises** in the ME slot.
  These are the bridge from prep to the rotational power demands of the
  session.
- **Strength / Hypertrophy athletes:** **1-2 ME exercises** is enough —
  they're still doing them, but they're not the centerpiece.
- **In-Season athletes:** keep ME volume conservative (single light set,
  low rep) — these are nervous-system fatiguing tools.

**Vary the ME tool families — and PRIORITIZE med balls over waterbags.**
Coaches use med ball drills ~3× more often than waterbag drills. Common
failure mode of the AI recommender: defaulting to Waterbag Skenes every
time. Don't do this. The four tool families ranked by coach prevalence:

1. **Med balls** (MB Chest Pass, MB Stomp, MB Scoop Toss, MB Kickstand Scoop,
   Seated Rotational Shotput) — **highest priority for rotational power priming.**
   Include at least one med ball drill in the ME slot for any Power / Speed
   athlete unless the candidate pool genuinely has no med ball entries.
2. **PUM (Pre-Use Movement) tools** (PUM Saddle, PUM Tornados, PUM Janitor,
   PUM Figure 8, PUM Hurdle CTP) — coach-built throwing-pattern rehearsal
   drills. **High signal for pitchers** — when [ME] entries with "PUM" in
   the name are in the candidate pool, include 1-2 for pitcher athletes.
3. **PVC / Indian clubs** (PVC Stomp & Twist, PVC Half Kneeling Hip Hinge,
   Indian Club Roll In Throw, Indian Club Rodeo Scap Load) — light cuff /
   wrist / shoulder rhythmic priming. Coaches use these heavily — when
   [ME] entries with "PVC" or "Indian Club" in the name are in the candidate
   pool, prefer them for athletes who need shoulder/wrist priming.
4. **Waterbags** (Waterbag Skenes, Waterbag Saddle Rotation, WB Stomp & Twist)
   — unpredictable-load core control. **Use sparingly** — only for athletes
   who need pelvic-trunk dissociation specifically, and never as the sole
   ME choice. Coaches use waterbags less often than med balls.

If the candidate pool surfaces multiple variants of the same rotational
pattern (e.g. several scoop-toss variants), pick the one with the highest
prevalence among the similar-athlete cohort.

If the candidate pool surfaces multiple variants of the same rotational
pattern (e.g. several scoop-toss variants), pick the one with the highest
prevalence among the similar-athlete cohort, not the most aggressive.

The ME slot is intentional: med ball and waterbag work that lives "inside" the
general activation block gets lost. Placing it last makes its role as a
session-priming closer explicit, and matches how coaches actually structure
the warmup.

## Coaching framing

Prep is the most *individualized* component of the program. Two athletes on the same lift template can have completely different prep blocks because their mobility profiles differ. Lean into that — generic warmups waste training time. Pick prep exercises that *fix* something for *this* athlete.
