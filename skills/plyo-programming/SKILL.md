---
name: plyo-programming
description: Generate athlete-specific daily plyo ball plans for 8ctane Baseball pitchers. Use this skill whenever a coach requests a plyo plan, asks what drills to run today, needs a plyo day assigned, or references a pitcher's plyo level (P0, P1, P2, P3). Also triggers when a coach mentions a pitcher's role (starter/reliever), recent outing, Qualisys 8-cylinder inefficiency flags, feel goals (arm path / lead leg / hip drive / intent), or annual training phase (recovery / rebuild / workload ramp / velocity / velocity transfer / preseason build / in-season / postseason). The skill produces a complete ready-to-use daily plan with drills, sets, reps, ball weights, and rationale that feeds directly into the 8ctane app.
---

# 8ctane Plyo Programming Skill

This skill turns three inputs into a complete daily plyo session that a coach can hand to a pitcher and an athlete can execute:

- **Plyo Day Level** (P0 / P1 / P2 / P3) — determined by role and where the pitcher is in their weekly cadence
- **Movement Inefficiency** — pulled from Qualisys 8-cylinder data and coach observation (which biomechanical gap is the pitcher working to close)
- **Feel Goal** — arm path / hip drive / lead leg / intent — the neuromuscular target the coach is cueing toward

The output is a session with drill selection, dose, ball weights, and the *why* behind each choice. The plyo level dictates volume and intensity guardrails; the inefficiency + feel goal dictate which drills get prioritized.

## When to invoke this skill

Trigger any time the conversation is about programming plyos for one or more days. Specifically:

- Coach asks "what plyo day should X be on today?"
- Coach asks for a plyo plan for a specific date
- An athlete's Qualisys data shows a specific inefficiency and we're picking drills around it
- Pitcher just had an outing or pen and we need their next day's prescription
- Designing a weekly plyo template tied to a training phase
- Auditing a current plyo plan against the system

## Three-step workflow

### Step 1 — Decide the Plyo Day Level

The plyo level encodes intent and recovery posture. It's set by the pitcher's role and where they are in their week. See `references/cadence-rules.md` for the full starter / reliever cadence rules. Quick map:

| Level | Day Type | Intent |
|---|---|---|
| **P0** | Recovery | Day after outing or pen. Low intent, high feel, tissue restoration. Never upgraded just because the athlete "feels good" — the day after any real throwing IS P0. |
| **P1** | Hybrid | Between recovery and work. Moderate intent, still feel-oriented. Most common day for relievers and starters who are 1-2 days from a high-intent day. |
| **P2** | Work | Pen day or higher intent. Full expression within controlled volume. Bullpen days, max-intent throwing days, mound work days. |
| **P3** | Intent Education | Rarely given. Only assigned when the coach has direct judgment that the athlete genuinely doesn't understand what max intent feels like. Never assigned by schedule alone. |

Also gate the level against the athlete's **annual training phase** — `references/annual-throwing-system.md` lists which plyo levels are permitted in each phase. Examples: the Recovery phase post-season permits only +0 work; the Velocity phase permits up to P3. If the day-level the cadence suggests isn't allowed by the phase, drop to the highest allowed level.

### Step 2 — Set the Volume, Reps, and Ball Weight Range

Once you have the plyo level, the volume and weight guardrails are non-negotiable:

| Level | Exercises | Sets | Reps | Ball Weights |
|---|---|---|---|---|
| P0 | 4–5 | 1–2 | 6 | 32 oz – 7 oz |
| P1 | 5–6 | 1–2 | 6–8 | 32 oz – 5 oz |
| P2 | 5–7 | 1–2 | 8–10 | 32 oz – 5 oz (some 3 oz) |
| P3 | 5 | 1 | 6 | 32 oz – 3 oz |

**P3 is volume-capped on purpose** — the goal is intent education, not accumulation. Don't push volume higher just because the athlete is fresh.

### Step 3 — Select the drills (the matrix)

Drill selection is gated equally by **plyo level**, **movement inefficiency**, and **feel goal**:

- The **plyo level** sets what drill intensities are allowable.
- The **inefficiency** narrows which drills are *prioritized* — pull from drills tagged to that biomechanical gap.
- The **feel goal** shapes sequencing and cueing — drills tagged to the same feel category should cluster together in the session and share cueing language.

**Three goals per session** (priority weighting depends on the day's level):

1. **Correct the inefficiency** — target the pitcher's known movement gap (hip-shoulder separation, lead-leg stiffness, horizontal abduction, kinematic sequence timing, etc.)
2. **Create a feel** — build positive neuromuscular feedback in the pattern they're cueing toward
3. **Work what they don't do well (secondary)** — address a secondary weakness while they're feeling good

Weighting by level:

- **P0** — feel and tissue restoration win. Inefficiency correction is minimal. Don't spike effort to chase a correction; the athlete needs recovery more than they need adjustment.
- **P1** — balanced. Touch the inefficiency without overcooking volume.
- **P2 / P3** — inefficiency correction and intent expression can both be pushed. Drill selection can be more aggressive.

**Drill matching rule.** Each drill in the 8ctane system should be tagged with (a) primary inefficiency target, (b) feel category, (c) allowable plyo levels, (d) appropriate ball weights. When picking drills:

- A drill matching **all three** (inefficiency + feel + level) is prioritized
- A drill matching **two of three** is secondary — used to round out the session
- A drill matching **only one** is filler / warmup-only — use sparingly

When tag data isn't available for a candidate drill (e.g. a historical drill name pulled from a database with no metadata), fall back to the drill name itself + coach judgment. Note in session notes when you're operating without complete tag coverage.

## Required output format

Return a session object with this structure:

```json
{
  "session_header": {
    "athlete_name": "...",
    "date": "YYYY-MM-DD or null",
    "role": "Starter | Reliever",
    "plyo_level": "P0 | P1 | P2 | P3",
    "annual_phase": "Recovery | Rebuild Capacity | Workload Ramp | Velocity | Velocity Transfer | Preseason Build | In-Season | Postseason Reset"
  },
  "session_intent": "1–2 sentences summarizing the goal of the day — what we're correcting, what we're trying to feel, what we want the athlete to walk away with",
  "drills": [
    {
      "drill_name": "exact drill name",
      "sets": 1,
      "reps": 8,
      "ball_weights": ["5 oz", "4 oz"],
      "inefficiency_target": "hip-shoulder separation | lead leg stiffness | horizontal abduction | pelvis-trunk amplification | kinematic sequence timing | other",
      "feel_category": "arm path | lead leg | hip drive | intent/effort | other",
      "why": "1 sentence tying this drill to the athlete's inefficiency and/or feel goal"
    }
  ],
  "session_notes": "Optional coach-facing notes — Qualisys flags that drove selection, cues to emphasize during execution, anything that won't fit in the why fields per drill"
}
```

## Weekly cadence — game-day aware

The plyo cadence has to be built around the athlete's actual throwing schedule.
Two iron-clad rules override everything else:

- **NEVER assign P2 the day after a game or pen day.** Connective tissue
  doesn't care if the athlete feels good. If yesterday was a game, today is
  P0 — non-negotiable. Same if yesterday was a P2 pen day.
- **The game day is itself P2.** When an athlete pitches in a game, the
  game IS the high-intent work. Do not add a P2 plyo session on top of it —
  the plyo block on a game day is light (P1 max) or skipped entirely.

### Default weekly cadence (summer-ball / college-fall, Saturday game day)

Most of our pitchers play summer ball or fall scrimmages with **Saturday games**.
**Only ONE P2 day per week** — and it's the game day. The plyo block on game
day exists to extend / support the game's high-intent stimulus. Mid-week is for
rebuilding intent gradually, not stacking a second P2.

Default cadence for a starter (in-season / summer-ball with Saturday games):

| Day | Level | Rationale |
|-----|-------|-----------|
| Sun | **P0** | MANDATORY recovery — day after Saturday game |
| Mon | **P0** | Continued recovery, light tissue work |
| Tue | **P1** | Begin rebuilding intent |
| Wed | **P1** | Continue building |
| Thu | **P0** | Mid-week structural rest — hard reset before pre-game ramp |
| Fri | **P1** | Pre-game ramp, light feel work |
| Sat | **P1** | Game day — plyo block is LIGHT (the game itself is the P2 event) |

Note two important things:

- **The Thursday P0** is consistent across coach data. It's a hard reset before
  Friday's pre-game ramp, even without a mid-week pen. Don't collapse it to P1.
- **Saturday is P1, NOT P2.** When the athlete is playing a game, the GAME is
  the P2 event — you don't stack a P2 plyo block on top. Coach data confirms:
  Saturday mode is P1 (7/13) with P2 only 4/13. The plyo block on game day
  exists to support the game, not add high-intent work.

For **off-season athletes** (no games, e.g., "24-25 Off Season" phase), the
P2 belongs to the mid-week pen day instead:

| Day | Off-season with Wed pen |
|-----|-------------------------|
| Sun | P0 |
| Mon | P0 |
| Tue | P1 |
| Wed | **P2** (pen day) |
| Thu | P0 (MANDATORY post-pen) |
| Fri | P1 |
| Sat | P1 |

Determine which cadence to use by checking `annual_phase`. If the phase name
contains "Off Season" / "Off-Season" / "Offseason", use the off-season cadence
(Wed=P2). Otherwise default to the in-season cadence above (no P2 in the week
— the game supersedes).

If the game day is not Saturday, shift by the same offset relative to game day:
- `game_day`       = P1   (plyo block is LIGHT; game is the P2 event)
- `(game_day + 1)` = P0   MANDATORY post-game recovery
- `(game_day + 2)` = P0   continued recovery
- `(game_day + 3)` = P1   rebuild
- `(game_day + 4)` = P1
- `(game_day + 5)` = P0   mid-week structural rest
- `(game_day + 6)` = P1   pre-game ramp

For **relievers**, cadence is more flexible — they can appear in multiple games
per week. Use the most recent appearance as the anchor and apply the same
post-throw P0 rule (the day after ANY appearance is P0).

If an athlete has a **dedicated mid-week pen day** (off-season or in-house
program with no game), THEN that pen day can be P2 — but then game day (if
any) becomes P1 because you can only have one P2 per week.

### Why the older "Mon = P2" default was wrong

Our earlier default put the work day on Monday. That's misaligned with how
coaches actually structure the week — they cycle around the game day, not the
calendar week. With Saturday games, Monday is still inside the recovery window
(2 days post-game). The P2 work day belongs mid-week, separated from both the
game and the next game by 2-3 recovery days on each side.

## Drill ordering rules

These rules constrain how drills are sequenced within any plyo session.
They reflect how the 8ctane coaches actually structure a plyo day.

- **Reverse throws come FIRST.** Any drill whose name contains "Reverse Throw"
  (or starts with "Throw to Reverse Throw") is the connection/priming drill of
  the session and belongs at order=0. It opens the body and gets the kinetic
  chain talking before higher-intent work.
- **Only ONE reverse throw per session.** A session opens with one reverse-throw
  drill at order=0. Duplicate reverse-throws in the same session are programming
  waste. (Across the three sessions in a P0/P1/P2 cycle, you can use different
  variants on different days — that's fine.)
- **Reverse Throw vs Throw to Reverse Throw** — these are TWO different drills,
  not weight variants. Coaches prescribe both about equally:
  - **Reverse Throw** = the rotational-load drill itself, typically lighter ball
    (5-7oz). Use it when the session intent emphasizes arm path / connection.
  - **Throw to Reverse Throw** = connection-prep drill that walks the body
    into the reverse-throw position, typically heavier (16-32oz). Use it when
    the session intent emphasizes priming / loading before higher intent.
  Pick whichever variant matches the session's feel goal. **Heavier ball =
  more priming/recovery**, lighter ball = more intent/feel. Don't always
  default to the same variant across all three sessions — rotate.
- **Recovery/light-intent drills go last** if any are included. Drills like
  "Walk Throughs" or low-intensity finishers belong at the tail end of the
  session, not the middle.
- **Within the middle of the session**, sequence by intensity ramp: connection
  → rotational → throwing-specific → max-output → cooldown. Different sessions
  emphasize different parts of this ramp depending on the plyo level.

## Edge cases and guardrails

- **Never assign P3 based on schedule alone.** P3 requires an explicit coach decision that the athlete lacks intent understanding. If the inputs say "P3" without that context, drop to P2 and note the override in session notes.
- **P0 is non-negotiable** the day after any outing or pen session. Don't upgrade to P1 because the athlete feels fresh. Connective tissue doesn't tell you it's overworked.
- **Reliever back-to-back appearances** — if a reliever threw on consecutive days, treat the day after the *second* appearance as P0 even if 48 hours have passed since the first.
- **No Qualisys data available** — fall back to coach observation for inefficiency tagging. Note the data gap in session notes so the next coach reading the plan knows the input quality.
- **In-season vs. off-season** — in-season plans should prioritize feel and recovery continuity even at P2. Off-season plans can weight inefficiency correction more heavily and allow more accumulation at P2.
- **Phase-level overrides matter more than day-level.** If the annual phase says "no P2 yet" (e.g., Rebuild Capacity), don't assign P2 even if the cadence would otherwise suggest it. The phase guardrails exist for connective-tissue safety.

## What this skill doesn't do

- Doesn't pick lifts, prep, bulletproofing, or hitting drills — those have their own skills.
- Doesn't decide whether an athlete is ready to train or should be shut down — that's coach judgment with input from athlete feel + workload monitoring.
- Doesn't tell you a pitcher's role (starter vs. reliever) — that's an input you bring in.
- Doesn't generate the throwing program itself (bullpens, long toss, pulldown progressions). It generates the plyo block that supports throwing.

## Reference files

Read these when the situation demands:

- `references/cadence-rules.md` — full starter and reliever weekly cadence rules, plus how to handle multi-day patterns
- `references/annual-throwing-system.md` — the 8-phase annual calendar and which plyo levels are allowed in each phase
