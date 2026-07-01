---
name: hitting-programming
description: Select hitting drills, bat work, and on-field hitting exercises for baseball or softball athletes. Use this skill when programming the hitting block of a daily training session — tee work, front-toss, live BP, swing mechanics drills, attack-angle work, and rotational hitting power. Trigger when programming for a hitter (an athlete with has_hitting_data=TRUE), when the athlete's hitting 3D data shows mechanical deficits (low hip-shoulder separation, suboptimal bat attack angles, poor lead leg block, etc.), or any mention of hitting drills, bat work, or batting practice.
---

# Hitting Programming Skill (8ctane)

The hitting block is the **on-field swing work** of a hitter's day. It varies more by athlete than any other component because hitting is mechanically expressive — coaches address each hitter's specific swing pattern. Typical structure: 5-10 drills, sets/reps vary widely (often described in rounds, not sets).

## What you'll receive

- Athlete profile per assessment domain, with hitting-specific signal:
  - **Hitting 3D mechanics** (z_hit_*): bat speed, attack angles, hip-shoulder separation, lead-leg block delta, lead-knee mechanics
  - **Proteus rotational power** (z_proteus_hitter_*): shot put, trunk rotation power/velocity
  - **Athletic screen**: rotational power proxy + lower-body force production
- Candidate hitting exercises with prevalence + typical dose from similar hitters

## What good hitting programming addresses

Hitting deficits are usually one of:

- **Bat speed deficits** (low max_bat_ang_vel) → overload/underload bat work, rotational power throws
- **Attack angle issues** (suboptimal horizontal/vertical attack angle at contact) → tee work targeting specific contact points, plane drills
- **Lower-half issues** (poor lead leg block, low knee extension velocity) → block drills, kettlebell hip work, single-leg patterns
- **Sequencing issues** (low hip-shoulder separation) → connection drills, dissociation work
- **Vision/timing** — usually live BP work, often not measurable in our profile

## Selection logic

1. **Start with the dominant deficit.** If bat speed is the biggest deficit, lean toward overload work. If attack angle is the issue, lean toward plane-specific tee work.
2. **Cover the structure**: most hitting blocks have a mix of (a) skill/timing work and (b) mechanics/output work. Don't be all one or the other.
3. **Match the focus block**: power focus → more overload + rotational power throws; in-season → more skill/maintenance work.
4. **Use the prevalence signal heavily**: hitting coaches develop strong individual preferences. If the candidate pool shows 60% of similar hitters got X drill, that's strong signal.

## Output format

Same JSON shape:

```json
{
  "reasoning": "1-3 sentences identifying the hitting deficit prioritized + block structure",
  "exercises": [
    {
      "exercise_name": "exact name from candidate pool",
      "slot": "tee work | front toss | live BP | rotational power | overload | block work | other",
      "sets": 3,
      "reps": 5,
      "duration_seconds": null,
      "rationale": "1 sentence"
    }
  ]
}
```

For exercises measured in rounds (e.g., "3 rounds of 10 swings"), put `sets: 3, reps: 10`.

## Coaching framing

Hitting programming is the *least catalog-able* of the components — coaches make a lot of intuitive decisions about which drill matches which feel. The candidate pool you're given was built by real coaches working with real athletes; respect that signal. If a drill appears commonly in similar athletes' programs, it probably maps to something about their profile that we can't see in metrics yet.
