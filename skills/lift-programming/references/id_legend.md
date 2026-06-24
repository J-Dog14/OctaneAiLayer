# Template ID Legend

The 8ctane template numbering is a fully self-describing code. A typical
template ID looks like `4222-11` or `5113-21`. Read the digits left-to-right;
the dash separates the "family" identifier from the "mesocycle + movement"
identifier.

```
        4    2    2    2    -    1    1
        │    │    │    │         │    │
        │    │    │    │         │    └─ 7th digit: lift # in the day
        │    │    │    │         │           (1 = 1st lift, 2 = 2nd lift)
        │    │    │    │         └────── 6th digit: movement bucket
        │    │    │    └──────────────── 5th digit: mesocycle number
        │    │    └───────────────────── 4th digit: identifier / variant
        │    └────────────────────────── 3rd digit: sprint days per week
        └─────────────────────────────── 2nd digit: focus
        ▲
        └─ 1st digit: athlete level
```

Some IDs are 5 or 7 digits long — that's the same scheme with mesocycle-number
spilling into two digits (e.g. `52210-12`, `6327-131`).

---

## 1st digit — Athlete Level

| Digit | Level                            | Typical athlete                          |
|-------|----------------------------------|------------------------------------------|
| 1     | Before Baseball Club (Beginner)  | Very young / first-time training         |
| 2     | Baseball Club (Advanced)         | Club-team caliber, has training base     |
| 3     | High School (Beginner)           | First-year HS, raw                       |
| 4     | High School                      | Mainstream HS athlete                    |
| 5     | College                          | NCAA / JUCO                              |
| 6     | Pro                              | Affiliated minor league + up             |
| 7     | Softball                         | Any softball athlete                     |
| 8     | Misc                             | Anything else (rehab, special cases)     |

---

## 2nd digit — Focus

| Digit | Focus       | When to use                                                |
|-------|-------------|------------------------------------------------------------|
| 1     | Strength    | Building base strength, off-season starter                 |
| 2     | Power       | Velocity/explosiveness phase, post-strength block          |
| 3     | Speed       | Sprint and acceleration emphasis                           |
| 4     | In-Season   | Maintenance during competitive season                      |
| 5     | Hypertrophy | Size-building block, usually early in off-season           |

---

## 3rd digit — Sprint days per week

| Digit | Sprint days |
|-------|-------------|
| 1     | 1 day/week  |
| 2     | 2 days/week |
| 3     | 3 days/week |

This is a *programming volume* indicator. Higher sprint-day templates pair with
more aggressive lower-body work.

---

## 4th digit — Identifier / Variant

Free-form variant tag. Two templates with the same first three digits but
different 4th digits are *parallel* variants of the same level/focus/sprint
combo (e.g. `4111`, `4112`, `4113`, `4114`, ...). They differ by emphasis,
movement style, or who they were built for. Look at the index `description`
or `made_for_athletes` to disambiguate.

---

## 5th digit — Mesocycle Number

A mesocycle is a 4-6 week training block. Mesocycle 1 is the start of a phase;
mesocycle 2 follows, etc. Templates at the same family `XXXX` come in
mesocycle 1, 2, 3, etc. (the digits after the dash usually start with
the mesocycle).

For multi-digit mesocycles (rare), see IDs like `4111-31` (mesocycle 3, lift 1).

---

## 6th digit — Movement Bucket

| Digit | Bucket      | What's in it                                  |
|-------|-------------|-----------------------------------------------|
| 1     | Legs        | Lower-body focused day                        |
| 2     | Upper       | Upper-body focused day                        |
| 3     | Total Body  | Full-body day                                 |
| 4     | Sprint      | Sprint mechanics + lifts that complement      |
| 5     | Jump        | Jump training + plyo-aligned strength         |
| 6     | Push        | Push-pattern dominant day                     |
| 7     | Pull        | Pull-pattern dominant day                     |
| 8     | Core        | Core stability / anti-rotation day            |
| 9     | Recovery    | Light work, restoration                       |

In a complete weekly program you typically see the 1-5 buckets used together,
sometimes with 6-8 for specialized phases.

---

## 7th digit — Lift # in the day

| Digit | Meaning   |
|-------|-----------|
| 1     | 1st lift  |
| 2     | 2nd lift  |

Used when an athlete does two lifting sessions in a single day. Most templates
omit this — they default to lift #1.

---

## Quick decode examples

| ID         | Decoded                                                           |
|------------|-------------------------------------------------------------------|
| `4111-11`  | HS · Strength · 1 sprint day · variant 1 · meso 1 · Legs · lift 1 |
| `4222-15`  | HS · Power · 2 sprint days · variant 2 · meso 1 · Jump · lift 1   |
| `5113-12`  | College · Strength · 1 sprint day · variant 3 · meso 1 · Upper    |
| `6327-14`  | Pro · Speed · 2 sprint days · variant 7 · meso 1 · Sprint         |
| `7111-13`  | Softball · Strength · 1 sprint day · variant 1 · meso 1 · Total   |

---

## Notes on irregularity

Some headers don't follow the numeric scheme at all — they use a named
prefix like `Baseball Club Total Body - 1 - 1`. Treat the named prefix as the
"family" name; the two trailing numbers are still mesocycle + day-of-week.
Named templates can be found in `template_catalog.json["blocks"]` where
`levels.L1.header.id_numeric` is `null`.
