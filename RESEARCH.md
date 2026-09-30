# Research layer

Deep research on the assessment corpus, and the coach-facing output built on
top of it.

This document covers what changed in v2, why, and how to run it. The short
version: the analysis was already doing sophisticated things, but nothing in it
answered the two questions a coach actually has — *is this number real?* and
*what does it mean?* — so the output was 500 rows of tables nobody read. v2
adds a measurement-noise layer, a plain-language dictionary, and a page that
leads with findings instead of data.

---

## Quick start

```bash
# One-off: create the persistence tables
python -m src.main research apply-migration

# Nightly: recompute group-level statistics once, for everyone
python -m src.main research cohort-refresh

# Per athlete: the coach one-pager
python -m src.main research coach-report "Carson Crider"
python -m src.main research coach-report joey@8ctanebaseball.com --offline

# The squad, sorted by who needs looking at
python -m src.main research roster
python -m src.main research roster --age-group PRO

# Housekeeping
python -m src.main research reliability --gaps   # what needs repeat captures
python -m src.main research metric-coverage      # what needs a coach-facing name

# Tests — no database needed
pytest tests/ -q
```

`--offline` embeds the chart library so a report works on a laptop with no
wifi and survives being emailed around. Costs ~3 MB per file.

---

## The one thing to do this week

Run `python -m src.main research reliability --gaps`.

It prints every measurement that has no repeat-capture data. For those, we can
tell you a number changed but not whether the change is bigger than what that
measurement does on its own from one day to the next — so the verdict falls
back to a **provisional** band built from pitch-to-pitch scatter, which is a
lower bound and therefore leans generous.

**Capturing a handful of athletes twice inside two weeks fixes this
permanently, for every metric those captures cover, in every future report.**
It is the highest-value data-collection change available and it costs one
afternoon. Nothing else in this codebase buys as much.

---

## Why v2

Six things were wrong. Each is now covered by a test.

**1. The within-session table was ranking noise.** It computed a correlation
between every mechanic and velocity across the pitches in one session, sorted
by |r|, and printed the top. With four pitches and ~150 metrics, the largest
|r| you get from *pure noise* is 1.00 — so the top of that table was
guaranteed to be a coin flip. One real report led with r = −0.80, p = 0.20,
n = 4 under the heading "top metrics correlated with velocity this session".

Now: a hard floor of 8 pitches (the section is withheld below it rather than
shown with caveats), a bootstrap interval on every correlation, and the
section states what the top |r| would have been under pure noise for that n
and that many metrics. A correlation only becomes a finding if its interval
excludes zero *and* it beats that scan-wide ceiling.

**2. Z-score deltas mixed athlete change with norm drift.** `refresh_norms`
does DELETE + INSERT with no version stamp, so two profiles built months apart
were scored against different norm tables. Part of every Z-delta was the
population moving.

Now: assessment deltas are computed on **raw values**. The migration adds
`norms_version` to both `assessment_norms` and `athlete_profiles` so the
mixing is at least visible when someone does want Z-deltas.

**3. Carried-forward values were being differenced as if they were new.**
`profiler` takes the latest session at-or-before the as-of date with no
recency limit. A round whose window contained no mobility test still carried
mobility values from whenever they were last measured — and the change table
differenced those against a fresh test, manufacturing change out of nothing.
One real report had a Round 1 with **zero** assessment sources in its window
and a full set of assessment deltas anyway.

Now: a metric only enters a round's delta if its own `source_dates` entry
falls inside that round's window. Everything excluded is listed with the
reason, because a silently shorter table is how this survived.

**4. Cohort percentiles double-counted.** The comparison cohort was one row
per athlete-*session*, so a pitcher with six captures contributed six rows and
bent the scale toward himself. And `min_n_cohort` was 6 — a percentile off six
observations has ~17-point resolution, so "p90" was not a real number.

Now: one row per athlete, the athlete himself excluded from his own cohort,
and below 20 athletes we report an honest rank ("3rd hardest of 9") instead of
inventing a percentile.

**5. Nothing asked whether a change beat the noise.** See the next section.

**6. One deep dive hit the warehouse ~14 times.** The full trial table was
pulled for the athlete context, again per stratum, again for the outlier
cohort, and again inside `correlate_kinematic_to_assessments` for every
flagged metric — each of those also re-pulling the whole profile matrix.

Now: one pull per process, strata derived in pandas. The test asserts ≤ 2
queries per report.

---

## The reliability layer

`src/research/reliability.py`. This is the centre of gravity of v2.

For every metric it estimates:

- **SEM** — typical error, how much a repeated measurement of the same athlete
  moves when nothing has changed
- **ICC** — the share of the total spread that is real difference between
  athletes rather than measurement scatter. Below 0.5, a ranking on that
  metric is mostly noise, and those metrics are excluded from coach-facing
  findings
- **MDC** — minimal detectable change, the size a change has to clear before
  it can be distinguished from measurement error

Every delta anywhere in the system is then classified:

| Verdict | Meaning | Fires on unchanged data |
|---|---|---|
| **Real change** | cleared the 95% band | ~5% |
| **Probably real** | cleared the 80% band | ~20% |
| **Inside the noise** | it did not move | — |
| **Can't tell yet** | no estimate for this metric | — |

Only *Real change* drives a finding. The false-positive rates are asserted in
`test_verdict_false_positive_rate_is_calibrated` — if they drift, every green
chip on every coach page becomes a lie, so it is worth keeping that test green.

### The trap, and why it matters

There are **two** error bands and using the wrong one is how a change table
fills with false positives.

Comparing two means from the *same* capture is limited only by pitch-to-pitch
scatter, so the band shrinks as you throw more pitches:

```
MDC(k1, k2) = 1.96 * SEM_within * sqrt(1/k1 + 1/k2)
```

Comparing the same athlete on *two different days* — which is what every
round-to-round and roster comparison actually is — is a different question.
Marker placement, recalibration and day-to-day biology are shared by every
pitch in a session, so they do not average out. Forty pitches pins down
exactly where the athlete was that day; it says nothing about whether the day
was typical. The correct band is:

```
MDC_between = 1.96 * SEM_between * sqrt(2)
```

and `SEM_between` can only come from repeat captures close enough together
that real adaptation is implausible.

This is not theoretical. On synthetic data with realistic day-to-day drift,
using the shrinking within-session band for a between-session question called
**about 60% of unchanged mechanics "real changes"**. That is the failure mode
`test_null_warehouse_yields_no_findings` now guards against.

Until repeat captures exist, the system falls back to a deliberately
conservative **provisional** band — the single-trial typical error, un-shrunk —
and labels every verdict from it as provisional, on the page and in the
terminal. Set `allow_provisional_band: false` in the config to refuse instead
and report "can't tell yet", which is the right setting once repeat captures
are routine.

---

## The metric dictionary

`src/research/references/metric_display.yaml` plus `metric_display.py`.

Turns `fm_lead_rfd_braking_bw_per_s` into *"Front-leg braking rate — how fast
the front leg gets the brakes on after the foot lands"*, with the unit, the
direction ("is higher better?"), a group, and the coaching cue it maps to.

Three tiers:

- **curated** — written by a human. Only these may appear on a coach page with
  plain language.
- **auto** — composed from the vocabulary tables for the ~600-key Visual3D
  long tail ("Pelvis rotation speed at ball release — axis Z"). Readable, but
  nobody signed off, so it stays in the analyst appendix.
- **raw** — nothing matched; the key is cleaned up cosmetically.

`research metric-coverage` prints the naming backlog ranked by how much data
actually carries each key, so the work happens in impact order.

**Direction matters more than it looks.** A metric marked `context` (foot
strike to release time) is never coloured good or bad, and a metric marked
`load` (elbow varus torque) is never described as improving or declining — it
*increased*. Telling a coach a pitcher is "weak at foot-strike-to-release
time" is the kind of thing that costs you the room.

`axis_labels` in the YAML is deliberately empty: Visual3D axis meanings are
model- and segment-dependent and guessing them would put confident wrong
statements in front of coaches. Fill them in as you confirm them.

---

## The output

### Coach one-pager — `research coach-report <athlete>`

Inverts the pyramid:

1. **Banner** — the one-line verdict on velocity, with the noise band
2. **Findings** — grouped into *what changed*, *what's holding him back*,
   *what he does well*, *worth watching*, *where he is unusual*. Each is a
   sentence with a confidence chip and, where we have one, a coaching cue.
3. **Three charts** — velocity by session with the noise band shaded; a
   "did it actually move?" bar chart expressing every change in multiples of
   its own error band so metrics in different units sit on one axis; a cohort
   strip showing every comparable athlete as a dot with this one marked
4. **The evidence** — the numbers behind each sentence
5. **What this report could not tell you** — stated plainly, not left to be
   inferred from a missing table
6. **Analyst appendix** — everything else, collapsed

Findings are built in one place (`build_headlines`) with explicit admission
rules, so "what we decided to tell the coach" is auditable rather than spread
across a renderer.

### Roster — `research roster`

Every athlete, one row, one column per KPI, coloured only when the change
cleared that metric's noise band *and* the metric has an agreed better
direction. Sparkline per athlete, sorted by an attention score that is
deliberately simple enough to explain: three points per real decline, four for
a real velocity drop, 1.5 for a capture gap over 120 days.

This is the page for "the pros are back, who do I look at today".

---

## Persistence

`sql/002_research_findings.sql` adds:

| Table | What |
|---|---|
| `ai_layer.metric_reliability` | the noise floor per metric |
| `ai_layer.cohort_findings` | group-level statistics, computed nightly |
| `ai_layer.research_findings` | structured per-athlete findings as JSONB |
| `norms_version` columns | on `assessment_norms` and `athlete_profiles` |

The point: a rendered HTML file is a *view* of a finding, never the finding
itself. With findings in a table you can answer what you told a coach last
month, diff this report against the previous one
(`findings_store.diff_against_previous`), and trace any claim to the config
that produced it.

Everything degrades gracefully — if the migration has not been applied, the
store logs once and becomes a no-op rather than taking down a report run.

---

## Config

`src/research/config.py` — every threshold in one frozen dataclass with a
hash. The hash is stamped into every report header and every persisted
finding, so when two reports disagree you can tell whether the thresholds
moved or the data did.

Override for one run:

```python
strict = CONFIG.replace(min_trials_within_session=12)
```

Or point `RESEARCH_CONFIG_PATH` at a YAML file with any subset of the fields.

---

## Tests

```bash
pytest tests/ -q      # 37 tests, ~8s, no database
```

`tests/fixtures.py` builds a synthetic warehouse with the same three-level
structure as the real one (athlete level + session drift + trial noise) and
patches it in at the loader boundary, so the tests exercise all of the real
analysis code and only the I/O is fake.

The two that matter:

- **`test_null_warehouse_yields_no_findings`** — pure noise in, nothing out,
  across several seeds. An analysis that invents findings from random data is
  worse than no analysis, because the fake findings look exactly like the real
  ones.
- **`test_planted_effect_is_recovered`** — the complement. Having proved it
  says nothing when there is nothing, prove it still says something when there
  is.

Run the null test before trusting any change to the statistics.

---

## Module map

```
src/research/
  config.py            every threshold, one hash
  metric_display.py    warehouse key -> coach language
  references/
    metric_display.yaml   the dictionary itself
  reliability.py       SEM / ICC / MDC, change verdicts
  stats_support.py     bootstrap CIs, the max-|r| null, shrinkage,
                       percentile-or-rank
  loaders.py           cached warehouse access, stratum resolution
  athlete_deep_dive.py the analysis
  coach_report.py      the coach one-pager
  roster.py            the squad view
  render_kit.py        shared HTML shell, palette, chart helpers
  cohort_job.py        the nightly group-level refresh
  findings_store.py    persistence, with graceful degradation

  correlations.py      (unchanged) cross-metric correlation + FDR
  pitching_deep.py     (unchanged) trial-level loaders, FE analyses
  profile_matrix.py    (unchanged) wide athlete x metric matrix
  assessment_rounds.py (unchanged) 3D-anchored round grouping
  kinematic_drivers.py (unchanged) mechanics -> assessments
  clustering.py        (unchanged)
  longitudinal.py      (unchanged)
  program_response.py  (unchanged)
  session_change.py    (unchanged)
  reports.py           the older analyst renderers; the deep-dive one is now
                       a shim onto coach_report
```

---

## Known gaps

Honest list of what is still missing, in rough priority order.

1. **Test-retest data.** See the top of this document. Everything downstream
   gets sharper the day this exists.
2. **Mixed-effects models.** `correlate_velocity_within_athlete` residualises
   by subtracting athlete means and then correlates. That is a reasonable
   fixed-effects approximation, but the standard errors it implies are wrong
   because the residualisation consumed degrees of freedom. `statsmodels
   MixedLM` with athlete and session random intercepts would give correct
   inference for roughly 40 lines. This is the one genuine "more advanced
   method" worth doing.
3. **Session fixed effects.** The within-athlete analysis pools trials across
   sessions, so between-session mechanical change loads as within-session
   variation. Adding session fixed effects would separate "what varies pitch
   to pitch" from "what changed over time"; `correlate_velocity_session_level`
   already answers the second question separately.
4. **A composite index.** A small weighted score over the handful of mechanics
   that actually predict velocity in a stratum, tracked over time, turns 200
   tests into one line — and it is far easier to show a coach than 200
   correlations.
5. **Maturation adjustment.** A round-to-round comparison spanning a year on a
   high-school athlete contains growth as well as training. Currently we warn;
   we do not adjust.
6. **Axis labels** in the metric dictionary.
7. **Report-level multiple comparisons.** Each section FDR-corrects within its
   own family. Nothing controls the error rate across a whole report. The
   practical mitigation is the small pre-registered finding set on page one;
   the exploratory scans live behind `<details>` and are labelled as such.
