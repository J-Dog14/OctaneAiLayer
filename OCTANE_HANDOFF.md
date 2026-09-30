# Backend Handoff — Feeding the Octane AI Program Pipeline

The Octane app now runs the full program-generation pipeline itself. This repo's remaining job is **data supply**: profiles, the reference corpus, and (unchanged) the skills. This doc is everything you need to do on this side.

## One-time setup

### 1. Env vars

Add to this repo's `.env`:

```
OCTANE_API_URL=https://<your-octane-host>        # no trailing slash; http://localhost:3000 for local testing
OCTANE_REPORTS_API_KEY=<same value as Octane's REPORTS_API_KEY env>
```

Same auth scheme your existing report sender uses (Bearer token checked by `requireReportsApiAuth`).

### 2. Initial corpus push

The Octane pipeline finds similar athletes and builds candidate exercise pools from the reference population. Push it once:

```bash
python -m src.main send-corpus --dry-run    # sanity: athlete + prescription counts
python -m src.main send-corpus              # one POST per athlete, idempotent
```

What it sends per athlete: latest `ai_layer.athlete_profiles` row (z-scores, age group), role flags (with the f_pitching_trials/f_hitting_trials staleness fallback applied), and all their `ai_layer.program_exercise_prescriptions` rows. Keyed by **warehouse athlete_uuid** — athletes don't need Octane accounts to be corpus members.

### 3. Skills seeding (runs from the Octane repo, reads this one)

```bash
# in the Octane repo:
npx tsx scripts/seed-ai-skills.ts --skills-dir ../OctaneAiLayer/skills
```

## Ongoing workflows

### Per-athlete: send a profile (the generation trigger)

An athlete becomes generatable in Octane the moment their profile lands there. After running the profiler:

```bash
python -m src.main profile <athlete_uuid> <YYYY-MM-DD>     # existing step, unchanged
python -m src.main send-profile <athlete_uuid> [--as-of YYYY-MM-DD] [--dry-run]
```

Requirements for `send-profile` to succeed: the athlete has an email in `analytics.d_athletes`, and that email matches an `app_db_snapshot."User"` row (i.e., they have an Octane account — refresh with `python -m src.main sync` if the snapshot is stale). Octane re-verifies uuid+email server-side before storing.

### When to re-run send-corpus

Re-run after any of: profile backfills (`backfill`), new program summaries (`summarize-all`), norm refreshes that change z-scores, or new athletes with meaningful history. It's a full upsert per athlete — safe to run whenever, `--athlete <uuid>` for one athlete, `--limit N` to test.

### When you edit a skill

Edit `skills/<name>/` here as usual, then re-seed from the Octane repo (command above). Content-hash versioning means unchanged skills are no-ops and every Octane generation records the skill version it used. If the coach updates the lift spreadsheet: regenerate `template_catalog.json` via the skill's parse script, then re-seed — Octane reads the catalog from the seeded skill, so `load-templates` is no longer needed for the Octane side (keep it if you still run local Python evals).

## What this repo still owns vs. what moved

Still owned here: warehouse ingestion pipelines, norms (`refresh_norms`), the profiler and backfills, program summarization into `program_exercise_prescriptions`, skills authoring, and the local eval harness. Moved to Octane: Gemini calls, recommendation, template materialization with 95% dedupe, applying to programs. The Python `recommend` / `compile-payload` commands still work against your warehouse for local testing/evals, but production generation should go through Octane so drafts, dedupe, and audit live in one place.

## New files/commands added to this repo

- `src/octane_sync.py` — `send_profile()` + `send_corpus()` and helpers (identity resolution mirrors `payload_builder.py`)
- `main.py` — `send-profile` and `send-corpus` CLI commands
- Nothing else changed; all existing commands behave as before.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `send-profile` → "No app_db_snapshot.User with email …" | Athlete has no Octane account, or snapshot stale → `python -m src.main sync` |
| `send-profile` → 404 from Octane | UUID+email mismatch on Octane side — check the user's email in the app |
| `send-corpus` skips an athlete | Empty/invalid z_scores map on their profile — re-run the profiler |
| Octane rejects with 400 (zod) | Metric keys must be `[a-z0-9_]+`; non-numeric values are dropped by the sender automatically — check for schema drift if it persists |
| 401 from Octane | `OCTANE_REPORTS_API_KEY` doesn't match Octane's `REPORTS_API_KEY` |
