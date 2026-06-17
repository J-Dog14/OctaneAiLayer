# 8ctane AI Layer

R&D pipeline that turns athlete assessment data + programming history into LLM-generated
analyses, stored in the `ai_layer` schema of the backend Neon DB.

## What this repo does

1. **Nightly snapshot** of relevant App DB tables into `app_db_snapshot.*` on the backend DB.
2. **Per-athlete pipelines** (athletic screen, pitching biomech, etc.) that pull data,
   apply a skill (system prompt) via Gemini, and write the result to `ai_layer.generated_reports`.
3. **Cost + version logging** so every call is auditable.

## Layout

```
src/
  db.py                  # backend + app db connection helpers
  gemini_client.py       # Gemini wrapper, logs every call to ai_layer.llm_call_log
  prompt_loader.py       # loads a skill from skills/, registers version in ai_layer.prompt_versions
  sync_app_db.py         # snapshot App DB tables into app_db_snapshot.*
  main.py                # CLI entry point (click)
  pipelines/
    athletic_screen.py   # one athlete + one session -> generated report
    biomech_pitching.py  # ditto for pitching biomech
skills/                  # your existing skill folders (biomech-pitching-breakdown, athletic-screen-analysis)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # mac/linux
# .venv\Scripts\activate    # windows
pip install -r requirements.txt
cp .env.example .env        # then fill in real values
```

## Usage

```bash
# Snapshot App DB into backend.app_db_snapshot.* (run once, then nightly via cron)
python -m src.main sync

# Snapshot a single table (for testing)
python -m src.main sync --tables User

# Generate an athletic-screen analysis for one athlete/session
python -m src.main screen <athlete_uuid> 2026-06-15

# Generate a pitching biomech breakdown
python -m src.main biomech <athlete_uuid> 2026-06-15
```

## Where outputs go

Every report lands in `ai_layer.generated_reports`. Every API call (cost, tokens, latency)
lands in `ai_layer.llm_call_log`. Skill prompt history lives in `ai_layer.prompt_versions`.

## Safety

- All writes happen on the backend DB only. The App DB connection is opened in read-only mode.
- Every Gemini call checks a daily cost cap before running (`DAILY_COST_CAP_USD` in `.env`).
- Snapshot uses atomic staging-table swap so partial-sync states are never visible to queries.
