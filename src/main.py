"""
Single CLI entry point.

Examples:
    python -m src.main sync
    python -m src.main sync --tables User --tables Program
    python -m src.main screen 8d3e... 2026-06-15
    python -m src.main biomech 8d3e... 2026-06-15
    python -m src.main norms
    python -m src.main profile 8d3e... 2026-06-15
    python -m src.main show-profile 8d3e... 2026-06-15
    python -m src.main cost
"""
from __future__ import annotations

import json

import click

from src.backfill_profiles import run_backfill
from src.correlation_report import generate_report
from src.db import backend_conn, query
from src.pipelines import athletic_screen, biomech_pitching
from src.profiler import build_and_save, build_profile, _json_default
from src.program_summarizer import summarize_all, summarize_one_and_save
from src.refresh_norms import refresh_all
from src.sync_app_db import run_full_sync


@click.group()
def cli() -> None:
    """8ctane AI Layer CLI."""


@cli.command()
@click.option("--tables", "-t", multiple=True,
              help="Specific tables to sync (repeatable). Default: all configured tables.")
def sync(tables: tuple[str, ...]) -> None:
    """Snapshot App DB tables into app_db_snapshot.*."""
    run_full_sync(list(tables) if tables else None)


@cli.command()
@click.argument("athlete_uuid")
@click.argument("session_date")
def screen(athlete_uuid: str, session_date: str) -> None:
    """Generate an athletic-screen analysis for one athlete + session date."""
    rid = athletic_screen.run(athlete_uuid, session_date)
    click.echo(f"Generated report id: {rid}")


@cli.command()
@click.argument("athlete_uuid")
@click.argument("session_date")
def biomech(athlete_uuid: str, session_date: str) -> None:
    """Generate a pitching biomechanics breakdown for one athlete + session date."""
    rid = biomech_pitching.run(athlete_uuid, session_date)
    click.echo(f"Generated report id: {rid}")


@cli.command()
@click.option("--keys", "-k", multiple=True,
              help="Specific metric keys to refresh (repeatable). Default: all.")
def norms(keys: tuple[str, ...]) -> None:
    """Recompute per-age-group norms in ai_layer.assessment_norms."""
    refresh_all(list(keys) if keys else None)


@cli.command()
@click.argument("athlete_uuid")
@click.argument("as_of_date")
def profile(athlete_uuid: str, as_of_date: str) -> None:
    """Build & persist a deficit profile to ai_layer.athlete_profiles."""
    pid, p = build_and_save(athlete_uuid, as_of_date)
    click.echo(f"Saved profile id: {pid} (role={p['role']}, age_group={p['age_group']})")


@cli.command()
@click.option("--latest-only", is_flag=True,
              help="Generate one profile per athlete (latest assessment date), "
                   "instead of one per (athlete, session_date).")
def backfill(latest_only: bool) -> None:
    """Bulk-generate ai_layer.athlete_profiles for the whole population."""
    run_backfill(latest_only=latest_only)


@cli.command("summarize-program")
@click.argument("program_id", type=int)
def summarize_program(program_id: int) -> None:
    """Summarize one program (by App DB id) and persist to ai_layer.program_summaries."""
    sid, s = summarize_one_and_save(program_id)
    click.echo(f"Saved summary id: {sid}")
    click.echo(json.dumps(s, default=_json_default, indent=2)[:3000])


@cli.command("summarize-all")
def summarize_all_cmd() -> None:
    """Walk every program in the snapshot and persist a summary row for each."""
    summarize_all()


@cli.command()
@click.option("--k", default=5, type=int, help="Number of archetype clusters.")
@click.option("--role", type=click.Choice(["pitcher", "hitter", "all"], case_sensitive=False),
              default="all", help="Filter to pitchers, hitters, or all athletes.")
@click.option("--domain", type=click.Choice(["movement", "functional", "all"], case_sensitive=False),
              default="all",
              help="movement = pitch/hit kinematics only. "
                   "functional = mobility + proteus + screen only.")
@click.option("--output", default=None, help="Output HTML path (defaults to outputs/...).")
def report(k: int, role: str, domain: str, output: str | None) -> None:
    """Generate a population correlation + archetype report (HTML).

    Recommended sweep:
      python -m src.main report --role pitcher --domain movement
      python -m src.main report --role pitcher --domain functional
      python -m src.main report --role hitter  --domain movement
      python -m src.main report --role hitter  --domain functional
    """
    r = None if role == "all" else role
    d = None if domain == "all" else domain
    path = generate_report(k=k, role=r, domain=d, output=output)
    click.echo(f"Report written: {path}")


@cli.command("show-profile")
@click.argument("athlete_uuid")
@click.argument("as_of_date")
@click.option("--only", type=click.Choice(["raw", "z"]), default=None,
              help="Show only raw values or only z-scores (default: both).")
def show_profile(athlete_uuid: str, as_of_date: str, only: str | None) -> None:
    """Build a profile and pretty-print it WITHOUT saving."""
    p = build_profile(athlete_uuid, as_of_date)
    click.echo(f"Athlete: {athlete_uuid}  role={p['role']}  age_group={p['age_group']}")
    click.echo(f"As of:   {as_of_date}\n")
    click.echo("Source dates per modality:")
    for mod, d in p["source_dates"].items():
        click.echo(f"  {mod:<24} {d or '—'}")
    click.echo("")
    raw = p["raw_values"]
    zs = p["z_scores"]
    click.echo(f"{'METRIC':<55} {'RAW':>12} {'Z':>8}")
    click.echo("-" * 77)
    for k in raw:
        rv = raw[k]
        zv = zs[k]
        rv_s = f"{rv:.3f}" if isinstance(rv, (int, float)) else "—"
        zv_s = f"{zv:+.2f}" if isinstance(zv, (int, float)) else "—"
        if only == "raw":
            click.echo(f"{k:<55} {rv_s:>12}")
        elif only == "z":
            click.echo(f"{k:<55} {zv_s:>8}")
        else:
            click.echo(f"{k:<55} {rv_s:>12} {zv_s:>8}")


@cli.command()
def cost() -> None:
    """Print rolling 24-hour Gemini spend and call count."""
    with backend_conn() as conn:
        rows = query(conn, """
            SELECT model_name,
                   COUNT(*) AS calls,
                   COALESCE(SUM(input_tokens), 0) AS in_toks,
                   COALESCE(SUM(output_tokens), 0) AS out_toks,
                   ROUND(COALESCE(SUM(cost_usd), 0)::numeric, 4) AS spend_usd
            FROM ai_layer.llm_call_log
            WHERE created_at > now() - interval '1 day'
            GROUP BY model_name
            ORDER BY spend_usd DESC
        """)
    if not rows:
        click.echo("No Gemini calls in the last 24h.")
        return
    click.echo(f"{'MODEL':<25} {'CALLS':>6} {'IN_TOKS':>10} {'OUT_TOKS':>10} {'$':>10}")
    for r in rows:
        click.echo(f"{r['model_name']:<25} {r['calls']:>6} {r['in_toks']:>10} "
                   f"{r['out_toks']:>10} {r['spend_usd']:>10}")


if __name__ == "__main__":
    cli()
