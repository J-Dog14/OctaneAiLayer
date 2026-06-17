"""
Single CLI entry point.

Examples:
    python -m src.main sync
    python -m src.main sync --tables User --tables Program
    python -m src.main screen 8d3e... 2026-06-15
    python -m src.main biomech 8d3e... 2026-06-15
    python -m src.main cost
"""
from __future__ import annotations

import click

from src.db import backend_conn, query
from src.pipelines import athletic_screen, biomech_pitching
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
