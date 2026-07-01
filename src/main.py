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
from src.load_templates import load_catalog, summary as load_templates_summary
from src.eval_deep_report import generate_deep_report
from src.eval_harness import (
    aggregate_eval_summary,
    link_coach_program_by_name,
    run_eval_for_athlete,
)
from src.payload_builder import compile_payload, save_payload
from src.pipelines import athletic_screen, biomech_pitching
from src.profiler import build_and_save, build_profile, _json_default
from src.program_summarizer import summarize_all, summarize_one_and_save
from src.recommender import (
    get_available_focuses,
    load_athlete_profile,
    recommend_lift_program,
    save_markdown,
)
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


@cli.command("available-focuses")
@click.argument("athlete_uuid")
def available_focuses_cmd(athlete_uuid: str) -> None:
    """Show which training focuses are programmable for this athlete.

    Filters by the loaded templates — focuses with fewer than 3 templates
    available for the athlete's age group are excluded as too thin.
    """
    profile = load_athlete_profile(athlete_uuid)
    ag = profile.get("age_group")
    focuses = get_available_focuses(ag)
    click.echo(f"Athlete: {profile.get('name')} (age_group: {ag})")
    click.echo(f"Available focuses ({len(focuses)}):")
    for f in focuses:
        click.echo(f"  - {f}")
    if not focuses:
        click.echo("  (none — no templates loaded for this age group)")


@cli.command("recommend")
@click.argument("athlete_uuid")
@click.option("--focus", required=True,
              type=click.Choice(["Strength", "Power", "Speed", "In-Season", "Hypertrophy"],
                                case_sensitive=False),
              help="Training focus for this block.")
@click.option("--role", type=click.Choice(["pitcher", "hitter", "both"], case_sensitive=False),
              default=None, help="Override the auto-detected role.")
@click.option("--plyo-day",
              type=click.Choice(["P0", "P1", "P2", "P3", "auto"], case_sensitive=False),
              default="auto",
              help="Plyo day level. 'auto' infers from focus (Power/Speed→P2, In-Season→P1).")
@click.option("--phase", default=None,
              help="Annual throwing phase override (e.g. 'Velocity Phase', 'In-Season Maintenance'). "
                   "Default: inferred from focus.")
@click.option("--athlete-role",
              type=click.Choice(["Starter", "Reliever"], case_sensitive=False),
              default="Starter",
              help="Pitcher role for plyo cadence rules (Starter vs Reliever). "
                   "Only used for the plyo component.")
@click.option("--game-day",
              type=click.Choice(["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY",
                                 "FRIDAY", "SATURDAY", "SUNDAY"],
                                case_sensitive=False),
              default="SATURDAY",
              help="Day of the week this athlete pitches games. Anchors the "
                   "plyo weekly cadence — day after game = P0 always. Default "
                   "SATURDAY for summer-ball / fall scrimmages.")
@click.option("--no-markdown", is_flag=True,
              help="Skip writing the markdown summary to outputs/.")
def recommend_cmd(athlete_uuid: str, focus: str, role: str | None,
                  plyo_day: str, phase: str | None, athlete_role: str,
                  game_day: str, no_markdown: bool) -> None:
    """Generate a draft full weekly program for an athlete + focus."""
    payload = recommend_lift_program(
        athlete_uuid=athlete_uuid,
        focus=focus.title() if focus != "In-Season" else "In-Season",
        role=role,
        plyo_day=None if plyo_day.lower() == "auto" else plyo_day.upper(),
        annual_phase=phase,
        athlete_role=athlete_role.title(),
        game_day=game_day.upper(),
    )
    click.echo(f"\n[recommender] saved id={payload['recommended_program_id']} "
               f"in {payload['total_elapsed_ms']} ms "
               f"(${payload.get('generation_cost_usd', 0):.4f})")
    click.echo(f"[recommender] {len(payload['selected_template_ids'])} templates selected: "
               f"{payload['selected_template_ids']}")
    if not no_markdown:
        path = save_markdown(payload)
        click.echo(f"[recommender] markdown summary: {path}")

    # Auto-compile the App DB payload alongside the markdown
    try:
        compiled = compile_payload(payload["recommended_program_id"])
        payload_path = save_payload(compiled, payload["recommended_program_id"])
        n_unmatched = len(compiled.get("unmatched_exercises") or [])
        click.echo(f"[recommender] App DB payload:    {payload_path}")
        if n_unmatched:
            click.echo(f"[recommender]   ⚠ {n_unmatched} exercise(s) couldn't be matched "
                       f"to Exercise.id — see payload.unmatched_exercises")
    except Exception as e:
        click.echo(f"[recommender] payload compilation failed: {e}")


@cli.command("compile-payload")
@click.argument("recommended_program_id", type=int)
def compile_payload_cmd(recommended_program_id: int) -> None:
    """Compile an existing recommended program into an App DB payload JSON."""
    compiled = compile_payload(recommended_program_id)
    path = save_payload(compiled, recommended_program_id)
    click.echo(f"Payload: {path}")
    n_at = len(compiled.get("activity_templates") or [])
    click.echo(f"  activity templates: {n_at}")

    # Count matches by method across every template
    n_exact = n_loose = n_fuzzy = n_unmatched = 0
    fuzzy_rows: list[tuple[str, str, str]] = []
    for at in compiled.get("activity_templates") or []:
        if at.get("role") == "plyo":
            iters = [(d.get("throwing_exercise_name"),
                      d.get("matched_exercise_name"), d.get("match_method"),
                      d.get("throwing_exercise_id"))
                     for sess in (at.get("sessions") or [])
                     for d in (sess.get("drills") or [])]
        else:
            iters = [(e.get("exercise_name"),
                      e.get("matched_exercise_name"), e.get("match_method"),
                      e.get("exercise_id"))
                     for e in (at.get("exercises") or [])]
        for name, matched, method, ex_id in iters:
            if ex_id is None:
                n_unmatched += 1
            elif method == "exact":
                n_exact += 1
            elif method == "loose":
                n_loose += 1
            elif method and method.startswith("fuzzy"):
                n_fuzzy += 1
                fuzzy_rows.append((name or "", matched or "", method))

    click.echo(f"  match: exact={n_exact} loose={n_loose} fuzzy={n_fuzzy} "
               f"unmatched={n_unmatched}")
    if fuzzy_rows:
        click.echo("  fuzzy matches (review these in the payload):")
        for src, dst, m in fuzzy_rows[:20]:
            click.echo(f"    [{m}] '{src}'  ->  '{dst}'")
        if len(fuzzy_rows) > 20:
            click.echo(f"    ... and {len(fuzzy_rows) - 20} more")


@cli.command("link-coach-program")
@click.argument("athlete_name")
@click.option("--program-id", type=int, default=None,
              help="Override which Program.id to summarize. Default: athlete's "
                   "most recent non-archived program.")
def link_coach_program_cmd(athlete_name: str, program_id: int | None) -> None:
    """Link a coach program into ai_layer for an athlete whose email link is broken.

    Use this when an athlete exists in both analytics.d_athletes (warehouse) and
    app_db_snapshot.User (App DB mirror) but their emails don't match — so the
    normal `summarize-all` pipeline never associated their programs with the
    warehouse athlete_uuid. This searches the App DB snapshot by name, picks
    the user's most recent non-archived program, and saves a summary keyed to
    the WAREHOUSE athlete_uuid (overriding the broken email-derived one).

    Typical use: prep an athlete for `eval-athlete` after the email-link failed.
    """
    result = link_coach_program_by_name(athlete_name, program_id_override=program_id)
    click.echo(f"\nLinked. You can now run:")
    click.echo(f"  python -m src.main eval-athlete \"{result['athlete_name']}\"")


@cli.command("eval-athlete")
@click.argument("athlete_name")
@click.option("--focus",
              type=click.Choice(["Strength", "Power", "Speed", "In-Season",
                                 "Hypertrophy"], case_sensitive=False),
              default=None,
              help="Override the auto-detected focus. Default: mirror the coach's "
                   "actual program goals.")
@click.option("--athlete-role",
              type=click.Choice(["Starter", "Reliever"], case_sensitive=False),
              default="Starter",
              help="Pitcher role for plyo cadence rules. Default: Starter.")
@click.option("--as-of", default=None,
              help="Profile as-of date (YYYY-MM-DD). Default: today.")
@click.option("--skip-profile", is_flag=True,
              help="Skip rebuilding the athlete profile. Use when the profile "
                   "is already current and you just want to re-eval.")
def eval_athlete_cmd(athlete_name: str, focus: str | None,
                     athlete_role: str, as_of: str | None,
                     skip_profile: bool) -> None:
    """Blind-eval the recommender against a coach's actual program.

    Looks up ATHLETE_NAME in analytics.d_athletes, finds their most recent
    coach-prescribed program, runs the recommender (without showing it the
    coach's program), then diffs the two and writes a side-by-side markdown
    report. Eval is persisted to ai_layer.eval_runs for aggregation.
    """
    result = run_eval_for_athlete(
        athlete_name=athlete_name,
        focus_override=focus.title() if focus and focus != "in-season" else focus,
        as_of_date=as_of,
        athlete_role=athlete_role.title(),
        skip_profile=skip_profile,
    )
    click.echo("")
    click.echo(f"[eval] saved eval id={result['eval_id']}")
    click.echo(f"[eval] overall overlap: {result['overall_overlap_score']:.1%}")
    click.echo(f"[eval] markdown: {result['markdown_path']}")
    click.echo("[eval] per-component Recall (headline) / Precision / F1:")
    for comp, d in result["comparison"]["by_component"].items():
        r = d.get("recall")
        if r is None:
            continue
        p_str = f"{d['precision']:.1%}" if d.get("precision") is not None else "—"
        f1_str = f"{d['f1']:.1%}" if d.get("f1") is not None else "—"
        click.echo(f"         {comp:<6} Recall={r:.1%}  "
                   f"P={p_str}  F1={f1_str}  "
                   f"(coach={d['n_coach']}, rec={d['n_rec']}, both={d['n_intersection']})")
    mr = result["comparison"]["plyo_cadence"].get("match_rate")
    if mr is not None:
        click.echo(f"[eval] plyo cadence match: {mr:.1%}")


@cli.command("eval-batch")
@click.argument("names_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--athlete-role",
              type=click.Choice(["Starter", "Reliever"], case_sensitive=False),
              default="Starter")
@click.option("--skip-profile", is_flag=True,
              help="Skip rebuilding profiles. Faster but riskier if profiles are stale.")
@click.option("--stop-on-error", is_flag=True,
              help="Abort on the first failure (default: skip and continue).")
@click.option("--no-retry-transient", is_flag=True,
              help="Skip the end-of-batch retry pass for 503/transient failures. "
                   "Default: retry all transient-failed athletes once after a 30s pause.")
def eval_batch_cmd(names_file: str, athlete_role: str,
                   skip_profile: bool, stop_on_error: bool,
                   no_retry_transient: bool) -> None:
    """Run eval-athlete against every name in NAMES_FILE (one per line).

    Lines starting with `#` are treated as comments and skipped. Each athlete's
    focus is auto-detected from their most recent coach program.

    Athletes that fail with Gemini transient errors (503/UNAVAILABLE) during
    the first pass are collected and retried once at the end after a 30s pause
    (Gemini spikes typically pass within 20-40 seconds).
    """
    import time as _time
    with open(names_file, encoding="utf-8") as f:
        names = [ln.strip() for ln in f
                 if ln.strip() and not ln.strip().startswith("#")]
    click.echo(f"[batch] running eval on {len(names)} athletes...")

    def _is_transient(err: Exception) -> bool:
        s = str(err).upper()
        return any(tok in s for tok in ("503", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "429"))

    def _run_one(idx: int, total: int, name: str) -> tuple[bool, Exception | None]:
        click.echo(f"\n[batch] [{idx}/{total}] {name}")
        try:
            res = run_eval_for_athlete(
                athlete_name=name,
                athlete_role=athlete_role.title(),
                skip_profile=skip_profile,
            )
            click.echo(f"[batch]   ✓ eval id={res['eval_id']}  "
                       f"overlap={res['overall_overlap_score']:.1%}")
            return True, None
        except Exception as e:
            click.echo(f"[batch]   ✗ failed: {e}")
            if stop_on_error:
                raise
            return False, e

    succeeded = failed_permanent = 0
    transient_failed: list[str] = []
    for i, name in enumerate(names, 1):
        ok, err = _run_one(i, len(names), name)
        if ok:
            succeeded += 1
        elif err is not None and _is_transient(err) and not no_retry_transient:
            transient_failed.append(name)
        else:
            failed_permanent += 1

    if transient_failed and not no_retry_transient:
        click.echo(f"\n[batch] {len(transient_failed)} athlete(s) failed with "
                   f"transient errors; pausing 30s and retrying once...")
        _time.sleep(30)
        retry_ok = retry_fail = 0
        for i, name in enumerate(transient_failed, 1):
            click.echo(f"\n[batch][retry] [{i}/{len(transient_failed)}] {name}")
            ok, err = _run_one(i, len(transient_failed), name)
            if ok:
                retry_ok += 1
                succeeded += 1
            else:
                retry_fail += 1
                failed_permanent += 1
        click.echo(f"[batch][retry] recovered {retry_ok}, still failing {retry_fail}")

    click.echo(f"\n[batch] done: {succeeded} succeeded, {failed_permanent} failed")
    click.echo("[batch] run `python -m src.main eval-summary` for aggregated insights")


@cli.command("eval-deep-report")
@click.option("--from", "from_id", type=int, default=None,
              help="Start of eval-id range (inclusive).")
@click.option("--to", "to_id", type=int, default=None,
              help="End of eval-id range (inclusive).")
@click.option("--last", "last_n", type=int, default=None,
              help="Take the last N eval rows instead of an id range.")
def eval_deep_report_cmd(from_id: int | None, to_id: int | None,
                          last_n: int | None) -> None:
    """Generate the deep cross-eval analysis markdown.

    Reads ai_layer.eval_runs rows (no LLM cost) and produces a rich report with:
      - Pattern-clustered intent recall (Trapbar Deadlift ~ Hex Bar Deadlift match)
      - Per-component movement-pattern overlap & gaps
      - Specific exercises systematically missed / over-prescribed
      - Day-by-day plyo cadence aggregate
      - Lift template family selection summary
      - Concrete recommendations ranked by signal strength

    Examples:
        python -m src.main eval-deep-report --from 31 --to 43
        python -m src.main eval-deep-report --last 13
        python -m src.main eval-deep-report                # all evals
    """
    path = generate_deep_report(from_id=from_id, to_id=to_id, last_n=last_n)
    click.echo(f"Deep eval report written: {path}")


@cli.command("eval-summary")
@click.option("--focus", default=None,
              help="Filter to evals run under a specific focus.")
def eval_summary_cmd(focus: str | None) -> None:
    """Aggregate stats across every eval-runs row.

    Surfaces systematic biases: exercises the recommender consistently picks
    that coaches don't (over-prescribed), exercises coaches pick that the
    recommender consistently misses (gaps), and per-component overlap averages.
    """
    s = aggregate_eval_summary(focus=focus)
    if s["n_evals"] == 0:
        click.echo("No eval runs found.")
        return
    click.echo(f"[summary] n_evals: {s['n_evals']}"
               + (f"  (focus={focus})" if focus else ""))
    click.echo(f"[summary] avg overall overlap:     {s['avg_overall_overlap']:.1%}")
    if s.get("avg_plyo_cadence_match") is not None:
        click.echo(f"[summary] avg plyo cadence match:  {s['avg_plyo_cadence_match']:.1%}")
    click.echo("")
    f1 = s.get("per_component_avg_f1") or {}
    rec = s.get("per_component_avg_recall") or {}
    pre = s.get("per_component_avg_precision") or {}
    dose = s.get("per_component_avg_dose_alignment") or {}
    click.echo("[summary] per-component averages (Recall is the headline metric):")
    click.echo(f"           {'comp':<6}  {'Recall':>7}  {'Precision':>10}  {'F1':>6}  {'Dose':>6}")
    for comp in ["lift", "plyo", "prep", "bp", "hit", "me"]:
        def _p(v):
            return f"{v:.1%}" if v is not None else "—"
        click.echo(f"           {comp:<6}  {_p(rec.get(comp)):>7}  "
                   f"{_p(pre.get(comp)):>10}  {_p(f1.get(comp)):>6}  {_p(dose.get(comp)):>6}")
    click.echo("")
    over = s.get("most_overprescribed") or []
    if over:
        click.echo("[summary] top exercises recommender added (coach didn't pick):")
        for key, n in over[:15]:
            click.echo(f"           {n:>3}×  {key}")
    miss = s.get("most_missed") or []
    if miss:
        click.echo("")
        click.echo("[summary] top exercises coach picked (recommender missed):")
        for key, n in miss[:15]:
            click.echo(f"           {n:>3}×  {key}")
    fams = s.get("lift_template_family_distribution") or []
    if fams:
        click.echo("")
        click.echo("[summary] lift template families the recommender picked:")
        for fam, n in fams[:15]:
            click.echo(f"           {n:>3}×  {fam}")


@cli.command("load-templates")
def load_templates_cmd() -> None:
    """Load the lift-programming template catalog into ai_layer.lift_templates.

    Reads skills/lift-programming/references/template_catalog.json (generated
    by skills/lift-programming/scripts/parse_templates.py). Idempotent —
    TRUNCATEs and reloads on each run.
    """
    result = load_catalog()
    click.echo(f"[templates] loaded {result['n_templates']} templates, "
               f"{result['n_exercises']} exercises across {result['n_families']} families")
    load_templates_summary()


@cli.command()
@click.option("--k", default=5, type=int, help="Number of archetype clusters.")
@click.option("--role", type=click.Choice(["pitcher", "hitter", "all"], case_sensitive=False),
              default="all", help="Filter to pitchers, hitters, or all athletes.")
@click.option("--domain",
              type=click.Choice(
                  ["movement", "mobility", "performance", "functional", "all"],
                  case_sensitive=False),
              default="all",
              help="movement = pitch/hit kinematics (drives plyo/drill rx). "
                   "mobility = ROM + soft tissue (drives prep/bulletproofing). "
                   "performance = athletic screen + proteus (drives lifts). "
                   "functional = mobility + performance combined (legacy wide view).")
@click.option("--output", default=None, help="Output HTML path (defaults to outputs/...).")
def report(k: int, role: str, domain: str, output: str | None) -> None:
    """Generate a population correlation + archetype report (HTML).

    Recommended sharp-archetype sweep (one report per prescription type):
      python -m src.main report --role pitcher --domain movement
      python -m src.main report --role pitcher --domain mobility
      python -m src.main report --role pitcher --domain performance
      python -m src.main report --role hitter  --domain movement     --k 3
      python -m src.main report --role hitter  --domain mobility     --k 3
      python -m src.main report --role hitter  --domain performance  --k 3

    Wide net for cross-domain discovery (when you want to find surprise
    correlations between mobility and power, etc.):
      python -m src.main report --role pitcher --domain functional
      python -m src.main report --role pitcher --domain all
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
