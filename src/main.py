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
import pandas as pd

from src.backfill_profiles import run_backfill
from src.correlation_report import generate_report
from src.db import backend_conn, query
from src.load_templates import load_catalog, summary as load_templates_summary
from src.eval_deep_report import generate_deep_report
from src.research import athlete_deep_dive as _r_dive
from src.research import clustering as _r_clustering
from src.research import correlations as _r_corr
from src.research import kinematic_drivers as _r_kdrv
from src.research import session_change as _r_schange
from src.research import longitudinal as _r_long
from src.research import pitching_deep as _r_pdeep
from src.research import profile_matrix as _r_matrix
from src.research import program_response as _r_prog
from src.research import reports as _r_reports
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


@cli.command("send-profile")
@click.argument("athlete_uuid")
@click.option("--as-of", "as_of", default=None,
              help="Profile as-of date (YYYY-MM-DD). Default: latest profile.")
@click.option("--dry-run", is_flag=True, default=False,
              help="Build and summarize the payload without POSTing.")
def send_profile_cmd(athlete_uuid: str, as_of: str | None, dry_run: bool) -> None:
    """Send a compiled athlete profile to Octane's /api/biomech/profile.

    Requires OCTANE_API_URL and OCTANE_REPORTS_API_KEY in .env. The profile
    must already exist (run the profiler first). Octane matches the athlete by
    octane user uuid + email and stores the document in its reports DB, where
    the admin template-generation page reads it.
    """
    from src.octane_sync import send_profile

    result = send_profile(athlete_uuid, as_of=as_of, dry_run=dry_run)
    if dry_run:
        click.echo(f"[send-profile] dry run — {result['athlete']} "
                   f"as of {result['asOfDate']} (role={result['role']}, "
                   f"{result['metrics']} metrics). Nothing sent.")
    else:
        click.echo(f"[send-profile] sent: {json.dumps(result)}")


@cli.command("send-corpus")
@click.option("--athlete", "athlete_uuid", default=None,
              help="Send only this warehouse athlete_uuid. Default: all athletes with profiles.")
@click.option("--limit", type=int, default=None, help="Cap the number of athletes sent.")
@click.option("--dry-run", is_flag=True, default=False,
              help="Count athletes/prescriptions without POSTing.")
def send_corpus_cmd(athlete_uuid: str | None, limit: int | None, dry_run: bool) -> None:
    """Push the reference corpus to Octane (/api/biomech/corpus).

    Sends every warehouse athlete's latest profile + coach prescription history
    (one POST per athlete, idempotent). Octane's AI program generation uses
    this corpus for similar-athlete search and candidate exercise pools.
    Re-run whenever profiles are backfilled or program summaries change.
    """
    from src.octane_sync import send_corpus

    result = send_corpus(athlete_uuid=athlete_uuid, limit=limit, dry_run=dry_run)
    if dry_run:
        click.echo(f"[send-corpus] dry run — {result['athletes']} athlete(s), "
                   f"{result['prescriptions']} prescription row(s). Nothing sent.")
    else:
        click.echo(f"[send-corpus] sent {result['sent']}/{result['athletes']} athlete(s)")
        for f in result["failed"]:
            click.echo(f"[send-corpus]   FAILED {f}")


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


# ──────────────────────────────────────────────────────────────────────────
# Research subcommand group — deep exploration of the warehouse
# ──────────────────────────────────────────────────────────────────────────

@cli.group("research")
def research_grp() -> None:
    """Deep research on the assessment corpus.

    Every subcommand writes an interactive HTML report to outputs/. Open in a
    browser to explore.
    """


@research_grp.command("coverage")
@click.option("--role", type=click.Choice(["pitcher", "hitter", "both", "all"],
                                          case_sensitive=False), default="all")
@click.option("--age-group", default=None,
              help="e.g. 'COLLEGE', 'HIGH SCHOOL'")
@click.option("--min-as-of-date", default=None,
              help="YYYY-MM-DD — drop profiles older than this. Use to exclude "
                   "legacy mobility data measured on the 1-3 scale.")
@click.option("--exclude-mobility", is_flag=True,
              help="Drop all mobility metrics. Mobility protocol changed over "
                   "the years so pooled data can be misleading.")
def research_coverage_cmd(role: str, age_group: str | None,
                           min_as_of_date: str | None,
                           exclude_mobility: bool) -> None:
    """What data do we actually have? Per-metric coverage across the cohort.

    Run this FIRST — tells you which metrics have enough non-null data to be
    worth correlating.
    """
    r = None if role == "all" else role
    df = _r_matrix.load_matrix(role=r, age_group=age_group, latest_only=True,
                                min_as_of_date=min_as_of_date,
                                exclude_mobility=exclude_mobility)
    if df.empty:
        click.echo("No profiles matched. Try loosening the filter.")
        return
    cov = _r_matrix.coverage_report(df)
    path = _r_reports.render_coverage_report(cov, n_athletes=len(df))
    click.echo(f"Coverage report: {path}")
    click.echo(f"  n athletes: {len(df)} · n metric columns: {len(cov)}")


@research_grp.command("list-metrics")
@click.option("--role", type=click.Choice(["pitcher", "hitter", "both", "all"],
                                          case_sensitive=False), default="all")
@click.option("--age-group", default=None)
@click.option("--domain", default=None,
              help="Filter to metrics in one domain (mobility, pitching_3d, ...)")
@click.option("--search", default=None,
              help="Substring filter (case-insensitive).")
def research_list_metrics_cmd(role: str, age_group: str | None,
                                domain: str | None, search: str | None) -> None:
    """List available metrics grouped by domain. Handy before `correlate`.

    Example:
        python -m src.main research list-metrics --domain pitching_3d
        python -m src.main research list-metrics --search hip_shoulder
    """
    r = None if role == "all" else role
    df = _r_matrix.load_matrix(role=r, age_group=age_group, latest_only=True)
    if df.empty:
        click.echo("No profiles matched.")
        return
    by_domain = _r_matrix.columns_by_domain(df)
    total = 0
    for d in sorted(by_domain):
        if domain and d != domain:
            continue
        cols = by_domain[d]
        if search:
            s = search.lower()
            cols = [c for c in cols if s in c.lower()]
        if not cols:
            continue
        click.echo(f"\n[{d}]  ({len(cols)} metrics)")
        for c in sorted(cols):
            coverage = df[c].notna().mean()
            click.echo(f"  {c:<50}  n={int(df[c].notna().sum()):>3}  ({coverage:.0%})")
            total += 1
    click.echo(f"\nTotal: {total} metrics.")


@research_grp.command("correlate")
@click.argument("target_metric")
@click.option("--role", type=click.Choice(["pitcher", "hitter", "both", "all"],
                                          case_sensitive=False), default="all")
@click.option("--age-group", default=None)
@click.option("--method", type=click.Choice(["spearman", "pearson"]), default="spearman")
@click.option("--min-n", default=8, type=int,
              help="Skip pairs with fewer athletes having both metrics.")
@click.option("--fdr-alpha", default=0.10, type=float,
              help="FDR threshold for the fdr_significant flag.")
@click.option("--exclude-same-domain", is_flag=True,
              help="Skip metrics in the same domain as the target — useful for "
                   "cross-modality-only findings.")
@click.option("--min-as-of-date", default=None,
              help="YYYY-MM-DD — drop profiles older than this.")
@click.option("--exclude-mobility", is_flag=True,
              help="Drop mobility metrics (protocol changed over the years).")
@click.option("--by-age-group", is_flag=True,
              help="Run correlate SEPARATELY per age group (YOUTH / HIGH SCHOOL / "
                   "COLLEGE / PRO) and produce a combined comparison report. "
                   "Reveals age-specific mechanics.")
def research_correlate_cmd(target_metric: str, role: str, age_group: str | None,
                            method: str, min_n: int, fdr_alpha: float,
                            exclude_same_domain: bool,
                            min_as_of_date: str | None,
                            exclude_mobility: bool,
                            by_age_group: bool) -> None:
    """Correlate ONE metric against every other metric across the cohort.

    Use for hypothesis-driven analysis. Run `research list-metrics` first to
    see what's available. Example:

        python -m src.main research correlate pitch_hip_shoulder_sep_at_fc --role pitcher
    """
    r = None if role == "all" else role

    # ── Optional: run stratified per age group and produce a comparison report
    if by_age_group:
        AGE_GROUPS = ["YOUTH", "HIGH SCHOOL", "COLLEGE", "PRO"]
        strata: dict[str, dict] = {}
        for ag in AGE_GROUPS:
            df_ag = _r_matrix.load_matrix(
                role=r, age_group=ag, latest_only=True,
                min_as_of_date=min_as_of_date,
                exclude_mobility=exclude_mobility,
            )
            if df_ag.empty or target_metric not in df_ag.columns:
                strata[ag] = {"correlations": pd.DataFrame(), "n_athletes": len(df_ag)}
                continue
            corr = _r_corr.correlate_target(
                df_ag, target_metric, method=method, min_n=min_n,
                fdr_alpha=fdr_alpha,
                exclude_same_domain=exclude_same_domain,
            )
            strata[ag] = {"correlations": corr, "n_athletes": len(df_ag)}
        # Drop age groups with zero athletes so the report is cleaner
        strata = {k: v for k, v in strata.items() if v["n_athletes"] > 0}
        if not strata:
            click.echo(f"No athletes with {target_metric!r} across any age group.")
            return
        path = _r_reports.render_stratified_correlation_report(
            target_metric, strata, role=r,
        )
        click.echo(f"Stratified correlation report: {path}")
        for ag, st in strata.items():
            corr = st["correlations"]
            n_sig = int(corr["fdr_significant"].sum()) if not corr.empty else 0
            click.echo(f"  {ag:<12} n={st['n_athletes']:>3} · "
                       f"{len(corr):>3} metrics · {n_sig} FDR-sig")
        return

    df = _r_matrix.load_matrix(role=r, age_group=age_group, latest_only=True,
                                min_as_of_date=min_as_of_date,
                                exclude_mobility=exclude_mobility)
    if df.empty:
        click.echo("No profiles matched.")
        return
    if target_metric not in df.columns:
        # Fuzzy-suggest similar names
        from difflib import get_close_matches
        all_metric_cols = _r_matrix.metric_columns(df)
        close = get_close_matches(target_metric, all_metric_cols, n=8, cutoff=0.4)
        click.echo(f"Metric {target_metric!r} not present in the cohort's profiles.")
        if close:
            click.echo("\nDid you mean one of these?")
            for c in close:
                click.echo(f"  {c}")
        click.echo("\nRun `python -m src.main research list-metrics --search "
                    f"{target_metric.split('_')[0]}` to browse.")
        return
    corr = _r_corr.correlate_target(
        df, target_metric, method=method, min_n=min_n, fdr_alpha=fdr_alpha,
        exclude_same_domain=exclude_same_domain,
    )
    path = _r_reports.render_target_correlation_report(
        target_metric, corr, n_athletes=len(df), role=r,
    )
    click.echo(f"Correlation report: {path}")
    click.echo(f"  n athletes: {len(df)} · {len(corr)} metrics correlated · "
               f"{int(corr['fdr_significant'].sum()) if not corr.empty else 0} "
               f"survived FDR α={fdr_alpha}")


@research_grp.command("cross")
@click.argument("domain_a")
@click.argument("domain_b")
@click.option("--role", type=click.Choice(["pitcher", "hitter", "both", "all"],
                                          case_sensitive=False), default="all")
@click.option("--age-group", default=None)
@click.option("--min-n", default=8, type=int)
@click.option("--fdr-alpha", default=0.10, type=float)
@click.option("--min-as-of-date", default=None,
              help="YYYY-MM-DD — drop profiles older than this.")
@click.option("--exclude-mobility", is_flag=True,
              help="Drop mobility metrics (protocol changed over the years).")
def research_cross_cmd(domain_a: str, domain_b: str, role: str,
                        age_group: str | None, min_n: int, fdr_alpha: float,
                        min_as_of_date: str | None,
                        exclude_mobility: bool) -> None:
    """Cross-domain full correlation matrix. Every metric in A × every metric in B.

    Domains: mobility, pitching_3d, hitting_3d, athletic_screen_dj, athletic_screen_cmj,
             athletic_screen_ppu, athletic_screen_slv, proteus, force_plate,
             readiness_screen, arm_action, curveball_test

    Example:
        python -m src.main research cross mobility pitching_3d --role pitcher
    """
    r = None if role == "all" else role
    df = _r_matrix.load_matrix(role=r, age_group=age_group, latest_only=True,
                                min_as_of_date=min_as_of_date,
                                exclude_mobility=exclude_mobility)
    if df.empty:
        click.echo("No profiles matched.")
        return
    try:
        mats = _r_corr.cross_domain_matrix(
            df, domain_a, domain_b, min_n=min_n, fdr_alpha=fdr_alpha,
        )
    except ValueError as e:
        click.echo(f"Error: {e}")
        return
    top = _r_corr.top_findings_from_matrix(mats, top_n=40)
    path = _r_reports.render_cross_domain_report(
        domain_a, domain_b, mats, top, n_athletes=len(df), role=r,
    )
    click.echo(f"Cross-domain report: {path}")
    n_sig = int(mats["sig"].fillna(False).values.sum())
    click.echo(f"  n athletes: {len(df)} · matrix {mats['r'].shape[0]}×{mats['r'].shape[1]} · "
               f"{n_sig} pairs survived FDR α={fdr_alpha}")


@research_grp.command("overview")
@click.option("--target-domain", default="pitching_3d",
              help="The 'right side' of every cross-domain matrix. "
                   "Default: pitching_3d.")
@click.option("--role", type=click.Choice(["pitcher", "hitter", "both", "all"],
                                          case_sensitive=False), default="pitcher")
@click.option("--age-group", default=None)
@click.option("--min-n", default=8, type=int)
@click.option("--fdr-alpha", default=0.10, type=float)
@click.option("--min-as-of-date", default=None,
              help="YYYY-MM-DD — drop profiles older than this.")
@click.option("--exclude-mobility", is_flag=True,
              help="Drop mobility metrics (protocol changed over the years).")
def research_overview_cmd(target_domain: str, role: str, age_group: str | None,
                           min_n: int, fdr_alpha: float,
                           min_as_of_date: str | None,
                           exclude_mobility: bool) -> None:
    """Run every non-target domain × target_domain in one shot.

    Produces one HTML with:
      - All cross-domain matrices as heatmaps
      - GLOBAL top-findings table with FDR applied across the whole test space

    Example:
        # What predicts pitching mechanics across ALL modalities:
        python -m src.main research overview --target-domain pitching_3d --role pitcher

        # What predicts hitting mechanics:
        python -m src.main research overview --target-domain hitting_3d --role hitter
    """
    r = None if role == "all" else role
    df = _r_matrix.load_matrix(role=r, age_group=age_group, latest_only=True,
                                min_as_of_date=min_as_of_date,
                                exclude_mobility=exclude_mobility)
    if df.empty:
        click.echo("No profiles matched.")
        return

    by_domain = _r_matrix.columns_by_domain(df)
    if target_domain not in by_domain:
        click.echo(f"Target domain {target_domain!r} not present. "
                   f"Available: {sorted(by_domain)}")
        return

    # Source domains = every domain that isn't the target or 'other'
    source_domains = [d for d in sorted(by_domain)
                       if d != target_domain and d != "other"]
    click.echo(f"[overview] {target_domain} vs {len(source_domains)} source domains: "
               f"{source_domains}")

    matrices_by_domain: dict[str, dict] = {}
    all_findings: list[pd.DataFrame] = []
    for src in source_domains:
        try:
            mats = _r_corr.cross_domain_matrix(
                df, src, target_domain, min_n=min_n, fdr_alpha=fdr_alpha,
            )
        except ValueError as e:
            click.echo(f"  [{src}] skipped: {e}")
            continue
        matrices_by_domain[src] = mats
        # Flatten to findings (without local FDR — we'll do global FDR at the end)
        rows = []
        r_mat, n_mat = mats["r"], mats["n"]
        p_mat = mats["p"]
        for a in r_mat.index:
            for b in r_mat.columns:
                r_val = r_mat.at[a, b]
                if pd.isna(r_val):
                    continue
                rows.append({
                    "source_domain": src,
                    "metric_a": a,
                    "metric_b": b,
                    "r": float(r_val),
                    "p_value": float(p_mat.at[a, b]) if not pd.isna(p_mat.at[a, b]) else None,
                    "n": int(n_mat.at[a, b]) if not pd.isna(n_mat.at[a, b]) else None,
                })
        if rows:
            all_findings.append(pd.DataFrame(rows))

    # ── Global FDR across ALL findings from ALL source domains
    if all_findings:
        top = pd.concat(all_findings, ignore_index=True)
        top["q_value"] = _r_corr._bh_q_values(top["p_value"])
        top["fdr_significant"] = top["q_value"] <= fdr_alpha
        top["abs_r"] = top["r"].abs()
        top = (top.sort_values("abs_r", ascending=False)
                   .drop(columns=["abs_r"])
                   .reset_index(drop=True))
    else:
        top = pd.DataFrame()

    path = _r_reports.render_overview_report(
        target_domain, matrices_by_domain, top,
        n_athletes=len(df), role=r,
    )
    click.echo(f"Overview report: {path}")
    n_sig = int(top["fdr_significant"].sum()) if not top.empty else 0
    click.echo(f"  n athletes: {len(df)} · "
               f"total pairs: {len(top) if not top.empty else 0} · "
               f"{n_sig} FDR-significant globally at α={fdr_alpha}")


@research_grp.command("clusters")
@click.option("--k", default=None, type=int,
              help="Number of clusters. Omit to auto-select via silhouette.")
@click.option("--role", type=click.Choice(["pitcher", "hitter", "both", "all"],
                                          case_sensitive=False), default="all")
@click.option("--age-group", default=None)
@click.option("--domain", "-d", multiple=True,
              help="Restrict clustering to one or more domains (repeatable). "
                   "e.g. -d mobility -d pitching_3d. Default: use all metrics.")
@click.option("--min-coverage", default=0.5, type=float,
              help="Drop metric columns with less than this fraction of "
                   "non-null values across the cohort.")
@click.option("--min-as-of-date", default=None,
              help="YYYY-MM-DD — drop profiles older than this.")
@click.option("--exclude-mobility", is_flag=True,
              help="Drop mobility metrics (protocol changed over the years).")
def research_clusters_cmd(k: int | None, role: str, age_group: str | None,
                           domain: tuple[str, ...], min_coverage: float,
                           min_as_of_date: str | None,
                           exclude_mobility: bool) -> None:
    """Discover athlete archetypes via KMeans + PCA.

    Example:
        python -m src.main research clusters --role pitcher -d mobility -d pitching_3d
    """
    r = None if role == "all" else role
    df = _r_matrix.load_matrix(role=r, age_group=age_group, latest_only=True,
                                min_non_null_metrics=8,
                                min_as_of_date=min_as_of_date,
                                exclude_mobility=exclude_mobility)
    if df.empty:
        click.echo("No profiles matched.")
        return
    domains = list(domain) if domain else None
    try:
        result = _r_clustering.cluster_athletes(
            df, k=k, restrict_to_domains=domains, min_coverage=min_coverage,
        )
    except ValueError as e:
        click.echo(f"Error: {e}")
        return
    path = _r_reports.render_clustering_report(result, role=r, domains=domains)
    click.echo(f"Cluster report: {path}")
    click.echo(f"  n athletes: {result['n_athletes']} · k: {result['k']} · "
               f"silhouette: {result['silhouette']}")


@research_grp.command("longitudinal")
@click.option("--role", type=click.Choice(["pitcher", "hitter", "both", "all"],
                                          case_sensitive=False), default="all")
@click.option("--min-span-days", default=30, type=int)
@click.option("--max-span-days", default=365, type=int)
@click.option("--min-n", default=6, type=int,
              help="Min athletes required per (pattern × delta) pair.")
@click.option("--skip-program-response", is_flag=True,
              help="Compute deltas only, skip the pattern-response correlation.")
def research_longitudinal_cmd(role: str, min_span_days: int, max_span_days: int,
                               min_n: int, skip_program_response: bool) -> None:
    """Pre/post analysis + program-response correlations.

    Requires athletes with ≥2 profile rows. Correlates exercise pattern volume
    (from prescriptions overlapping the window) with metric-delta.
    """
    r = None if role == "all" else role
    deltas = _r_long.load_deltas(
        role=r, min_span_days=min_span_days, max_span_days=max_span_days,
    )
    if deltas.empty:
        click.echo("No athletes with ≥2 profiles in span window.")
        return
    summary = _r_long.delta_summary(deltas)
    if skip_program_response:
        findings = pd.DataFrame()
    else:
        click.echo(f"[longitudinal] loading prescriptions for {len(deltas)} athletes...")
        volumes = _r_prog.pattern_volume_by_athlete(deltas)
        findings = _r_prog.correlate_pattern_volume_to_delta(
            deltas, volumes, min_n=min_n,
        )
    path = _r_reports.render_longitudinal_report(deltas, summary, findings, role=r)
    click.echo(f"Longitudinal report: {path}")
    click.echo(f"  n athletes with serial profiles: {len(deltas)}")
    if not findings.empty:
        click.echo(f"  {len(findings)} pattern × metric pairs correlated")


@research_grp.command("trial-metrics")
@click.option("--age-group", default=None,
              help="e.g. 'COLLEGE', 'HIGH SCHOOL', 'PRO'. Filters at the trial level.")
@click.option("--min-as-of-date", default=None,
              help="YYYY-MM-DD — drop trials older than this.")
@click.option("--search", default=None,
              help="Substring filter on metric name.")
def research_trial_metrics_cmd(age_group: str | None,
                                min_as_of_date: str | None,
                                search: str | None) -> None:
    """List every metric present in the trial-level tables (f_pitching_trials
    JSON metrics + f_pitching_force_metrics), with coverage stats.

    Use this to see what's actually available BEFORE running velocity-deep.
    """
    df = _r_pdeep.load_pitching_trials_wide(
        age_group=age_group, min_as_of_date=min_as_of_date,
    )
    if df.empty:
        click.echo("No pitching trials matched.")
        return
    cov = _r_pdeep.list_trial_metrics(df)
    if search:
        s = search.lower()
        cov = cov[cov["metric"].str.lower().str.contains(s)]
    click.echo(f"Trials: {len(df):,} · Athletes: {df['athlete_uuid'].nunique()}")
    click.echo(f"Total metric columns: {len(cov)}\n")
    for family in sorted(cov["family"].unique()):
        sub = cov[cov["family"] == family]
        click.echo(f"[{family}]  ({len(sub)} metrics)")
        for _, r in sub.iterrows():
            click.echo(f"  {r['metric']:<55}  n={int(r['n_trials_non_null']):>4}  ({r['coverage_pct']:.0%})")
        click.echo()


@research_grp.command("velocity-deep")
@click.option("--age-group", default=None,
              help="Restrict to one age group at load time (also compared to "
                   "stratified analysis below).")
@click.option("--min-velocity", default=None, type=float,
              help="Drop trials below this velocity (removes warm-up/off throws).")
@click.option("--max-velocity", default=None, type=float,
              help="Drop trials above this velocity (rare outliers).")
@click.option("--min-trials-per-athlete", default=3, type=int,
              help="Min trials per athlete for within-athlete analysis.")
@click.option("--min-sessions-per-athlete", default=2, type=int,
              help="Min sessions per athlete for session-level analysis.")
@click.option("--min-n-pooled", default=30, type=int,
              help="Min trials for pooled correlation to be considered.")
@click.option("--min-as-of-date", default=None,
              help="YYYY-MM-DD — drop trials older than this.")
@click.option("--top-k-backwards", default=8, type=int,
              help="Number of top within-athlete velocity correlates to run "
                   "backwards-chain on (0 to skip backwards chain).")
@click.option("--processed-only", is_flag=True,
              help="Drop kin_INCREMENT.* timepoint metrics — keep only "
                   "kin_PROCESSED.* summary metrics. Cuts ~800 noise columns "
                   "and dramatically improves FDR power.")
@click.option("--exclude-symptomatic", is_flag=True,
              help="Drop metrics that are OUTPUTS of throwing hard rather than "
                   "causes: elbow/shoulder torque, distraction force, humerus "
                   "angular ACC. Focuses the analysis on causal-adjacent mechanics.")
def research_velocity_deep_cmd(age_group: str | None,
                                min_velocity: float | None,
                                max_velocity: float | None,
                                min_trials_per_athlete: int,
                                min_sessions_per_athlete: int,
                                min_n_pooled: int,
                                min_as_of_date: str | None,
                                top_k_backwards: int,
                                processed_only: bool,
                                exclude_symptomatic: bool) -> None:
    """The BIG velocity search.

    Runs 4 levels of trial-level analysis simultaneously:
      1. Pooled — every trial across every athlete
      2. Age-group stratified — same but split
      3. Within-athlete (fixed effects) — the strongest control
      4. Session-level — for athletes with ≥2 sessions

    Plus optional backwards-chain: for the top within-athlete velocity
    correlates, find what correlates with THEM.

    Example:
        # Full analysis — recommended first run
        python -m src.main research velocity-deep

        # College pitchers only, filter out warmup throws under 75 mph
        python -m src.main research velocity-deep --age-group COLLEGE --min-velocity 75
    """
    click.echo(f"[velocity-deep] loading trial-level data...")
    df = _r_pdeep.load_pitching_trials_wide(
        age_group=age_group,
        min_velocity=min_velocity,
        max_velocity=max_velocity,
        min_as_of_date=min_as_of_date,
    )
    if df.empty:
        click.echo("No pitching trials matched.")
        return

    n_trials = len(df)
    n_athletes = df["athlete_uuid"].nunique()
    click.echo(f"  {n_trials:,} trials across {n_athletes} athletes")

    if processed_only:
        click.echo("[velocity-deep] filter: PROCESSED metrics only (dropping INCREMENT.*)")
    if exclude_symptomatic:
        click.echo("[velocity-deep] filter: excluding symptomatic metrics "
                   "(elbow/shoulder torque, distraction force, humerus angular acc)")

    click.echo(f"[velocity-deep] pooled analysis...")
    pooled = _r_pdeep.correlate_velocity_pooled(
        df, min_n=min_n_pooled,
        processed_only=processed_only, exclude_symptomatic=exclude_symptomatic,
    )

    click.echo(f"[velocity-deep] stratified by age group...")
    stratified = _r_pdeep.correlate_velocity_stratified(
        df, min_n=15,
        processed_only=processed_only, exclude_symptomatic=exclude_symptomatic,
    )

    click.echo(f"[velocity-deep] within-athlete fixed effects "
               f"(min {min_trials_per_athlete} trials/athlete)...")
    within = _r_pdeep.correlate_velocity_within_athlete(
        df, min_trials_per_athlete=min_trials_per_athlete, min_n=30,
        processed_only=processed_only, exclude_symptomatic=exclude_symptomatic,
    )

    click.echo(f"[velocity-deep] session-level within-athlete "
               f"(min {min_sessions_per_athlete} sessions/athlete)...")
    session = _r_pdeep.correlate_velocity_session_level(
        df, min_sessions_per_athlete=min_sessions_per_athlete, min_n=20,
        processed_only=processed_only, exclude_symptomatic=exclude_symptomatic,
    )

    backwards: dict | None = None
    if top_k_backwards > 0 and not within.empty:
        top_targets = within.head(top_k_backwards)["metric"].tolist()
        click.echo(f"[velocity-deep] backwards-chain for top {len(top_targets)} "
                   f"within-athlete velocity correlates...")
        backwards = _r_pdeep.backwards_chain(
            df, top_targets,
            min_trials_per_athlete=min_trials_per_athlete, min_n=30,
            processed_only=processed_only, exclude_symptomatic=exclude_symptomatic,
        )

    path = _r_reports.render_velocity_deep_report(
        n_trials=n_trials, n_athletes=n_athletes, age_group=age_group,
        pooled=pooled, stratified=stratified,
        within_athlete=within, session_level=session,
        backwards=backwards,
    )
    click.echo(f"\nVelocity-deep report: {path}")
    click.echo(f"  pooled           · {len(pooled):>4} metrics · "
               f"{int(pooled['fdr_significant'].sum()) if not pooled.empty else 0} FDR-sig")
    click.echo(f"  within-athlete   · {len(within):>4} metrics · "
               f"{int(within['fdr_significant'].sum()) if not within.empty else 0} FDR-sig")
    click.echo(f"  session-level    · {len(session):>4} metrics · "
               f"{int(session['fdr_significant'].sum()) if not session.empty else 0} FDR-sig")
    for ag, sf in stratified.items():
        click.echo(f"  stratified[{ag:<12}] · {len(sf):>4} metrics · "
                   f"{int(sf['fdr_significant'].sum()) if not sf.empty else 0} FDR-sig")


@research_grp.command("stratified-chain")
@click.option("--top-k", default=5, type=int,
              help="Top K velocity correlates per stratum to chain to assessments.")
@click.option("--strata", multiple=True,
              help="Age groups to include. Default: HS + COLLEGE + PRO + YOUTH. "
                   "Repeatable: --strata COLLEGE --strata PRO")
@click.option("--min-trials-per-athlete", default=3, type=int)
@click.option("--min-trials-agg", default=3, type=int,
              help="Min trials for kinematic→athlete aggregation.")
@click.option("--min-n-within", default=30, type=int)
@click.option("--min-n-assessment", default=8, type=int)
@click.option("--aggregation", type=click.Choice(["mean", "median", "max"]),
              default="mean")
@click.option("--fdr-alpha", default=0.10, type=float)
@click.option("--min-as-of-date", default=None)
@click.option("--exclude-mobility", is_flag=True)
def research_stratified_chain_cmd(
    top_k: int, strata: tuple[str, ...], min_trials_per_athlete: int,
    min_trials_agg: int, min_n_within: int, min_n_assessment: int,
    aggregation: str, fdr_alpha: float, min_as_of_date: str | None,
    exclude_mobility: bool,
) -> None:
    """Per-age-group triangulation: within THAT stratum, find top velocity
    correlates AND their assessment predictors — using only athletes in that
    stratum for both stages.

    Answers stratum-specific coaching questions:
      - What assessment predicts a PRO's velocity mechanic (in PROS)?
      - What assessment predicts a HS pitcher's velocity mechanic (in HS)?

    Example:
        python -m src.main research stratified-chain --top-k 5
        python -m src.main research stratified-chain --top-k 8 --strata PRO --strata COLLEGE
    """
    strata_list = list(strata) if strata else None
    click.echo(f"[stratified-chain] running for strata: "
               f"{strata_list or 'default (HS/COLLEGE/PRO/YOUTH)'}")
    results = _r_kdrv.stratified_chain(
        strata=strata_list, top_k_per_stratum=top_k,
        min_trials_per_athlete=min_trials_per_athlete,
        min_trials_agg=min_trials_agg,
        min_n_within=min_n_within,
        min_n_assessment=min_n_assessment,
        aggregation=aggregation, fdr_alpha=fdr_alpha,
        min_as_of_date=min_as_of_date,
        exclude_mobility=exclude_mobility,
        processed_only=True, exclude_symptomatic=True,
    )
    path = _r_reports.render_stratified_chain_report(
        results, top_k_per_stratum=top_k,
    )
    click.echo(f"Stratified chain report: {path}")
    for stratum, sr in results.items():
        if "error" in sr and "chain" not in sr:
            click.echo(f"  [{stratum}] {sr['error']}")
            continue
        n_ath = sr.get("n_athletes_profile", 0)
        n_kin = len(sr.get("top_velocity_correlates", pd.DataFrame()))
        n_chain_sig = sum(
            int(cr.get("correlations", pd.DataFrame()).get("fdr_significant",
                pd.Series(dtype=bool)).sum())
            for cr in sr.get("chain", {}).values() if "error" not in cr
        )
        click.echo(f"  [{stratum}] {n_kin} kin correlates · "
                   f"n_ath_profile={n_ath} · "
                   f"{n_chain_sig} total FDR-sig predictors across the chain")


@research_grp.command("kinematic-drivers")
@click.argument("kin_metrics", nargs=-1)
@click.option("--role", type=click.Choice(["pitcher", "hitter", "both", "all"],
                                          case_sensitive=False), default="pitcher")
@click.option("--age-group", default=None)
@click.option("--min-trials", default=3, type=int,
              help="Min trials per athlete for the trial→athlete aggregation.")
@click.option("--aggregation", type=click.Choice(["mean", "median", "max"]),
              default="mean",
              help="How to collapse an athlete's multiple trials to one value.")
@click.option("--min-n", default=8, type=int,
              help="Min athletes for a correlation to be considered.")
@click.option("--fdr-alpha", default=0.10, type=float)
@click.option("--min-as-of-date", default=None,
              help="YYYY-MM-DD — drop profiles older than this.")
@click.option("--exclude-mobility", is_flag=True,
              help="Drop mobility metrics.")
@click.option("--from-velocity-deep", default=None, type=int,
              help="Auto-run on the top-N within-athlete velocity correlates "
                   "from the last velocity-deep run. Use this to chain "
                   "velocity ← kinematic ← assessment automatically.")
def research_kinematic_drivers_cmd(
    kin_metrics: tuple[str, ...], role: str, age_group: str | None,
    min_trials: int, aggregation: str, min_n: int, fdr_alpha: float,
    min_as_of_date: str | None, exclude_mobility: bool,
    from_velocity_deep: int | None,
) -> None:
    """For each kinematic metric, find which ASSESSMENT metrics predict it.

    Aggregates each athlete's trials to a single value, correlates that value
    across athletes against ai_layer.athlete_profiles Z-scores (mobility,
    athletic screen, proteus, ...).

    Examples:
        # One specific kinematic — pelvis brake mechanic
        python -m src.main research kinematic-drivers "kin_PROCESSED.Pelvis_Ang_Vel@Release.Z"

        # Multiple in one shot
        python -m src.main research kinematic-drivers \\
            "kin_PROCESSED.Pelvis_Ang_Vel@Release.Z" \\
            "kin_PROCESSED.Lead_Knee_Angle@Release.X"

        # Auto-chain from velocity-deep: take top-5 within-athlete velocity
        # correlates and find what predicts them
        python -m src.main research kinematic-drivers --from-velocity-deep 5
    """
    if from_velocity_deep and not kin_metrics:
        # Discover top within-athlete velocity correlates automatically
        click.echo(f"[kinematic-drivers] auto-discovering top {from_velocity_deep} "
                   f"velocity correlates from velocity-deep...")
        trial_df = _r_pdeep.load_pitching_trials_wide(age_group=age_group)
        if trial_df.empty:
            click.echo("No trial data.")
            return
        within = _r_pdeep.correlate_velocity_within_athlete(
            trial_df, min_trials_per_athlete=min_trials, min_n=30,
            processed_only=True, exclude_symptomatic=True,
        )
        if within.empty:
            click.echo("No within-athlete velocity correlates found.")
            return
        kin_metrics = tuple(within.head(from_velocity_deep)["metric"].tolist())
        click.echo(f"  targeting: {list(kin_metrics)}")

    if not kin_metrics:
        click.echo("No kinematic metrics specified. Provide one or more as args "
                   "or use --from-velocity-deep N.")
        return

    r = None if role == "all" else role
    click.echo(f"[kinematic-drivers] analyzing {len(kin_metrics)} metric(s)...")
    results = _r_kdrv.batch_kinematic_drivers(
        list(kin_metrics),
        role=r, age_group=age_group,
        min_trials=min_trials, aggregation=aggregation,
        min_n=min_n, fdr_alpha=fdr_alpha,
        min_as_of_date=min_as_of_date,
        exclude_mobility=exclude_mobility,
    )
    path = _r_reports.render_kinematic_drivers_report(
        results, role=r, age_group=age_group, aggregation=aggregation,
    )
    click.echo(f"Kinematic-drivers report: {path}")
    for kin, res in results.items():
        if "error" in res:
            click.echo(f"  {kin:<60} ERROR: {res['error']}")
            continue
        corr = res["correlations"]
        n_sig = int(corr["fdr_significant"].sum()) if not corr.empty else 0
        click.echo(f"  {kin:<60} n_ath={res['n_athletes']:>3} · "
                   f"{len(corr):>3} predictors · {n_sig} FDR-sig")


@research_grp.command("session-change")
@click.option("--age-group", default=None,
              help="Restrict trials + profiles to one age group. Answers "
                   "stratum-specific 'what changed in HIGH SCHOOLERS over their "
                   "2 sessions' etc.")
@click.option("--min-span-days", default=30, type=int)
@click.option("--max-span-days", default=730, type=int)
@click.option("--max-profile-gap-days", default=90, type=int,
              help="Max days between session date and its matched profile date. "
                   "Larger allows more athletes but assessment→session link is fuzzier.")
@click.option("--aggregation", type=click.Choice(["mean", "median"]),
              default="mean")
@click.option("--min-n", default=5, type=int,
              help="Min paired athletes for a predictor correlation. Default 5 "
                   "matches the ~14-athlete session-change cohort where profile "
                   "keys are sparse per-column.")
@click.option("--target", "-t", multiple=True,
              help="Delta target(s) to correlate everything else against. "
                   "Default: delta_velocity_mph + auto-detected GRF/impulse deltas. "
                   "Repeatable, e.g. -t delta_velocity_mph -t fm_lead_peak_vertical_bw_delta.")
@click.option("--predictor-domain",
              type=click.Choice(["all", "assessment_only", "assessment_delta",
                                 "assessment_baseline", "kinematic_only"]),
              default="assessment_baseline",
              help="Restrict what counts as a predictor. Default "
                   "'assessment_baseline' correlates each athlete's BASELINE "
                   "mob_/screen_/proteus_ Z-score against their kinematic/velocity "
                   "DELTA (needs only 1 profile per athlete → usually the most "
                   "usable mode). 'assessment_delta' (a.k.a. legacy "
                   "'assessment_only') needs 2 distinct profiles per athlete "
                   "(rare — most athletes have only one). 'all' includes "
                   "cross-force/kinematic correlations. 'kinematic_only' "
                   "restricts to kin_/fm_ deltas (mostly autocorrelation).")
@click.option("--cross-target-min-hits", default=2, type=int,
              help="Min number of target hits for a predictor to appear in the "
                   "cross-target summary.")
def research_session_change_cmd(
    age_group: str | None, min_span_days: int, max_span_days: int,
    max_profile_gap_days: int, aggregation: str, min_n: int,
    target: tuple[str, ...],
    predictor_domain: str,
    cross_target_min_hits: int,
) -> None:
    """Session-to-session delta correlations.

    For athletes with 2+ pitching-3D sessions AND 2+ profiles, compute the
    session1→session2 change in velocity, kinematics, force-plate metrics, AND
    assessment Z-scores. Then correlate the deltas across athletes.

    Example:
        # Default: velocity + GRF targets, all age groups
        python -m src.main research session-change

        # Only PRO cohort, specific target
        python -m src.main research session-change --age-group PRO \\
            -t delta_velocity_mph -t fm_lead_peak_vertical_bw_delta
    """
    click.echo(f"[session-change] loading trials + profiles...")
    delta_df = _r_schange.build_session_delta_frame(
        min_span_days=min_span_days,
        max_span_days=max_span_days,
        age_group=age_group,
        aggregation=aggregation,
        max_profile_gap_days=max_profile_gap_days,
    )
    if delta_df.empty:
        click.echo("No athletes with 2+ session windows in range.")
        return
    click.echo(f"  {len(delta_df)} athletes with paired sessions")
    # Profile coverage diagnostics — tells you when a domain will be data-starved.
    n_baseline = int(delta_df.get("_has_baseline_profile", False).sum()) \
        if "_has_baseline_profile" in delta_df.columns else 0
    n_delta = int(delta_df.get("_has_delta_profile", False).sum()) \
        if "_has_delta_profile" in delta_df.columns else 0
    n_asmt_delta_cols = sum(1 for c in delta_df.columns
                            if c.endswith("_delta")
                            and _r_schange._is_assessment_delta(c))
    n_asmt_base_cols = sum(1 for c in delta_df.columns
                           if c.endswith("_baseline")
                           and _r_schange._is_assessment_baseline(c))
    click.echo(f"  profile coverage: {n_baseline} w/ baseline profile · "
               f"{n_delta} w/ two DISTINCT profiles (delta)")
    click.echo(f"  assessment predictor cols: {n_asmt_base_cols} baseline · "
               f"{n_asmt_delta_cols} delta")
    if predictor_domain in ("assessment_only", "assessment_delta") and n_delta < min_n:
        click.echo(f"  WARNING: only {n_delta} athletes have 2 distinct profiles — "
                   f"below --min-n {min_n}. Consider "
                   f"'--predictor-domain assessment_baseline' instead.")
    if predictor_domain == "assessment_baseline" and n_baseline < min_n:
        click.echo(f"  WARNING: only {n_baseline} athletes have any profile — "
                   f"below --min-n {min_n}.")

    targets = list(target) if target else None
    correlations = _r_schange.correlate_deltas(
        delta_df, targets=targets, min_n=min_n,
        predictor_domain=predictor_domain,
    )
    # Cross-target aggregate — which predictor hits the most force targets consistently?
    cross_summary = _r_schange.cross_target_predictor_summary(
        correlations, min_hits=cross_target_min_hits,
    )
    # Density-drop diagnostic
    drop = _r_schange._last_predictor_density_drop
    if drop.get("before", 0) > 0 and drop["before"] > drop["after"]:
        click.echo(f"  density filter: {drop['before']} → {drop['after']} predictor cols "
                   f"kept (min_n={drop['min_n']}); {drop['before'] - drop['after']} "
                   f"dropped as too sparse — this is the usual reason profile-based "
                   f"predictors return 0. Consider --min-n 4 or expand the profile "
                   f"backfill.")
    path = _r_reports.render_session_change_report(
        delta_df, correlations, aggregation=aggregation,
        cross_target_summary=cross_summary,
        predictor_domain=predictor_domain,
    )
    if not cross_summary.empty:
        click.echo(f"  cross-target summary: {len(cross_summary)} predictor(s) "
                   f"hit ≥{cross_target_min_hits} target(s)")
        top5 = cross_summary.head(5)
        for _, row in top5.iterrows():
            click.echo(f"    {row['predictor']:<50} hits {row['n_targets_hit']:>2} "
                       f"targets · consistency={row['direction_consistency']:.0%} · "
                       f"mean_r={row['mean_r']:+.2f}")
    click.echo(f"Session-change report: {path}")
    for tgt, res in correlations.items():
        n_sig = int(res["fdr_significant"].sum()) if not res.empty else 0
        click.echo(f"  target {tgt:<55} · {len(res):>3} predictors · {n_sig} FDR-sig")


@research_grp.command("athlete-deep-dive")
@click.argument("athlete_query")
@click.option("--focus-session", "focus_session", default=None,
              help="YYYY-MM-DD of the session to focus. Defaults to latest.")
@click.option("--outlier-p", "outlier_percentile", default=90.0, type=float,
              help="Percentile threshold for cohort-outlier flags. "
                   "e.g. 90 flags athlete's metrics >p90 or <p10.")
@click.option("--min-n-cohort", default=6, type=int,
              help="Min athletes in cohort to compute a percentile.")
@click.option("--top-k-velo-drivers", default=10, type=int,
              help="Top-K kin↔velo correlates to show for the age group.")
def research_athlete_deep_dive_cmd(
    athlete_query: str, focus_session: str | None,
    outlier_percentile: float, min_n_cohort: int, top_k_velo_drivers: int,
) -> None:
    """Comprehensive per-athlete research package for one athlete.

    Sections:
      1. Fingerprint — data on record, current Z-score profile
      2. Trial variability — CV + within-session velo correlations for focus session
      3. Session change — first→latest deltas, joined against age-group velo drivers
      4. Outlier flags — where athlete sits vs cohort, with candidate assessment levers
      5. Velo drivers — top age-group kin↔velo correlates, personalized

    Examples:
        python -m src.main research athlete-deep-dive "joey@8ctanebaseball.com"
        python -m src.main research athlete-deep-dive "Carson Crider"
        python -m src.main research athlete-deep-dive <uuid> --focus-session 2025-11-19
    """
    from datetime import date
    focus_date = None
    if focus_session:
        try:
            focus_date = date.fromisoformat(focus_session)
        except ValueError:
            click.echo(f"[athlete-deep-dive] bad --focus-session {focus_session!r}; "
                       f"expected YYYY-MM-DD.")
            return

    click.echo(f"[athlete-deep-dive] resolving athlete: {athlete_query!r}...")
    try:
        report = _r_dive.run_athlete_deep_dive(
            athlete_query,
            focus_session_date=focus_date,
            outlier_percentile=outlier_percentile,
            min_n_cohort=min_n_cohort,
            top_k_velo_drivers=top_k_velo_drivers,
        )
    except ValueError as e:
        click.echo(f"[athlete-deep-dive] {e}")
        return

    ath = report.athlete
    click.echo(f"  {ath.name} ({ath.age_group or '?'}) · "
               f"{len(ath.sessions)} pitching sessions · "
               f"{len(ath.profile_df)} profiles")
    click.echo(f"  focus session: {ath.focus_session_date}")

    _echo_deep_dive_summary(report)

    path = _r_reports.render_athlete_deep_dive_report(report)
    click.echo(f"Report: {path}")
    for w in report.warnings:
        click.echo(f"  WARN: {w}")


# ══════════════════════════════════════════════════════════════════════════
# Research v2 — coach-facing output, cohort job, reliability, dictionary
# ══════════════════════════════════════════════════════════════════════════

def _echo_deep_dive_summary(report) -> None:
    """The terminal view: the findings, not the section internals. If the
    headline list is empty that is itself the news, so say it."""
    s = report.summary or {}
    if report.headlines:
        click.echo(f"  {len(report.headlines)} finding(s):")
        for f in report.headlines:
            mark = {
                "beyond_typical": "[big for him]",
                "at_edge": "[typical]",
                "within_typical": "[small]",
                "no_baseline": "[no baseline]",
            }.get(f.verdict, "")
            prov = " (provisional band)" if getattr(f, "provisional", False) else ""
            click.echo(f"    {mark:9s} {f.headline}{prov}")
    else:
        click.echo("  no findings cleared the bar — see the report for what "
                   "was looked at")
    if s.get("n_joined"):
        click.echo(f"  {s.get('aligned_drivers', 0)}/{s['n_joined']} velocity-"
                   f"relevant mechanics moved the right way by more than noise")


def _apply_render_flags(offline: bool) -> None:
    from src.research import render_kit
    render_kit.set_plotly_mode("inline" if offline else "cdn")


@research_grp.command("coach-report")
@click.argument("athlete_query")
@click.option("--focus-session", default=None,
              help="YYYY-MM-DD of the session to focus. Defaults to latest.")
@click.option("--offline", is_flag=True,
              help="Embed the chart library so the page works with no internet "
                   "(bigger file, survives being emailed around).")
@click.option("--save/--no-save", default=True,
              help="Persist the structured findings to ai_layer.research_findings.")
@click.option("--open-previous", is_flag=True,
              help="Also print what changed since the last stored report.")
def research_coach_report_cmd(athlete_query, focus_session, offline, save,
                              open_previous):
    """Coach-facing one-pager for a single athlete.

    Leads with the findings that survived every filter, draws the three charts
    that answer 'did it move / is he good at it / where does he sit', and puts
    the full analysis in a collapsed appendix.

        python -m src.main research coach-report "Carson Crider"
        python -m src.main research coach-report joey@8ctanebaseball.com --offline
    """
    from datetime import date as _date
    from src.research import findings_store
    from src.research.coach_report import render_coach_report

    _apply_render_flags(offline)
    focus = None
    if focus_session:
        try:
            focus = _date.fromisoformat(focus_session)
        except ValueError:
            click.echo(f"bad --focus-session {focus_session!r}; expected YYYY-MM-DD")
            return
    try:
        report = _r_dive.run_athlete_deep_dive(athlete_query,
                                               focus_session_date=focus)
    except ValueError as e:
        click.echo(str(e))
        return

    ath = report.athlete
    click.echo(f"{ath.name} ({ath.age_group or '?'}) · {len(ath.sessions)} "
               f"session(s) · focus {ath.focus_session_date}")
    _echo_deep_dive_summary(report)

    if open_previous:
        diff = findings_store.diff_against_previous(report)
        if diff.get("has_previous"):
            click.echo(f"  since {diff['previous_generated_at']:%Y-%m-%d}: "
                       f"{len(diff['new'])} new, {len(diff['resolved'])} resolved, "
                       f"{len(diff['persisting'])} still there")
        else:
            click.echo("  no previous stored report to compare against")

    path = render_coach_report(report)
    click.echo(f"Coach report: {path}")

    if save:
        row_id = findings_store.save_deep_dive(report,
                                               output_paths={"coach": str(path)})
        if row_id:
            click.echo(f"Saved findings as ai_layer.research_findings id={row_id}")

    for w in report.warnings:
        click.echo(f"  WARN: {w}")


@research_grp.command("roster")
@click.option("--age-group", default=None, help="Limit to one level.")
@click.option("--limit", default=None, type=int, help="Show only the top N.")
@click.option("--min-sessions", default=1, type=int)
@click.option("--offline", is_flag=True, help="Embed the chart library.")
@click.option("--kpi", "kpis", multiple=True,
              help="Override the KPI columns (repeatable).")
def research_roster_cmd(age_group, limit, min_sessions, offline, kpis):
    """Squad overview — every athlete, sorted by who needs looking at.

        python -m src.main research roster
        python -m src.main research roster --age-group PRO --offline
    """
    from src.research.roster import build_roster, render_roster_report

    _apply_render_flags(offline)
    roster = build_roster(age_group=age_group, limit=limit,
                          min_sessions=min_sessions,
                          kpis=list(kpis) or None)
    click.echo(f"{len(roster.rows)} athlete(s) · {roster.stratum}")
    for r in roster.rows[:10]:
        click.echo(f"  {r.name:<24s} {r.attention:5.1f}  "
                   f"{r.attention_reason or '—'}")
    for n in roster.notes:
        click.echo(f"  note: {n}")
    click.echo(f"Roster: {render_roster_report(roster)}")


@research_grp.command("cohort-refresh")
@click.option("--strata", multiple=True,
              help="Age groups to refresh (default PRO/COLLEGE/HIGH SCHOOL/YOUTH).")
@click.option("--top-k", default=5, type=int,
              help="Velocity correlates per stratum to chain to assessments.")
@click.option("--dry-run", is_flag=True, help="Compute but do not persist.")
def research_cohort_refresh_cmd(strata, top_k, dry_run):
    """Recompute the group-level statistics. Run this nightly.

    Reliability, per-stratum velocity correlates and the assessment chain are
    computed once here and stored, instead of being recomputed inside every
    per-athlete report where they could silently disagree between runs.
    """
    from src.research.cohort_job import refresh_cohort_findings
    res = refresh_cohort_findings(strata=list(strata) or None,
                                  top_k_chain=top_k, persist=not dry_run)
    click.echo(res.describe())
    if dry_run:
        click.echo("  (dry run — nothing written)")


@research_grp.command("reliability")
@click.option("--metric", default=None, help="Show one metric in detail.")
@click.option("--gaps", is_flag=True,
              help="Only list the measures still missing a test-retest estimate.")
@click.option("--save/--no-save", default=False,
              help="Persist to ai_layer.metric_reliability.")
def research_reliability_cmd(metric, gaps, save):
    """How much does each measurement wobble when nothing changed?

    Everything that says "this is a real change" resolves through this table,
    so it is worth looking at directly. `--gaps` prints the measures that need
    repeat captures before their change verdicts can be trusted — the single
    highest-value data-collection job available.
    """
    from src.research import findings_store, loaders
    from src.research.config import CONFIG as _CFG
    from src.research.metric_display import DISPLAY as _D

    table = loaders.reliability_table()
    if len(table) == 0:
        click.echo("No reliability estimates could be computed. That usually "
                   "means too few sessions with enough trials each "
                   f"(need {_CFG.reliability_min_sessions} sessions of "
                   f"{_CFG.reliability_min_trials_per_session}+ trials).")
        return

    summ = table.summary()
    click.echo(f"{summ['n_metrics']} metrics · "
               f"{summ['n_between_session']} with real test-retest data · "
               f"median ICC {summ.get('median_icc') and round(summ['median_icc'], 2)}")

    g = table.reliability_gaps()
    if gaps or g["needs_retest"]:
        click.echo(f"\n{len(g['needs_retest'])} measure(s) have no repeat-capture "
                   f"data. Their change verdicts use a provisional band that "
                   f"leans generous:")
        for m in g["needs_retest"][:40]:
            click.echo(f"    {_D.short(m)}  ({m})")
        if len(g["needs_retest"]) > 40:
            click.echo(f"    ... and {len(g['needs_retest']) - 40} more")
        click.echo("\n  Fix: capture a handful of athletes twice inside two "
                   "weeks. Every metric those captures cover gains a real "
                   "band for every future report.")
    if gaps:
        return

    if metric:
        r = table.get(metric)
        if r is None:
            click.echo(f"No estimate for {metric!r}.")
            return
        click.echo(f"\n{_D.name(metric)}  ({metric})")
        click.echo(f"  typical error   {r.sem:.4g} {_D.unit(metric)}")
        click.echo(f"  MDC95 (1 trial) {r.mdc95_single:.4g} {_D.unit(metric)}")
        click.echo(f"  ICC             {r.icc if r.icc is None else round(r.icc, 3)}")
        click.echo(f"  from            {r.source} · {r.n_sessions} sessions · "
                   f"{r.n_athletes} athletes")
        return

    df = table.to_frame().sort_values("icc", ascending=False, na_position="last")
    click.echo("\nmetric                                   SEM      MDC95    ICC   source")
    for _, r in df.head(40).iterrows():
        icc = "  —  " if pd.isna(r["icc"]) else f"{r['icc']:.2f} "
        click.echo(f"{r['metric'][:40]:<40s} {r['sem']:>8.3g} "
                   f"{r['mdc95_single']:>8.3g} {icc} {r['source']}")

    if save:
        n = findings_store.save_reliability(
            table.to_frame(), config_hash=_CFG.fingerprint(),
            analysis_version=_CFG.analysis_version)
        click.echo(f"\nSaved {n} rows to ai_layer.metric_reliability")


@research_grp.command("metric-coverage")
@click.option("--status", type=click.Choice(["all", "raw", "auto", "curated"]),
              default="all")
@click.option("--limit", default=60, type=int)
def research_metric_coverage_cmd(status, limit):
    """Which metrics have a coach-facing name, and which still need one.

    Only curated metrics are allowed to appear on a coach page with plain
    language, so this is the naming backlog. It is ranked by how often each
    key actually shows up in the warehouse, so the work happens in impact
    order rather than alphabetically.
    """
    from src.research import loaders
    from src.research.metric_display import DISPLAY as _D
    from src.research.pitching_deep import metric_columns_pitching

    trials = loaders.trials(copy=False)
    if trials.empty:
        click.echo("No trial data.")
        return
    cols = metric_columns_pitching(trials, processed_only=False,
                                   exclude_symptomatic=False, role="pitcher")
    # Weight by how many trials actually carry a value — a key that is null
    # everywhere is not worth naming.
    weights = {c: int(trials[c].notna().sum()) for c in cols}

    prof = loaders.profiles(copy=False)
    if not prof.empty:
        from src.research.profile_matrix import metric_columns as _pm
        for c in _pm(prof):
            weights[c] = int(prof[c].notna().sum())

    cov = _D.coverage(weights.keys(), weights=weights)
    if status != "all":
        cov = cov[cov["status"] == status]
    counts = _D.coverage(weights.keys())["status"].value_counts().to_dict()
    click.echo(f"curated {counts.get('curated', 0)} · "
               f"auto {counts.get('auto', 0)} · raw {counts.get('raw', 0)}")
    click.echo("\n(ranked by how much data actually carries the metric)\n")
    for _, r in cov.head(limit).iterrows():
        click.echo(f"  {r['status']:<8s} {r['weight']:>7d}  {r['metric'][:52]:<52s} "
                   f"{r['display_name'][:44]}")
    needs = _D.needs_review(weights.keys())
    if needs:
        click.echo(f"\n{len(needs)} curated metric(s) still have "
                   f"direction: review — nobody has decided whether higher is "
                   f"better: {', '.join(needs[:12])}")


@research_grp.command("diagnose")
@click.argument("athlete_query")
def research_diagnose_cmd(athlete_query):
    """Why isn't this athlete's newest session showing up?

    This repo only READS the warehouse — `public.f_*` is written by the UAIS
    runners in OctaneBiomechBackend, so no command here can pull in a capture
    that was never ingested. This tells you which link in the chain broke:
    wrong Neon branch, duplicate athlete record, never ingested, or ingested
    with a problem.

        python -m src.main research diagnose "Ryan Chasse"
    """
    from src.research.diagnose import diagnose_athlete, format_diagnosis
    try:
        d = diagnose_athlete(athlete_query)
    except Exception as e:
        click.echo(f"diagnose failed: {e}")
        return
    click.echo(format_diagnosis(d))


@research_grp.command("unit-audit")
@click.argument("athlete_query")
@click.option("--limit", default=40, type=int, help="Rows to print.")
def research_unit_audit_cmd(athlete_query, limit):
    """Show every scale break the guard found, with the numbers behind it.

    The guard withholds comparisons when it believes the pipeline changed
    units. That is the right call for a real unit change and infuriating for a
    false positive, so this prints the evidence: which metrics, which
    boundary, what factor, and the session means on both sides — so you can
    read them yourself and decide whether the guard is right.

        python -m src.main research unit-audit "Ryan Chasse"
    """
    from src.research import loaders
    from src.research.athlete_deep_dive import resolve_athlete
    from src.research.pitching_deep import metric_columns_pitching
    from src.research.unit_guard import audit_scale_changes

    uuid, name = resolve_athlete(athlete_query)
    df = loaders.trials_for_athlete(uuid)
    if df.empty:
        click.echo(f"{name}: no pitching trials on file.")
        return
    metrics = metric_columns_pitching(df, processed_only=True,
                                      exclude_symptomatic=False, role="pitcher")
    audit = audit_scale_changes(df, metrics)
    means = df.groupby("session_date")[metrics].mean()

    click.echo(f"{name} — {len(metrics)} metrics across "
               f"{means.shape[0]} sessions\n")
    if not audit.has_findings:
        click.echo("No scale breaks found. Nothing is being withheld.")
        return

    for sysb in audit.systemic:
        click.echo(f"WITHHELD — {len(sysb.metrics)} metric(s) {sysb.direction} "
                   f"by ~{sysb.median_ratio:,.0f}x between {sysb.from_date} "
                   f"and {sysb.to_date}"
                   + (f" ({sysb.likely_cause})" if sysb.likely_cause else ""))
        for m in sysb.metrics[:limit]:
            a = means[m].get(sysb.from_date)
            b = means[m].get(sysb.to_date)
            click.echo(f"    {m:<62} {a:>14,.4g} -> {b:>14,.4g}")
        click.echo("")

    singles = [b for b in audit.breaks
               if b.metric not in {m for s in audit.systemic for m in s.metrics}]
    if singles:
        click.echo(f"REPORTED ONLY — {len(singles)} metric(s) moved >20x alone. "
                   f"These are still compared; one metric moving by itself is "
                   f"usually a value that passed near zero.")
        for b in singles[:limit]:
            a = means[b.metric].get(b.from_date)
            c = means[b.metric].get(b.to_date)
            click.echo(f"    {b.metric:<62} {a:>14,.4g} -> {c:>14,.4g}"
                       f"  ({b.ratio:,.0f}x {b.direction})")


@research_grp.command("apply-migration")
def research_apply_migration_cmd():
    """Create the research persistence tables (sql/002_research_findings.sql).

    Idempotent. Without it the analysis still runs; it just cannot keep an
    audit trail, and `findings_store` degrades to a no-op.
    """
    from pathlib import Path as _P
    from src.db import backend_conn as _conn
    sql_path = _P(__file__).resolve().parents[1] / "sql" / "002_research_findings.sql"
    if not sql_path.exists():
        click.echo(f"Missing {sql_path}")
        return
    sql = sql_path.read_text(encoding="utf-8")
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
    click.echo(f"Applied {sql_path.name}")


if __name__ == "__main__":
    cli()
