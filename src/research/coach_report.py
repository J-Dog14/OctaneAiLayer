"""
The coach one-pager.

The old deep-dive HTML was 511 table rows and zero charts. No coach reads
that, so the analysis may as well not have run. This renderer inverts the
pyramid:

  Page 1   five sentences and three charts — what changed, is it real, what
           he is good at, what is holding him back, what to check next.
  Below    the same content as evidence: the numbers behind each sentence.
  Appendix everything else, collapsed, for whoever wants to audit it.

Rules it follows:
  * Every claim carries a confidence marker, and the marker means the same
    thing on every page (see `reliability.VERDICT_*`).
  * No metric appears with a plain-language name unless a human curated that
    name in metric_display.yaml. Machine-composed names stay in the appendix.
  * Nothing is coloured good or bad unless the dictionary says the metric has
    a good direction. Arm-load metrics are never coloured.
  * Every measurement is taken as fact. Numbers are accompanied by the
    comparison that gives them scale — how much that measure usually varies
    for this athlete, or the group he sits in — never by a claim that a
    recorded difference did not happen.
"""
from __future__ import annotations

import re

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.research import render_kit as rk
from src.research.config import CONFIG
from src.research.metric_display import DISPLAY
from src.research.reliability import (
    VERDICT_LIKELY,
    VERDICT_NOISE,
    VERDICT_REAL,
    VERDICT_UNKNOWN,
)


def _unit_break_note(report) -> str:
    """Scale/unit discontinuities go at the very top, above the findings.

    A unit change invalidates comparisons rather than qualifying them, so it
    cannot sit in a caveats list at the bottom of the page.
    """
    audit = getattr(report.athlete, "unit_audit", None)
    if audit is None or not audit.has_findings:
        return ""
    return rk.note(
        "Some measurements changed scale between captures — this is a data "
        "problem, not the athlete:", items=audit.summary_lines())


def _provisional_note(report) -> str:
    n = (report.summary or {}).get("n_provisional") or 0
    if not n:
        return ""
    return rk.note(
        f"{n} of the calls above compare against a provisional baseline. We do "
        f"not yet have repeat captures for those measures, so 'usual variation' "
        f"is estimated from his pitch-to-pitch spread rather than his "
        f"day-to-day spread. Day to day is wider, so those moves look larger "
        f"here than they will once the baseline is real. Capturing a few "
        f"athletes twice inside two weeks fixes it for every future report.")


def _tone(good: int | None) -> str:
    return {1: "good", -1: "bad"}.get(good, "neutral")


def _kind_heading(kind: str) -> str:
    return {
        "change": "What changed",
        "standout": "What he does well",
        "limiter": "What is holding him back",
        "profile": "Where he is unusual",
        "variability": "Worth watching",
        "load": "Arm load",
    }.get(kind, kind.title())


def render_coach_report(report, *, output_dir: Path | None = None) -> Path:
    """One athlete, one page. `report` is a DeepDiveReport."""
    ath = report.athlete
    cfg = report.config
    sc = report.session_change if isinstance(report.session_change, dict) else {}
    o = report.outliers if isinstance(report.outliers, dict) else {}
    v = report.variability if isinstance(report.variability, dict) else {}
    fp = report.fingerprint if isinstance(report.fingerprint, dict) else {}

    body: list[str] = []
    first_chart = True   # only the first figure pulls plotly.js

    # ── Header ────────────────────────────────────────────────────────────
    bits = []
    if ath.age_group:
        bits.append(ath.age_group)
    if ath.handedness:
        bits.append(f"{ath.handedness}HP")
    if ath.focus_session_date is not None:
        bits.append(f"Session {ath.focus_session_date}")
    if fp.get("n_pitching_sessions"):
        bits.append(f"{fp['n_pitching_sessions']} sessions on file")
    body.append(rk.page_header(
        ath.name, bits,
        f"Generated {report.generated_at:%Y-%m-%d %H:%M} · {cfg.describe()}",
    ))

    # ── Verdict banner ────────────────────────────────────────────────────
    body.append(_banner(report, sc))
    body.append(_unit_break_note(report))

    # ── The findings ──────────────────────────────────────────────────────
    if report.headlines:
        order = ["change", "limiter", "standout", "variability", "profile", "load"]
        by_kind: dict[str, list] = {}
        for f in report.headlines:
            by_kind.setdefault(f.kind, []).append(f)
        for kind in order:
            items = by_kind.get(kind)
            if not items:
                continue
            body.append(f"<h2>{rk.esc(_kind_heading(kind))}</h2>")
            body.append("<div class='cards'>")
            for f in items:
                body.append(rk.finding_card(
                    f.headline, f.detail, tone=_tone(f.good),
                    verdict=f.verdict if f.verdict != VERDICT_UNKNOWN else None,
                    cue=f.cue,
                ))
            body.append("</div>")
    else:
        body.append("<h2>Findings</h2>")
        body.append(rk.note(
            "Nothing this session was large enough relative to his own usual "
            "variation to lead with. Every measurement is still below — the "
            "evidence section has the per-pitch values, the movement changes "
            "and where he sits in the group."))

    body.append(_provisional_note(report))

    # ── Charts ────────────────────────────────────────────────────────────
    body.append("<h2>The picture</h2>")

    sessions = fp.get("sessions")
    if isinstance(sessions, pd.DataFrame) and not sessions.empty:
        mdc_velo = ath.reliability.mdc_for_means(
            "velocity_mph",
            n_before=int(sessions["n_trials"].median() or 1),
            n_after=int(sessions["n_trials"].iloc[-1] or 1),
        )
        body.append(rk.chart_velocity_trend(sessions, mdc=mdc_velo,
                                            first=first_chart))
        first_chart = False

    sm = sc.get("matrix")
    if sm is not None and not sm.frame.empty and sm.has_comparison:
        movers = sm.movers(10).copy()
        if not movers.empty:
            movers["from_value"] = movers[f"s_{sm.previous}"]
            movers["to_value"] = movers[f"s_{sm.latest}"]
            movers["delta"] = movers["change"]
            movers["mdc"] = movers["usual_spread"]
            movers["mdc_ratio"] = movers["spread_multiple"]
            body.append(rk.chart_change_vs_noise(movers, first=first_chart))
            first_chart = False
            body.append(
                f"<div class='legend'>Comparing {sm.previous} to {sm.latest}"
                + (f", {sm.days_between} days apart." if sm.days_between else ".")
                + "</div>")
        if sm.days_between and sm.days_between > CONFIG.long_span_warn_days:
            body.append(rk.note(
                f"These two captures are {sm.days_between} days apart — long "
                f"enough that a training block, a season and (for a younger "
                f"athlete) growth all sit inside the difference. It describes "
                f"what is different, not what any one thing did."))

    strip_rows = _cohort_strip_rows(ath, o, limit=6)
    if strip_rows:
        body.append(rk.chart_cohort_strip(strip_rows, first=first_chart))
        first_chart = False

    credible = v.get("credible_correlates")
    if isinstance(credible, pd.DataFrame) and not credible.empty:
        m = credible.iloc[0]["metric"]
        focus = ath.trials_on(ath.focus_session_date)
        body.append(rk.chart_trial_scatter(focus, m, first=first_chart))
        first_chart = False

    # ── Evidence ──────────────────────────────────────────────────────────
    body.append("<h2>The evidence</h2>")
    body.append(_how_to_read())
    body.append(_evidence_changes(sc))
    body.append(_evidence_assessments(sc))
    body.append(_evidence_position(o, cfg))
    body.append(_evidence_coverage(report))
    body.append(_evidence_within_session(v, cfg))
    body.append(_evidence_alignment(sc))

    # ── What we could not say ─────────────────────────────────────────────
    body.append("<h2>What this report could not tell you</h2>")
    limits = _limitations(report, sc, o, v, cfg)
    if limits:
        body.append(rk.note(
            "Stated plainly so nobody has to infer it from a missing table:",
            items=limits))
    else:
        body.append(rk.empty("No coverage gaps worth flagging."))

    if report.warnings:
        body.append(rk.details(
            f"Data caveats ({len(report.warnings)})",
            "<ul>" + "".join(f"<li>{rk.esc(w)}</li>" for w in report.warnings)
            + "</ul>"))

    # ── Analyst appendix ──────────────────────────────────────────────────
    body.append("<h2>Analyst appendix</h2>")
    body.append(_appendix(report, sc, o, v))

    filename = (f"coach_report_{_slug(ath.name)}_"
                f"{rk.timestamp_slug()}.html")
    return rk.write_html("\n".join(body), filename,
                         f"{ath.name} — coach report", output_dir=output_dir)


# ──────────────────────────────────────────────────────────────────────────
# Pieces
# ──────────────────────────────────────────────────────────────────────────

def _banner(report, sc: dict) -> str:
    dv = sc.get("delta_velocity_first")
    verdict = sc.get("delta_velocity_first_verdict")
    span = sc.get("span_days")
    fp = report.fingerprint or {}
    latest = fp.get("latest_session_mean_velocity")
    n_tr = fp.get("latest_session_n_trials")

    left = (f"{latest:.1f} mph average this session"
            if latest is not None else "No velocity on record this session")
    if n_tr:
        left += f" over {n_tr} pitches"

    if dv is None or verdict is None:
        return rk.banner(left, "Not enough history yet to say which way he is "
                               "trending.", tone="flat")
    if verdict.verdict == VERDICT_NOISE:
        return rk.banner(
            left,
            f"Holding steady — the {dv:+.1f} mph difference from "
            f"{sc.get('round1_id')} is inside normal session-to-session "
            f"variation.", tone="flat")
    tone = "good" if dv > 0 else "bad"
    word = "up" if dv > 0 else "down"
    span_txt = f" across {span} days" if span else ""
    return rk.banner(
        f"{left} — {word} {abs(dv):.1f} mph{span_txt}",
        verdict.explain(), tone=tone)


def _cohort_strip_rows(ath, o: dict, *, limit: int = 6) -> list[dict]:
    """Pick the metrics worth drawing a strip for: curated, reliable, and
    actually flagged — falling back to the most extreme ones if nothing was
    flagged."""
    allm = o.get("all_metrics")
    if not isinstance(allm, pd.DataFrame) or allm.empty:
        return []
    from src.research import loaders
    from src.research.pitching_deep import metric_columns_pitching

    cand = allm[allm["metric"].map(DISPLAY.is_coach_ready)]
    if cand.empty:
        return []
    if "reliable" in cand.columns and cand["reliable"].any():
        cand = cand[cand["reliable"]]
    flagged = cand[cand["flag"].notna()]
    pick = flagged if not flagged.empty else cand.assign(
        _d=(cand["percentile"].fillna(50) - 50).abs()
    ).sort_values("_d", ascending=False)
    pick = pick.head(limit)

    stratum = (o.get("stratum") or {}).get("used")
    age_group = None if (o.get("stratum") or {}).get("fell_back") else \
        (o.get("stratum") or {}).get("requested")
    try:
        cohort = loaders.trials(age_group=age_group, copy=False)
    except Exception:
        return []
    if cohort.empty:
        return []

    rows: list[dict] = []
    for _, r in pick.iterrows():
        m = r["metric"]
        if m not in cohort.columns:
            continue
        per_session = cohort.groupby(["athlete_uuid", "session_date"])[m].mean()
        per_athlete = per_session.groupby("athlete_uuid").mean().dropna()
        rows.append({
            "metric": m,
            "athlete_value": float(r["athlete_value"]),
            "cohort_values": per_athlete.drop(index=ath.uuid, errors="ignore").tolist(),
            "cohort_label": stratum or "",
        })
    return rows


def _how_to_read() -> str:
    """One legend, once, because every table below shares a shape.

    The report used to change layout between sections — a wide pitch dump, a
    Before/After pair, a spread table — and a coach had to re-learn how to read
    each one. They are now the same table with different columns of history, so
    the instructions are written down a single time.
    """
    rows = [
        ("Metric", "on the left, with its unit underneath."),
        ("One column per capture", "oldest on the left, the most recent one "
         "in bold on the right. A blank cell means it was not measured then — "
         "it never means zero."),
        ("Change (last 2)", "the most recent column minus the one before it. "
         "Not a first-to-last difference with other captures hidden inside."),
        ("x usual", "that change divided by how much the measure normally "
         "moves for him. 1.0x is the size of his usual swing, 3x is three "
         "times it. Bold and coloured means bigger than his usual."),
        ("Trend", "the same row drawn as a line, so a one-off move and a "
         "steady drift look different at a glance."),
    ]
    lis = "".join(f"<li><b>{rk.esc(k)}</b> — {rk.esc(t)}</li>" for k, t in rows)
    return ("<div class='note'>How to read every table below — they all share "
            f"one shape.<ul>{lis}</ul></div>")


def _matrix_table(sm, sub: pd.DataFrame, *, count_label: str | None = "pitches",
                  context: str = "spread") -> str:
    """The canonical table: metric, one column per capture, then the change.

    Every table on the page uses this shape, so a coach learns one layout and
    reads the rest for free: name on the left, one dated column per capture
    oldest-to-newest, the last-two change, one column sizing that change, and a
    sparkline. ``context`` picks the sizing column — "spread" for pitching,
    where trial repeats give a usual swing to measure against, "pct" for
    assessments, which are single measurements with no within-round repeats.
    """
    if sub is None or sub.empty:
        return rk.empty("Nothing in this group.")
    dates = sm.session_dates
    latest = sm.latest
    by_pct = context == "pct"

    head = ["<th>Metric</th>"]
    for d in dates:
        emph = " style='color:#1c2024;font-weight:700'" if d == latest else ""
        n = sm.session_trials.get(d, 0) if count_label else 0
        sub_head = (f"<br><span style='font-weight:400;text-transform:none;"
                    f"color:#8c959f'>{n} {count_label}</span>"
                    if count_label and n else "")
        head.append(f"<th class='num'{emph}>{rk.esc(d)}{sub_head}</th>")
    head.append("<th class='num'>Change<br>"
                "<span style='font-weight:400;text-transform:none'>last 2</span></th>")
    head.append("<th class='num'>% change</th>" if by_pct
                else "<th class='num'>x usual</th>")
    head.append("<th>Trend</th>")

    rows = []
    for _, r in sub.iterrows():
        cells = [f"<td>{rk.esc(r['display_name'])}"
                 f"<div style='font-size:11px;color:#8c959f'>"
                 f"{rk.esc(r.get('unit') or '')}</div></td>"]
        for d in dates:
            v = r.get(f"s_{d}")
            txt = "—" if v is None or pd.isna(v) else f"{v:,.4g}"
            emph = " style='font-weight:650'" if d == latest else ""
            cells.append(f"<td class='num'{emph}>{txt}</td>")

        if r.get("blocked"):
            why = str(r.get("blocked"))
            factor = ""
            m = re.search(r"about ([\d,]+)x", why)
            if m:
                factor = f"<div style='font-size:11px;font-weight:400'>" \
                         f"&times;{m.group(1)}</div>"
            cells.append(f"<td class='num' style='color:#b42318' "
                         f"title='{rk.esc(why)}'>scale changed{factor}</td>")
            cells.append("<td class='num'>—</td>")
        elif r.get("change") is None or pd.isna(r.get("change")):
            cells.append("<td class='num'>—</td>")
            cells.append("<td class='num'>—</td>")
        else:
            good = r.get("toward_better")
            pct_v = pd.to_numeric(pd.Series([r.get("pct_change")]),
                                  errors="coerce").iloc[0]
            if by_pct:
                # Assessments carry no test-retest band, so nothing here can be
                # called big or small against his own repeats. Colour shows the
                # direction; the bold is simply the larger moves, by percent.
                beyond = bool(pd.notna(pct_v) and abs(pct_v) >= 10.0)
                colour = {1: rk.GOOD, -1: rk.BAD}.get(good, rk.NEUTRAL)
            else:
                beyond = r.get("verdict") == VERDICT_REAL
                colour = ({1: rk.GOOD, -1: rk.BAD}.get(good, rk.NEUTRAL)
                          if beyond else rk.MUTED)
            weight = "700" if beyond else "500"
            cells.append(f"<td class='num' style='color:{colour};"
                         f"font-weight:{weight}'>{rk.esc(r.get('change_text'))}</td>")
            if by_pct:
                ctx_txt = "—" if pd.isna(pct_v) else f"{float(pct_v):+.1f}%"
            else:
                mult = r.get("spread_multiple")
                ctx_txt = ("" if mult is None or pd.isna(mult)
                           else f"{mult:.1f}x")
            cells.append(f"<td class='num' style='color:{colour}'>"
                         f"{ctx_txt}</td>")
        cells.append(f"<td>{r.get('trend') or ''}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")

    return ("<div class='scroll-x'><table><thead><tr>" + "".join(head)
            + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")


def _evidence_changes(sc: dict) -> str:
    """Movement across every 3D capture, compared latest-vs-previous.

    Two things this gets right that a first-to-last delta cannot:

      * the comparison is the last two captures, which is the question a coach
        has after a session — not a difference accumulated over two years with
        other captures hidden inside it
      * the earlier captures stay on the page as columns, so a move can be read
        against its own history. A metric that jumped at capture two and held
        looks nothing like one that moved for the first time last week, and a
        single delta cannot tell them apart.
    """
    from src.research.metric_filters import FAMILY_LABEL, FAMILY_ORDER

    sm = sc.get("matrix")
    head = "<h3>Movement across captures</h3>"
    if sm is None or sm.frame.empty:
        return head + rk.empty("No 3D captures to compare.")

    out = [head, f"<p>{sc.get('matrix_window', '')}</p>"]
    out.append(
        "<p class='subtitle'>Every value as measured, one column per capture, "
        "newest on the right. The last two columns size the most recent move "
        "against how much that measure usually varies for him. Sorted biggest "
        "move first.</p>")

    thin = sm.thin_captures()
    if thin:
        detail = "; ".join(f"{d}: {n:,} of {best:,}" for d, n, best in thin)
        out.append(rk.note(
            f"Some captures carry far fewer measures than others, so their "
            f"columns are mostly dashes — {detail}. That is a processing gap, "
            f"not a testing gap: those sessions were written to the warehouse "
            f"without their full metric set. Anything missing on one side of "
            f"the comparison cannot be differenced, so those rows show values "
            f"without a change."))

    by_family = sm.by_family()
    for fam in FAMILY_ORDER:
        sub = by_family.get(fam)
        if sub is None or sub.empty:
            continue
        out.append(f"<h4>{rk.esc(FAMILY_LABEL.get(fam, fam))}</h4>")
        out.append(_matrix_table(sm, sub.head(14)))

    if sm.blocked_pairs:
        out.append(rk.note(
            f"{len(sm.blocked_pairs)} metric(s) changed units or scale between "
            f"the last two captures, so their values are shown but not "
            f"differenced — see the note at the top of the page."))
    if sm.excluded:
        sample = list(sm.excluded)[:4]
        out.append(rk.note(
            f"{len(sm.excluded)} metric(s) are left out because they are fixed "
            f"by the marker set or the capture rig rather than measured from "
            f"the athlete — segment lengths, frame rate, duplicate unit "
            f"columns. Example: {', '.join(sample)}.", quiet=True))
    return "".join(out)


def _evidence_assessments(sc: dict) -> str:
    """Assessment history in the same shape as the movement tables.

    Same layout, same reading order, same place for the change — so a coach who
    has read the table above already knows how to read this one. An empty cell
    means he was not tested in that round; it never means zero.
    """
    from src.research.session_matrix import ASSESSMENT_LABEL

    history = sc.get("assessment_history") or {}
    meta = sc.get("assessment_delta_meta") or {}
    head = "<h3>Assessment history</h3>"

    if history:
        out = [head,
               "<p>Each test on its own dates — these are not tied to when he "
               "was captured in 3D, so every screen he has taken shows up "
               "here.</p>",
               "<p class='subtitle'>Same table shape as above. Raw measured "
               "values, not Z-scores: a Z-score is scored against a norms "
               "table that gets rebuilt in place, so part of any Z change is "
               "the population moving rather than the athlete. A blank cell "
               "means that measure was not taken that day. <b>Change</b> is "
               "the last two times that measure was actually taken, which is "
               "not always the last two columns.</p>"]
        # Some sources are only read for part of what they record. Saying so
        # next to the table stops it looking like missing data: an athlete who
        # is on Proteus twice a week has plenty of dates that carry none of
        # the movements we score.
        domain_note = {
            "proteus_pitcher": "Only the Shot Put and D2 Extension movements "
                               "are scored here, so a session where he did "
                               "neither is not a column.",
            "proteus_hitter":  "Only the Shot Put and Straight Arm Trunk "
                               "Rotation movements are scored here.",
        }
        for mod, sm in history.items():
            if sm is None or sm.frame.empty:
                continue
            label = ASSESSMENT_LABEL.get(mod, mod.replace("_", " ").title())
            n_dates = len(sm.session_dates)
            out.append(f"<h4>{rk.esc(label)} — {n_dates} test "
                       f"date{'s' if n_dates != 1 else ''}</h4>")
            if mod in domain_note:
                out.append(f"<p class='subtitle'>{domain_note[mod]}</p>")
            out.append(_matrix_table(sm, sm.frame.head(16), count_label=None,
                                     context="pct"))
        out.append(rk.note(
            "These have no test-retest band — one measurement per day, so "
            "there is nothing to size the change against beyond percent. "
            "Repeating a screen inside a week or two on a few athletes would "
            "give every one of these a real 'is this bigger than his usual' "
            "number, the way the pitching metrics have one.", quiet=True))
        return "".join(out)

    # ── Fallback: the old 3D-anchored matrix ──────────────────────────────
    am = sc.get("assessment_matrix")
    if am is None or getattr(am, "frame", None) is None or am.frame.empty:
        reason = meta.get("error") or (
            "No assessment values are on file for him in "
            "ai_layer.athlete_profiles, so there is nothing to show here. "
            "That is a profile-table gap, not an absence of testing — check "
            "the coverage section below, then rebuild profiles.")
        return head + rk.note(reason, quiet=True)

    frame = am.frame
    curated = frame[frame["metric"].map(DISPLAY.is_coach_ready)]
    show = curated if not curated.empty else frame
    # Rows he was never tested on twice add columns of dashes and nothing else.
    tested = show[show[[f"s_{d}" for d in am.session_dates]]
                  .notna().sum(axis=1) >= 1]
    show = tested if not tested.empty else show

    out = [head]
    if am.latest and am.previous:
        gap = f" ({am.days_between} days apart)" if am.days_between else ""
        out.append(f"<p>Change is <strong>{rk.esc(am.previous)} &rarr; "
                   f"{rk.esc(am.latest)}</strong>{gap}. Earlier rounds are "
                   f"shown so the move can be read against its history.</p>")
    out.append(
        "<p class='subtitle'>Raw measured values, not Z-scores — Z-scores are "
        "scored against a norms table that gets rebuilt in place, so part of "
        "any Z change is the population moving rather than the athlete. A "
        "blank cell means he was not tested that round; a value carried "
        "forward from an older test is left blank rather than shown as a "
        "re-test. Sorted biggest move first.</p>")
    out.append(_matrix_table(am, show.head(18), count_label=None,
                             context="pct"))
    out.append(rk.note(
        "There is no test-retest band for these, so the change is sized by "
        "percent rather than against his own repeats. Two assessments a couple "
        "of weeks apart would fix that.", quiet=True))

    dropped = (len(meta.get("excluded_stale", []))
               + len(meta.get("excluded_same_source", [])))
    if dropped:
        out.append(rk.note(
            f"{dropped} assessment metric(s) were left out of the change "
            f"because the value was carried forward rather than re-measured. "
            f"Differencing a carry-forward against a fresh test manufactures "
            f"change.", quiet=True))
    return "".join(out)


def _evidence_coverage(report) -> str:
    """What testing exists, and what of it reached this report."""
    cov = getattr(report, "assessment_coverage", None) or {}
    df = cov.get("by_source")
    head = "<h3>Assessment coverage</h3>"
    if not isinstance(df, pd.DataFrame) or df.empty:
        return head + rk.empty("No assessment dates on record.")
    out = head
    pct = cov.get("pct_used")
    if pct is not None:
        out += (f"<p><b>{cov['n_dates_total'] - cov['n_dates_outside']} of "
                f"{cov['n_dates_total']}</b> assessment dates ({pct:.0f}%) are "
                f"paired with a 3D capture and used above.</p>")
    out += rk.table(
        df, columns=["source", "total_dates", "in_a_round",
                     "outside_every_round", "first", "latest", "latest_used"],
        rename={"source": "Assessment", "total_dates": "Dates on file",
                "in_a_round": "Paired with 3D",
                "outside_every_round": "Not paired",
                "first": "First", "latest": "Latest",
                "latest_used": "Latest paired"})
    if cov.get("n_dates_outside"):
        out += rk.note(str(cov.get("note")), quiet=True)

    # The one gap that is a real gap: a test date the profiler has never run
    # over. Nothing in this report can show it, because every assessment value
    # is read from ai_layer.athlete_profiles.
    unprof = cov.get("unprofiled_dates") or []
    if unprof:
        sample = ", ".join(unprof[-6:])
        out += rk.note(
            f"{len(unprof)} test date(s) exist in the assessment tables but "
            f"have no row in ai_layer.athlete_profiles, so their values cannot "
            f"appear anywhere in this report — including the history above. "
            f"Most recent: {sample}. Rebuild with "
            f"`python -m src.main backfill` (one profile per assessment date), "
            f"then re-run this report.")
    return out


def _evidence_position(o: dict, cfg) -> str:
    head = "<h3>Where he sits in the group</h3>"
    if o.get("error"):
        return head + rk.empty(str(o["error"]))
    stratum = o.get("stratum") or {}
    flagged = o.get("flagged")
    intro = (f"<p class='subtitle'>Compared against "
             f"<b>{rk.esc(stratum.get('used', '—'))}</b> "
             f"({stratum.get('n_athletes', 0)} athletes, one row each).</p>")
    out = head + intro
    if stratum.get("fell_back") and stratum.get("note"):
        out += rk.note(stratum["note"])
    if not isinstance(flagged, pd.DataFrame) or flagged.empty:
        return out + rk.note("Nothing sits at either extreme of the group — "
                             "he looks typical across the board.", quiet=True)
    curated = flagged[flagged["metric"].map(DISPLAY.is_coach_ready)]
    show = curated if not curated.empty else flagged
    out += rk.table(
        show, columns=["display_name", "athlete_value", "cohort_median",
                       "percentile", "rank", "cohort_n", "icc", "flag"],
        rename={"display_name": "Metric", "athlete_value": "His value",
                "cohort_median": "Group median", "percentile": "Percentile",
                "rank": "Rank", "cohort_n": "Group n", "icc": "Reliability",
                "flag": "Flag"})
    out += ("<div class='legend'>Reliability is the share of the spread in "
            "this metric that is real difference between athletes rather than "
            "measurement scatter. Below 0.5, a ranking on it is mostly noise "
            "and is excluded from the findings above.</div>")
    if o.get("driver_note"):
        out += rk.note(str(o["driver_note"]), quiet=True)
    return out


def _evidence_within_session(v: dict, cfg) -> str:
    """Pitch by pitch. Always shown when there are two or more pitches.

    The spread between pitches is a finding in its own right — a pitcher who
    moves eight degrees of separation pitch to pitch is a different pitcher
    from one who moves two — so the per-pitch values go on the page rather
    than being collapsed into a mean.
    """
    from src.research.metric_filters import FAMILY_LABEL, FAMILY_ORDER

    head = f"<h3>Pitch to pitch — {v.get('focus_session_date')}</h3>"
    if v.get("error"):
        return head + rk.empty(str(v["error"]))

    out = [head]
    n = v.get("n_trials")
    vs = v.get("velo_spread") or {}
    if vs:
        out.append(
            f"<p><b>{n} pitches.</b> Velocity {vs['min']}–{vs['max']} mph "
            f"(mean {vs['mean']}, spread {vs['range']} mph).</p>")
    else:
        out.append(f"<p><b>{n} pitches.</b></p>")

    # ── Every pitch, in the canonical shape ───────────────────────────────
    pm = v.get("pitch_matrix")
    if pm is not None and not pm.frame.empty:
        out.append(
            "<p class='subtitle'>Same table as above, with his pitches as the "
            "columns instead of his captures. Every pitch as recorded, then "
            "the difference between the last two sized against how much that "
            "measure usually moves from pitch to pitch. Sorted by the metrics "
            "he repeats least.</p>")
        by_family = pm.by_family()
        for fam in FAMILY_ORDER:
            sub = by_family.get(fam)
            if sub is None or sub.empty:
                continue
            out.append(f"<h4>{rk.esc(FAMILY_LABEL.get(fam, fam))}</h4>")
            out.append(_matrix_table(pm, sub.head(12), count_label=None))
    else:
        # Fall back to the flat per-pitch dump if the matrix could not build.
        per_pitch = v.get("per_pitch")
        if isinstance(per_pitch, pd.DataFrame) and not per_pitch.empty:
            disp = per_pitch.copy()
            labels: dict[str, str] = {"trial_index": "Pitch",
                                      "velocity_mph": "Velo (mph)"}
            seen: dict[str, int] = {}
            for c in disp.columns:
                if c in labels:
                    continue
                base = DISPLAY.short(c)
                seen[base] = seen.get(base, 0) + 1
                labels[c] = base if seen[base] == 1 else f"{base} ({seen[base]})"
            out.append("<h4>Every pitch</h4>")
            out.append("<div class='scroll-x'>"
                       + rk.table(disp.rename(columns=labels), precision=4)
                       + "</div>")

    # ── The velocity scan, labelled as a scan ─────────────────────────────
    null = v.get("noise_ceiling")
    out.append("<h4>Relationship with velocity on the day</h4>")
    if v.get("correlations_underpowered"):
        out.append(rk.note(
            f"With {n} pitches these correlations are shown for completeness, "
            f"not as findings. Ranking {v.get('n_metrics_scanned', 0)} metrics "
            f"and reading the top of the list produces a large value even when "
            f"the columns are unrelated — that is a property of the ranking, "
            f"not of the pitches. The per-pitch values above are the reliable "
            f"part of this session.", quiet=True))
    elif null is not None:
        out.append(f"<p class='subtitle'>{rk.esc(null.explain())}</p>")

    top = v.get("top_velo_correlates")
    if isinstance(top, pd.DataFrame) and not top.empty:
        out.append(rk.table(
            top, columns=["display_name", "r_vs_velo", "ci", "n_trials",
                          "mean", "range"], max_rows=12,
            rename={"display_name": "Metric", "r_vs_velo": "r vs velo",
                    "ci": "90% interval", "n_trials": "Pitches",
                    "mean": "Mean", "range": "Spread"}))
    else:
        out.append(rk.empty("Not enough pitches to compute a relationship."))

    dstb = v.get("destabilizing_metrics")
    if isinstance(dstb, pd.DataFrame) and not dstb.empty:
        out.append("<h4>Moving more than this athlete usually does</h4>")
        out.append(rk.table(
            dstb, columns=["display_name", "cv", "hist_cv", "cv_vs_hist_ratio"],
            max_rows=10,
            rename={"display_name": "Metric", "cv": "This session",
                    "hist_cv": "His usual", "cv_vs_hist_ratio": "x usual"}))
    return "".join(out)


def _evidence_alignment(sc: dict) -> str:
    vcj = sc.get("velo_correlate_join")
    head = "<h3>Is he moving like a harder thrower?</h3>"
    if not isinstance(vcj, pd.DataFrame) or vcj.empty:
        return head + rk.empty(
            "No stratum-level velocity correlates could be joined to this "
            "athlete's changes.")
    n_aligned = sc.get("n_aligned", 0)
    n_joined = sc.get("n_joined", 0)
    out = head + (
        f"<p><b>{n_aligned} of {n_joined}</b> of the mechanics that track with "
        f"velocity in his group moved the right way by more than measurement "
        f"noise.</p>"
        "<p class='subtitle'>A mechanic only counts when the move was both in "
        "the right direction and bigger than the error band — signing a change "
        "smaller than the noise is coin-flipping with extra steps.</p>")
    return out + rk.table(
        vcj, columns=["display_name", "stratum_r_vs_velo", "athlete_delta",
                      "mdc", "verdict", "moved_right_way", "counts"],
        max_rows=15,
        rename={"display_name": "Metric", "stratum_r_vs_velo": "r in group",
                "athlete_delta": "His change", "mdc": "Noise band",
                "verdict": "Verdict", "moved_right_way": "Right direction",
                "counts": "Counts"})


def _limitations(report, sc: dict, o: dict, v: dict, cfg) -> list[str]:
    out: list[str] = []
    if v.get("correlations_underpowered"):
        out.append(
            f"With {v.get('n_trials')} pitches in the focus session, the "
            f"velocity correlations are descriptive only — "
            f"{cfg.min_trials_within_session} pitches is where a ranked scan "
            f"starts to mean something. The per-pitch values are unaffected.")
    if len(report.athlete.reliability) == 0:
        out.append("No repeat-capture baselines exist yet, so changes are "
                   "reported without a sense of how much each measure "
                   "usually moves for him.")
    else:
        gaps = report.athlete.reliability.reliability_gaps()
        if gaps.get("needs_retest"):
            out.append(
                f"{len(gaps['needs_retest'])} of "
                f"{len(report.athlete.reliability)} measures have no repeat "
                f"captures, so their 'usual variation' is estimated from "
                f"pitch-to-pitch spread and reads narrower than the truth.")
    stratum = o.get("stratum") or {}
    if stratum.get("fell_back"):
        out.append(
            f"He was compared against all tested pitchers rather than "
            f"{stratum.get('requested')} — only "
            f"{stratum.get('n_athletes') if not stratum.get('fell_back') else '<' + str(cfg.min_athletes_for_stratum)} "
            f"athletes at his level are on file.")
    # The gap that matters is between the two captures actually being
    # compared, not between his first and his last. Quoting the whole history
    # here said "735 days apart" under a table comparing captures 326 days
    # apart, which is a different and much scarier claim than the true one.
    sm = sc.get("matrix")
    span = getattr(sm, "days_between", None) if sm is not None else None
    if span is None:
        span = sc.get("span_days")
    if span is not None and span > cfg.long_span_warn_days:
        out.append(
            f"The two compared captures are {span} days apart, so maturation "
            f"and a full training block are inside every difference reported.")
    if sc.get("skipped"):
        out.append(str(sc["skipped"]))
    held = (report.summary or {}).get("held_back_uncurated") or []
    if held:
        out.append(
            f"{len(held)} finding(s) were withheld because their metric has no "
            f"coach-facing name yet; they are in the appendix under their "
            f"warehouse key.")
    meta = sc.get("assessment_delta_meta") or {}
    n_stale = len(meta.get("excluded_stale", []))
    if n_stale:
        out.append(
            f"{n_stale} assessment metric(s) had no fresh re-test and were "
            f"excluded from the change table rather than differenced against "
            f"a carried-forward value.")
    return out


def _appendix(report, sc: dict, o: dict, v: dict) -> str:
    parts: list[str] = []

    if report.athlete.rounds:
        from src.research.assessment_rounds import rounds_to_dataframe
        rdf = rounds_to_dataframe(report.athlete.rounds)
        keep = [c for c in rdf.columns if rdf[c].notna().any()]
        parts.append(rk.details("Assessment rounds",
                                rk.table(rdf[keep], precision=4)))

    mech = sc.get("mechanic_deltas")
    if isinstance(mech, pd.DataFrame) and not mech.empty:
        parts.append(rk.details(
            f"Every mechanical delta ({len(mech)})",
            rk.table(mech, columns=["metric", "display_name", "from_value",
                                    "to_value", "delta", "mdc", "mdc_ratio",
                                    "verdict"], max_rows=400)))

    # What the page had to leave out, and what it would take to include it.
    counts = (report.summary or {}).get("held_back_counts") or {}
    if counts:
        held_df = pd.DataFrame(
            [{"metric": m, "findings": n,
              "auto_name": DISPLAY.name(m),
              "status": DISPLAY.get(m).status}
             for m, n in sorted(counts.items(), key=lambda kv: -kv[1])])
        parts.append(rk.details(
            f"Findings held back for want of a curated name "
            f"({int(held_df['findings'].sum())} across {len(held_df)} metrics)",
            "<p class='subtitle'>These are real measurements. The name beside "
            "each one was composed by the machine from the warehouse key, and "
            "nothing machine-named is allowed onto the coach page — a wrong "
            "name is worse than no name. Adding an entry to "
            "metric_display.yaml promotes it, most-used first.</p>"
            + rk.table(held_df, max_rows=120)))

    allm = o.get("all_metrics")
    if isinstance(allm, pd.DataFrame) and not allm.empty:
        parts.append(rk.details(
            f"Every cohort position ({len(allm)})",
            rk.table(allm, columns=["metric", "display_name", "athlete_value",
                                    "cohort_median", "percentile", "rank",
                                    "cohort_n", "mode", "icc"], max_rows=400)))

    allv = v.get("all_metrics")
    if isinstance(allv, pd.DataFrame) and not allv.empty:
        parts.append(rk.details(
            f"Every within-session correlation ({len(allv)}) — unfiltered, "
            f"exploratory",
            rk.note("This is the raw scan. It is here for auditing, not for "
                    "acting on: the top of an unfiltered ranking over this "
                    "many metrics is the largest of that many coin flips.",
                    quiet=True)
            + rk.table(allv, columns=["metric", "r_vs_velo", "ci", "p_vs_velo",
                                      "n_trials", "cv", "tier"], max_rows=400)))

    bp = report.big_picture or {}
    if bp.get("transitions"):
        inner = []
        for tx in bp["transitions"]:
            inner.append(f"<h4>{rk.esc(tx['from_round'])} → "
                         f"{rk.esc(tx['to_round'])} · "
                         f"{rk.esc(tx.get('span_days'))} days · "
                         f"Δ velo {tx.get('delta_velocity')}</h4>")
            tmc = tx.get("top_mechanic_changes")
            if isinstance(tmc, pd.DataFrame) and not tmc.empty:
                inner.append(rk.table(
                    tmc, columns=["display_name", "from_value", "to_value",
                                  "delta", "mdc_ratio", "verdict"]))
            tac = tx.get("top_assessment_changes")
            if isinstance(tac, pd.DataFrame) and not tac.empty:
                inner.append(rk.table(
                    tac, columns=["display_name", "from_value", "to_value",
                                  "delta", "change"]))
        parts.append(rk.details("Every round transition", "".join(inner)))

    vd = report.velo_drivers or {}
    tc = vd.get("top_correlates")
    if isinstance(tc, pd.DataFrame) and not tc.empty:
        parts.append(rk.details(
            "Group velocity drivers (within-athlete fixed effects)",
            f"<p class='subtitle'>{vd.get('n_fdr_significant', 0)} of "
            f"{vd.get('n_tested', 0)} survived FDR correction.</p>"
            + rk.table(tc, max_rows=40)))

    findings = report.findings_frame()
    if not findings.empty:
        parts.append(rk.details(
            "Findings as data",
            rk.table(findings, columns=["kind", "metric", "headline", "tier",
                                        "verdict", "good", "delta", "mdc",
                                        "percentile", "priority"],
                     max_rows=100)))

    return "".join(parts) if parts else rk.empty("Nothing further.")


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in (name or "athlete")).strip("_")


__all__ = ["render_coach_report"]
