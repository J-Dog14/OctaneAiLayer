"""
The roster view — every athlete on one page.

With pros coming back in, the first question a coach has is not "tell me
everything about one pitcher", it is "who moved, and who do I need to look at
today". That question had no artifact at all: you could generate one deep dive
at a time and compare them by eye.

One row per athlete, one column per KPI, coloured by whether the change since
their previous capture cleared that metric's measurement-error band, with a
sparkline for the trend and a link through to the full report.

Cheap by construction: one warehouse read for the whole squad (see `loaders`),
then everything is pandas.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from src.research import loaders, render_kit as rk
from src.research.config import CONFIG, ResearchConfig
from src.research.metric_display import DISPLAY
from src.research.reliability import (
    VERDICT_LIKELY,
    VERDICT_NOISE,
    VERDICT_REAL,
    VERDICT_UNKNOWN,
    ReliabilityTable,
)

# The KPI set. Deliberately short: a roster with 20 columns is a spreadsheet,
# and the point of this page is that a coach can read it in ten seconds.
# Every entry must be curated in metric_display.yaml.
DEFAULT_KPIS = [
    "velocity_mph",
    "fm_lead_peak_braking_bw",
    "fm_lead_rfd_braking_bw_per_s",
    "fm_lead_peak_vertical_bw",
    "fm_peak_v_to_peak_b_lag_ms",
]


@dataclass
class AthleteRow:
    athlete_uuid: str
    name: str
    age_group: str | None
    n_sessions: int
    last_session: Any
    days_since: int | None
    velocity_series: list[float] = field(default_factory=list)
    kpis: dict[str, dict] = field(default_factory=dict)
    n_real_changes: int = 0
    n_real_improvements: int = 0
    n_real_declines: int = 0
    attention: float = 0.0
    attention_reason: str = ""


@dataclass
class Roster:
    rows: list[AthleteRow]
    kpis: list[str]
    stratum: str
    generated_at: datetime
    config: ResearchConfig
    reliability: ReliabilityTable
    notes: list[str] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        recs = []
        for r in self.rows:
            rec = {
                "name": r.name, "age_group": r.age_group,
                "n_sessions": r.n_sessions, "last_session": r.last_session,
                "days_since": r.days_since,
                "real_changes": r.n_real_changes,
                "improvements": r.n_real_improvements,
                "declines": r.n_real_declines,
                "attention": round(r.attention, 2),
            }
            for m, k in r.kpis.items():
                rec[f"{DISPLAY.short(m)}"] = k.get("latest")
                rec[f"{DISPLAY.short(m)} Δ"] = k.get("delta")
            recs.append(rec)
        return pd.DataFrame(recs)


def build_roster(
    *,
    age_group: str | None = None,
    kpis: Sequence[str] | None = None,
    min_sessions: int = 1,
    limit: int | None = None,
    config: ResearchConfig = CONFIG,
) -> Roster:
    """One row per athlete with the change since their previous capture."""
    kpis = list(kpis or DEFAULT_KPIS)
    notes: list[str] = []

    trials = loaders.trials(age_group=age_group, copy=False)
    if trials.empty:
        return Roster([], kpis, age_group or "ALL", datetime.now(), config,
                      ReliabilityTable(), ["No trial data available."])

    reliability = loaders.reliability_table()
    present = [m for m in kpis if m in trials.columns]
    missing = [m for m in kpis if m not in trials.columns]
    if missing:
        notes.append(f"Not in the warehouse, so left off: {', '.join(missing)}")

    uncurated = [m for m in present if not DISPLAY.is_coach_ready(m)]
    if uncurated:
        notes.append(
            f"{len(uncurated)} KPI(s) have no coach-facing name yet and are "
            f"shown under their warehouse key: {', '.join(uncurated)}")

    # Session means + trial counts, one pass.
    sess = (trials.groupby(["athlete_uuid", "name", "age_group", "session_date"])
                  [present].mean().reset_index())
    n_by_session = (trials.groupby(["athlete_uuid", "session_date"]).size()
                          .rename("n_trials").reset_index())
    sess = sess.merge(n_by_session, on=["athlete_uuid", "session_date"], how="left")
    sess = sess.sort_values(["athlete_uuid", "session_date"])

    today = date.today()
    rows: list[AthleteRow] = []
    for uuid, g in sess.groupby("athlete_uuid"):
        g = g.sort_values("session_date")
        if len(g) < min_sessions:
            continue
        last = g.iloc[-1]
        prev = g.iloc[-2] if len(g) >= 2 else None

        last_date = last["session_date"]
        days_since = None
        try:
            days_since = (today - pd.to_datetime(last_date).date()).days
        except Exception:
            pass

        row = AthleteRow(
            athlete_uuid=uuid,
            name=str(last["name"]),
            age_group=str(last["age_group"]) if pd.notna(last["age_group"]) else None,
            n_sessions=int(len(g)),
            last_session=last_date,
            days_since=days_since,
            velocity_series=[float(v) for v in g.get("velocity_mph", pd.Series(dtype=float)).tolist()
                             if pd.notna(v)],
        )

        for m in present:
            latest = last.get(m)
            entry: dict[str, Any] = {
                "latest": float(latest) if pd.notna(latest) else None,
                "delta": None, "verdict": VERDICT_UNKNOWN, "mdc": None,
                "toward_better": None,
                "series": [float(v) for v in g[m].tolist() if pd.notna(v)],
            }
            if prev is not None and pd.notna(latest) and pd.notna(prev.get(m)):
                delta = float(latest) - float(prev[m])
                cv = reliability.classify(
                    m, delta,
                    n_before=int(prev.get("n_trials") or 1),
                    n_after=int(last.get("n_trials") or 1))
                entry.update(delta=delta, verdict=cv.verdict, mdc=cv.mdc,
                             toward_better=DISPLAY.signed_toward_better(m, delta))
            row.kpis[m] = entry

        reals = [k for k in row.kpis.values() if k["verdict"] == VERDICT_REAL]
        row.n_real_changes = len(reals)
        row.n_real_improvements = sum(1 for k in reals if k["toward_better"] == 1)
        row.n_real_declines = sum(1 for k in reals if k["toward_better"] == -1)
        row.attention, row.attention_reason = _attention(row, config)
        rows.append(row)

    rows.sort(key=lambda r: (-r.attention, r.name))
    if limit:
        rows = rows[:limit]

    if not any(len(reliability) for _ in [0]):
        notes.append("No reliability estimates available, so no change could "
                     "be separated from measurement noise.")
    return Roster(rows, present, age_group or "ALL", datetime.now(), config,
                  reliability, notes)


def _attention(row: AthleteRow, config: ResearchConfig) -> tuple[float, str]:
    """A single 'look at this one' score, so the page sorts itself.

    Weighted toward things a coach would want surfaced without asking: real
    declines first, then a velocity drop, then a long gap since last capture.
    Deliberately simple and readable — a score nobody can explain is a score
    nobody trusts.
    """
    score = 0.0
    reasons: list[str] = []

    if row.n_real_declines:
        score += 3.0 * row.n_real_declines
        reasons.append(f"{row.n_real_declines} measure(s) down beyond noise")

    velo = row.kpis.get("velocity_mph") or {}
    if velo.get("verdict") == VERDICT_REAL and (velo.get("delta") or 0) < 0:
        score += 4.0
        reasons.append(f"velo down {abs(velo['delta']):.1f} mph")
    elif velo.get("verdict") == VERDICT_REAL and (velo.get("delta") or 0) > 0:
        score += 1.0
        reasons.append(f"velo up {velo['delta']:.1f} mph")

    if row.days_since is not None and row.days_since > 120:
        score += 1.5
        reasons.append(f"no capture in {row.days_since} days")

    if row.n_sessions == 1:
        score += 0.5
        reasons.append("only one capture on file")

    return score, "; ".join(reasons)


# ──────────────────────────────────────────────────────────────────────────
# Rendering
# ──────────────────────────────────────────────────────────────────────────

def render_roster_report(roster: Roster, *,
                         report_links: dict[str, str] | None = None,
                         output_dir: Path | None = None) -> Path:
    links = report_links or {}
    body: list[str] = []

    body.append(rk.page_header(
        "Squad overview",
        [f"{len(roster.rows)} athletes",
         roster.stratum if roster.stratum != "ALL" else "all levels",
         f"{len(roster.kpis)} KPIs"],
        f"Generated {roster.generated_at:%Y-%m-%d %H:%M} · "
        f"{roster.config.describe()}"))

    if not roster.rows:
        body.append(rk.empty("No athletes matched."))
        return rk.write_html("\n".join(body), "roster.html", "Squad overview",
                             output_dir=output_dir, wide=True)

    total = len(roster.rows)
    n_declining = sum(1 for r in roster.rows if r.n_real_declines)
    n_improving = sum(1 for r in roster.rows if r.n_real_improvements)
    n_stale = sum(1 for r in roster.rows
                  if r.days_since is not None and r.days_since > 120)
    body.append(rk.banner(
        f"{n_declining} of {total} have a measure down beyond noise · "
        f"{n_improving} have one up · {n_stale} overdue for a capture",
        "Sorted so whoever needs looking at is at the top. 'Real' means the "
        "change cleared that measurement's own noise band — everything else "
        "is treated as unchanged.",
        tone="bad" if n_declining else "flat"))

    body.append("<div class='scroll-x'>" + _roster_table(roster, links) + "</div>")

    body.append("<h2>Velocity trends</h2>")
    body.append(_velocity_small_multiples(roster))

    if roster.notes:
        body.append(rk.note("Coverage notes:", items=roster.notes, quiet=True))

    body.append(rk.details(
        "How to read this",
        "<p>Each cell shows the athlete's latest session mean and, underneath, "
        "the change from their previous capture. Colour is only applied when "
        "the change cleared that metric's measurement-error band <em>and</em> "
        "the metric has an agreed better direction — green toward better, red "
        "away. A grey change is inside the noise: it did not move. A dash "
        "means we have only one capture, so there is nothing to compare.</p>"
        "<p>The attention score is deliberately simple: three points per real "
        "decline, four for a real velocity drop, one and a half for a capture "
        "gap over 120 days, half a point for a single-session athlete. It "
        "sorts the page; it is not a rating of the athlete.</p>"))

    fname = f"roster_{_slug(roster.stratum)}_{rk.timestamp_slug()}.html"
    return rk.write_html("\n".join(body), fname, "Squad overview",
                         output_dir=output_dir, wide=True)


def _cell(entry: dict, metric: str) -> str:
    latest = entry.get("latest")
    if latest is None:
        return "<td class='num'>—</td>"
    main = DISPLAY.format_value(metric, latest)
    delta = entry.get("delta")
    verdict = entry.get("verdict")
    if delta is None:
        sub = "<span style='color:#8c959f'>first capture</span>"
    elif verdict in (VERDICT_REAL, VERDICT_LIKELY):
        good = entry.get("toward_better")
        colour = {1: rk.GOOD, -1: rk.BAD}.get(good, rk.NEUTRAL)
        weight = "600" if verdict == VERDICT_REAL else "500"
        mark = "" if verdict == VERDICT_REAL else "?"
        sub = (f"<span style='color:{colour};font-weight:{weight}'>"
               f"{DISPLAY.format_delta(metric, delta)}{mark}</span>")
    else:
        sub = (f"<span style='color:#8c959f'>"
               f"{DISPLAY.format_delta(metric, delta)}</span>")
    return (f"<td class='num'><div>{rk.esc(main)}</div>"
            f"<div style='font-size:11.5px'>{sub}</div></td>")


def _roster_table(roster: Roster, links: dict[str, str]) -> str:
    head = ["<th>Athlete</th>", "<th>Level</th>", "<th class='num'>Last</th>",
            "<th>Velo trend</th>"]
    for m in roster.kpis:
        unit = DISPLAY.unit(m)
        head.append(f"<th class='num'>{rk.esc(DISPLAY.short(m))}"
                    + (f"<br><span style='font-weight:400;text-transform:none'>"
                       f"{rk.esc(unit)}</span>" if unit else "")
                    + "</th>")

    rows_html = []
    for r in roster.rows:
        name = rk.esc(r.name)
        link = links.get(r.athlete_uuid)
        name_cell = (f"<a href='{rk.esc(link)}'>{name}</a>" if link else name)
        last = rk.esc(r.last_session)
        if r.days_since is not None:
            colour = rk.BAD if r.days_since > 120 else rk.MUTED
            last += (f"<div style='font-size:11.5px;color:{colour}'>"
                     f"{r.days_since}d ago</div>")
        cells = [
            # The reason lives under the name rather than in its own column:
            # a tenth column pushed the table off the page, and this is where
            # the eye already is when it lands on the row.
            f"<td style='min-width:190px'><b>{name_cell}</b>"
            f"<div style='font-size:11.5px;color:#5b6570'>"
            f"{r.n_sessions} session(s)</div>"
            f"<div style='font-size:11.5px;color:{rk.BAD if r.n_real_declines else rk.MUTED};"
            f"margin-top:2px'>{rk.esc(r.attention_reason)}</div></td>",
            f"<td class='nowrap'>{rk.esc(r.age_group or '—')}</td>",
            f"<td class='num'>{last}</td>",
            f"<td>{rk.sparkline_svg(r.velocity_series)}</td>",
        ]
        for m in roster.kpis:
            cells.append(_cell(r.kpis.get(m, {}), m))
        rows_html.append("<tr>" + "".join(cells) + "</tr>")

    return (f"<table><thead><tr>{''.join(head)}</tr></thead>"
            f"<tbody>{''.join(rows_html)}</tbody></table>")


def _velocity_small_multiples(roster: Roster, *, max_athletes: int = 24) -> str:
    """One line per athlete on a shared axis. Shows the spread of the whole
    group and who is moving, in one glance."""
    import plotly.graph_objects as go
    rows = [r for r in roster.rows if len(r.velocity_series) >= 2][:max_athletes]
    if not rows:
        return rk.empty("Not enough repeat captures to draw a trend.")

    fig = go.Figure()
    for i, r in enumerate(rows):
        colour = rk.SERIES[i % len(rk.SERIES)]
        velo = r.kpis.get("velocity_mph") or {}
        width = 3 if velo.get("verdict") == VERDICT_REAL else 1.6
        fig.add_trace(go.Scatter(
            y=r.velocity_series, x=list(range(1, len(r.velocity_series) + 1)),
            mode="lines+markers", name=r.name,
            line=dict(color=colour, width=width),
            marker=dict(size=6),
            hovertemplate=f"<b>{rk.esc(r.name)}</b><br>capture %{{x}}"
                          "<br>%{y:.1f} mph<extra></extra>",
        ))
    fig.update_layout(
        font=dict(family=rk.PLOT_FONT, size=12, color=rk.INK),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=60, r=28, t=40, b=44),
        title=dict(text="Session-mean velocity by capture number", x=0,
                   font=dict(size=15)),
        xaxis=dict(title="capture", gridcolor=rk.LINE, dtick=1),
        yaxis=dict(title="mph", gridcolor=rk.LINE),
        height=420, showlegend=True,
        legend=dict(font=dict(size=11), orientation="v", x=1.02, y=1),
        hovermode="closest",
    )
    return rk.fig_html(fig, first=True) + (
        "<div class='legend'>Thick lines are athletes whose latest velocity "
        "change cleared the measurement-noise band. Captures are plotted by "
        "order, not date, so trajectories line up.</div>")


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in (s or "all")).strip("_").lower()


__all__ = ["build_roster", "render_roster_report", "Roster", "AthleteRow",
           "DEFAULT_KPIS"]
