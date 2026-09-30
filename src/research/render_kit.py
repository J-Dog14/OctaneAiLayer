"""
Shared HTML/chart furniture for coach-facing research output.

Everything here exists to make the coach pages look like one product rather
than five renderers that each grew their own table style: one shell, one
palette, one set of chart defaults, one traffic-light vocabulary.

Design rules baked in:
  * Charts are the default, tables are the fallback. A number in a cell makes
    a coach do the comparison in their head; a dot on a strip does it for them.
  * Nothing renders as good-or-bad unless the metric dictionary says the
    metric HAS a good direction. Load metrics and context metrics stay neutral.
  * Everything prints. These get taken to the field on paper.
  * Self-contained: one HTML file, Plotly from CDN, no other assets.
"""
from __future__ import annotations

import html as _html
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from src.research.metric_display import DISPLAY
from src.research.reliability import (
    VERDICT_COLOR,
    VERDICT_LABEL,
    VERDICT_LIKELY,
    VERDICT_NOISE,
    VERDICT_REAL,
    VERDICT_UNKNOWN,
)

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────
INK = "#1c2024"
MUTED = "#5b6570"
LINE = "#e3e6ea"
CANVAS = "#ffffff"
WASH = "#f6f8fa"

GOOD = "#1a7f37"
BAD = "#b42318"
NEUTRAL = "#4a6fa5"
WARN = "#bf8700"
GREY = "#8c959f"

# Categorical series colours — colour-blind safe, distinguishable in greyscale
SERIES = ["#4a6fa5", "#c1663b", "#4f8a6b", "#8a5fa8", "#b0873b", "#7a8794"]

PLOT_FONT = ("-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, "
             "'Helvetica Neue', Arial, sans-serif")


def esc(s: Any) -> str:
    return _html.escape(str(s if s is not None else ""))


# ──────────────────────────────────────────────────────────────────────────
# Shell
# ──────────────────────────────────────────────────────────────────────────

_CSS = f"""
:root {{
  --ink: {INK}; --muted: {MUTED}; --line: {LINE}; --canvas: {CANVAS};
  --wash: {WASH}; --good: {GOOD}; --bad: {BAD}; --neutral: {NEUTRAL};
  --warn: {WARN}; --grey: {GREY};
}}
* {{ box-sizing: border-box; }}
body {{
  font-family: {PLOT_FONT};
  color: var(--ink); background: var(--canvas);
  max-width: 1100px; margin: 0 auto; padding: 28px 20px 80px;
  line-height: 1.5; font-size: 15px;
}}
/* Roster-style pages carry a column per KPI and need the room. */
body.wide {{ max-width: 1500px; }}
.scroll-x {{ overflow-x: auto; }}
h1 {{ font-size: 26px; margin: 0 0 4px; letter-spacing: -0.01em; }}
h2 {{ font-size: 19px; margin: 40px 0 12px; padding-bottom: 6px;
     border-bottom: 2px solid var(--line); letter-spacing: -0.01em; }}
h3 {{ font-size: 16px; margin: 24px 0 8px; }}
h4 {{ font-size: 14px; margin: 16px 0 6px; color: var(--muted);
     text-transform: uppercase; letter-spacing: 0.04em; }}
p {{ margin: 8px 0; }}
code {{ background: var(--wash); padding: 1px 5px; border-radius: 3px;
       font-size: 0.88em; }}
.subtitle {{ color: var(--muted); font-size: 13.5px; margin: 0 0 4px; }}
.stamp {{ color: var(--grey); font-size: 11.5px; font-family: ui-monospace,
         SFMono-Regular, Menlo, monospace; margin-top: 6px; }}

/* ── Verdict banner ─────────────────────────────────────────── */
.banner {{ border-radius: 10px; padding: 18px 20px; margin: 20px 0 8px;
          background: var(--wash); border: 1px solid var(--line); }}
.banner .big {{ font-size: 22px; font-weight: 620; letter-spacing: -0.01em; }}
.banner .sub {{ color: var(--muted); font-size: 14px; margin-top: 4px; }}
.banner.good {{ background: #f0f8f2; border-color: #b7dfc4; }}
.banner.bad  {{ background: #fdf3f2; border-color: #f2c2bd; }}
.banner.flat {{ background: var(--wash); }}

/* ── Finding cards ──────────────────────────────────────────── */
.cards {{ display: grid; gap: 10px; margin: 12px 0 4px; }}
.card {{ border: 1px solid var(--line); border-left: 4px solid var(--grey);
        border-radius: 8px; padding: 12px 14px; background: var(--canvas); }}
.card.good {{ border-left-color: var(--good); }}
.card.bad {{ border-left-color: var(--bad); }}
.card.neutral {{ border-left-color: var(--neutral); }}
.card .head {{ font-weight: 600; font-size: 15px; }}
.card .detail {{ color: var(--muted); font-size: 13.5px; margin-top: 4px; }}
.card .cue {{ font-size: 13.5px; margin-top: 6px; padding: 6px 10px;
             background: var(--wash); border-radius: 6px; }}
.card .cue b {{ font-weight: 600; }}

/* ── Chips ──────────────────────────────────────────────────── */
.chip {{ display: inline-block; font-size: 11px; font-weight: 600;
        padding: 2px 8px; border-radius: 999px; letter-spacing: 0.02em;
        vertical-align: 2px; margin-left: 6px; white-space: nowrap; }}
.chip.real {{ background: #e7f4ec; color: {GOOD}; }}
.chip.likely {{ background: #fdf5e3; color: {WARN}; }}
.chip.noise {{ background: #eef1f4; color: {MUTED}; }}
.chip.unknown {{ background: #f2f3f5; color: {GREY}; }}

/* ── Two-column ─────────────────────────────────────────────── */
.cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
@media (max-width: 760px) {{ .cols {{ grid-template-columns: 1fr; }} }}

/* ── Tables ─────────────────────────────────────────────────── */
table {{ border-collapse: collapse; width: 100%; font-size: 13px;
        margin: 10px 0; }}
th, td {{ text-align: left; padding: 6px 10px;
         border-bottom: 1px solid var(--line); }}
th {{ background: var(--wash); font-weight: 600; font-size: 12px;
     text-transform: uppercase; letter-spacing: 0.03em; color: var(--muted); }}
td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums;
                  white-space: nowrap; }}
td.nowrap, th.nowrap {{ white-space: nowrap; }}
tbody tr:hover {{ background: var(--wash); }}

/* ── Notes ──────────────────────────────────────────────────── */
.note {{ background: #fffbf0; border-left: 3px solid var(--warn);
        padding: 10px 14px; margin: 14px 0; font-size: 13.5px;
        border-radius: 0 6px 6px 0; }}
.note.quiet {{ background: var(--wash); border-left-color: var(--grey); }}
.note ul {{ margin: 6px 0 0; padding-left: 20px; }}
.empty {{ color: var(--muted); font-style: italic; font-size: 13.5px; }}

details {{ margin: 12px 0; border: 1px solid var(--line); border-radius: 8px;
          padding: 0 14px; }}
details[open] {{ padding-bottom: 12px; }}
summary {{ cursor: pointer; padding: 10px 0; font-weight: 600;
          font-size: 14px; color: var(--muted); }}
summary:hover {{ color: var(--ink); }}

.plot {{ margin: 6px 0 18px; }}
.legend {{ font-size: 12px; color: var(--muted); margin: -10px 0 16px; }}

a {{ color: var(--neutral); }}

@media print {{
  body {{ max-width: none; padding: 0; font-size: 11pt; }}
  h2 {{ page-break-after: avoid; }}
  .card, .banner {{ page-break-inside: avoid; }}
  details {{ page-break-inside: avoid; }}
  details:not([open]) > *:not(summary) {{ display: none; }}
  .no-print {{ display: none; }}
}}
"""


def write_html(body: str, filename: str, title: str,
               *, output_dir: Path | None = None, wide: bool = False) -> Path:
    out = (output_dir or OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    cls = " class='wide'" if wide else ""
    doc = (
        "<!DOCTYPE html>\n<html lang='en'><head>\n"
        "<meta charset='utf-8'>\n"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>\n"
        f"<title>{esc(title)}</title>\n<style>{_CSS}</style>\n"
        f"</head><body{cls}>\n{body}\n</body></html>\n"
    )
    path.write_text(doc, encoding="utf-8")
    return path


# ──────────────────────────────────────────────────────────────────────────
# Components
# ──────────────────────────────────────────────────────────────────────────

_VERDICT_CLASS = {
    VERDICT_REAL: "real", VERDICT_LIKELY: "likely",
    VERDICT_NOISE: "noise", VERDICT_UNKNOWN: "unknown",
}


def verdict_chip(verdict: str | None) -> str:
    if not verdict:
        return ""
    cls = _VERDICT_CLASS.get(verdict, "unknown")
    return f"<span class='chip {cls}'>{esc(VERDICT_LABEL.get(verdict, verdict))}</span>"


def banner(big: str, sub: str = "", tone: str = "flat") -> str:
    sub_html = f"<div class='sub'>{esc(sub)}</div>" if sub else ""
    return (f"<div class='banner {tone}'><div class='big'>{esc(big)}</div>"
            f"{sub_html}</div>")


def finding_card(headline: str, detail: str = "", *, tone: str = "neutral",
                 verdict: str | None = None, cue: str | None = None) -> str:
    parts = [f"<div class='card {tone}'>",
             f"<div class='head'>{esc(headline)}{verdict_chip(verdict)}</div>"]
    if detail:
        parts.append(f"<div class='detail'>{esc(detail)}</div>")
    if cue:
        parts.append(f"<div class='cue'><b>Cue:</b> {esc(cue)}</div>")
    parts.append("</div>")
    return "".join(parts)


def note(text: str, *, quiet: bool = False, items: Iterable[str] | None = None) -> str:
    cls = "note quiet" if quiet else "note"
    body = esc(text)
    if items:
        lis = "".join(f"<li>{esc(i)}</li>" for i in items)
        body += f"<ul>{lis}</ul>"
    return f"<div class='{cls}'>{body}</div>"


def empty(text: str) -> str:
    return f"<p class='empty'>{esc(text)}</p>"


def details(summary: str, inner: str, *, open_: bool = False) -> str:
    o = " open" if open_ else ""
    return f"<details{o}><summary>{esc(summary)}</summary>{inner}</details>"


_NUMERIC_KINDS = "iuf"


def table(df: pd.DataFrame, *, columns: Sequence[str] | None = None,
          rename: dict[str, str] | None = None,
          precision: int = 3, max_rows: int | None = None) -> str:
    """A table that right-aligns numbers and never dumps an index column."""
    if df is None or df.empty:
        return empty("No rows.")
    d = df if columns is None else df[[c for c in columns if c in df.columns]]
    if max_rows is not None and len(d) > max_rows:
        d = d.head(max_rows)
    rename = rename or {}
    # Duplicate column labels make d[c] return a DataFrame instead of a Series,
    # which blows up on .dtype. Callers can produce them innocently — two
    # metrics can share a short display name — so make the renderer safe rather
    # than relying on every call site to be careful.
    if d.columns.duplicated().any():
        d = d.loc[:, ~d.columns.duplicated()]
    numeric = {c for c, dt in zip(d.columns, d.dtypes) if dt.kind in _NUMERIC_KINDS}

    head = "".join(
        f"<th class='num'>{esc(rename.get(c, c))}</th>" if c in numeric
        else f"<th>{esc(rename.get(c, c))}</th>"
        for c in d.columns
    )
    rows = []
    for _, r in d.iterrows():
        cells = []
        for c in d.columns:
            v = r[c]
            if isinstance(v, pd.Series):      # belt and braces
                v = v.iloc[0]
            if c in numeric and pd.notna(v):
                cells.append(f"<td class='num'>{v:,.{precision}g}</td>")
            elif isinstance(v, bool):
                cells.append(f"<td>{'yes' if v else 'no'}</td>")
            elif pd.isna(v):
                cells.append("<td>—</td>")
            else:
                cells.append(f"<td>{esc(v)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (f"<table><thead><tr>{head}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>")


# ──────────────────────────────────────────────────────────────────────────
# Charts
# ──────────────────────────────────────────────────────────────────────────

def _layout(**kw) -> dict:
    base = dict(
        font=dict(family=PLOT_FONT, size=12, color=INK),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=60, r=28, t=44, b=44),
        hoverlabel=dict(font_size=12, font_family=PLOT_FONT),
        xaxis=dict(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE),
        yaxis=dict(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE),
        showlegend=False,
    )
    base.update(kw)
    return base


# ── Plotly delivery ───────────────────────────────────────────────────────
# 'cdn'    — small files, needs internet when the page is opened.
# 'inline' — the plotly bundle is embedded, so the report works on a field
#            laptop with no wifi and survives being emailed around. Costs
#            ~3 MB per file, which is the right trade for something a coach
#            opens at a facility.
_PLOTLY_MODE = os.getenv("RESEARCH_PLOTLY", "cdn").strip().lower()


def set_plotly_mode(mode: str) -> None:
    global _PLOTLY_MODE
    if mode not in ("cdn", "inline"):
        raise ValueError("plotly mode must be 'cdn' or 'inline'")
    _PLOTLY_MODE = mode


def plotly_mode() -> str:
    return _PLOTLY_MODE


def fig_html(fig, *, first: bool = False) -> str:
    """Embed a figure. The plotly bundle is attached once per document."""
    if first:
        include = True if _PLOTLY_MODE == "inline" else "cdn"
    else:
        include = False
    inner = fig.to_html(include_plotlyjs=include, full_html=False,
                        config={"displayModeBar": False, "responsive": True})
    return f"<div class='plot'>{inner}</div>"


def chart_velocity_trend(sessions: pd.DataFrame, *, mdc: float | None = None,
                         first: bool = False) -> str:
    """Session-mean velocity over time with the within-session spread, and —
    when we have a reliability estimate — a shaded band showing how big a
    change has to be before it means anything."""
    import plotly.graph_objects as go
    if sessions is None or sessions.empty:
        return empty("No sessions on record.")
    d = sessions.dropna(subset=["mean_velocity"]).copy()
    if d.empty:
        return empty("No velocity recorded.")
    d["session_date"] = pd.to_datetime(d["session_date"])
    d = d.sort_values("session_date")

    fig = go.Figure()
    # Band is anchored on the PREVIOUS session, so the question it answers is
    # the one a coach actually asks: did the latest number leave the range the
    # last one could have produced by measurement error alone?
    ref = None
    if mdc and len(d) >= 2:
        ref = float(d["mean_velocity"].iloc[-2])
        fig.add_hrect(y0=ref - mdc, y1=ref + mdc, line_width=0,
                      fillcolor=GREY, opacity=0.13, layer="below")
    if "max_velocity" in d.columns and d["max_velocity"].notna().any():
        fig.add_trace(go.Scatter(
            x=d["session_date"], y=d["max_velocity"], mode="markers",
            marker=dict(size=7, color=GREY, symbol="line-ew-open",
                        line=dict(width=2, color=GREY)),
            name="session max",
            hovertemplate="%{x|%b %d %Y}<br>max %{y:.1f} mph<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=d["session_date"], y=d["mean_velocity"], mode="lines+markers+text",
        line=dict(color=NEUTRAL, width=2.5),
        marker=dict(size=10, color=NEUTRAL,
                    line=dict(width=2, color=CANVAS)),
        text=[f"{v:.1f}" for v in d["mean_velocity"]],
        textposition="top center",
        textfont=dict(size=11, color=INK),
        name="session mean",
        customdata=d[["n_trials"]].to_numpy(),
        hovertemplate=("%{x|%b %d %Y}<br>mean %{y:.1f} mph"
                       "<br>%{customdata[0]} pitches<extra></extra>"),
    ))
    band = ""
    if ref is not None and mdc:
        band = (f"Shaded band is ±{mdc:.1f} mph around the previous session — "
                f"the range his session velocity moves through anyway. ")
    fig.update_layout(**_layout(
        title=dict(text="Velocity by session", x=0, font=dict(size=15)),
        yaxis_title="mph", height=320,
    ))
    return fig_html(fig, first=first) + (
        f"<div class='legend'>{esc(band)}Dashes mark that session's "
        f"hardest pitch.</div>")


def chart_change_vs_noise(deltas: pd.DataFrame, *, max_rows: int = 10,
                          first: bool = False) -> str:
    """Horizontal bars of change expressed in multiples of how much each
    measure usually varies for this athlete, so metrics in different units sit
    on one comparable axis.

    Answers 'how big is this move, for him?' — a 3-degree change reads
    differently for a measure that holds within half a degree across his own
    pitches than for one that swings six.
    """
    import plotly.graph_objects as go
    if deltas is None or deltas.empty:
        return empty("No changes to plot.")
    d = deltas.dropna(subset=["mdc_ratio"]).copy()
    if d.empty:
        return empty("No metric has a repeat-capture baseline yet, so these "
                     "changes cannot be placed in proportion.")
    d["signed_ratio"] = d["mdc_ratio"] * np.sign(d["delta"].fillna(0))
    d["abs_ratio"] = d["mdc_ratio"].abs()
    d = d.sort_values("abs_ratio", ascending=False).head(max_rows)
    d = d.iloc[::-1]

    def _colour(row) -> str:
        good = row.get("toward_better")
        if row.get("verdict") not in (VERDICT_REAL, VERDICT_LIKELY):
            return GREY
        if good == 1:
            return GOOD
        if good == -1:
            return BAD
        return NEUTRAL

    colours = [_colour(r) for _, r in d.iterrows()]
    labels = [DISPLAY.short(m) for m in d["metric"]]
    hover = [
        (f"<b>{DISPLAY.name(r['metric'])}</b><br>"
         f"{r['from_value']:g} → {r['to_value']:g} {DISPLAY.unit(r['metric'])}"
         f"<br>change {r['delta']:+g} · usual variation {r['mdc']:g}"
         f"<br>{VERDICT_LABEL.get(r['verdict'], '')}<extra></extra>")
        for _, r in d.iterrows()
    ]
    fig = go.Figure(go.Bar(
        x=d["signed_ratio"], y=labels, orientation="h",
        marker=dict(color=colours),
        hovertemplate=hover,
        text=[DISPLAY.format_delta(r["metric"], r["delta"])
              for _, r in d.iterrows()],
        textposition="outside", textfont=dict(size=11),
        cliponaxis=False,
    ))
    lim = float(max(2.0, d["abs_ratio"].max() * 1.45))
    for x in (-1, 1):
        fig.add_vline(x=x, line_width=1, line_dash="dot", line_color=MUTED)
    fig.add_vrect(x0=-1, x1=1, line_width=0, fillcolor=GREY, opacity=0.09,
                  layer="below")
    fig.update_layout(**_layout(
        title=dict(text="How big was the move, for him?", x=0, font=dict(size=15)),
        xaxis=dict(title="change, as a multiple of his usual variation",
                   range=[-lim, lim], gridcolor=LINE, zerolinecolor=MUTED,
                   zerolinewidth=1.5),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        height=max(230, 34 * len(d) + 110),
        margin=dict(l=210, r=90, t=44, b=52),
    ))
    return fig_html(fig, first=first) + (
        "<div class='legend'>The grey strip is the range each measure moves "
        "across his own pitches. A bar inside it changed by less than the "
        "measure's own swing; a bar outside it moved further than it usually "
        "does. Green moved toward better, red away, blue means the metric has "
        "no agreed good direction.</div>")


def chart_cohort_strip(rows: Sequence[dict], *, first: bool = False) -> str:
    """One strip per metric: every comparable athlete as a dot, this athlete
    marked. Replaces a percentile column with the picture the percentile was
    trying to describe — including how few dots there are.

    Each row: {metric, athlete_value, cohort_values, cohort_label}
    """
    import plotly.graph_objects as go
    rows = [r for r in rows if r.get("cohort_values") is not None
            and len(r["cohort_values"]) >= 3]
    if not rows:
        return empty("Not enough comparable athletes to place him against.")

    n = len(rows)
    fig = go.Figure()
    labels = []
    for i, r in enumerate(rows):
        vals = pd.Series(list(r["cohort_values"]), dtype="float64").dropna()
        if vals.empty:
            continue
        lo, hi = float(vals.min()), float(vals.max())
        rng = (hi - lo) or 1.0
        scaled = (vals - lo) / rng
        x_ath = (float(r["athlete_value"]) - lo) / rng
        # A little deterministic vertical jitter so athletes with similar
        # values do not hide behind each other — with cohorts this small,
        # one overlapping dot is a meaningful fraction of the group.
        jitter = np.linspace(-0.16, 0.16, num=len(scaled)) if len(scaled) > 1 else [0.0]
        y = [i + float(j) for j in jitter]
        fig.add_trace(go.Scatter(
            x=scaled, y=y, mode="markers",
            marker=dict(size=9, color=GREY, opacity=0.5,
                        line=dict(width=0)),
            hovertemplate=(f"{DISPLAY.name(r['metric'])}<br>"
                           "another athlete: %{customdata:.3g}<extra></extra>"),
            customdata=vals.to_numpy(),
        ))
        fig.add_trace(go.Scatter(
            x=[x_ath], y=[i], mode="markers",
            marker=dict(size=15, color=NEUTRAL, symbol="diamond",
                        line=dict(width=2, color=CANVAS)),
            hovertemplate=(f"<b>{DISPLAY.name(r['metric'])}</b><br>"
                           f"this athlete: {r['athlete_value']:.3g} "
                           f"{DISPLAY.unit(r['metric'])}<extra></extra>"),
        ))
        labels.append(f"{DISPLAY.short(r['metric'])}  (n={len(vals)})")

    fig.update_layout(**_layout(
        title=dict(text="Where he sits in the group", x=0, font=dict(size=15)),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                   range=[-0.08, 1.08], title="lower ← group range → higher"),
        yaxis=dict(tickmode="array", tickvals=list(range(len(labels))),
                   ticktext=labels, gridcolor="rgba(0,0,0,0)",
                   autorange="reversed"),
        height=max(200, 46 * n + 100),
        margin=dict(l=240, r=40, t=44, b=52),
    ))
    return fig_html(fig, first=first) + (
        "<div class='legend'>Each grey dot is one comparable athlete we have "
        "tested; the blue diamond is him. Strips are scaled to the group's own "
        "range, so position — not spacing — is the message. Few dots means "
        "read it as a rank, not a percentile.</div>")


def chart_trial_scatter(focus: pd.DataFrame, metric: str, *,
                        first: bool = False) -> str:
    """Pitch-by-pitch scatter of one mechanic against velocity, with a fitted
    line. Only ever drawn for a relationship that already cleared the n floor
    and the scan-wide noise ceiling."""
    import plotly.graph_objects as go
    if focus is None or focus.empty or metric not in focus.columns:
        return empty("No trial data for this metric.")
    d = focus[[metric, "velocity_mph"]].dropna()
    if len(d) < 4:
        return empty("Too few pitches to plot.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d[metric], y=d["velocity_mph"], mode="markers",
        marker=dict(size=11, color=NEUTRAL, opacity=0.8,
                    line=dict(width=1.5, color=CANVAS)),
        hovertemplate=(f"{DISPLAY.short(metric)} %{{x:.3g}}"
                       "<br>%{y:.1f} mph<extra></extra>"),
    ))
    if d[metric].std() > 0:
        coef = np.polyfit(d[metric], d["velocity_mph"], 1)
        xs = np.linspace(d[metric].min(), d[metric].max(), 50)
        fig.add_trace(go.Scatter(
            x=xs, y=np.polyval(coef, xs), mode="lines",
            line=dict(color=MUTED, width=1.5, dash="dash"),
            hoverinfo="skip",
        ))
    unit = DISPLAY.unit(metric)
    fig.update_layout(**_layout(
        title=dict(text=f"{DISPLAY.name(metric)} vs velocity, pitch by pitch",
                   x=0, font=dict(size=15)),
        xaxis_title=f"{DISPLAY.short(metric)}" + (f" ({unit})" if unit else ""),
        yaxis_title="mph", height=330,
    ))
    return fig_html(fig, first=first)


def sparkline_svg(values: Sequence[float], *, width: int = 86, height: int = 22,
                  color: str = NEUTRAL) -> str:
    """Tiny inline trend for roster tables. SVG so it prints and needs no JS."""
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    if len(vals) < 2:
        return "<span class='empty'>—</span>"
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    step = width / (len(vals) - 1)
    pts = " ".join(
        f"{i * step:.1f},{height - 3 - ((v - lo) / rng) * (height - 6):.1f}"
        for i, v in enumerate(vals)
    )
    last_x = width
    last_y = height - 3 - ((vals[-1] - lo) / rng) * (height - 6)
    return (
        f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' "
        f"style='vertical-align:middle'>"
        f"<polyline points='{pts}' fill='none' stroke='{color}' "
        f"stroke-width='1.6' stroke-linejoin='round' stroke-linecap='round'/>"
        f"<circle cx='{last_x:.1f}' cy='{last_y:.1f}' r='2.4' fill='{color}'/>"
        f"</svg>"
    )


def page_header(title: str, subtitle_bits: Sequence[str],
                stamp: str) -> str:
    sub = " · ".join(esc(b) for b in subtitle_bits if b)
    return (f"<h1>{esc(title)}</h1>"
            f"<p class='subtitle'>{sub}</p>"
            f"<div class='stamp'>{esc(stamp)}</div>")


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M")


__all__ = [
    "OUTPUT_DIR", "write_html", "esc", "banner", "finding_card", "note",
    "empty", "details", "table", "verdict_chip", "page_header",
    "chart_velocity_trend", "chart_change_vs_noise", "chart_cohort_strip",
    "chart_trial_scatter", "sparkline_svg", "fig_html", "timestamp_slug",
    "set_plotly_mode", "plotly_mode",
    "GOOD", "BAD", "NEUTRAL", "GREY", "WARN", "SERIES",
]
