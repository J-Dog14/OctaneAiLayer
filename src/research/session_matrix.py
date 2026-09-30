"""
One table shape, used everywhere: metrics down the side, 3D sessions across.

The problem this replaces
-------------------------
The change table compared Round 1 to Round N. For Ryan Chasse that was
2024-09-10 against 2026-09-15 — 735 days with two captures sitting in between,
invisible. A metric that went up, came back down and ended where it started
showed as "no change"; one that moved 12 degrees at the second capture and held
showed as a change that looks like it happened last month. Neither is what
happened.

What this does instead
----------------------
The headline comparison is the LATEST capture against the one immediately
BEFORE it — the question a coach actually has after a session. Every other 3D
capture is still on the page, as its own column, so the number has its history
next to it:

    Metric              2024-09-10  2025-01-27  2025-10-24  2026-09-15   Change   x usual
    Max external rot         168.2       171.4       170.9       158.6    -12.3      2.8
    Hip-shoulder sep          41.0        43.2        42.8        43.1     +0.3      0.1

Read left to right and you can see whether the latest move is a departure from
a stable baseline or the continuation of a drift. That distinction is the whole
reason to keep the intermediate sessions on screen, and it is impossible to see
from a single first-to-last delta.

The same shape is used for kinetics, kinematics, timing and the appendix, so a
coach learns one table and reads them all.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from src.research.config import CONFIG, ResearchConfig
from src.research.metric_display import DISPLAY
from src.research.metric_filters import classify_family, filter_for_change
from src.research.reliability import (
    VERDICT_BEYOND,
    VERDICT_NO_BASELINE,
    ChangeVerdict,
    ReliabilityTable,
)


@dataclass
class SessionMatrix:
    """Metrics x sessions, with the latest-vs-previous change attached."""
    frame: pd.DataFrame                    # one row per metric
    session_dates: list[Any]               # chronological, oldest first
    session_trials: dict[Any, int]         # date -> pitches captured
    latest: Any | None
    previous: Any | None
    days_between: int | None
    excluded: dict[str, str] = field(default_factory=dict)
    blocked_pairs: dict[str, str] = field(default_factory=dict)
    coverage: dict[Any, int] = field(default_factory=dict)  # date -> metrics present

    def thin_captures(self, *, floor: float = 0.75) -> list[tuple[Any, int, int]]:
        """Captures carrying far fewer measures than the fullest one.

        A capture processed by an older pipeline, or exported without part of
        its signal set, shows up as a column of dashes. Without this the reader
        is left to guess whether the athlete was not measured or the data never
        made it in.
        """
        if not self.coverage:
            return []
        best = max(self.coverage.values() or [0])
        if best <= 0:
            return []
        return [(d, n, best) for d, n in self.coverage.items()
                if n < floor * best]

    @property
    def value_columns(self) -> list[str]:
        return [_col(d) for d in self.session_dates]

    @property
    def has_comparison(self) -> bool:
        return self.latest is not None and self.previous is not None

    def by_family(self) -> dict[str, pd.DataFrame]:
        if self.frame.empty:
            return {}
        return {fam: sub.reset_index(drop=True)
                for fam, sub in self.frame.groupby("family")}

    def movers(self, n: int = 8, *, curated_only: bool = True) -> pd.DataFrame:
        """Biggest movers between the last two captures."""
        if self.frame.empty:
            return self.frame
        f = self.frame
        if curated_only:
            c = f[f["metric"].map(DISPLAY.is_coach_ready)]
            f = c if not c.empty else f
        return (f.dropna(subset=["spread_multiple"])
                 .sort_values("spread_multiple", ascending=False)
                 .head(n).reset_index(drop=True))

    def display_columns(self) -> tuple[list[str], dict[str, str]]:
        """The canonical column order and headers, so every table matches."""
        cols = ["display_name"] + self.value_columns + [
            "change", "pct_change", "usual_spread", "spread_multiple", "trend"]
        rename = {"display_name": "Metric",
                  "change": "Change (last 2)",
                  "pct_change": "% change",
                  "usual_spread": "Usual variation",
                  "spread_multiple": "x usual",
                  "trend": "Trend"}
        for d in self.session_dates:
            rename[_col(d)] = str(d)
        return cols, rename


def _col(d: Any) -> str:
    return f"s_{d}"


def build_session_matrix(
    ctx,
    metrics: Sequence[str],
    *,
    config: ResearchConfig = CONFIG,
    max_sessions: int = 6,
) -> SessionMatrix:
    """Build the metrics x sessions table for one athlete.

    `ctx` is an AthleteContext. Sessions are the athlete's own 3D captures, in
    chronological order; the most recent `max_sessions` are kept so the table
    stays readable when someone has a long history.
    """
    trial_df = getattr(ctx, "trial_df", None)
    if trial_df is None or trial_df.empty:
        return SessionMatrix(pd.DataFrame(), [], {}, None, None, None)

    metrics, excluded = filter_for_change(list(metrics))
    metrics = [m for m in metrics if m in trial_df.columns]
    if not metrics:
        return SessionMatrix(pd.DataFrame(), [], {}, None, None, None,
                             excluded=excluded)

    dates = sorted(trial_df["session_date"].dropna().unique())
    if len(dates) > max_sessions:
        dates = dates[-max_sessions:]

    per_session = {
        d: trial_df[trial_df["session_date"] == d] for d in dates
    }
    n_trials = {d: int(len(g)) for d, g in per_session.items()}
    means = {d: g[metrics].mean() for d, g in per_session.items()}

    latest = dates[-1] if dates else None
    previous = dates[-2] if len(dates) >= 2 else None
    days_between = None
    if latest is not None and previous is not None:
        try:
            days_between = (pd.to_datetime(latest) - pd.to_datetime(previous)).days
        except Exception:
            days_between = None

    # Metrics whose scale changed between the two compared captures cannot be
    # differenced — the values still show in their columns, but the change
    # cell says so instead of reporting a fabricated delta.
    blocked: dict[str, str] = {}
    audit = getattr(ctx, "unit_audit", None)
    if audit is not None and previous is not None:
        try:
            blocked.update(audit.blocked_details_between(previous, latest))
        except Exception:
            pass

    reliability: ReliabilityTable = getattr(ctx, "reliability", None) or ReliabilityTable()

    rows: list[dict[str, Any]] = []
    coverage: dict[Any, int] = {d: 0 for d in dates}
    for m in metrics:
        rec: dict[str, Any] = {
            "metric": m,
            "display_name": DISPLAY.name(m),
            "short_name": DISPLAY.short(m),
            "family": classify_family(m),
            "unit": DISPLAY.unit(m),
        }
        series: list[float] = []
        for d in dates:
            v = means[d].get(m)
            val = float(v) if pd.notna(v) else None
            rec[_col(d)] = round(val, 3) if val is not None else None
            series.append(val if val is not None else np.nan)
        rec["series"] = series
        for d, v in zip(dates, series):
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                coverage[d] = coverage.get(d, 0) + 1

        v_prev = rec.get(_col(previous)) if previous is not None else None
        v_last = rec.get(_col(latest)) if latest is not None else None

        if m in blocked:
            rec.update(change=None, change_text="units changed", pct_change=None,
                       usual_spread=None, spread_multiple=None,
                       verdict=VERDICT_NO_BASELINE, toward_better=None,
                       provisional=False, blocked=blocked[m])
        elif v_prev is None or v_last is None:
            rec.update(change=None, change_text=None, pct_change=None,
                       usual_spread=None, spread_multiple=None,
                       verdict=VERDICT_NO_BASELINE, toward_better=None,
                       provisional=False, blocked=None)
        else:
            delta = v_last - v_prev
            cv: ChangeVerdict = reliability.classify(
                m, delta,
                n_before=n_trials.get(previous, 1),
                n_after=n_trials.get(latest, 1))
            verb, formatted = DISPLAY.describe_change(m, delta)
            rec.update(
                change=round(delta, 3),
                change_text=formatted,
                change_verb=verb,
                pct_change=(round(100.0 * delta / abs(v_prev), 1)
                            if v_prev != 0 else None),
                usual_spread=round(cv.mdc, 3) if cv.mdc is not None else None,
                spread_multiple=(round(cv.ratio, 2)
                                 if _usable_ratio(cv, v_prev, v_last) else None),
                verdict=(cv.verdict if _usable_ratio(cv, v_prev, v_last)
                         else VERDICT_NO_BASELINE),
                toward_better=DISPLAY.signed_toward_better(m, delta),
                provisional=cv.provisional,
                blocked=None,
            )
        rows.append(rec)

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["spread_multiple"] = pd.to_numeric(frame["spread_multiple"],
                                                 errors="coerce")
        frame = (frame.sort_values("spread_multiple", ascending=False,
                                   na_position="last")
                      .reset_index(drop=True))
        frame["trend"] = [_trend_cell(s) for s in frame["series"]]

    return SessionMatrix(
        frame=frame, session_dates=list(dates), session_trials=n_trials,
        latest=latest, previous=previous, days_between=days_between,
        excluded=excluded, blocked_pairs=blocked, coverage=coverage,
    )


def _usable_ratio(cv: ChangeVerdict, v_prev: float, v_last: float) -> bool:
    """Reject a band so small it is arithmetic noise rather than a baseline.

    Body weight is the case that exposed this: it is one number repeated on
    every pitch of a session, so its within-session spread is float dust and
    the ratio came out as 391,482,406,602,053x. A band has to be a real
    fraction of the quantity to mean anything.
    """
    if cv is None or cv.ratio is None or cv.mdc is None:
        return False
    scale = max(abs(v_prev or 0.0), abs(v_last or 0.0), 1e-12)
    return bool(cv.mdc > 1e-6 * scale)


def _trend_cell(series: Sequence[float]) -> str:
    """Inline sparkline so the shape of the history reads at a glance.

    A column of numbers tells you the values; the sparkline tells you whether
    the latest move continues a drift or breaks from a flat baseline, which is
    the thing the intermediate sessions were added to show.
    """
    from src.research import render_kit as rk
    vals = [v for v in series if v is not None and np.isfinite(v)]
    if len(vals) < 2:
        return ""
    return rk.sparkline_svg(vals, width=64, height=18)


def describe_change_window(sm: SessionMatrix) -> str:
    """One sentence naming exactly what is being compared."""
    if not sm.has_comparison:
        return ("Only one 3D capture on record, so there is nothing to compare "
                "it against yet.")
    span = f" ({sm.days_between} days apart)" if sm.days_between else ""
    n_prev = sm.session_trials.get(sm.previous, 0)
    n_last = sm.session_trials.get(sm.latest, 0)
    older = len(sm.session_dates) - 2
    history = (f" The {older} earlier capture{'s' if older != 1 else ''} "
               f"{'are' if older != 1 else 'is'} shown in the columns to the "
               f"left so you can see whether this is a break from his baseline "
               f"or the continuation of a drift.") if older > 0 else ""
    return (f"Change is <b>{sm.previous} → {sm.latest}</b>{span} — his last two "
            f"3D captures, {n_prev} and {n_last} pitches.{history}")


def build_assessment_matrix(ctx, *, config: ResearchConfig = CONFIG,
                            max_sessions: int = 6) -> SessionMatrix:
    """The same shape, for assessments.

    Columns are the athlete's assessment rounds — a 3D capture plus the testing
    around it — so this table lines up with the movement table above it and a
    coach reads one layout, not two. Values are RAW measurements, never
    Z-scores, and a value only appears in a round if it was actually measured
    inside that round's window; a carry-forward from an older test leaves the
    cell empty rather than pretending it was re-tested.
    """
    rounds = [r for r in (getattr(ctx, "rounds", None) or [])
              if r.profile_id is not None]
    if not rounds:
        return SessionMatrix(pd.DataFrame(), [], {}, None, None, None)
    rounds = rounds[-max_sessions:]

    snapshots = getattr(ctx, "snapshots", {}) or {}
    dates = [r.anchor_date for r in rounds]
    by_date = {r.anchor_date: snapshots.get(r.profile_id) for r in rounds}
    rounds_by_date = {r.anchor_date: r for r in rounds}

    stale_days = config.max_source_staleness_days
    metrics: set[str] = set()
    for snap in by_date.values():
        if snap is not None:
            metrics.update(snap.raw_values)

    latest = dates[-1] if dates else None
    previous = dates[-2] if len(dates) >= 2 else None
    days_between = None
    if latest is not None and previous is not None:
        try:
            days_between = (pd.to_datetime(latest) - pd.to_datetime(previous)).days
        except Exception:
            days_between = None

    rows: list[dict[str, Any]] = []
    for m in sorted(metrics):
        rec: dict[str, Any] = {
            "metric": m, "display_name": DISPLAY.name(m),
            "short_name": DISPLAY.short(m),
            "family": "assessment", "unit": DISPLAY.unit(m),
        }
        series: list[float] = []
        measured_on: dict[Any, Any] = {}
        for d in dates:
            snap = by_date.get(d)
            val = None
            if snap is not None:
                raw = snap.raw_values.get(m)
                src = snap.source_date_for(m)
                anchor = rounds_by_date[d].anchor_date
                gap = None
                if src is not None:
                    try:
                        gap = abs((pd.to_datetime(src) - pd.to_datetime(anchor)).days)
                    except Exception:
                        gap = None
                # Only count it if it was measured near this capture. A value
                # carried forward from an older test is not a re-test.
                if raw is not None and (gap is None or gap <= stale_days):
                    val = float(raw)
                    measured_on[d] = src
            rec[_col(d)] = round(val, 3) if val is not None else None
            series.append(val if val is not None else np.nan)
        rec["series"] = series
        rec["measured_on"] = measured_on

        v_prev = rec.get(_col(previous)) if previous is not None else None
        v_last = rec.get(_col(latest)) if latest is not None else None
        if v_prev is None or v_last is None:
            rec.update(change=None, change_text=None, pct_change=None,
                       usual_spread=None, spread_multiple=None,
                       verdict=VERDICT_NO_BASELINE, toward_better=None,
                       provisional=False, blocked=None)
        else:
            delta = v_last - v_prev
            verb, formatted = DISPLAY.describe_change(m, delta)
            rec.update(
                change=round(delta, 3), change_text=formatted, change_verb=verb,
                pct_change=(round(100.0 * delta / abs(v_prev), 1)
                            if v_prev != 0 else None),
                # Assessments have no trial-level repeats here, so there is no
                # within-athlete spread to size the move against. Percent
                # change is the honest scale.
                usual_spread=None, spread_multiple=None,
                verdict=VERDICT_NO_BASELINE,
                toward_better=DISPLAY.signed_toward_better(m, delta),
                provisional=False, blocked=None)
        rows.append(rec)

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["_abs"] = pd.to_numeric(frame["pct_change"], errors="coerce").abs()
        frame = (frame.sort_values("_abs", ascending=False, na_position="last")
                      .drop(columns=["_abs"]).reset_index(drop=True))
        frame["trend"] = [_trend_cell(s) for s in frame["series"]]

    return SessionMatrix(
        frame=frame, session_dates=list(dates),
        session_trials={d: 0 for d in dates},
        latest=latest, previous=previous, days_between=days_between)


def build_pitch_matrix(focus: pd.DataFrame, metrics: Sequence[str],
                       reliability, *, config: ResearchConfig = CONFIG,
                       max_pitches: int = 12) -> SessionMatrix:
    """The same shape again, one column per pitch inside a single session.

    A coach who has read the capture table above should not have to learn a
    second layout to read the session underneath it. So the pitches become the
    columns exactly as the captures did: metric on the left, every pitch as
    recorded, then the last-two difference sized against his usual swing.

    The band here is the WITHIN-session one — two pitches on the same day share
    the day, the marker set and the calibration, so the only error between them
    is pitch-to-pitch scatter.
    """
    from src.research.reliability import COMPARISON_WITHIN

    if focus is None or focus.empty:
        return SessionMatrix(pd.DataFrame(), [], {}, None, None, None)
    metrics = [m for m in metrics if m in focus.columns]
    if not metrics:
        return SessionMatrix(pd.DataFrame(), [], {}, None, None, None)

    ordered = (focus.sort_values("trial_index")
               if "trial_index" in focus.columns else focus)
    # Number the pitches by their order in the session, 1-based, because the
    # stored trial_index is 0-based in some captures and a coach counting
    # pitches does not start at zero. Numbering before any truncation keeps
    # "P14" meaning the fourteenth pitch even when only the last few are shown.
    labels_all = [f"P{i}" for i in range(1, len(ordered) + 1)]
    if len(ordered) > max_pitches:
        ordered = ordered.tail(max_pitches)
        labels_all = labels_all[-max_pitches:]
    labels: list[str] = labels_all

    latest = labels[-1] if labels else None
    previous = labels[-2] if len(labels) >= 2 else None

    rows: list[dict[str, Any]] = []
    for m in metrics:
        vals = [(None if pd.isna(v) else float(v))
                for v in ordered[m].tolist()]
        if sum(v is not None for v in vals) < 2:
            continue
        rec: dict[str, Any] = {
            "metric": m, "display_name": DISPLAY.name(m),
            "short_name": DISPLAY.short(m), "family": classify_family(m),
            "unit": DISPLAY.unit(m),
        }
        for lab, v in zip(labels, vals):
            rec[_col(lab)] = None if v is None else round(v, 3)
        rec["series"] = [np.nan if v is None else v for v in vals]

        present = [v for v in vals if v is not None]
        rec["low"] = round(min(present), 3)
        rec["high"] = round(max(present), 3)
        rec["spread"] = round(max(present) - min(present), 3)

        v_prev = rec.get(_col(previous)) if previous else None
        v_last = rec.get(_col(latest)) if latest else None
        if v_prev is None or v_last is None:
            rec.update(change=None, change_text=None, pct_change=None,
                       usual_spread=None, spread_multiple=None,
                       verdict=VERDICT_NO_BASELINE, toward_better=None,
                       provisional=False, blocked=None)
        else:
            delta = v_last - v_prev
            verb, formatted = DISPLAY.describe_change(m, delta)
            cv: ChangeVerdict = reliability.classify(
                m, delta, n_before=1, n_after=1, comparison=COMPARISON_WITHIN)
            rec.update(
                change=round(delta, 3), change_text=formatted, change_verb=verb,
                pct_change=(round(100.0 * delta / abs(v_prev), 1)
                            if v_prev != 0 else None),
                usual_spread=(round(cv.mdc, 3) if cv.mdc is not None else None),
                spread_multiple=(round(cv.ratio, 2)
                                 if cv.ratio is not None else None),
                verdict=cv.verdict,
                toward_better=DISPLAY.signed_toward_better(m, delta),
                provisional=bool(cv.provisional), blocked=None)
        rows.append(rec)

    frame = pd.DataFrame(rows)
    if not frame.empty:
        # Sorted by how far the pitches sit apart relative to his usual swing,
        # so the metrics he is least repeatable on come first.
        frame["_s"] = pd.to_numeric(frame["spread_multiple"],
                                    errors="coerce").abs()
        frame = (frame.sort_values("_s", ascending=False, na_position="last")
                      .drop(columns=["_s"]).reset_index(drop=True))
        frame["trend"] = [_trend_cell(s) for s in frame["series"]]

    return SessionMatrix(
        frame=frame, session_dates=labels,
        session_trials={lab: 0 for lab in labels},
        latest=latest, previous=previous, days_between=None)


# ──────────────────────────────────────────────────────────────────────────
# Assessment history — his own test dates, not the 3D calendar
# ──────────────────────────────────────────────────────────────────────────

# Modalities that are measured FROM a 3D capture. They belong in the movement
# tables; putting them in the assessment table was a bug that made pitching
# kinematics look like screen results.
_MOVEMENT_MODALITIES = {"pitching", "hitting", "pitching_force"}

ASSESSMENT_LABEL: dict[str, str] = {
    "athletic_screen_cmj": "Counter-movement jump",
    "athletic_screen_dj":  "Drop jump",
    "athletic_screen_ppu": "Plyo push-up",
    "athletic_screen_slv": "Single-leg vertical",
    "mobility":            "Mobility screen",
    "proteus_pitcher":     "Proteus",
    "proteus_hitter":      "Proteus (hitting)",
}
ASSESSMENT_ORDER = ["athletic_screen_cmj", "athletic_screen_dj",
                    "athletic_screen_ppu", "athletic_screen_slv",
                    "mobility", "proteus_pitcher", "proteus_hitter"]


def _modality_map() -> dict[str, str]:
    from src.metrics_spec import METRICS
    return {m["key"]: m["modality"] for m in METRICS}


def build_assessment_history(ctx, *, config: ResearchConfig = CONFIG,
                             max_dates: int = 6) -> dict[str, SessionMatrix]:
    """One matrix per assessment type, keyed on that assessment's OWN dates.

    Why this replaces the 3D-anchored version
    -----------------------------------------
    Assessment change used to be read off the rounds — a 3D capture plus the
    testing near it. For an athlete who screens quarterly and throws 3D twice a
    year that hides most of his testing: one real athlete had 29 of his 39 test
    dates fall outside every 3D window, and a whole year of screens after his
    last capture could never appear no matter how wide the window went.

    So the assessment table no longer asks when he was captured. Each type gets
    its own columns — the dates he actually took that test — and the change is
    the last two times THAT measure was taken. The round window still governs
    the mechanics-linking analysis, where pairing is the whole point.

    Values come from the profile snapshots, attributed to the session date the
    profiler recorded for that modality, so a value carried forward into a
    later profile lands in the column where it was measured instead of being
    counted twice.
    """
    snaps = sorted((getattr(ctx, "snapshots", {}) or {}).values(),
                   key=lambda s: s.as_of_date)
    if not snaps:
        return {}

    modality = _modality_map()
    # modality -> metric -> {date: value}
    collected: dict[str, dict[str, dict[Any, float]]] = {}
    for snap in snaps:
        for m, v in (snap.raw_values or {}).items():
            mod = modality.get(m)
            if mod is None or mod in _MOVEMENT_MODALITIES:
                continue
            d = snap.source_date_for(m) or snap.as_of_date
            if d is None:
                continue
            collected.setdefault(mod, {}).setdefault(m, {})[d] = float(v)

    out: dict[str, SessionMatrix] = {}
    for mod, metrics in collected.items():
        dates = sorted({d for vals in metrics.values() for d in vals})
        if len(dates) > max_dates:
            dates = dates[-max_dates:]
        latest = dates[-1] if dates else None
        previous = dates[-2] if len(dates) >= 2 else None

        rows: list[dict[str, Any]] = []
        for m, vals in metrics.items():
            rec: dict[str, Any] = {
                "metric": m, "display_name": DISPLAY.name(m),
                "short_name": DISPLAY.short(m), "family": mod,
                "unit": DISPLAY.unit(m),
            }
            series: list[float] = []
            for d in dates:
                v = vals.get(d)
                rec[_col(d)] = None if v is None else round(v, 3)
                series.append(np.nan if v is None else v)
            rec["series"] = series

            # The change is the last two times this measure was TAKEN, which
            # is not always the last two columns: an athlete can skip a screen.
            taken = [(d, vals[d]) for d in dates if vals.get(d) is not None]
            if len(taken) < 2:
                rec.update(change=None, change_text=None, pct_change=None,
                           usual_spread=None, spread_multiple=None,
                           verdict=VERDICT_NO_BASELINE, toward_better=None,
                           provisional=False, blocked=None,
                           measured_from=None, measured_to=None)
            else:
                (d_prev, v_prev), (d_last, v_last) = taken[-2], taken[-1]
                delta = v_last - v_prev
                verb, formatted = DISPLAY.describe_change(m, delta)
                rec.update(
                    change=round(delta, 3), change_text=formatted,
                    change_verb=verb,
                    pct_change=(round(100.0 * delta / abs(v_prev), 1)
                                if v_prev != 0 else None),
                    # No test-retest repeats for a screen, so there is no band
                    # to size the move against. Percent is the honest scale.
                    usual_spread=None, spread_multiple=None,
                    verdict=VERDICT_NO_BASELINE,
                    toward_better=DISPLAY.signed_toward_better(m, delta),
                    provisional=False, blocked=None,
                    measured_from=d_prev, measured_to=d_last)
            rows.append(rec)

        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame["_abs"] = pd.to_numeric(frame["pct_change"],
                                          errors="coerce").abs()
            frame = (frame.sort_values("_abs", ascending=False,
                                       na_position="last")
                          .drop(columns=["_abs"]).reset_index(drop=True))
            frame["trend"] = [_trend_cell(s) for s in frame["series"]]

        days_between = None
        if latest is not None and previous is not None:
            try:
                days_between = (pd.to_datetime(latest)
                                - pd.to_datetime(previous)).days
            except Exception:
                days_between = None

        out[mod] = SessionMatrix(
            frame=frame, session_dates=list(dates),
            session_trials={d: 0 for d in dates},
            latest=latest, previous=previous, days_between=days_between)

    return {k: out[k] for k in ASSESSMENT_ORDER if k in out} | {
        k: v for k, v in out.items() if k not in ASSESSMENT_ORDER}


__all__ = ["SessionMatrix", "build_session_matrix", "build_assessment_matrix",
           "build_assessment_history", "build_pitch_matrix",
           "ASSESSMENT_LABEL", "ASSESSMENT_ORDER", "describe_change_window"]
