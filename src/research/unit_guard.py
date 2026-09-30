"""
Catch unit and scale changes before they get reported as athletic change.

Found in real data, Ryan Chasse, 2024-09-10 vs 2026-09-15:

    Back leg peak total ground force   2110  ->  1.95     ratio 1082
    Back leg peak ground force Z       1981  ->  1.83     ratio 1082
    Back leg peak ground force Y        729  ->  0.682    ratio 1069
    Back leg minimum ground force X    -214  -> -0.197    ratio 1086
    Front leg minimum ground force Y  -1200  -> -1.17     ratio 1026

Every ground-force metric in the newer capture is roughly a thousand times
smaller. That is a switch in what the pipeline writes — newtons to
bodyweights, or to kilonewtons — not a pitcher who lost 2,106 N of drive-leg
force. Reported as change it is nonsense; pooled into a velocity correlation
across the boundary it is worse, because the "effect" is enormous, perfectly
consistent, and entirely an artifact.

The detector is deliberately blunt, because unit changes are blunt:

  * compare session means for one metric across consecutive sessions
  * a jump of more than `min_ratio` (default 20x) in either direction is not a
    biological change in any biomechanical variable — no pitcher's peak force
    moves 20-fold
  * when several metrics that share a family jump by a SIMILAR ratio at the
    SAME session boundary, that is a pipeline change, not a set of coincidences

The second condition is what separates "a sensor broke on one channel" from
"the processing changed", and it is the one worth acting on.

This never silently drops data. It returns findings that the report states
out loud, and the change analysis refuses to compare across a flagged
boundary for the affected metrics — reporting "units changed here" instead of
a fabricated delta.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

# A biological variable does not move 20-fold between captures. Anything past
# this is a unit, scaling or pipeline change.
DEFAULT_MIN_RATIO = 20.0
# Ratios within this tolerance of each other count as "the same jump".
RATIO_CLUSTER_TOL = 0.25
# How many metrics must share a jump before we call it a pipeline change.
MIN_METRICS_FOR_SYSTEMIC = 3
# A unit change is permanent, so the new scale must hold on the far side of
# the boundary by at least this share of the jump. Below it, the "jump" is one
# capture's excursion — an angle through zero, a timing value at its event.
PERSISTENCE_FRACTION = 0.5

# Ratios worth naming when we see them.
KNOWN_CONVERSIONS = [
    (1000.0, "newtons to kilonewtons, or a factor-of-1000 scaling"),
    (9.80665, "kilograms to newtons (or the reverse) — a gravity factor"),
    (2.20462, "kilograms to pounds (or the reverse)"),
    (57.2958, "radians to degrees (or the reverse)"),
    (1000.0, "seconds to milliseconds (or the reverse)"),
    (0.0254, "inches to metres (or the reverse)"),
]


@dataclass
class ScaleBreak:
    """One metric whose scale jumped at one session boundary."""
    metric: str
    from_date: Any
    to_date: Any
    from_value: float
    to_value: float
    ratio: float          # always >= 1; direction carries the sign of change
    direction: str        # 'shrank' | 'grew'

    @property
    def magnitude(self) -> float:
        return self.ratio


@dataclass
class SystemicBreak:
    """Several metrics jumping by the same factor at the same boundary — a
    pipeline change rather than a measurement."""
    from_date: Any
    to_date: Any
    median_ratio: float
    direction: str
    metrics: list[str] = field(default_factory=list)
    likely_cause: str | None = None

    def describe(self) -> str:
        n = len(self.metrics)
        cause = f" That is consistent with {self.likely_cause}." if self.likely_cause else ""
        return (
            f"{n} metrics all {self.direction} by about {self.median_ratio:,.0f}x "
            f"between {self.from_date} and {self.to_date}. Measurements of a "
            f"pitcher do not move together by a constant factor, so this is a "
            f"change in what the processing pipeline writes, not a change in "
            f"the athlete.{cause} Comparisons across this date are withheld for "
            f"these metrics."
        )


@dataclass
class UnitAudit:
    breaks: list[ScaleBreak] = field(default_factory=list)
    systemic: list[SystemicBreak] = field(default_factory=list)

    @property
    def has_findings(self) -> bool:
        return bool(self.breaks or self.systemic)

    def blocked(self) -> set[tuple[str, Any, Any]]:
        """(metric, from_date, to_date) triples that must not be differenced."""
        out = {(b.metric, b.from_date, b.to_date) for b in self.breaks}
        for s in self.systemic:
            for m in s.metrics:
                out.add((m, s.from_date, s.to_date))
        return out

    def blocked_details_between(self, from_date: Any,
                                to_date: Any) -> dict[str, str]:
        """Metrics whose comparison across this window must be withheld, with
        the factor that earned the block.

        ONLY systemic breaks block. A single metric jumping 20x on its own is
        usually a small denominator — an angle that passed through zero, a
        timing value near the event it is measured from — not a unit change.
        Blocking those made a real report show "units changed" against
        2,136 N → 2,515 N, which is an ordinary session difference and reads
        as a bug to anyone who knows the data. Isolated jumps are still
        reported; they are just not treated as proof of a pipeline change.
        """
        out: dict[str, str] = {}
        for s in self.systemic:
            if _overlaps(s.from_date, s.to_date, from_date, to_date):
                for m in s.metrics:
                    out[m] = (f"scale changed by about {s.median_ratio:,.0f}x "
                              f"between {s.from_date} and {s.to_date}")
        return out

    def blocked_metrics_between(self, from_date: Any, to_date: Any) -> set[str]:
        """Back-compat wrapper over `blocked_details_between`."""
        return set(self.blocked_details_between(from_date, to_date))

    def isolated_breaks_between(self, from_date: Any,
                                to_date: Any) -> list[ScaleBreak]:
        """Single-metric jumps in this window — reported, never blocking."""
        systemic = {m for s in self.systemic for m in s.metrics}
        return [b for b in self.breaks
                if b.metric not in systemic
                and _overlaps(b.from_date, b.to_date, from_date, to_date)]

    def summary_lines(self) -> list[str]:
        lines = [s.describe() for s in self.systemic]
        systemic_metrics = {m for s in self.systemic for m in s.metrics}
        singles = [b for b in self.breaks if b.metric not in systemic_metrics]
        if singles:
            names = ", ".join(b.metric for b in singles[:6])
            more = f" and {len(singles) - 6} more" if len(singles) > 6 else ""
            lines.append(
                f"{len(singles)} further metric(s) moved more than 20x at a "
                f"session boundary on their own: {names}{more}. One metric "
                f"moving alone is usually a value that passed near zero rather "
                f"than a pipeline change, so these are still compared — read "
                f"them with an eye on the raw values.")
        return lines


def _overlaps(b_from, b_to, w_from, w_to) -> bool:
    """Does a break between b_from and b_to sit inside the window w_from..w_to?"""
    try:
        return bool(pd.to_datetime(w_from) <= pd.to_datetime(b_from)
                    and pd.to_datetime(b_to) <= pd.to_datetime(w_to))
    except Exception:
        return False


def _persists(g: pd.DataFrame, metric: str, i: int, min_ratio: float) -> bool:
    """Does the scale hold after the boundary, or snap back?

    Compares the median magnitude of every capture up to and including the
    boundary against every capture after it. A genuine unit change moves the
    whole series; a value that dipped near zero leaves the series where it was.
    """
    vals = pd.to_numeric(g[metric], errors="coerce").abs()
    pre = vals.iloc[:i + 1].dropna()
    post = vals.iloc[i + 1:].dropna()
    if pre.empty or post.empty:
        return True                      # nothing to test against — keep it
    a, b = float(pre.median()), float(post.median())
    if a <= 0 or b <= 0:
        return True
    held = max(a, b) / min(a, b)
    return held >= min_ratio * PERSISTENCE_FRACTION


def _name_conversion(ratio: float) -> str | None:
    for factor, label in KNOWN_CONVERSIONS:
        if factor <= 0:
            continue
        rel = abs(ratio - factor) / factor
        if rel <= 0.15:
            return label
    return None


def audit_scale_changes(
    trial_df: pd.DataFrame,
    metrics: Iterable[str],
    *,
    min_ratio: float = DEFAULT_MIN_RATIO,
    min_abs: float = 1e-9,
) -> UnitAudit:
    """Look for scale discontinuities between consecutive sessions.

    Works on session means, per athlete, then pools the boundaries: a pipeline
    change affects everyone captured after it, so a break seen in several
    athletes at the same date is the strongest possible signal. With a single
    athlete it still fires on the same-ratio-many-metrics rule.
    """
    audit = UnitAudit()
    metrics = [m for m in metrics if m in trial_df.columns]
    if not metrics or trial_df.empty:
        return audit

    sess = (trial_df.groupby(["athlete_uuid", "session_date"])[metrics]
                    .mean().reset_index()
                    .sort_values(["athlete_uuid", "session_date"]))

    # boundary -> metric -> list of ratios (one per athlete)
    by_boundary: dict[tuple, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list))

    for _, g in sess.groupby("athlete_uuid"):
        g = g.sort_values("session_date")
        if len(g) < 2:
            continue
        for i in range(len(g) - 1):
            a, b = g.iloc[i], g.iloc[i + 1]
            key = (a["session_date"], b["session_date"])
            for m in metrics:
                va, vb = a.get(m), b.get(m)
                if pd.isna(va) or pd.isna(vb):
                    continue
                va, vb = float(va), float(vb)
                if abs(va) < min_abs or abs(vb) < min_abs:
                    continue
                if not _persists(g, m, i, min_ratio):
                    # The jump does not hold: the metric goes back to its old
                    # magnitude at the next capture. A unit change is
                    # permanent — everything written after the boundary is on
                    # the new scale. A one-capture excursion is a value that
                    # passed near zero, which is arithmetic, not a pipeline
                    # change.
                    continue
                if np.sign(va) != np.sign(vb):
                    continue  # a sign flip is a different problem
                ratio = abs(va) / abs(vb) if abs(va) > abs(vb) else abs(vb) / abs(va)
                if ratio >= min_ratio:
                    by_boundary[key][m].append(
                        ratio if abs(va) > abs(vb) else -ratio)

    for (d_from, d_to), per_metric in by_boundary.items():
        # Record the individual breaks.
        metric_ratio: dict[str, float] = {}
        for m, ratios in per_metric.items():
            signed = float(np.median(ratios))
            metric_ratio[m] = signed
            audit.breaks.append(ScaleBreak(
                metric=m, from_date=d_from, to_date=d_to,
                from_value=float("nan"), to_value=float("nan"),
                ratio=abs(signed),
                direction="shrank" if signed > 0 else "grew",
            ))

        # Cluster metrics by similar ratio → systemic pipeline change.
        for direction, sign in (("shrank", 1), ("grew", -1)):
            vals = {m: abs(r) for m, r in metric_ratio.items()
                    if np.sign(r) == sign}
            if len(vals) < MIN_METRICS_FOR_SYSTEMIC:
                continue
            arr = np.array(sorted(vals.values()))
            med = float(np.median(arr))
            close = [m for m, r in vals.items()
                     if abs(r - med) / med <= RATIO_CLUSTER_TOL]
            if len(close) >= MIN_METRICS_FOR_SYSTEMIC:
                audit.systemic.append(SystemicBreak(
                    from_date=d_from, to_date=d_to,
                    median_ratio=med, direction=direction,
                    metrics=sorted(close),
                    likely_cause=_name_conversion(med),
                ))
    return audit


__all__ = ["audit_scale_changes", "UnitAudit", "ScaleBreak", "SystemicBreak",
           "DEFAULT_MIN_RATIO"]
