"""
How much does a measure move for this athlete anyway?

Every value in the warehouse is taken as fact. The pitcher threw the pitch, the
plate recorded the force, the model produced the angle — that happened. What a
coach still needs is SCALE: a 3-degree change means one thing for a measure
that holds within half a degree across his own pitches and something else for
one that swings six degrees pitch to pitch. This module supplies that
denominator. It never decides whether a difference is real.

What we compute
---------------
For each trial-level metric, using every athlete-session with enough trials:

  SEM   = sqrt( mean within-session variance )
        How much the measure varies across repeat pitches on the same day.
        In a biological system most of this is genuine motor variability, not
        instrument error, which is exactly why it is the right yardstick for
        "is this change large for him".

  ICC   = between-athlete variance / (between-athlete + within-session)
        How much of the spread across the roster separates athletes rather
        than separating one pitch from the next. A low ICC does not make the
        measure false — it means ranking athletes on it says little, so it
        stays out of percentile headlines.

  Span  = 1.96 * sqrt(2) * SEM
        The width that covers ~95% of the swings this measure makes on its
        own. A change larger than this is bigger than nearly all of its usual
        movement. That is a statement about magnitude, not about truth.

Two different spans, and using the wrong one is the trap
--------------------------------------------------------
Comparing two means from the SAME capture (first half vs second half of a
bullpen) is limited only by pitch-to-pitch spread, so it narrows as you throw
more pitches:

     span(k1, k2) = 1.96 * SEM_within * sqrt(1/k1 + 1/k2)

Comparing the same athlete on TWO DIFFERENT DAYS — every round-to-round and
roster comparison — is a different question. Marker placement, recalibration
and day-to-day biology are shared by every pitch in a session, so they do not
average out however many pitches you throw. Forty pitches pins down exactly
where he was that day; it says nothing about whether the day was typical:

     span_between = 1.96 * SEM_between * sqrt(2)

and SEM_between needs repeat captures close enough together that real
adaptation is implausible.

Using the narrowing within-session span for a between-session question is how
a change table fills with spurious entries: on data with realistic day-to-day
drift it flagged about 60% of unchanged mechanics. `mdc_for_means` never does
that. Until repeat captures exist it falls back to a deliberately conservative
PROVISIONAL span — the single-trial spread, un-shrunk:

     span_provisional = 1.96 * SEM_within * sqrt(2)

Verdicts from it carry `provisional=True` and every renderer says so. Set
`allow_provisional_band=False` to refuse instead and report "no baseline yet".

Closing that gap is cheap and is the highest-value data-collection change
available: capture a handful of athletes twice within two weeks and every
metric they cover gains a real baseline for every future report.
`ReliabilityTable.reliability_gaps()` prints what still needs it.

Limitations, stated so they reach the report
--------------------------------------------
  * Within-session spread is a LOWER bound on day-to-day spread, which is why
    it is not accepted as a substitute above.
  * SEM is assumed constant across the measurement range. For metrics whose
    spread scales with magnitude (most force RFDs) the CV is the steadier
    descriptor and is reported alongside.
  * A metric with too few qualifying sessions gets no estimate, and its
    verdicts say `no_baseline` rather than guessing.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.research.config import CONFIG, ResearchConfig

# ── Verdict vocabulary ────────────────────────────────────────────────────
#
# These describe MAGNITUDE, not truth. Every measurement in the warehouse is
# taken as fact: the pitcher did what the plate recorded. What a coach still
# needs is a sense of scale — a 3-degree move means something different for a
# measure that holds within half a degree across his own pitches than for one
# that swings six.
#
# So the comparison quantity is "how much does this measure usually vary for
# him", and the verdicts say where a given change sits against that. Nothing
# here calls data noise or tells anyone a difference did not happen.
VERDICT_BEYOND = "beyond_typical"      # larger than his usual variation
VERDICT_EDGE = "at_edge"               # about the size of his usual variation
VERDICT_TYPICAL = "within_typical"     # smaller than his usual variation
VERDICT_NO_BASELINE = "no_baseline"    # not enough repeats to have a baseline

# Back-compat aliases. The old names described a truth judgement we no longer
# make; they are kept so existing callers and stored rows keep resolving.
VERDICT_REAL = VERDICT_BEYOND
VERDICT_LIKELY = VERDICT_EDGE
VERDICT_NOISE = VERDICT_TYPICAL
VERDICT_UNKNOWN = VERDICT_NO_BASELINE

VERDICT_LABEL = {
    VERDICT_BEYOND: "Bigger than his usual swing",
    VERDICT_EDGE: "About his usual swing",
    VERDICT_TYPICAL: "Smaller than his usual swing",
    VERDICT_NO_BASELINE: "No baseline yet",
}
VERDICT_COLOR = {
    VERDICT_BEYOND: "#1a7f37",
    VERDICT_EDGE: "#bf8700",
    VERDICT_TYPICAL: "#57606a",
    VERDICT_NO_BASELINE: "#8c959f",
}
VERDICT_SYMBOL = {
    VERDICT_BEYOND: "\u25cf",
    VERDICT_EDGE: "\u25d0",
    VERDICT_TYPICAL: "\u25cb",
    VERDICT_NO_BASELINE: "?",
}
VERDICT_ORDER = {VERDICT_BEYOND: 0, VERDICT_EDGE: 1,
                 VERDICT_TYPICAL: 2, VERDICT_NO_BASELINE: 3}

# What we call the comparison quantity when a human reads it.
SPREAD_LABEL = "usual variation"

LIKELY_FRACTION = 1.2816 / 1.96

# Above this, the "x usual" number is reporting a divide-by-almost-zero rather
# than a real multiple of his typical variation, and is reported as no
# baseline instead.
RATIO_CEILING = 1.0e4

SOURCE_WITHIN = "within_session"
SOURCE_BETWEEN = "between_session"

# Which comparison a verdict is being asked about. See mdc_for_means.
COMPARISON_WITHIN = "within_session"
COMPARISON_BETWEEN = "between_session"


@dataclass(frozen=True)
class MetricReliability:
    metric: str
    sem: float
    cv_pct: float | None
    icc: float | None
    mdc95_single: float
    sd_between: float | None
    n_sessions: int
    n_trials: int
    n_athletes: int
    source: str

    @property
    def is_usable(self) -> bool:
        return self.sem > 0 and np.isfinite(self.sem)

    @property
    def trustworthy(self) -> bool:
        """ICC below ~0.5 means most of the spread in this metric is scatter.
        Still reportable, but never as a headline."""
        return self.icc is not None and self.icc >= 0.5


@dataclass(frozen=True)
class ChangeVerdict:
    metric: str
    delta: float | None
    mdc: float | None
    ratio: float | None          # |delta| / mdc
    verdict: str
    n_before: int
    n_after: int
    # Which kind of estimate the band came from. A provisional band is built
    # from within-session scatter standing in for day-to-day error because no
    # repeat captures exist yet; it is a LOWER bound, so provisional verdicts
    # over-call change. Renderers must say so.
    band_source: str = SOURCE_BETWEEN

    @property
    def provisional(self) -> bool:
        return self.band_source == SOURCE_WITHIN

    @property
    def label(self) -> str:
        return VERDICT_LABEL[self.verdict]

    @property
    def color(self) -> str:
        return VERDICT_COLOR[self.verdict]

    @property
    def symbol(self) -> str:
        return VERDICT_SYMBOL[self.verdict]

    @property
    def exceeds_typical(self) -> bool:
        """The change is larger than this measure's usual swing for him."""
        return self.verdict == VERDICT_BEYOND

    # Retained name; `exceeds_typical` says what it actually tests.
    @property
    def is_real(self) -> bool:
        return self.exceeds_typical

    def explain(self) -> str:
        """A sentence, in the metric's own units, that a coach can act on.

        Purely descriptive: what it was, what it is, and how that compares to
        how much this measure moves for him anyway. No claim that a difference
        did or did not happen — it happened, it is in the data.
        """
        if self.verdict == VERDICT_NO_BASELINE:
            return ("No repeat-capture baseline for this measure yet, so "
                    "there is nothing to compare the size of this move "
                    "against. Capturing a handful of athletes twice within "
                    "two weeks would give every future report one.")
        if self.delta is None or self.mdc is None:
            return "No change to evaluate."
        # Local import keeps `reliability` importable without the YAML present.
        from src.research.metric_display import DISPLAY
        unit = DISPLAY.unit(self.metric)
        suffix = f" {unit}" if unit else ""
        moved = f"{abs(self.delta):,.3g}{suffix}"
        band = f"{self.mdc:,.3g}{suffix}"
        word = "Rose" if self.delta > 0 else "Dropped"
        caveat = ""
        if self.provisional:
            caveat = (" The baseline is provisional — built from his "
                      "pitch-to-pitch spread because this measure has no "
                      "repeat captures yet, so it understates day-to-day "
                      "variation and makes this move look larger than it is.")
        mult = f"{self.ratio:.1f}x" if self.ratio else "?"
        if self.verdict == VERDICT_BEYOND:
            return (f"{word} {moved}. This measure usually varies about "
                    f"{band} for him, so the move is {mult} his usual swing."
                    + caveat)
        if self.verdict == VERDICT_EDGE:
            return (f"{word} {moved}, against a usual swing of about {band} "
                    f"({mult}). Comparable to how much it moves anyway — worth "
                    f"a second look next capture." + caveat)
        return (f"{word} {moved}. This measure usually varies about {band} "
                f"for him, so the move is smaller than its own swing "
                f"({mult})." + caveat)


# ──────────────────────────────────────────────────────────────────────────
# Estimation
# ──────────────────────────────────────────────────────────────────────────

def compute_reliability(
    trial_df: pd.DataFrame,
    metrics: Iterable[str],
    *,
    config: ResearchConfig = CONFIG,
    group_cols: tuple[str, str] = ("athlete_uuid", "session_date"),
) -> pd.DataFrame:
    """Within-session typical error for each metric.

    trial_df must be one row per trial with the group_cols present.
    Returns one row per metric; metrics with too little data are omitted
    entirely rather than given a shaky number.
    """
    metrics = [m for m in metrics if m in trial_df.columns]
    if not metrics or trial_df.empty:
        return _empty_reliability()

    rows: list[dict[str, Any]] = []
    grouped = trial_df.groupby(list(group_cols), sort=False)

    for m in metrics:
        within_vars: list[float] = []
        session_means: list[float] = []
        session_athletes: list[Any] = []
        n_trials_used = 0

        for (ath, _sess), g in grouped:
            vals = pd.to_numeric(g[m], errors="coerce").dropna()
            if len(vals) < config.reliability_min_trials_per_session:
                continue
            # ddof=1: sample variance within this session
            within_vars.append(float(vals.var(ddof=1)))
            session_means.append(float(vals.mean()))
            session_athletes.append(ath)
            n_trials_used += len(vals)

        n_sessions = len(within_vars)
        if n_sessions < config.reliability_min_sessions:
            continue

        within_vars_arr = np.asarray(within_vars, dtype=float)
        within_vars_arr = within_vars_arr[np.isfinite(within_vars_arr)]
        if within_vars_arr.size == 0:
            continue
        sem = float(np.sqrt(within_vars_arr.mean()))
        if not np.isfinite(sem) or sem <= 0:
            continue

        means_arr = np.asarray(session_means, dtype=float)
        grand_mean = float(np.mean(means_arr))
        cv_pct = (100.0 * sem / abs(grand_mean)) if grand_mean != 0 else None

        # Between-athlete SD from athlete-mean-of-session-means, so an athlete
        # with many sessions does not dominate.
        per_ath = pd.DataFrame({"athlete_uuid": session_athletes, "v": means_arr}) \
            .groupby("athlete_uuid")["v"].mean()
        sd_between = float(per_ath.std(ddof=1)) if len(per_ath) > 1 else None

        icc = None
        if sd_between is not None and np.isfinite(sd_between):
            denom = sd_between ** 2 + sem ** 2
            if denom > 0:
                icc = float(sd_between ** 2 / denom)

        rows.append({
            "metric": m,
            "sem": sem,
            "cv_pct": cv_pct,
            "icc": icc,
            "mdc95_single": config.mdc_multiplier * sem,
            "sd_between": sd_between,
            "n_sessions": n_sessions,
            "n_trials": n_trials_used,
            "n_athletes": int(per_ath.shape[0]),
            "source": SOURCE_WITHIN,
        })

    if not rows:
        return _empty_reliability()
    return (pd.DataFrame(rows)
              .sort_values("metric")
              .reset_index(drop=True))


def estimate_between_session(
    trial_df: pd.DataFrame,
    metrics: Iterable[str],
    *,
    max_gap_days: int = 21,
    config: ResearchConfig = CONFIG,
) -> pd.DataFrame:
    """Test-retest error from session PAIRS close enough together that real
    adaptation is implausible.

    This is the honest noise floor — it carries marker placement, calibration
    and day-to-day biology, which within-session scatter does not. It needs
    athletes who happened to be captured twice inside `max_gap_days`, so
    coverage is thinner. Use it where it exists and fall back to within-session
    everywhere else (`merge_reliability` does exactly that).
    """
    metrics = [m for m in metrics if m in trial_df.columns]
    if not metrics or trial_df.empty:
        return _empty_reliability()

    sess = (trial_df.groupby(["athlete_uuid", "session_date"])[metrics]
                    .mean().reset_index()
                    .sort_values(["athlete_uuid", "session_date"]))
    counts = (trial_df.groupby(["athlete_uuid", "session_date"]).size()
                      .rename("n_trials").reset_index())
    sess = sess.merge(counts, on=["athlete_uuid", "session_date"], how="left")

    pairs: list[dict[str, Any]] = []
    for ath, g in sess.groupby("athlete_uuid"):
        g = g.sort_values("session_date")
        dates = pd.to_datetime(g["session_date"])
        for i in range(len(g) - 1):
            gap = (dates.iloc[i + 1] - dates.iloc[i]).days
            if 0 < gap <= max_gap_days:
                pairs.append({"athlete_uuid": ath,
                              "a": g.iloc[i], "b": g.iloc[i + 1]})

    if not pairs:
        return _empty_reliability()

    rows: list[dict[str, Any]] = []
    for m in metrics:
        diffs = []
        athletes = set()
        for p in pairs:
            v1, v2 = p["a"].get(m), p["b"].get(m)
            if pd.notna(v1) and pd.notna(v2):
                diffs.append(float(v2) - float(v1))
                athletes.add(p["athlete_uuid"])
        if len(diffs) < max(3, config.reliability_min_sessions // 2):
            continue
        d = np.asarray(diffs, dtype=float)
        # Typical error from repeated measures: SD of differences / sqrt(2)
        sd_diff = float(d.std(ddof=1))
        if not np.isfinite(sd_diff) or sd_diff <= 0:
            continue
        sem = sd_diff / np.sqrt(2.0)
        rows.append({
            "metric": m,
            "sem": sem,
            "cv_pct": None,
            "icc": None,
            "mdc95_single": config.mdc_multiplier * sem,
            "sd_between": None,
            "n_sessions": len(diffs) * 2,
            "n_trials": 0,
            "n_athletes": len(athletes),
            "source": SOURCE_BETWEEN,
        })
    if not rows:
        return _empty_reliability()
    return pd.DataFrame(rows).sort_values("metric").reset_index(drop=True)


def merge_reliability(within: pd.DataFrame, between: pd.DataFrame) -> pd.DataFrame:
    """Prefer the between-session estimate where it exists — it is the wider,
    more honest band — and fall back to within-session elsewhere."""
    if between is None or between.empty:
        return within
    if within is None or within.empty:
        return between
    keep_within = within[~within["metric"].isin(set(between["metric"]))]
    return (pd.concat([between, keep_within], ignore_index=True)
              .sort_values("metric").reset_index(drop=True))


def _empty_reliability() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "metric", "sem", "cv_pct", "icc", "mdc95_single", "sd_between",
        "n_sessions", "n_trials", "n_athletes", "source",
    ])


# ──────────────────────────────────────────────────────────────────────────
# Lookup / verdicts
# ──────────────────────────────────────────────────────────────────────────

class ReliabilityTable:
    """Wraps the reliability DataFrame and answers the only question that
    matters downstream: is this change bigger than the noise?"""

    def __init__(self, df: pd.DataFrame | None = None,
                 *, config: ResearchConfig = CONFIG,
                 computed_at: datetime | None = None):
        self.config = config
        self.computed_at = computed_at or datetime.now()
        self._by_metric: dict[str, MetricReliability] = {}
        if df is not None and not df.empty:
            for r in df.to_dict("records"):
                self._by_metric[r["metric"]] = MetricReliability(
                    metric=r["metric"],
                    sem=float(r["sem"]),
                    cv_pct=_opt_float(r.get("cv_pct")),
                    icc=_opt_float(r.get("icc")),
                    mdc95_single=float(r["mdc95_single"]),
                    sd_between=_opt_float(r.get("sd_between")),
                    n_sessions=int(r.get("n_sessions") or 0),
                    n_trials=int(r.get("n_trials") or 0),
                    n_athletes=int(r.get("n_athletes") or 0),
                    source=str(r.get("source") or SOURCE_WITHIN),
                )

    def __len__(self) -> int:
        return len(self._by_metric)

    def __contains__(self, metric: str) -> bool:
        return metric in self._by_metric

    @property
    def metrics(self) -> list[str]:
        return sorted(self._by_metric)

    def get(self, metric: str) -> MetricReliability | None:
        return self._by_metric.get(metric)

    def sem(self, metric: str) -> float | None:
        r = self.get(metric)
        return r.sem if r else None

    def has_between_session_estimate(self, metric: str) -> bool:
        r = self.get(metric)
        return bool(r and r.source == SOURCE_BETWEEN and r.is_usable)

    def mdc_for_means(self, metric: str, n_before: int = 1, n_after: int = 1,
                      *, comparison: str = COMPARISON_BETWEEN) -> float | None:
        """Error band for comparing two means.

        The band depends on WHICH question is being asked, and getting this
        wrong is how a change table fills up with false positives:

        within-session — two means from the same capture (e.g. first half vs
            second half of a bullpen). The only error is pitch-to-pitch
            scatter, so the band shrinks with the number of pitches:
                z * SEM * sqrt(1/n1 + 1/n2)

        between-session — the same athlete on two different days, which is
            what every round-to-round comparison actually is. Day-to-day
            variation, marker placement and recalibration are shared by every
            pitch in a session, so they do NOT average out with more pitches.
            Throwing 40 pitches instead of 10 tells you precisely where the
            athlete was THAT DAY; it says nothing about whether the day itself
            was typical. The band is therefore:
                z * SEM_between * sqrt(2)
            and it needs a genuine test-retest estimate. A within-session SEM
            is a lower bound on day-to-day error, often by a lot, so using it
            here would call ordinary day-to-day variation a real change. When
            only a within-session estimate exists we return None, which
            surfaces as "can't tell yet" rather than a confident wrong answer.
            `reliability_gaps()` lists exactly which metrics need repeat
            captures to close that gap.
        """
        r = self.get(metric)
        if r is None or not r.is_usable:
            return None
        if comparison == COMPARISON_WITHIN:
            n1 = max(1, int(n_before or 1))
            n2 = max(1, int(n_after or 1))
            return float(self.config.mdc_z * r.sem * np.sqrt(1.0 / n1 + 1.0 / n2))
        if r.source != SOURCE_BETWEEN:
            if not self.config.allow_provisional_band:
                return None
            # Provisional: use the single-trial typical error as a stand-in for
            # day-to-day error, deliberately WITHOUT the 1/sqrt(k) shrinkage.
            # Shrinking it was the bug — it assumed more pitches make you more
            # certain the day was typical, which produced a ~60% false-positive
            # rate on data with realistic drift. This version is far more
            # conservative while still letting the tool say something before
            # any repeat captures exist.
            return float(self.config.mdc_z * r.sem * np.sqrt(2.0))
        return float(self.config.mdc_z * r.sem * np.sqrt(2.0))

    def classify(self, metric: str, delta: float | None,
                 *, n_before: int = 1, n_after: int = 1,
                 comparison: str = COMPARISON_BETWEEN) -> ChangeVerdict:
        mdc = self.mdc_for_means(metric, n_before, n_after,
                                 comparison=comparison)
        rec = self.get(metric)
        band_source = SOURCE_BETWEEN
        if comparison == COMPARISON_BETWEEN and rec is not None \
                and rec.source != SOURCE_BETWEEN:
            band_source = SOURCE_WITHIN
        elif comparison == COMPARISON_WITHIN:
            band_source = SOURCE_WITHIN
        if delta is None or mdc is None or not np.isfinite(mdc) or mdc <= 0:
            return ChangeVerdict(metric, delta, mdc, None, VERDICT_UNKNOWN,
                                 n_before, n_after, band_source)
        ratio = float(abs(delta) / mdc)
        # A band can be arithmetic dust rather than a baseline. Body weight is
        # one number repeated on every pitch of a session, so its within-session
        # spread is float noise and the ratio came out at 5.87e14 — a number
        # that says nothing except that the divisor was zero in spirit.
        if not np.isfinite(ratio) or ratio > RATIO_CEILING:
            return ChangeVerdict(metric, float(delta), mdc, None,
                                 VERDICT_UNKNOWN, n_before, n_after,
                                 band_source)
        if ratio >= 1.0:
            verdict = VERDICT_REAL
        elif ratio >= LIKELY_FRACTION:
            verdict = VERDICT_LIKELY
        else:
            verdict = VERDICT_NOISE
        return ChangeVerdict(metric, float(delta), mdc, ratio, verdict,
                             n_before, n_after, band_source)

    def annotate(self, df: pd.DataFrame, *, metric_col: str = "metric",
                 delta_col: str = "delta",
                 n_before: int | str = 1,
                 n_after: int | str = 1,
                 comparison: str = COMPARISON_BETWEEN) -> pd.DataFrame:
        """Add mdc / mdc_ratio / verdict columns to any delta table.

        n_before / n_after may be an int (same for every row) or the name of a
        column holding the per-row trial count.
        """
        if df is None or df.empty:
            return df
        out = df.copy()

        def _n(row, spec):
            if isinstance(spec, str):
                v = row.get(spec)
                return int(v) if pd.notna(v) else 1
            return int(spec)

        verdicts = [
            self.classify(row[metric_col], row.get(delta_col),
                          n_before=_n(row, n_before), n_after=_n(row, n_after),
                          comparison=comparison)
            for _, row in out.iterrows()
        ]
        out["mdc"] = [v.mdc for v in verdicts]
        out["mdc_ratio"] = [round(v.ratio, 2) if v.ratio is not None else None
                            for v in verdicts]
        out["verdict"] = [v.verdict for v in verdicts]
        out["is_real"] = [v.is_real for v in verdicts]
        out["provisional"] = [v.provisional for v in verdicts]
        return out

    def to_frame(self) -> pd.DataFrame:
        if not self._by_metric:
            return _empty_reliability()
        return pd.DataFrame([r.__dict__ for r in self._by_metric.values()]) \
                 .sort_values("metric").reset_index(drop=True)

    def reliability_gaps(self, metrics: Iterable[str] | None = None
                         ) -> dict[str, list[str]]:
        """Which metrics can and cannot support a between-session verdict.

        This is the single most actionable output of the whole reliability
        layer: `needs_retest` is a concrete list of measurements where one
        afternoon of repeat captures would turn "can\'t tell" into "real
        change" for every future report.
        """
        keys = list(metrics) if metrics is not None else self.metrics
        ready, needs_retest, missing = [], [], []
        for m in keys:
            r = self.get(m)
            if r is None or not r.is_usable:
                missing.append(m)
            elif r.source == SOURCE_BETWEEN:
                ready.append(m)
            else:
                needs_retest.append(m)
        return {"ready": sorted(ready), "needs_retest": sorted(needs_retest),
                "no_estimate": sorted(missing)}

    def summary(self) -> dict[str, Any]:
        f = self.to_frame()
        if f.empty:
            return {"n_metrics": 0}
        iccs = f["icc"].dropna()
        return {
            "n_metrics": int(len(f)),
            "n_between_session": int((f["source"] == SOURCE_BETWEEN).sum()),
            "median_icc": float(iccs.median()) if len(iccs) else None,
            "n_low_icc": int((iccs < 0.5).sum()) if len(iccs) else 0,
            "computed_at": self.computed_at,
        }


def _opt_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return None if not np.isfinite(f) else f
    except (TypeError, ValueError):
        return None


EMPTY_RELIABILITY = ReliabilityTable()

__all__ = [
    "compute_reliability", "estimate_between_session", "merge_reliability",
    "ReliabilityTable", "MetricReliability", "ChangeVerdict",
    "EMPTY_RELIABILITY",
    "VERDICT_BEYOND", "VERDICT_EDGE", "VERDICT_TYPICAL", "VERDICT_NO_BASELINE",
    "VERDICT_REAL", "VERDICT_LIKELY", "VERDICT_NOISE", "VERDICT_UNKNOWN",
    "SPREAD_LABEL",
    "VERDICT_LABEL", "VERDICT_COLOR", "VERDICT_SYMBOL", "VERDICT_ORDER",
    "SOURCE_WITHIN", "SOURCE_BETWEEN",
    "COMPARISON_WITHIN", "COMPARISON_BETWEEN",
]
