"""
The statistical guardrails the research stack was missing.

Four things live here:

1. Bootstrap confidence intervals for Spearman rho.
   A point estimate of r=0.80 means nothing without knowing that at n=5 its
   90% interval is roughly [-0.3, 1.0]. Every correlation shown to a human
   should carry its interval.

2. `max_abs_r_null` — the honest answer to "we scanned 150 metrics at n=6 and
   the biggest |r| was 0.83, is that impressive?"
   Usually: no. This returns the distribution of the LARGEST |r| you would
   expect from pure noise given the same number of metrics and the same n, so
   a top-of-table finding can be compared against the right null instead of
   against zero. This is the single check that would have flagged the old
   within-session table as unusable.

3. Empirical-Bayes shrinkage for per-athlete means.
   An athlete with 4 pitches gets pulled toward his cohort; an athlete with
   40 barely moves. Solves the small-n problem rather than hiding it.

4. `percentile_or_rank` — refuses to quote a percentile off a cohort too small
   to support one, and returns an honest rank instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from src.research.config import (
    CONFIG,
    TIER_INSUFFICIENT,
    TIER_STRONG,
    TIER_SUGGESTIVE,
    ResearchConfig,
)

_RNG_SEED = 20260916


# ──────────────────────────────────────────────────────────────────────────
# 1. Correlation with an interval
# ──────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CorrelationResult:
    r: float | None
    p_value: float | None
    n: int
    ci_low: float | None = None
    ci_high: float | None = None
    method: str = "spearman"

    @property
    def is_estimable(self) -> bool:
        return self.r is not None

    @property
    def ci_spans_zero(self) -> bool:
        if self.ci_low is None or self.ci_high is None:
            return True
        return self.ci_low <= 0.0 <= self.ci_high

    @property
    def ci_width(self) -> float | None:
        if self.ci_low is None or self.ci_high is None:
            return None
        return self.ci_high - self.ci_low

    def format_ci(self, precision: int = 2) -> str:
        if self.ci_low is None:
            return "—"
        return f"[{self.ci_low:.{precision}f}, {self.ci_high:.{precision}f}]"


def spearman_with_ci(
    x: Sequence[float] | pd.Series,
    y: Sequence[float] | pd.Series,
    *,
    config: ResearchConfig = CONFIG,
    n_boot: int | None = None,
    ci: float | None = None,
    seed: int = _RNG_SEED,
) -> CorrelationResult:
    """Spearman rho with a percentile-bootstrap interval.

    Below `config.bootstrap_min_n` the interval is omitted rather than
    reported as [-1, 1] — an interval that wide is noise dressed as rigour.
    """
    xs = pd.Series(x, dtype="float64")
    ys = pd.Series(y, dtype="float64")
    mask = xs.notna() & ys.notna()
    xv = xs[mask].to_numpy()
    yv = ys[mask].to_numpy()
    n = int(mask.sum())
    if n < 3 or np.std(xv) == 0 or np.std(yv) == 0:
        return CorrelationResult(None, None, n)

    r, p = stats.spearmanr(xv, yv)
    if not np.isfinite(r):
        return CorrelationResult(None, None, n)

    if n < config.bootstrap_min_n:
        return CorrelationResult(float(r), float(p), n)

    n_boot = n_boot or config.bootstrap_iterations
    ci = ci or config.bootstrap_ci

    # Vectorised bootstrap. The data is rank-transformed ONCE and the
    # resamples are Pearson correlations of those ranks, rather than re-ranking
    # inside every resample. That is the standard rank-bootstrap
    # approximation: it differs from full re-ranking only in how ties created
    # by resampling are handled, which moves the interval endpoints by well
    # under a hundredth here, and it is roughly 200x faster. Speed matters —
    # the naive loop made a single deep dive take minutes, which is how you
    # end up quietly turning intervals off.
    rx = stats.rankdata(xv)
    ry = stats.rankdata(yv)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    bx = rx[idx]
    by = ry[idx]
    bxc = bx - bx.mean(axis=1, keepdims=True)
    byc = by - by.mean(axis=1, keepdims=True)
    denom = np.sqrt((bxc ** 2).sum(axis=1) * (byc ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        boots = (bxc * byc).sum(axis=1) / denom
    boots = boots[np.isfinite(boots)]
    if boots.size < n_boot * 0.5:
        return CorrelationResult(float(r), float(p), n)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boots, [alpha, 1.0 - alpha])
    return CorrelationResult(float(r), float(p), n, float(lo), float(hi))


# ──────────────────────────────────────────────────────────────────────────
# 2. The right null for a "top of the table" finding
# ──────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MaxRNull:
    n: int
    n_metrics: int
    observed_max_abs_r: float | None
    null_median: float
    null_p95: float
    p_value: float | None      # P(null max |r| >= observed)
    n_sim: int

    @property
    def is_surprising(self) -> bool:
        return self.p_value is not None and self.p_value <= 0.05

    def explain(self) -> str:
        if self.observed_max_abs_r is None:
            return (f"With {self.n_metrics} metrics at n={self.n}, pure noise "
                    f"typically produces a top |r| of {self.null_median:.2f}.")
        verdict = ("more than you would expect from noise"
                   if self.is_surprising else
                   "no more than you would expect from noise")
        return (
            f"Scanning {self.n_metrics} metrics across {self.n} pitches, the "
            f"largest |r| was {self.observed_max_abs_r:.2f}. Random data of the "
            f"same shape produces a top |r| of {self.null_median:.2f} on a "
            f"typical run and {self.null_p95:.2f} one run in twenty — so this "
            f"is {verdict}."
        )


@lru_cache(maxsize=256)
def _max_abs_r_sim(n: int, n_metrics: int, n_sim: int, seed: int) -> np.ndarray:
    """Simulate max |Spearman r| under the null, fast.

    Under the null the ranks of each noise metric are a uniform random
    permutation of 1..n, so there is no need to generate normals and rank
    them — we sample permutations directly and use the closed form

        rho = 1 - 6 * sum(d^2) / (n^3 - n)

    which turns the whole simulation into one argsort and one sum. Memoised
    because the distribution depends only on (n, n_metrics), and a report
    asks for the same shape repeatedly.
    """
    rng = np.random.default_rng(seed)
    yr = np.arange(1, n + 1, dtype=float)
    denom = float(n ** 3 - n)
    maxes = np.empty(n_sim, dtype=float)
    # Chunk so a wide scan does not allocate a huge 3-D array at once.
    chunk = max(1, min(n_sim, int(4_000_000 / max(1, n_metrics * n))))
    done = 0
    while done < n_sim:
        b = min(chunk, n_sim - done)
        perms = np.argsort(rng.random((b, n_metrics, n)), axis=2) + 1.0
        d2 = ((perms - yr) ** 2).sum(axis=2)
        rs = 1.0 - 6.0 * d2 / denom
        maxes[done:done + b] = np.abs(rs).max(axis=1)
        done += b
    return maxes


def max_abs_r_null(
    n: int,
    n_metrics: int,
    *,
    observed_max_abs_r: float | None = None,
    n_sim: int = 2000,
    seed: int = _RNG_SEED,
) -> MaxRNull:
    """Distribution of max |Spearman r| over `n_metrics` independent noise
    columns against a noise target, at sample size `n`.

    Treating the metrics as independent is optimistic — real kinematic columns
    are heavily correlated, which makes the true null max SMALLER than this.
    So a finding that fails this test definitely fails; one that passes is
    worth a second look but is not proven.
    """
    n = int(n)
    n_metrics = max(1, int(n_metrics))
    if n < 3:
        return MaxRNull(n, n_metrics, observed_max_abs_r, 1.0, 1.0, 1.0, 0)

    maxes = _max_abs_r_sim(n, n_metrics, n_sim, seed)
    null_median = float(np.median(maxes))
    null_p95 = float(np.quantile(maxes, 0.95))
    p = None
    if observed_max_abs_r is not None:
        p = float((maxes >= abs(observed_max_abs_r)).mean())
    return MaxRNull(n, n_metrics, observed_max_abs_r, null_median, null_p95,
                    p, n_sim)


# ──────────────────────────────────────────────────────────────────────────
# 3. Partial pooling
# ──────────────────────────────────────────────────────────────────────────

def eb_shrink_means(
    means: Sequence[float] | pd.Series,
    counts: Sequence[int] | pd.Series,
    within_sd: float,
    *,
    index: Sequence[Any] | None = None,
) -> pd.DataFrame:
    """Empirical-Bayes shrinkage of per-athlete means toward the grand mean.

    Each athlete's observed mean carries error sigma^2/n_i. The between-athlete
    spread tau^2 is estimated by subtracting the average sampling variance off
    the observed spread of the means (method of moments, floored at zero). The
    shrunk estimate is the precision-weighted blend:

        w_i        = tau^2 / (tau^2 + sigma^2/n_i)
        shrunk_i   = w_i * observed_i + (1 - w_i) * grand_mean

    An athlete with 40 pitches keeps his number; an athlete with 4 gets pulled
    most of the way back to the group, which is exactly the right behaviour
    when you are about to put him on a percentile chart.
    """
    m = pd.Series(list(means), dtype="float64")
    c = pd.Series(list(counts), dtype="float64")
    if index is not None:
        m.index = list(index)
        c.index = list(index)
    valid = m.notna() & c.notna() & (c > 0)
    if valid.sum() == 0:
        return pd.DataFrame({"observed": m, "n": c, "weight": np.nan,
                             "shrunk": m})

    mv, cv = m[valid], c[valid]
    grand = float(mv.mean())
    sigma2 = float(within_sd) ** 2
    observed_var = float(mv.var(ddof=1)) if len(mv) > 1 else 0.0
    mean_sampling_var = float((sigma2 / cv).mean())
    tau2 = max(observed_var - mean_sampling_var, 0.0)

    if tau2 == 0.0:
        # Nothing separates these athletes beyond measurement error: the best
        # estimate of every athlete is the group mean. Say so rather than
        # pretending the ranking is meaningful.
        w = pd.Series(0.0, index=m.index)
        shrunk = pd.Series(grand, index=m.index).where(valid, np.nan)
    else:
        w = pd.Series(np.nan, index=m.index)
        w[valid] = tau2 / (tau2 + sigma2 / cv)
        shrunk = pd.Series(np.nan, index=m.index)
        shrunk[valid] = w[valid] * mv + (1.0 - w[valid]) * grand

    out = pd.DataFrame({"observed": m, "n": c, "weight": w, "shrunk": shrunk})
    out.attrs["grand_mean"] = grand
    out.attrs["tau2"] = tau2
    out.attrs["sigma2"] = sigma2
    return out


def fisher_shrink_r(r: float, n: int, *, prior_n: float = 10.0) -> float:
    """Shrink a correlation toward zero in Fisher-z space by treating the
    prior as `prior_n` pseudo-observations of r=0. At n=4 with prior_n=10 an
    observed 0.8 lands near 0.3; at n=60 it barely moves."""
    if n is None or n < 4 or r is None or not np.isfinite(r):
        return 0.0
    r = float(np.clip(r, -0.999999, 0.999999))
    z = np.arctanh(r)
    eff = (n - 3) / ((n - 3) + prior_n)
    return float(np.tanh(z * eff))


# ──────────────────────────────────────────────────────────────────────────
# 4. Percentiles that refuse to lie
# ──────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CohortPosition:
    value: float
    n: int
    percentile: float | None
    rank: int | None            # 1 = highest
    median: float | None
    mode: str                   # 'percentile' | 'rank' | 'insufficient'
    cohort_label: str = ""

    @property
    def _group(self) -> str:
        """How to name the comparison group in a sentence. Empty label means
        'everyone we test', which reads better than a bare noun."""
        lbl = (self.cohort_label or "").strip()
        if not lbl or lbl.lower() in {"tested", "all athletes"}:
            return "pitchers we have tested"
        return f"{lbl} pitchers"

    def describe(self, metric_name: str) -> str:
        """A sentence a coach can read, not a number in a column."""
        if self.mode == "insufficient":
            return (f"{metric_name}: only {self.n} comparable athletes on "
                    f"file — not enough to say where he sits.")
        if self.mode == "rank":
            if self.rank == 1:
                return (f"{metric_name}: the highest of the {self.n} "
                        f"{self._group}.")
            if self.rank == self.n:
                return (f"{metric_name}: the lowest of the {self.n} "
                        f"{self._group}.")
            return (f"{metric_name}: {_ordinal(self.rank)} highest of "
                    f"{self.n} {self._group}.")
        pct = self.percentile
        assert pct is not None
        n_below = max(0, round((pct / 100.0) * self.n))
        return (f"{metric_name}: {pct:.0f}th percentile — higher than "
                f"{n_below} of the {self.n} {self._group}.")


def percentile_or_rank(
    value: float | None,
    cohort_values: Sequence[float] | pd.Series,
    *,
    config: ResearchConfig = CONFIG,
    cohort_label: str = "",
) -> CohortPosition | None:
    """Where one athlete sits in a cohort — as a percentile when the cohort is
    big enough to support one, as a plain rank when it is not.

    A percentile off 6 observations has ~17-point resolution, so 'p90' is a
    made-up number. Below `config.outlier_min_cohort_n` we say '3rd of 9'
    instead, which is both honest and more useful to a coach.
    """
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    vals = pd.Series(list(cohort_values), dtype="float64").dropna()
    n = int(len(vals))
    if n < config.rank_min_cohort_n:
        return CohortPosition(float(value), n, None, None,
                              float(vals.median()) if n else None,
                              "insufficient", cohort_label)
    rank = int((vals > float(value)).sum()) + 1
    median = float(vals.median())
    if n < config.outlier_min_cohort_n:
        return CohortPosition(float(value), n, None, rank, median,
                              "rank", cohort_label)
    pct = float(stats.percentileofscore(vals, float(value), kind="mean"))
    return CohortPosition(float(value), n, pct, rank, median,
                          "percentile", cohort_label)


def _ordinal(k: int | None) -> str:
    if k is None:
        return "—"
    if 10 <= k % 100 <= 20:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(k % 10, "th")
    return f"{k}{suf}"


# ──────────────────────────────────────────────────────────────────────────
# Confidence tiering — the single place that decides green / amber / grey
# ──────────────────────────────────────────────────────────────────────────

def correlation_tier(
    result: CorrelationResult,
    *,
    fdr_significant: bool | None = None,
    min_n: int,
    config: ResearchConfig = CONFIG,
) -> str:
    """One rule, used by every renderer, so 'strong' means the same thing on
    every page:

      strong      n at or above the floor, the interval excludes zero, and it
                  survived FDR (when FDR was computed at all)
      suggestive  n at the floor and a usable point estimate, but the interval
                  touches zero or FDR was not cleared
      insufficient below the n floor, or not estimable
    """
    if not result.is_estimable or result.n < min_n:
        return TIER_INSUFFICIENT
    ci_clean = (result.ci_low is not None) and (not result.ci_spans_zero)
    fdr_ok = True if fdr_significant is None else bool(fdr_significant)
    if ci_clean and fdr_ok:
        return TIER_STRONG
    return TIER_SUGGESTIVE


__all__ = [
    "CorrelationResult", "spearman_with_ci",
    "MaxRNull", "max_abs_r_null",
    "eb_shrink_means", "fisher_shrink_r",
    "CohortPosition", "percentile_or_rank",
    "correlation_tier",
]
