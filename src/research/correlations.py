"""
Cross-metric correlation analysis with statistical guardrails.

Two main entry points:

  correlate_target(df, target_metric, method='spearman', min_n=8)
      One-to-many: correlate ONE metric against every other. Returns ranked list.
      Use when you have a specific hypothesis: "what predicts hip-shoulder sep?"

  cross_domain_matrix(df, domain_a, domain_b, method='spearman', min_n=8)
      Full cross-modality matrix. Every metric in domain A vs every metric in
      domain B. Returns a DataFrame ready to plot as a heatmap.
      Use for hypothesis generation: "what does mobility relate to in 3D?"

Both apply Benjamini-Hochberg FDR correction so multiple-comparison inflation
is controlled.

At small n (biomech data is always small):
  - Spearman > Pearson (rank-based, robust to outliers)
  - Effect size matters more than p-value — with n=40 an |r|>0.5 is a real
    signal even before FDR correction
  - min_n guard drops pairs where too few athletes have both metrics
"""
from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

# Suppress ConstantInputWarning globally in this module — we already guard for
# zero-variance columns, but scipy still warns before we check.
warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)

from src.research.profile_matrix import (
    _classify_metric,
    columns_by_domain,
    metric_columns,
)


CorrelationMethod = Literal["spearman", "pearson"]


# ──────────────────────────────────────────────────────────────────────────
# Core correlation primitive
# ──────────────────────────────────────────────────────────────────────────

def _pairwise_corr(a: pd.Series, b: pd.Series, method: CorrelationMethod
                    ) -> tuple[float | None, float | None, int]:
    """Return (r, p, n) for one pair of series, using only rows where both are
    non-null. None values if too few observations."""
    mask = a.notna() & b.notna()
    n = int(mask.sum())
    if n < 3:
        return None, None, n
    x = a[mask].astype(float).values
    y = b[mask].astype(float).values
    # Zero-variance guard
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None, None, n
    try:
        if method == "spearman":
            r, p = stats.spearmanr(x, y)
        else:
            r, p = stats.pearsonr(x, y)
    except Exception:
        return None, None, n
    if isinstance(r, float) and (r != r):  # NaN check
        return None, None, n
    return float(r), float(p), n


def _bh_fdr(p_values: pd.Series, alpha: float = 0.10) -> pd.Series:
    """Benjamini-Hochberg FDR correction. Returns boolean Series `True` where
    the null (r=0) is rejected at the given FDR alpha. Also returns q-values.

    Not using statsmodels to avoid another dep."""
    p = p_values.dropna().sort_values()
    m = len(p)
    if m == 0:
        return pd.Series(dtype=bool)
    ranks = np.arange(1, m + 1)
    thresholds = (ranks / m) * alpha
    passes = p.values <= thresholds
    # BH: reject all p up to the largest index that passes
    if not passes.any():
        return pd.Series([False] * len(p_values), index=p_values.index)
    cutoff_idx = int(np.max(np.where(passes)[0]))
    cutoff_p = float(p.values[cutoff_idx])
    return p_values <= cutoff_p


def _bh_q_values(p_values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg q-values (FDR-adjusted p-values)."""
    p = p_values.copy()
    valid = p.notna()
    if not valid.any():
        return p
    idx = p[valid].sort_values().index
    ranked = p[valid].sort_values().values
    m = len(ranked)
    ranks = np.arange(1, m + 1)
    q = ranked * m / ranks
    # Enforce monotonicity from the back
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = p.copy()
    out.loc[idx] = q
    return out


# ──────────────────────────────────────────────────────────────────────────
# Target correlation: ONE metric vs all others
# ──────────────────────────────────────────────────────────────────────────

def correlate_target(
    df: pd.DataFrame,
    target_metric: str,
    *,
    method: CorrelationMethod = "spearman",
    min_n: int = 8,
    fdr_alpha: float = 0.10,
    exclude_same_domain: bool = False,
) -> pd.DataFrame:
    """Correlate one target metric against every other metric in the DataFrame.

    Returns a DataFrame sorted by |r| descending with columns:
        metric, domain, r, p_value, q_value, n, fdr_significant

    exclude_same_domain: when True, drop metrics that share the target's domain
    prefix. Useful when you want to focus on cross-modality relationships.
    """
    if target_metric not in df.columns:
        raise ValueError(f"{target_metric!r} is not a column in the profile matrix.")

    target_domain = _classify_metric(target_metric)
    metric_cols = metric_columns(df)
    if target_metric in metric_cols:
        metric_cols.remove(target_metric)

    rows: list[dict] = []
    for m in metric_cols:
        if exclude_same_domain and _classify_metric(m) == target_domain:
            continue
        r, p, n = _pairwise_corr(df[target_metric], df[m], method)
        if r is None or n < min_n:
            continue
        rows.append({
            "metric": m,
            "domain": _classify_metric(m),
            "r": r,
            "p_value": p,
            "n": n,
        })
    if not rows:
        return pd.DataFrame(columns=["metric", "domain", "r", "p_value",
                                     "q_value", "n", "fdr_significant"])
    out = pd.DataFrame(rows)
    out["q_value"] = _bh_q_values(out["p_value"])
    out["fdr_significant"] = out["q_value"] <= fdr_alpha
    out["abs_r"] = out["r"].abs()
    return (out.sort_values("abs_r", ascending=False)
               .drop(columns=["abs_r"])
               .reset_index(drop=True))


# ──────────────────────────────────────────────────────────────────────────
# Cross-domain matrix: full domain-A × domain-B correlation grid
# ──────────────────────────────────────────────────────────────────────────

def cross_domain_matrix(
    df: pd.DataFrame,
    domain_a: str,
    domain_b: str,
    *,
    method: CorrelationMethod = "spearman",
    min_n: int = 8,
    fdr_alpha: float = 0.10,
) -> dict[str, pd.DataFrame]:
    """Full cross-modality matrix. Every metric in domain_a × every metric in
    domain_b. Returns a dict with three DataFrames:
        'r'       — Spearman r values (may be NaN where under-powered)
        'p'       — raw p-values
        'q'       — BH-adjusted q-values
        'sig'     — boolean matrix, True where q ≤ fdr_alpha
        'n'       — sample size for each pair
    Rows = domain_a metrics, columns = domain_b metrics.

    domain_a / domain_b are the values produced by _classify_metric (e.g.,
    'mobility', 'pitching_3d', 'athletic_screen_dj', 'proteus').
    """
    by_domain = columns_by_domain(df)
    if domain_a not in by_domain:
        raise ValueError(f"Domain {domain_a!r} not present. Available: {sorted(by_domain)}")
    if domain_b not in by_domain:
        raise ValueError(f"Domain {domain_b!r} not present. Available: {sorted(by_domain)}")

    cols_a = by_domain[domain_a]
    cols_b = by_domain[domain_b]

    r_mat = pd.DataFrame(index=cols_a, columns=cols_b, dtype=float)
    p_mat = pd.DataFrame(index=cols_a, columns=cols_b, dtype=float)
    n_mat = pd.DataFrame(index=cols_a, columns=cols_b, dtype=int)

    for a in cols_a:
        for b in cols_b:
            r, p, n = _pairwise_corr(df[a], df[b], method)
            n_mat.at[a, b] = n
            if r is None or n < min_n:
                continue
            r_mat.at[a, b] = r
            p_mat.at[a, b] = p

    # Flatten p-values for FDR, then unflatten. In newer pandas, stack()
    # no longer accepts dropna=True; use .stack().dropna() instead.
    flat_p = p_mat.stack().dropna()
    flat_q = _bh_q_values(flat_p)
    q_mat = pd.DataFrame(index=cols_a, columns=cols_b, dtype=float)
    for (a, b), q in flat_q.items():
        q_mat.at[a, b] = q
    sig_mat = q_mat.le(fdr_alpha)

    return {
        "r": r_mat,
        "p": p_mat,
        "q": q_mat,
        "sig": sig_mat,
        "n": n_mat,
    }


def top_findings_from_matrix(
    matrices: dict[str, pd.DataFrame],
    *,
    top_n: int = 25,
    require_sig: bool = False,
) -> pd.DataFrame:
    """Flatten a cross-domain matrix into a ranked list of top findings."""
    r = matrices["r"]
    q = matrices["q"]
    n = matrices["n"]
    sig = matrices["sig"]

    rows = []
    for a in r.index:
        for b in r.columns:
            r_val = r.at[a, b]
            if pd.isna(r_val):
                continue
            rows.append({
                "metric_a": a,
                "metric_b": b,
                "r": float(r_val),
                "q_value": float(q.at[a, b]) if not pd.isna(q.at[a, b]) else None,
                "n": int(n.at[a, b]) if not pd.isna(n.at[a, b]) else None,
                "fdr_significant": bool(sig.at[a, b]) if not pd.isna(sig.at[a, b]) else False,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    if require_sig:
        out = out[out["fdr_significant"]]
    out["abs_r"] = out["r"].abs()
    return (out.sort_values("abs_r", ascending=False)
               .drop(columns=["abs_r"])
               .head(top_n)
               .reset_index(drop=True))
