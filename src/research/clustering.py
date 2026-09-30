"""
Archetype discovery via KMeans + PCA.

Goal: find natural deficit-pattern groups in your athletes without preset
categories. Answers questions like:
    - How many distinct pitcher archetypes exist in the corpus?
    - Do the archetypes align with mechanical vs mobility vs strength patterns?
    - Which athletes cluster together, and what distinguishes each cluster?

Method:
1. Restrict to numeric metric columns (drop metadata)
2. Impute missing (median per column) — biomech data has holes
3. Standardize (z-scores are already standardized, but this equalizes the
   imputed-vs-original distribution)
4. Optional PCA for viz (2D projection)
5. KMeans clustering (or auto-select k via silhouette if k=None)
6. Compute distinguishing metrics per cluster (t-score vs the rest)

Returns a dict with everything needed to render an interactive report.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.research.profile_matrix import _classify_metric, metric_columns


def _prep_matrix(df: pd.DataFrame,
                 restrict_to_domains: list[str] | None = None,
                 min_coverage: float = 0.5,
                 ) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """Prep the numeric matrix for clustering.

    Drops columns with < min_coverage non-null values, then imputes remaining
    NaNs with the column median and standardizes.

    Returns (X_scaled, feature_names, filtered_df_with_metadata).
    """
    cols = metric_columns(df)
    if restrict_to_domains:
        cols = [c for c in cols if _classify_metric(c) in restrict_to_domains]
    if not cols:
        raise ValueError("No metric columns match the requested domains.")
    # Column coverage filter
    coverage = df[cols].notna().mean()
    cols = [c for c in cols if coverage[c] >= min_coverage]
    if not cols:
        raise ValueError(
            f"No metric columns meet min_coverage={min_coverage}. "
            f"Try lowering the threshold or restricting to well-populated domains."
        )
    numeric = df[cols].astype(float)
    # Drop rows where ALL are null (edge case, coverage filter should prevent)
    valid_rows = numeric.notna().any(axis=1)
    numeric = numeric[valid_rows]
    df_filtered = df[valid_rows].reset_index(drop=True)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_imputed = imputer.fit_transform(numeric.values)
        X_scaled = scaler.fit_transform(X_imputed)
    return X_scaled, cols, df_filtered


def cluster_athletes(
    df: pd.DataFrame,
    *,
    k: int | None = None,               # None → auto-select via silhouette
    k_range: tuple[int, int] = (2, 7),
    restrict_to_domains: list[str] | None = None,
    min_coverage: float = 0.5,
    random_state: int = 42,
) -> dict:
    """Run KMeans clustering + PCA on the athlete profile matrix.

    Returns a dict with:
      k: selected number of clusters
      silhouette: silhouette score at that k
      labels: cluster label per athlete (indexed same as filtered df)
      centroids: DataFrame (k rows × features) of cluster centroids in the
                 SCALED feature space (z-space of imputed values)
      distinguishing: dict[cluster_id → DataFrame] of per-cluster top
                      distinguishing metrics (highest |t| vs everyone else)
      pca_2d: DataFrame (n_athletes × 3) with athlete_uuid, PC1, PC2 for viz
      feature_names: list of the metric columns used
      cohort: filtered df with a 'cluster' column appended (for downstream use)
      silhouette_by_k: dict[k → score] from the auto-search (if k was None)
    """
    X, cols, cohort = _prep_matrix(
        df, restrict_to_domains=restrict_to_domains, min_coverage=min_coverage
    )
    n = X.shape[0]
    if n < 4:
        raise ValueError(f"Only {n} athletes available; need at least 4 to cluster.")

    silhouette_by_k: dict[int, float] = {}
    if k is None:
        k_lo, k_hi = k_range
        k_hi = min(k_hi, n - 1)
        best_k, best_s = k_lo, -1.0
        for k_try in range(k_lo, k_hi + 1):
            km = KMeans(n_clusters=k_try, n_init=10, random_state=random_state)
            labels = km.fit_predict(X)
            if len(set(labels)) < 2:
                continue
            try:
                s = float(silhouette_score(X, labels))
            except Exception:
                continue
            silhouette_by_k[k_try] = round(s, 3)
            if s > best_s:
                best_s = s
                best_k = k_try
        k = best_k

    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = km.fit_predict(X)
    silhouette = float(silhouette_score(X, labels)) if len(set(labels)) >= 2 else None

    cohort = cohort.copy()
    cohort["cluster"] = labels

    # Centroids (in scaled space — z-scores of the imputed matrix)
    centroids = pd.DataFrame(km.cluster_centers_, columns=cols)
    centroids.index.name = "cluster"

    # Distinguishing metrics per cluster: for each metric, compare mean of
    # cluster to mean of everyone else. Use t-score approximation.
    X_df = pd.DataFrame(X, columns=cols)
    X_df["cluster"] = labels
    distinguishing: dict[int, pd.DataFrame] = {}
    for c in sorted(set(labels)):
        in_c = X_df[X_df["cluster"] == c][cols]
        out_c = X_df[X_df["cluster"] != c][cols]
        if len(in_c) == 0 or len(out_c) == 0:
            distinguishing[c] = pd.DataFrame()
            continue
        # Welch's t-approx: (mean1 - mean2) / sqrt(var1/n1 + var2/n2)
        m_in = in_c.mean()
        m_out = out_c.mean()
        v_in = in_c.var(ddof=1).replace(0, np.nan)
        v_out = out_c.var(ddof=1).replace(0, np.nan)
        se = np.sqrt(v_in / max(len(in_c), 1) + v_out / max(len(out_c), 1))
        t = (m_in - m_out) / se
        rank = pd.DataFrame({
            "metric": cols,
            "domain": [_classify_metric(c) for c in cols],
            "cluster_mean_z": m_in.values,
            "other_mean_z": m_out.values,
            "delta": (m_in - m_out).values,
            "t_score": t.values,
        })
        rank["abs_t"] = rank["t_score"].abs()
        distinguishing[c] = (rank.sort_values("abs_t", ascending=False)
                                 .drop(columns=["abs_t"])
                                 .head(15)
                                 .reset_index(drop=True))

    # PCA for 2D viz
    pca = PCA(n_components=2, random_state=random_state)
    pcs = pca.fit_transform(X)
    pca_df = pd.DataFrame({
        "athlete_uuid": cohort["athlete_uuid"].values,
        "name": cohort["name"].values,
        "cluster": labels,
        "PC1": pcs[:, 0],
        "PC2": pcs[:, 1],
    })

    return {
        "k": k,
        "silhouette": round(silhouette, 3) if silhouette is not None else None,
        "silhouette_by_k": silhouette_by_k,
        "labels": labels.tolist(),
        "centroids": centroids,
        "distinguishing": distinguishing,
        "pca_2d": pca_df,
        "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
        "feature_names": cols,
        "cohort": cohort,
        "n_athletes": n,
    }
