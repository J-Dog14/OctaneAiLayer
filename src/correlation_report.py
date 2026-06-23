"""
Build a population-level R&D report from ai_layer.athlete_profiles +
ai_layer.program_summaries.

Output: a single self-contained HTML file at outputs/correlation_report_<ts>.html
containing:
  1. Population summary (counts by age_group / thrower_type)
  2. Metric coverage (how many athletes have each metric)
  3. Correlation heatmap: profile Z-scores ↔ program features
  4. Archetype clusters (K-means on Z-score vectors)
  5. Per-archetype average programming patterns
  6. Top correlations (sorted, easy scan for "what coaches do when they see X")

Run:
    python -m src.main report
    python -m src.main report --k 6 --output my_report.html
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.db import backend_conn, query
from src.metrics_spec import metric_keys

# Output directory — defaults to repo-relative ./outputs/
OUTPUTS_DIR = Path(__file__).resolve().parents[1] / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────── Data loading ───────────────────────────────────

# Metric prefixes that classify each Z-column into a domain bucket
_MOVEMENT_PREFIXES   = ("z_pitch_", "z_hit_")
_FUNCTIONAL_PREFIXES = ("z_mob_", "z_proteus_", "z_screen_")


def _build_load_sql(role: str | None) -> str:
    """Build the joined-corpus SELECT with an optional role filter."""
    role_clause = ""
    if role == "pitcher":
        role_clause = "WHERE d.has_pitching_data = true"
    elif role == "hitter":
        role_clause = "WHERE d.has_hitting_data = true"
    return f"""
WITH latest_profile AS (
    SELECT DISTINCT ON (athlete_uuid) *
    FROM ai_layer.athlete_profiles
    ORDER BY athlete_uuid, as_of_date DESC
),
latest_program AS (
    SELECT DISTINCT ON (athlete_uuid)
        athlete_uuid, program_id, program_type, thrower_type, skill_level,
        enable_pitching, enable_hitting, duration_days,
        n_program_days, n_lift_prescriptions, n_unique_lifts,
        n_plyo_prescriptions, n_prep_prescriptions, n_bp_prescriptions,
        n_hit_prescriptions, n_me_prescriptions
    FROM ai_layer.program_summaries
    WHERE athlete_uuid IS NOT NULL
      AND (n_lift_prescriptions > 0
        OR n_plyo_prescriptions > 0
        OR n_prep_prescriptions > 0
        OR n_bp_prescriptions   > 0
        OR n_hit_prescriptions  > 0
        OR n_me_prescriptions   > 0)
    ORDER BY athlete_uuid, created_at_app DESC NULLS LAST
)
SELECT
    p.athlete_uuid, p.as_of_date,
    p.age_group AS profile_age_group,
    p.raw_values, p.z_scores,
    pr.program_id, pr.program_type, pr.thrower_type, pr.skill_level,
    pr.enable_pitching, pr.enable_hitting, pr.duration_days,
    pr.n_program_days, pr.n_lift_prescriptions, pr.n_unique_lifts,
    pr.n_plyo_prescriptions, pr.n_prep_prescriptions, pr.n_bp_prescriptions,
    pr.n_hit_prescriptions, pr.n_me_prescriptions,
    d.has_pitching_data, d.has_hitting_data
FROM latest_profile p
INNER JOIN latest_program pr USING (athlete_uuid)
INNER JOIN analytics.d_athletes d USING (athlete_uuid)
{role_clause}
"""


def load_corpus(role: str | None = None) -> pd.DataFrame:
    """Load joined corpus. role ∈ {'pitcher', 'hitter', None (=both)}."""
    with backend_conn() as conn:
        rows = query(conn, _build_load_sql(role))
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Expand the z_scores JSONB into z_<metric> columns
    z = pd.json_normalize(df["z_scores"]).add_prefix("z_")
    df = pd.concat([df.drop(["z_scores", "raw_values"], axis=1), z], axis=1)
    return df


def _filter_z_cols_by_domain(z_cols: list[str], domain: str | None) -> list[str]:
    """domain ∈ {'movement', 'functional', None (=all)}."""
    if domain == "movement":
        return [c for c in z_cols if c.startswith(_MOVEMENT_PREFIXES)]
    if domain == "functional":
        return [c for c in z_cols if c.startswith(_FUNCTIONAL_PREFIXES)]
    return z_cols


# ─────────────────────────── Analysis helpers ───────────────────────────────

def cluster_athletes(df: pd.DataFrame, z_cols: list[str], k: int = 5
                     ) -> tuple[pd.Series, np.ndarray, list[str]]:
    """K-means on Z-score vectors. Drops athletes with >50% missing metrics.

    Returns:
      mask: boolean Series, True where the athlete was clustered
      labels: cluster ids (length = sum(mask))
      used_cols: metrics actually used (excludes any column that's >70% null overall)
    """
    # Drop metrics that are basically empty across the whole population
    coverage = df[z_cols].notna().mean()
    used_cols = coverage[coverage >= 0.30].index.tolist()
    X = df[used_cols].copy()

    # Drop athletes missing >50% of the remaining metrics
    mask = X.isna().sum(axis=1) <= len(used_cols) * 0.5

    # Impute remaining missing values with 0 (= "at age-group mean")
    X_imp = X[mask].fillna(0)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_imp)

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(Xs)
    return mask, labels, used_cols


def correlation_matrix(df: pd.DataFrame, z_cols: list[str], prog_cols: list[str]
                       ) -> pd.DataFrame:
    """Pearson correlation: every Z-score vs every program feature. NaN-tolerant."""
    sub = df[z_cols + prog_cols].apply(pd.to_numeric, errors="coerce")
    # Use min_periods so we only compute correlations with enough overlap
    full = sub.corr(method="pearson", min_periods=15)
    return full.loc[z_cols, prog_cols]


def top_correlations(corr: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Flatten the correlation matrix and return the n strongest (by absolute value)."""
    long = (
        corr.stack(future_stack=True).rename("r").reset_index()
            .rename(columns={"level_0": "z_metric", "level_1": "program_feature"})
    )
    long["abs_r"] = long["r"].abs()
    long = long.dropna().sort_values("abs_r", ascending=False).head(n)
    return long.drop(columns="abs_r").reset_index(drop=True)


def archetype_summary(df: pd.DataFrame, mask: pd.Series, labels: np.ndarray,
                       z_cols: list[str], prog_cols: list[str]
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean Z-score and mean programming per archetype."""
    sub = df[mask].copy()
    sub["archetype"] = labels
    by_z = sub.groupby("archetype")[z_cols].mean().round(2)
    by_p = sub.groupby("archetype")[prog_cols].mean().round(1)
    by_p["n_athletes"] = sub.groupby("archetype").size()
    return by_z, by_p


def _pretty_metric(z_col: str) -> str:
    """Turn 'z_pitch_max_external_rotation' into 'pitch max external rotation'."""
    s = z_col.removeprefix("z_").replace("_", " ")
    s = s.replace("rotation", "rot").replace("horizontal", "horiz").replace("velocity", "vel")
    return s


def label_archetypes(by_z: pd.DataFrame,
                     pos_threshold: float = 0.4,
                     neg_threshold: float = -0.4,
                     max_terms: int = 4) -> dict[int, str]:
    """Auto-generate a short descriptor for each archetype based on its extreme metrics.

    Example output:  '↑ DJ RSI, ↑ max ER, ↓ shoulder IR'
    """
    out: dict[int, str] = {}
    for arch_id, row in by_z.iterrows():
        ordered = row.dropna().sort_values()
        positives = ordered[ordered >= pos_threshold].tail(max_terms)
        negatives = ordered[ordered <= neg_threshold].head(max_terms)
        bits = []
        for k, v in negatives.items():
            bits.append(f"↓ {_pretty_metric(k)} ({v:+.1f})")
        for k, v in positives[::-1].items():
            bits.append(f"↑ {_pretty_metric(k)} ({v:+.1f})")
        out[int(arch_id)] = " · ".join(bits) if bits else "near-population-average"
    return out


# ─────────────────────────── Chart builders ─────────────────────────────────

def fig_heatmap(corr: pd.DataFrame) -> go.Figure:
    fig = px.imshow(
        corr, color_continuous_scale="RdBu_r", zmin=-0.6, zmax=0.6,
        aspect="auto", title="Profile Z-score ↔ Program feature correlations",
    )
    fig.update_layout(height=max(450, 22 * len(corr)), margin=dict(l=200, r=40, t=60, b=80))
    return fig


def fig_archetype_parallel(by_z: pd.DataFrame) -> go.Figure:
    """Parallel coordinates of mean Z-scores per archetype."""
    df_long = by_z.reset_index().melt(id_vars="archetype",
                                      var_name="metric", value_name="mean_z")
    fig = px.line(
        df_long, x="metric", y="mean_z", color="archetype",
        markers=True, title="Archetype mean Z-score profile (each line = one archetype)",
    )
    fig.add_hline(y=0, line_dash="dot", line_color="grey")
    fig.update_layout(height=520, xaxis_tickangle=-45, margin=dict(l=60, r=40, t=60, b=140))
    fig.update_xaxes(title="")
    fig.update_yaxes(title="Mean Z-score (population)")
    return fig


def fig_archetype_scatter(df: pd.DataFrame, mask: pd.Series, labels: np.ndarray,
                          used_cols: list[str]) -> go.Figure:
    """2D PCA projection colored by archetype."""
    X = df[mask][used_cols].fillna(0).values
    if X.shape[0] < 2:
        return go.Figure()
    Xs = StandardScaler().fit_transform(X)
    pcs = PCA(n_components=2, random_state=42).fit_transform(Xs)

    plot_df = pd.DataFrame({
        "PC1": pcs[:, 0], "PC2": pcs[:, 1],
        "archetype": labels.astype(str),
        "athlete_uuid": df[mask]["athlete_uuid"].values,
        "age_group": df[mask]["profile_age_group"].values,
    })
    fig = px.scatter(
        plot_df, x="PC1", y="PC2", color="archetype",
        hover_data=["athlete_uuid", "age_group"],
        title="Athletes projected to 2D (PCA), colored by archetype",
    )
    fig.update_layout(height=520)
    return fig


def fig_pop_by_age(df: pd.DataFrame) -> go.Figure:
    counts = (df.groupby("profile_age_group").size()
              .reset_index(name="athletes")
              .sort_values("athletes", ascending=False))
    fig = px.bar(counts, x="profile_age_group", y="athletes",
                 title="Athletes in corpus by age group", text_auto=True)
    fig.update_layout(height=320)
    return fig


# ─────────────────────────── HTML rendering ─────────────────────────────────

_PAGE_CSS = """
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         max-width: 1200px; margin: 24px auto; padding: 0 16px; color: #222; }
  h1 { border-bottom: 2px solid #333; padding-bottom: 6px; }
  h2 { margin-top: 36px; color: #1a4a82; }
  table { border-collapse: collapse; margin: 16px 0; }
  th, td { border: 1px solid #ccc; padding: 6px 10px; font-size: 13px; }
  th { background: #f0f4f8; text-align: left; }
  .meta { color: #666; font-size: 13px; }
  .note { background: #fff8d6; padding: 8px 12px; border-left: 4px solid #f1c40f;
          margin: 12px 0; font-size: 13px; }
  pre { background: #f6f8fa; padding: 12px; border-radius: 4px; overflow-x: auto; }
</style>
"""


def _table_html(df: pd.DataFrame, caption: str = "") -> str:
    out = df.to_html(border=0, classes="report-table", justify="left", float_format="%.2f")
    if caption:
        out = f"<p class='meta'>{caption}</p>" + out
    return out


def render_html(
    df: pd.DataFrame, *, k: int,
    used_cols: list[str], mask: pd.Series, labels: np.ndarray,
    corr: pd.DataFrame, top_corr: pd.DataFrame,
    by_z: pd.DataFrame, by_p: pd.DataFrame,
    archetype_labels: dict[int, str] | None = None,
    role: str | None = None, domain: str | None = None,
) -> str:
    archetype_labels = archetype_labels or {}
    slice_tag = f"role={role or 'all'} · domain={domain or 'all'}"
    n_total = len(df)
    n_clustered = int(mask.sum())
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    pop_fig = fig_pop_by_age(df).to_html(full_html=False, include_plotlyjs="cdn")
    heatmap = fig_heatmap(corr).to_html(full_html=False, include_plotlyjs=False)
    parallel = fig_archetype_parallel(by_z).to_html(full_html=False, include_plotlyjs=False)
    scatter = fig_archetype_scatter(df, mask, labels, used_cols).to_html(
        full_html=False, include_plotlyjs=False)

    # Build the archetype-programming table with descriptive labels
    by_p_show = by_p.copy()
    by_p_show.insert(
        0, "label",
        ["Archetype %d — %s" % (i, archetype_labels.get(int(i), "")) for i in by_p_show.index],
    )
    by_p_show = by_p_show.set_index("label")
    archetype_tbl = by_p_show.to_html(border=0, justify="left", float_format="%.1f")

    # Standalone "what each archetype is" cheat sheet
    if archetype_labels:
        cheat_rows = "".join(
            f"<tr><td><b>Archetype {i}</b></td><td>{lbl}</td>"
            f"<td>{int(by_p.loc[i, 'n_athletes']) if i in by_p.index else 0}</td></tr>"
            for i, lbl in sorted(archetype_labels.items())
        )
        cheat_html = (
            "<table><tr><th>Archetype</th><th>Distinguishing profile</th>"
            "<th>N athletes</th></tr>" + cheat_rows + "</table>"
        )
    else:
        cheat_html = ""

    top_corr_tbl = top_corr.to_html(border=0, justify="left", float_format="%.3f", index=False)

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>8ctane R&D Report — {ts}</title>
{_PAGE_CSS}
</head><body>

<h1>8ctane R&amp;D Correlation Report — {slice_tag}</h1>
<p class="meta">Generated: {ts}<br>
Slice: {slice_tag}<br>
Corpus: {n_total} athletes with a profile + non-empty program. Clustering used {n_clustered} of them
(athletes missing &gt;50% of metrics were excluded). K = {k}.</p>

<div class="note"><b>How to read this:</b> Z-scores are deviations from the athlete's
age-group population. Negative = below the group mean (a deficit), positive = above.
The correlation column "r" is Pearson; a positive r between a Z-score and a program
feature means coaches tended to <i>prescribe more</i> of that feature when the athlete
was <i>above</i> the population mean on that metric.</p></div>

<h2>1. Population</h2>
{pop_fig}

<h2>2. Profile ↔ Program correlations</h2>
{heatmap}

<h2>3. Top 20 strongest correlations</h2>
<p class="meta">Strongest associations between a profile Z-score and a program feature.
Use these as candidate hypotheses for "this kind of athlete tends to get this kind of program."</p>
{top_corr_tbl}

<h2>4. Athlete archetypes (K = {k})</h2>
<p class="meta">K-means clustered the athletes on their full Z-score vector. Each line below
is one archetype's <i>average</i> profile. Where lines diverge is where archetypes meaningfully
differ. Labels below are auto-generated from each cluster's most extreme metrics
(Z &gt; +0.4 or Z &lt; -0.4).</p>
{cheat_html}
{parallel}
{scatter}

<h2>5. Typical programming per archetype</h2>
<p class="meta">Average program characteristics for athletes in each cluster.
This is the closest thing to "what coaches do for this kind of athlete" we can derive without
yet modeling individual exercise choices.</p>
{archetype_tbl}

</body></html>"""


# ─────────────────────────── Orchestrator ───────────────────────────────────

def generate_report(k: int = 5, output: str | None = None,
                    role: str | None = None, domain: str | None = None) -> Path:
    """Generate the population correlation + archetype report.

    role:   None=all athletes, 'pitcher'=has_pitching_data, 'hitter'=has_hitting_data
    domain: None=all metrics, 'movement'=pitch_/hit_ only, 'functional'=mob_/proteus_/screen_
    """
    print(f"[report] loading corpus (role={role or 'all'})...")
    df = load_corpus(role=role)
    if df.empty:
        raise RuntimeError(
            f"No rows in the joined corpus for role={role}. "
            "Check ai_layer.athlete_profiles and ai_layer.program_summaries."
        )

    # Profiles' JSONB may contain stale z_* keys from previously-removed metrics.
    # Filter to only the keys currently declared in metrics_spec.METRICS so
    # spec changes are reflected immediately without requiring a re-backfill.
    current_spec_z_cols = {f"z_{k}" for k in metric_keys()}
    all_z_cols = [c for c in df.columns
                  if c.startswith("z_") and c in current_spec_z_cols]
    z_cols = _filter_z_cols_by_domain(all_z_cols, domain)
    if not z_cols:
        raise RuntimeError(f"No Z-metrics survived domain filter '{domain}'.")

    # Program columns (drop ones that are 100% null in this corpus, e.g. goals)
    prog_cols = [c for c in df.columns
                 if (c.startswith("n_") or c == "duration_days") and df[c].notna().any()]

    print(f"[report] {len(df)} athletes, {len(z_cols)} Z-metrics (domain={domain or 'all'}), "
          f"{len(prog_cols)} program features (after dropping all-null)")

    print("[report] clustering...")
    mask, labels, used_cols = cluster_athletes(df, z_cols, k=k)
    print(f"[report] clustered {int(mask.sum())} athletes on {len(used_cols)} metrics")

    print("[report] correlations + archetype stats...")
    corr = correlation_matrix(df, z_cols, prog_cols)
    top = top_correlations(corr, n=20)
    by_z, by_p = archetype_summary(df, mask, labels, used_cols, prog_cols)

    print("[report] labeling archetypes...")
    archetype_labels = label_archetypes(by_z)
    for i in sorted(archetype_labels):
        print(f"[report]   archetype {i}: {archetype_labels[i]}")

    print("[report] rendering HTML...")
    html = render_html(df, k=k, used_cols=used_cols,
                       mask=mask, labels=labels,
                       corr=corr, top_corr=top, by_z=by_z, by_p=by_p,
                       archetype_labels=archetype_labels,
                       role=role, domain=domain)

    if output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        slug = f"{role or 'all'}_{domain or 'all'}"
        output = OUTPUTS_DIR / f"report_{slug}_{ts}.html"
    else:
        output = Path(output)
    output.write_text(html, encoding="utf-8")
    print(f"[report] wrote {output}")
    return output


if __name__ == "__main__":
    import sys
    k = 5
    out = None
    args = sys.argv[1:]
    while args:
        a = args.pop(0)
        if a == "--k" and args:    k = int(args.pop(0))
        elif a == "--output" and args: out = args.pop(0)
    generate_report(k=k, output=out)
