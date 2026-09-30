"""
HTML report generator for research outputs. Uses Plotly for interactivity.

Each report is a single self-contained HTML file written to outputs/. Open in
a browser — no server, no notebook, coach-shareable via link or file transfer.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def _write_html(html_body: str, filename: str, title: str) -> Path:
    """Wrap the body in a minimal HTML shell and write to outputs/."""
    path = OUTPUT_DIR / filename
    doc = f"""<!DOCTYPE html>
<html><head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                        Roboto, sans-serif; max-width: 1200px; margin: 2em auto;
                        padding: 0 1em; color: #222; }}
    h1 {{ border-bottom: 3px solid #333; padding-bottom: 0.3em; }}
    h2 {{ border-bottom: 1px solid #999; padding-bottom: 0.2em; margin-top: 2em; }}
    table {{ border-collapse: collapse; margin: 1em 0; font-size: 0.9em; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 12px; }}
    th {{ background: #f4f4f4; text-align: left; }}
    tr:nth-child(even) {{ background: #fafafa; }}
    .meta {{ color: #666; font-size: 0.85em; }}
    .caveat {{ background: #fff9e6; border-left: 4px solid #f0c000;
              padding: 0.8em 1em; margin: 1em 0; }}
    .plot {{ margin: 1em 0; }}
    code {{ background: #f0f0f0; padding: 1px 4px; border-radius: 3px; }}
  </style>
</head><body>
{html_body}
</body></html>"""
    path.write_text(doc, encoding="utf-8")
    return path


def _df_to_html(df: pd.DataFrame, precision: int = 3) -> str:
    """Render a DataFrame as HTML with number formatting."""
    styled = df.copy()
    for c in styled.select_dtypes(include=[float]).columns:
        styled[c] = styled[c].round(precision)
    return styled.to_html(index=False, escape=False)


# ──────────────────────────────────────────────────────────────────────────
# Report 1: Target correlations
# ──────────────────────────────────────────────────────────────────────────

def render_target_correlation_report(
    target_metric: str,
    correlations: pd.DataFrame,
    *,
    n_athletes: int,
    role: str | None,
    top_n: int = 30,
) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    slug = target_metric.replace("/", "_")
    filename = f"research_correlate_{slug}_{ts}.html"

    body = [f"<h1>Correlations against <code>{target_metric}</code></h1>"]
    body.append(f"<p class='meta'>Generated {datetime.now():%Y-%m-%d %H:%M} · "
                f"Cohort n={n_athletes} · Role filter: {role or 'all'}</p>")
    body.append("<div class='caveat'>Small-sample warning: with n≈40, "
                "|r|&gt;0.5 is a meaningful signal even before FDR correction. "
                "Prioritize by effect size, use q-values for guardrails.</div>")

    if correlations.empty:
        body.append("<p>No metric pairs met the min-n threshold. Try a larger cohort.</p>")
    else:
        top = correlations.head(top_n)
        body.append(f"<h2>Top {len(top)} correlations by |r|</h2>")
        # Highlight FDR-significant rows
        html_tbl = top.to_html(index=False, escape=False,
                               classes=['corr-table'],
                               float_format=lambda x: f"{x:.3f}")
        body.append(html_tbl)

        # Plotly horizontal bar of top 20
        top20 = correlations.head(20).copy()
        top20 = top20.iloc[::-1]  # reverse for chart readability
        fig = go.Figure(go.Bar(
            x=top20["r"],
            y=top20["metric"],
            orientation="h",
            marker_color=["#c0392b" if r < 0 else "#2874a6" for r in top20["r"]],
            text=[f"n={n}, q={q:.3f}" if pd.notna(q) else f"n={n}"
                  for n, q in zip(top20["n"], top20["q_value"])],
            textposition="outside",
        ))
        fig.update_layout(
            title=f"Top 20 correlations vs {target_metric}",
            xaxis_title="Spearman r",
            xaxis=dict(range=[-1.05, 1.05], zeroline=True),
            height=max(400, 25 * len(top20) + 100),
            margin=dict(l=250, r=40, t=60, b=40),
        )
        body.append(f"<div class='plot'>{fig.to_html(include_plotlyjs='cdn', full_html=False)}</div>")

    return _write_html("\n".join(body), filename, f"Correlations vs {target_metric}")


# ──────────────────────────────────────────────────────────────────────────
# Report 2: Cross-domain matrix (heatmap)
# ──────────────────────────────────────────────────────────────────────────

def render_cross_domain_report(
    domain_a: str,
    domain_b: str,
    matrices: dict[str, pd.DataFrame],
    top_findings: pd.DataFrame,
    *,
    n_athletes: int,
    role: str | None,
) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"research_cross_{domain_a}_vs_{domain_b}_{ts}.html"

    r_mat = matrices["r"]
    sig_mat = matrices["sig"]

    body = [f"<h1>{domain_a} × {domain_b} correlation matrix</h1>"]
    body.append(f"<p class='meta'>Generated {datetime.now():%Y-%m-%d %H:%M} · "
                f"Cohort n={n_athletes} · Role: {role or 'all'} · "
                f"Method: Spearman · FDR: Benjamini-Hochberg, α=0.10</p>")
    body.append("<div class='caveat'>Cells show Spearman r. Bordered cells "
                "survive FDR correction. NaN cells had &lt; min_n athletes with "
                "both metrics.</div>")

    # Heatmap
    fig = go.Figure(data=go.Heatmap(
        z=r_mat.values,
        x=r_mat.columns.tolist(),
        y=r_mat.index.tolist(),
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        colorbar=dict(title="r"),
        hovertemplate="%{y}<br>%{x}<br>r=%{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"{domain_a} (rows) × {domain_b} (cols) — Spearman r",
        height=max(500, 22 * len(r_mat.index) + 150),
        width=max(700, 22 * len(r_mat.columns) + 250),
        xaxis=dict(tickangle=-45),
        margin=dict(l=250, r=40, t=60, b=200),
    )
    body.append("<h2>Full correlation heatmap</h2>")
    body.append(f"<div class='plot'>{fig.to_html(include_plotlyjs='cdn', full_html=False)}</div>")

    body.append(f"<h2>Top {len(top_findings)} findings by |r|</h2>")
    if top_findings.empty:
        body.append("<p>No pairs met the min-n threshold.</p>")
    else:
        body.append(_df_to_html(top_findings))

    return _write_html("\n".join(body), filename,
                       f"{domain_a} × {domain_b} correlations")


# ──────────────────────────────────────────────────────────────────────────
# Report 3: Cluster archetypes
# ──────────────────────────────────────────────────────────────────────────

def render_clustering_report(
    result: dict,
    *,
    role: str | None,
    domains: list[str] | None,
) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    domain_slug = "_".join(domains) if domains else "all"
    filename = f"research_clusters_{domain_slug}_k{result['k']}_{ts}.html"

    body = [f"<h1>Athlete archetypes — k={result['k']} clusters</h1>"]
    body.append(f"<p class='meta'>Generated {datetime.now():%Y-%m-%d %H:%M} · "
                f"Cohort n={result['n_athletes']} · Role: {role or 'all'} · "
                f"Domains: {', '.join(domains) if domains else 'all'}</p>")
    if result.get("silhouette") is not None:
        body.append(f"<p class='meta'>Silhouette score: <code>{result['silhouette']}</code> "
                    f"(higher is better; typically 0.15–0.40 for real-world biomech data)</p>")
    if result.get("silhouette_by_k"):
        body.append("<p class='meta'>Silhouette across k tried: "
                    f"<code>{result['silhouette_by_k']}</code></p>")

    body.append("<div class='caveat'>Cluster labels are arbitrary (0, 1, 2, ...); "
                "interpret them by their DISTINGUISHING METRICS section below. "
                "Small clusters (n&lt;5) are usually noise, not real archetypes.</div>")

    # PCA 2D scatter
    pca_df = result["pca_2d"]
    ev = result["pca_explained_variance"]
    fig = px.scatter(
        pca_df, x="PC1", y="PC2", color=pca_df["cluster"].astype(str),
        hover_data=["name"], title="PCA 2D projection colored by cluster",
    )
    fig.update_layout(
        xaxis_title=f"PC1 ({ev[0]:.0%} variance)",
        yaxis_title=f"PC2 ({ev[1]:.0%} variance)",
        legend_title="Cluster",
        height=500,
    )
    body.append(f"<div class='plot'>{fig.to_html(include_plotlyjs='cdn', full_html=False)}</div>")

    # Cluster-size + roster
    body.append("<h2>Cluster membership</h2>")
    cohort = result["cohort"]
    for c in sorted(cohort["cluster"].unique()):
        members = cohort[cohort["cluster"] == c][["name", "age_group", "role"]]
        body.append(f"<h3>Cluster {c} — {len(members)} athletes</h3>")
        body.append(_df_to_html(members.reset_index(drop=True)))

    # Distinguishing metrics per cluster
    body.append("<h2>Distinguishing metrics per cluster</h2>")
    body.append("<p class='meta'>For each cluster, top 15 metrics ranked by |t| "
                "(cluster mean vs everyone else). Positive delta = cluster is "
                "ABOVE population avg; negative = BELOW. High |t| = the metric "
                "meaningfully separates this cluster from the rest.</p>")
    for c, dist in result["distinguishing"].items():
        body.append(f"<h3>Cluster {c}</h3>")
        if dist.empty:
            body.append("<p>No distinguishing metrics computed.</p>")
        else:
            body.append(_df_to_html(dist))

    return _write_html("\n".join(body), filename, f"Archetypes k={result['k']}")


# ──────────────────────────────────────────────────────────────────────────
# Report 4: Longitudinal + program-response
# ──────────────────────────────────────────────────────────────────────────

def render_longitudinal_report(
    delta_df: pd.DataFrame,
    delta_summary: pd.DataFrame,
    program_findings: pd.DataFrame,
    *,
    role: str | None,
) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"research_longitudinal_{ts}.html"

    body = [f"<h1>Longitudinal + program-response analysis</h1>"]
    body.append(f"<p class='meta'>Generated {datetime.now():%Y-%m-%d %H:%M} · "
                f"n athletes with serial profiles: {len(delta_df)} · Role: {role or 'all'}</p>")
    body.append("<div class='caveat'>Program-response findings are CORRELATIONAL, "
                "not causal. Coach selection bias inflates apparent effects: athletes "
                "who NEEDED T-spine work got the drills that address T-spine, and "
                "their T-spine got tested again. Some of that improvement is "
                "regression to the mean, not the drill working. Treat as hypothesis "
                "generation, not proof.</div>")

    if delta_df.empty:
        body.append("<p><strong>No athletes with ≥2 profiles found</strong> — "
                    "run <code>python -m src.main backfill</code> across a wider "
                    "date range, or wait until more athletes get re-assessed.</p>")
        return _write_html("\n".join(body), filename, "Longitudinal analysis")

    body.append("<h2>Cohort — athletes with ≥2 profiles in window</h2>")
    body.append(_df_to_html(
        delta_df[["name", "first_date", "last_date", "span_days"]].reset_index(drop=True)
    ))

    body.append("<h2>Per-metric pre/post summary</h2>")
    body.append("<p class='meta'><code>delta_mean</code> is average Z-score "
                "change (positive = improved). <code>pct_improved</code> is the "
                "fraction of athletes whose Z went up.</p>")
    body.append(_df_to_html(delta_summary))

    body.append("<h2>Top program-response correlations</h2>")
    body.append("<p class='meta'>For each (exercise pattern × metric) pair with "
                "enough athletes, correlate how much of that pattern the athlete "
                "was prescribed against how much the metric changed. "
                "<strong>Look at q_value column: q ≤ 0.10 = FDR-significant across "
                "the full test space.</strong> Everything else is likely noise "
                "given the number of pairs tested.</p>")
    if program_findings.empty:
        body.append("<p>No pattern × delta pairs met the min-n threshold. Wait for "
                    "more serial data.</p>")
    else:
        n_total = len(program_findings)
        n_sig = int(program_findings["q_value"].le(0.10).sum()) \
                if "q_value" in program_findings.columns else 0
        body.append(f"<p class='meta'>Total pairs correlated: <code>{n_total}</code> · "
                    f"FDR-significant at q≤0.10: <strong>{n_sig}</strong></p>")

        # If FDR-significant findings exist, lead with those
        if "q_value" in program_findings.columns and n_sig > 0:
            sig = program_findings[program_findings["q_value"] <= 0.10].copy()
            body.append(f"<h3>FDR-significant findings ({len(sig)})</h3>")
            body.append(_df_to_html(sig))

        # Then show top 100 by |r| (whether or not FDR-significant)
        top = program_findings.head(100)
        body.append(f"<h3>Top 100 by |r| (may include noise — check q_value)</h3>")
        body.append(_df_to_html(top))

    return _write_html("\n".join(body), filename, "Longitudinal analysis")


# ──────────────────────────────────────────────────────────────────────────
# Report 0: Coverage — what data do we have?
# ──────────────────────────────────────────────────────────────────────────

def render_stratified_correlation_report(
    target_metric: str,
    strata: dict[str, dict],
    *,
    role: str | None,
    top_n_per_stratum: int = 25,
) -> Path:
    """Render one HTML with correlate results per age group side-by-side.

    strata: {age_group_label: {"correlations": DataFrame, "n_athletes": int}}
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    slug = target_metric.replace("/", "_")
    filename = f"research_correlate_stratified_{slug}_{ts}.html"

    body = [f"<h1>Correlations against <code>{target_metric}</code> — by age group</h1>"]
    body.append(f"<p class='meta'>Generated {datetime.now():%Y-%m-%d %H:%M} · "
                f"Role filter: {role or 'all'} · Strata: {list(strata.keys())}</p>")
    body.append("<div class='caveat'>Each age group is analyzed independently. "
                "A metric that's a strong correlate in one age group but not "
                "others suggests age-specific mechanics. A metric that's "
                "consistent across groups is a robust finding.</div>")

    # ── Comparison table: pull top-K from each stratum, merge into one wide table
    all_metrics: set[str] = set()
    for st_name, st in strata.items():
        corr = st["correlations"]
        if corr.empty:
            continue
        all_metrics.update(corr.head(top_n_per_stratum)["metric"].tolist())

    if all_metrics:
        rows = []
        for m in all_metrics:
            row = {"metric": m}
            # Add domain from any non-empty stratum
            for st_name, st in strata.items():
                corr = st["correlations"]
                if not corr.empty:
                    match = corr[corr["metric"] == m]
                    if not match.empty:
                        row["domain"] = match.iloc[0]["domain"]
                        break
            for st_name, st in strata.items():
                corr = st["correlations"]
                if corr.empty:
                    row[f"{st_name}_r"] = None
                    row[f"{st_name}_n"] = None
                    row[f"{st_name}_sig"] = ""
                    continue
                match = corr[corr["metric"] == m]
                if match.empty:
                    row[f"{st_name}_r"] = None
                    row[f"{st_name}_n"] = None
                    row[f"{st_name}_sig"] = ""
                else:
                    r = match.iloc[0]
                    row[f"{st_name}_r"] = round(float(r["r"]), 3)
                    row[f"{st_name}_n"] = int(r["n"])
                    row[f"{st_name}_sig"] = "★" if bool(r.get("fdr_significant", False)) else ""
            rows.append(row)
        merged = pd.DataFrame(rows)
        # Sort by max |r| across strata
        r_cols = [c for c in merged.columns if c.endswith("_r")]
        merged["_max_abs_r"] = merged[r_cols].abs().max(axis=1)
        merged = merged.sort_values("_max_abs_r", ascending=False).drop(columns=["_max_abs_r"])
        merged = merged.reset_index(drop=True)

        body.append("<h2>Union of top-25 correlations across age groups</h2>")
        body.append("<p class='meta'>★ = FDR-significant (q ≤ 0.10) in that stratum. "
                    "Blank cells = metric didn't make top-25 in that stratum. "
                    "Sort by max |r| across strata.</p>")
        body.append(_df_to_html(merged))

    # ── Individual stratum breakdowns
    body.append("<h2>Individual age-group breakdowns</h2>")
    for st_name, st in strata.items():
        corr = st["correlations"]
        n_ath = st["n_athletes"]
        body.append(f"<h3>{st_name} — n={n_ath} athletes</h3>")
        if corr.empty:
            body.append("<p>No metric pairs met the min-n threshold.</p>")
            continue
        n_sig = int(corr["fdr_significant"].sum())
        body.append(f"<p class='meta'>{len(corr)} metrics correlated · "
                    f"{n_sig} survived FDR</p>")
        body.append(_df_to_html(corr.head(top_n_per_stratum)))

    return _write_html("\n".join(body), filename,
                       f"{target_metric} by age group")


def render_overview_report(
    target_domain: str,
    matrices_by_domain: dict[str, dict],
    top_findings: pd.DataFrame,
    *,
    n_athletes: int,
    role: str | None,
) -> Path:
    """Render a comprehensive cross-domain overview against a target domain.

    matrices_by_domain: {source_domain: cross_domain_matrix result dict}
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"research_overview_vs_{target_domain}_{ts}.html"

    body = [f"<h1>Cross-domain overview vs <code>{target_domain}</code></h1>"]
    body.append(f"<p class='meta'>Generated {datetime.now():%Y-%m-%d %H:%M} · "
                f"n athletes: {n_athletes} · Role: {role or 'all'} · "
                f"Source domains: {list(matrices_by_domain.keys())}</p>")
    body.append("<div class='caveat'>Each source-domain × target-domain matrix "
                "is run independently. The top-findings table combines all pairs "
                "across all matrices and applies GLOBAL BH-FDR correction across "
                "the whole test space.</div>")

    # ── Global top findings
    body.append("<h2>Global top findings (all source domains, FDR-corrected globally)</h2>")
    if top_findings.empty:
        body.append("<p>No pairs met the min-n threshold across any source domain.</p>")
    else:
        n_sig = int(top_findings["fdr_significant"].sum()) if "fdr_significant" in top_findings.columns else 0
        body.append(f"<p class='meta'>Top 50 by |r| · {n_sig} FDR-significant "
                    f"across the full test space.</p>")
        body.append(_df_to_html(top_findings.head(50)))

    # ── Per-source-domain heatmaps
    body.append("<h2>Per-source-domain heatmaps</h2>")
    for src, mats in matrices_by_domain.items():
        r_mat = mats["r"]
        body.append(f"<h3>{src} × {target_domain} — {r_mat.shape[0]}×{r_mat.shape[1]}</h3>")
        if r_mat.empty:
            body.append("<p>No matrix (domain not present in cohort).</p>")
            continue
        fig = go.Figure(data=go.Heatmap(
            z=r_mat.values,
            x=r_mat.columns.tolist(),
            y=r_mat.index.tolist(),
            colorscale="RdBu_r",
            zmin=-1, zmax=1,
            colorbar=dict(title="r"),
            hovertemplate="%{y}<br>%{x}<br>r=%{z:.3f}<extra></extra>",
        ))
        fig.update_layout(
            title=f"{src} (rows) × {target_domain} (cols)",
            height=max(400, 20 * len(r_mat.index) + 150),
            width=max(700, 20 * len(r_mat.columns) + 250),
            xaxis=dict(tickangle=-45),
            margin=dict(l=250, r=40, t=60, b=200),
        )
        body.append(f"<div class='plot'>{fig.to_html(include_plotlyjs='cdn', full_html=False)}</div>")

    return _write_html("\n".join(body), filename, f"Overview vs {target_domain}")


def render_velocity_deep_report(
    *,
    n_trials: int,
    n_athletes: int,
    age_group: str | None,
    pooled: pd.DataFrame,
    stratified: dict[str, pd.DataFrame],
    within_athlete: pd.DataFrame,
    session_level: pd.DataFrame,
    backwards: dict[str, pd.DataFrame] | None = None,
    top_k: int = 30,
) -> Path:
    """Combined velocity-deep report showing all 4 analysis levels."""
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"research_velocity_deep_{ts}.html"

    body = ["<h1>Velocity deep-dive — trial-level analysis</h1>"]
    body.append(f"<p class='meta'>Generated {datetime.now():%Y-%m-%d %H:%M} · "
                f"Trials analyzed: {n_trials:,} · Unique athletes: {n_athletes} · "
                f"Age filter: {age_group or 'all'}</p>")
    body.append("<div class='caveat'>"
                "<strong>The four levels answer different questions:</strong> "
                "POOLED shows raw associations but confounds age/size. "
                "STRATIFIED controls for age group. "
                "<strong>WITHIN-ATHLETE (fixed effects) is the strongest — "
                "it asks 'when this athlete throws harder than his own average, "
                "what changes?' and controls for every fixed characteristic of the person.</strong> "
                "SESSION-LEVEL shows session-mean changes for athletes with multiple 3Ds. "
                "Prioritize findings that replicate across levels — those are the "
                "most robust."
                "</div>")

    # ── Executive comparison table
    body.append("<h2>Cross-level comparison — top 30 within-athlete metrics + how they show up elsewhere</h2>")
    body.append("<p class='meta'>Metrics ranked by |r| in the within-athlete "
                "analysis, with r-values from other analyses shown alongside. "
                "★ = FDR-significant in that column.</p>")
    if within_athlete.empty:
        body.append("<p>No within-athlete correlations produced. Check that you "
                    "have athletes with ≥3 trials.</p>")
    else:
        top_wa = within_athlete.head(top_k)
        compare_rows = []
        for _, r in top_wa.iterrows():
            m = r["metric"]
            row = {"metric": m, "family": r["family"]}
            row["wa_r"] = round(float(r["r"]), 3)
            row["wa_n_trials"] = int(r["n_trials"])
            row["wa_sig"] = "★" if bool(r.get("fdr_significant")) else ""
            # Pooled
            p_match = pooled[pooled["metric"] == m] if not pooled.empty else pd.DataFrame()
            row["pooled_r"] = round(float(p_match.iloc[0]["r"]), 3) if not p_match.empty else None
            row["pooled_sig"] = "★" if not p_match.empty and bool(p_match.iloc[0]["fdr_significant"]) else ""
            # Session-level
            s_match = session_level[session_level["metric"] == m] if not session_level.empty else pd.DataFrame()
            row["session_r"] = round(float(s_match.iloc[0]["r"]), 3) if not s_match.empty else None
            row["session_sig"] = "★" if not s_match.empty and bool(s_match.iloc[0]["fdr_significant"]) else ""
            # Per age group
            for ag, sf in stratified.items():
                if sf.empty:
                    row[f"{ag}_r"] = None
                    continue
                sm = sf[sf["metric"] == m]
                row[f"{ag}_r"] = round(float(sm.iloc[0]["r"]), 3) if not sm.empty else None
            compare_rows.append(row)
        body.append(_df_to_html(pd.DataFrame(compare_rows)))

    # ── Per-level detail
    def _level_section(title: str, note: str, df: pd.DataFrame) -> None:
        body.append(f"<h2>{title}</h2>")
        body.append(f"<p class='meta'>{note}</p>")
        if df.empty:
            body.append("<p>No results at this level (insufficient data).</p>")
            return
        n_sig = int(df["fdr_significant"].sum())
        body.append(f"<p class='meta'>{len(df)} metrics correlated · "
                    f"{n_sig} FDR-significant (q ≤ 0.10)</p>")
        body.append(_df_to_html(df.head(top_k)))

    _level_section(
        "Within-athlete fixed-effects (strongest control)",
        "Subtract each athlete's mean from each trial. Correlate residuals. "
        "Answers: 'When THIS pitcher throws harder than his own average, what "
        "kinematic/force changes come with it?'",
        within_athlete,
    )
    _level_section(
        "Pooled — all trials, all athletes (confounded, use with caution)",
        "Cross-sectional across the whole cohort. Athletes who tend to throw "
        "harder ALSO tend to be older/bigger, so this mixes those effects.",
        pooled,
    )
    _level_section(
        "Session-level — repeated 3Ds within-athlete",
        "For athletes with ≥2 sessions, compute per-session means, then "
        "residualize within-athlete. Answers: 'When this athlete's average "
        "velocity changed between sessions, what mechanics changed with it?'",
        session_level,
    )

    body.append("<h2>Stratified — pooled per age group</h2>")
    body.append("<p class='meta'>Same as pooled but split by age group. "
                "A finding consistent across HS/COLLEGE/PRO is much more robust "
                "than one that's only present in a single stratum.</p>")
    for ag, df_ag in stratified.items():
        body.append(f"<h3>{ag}</h3>")
        if df_ag.empty:
            body.append("<p>Insufficient trials in this stratum.</p>")
            continue
        n_sig = int(df_ag["fdr_significant"].sum())
        body.append(f"<p class='meta'>{len(df_ag)} metrics · {n_sig} FDR-significant</p>")
        body.append(_df_to_html(df_ag.head(20)))

    # ── Backwards chain
    if backwards:
        body.append("<h2>Backwards chain — what drives the top velocity correlates?</h2>")
        body.append("<p class='meta'>For each of the top velocity correlates "
                    "(from within-athlete), the top drivers of THAT metric are "
                    "shown. Read as: 'velocity ← metric X ← its drivers'.</p>")
        for target, drivers in backwards.items():
            body.append(f"<h3>← <code>{target}</code></h3>")
            if drivers.empty:
                body.append("<p>No drivers surfaced.</p>")
                continue
            body.append(_df_to_html(drivers))

    return _write_html("\n".join(body), filename, "Velocity deep dive")


def render_session_change_report(
    delta_df: pd.DataFrame,
    delta_correlations: dict[str, pd.DataFrame],
    *,
    aggregation: str,
    cross_target_summary: pd.DataFrame | None = None,
    predictor_domain: str = "all",
) -> Path:
    """Session-change deltas — natural-experiment-style correlations."""
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"research_session_change_{ts}.html"

    body = [f"<h1>Session-change drivers — what changed with velocity/GRF between 3Ds?</h1>"]
    body.append(f"<p class='meta'>Generated {datetime.now():%Y-%m-%d %H:%M} · "
                f"Athletes with 2+ sessions: {len(delta_df)} · "
                f"Aggregation: {aggregation}</p>")
    body.append("<div class='caveat'>"
                "<strong>Natural-experiment lens.</strong> Each athlete's "
                "session1-to-session2 change in a metric is a natural mini-experiment: "
                "what changed between two states of the same person. Positive correlation "
                "between delta_A and delta_B means these adaptations tended to happen "
                "together across athletes. Stronger evidence than pooled cross-sectional "
                "correlation, but still not causal — both changes could be driven by "
                "a hidden training-block variable."
                "</div>")
    if predictor_domain == "assessment_baseline":
        body.append("<div class='caveat' style='border-left-color:#0080d0;background:#eef7ff'>"
                    "<strong>Baseline mode.</strong> Predictors are each athlete's "
                    "BASELINE assessment Z-score (from the profile closest to session 1), "
                    "not a change in the assessment. This answers "
                    "<em>'athletes who came in with more X gained more Y between sessions.'</em> "
                    "Uses every athlete with at least one profile — usually 3× more coverage "
                    "than the two-profile delta mode."
                    "</div>")
    elif predictor_domain in ("assessment_only", "assessment_delta"):
        body.append("<div class='caveat' style='border-left-color:#d08000;background:#fff8e6'>"
                    "<strong>Delta mode — requires 2 profile snapshots per athlete.</strong> "
                    "If this table looks empty, most athletes in the cohort have only one "
                    "profile in <code>ai_layer.athlete_profiles</code>. Re-run with "
                    "<code>--predictor-domain assessment_baseline</code>."
                    "</div>")

    # Cross-target summary first — the "which single thing predicts everything"
    if cross_target_summary is not None and not cross_target_summary.empty:
        body.append(f"<h2>Cross-target summary — which predictors hit the MOST targets consistently?</h2>")
        body.append(f"<p class='meta'>Predictor domain filter: <code>{predictor_domain}</code>. "
                    f"A predictor scoring high on <code>n_targets_hit</code> means it moved with "
                    f"many force/impulse metrics between sessions — <strong>the single most likely "
                    f"'if I train this, everything else improves' finding.</strong> "
                    f"<code>direction_consistency</code> = fraction of hits in the same direction "
                    f"(1.0 = all positive or all negative).</p>")
        body.append(_df_to_html(cross_target_summary.head(30)))

    body.append("<h2>Cohort</h2>")
    body.append(f"<p class='meta'>{len(delta_df)} athletes with 2+ pitching-3D "
                f"sessions within the span window.</p>")
    if not delta_df.empty:
        show_cols = ["name", "session1_date", "session2_date",
                     "span_days", "delta_velocity_mph"]
        cohort_show = delta_df[[c for c in show_cols if c in delta_df.columns]].copy()
        body.append(_df_to_html(cohort_show))

    body.append("<h2>Per-target delta correlations</h2>")
    body.append("<p class='meta'>For each target delta, correlate against every "
                "OTHER delta (assessment + kinematic + force-metric).</p>")
    for tgt, res in delta_correlations.items():
        body.append(f"<h3>Target: <code>{tgt}</code></h3>")
        if res.empty:
            body.append("<p>Not enough athletes with paired deltas on this target.</p>")
            continue
        n_sig = int(res["fdr_significant"].sum())
        body.append(f"<p class='meta'>{len(res)} predictors tested · "
                    f"<strong>{n_sig} FDR-significant</strong></p>")
        body.append(_df_to_html(res.head(30)))

    return _write_html("\n".join(body), filename, "Session-change drivers")


def render_stratified_chain_report(
    strata_results: dict[str, dict],
    *,
    top_k_per_stratum: int,
) -> Path:
    """Stratified chain report: velocity ← kinematic ← assessment, per age group."""
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"research_stratified_chain_{ts}.html"

    body = [f"<h1>Stratified triangulation — velocity ← kinematic ← assessment</h1>"]
    body.append(f"<p class='meta'>Generated {datetime.now():%Y-%m-%d %H:%M} · "
                f"Strata: {list(strata_results.keys())} · "
                f"Top {top_k_per_stratum} kinematic velocity correlates per stratum</p>")
    body.append("<div class='caveat'>"
                "<strong>Each age group is analyzed as an ISOLATED cohort.</strong> "
                "Top kinematic velocity correlates are picked WITHIN that stratum "
                "(within-athlete fixed effects), then predictors of those "
                "kinematics are searched using ONLY athletes in that stratum's "
                "profile cohort. This is the stratum-specific triangulation — "
                "the answer to 'what makes a PRO throw hard' is entirely separate "
                "from 'what makes a HIGH SCHOOLER throw hard.'"
                "</div>")

    # Executive summary — top predictor per (stratum, kinematic) pair
    body.append("<h2>Executive summary — top FDR-significant assessment predictor per (stratum × kinematic)</h2>")
    summary_rows = []
    for stratum, sr in strata_results.items():
        if "error" in sr and "chain" not in sr:
            summary_rows.append({
                "stratum": stratum, "kinematic": f"ERROR: {sr['error']}",
                "kin_velocity_r": None,
                "top_predictor": None, "top_r": None, "top_domain": None,
                "n_predictors_fdr_sig": None,
            })
            continue
        top_vc = sr.get("top_velocity_correlates")
        if top_vc is None or top_vc.empty:
            summary_rows.append({
                "stratum": stratum, "kinematic": "(no velocity correlates)",
                "kin_velocity_r": None,
                "top_predictor": None, "top_r": None, "top_domain": None,
                "n_predictors_fdr_sig": None,
            })
            continue
        for _, kvc in top_vc.iterrows():
            m = kvc["metric"]
            chain_r = sr["chain"].get(m, {})
            corr = chain_r.get("correlations") if "error" not in chain_r else None
            if corr is None or corr.empty:
                summary_rows.append({
                    "stratum": stratum, "kinematic": m,
                    "kin_velocity_r": round(float(kvc["r"]), 3),
                    "top_predictor": chain_r.get("error", "(no predictors)"),
                    "top_r": None, "top_domain": None, "n_predictors_fdr_sig": 0,
                })
                continue
            top = corr.iloc[0]
            summary_rows.append({
                "stratum": stratum, "kinematic": m,
                "kin_velocity_r": round(float(kvc["r"]), 3),
                "top_predictor": top["assessment_metric"],
                "top_r": round(float(top["r"]), 3),
                "top_domain": top["domain"],
                "n_predictors_fdr_sig": int(corr["fdr_significant"].sum()),
            })
    body.append(_df_to_html(pd.DataFrame(summary_rows)))

    # Per-stratum detail
    for stratum, sr in strata_results.items():
        body.append(f"<h2>[{stratum}] Detail</h2>")
        if "error" in sr and "chain" not in sr:
            body.append(f"<p><em>{sr['error']}</em></p>")
            continue
        body.append(f"<p class='meta'>Trial cohort: {sr.get('n_athletes_trials', 0)} athletes "
                    f"({sr.get('n_trials', 0):,} trials) · "
                    f"Profile cohort intersection: {sr.get('n_athletes_profile', 0)} athletes</p>")
        top_vc = sr.get("top_velocity_correlates")
        if top_vc is None or top_vc.empty:
            body.append("<p>No within-athlete velocity correlates in this stratum.</p>")
            continue
        body.append(f"<h3>Top {len(top_vc)} kinematic velocity correlates (within-athlete)</h3>")
        body.append(_df_to_html(top_vc))
        body.append(f"<h3>Assessment predictors of each top kinematic</h3>")
        for m, chain_r in sr["chain"].items():
            body.append(f"<h4>← <code>{m}</code></h4>")
            if "error" in chain_r:
                body.append(f"<p><em>{chain_r['error']}</em></p>")
                continue
            corr = chain_r.get("correlations")
            if corr is None or corr.empty:
                body.append("<p>No eligible predictors.</p>")
                continue
            n_sig = int(corr["fdr_significant"].sum())
            body.append(f"<p class='meta'>n_athletes: {chain_r['n_athletes']} · "
                        f"{len(corr)} predictors tested · "
                        f"<strong>{n_sig} FDR-significant</strong></p>")
            body.append(_df_to_html(corr.head(15)))

    return _write_html("\n".join(body), filename, "Stratified triangulation")


def render_kinematic_drivers_report(
    results: dict[str, dict],
    *,
    role: str | None,
    age_group: str | None,
    aggregation: str,
) -> Path:
    """Multi-kinematic drivers report. `results` is dict[kin_metric → result]."""
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"research_kinematic_drivers_{ts}.html"

    body = [f"<h1>Kinematic drivers — what assessment predicts each mechanic?</h1>"]
    body.append(f"<p class='meta'>Generated {datetime.now():%Y-%m-%d %H:%M} · "
                f"Role: {role or 'all'} · Age group: {age_group or 'all'} · "
                f"Aggregation: {aggregation} across trials per athlete · "
                f"Targets: {len(results)}</p>")
    body.append("<div class='caveat'>"
                "<strong>Two-stage inference.</strong> Each kinematic was aggregated "
                "ACROSS an athlete's trials to a single value, then correlated "
                "across athletes against physical assessment Z-scores. That means "
                "these correlations are cross-sectional at the athlete level "
                "(NOT within-athlete). "
                "Read a finding as 'athletes who score higher on assessment X "
                "tend to display more of kinematic Y' — hypothesis-generating, "
                "not causal. Combine with the within-athlete velocity findings "
                "to form: <em>velocity ← kinematic driver ← physical predictor</em>."
                "</div>")

    # ── Cross-target summary (top 5 significant per target)
    body.append("<h2>Executive summary — top FDR-significant assessment predictor per kinematic</h2>")
    summary_rows = []
    for kin, res in results.items():
        if "error" in res:
            summary_rows.append({
                "kinematic": kin,
                "n_athletes": None,
                "top_predictor": f"ERROR: {res['error']}",
                "top_r": None,
                "top_domain": None,
                "n_fdr_sig": None,
            })
            continue
        corr = res["correlations"]
        if corr.empty:
            summary_rows.append({
                "kinematic": kin,
                "n_athletes": res["n_athletes"],
                "top_predictor": "(no eligible pairs)",
                "top_r": None,
                "top_domain": None,
                "n_fdr_sig": 0,
            })
            continue
        top = corr.iloc[0]
        summary_rows.append({
            "kinematic": kin,
            "n_athletes": res["n_athletes"],
            "top_predictor": top["assessment_metric"],
            "top_r": round(float(top["r"]), 3),
            "top_domain": top["domain"],
            "n_fdr_sig": int(corr["fdr_significant"].sum()),
        })
    body.append(_df_to_html(pd.DataFrame(summary_rows)))

    # ── Per-kinematic detail
    body.append("<h2>Per-kinematic detail — top 20 predictors</h2>")
    for kin, res in results.items():
        body.append(f"<h3><code>{kin}</code></h3>")
        if "error" in res:
            body.append(f"<p><em>Error: {res['error']}</em></p>")
            continue
        corr = res["correlations"]
        body.append(f"<p class='meta'>Athletes with this kinematic aggregated: "
                    f"<code>{res['n_athletes']}</code></p>")
        if corr.empty:
            body.append("<p>No pairs met the min-n threshold.</p>")
        else:
            n_sig = int(corr["fdr_significant"].sum())
            body.append(f"<p class='meta'>{len(corr)} predictors tested · "
                        f"<strong>{n_sig} survived FDR (q ≤ 0.10)</strong></p>")
            body.append(_df_to_html(corr.head(20)))

    return _write_html("\n".join(body), filename, "Kinematic drivers")


def render_coverage_report(coverage_df: pd.DataFrame, n_athletes: int) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"research_coverage_{ts}.html"

    body = ["<h1>Assessment coverage report</h1>"]
    body.append(f"<p class='meta'>Generated {datetime.now():%Y-%m-%d %H:%M} · "
                f"Cohort n={n_athletes}</p>")
    body.append("<div class='caveat'>"
                "<strong>Data quality watch — mobility metrics.</strong> "
                "The 8ctane mobility protocol has changed over the years: tests "
                "were added and removed, and the measurement scale switched from "
                "a 1-3 rubric to actual measurements. Older profiles have "
                "mobility Z-scores computed against a different normative "
                "population than newer ones. If you're going to correlate on "
                "mobility metrics, either (a) restrict to recent profiles with "
                "<code>--min-as-of-date 2025-01-01</code>, or (b) drop mobility "
                "entirely with <code>--exclude-mobility</code>."
                "</div>")
    body.append("<p>Per-metric coverage across the cohort. Metrics with low "
                "coverage will limit which correlations / cluster runs are "
                "meaningful.</p>")

    # Coverage bar chart by domain
    if not coverage_df.empty:
        fig = px.bar(
            coverage_df,
            x="metric",
            y="coverage_pct",
            color="domain",
            title="Per-metric coverage across cohort",
            hover_data=["n_non_null", "mean_z", "std_z"],
        )
        fig.update_layout(
            xaxis_tickangle=-70,
            height=600,
            yaxis=dict(range=[0, 1], tickformat=".0%"),
            margin=dict(b=200),
        )
        body.append(f"<div class='plot'>{fig.to_html(include_plotlyjs='cdn', full_html=False)}</div>")
        body.append("<h2>Full coverage table</h2>")
        body.append(_df_to_html(coverage_df))

    return _write_html("\n".join(body), filename, "Coverage report")


# ──────────────────────────────────────────────────────────────────────────
# Athlete deep-dive
# ──────────────────────────────────────────────────────────────────────────

def render_athlete_deep_dive_report(report) -> Path:  # DeepDiveReport
    """Deprecated shim.

    This used to be a 250-line renderer that emitted 500+ table rows and no
    charts. Everything it showed now lives in the analyst appendix of the
    coach report, which also leads with the findings and draws the picture, so
    keeping two renderers in step was pure cost. Existing callers keep working
    and get the better page.
    """
    from src.research.coach_report import render_coach_report
    return render_coach_report(report)
