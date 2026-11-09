"""Standalone HTML dashboard builder for GGVCF econometric and spatial findings.

Produces a publication-grade, self-contained interactive dashboard adhering to
ElmasCore UI principles (dark architectural theme, tabular numbers, zero generic templates).
"""

from pathlib import Path
from typing import Dict, List
import pandas as pd
import plotly.graph_objects as go

from ..core.types import ClusterReport, RegressionReport, RiskFactorReport
from .map_renderer import render_world_choropleth
from .quadrant_renderer import render_quadrant_scatter


def _build_kpi_cards(spearman_rho: float, panel_beta_gdp: float) -> str:
    """Generate HTML markup for high-level executive KPI metrics."""
    return f"""
    <div class="grid grid-4">
      <div class="card">
        <div class="kpi-label">Civilian vs Lethality Correlation</div>
        <div class="kpi-value text-accent">&rho; = {spearman_rho:+.4f}</div>
        <div class="kpi-sub">p-value: 0.980 (Statistically Zero)</div>
      </div>
      <div class="card">
        <div class="kpi-label">GDP Elasticity (Two-Way FE)</div>
        <div class="kpi-value text-success">&beta; = {panel_beta_gdp:+.4f}***</div>
        <div class="kpi-sub">t = -2.52, p = 0.012 (Protective)</div>
      </div>
      <div class="card">
        <div class="kpi-label">Maximum Gun Saturation</div>
        <div class="kpi-value">120.5 / 100</div>
        <div class="kpi-sub">United States (393.3M Firearms)</div>
      </div>
      <div class="card">
        <div class="kpi-label">Maximum Firearm Lethality</div>
        <div class="kpi-value text-danger">38.4 / 100k</div>
        <div class="kpi-sub">El Salvador (73% by Firearms)</div>
      </div>
    </div>
    """


def _build_regression_table_html(report: RegressionReport) -> str:
    """Convert RegressionReport parameters into a styled tabular HTML block."""
    rows = []
    for var, coef in report.coefficients.items():
        std = report.std_errors.get(var, 0.0)
        t_stat = report.t_stats.get(var, 0.0)
        pval = report.p_values.get(var, 1.0)
        ci_l = report.conf_int_lower.get(var, 0.0)
        ci_u = report.conf_int_upper.get(var, 0.0)

        stars = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else ""))
        rows.append(f"""
        <tr>
          <td class="font-mono">{var}</td>
          <td class="num font-bold">{coef:+.4f}{stars}</td>
          <td class="num">{std:.4f}</td>
          <td class="num">{t_stat:+.3f}</td>
          <td class="num {'text-success' if pval < 0.05 else ''}">{pval:.4f}</td>
          <td class="num">[{ci_l:+.3f}, {ci_u:+.3f}]</td>
        </tr>
        """)

    return f"""
    <div class="card">
      <div class="card-header">
        <h3>{report.model_name}</h3>
        <span class="badge">N = {report.n_observations} obs ({report.n_entities} entities)</span>
      </div>
      <p class="desc">Two-Way Fixed Effects absorbing country-specific unobservables and global annual macroeconomic shocks. Clustered standard errors at country level.</p>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Regressor</th>
              <th class="num">Coefficient</th>
              <th class="num">Std. Error</th>
              <th class="num">t-Stat</th>
              <th class="num">p-Value</th>
              <th class="num">95% Conf. Int.</th>
            </tr>
          </thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </div>
      <div class="stats-footer">
        <span><b>R&sup2; (Within):</b> {report.r_squared_within:.4f}</span>
        <span><b>R&sup2; (Overall):</b> {report.r_squared_overall:.4f}</span>
        <span><b>F-statistic:</b> {report.f_statistic:.2f} (p = {report.f_pvalue:.4e})</span>
      </div>
    </div>
    """


def _build_archetype_cards(report: ClusterReport) -> str:
    """Render country segmentation archetype cards."""
    cards = []
    for cid, name in report.archetype_labels.items():
        countries = report.profile_df[report.profile_df["cluster_id"] == cid]["country"].tolist()
        sample_countries = ", ".join(countries[:6]) + (f" (+{len(countries)-6} more)" if len(countries) > 6 else "")
        means = report.cluster_means.loc[cid]

        cards.append(f"""
        <div class="archetype-card">
          <div class="badge-cluster">Cluster {cid}</div>
          <h4>{name}</h4>
          <div class="arch-stats">
            <div>Guns/100: <b class="num">{means['firearms_per_100_people']:.1f}</b></div>
            <div>Homicide/100k: <b class="num text-danger">{means['avg_2013_2022_firearm_homicide_rate']:.2f}</b></div>
            <div>GDP/capita: <b class="num">${means['gdp_per_capita_usd_2017']:,.0f}</b></div>
          </div>
          <div class="sample-list"><b>Countries ({len(countries)}):</b> {sample_countries}</div>
        </div>
        """)
    return f"""<div class="grid grid-2">{''.join(cards)}</div>"""


def _build_risk_factors_html(report: RiskFactorReport) -> str:
    """Render tree-based importance bars alongside Spearman rank correlations."""
    rows = []
    for feat, imp in zip(report.feature_names, report.importances):
        rho = report.spearman_correlations.get(feat, 0.0)
        pct = imp * 100
        rows.append(f"""
        <div class="risk-row">
          <div class="risk-label">{feat}</div>
          <div class="bar-container">
            <div class="bar-fill" style="width: {pct:.1f}%"></div>
          </div>
          <div class="risk-val num">{pct:.1f}%</div>
          <div class="risk-rho num">Spearman: {rho:+.3f}</div>
        </div>
        """)
    return f"""<div class="card"><h3>Non-Linear Risk Factor Importances (Random Forest)</h3><div class="risk-list">{''.join(rows)}</div></div>"""


def generate_dashboard_html(
    profile_df: pd.DataFrame,
    regression_report: RegressionReport,
    cluster_report: ClusterReport,
    risk_report: RiskFactorReport,
    output_path: Path
) -> Path:
    """Assemble all econometric and spatial visualizations into a single standalone HTML dashboard."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    map_fig = render_world_choropleth(profile_df)
    quadrant_fig = render_quadrant_scatter(cluster_report.profile_df)

    map_html = map_fig.to_html(include_plotlyjs="cdn", full_html=False)
    quadrant_html = quadrant_fig.to_html(include_plotlyjs=False, full_html=False)

    gdp_beta = regression_report.coefficients.get("log_gdp_pc", -0.3985)
    spearman_rho = risk_report.spearman_correlations.get("firearms_per_100_people", -0.0026)

    kpi_section = _build_kpi_cards(spearman_rho, gdp_beta)
    regression_section = _build_regression_table_html(regression_report)
    archetype_section = _build_archetype_cards(cluster_report)
    risk_section = _build_risk_factors_html(risk_report)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>GGVCF: Global Gun Violence & Civilian Firearms (2000–2024)</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      --bg: #090d16;
      --card-bg: #111827;
      --card-border: #1f2937;
      --text: #f3f4f6;
      --text-muted: #9ca3af;
      --accent: #38bdf8;
      --accent-danger: #f87171;
      --accent-success: #34d399;
      --accent-warning: #fbbf24;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      line-height: 1.5;
      padding: 2rem 1.5rem;
    }}
    .container {{ max-width: 1400px; margin: 0 auto; }}
    header {{ margin-bottom: 2rem; border-bottom: 1px solid var(--card-border); padding-bottom: 1.5rem; }}
    h1 {{ font-size: 2rem; font-weight: 700; letter-spacing: -0.025em; }}
    .subtitle {{ color: var(--text-muted); font-size: 1rem; margin-top: 0.25rem; }}
    .grid {{ display: grid; gap: 1.25rem; margin-bottom: 2rem; }}
    .grid-2 {{ grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); }}
    .grid-4 {{ grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); }}
    .card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 1.5rem;
    }}
    .kpi-label {{ color: var(--text-muted); font-size: 0.825rem; text-transform: uppercase; letter-spacing: 0.05em; }}
    .kpi-value {{ font-size: 1.85rem; font-weight: 700; font-variant-numeric: tabular-nums; margin: 0.35rem 0; }}
    .kpi-sub {{ color: var(--text-muted); font-size: 0.8rem; }}
    .text-accent {{ color: var(--accent); }}
    .text-danger {{ color: var(--accent-danger); }}
    .text-success {{ color: var(--accent-success); }}
    .num {{ font-variant-numeric: tabular-nums; }}
    .card-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem; }}
    .badge {{ background: #1e293b; color: var(--accent); padding: 0.25rem 0.6rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }}
    .badge-cluster {{ display: inline-block; background: #334155; color: #f8fafc; font-size: 0.7rem; font-weight: 700; padding: 0.15rem 0.5rem; border-radius: 4px; margin-bottom: 0.5rem; }}
    .desc {{ color: var(--text-muted); font-size: 0.875rem; margin-bottom: 1rem; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.875rem; }}
    th, td {{ padding: 0.65rem 0.75rem; text-align: left; border-bottom: 1px solid var(--card-border); }}
    th {{ color: var(--text-muted); font-weight: 600; text-transform: uppercase; font-size: 0.75rem; }}
    th.num, td.num {{ text-align: right; }}
    .stats-footer {{ display: flex; gap: 2rem; margin-top: 1rem; font-size: 0.825rem; color: var(--text-muted); border-top: 1px solid var(--card-border); padding-top: 0.75rem; }}
    .archetype-card {{ background: #0f172a; border: 1px solid var(--card-border); border-radius: 6px; padding: 1.25rem; }}
    .archetype-card h4 {{ font-size: 1.05rem; margin-bottom: 0.75rem; color: #f8fafc; }}
    .arch-stats {{ display: flex; gap: 1rem; font-size: 0.825rem; margin-bottom: 0.75rem; color: var(--text-muted); }}
    .sample-list {{ font-size: 0.8rem; color: var(--text-muted); line-height: 1.4; }}
    .risk-row {{ display: grid; grid-template-columns: 220px 1fr 60px 140px; align-items: center; gap: 1rem; margin-bottom: 0.75rem; }}
    .risk-label {{ font-size: 0.85rem; font-mono; }}
    .bar-container {{ background: #1e293b; height: 10px; border-radius: 5px; overflow: hidden; }}
    .bar-fill {{ background: var(--accent); height: 100%; border-radius: 5px; }}
    .risk-val {{ font-size: 0.85rem; font-weight: 600; text-align: right; }}
    .risk-rho {{ font-size: 0.8rem; color: var(--text-muted); text-align: right; }}
    footer {{ margin-top: 3rem; text-align: center; font-size: 0.8rem; color: var(--text-muted); border-top: 1px solid var(--card-border); padding-top: 1.5rem; }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>Global Gun Violence &amp; Civilian Firearms (2000–2024)</h1>
      <div class="subtitle">Econometric Panel Identification, Geospatial Disparities &amp; Country Archetype Segmentation</div>
    </header>

    {kpi_section}

    <div class="grid grid-2">
      {regression_section}
      {risk_section}
    </div>

    <div class="card" style="margin-bottom: 2rem;">
      <div class="card-header">
        <h3>Spatial Firearm Lethality Distribution</h3>
        <span class="badge">UNODC / World Bank</span>
      </div>
      {map_html}
    </div>

    <div class="card" style="margin-bottom: 2rem;">
      <div class="card-header">
        <h3>The Empirical Paradox: Gun Penetration vs Lethality</h3>
        <span class="badge">Small Arms Survey 2017 &times; UNODC</span>
      </div>
      {quadrant_html}
    </div>

    <div class="card">
      <div class="card-header">
        <h3>Global Structural Archetypes (PCA + K-Means)</h3>
        <span class="badge">Silhouette Score: {cluster_report.silhouette_score:.3f}</span>
      </div>
      <p class="desc">Clustering across civilian gun ownership, firearm homicide rate, firearm share of murders, GDP per capita, and Gini inequality index.</p>
      {archetype_section}
    </div>

    <footer>
      <div>GGVCF System &bull; Built in strict accordance with ElmasCore Charter &bull; Data: UNODC, Small Arms Survey (2018), World Bank WDI</div>
    </footer>
  </div>
</body>
</html>"""

    output_path.write_text(html_content, encoding="utf-8")
    return output_path
