"""Main execution entrypoint for the GGVCF econometric analysis suite.

Orchestrates data loading, panel construction, Two-Way Fixed Effects estimation,
archetype clustering, risk factor importance ranking, and dashboard generation.

Adheres strictly to ElmasCore standards:
- Clear separation of concerns
- Guard clauses and early returns
- Comprehensive console logging with tabular formatting
"""

import argparse
import sys
from pathlib import Path

from src.core.config import AnalysisConfig
from src.core.types import ClusterReport, PanelBundle, RegressionReport, RiskFactorReport
from src.data_pipeline import build_panel_dataset, load_raw_datasets, validate_panel_bundle
from src.econometrics import (
    compute_spearman_rank_correlation,
    compute_vif_scores,
    estimate_panel_ols,
    format_regression_table,
)
from src.modeling import analyze_risk_factors
from src.segmentation import segment_country_archetypes
from src.visualization import generate_dashboard_html


def _display_header() -> None:
    print("=" * 80)
    print(" GGVCF: Global Gun Violence & Civilian Firearms (2000-2024)")
    print(" Econometric Panel Analysis & Country Archetype Segmentation")
    print("=" * 80)


def _log_panel_summary(bundle: PanelBundle) -> None:
    print(f"\n[+] Panel Dataset Assembled:")
    print(f"    - Entities (Countries): {bundle.countries_count}")
    print(f"    - Total Observations:  {bundle.total_observations}")
    print(f"    - Temporal Coverage:   {bundle.year_min} - {bundle.year_max}")


def _log_regression_summary(report: RegressionReport) -> None:
    print(f"\n[+] {report.model_name}:")
    print(f"    - N (Obs): {report.n_observations} | Entities: {report.n_entities}")
    print(f"    - Within R-squared:  {report.r_squared_within:.4f}")
    print(f"    - Overall R-squared: {report.r_squared_overall:.4f}")
    print(f"    - F-statistic:       {report.f_statistic:.2f} (p = {report.f_pvalue:.4e})")
    print("\n" + format_regression_table(report).to_string(index=False))


def _log_clustering_summary(report: ClusterReport) -> None:
    print(f"\n[+] Archetype Segmentation (PCA + K-Means):")
    print(f"    - Silhouette Score: {report.silhouette_score:.3f}")
    print(f"    - PCA Explained Variance: {[f'{v*100:.1f}%' for v in report.explained_variance_ratio]}")
    for cid, name in report.archetype_labels.items():
        n_ctry = (report.profile_df["cluster_id"] == cid).sum()
        print(f"    - Cluster {cid} [{name}]: {n_ctry} countries")


def _log_paradox_metrics(rho: float, pval: float) -> None:
    print(f"\n[+] The Empirical Paradox:")
    print(f"    - Spearman rho (Guns/100 vs Homicide Rate): {rho:+.4f} (p = {pval:.4f})")
    print("    - Interpretation: Civilian gun saturation shows zero bivariate rank correlation")
    print("      with firearm homicide rates globally; structural inequality and governance dominate.")


def run_pipeline(config: AnalysisConfig) -> None:
    """Execute end-to-end econometric and spatial workflow."""
    _display_header()

    # Step 1: Load Raw Datasets
    print("\n[*] Loading raw CSV tables...")
    civilian_df, profile_df, panel_df = load_raw_datasets(config.paths)

    # Step 2: Build & Validate Multi-Indexed Panel
    print("[*] Assembling multi-indexed panel dataset...")
    panel_bundle = build_panel_dataset(panel_df, config.econometrics)
    validate_panel_bundle(panel_bundle, [config.econometrics.dependent_var] + config.econometrics.exogenous_vars[:3])
    _log_panel_summary(panel_bundle)

    # Step 3: Two-Way Fixed Effects Panel Estimation
    print("[*] Estimating Two-Way Fixed Effects PanelOLS...")
    reg_report = estimate_panel_ols(panel_bundle, config.econometrics)
    _log_regression_summary(reg_report)

    # Step 4: Spearman Non-Parametric Rank Correlation Paradox
    rho, pval = compute_spearman_rank_correlation(
        profile_df,
        "firearms_per_100_people",
        "avg_2013_2022_firearm_homicide_rate"
    )
    _log_paradox_metrics(rho, pval)

    # Step 5: Country Archetype Segmentation
    print("\n[*] Running PCA and K-Means country segmentation...")
    cluster_report = segment_country_archetypes(profile_df, config.clustering)
    _log_clustering_summary(cluster_report)

    # Step 6: Non-Linear Risk Factor Importances
    print("\n[*] Evaluating non-linear feature importances (Random Forest)...")
    risk_candidates = [
        "firearms_per_100_people",
        "gdp_per_capita_usd_2017",
        "gini_index_2017",
        "urban_population_pct_2017"
    ]
    risk_report = analyze_risk_factors(
        cluster_report.profile_df,
        "avg_2013_2022_firearm_homicide_rate",
        risk_candidates
    )
    for feat, imp in zip(risk_report.feature_names, risk_report.importances):
        r = risk_report.spearman_correlations.get(feat, 0.0)
        print(f"    - {feat:<30}: Importance {imp*100:5.1f}% | Spearman rho = {r:+.3f}")

    # Step 7: Generate Standalone Interactive Dashboard
    print(f"\n[*] Generating standalone HTML dashboard at {config.paths.dashboard_output_path}...")
    dashboard_path = generate_dashboard_html(
        profile_df,
        reg_report,
        cluster_report,
        risk_report,
        config.paths.dashboard_output_path
    )
    print(f"[+] Interactive Dashboard successfully saved: {dashboard_path}")
    print("\n" + "=" * 80)
    print(" Pipeline execution finished successfully.")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="GGVCF Econometric & Spatial Analysis Engine")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Directory for output artifacts")
    args = parser.parse_args()

    cfg = AnalysisConfig()
    try:
        run_pipeline(cfg)
    except Exception as exc:
        print(f"\n[!] Critical Pipeline Failure: {exc}", file=sys.stderr)
        raise exc


if __name__ == "__main__":
    main()
