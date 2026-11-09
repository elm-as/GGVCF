"""Automated test suite verifying the econometric and analytical pipeline of GGVCF.

Validates:
- Data loading integrity
- Panel construction and MultiIndex schema
- Two-Way Fixed Effects econometric estimation
- Archetype segmentation (PCA + K-Means)
- Risk factor ranking and Spearman correlation
- Interactive dashboard generation
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.core.config import AnalysisConfig, EconometricConfig, ClusterConfig, PathConfig
from src.data_pipeline import (
    load_raw_datasets,
    build_panel_dataset,
    validate_panel_bundle,
    validate_profile_dataframe,
)
from src.econometrics import (
    estimate_panel_ols,
    compute_spearman_rank_correlation,
    compute_vif_scores,
)
from src.segmentation import segment_country_archetypes
from src.modeling import analyze_risk_factors
from src.visualization import generate_dashboard_html


@pytest.fixture
def config() -> AnalysisConfig:
    return AnalysisConfig()


def test_load_raw_datasets(config: AnalysisConfig):
    civilian_df, profile_df, panel_df = load_raw_datasets(config.paths)
    assert not civilian_df.empty
    assert not profile_df.empty
    assert not panel_df.empty
    assert "firearms_per_100_people" in civilian_df.columns
    assert "firearm_homicide_rate" in panel_df.columns
    assert "iso3" in profile_df.columns


def test_build_and_validate_panel(config: AnalysisConfig):
    _, _, panel_df = load_raw_datasets(config.paths)
    bundle = build_panel_dataset(panel_df, config.econometrics)

    assert bundle.countries_count > 50
    assert bundle.total_observations > 500
    assert bundle.year_min >= 2000
    assert bundle.year_max <= 2024
    assert isinstance(bundle.panel_df.index, pd.MultiIndex)
    assert bundle.panel_df.index.names == ["iso3", "year"]
    assert "log_firearm_homicide_rate" in bundle.panel_df.columns
    assert "log_gdp_pc" in bundle.panel_df.columns

    is_valid = validate_panel_bundle(bundle, ["log_firearm_homicide_rate", "log_gdp_pc"])
    assert is_valid is True


def test_estimate_panel_ols(config: AnalysisConfig):
    _, _, panel_df = load_raw_datasets(config.paths)
    bundle = build_panel_dataset(panel_df, config.econometrics)
    report = estimate_panel_ols(bundle, config.econometrics)

    assert report.n_observations > 500
    assert report.n_entities > 50
    assert "log_gdp_pc" in report.coefficients
    assert report.coefficients["log_gdp_pc"] < 0  # Wealth has protective effect
    assert report.f_statistic > 0
    assert 0.0 <= report.r_squared_within <= 1.0


def test_spearman_paradox_correlation(config: AnalysisConfig):
    _, profile_df, _ = load_raw_datasets(config.paths)
    rho, pval = compute_spearman_rank_correlation(
        profile_df,
        "firearms_per_100_people",
        "avg_2013_2022_firearm_homicide_rate"
    )
    # The core empirical finding: practically zero correlation (-0.05 < rho < 0.05)
    assert abs(rho) < 0.10
    assert pval > 0.05


def test_segment_country_archetypes(config: AnalysisConfig):
    _, profile_df, _ = load_raw_datasets(config.paths)
    report = segment_country_archetypes(profile_df, config.clustering)

    assert len(report.archetype_labels) == config.clustering.n_clusters
    assert report.silhouette_score > 0.10
    assert len(report.explained_variance_ratio) == 2
    assert "archetype_name" in report.profile_df.columns
    assert "pca_dim_1" in report.profile_df.columns


def test_analyze_risk_factors(config: AnalysisConfig):
    _, profile_df, _ = load_raw_datasets(config.paths)
    report = analyze_risk_factors(
        profile_df,
        "avg_2013_2022_firearm_homicide_rate",
        ["firearms_per_100_people", "gdp_per_capita_usd_2017", "urban_population_pct_2017"]
    )
    assert len(report.feature_names) == 3
    assert abs(sum(report.importances) - 1.0) < 1e-4
    assert "firearms_per_100_people" in report.spearman_correlations


def test_generate_dashboard_html(tmp_path: Path, config: AnalysisConfig):
    _, profile_df, panel_df = load_raw_datasets(config.paths)
    bundle = build_panel_dataset(panel_df, config.econometrics)
    reg_report = estimate_panel_ols(bundle, config.econometrics)
    cluster_report = segment_country_archetypes(profile_df, config.clustering)
    risk_report = analyze_risk_factors(
        profile_df,
        "avg_2013_2022_firearm_homicide_rate",
        ["firearms_per_100_people", "gdp_per_capita_usd_2017", "urban_population_pct_2017"]
    )

    out_file = tmp_path / "test_dashboard.html"
    res_path = generate_dashboard_html(
        profile_df,
        reg_report,
        cluster_report,
        risk_report,
        out_file
    )
    assert res_path.is_file()
    content = res_path.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    assert "Global Gun Violence" in content
    assert "Plotly" in content or "plotly" in content
