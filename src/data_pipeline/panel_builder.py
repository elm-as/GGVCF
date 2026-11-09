"""Panel dataset constructor for GGVCF econometric modeling.

Transforms country-year observations into a strictly formatted, multi-indexed
panel DataFrame suited for Two-Way Fixed Effects PanelOLS.
"""

from typing import List, Optional
import numpy as np
import pandas as pd

from ..core.config import EconometricConfig
from ..core.types import PanelBundle


def _apply_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """Compute logarithmic transformations on strictly positive quantities."""
    out = df.copy()
    if "firearm_homicide_rate" in out.columns:
        out["log_firearm_homicide_rate"] = np.log1p(out["firearm_homicide_rate"].clip(lower=0.0))
    if "homicide_rate" in out.columns:
        out["log_homicide_rate"] = np.log1p(out["homicide_rate"].clip(lower=0.0))
    if "gdp_per_capita_usd" in out.columns:
        out["log_gdp_pc"] = np.log(out["gdp_per_capita_usd"].clip(lower=1.0))
    return out


def _impute_structural_indicators(df: pd.DataFrame, indicator_cols: List[str]) -> pd.DataFrame:
    """Forward and backward fill slow-moving structural World Bank indicators per country."""
    out = df.copy()
    for col in indicator_cols:
        if col in out.columns:
            out[col] = out.groupby("iso3")[col].transform(lambda s: s.ffill().bfill())
    return out


def build_panel_dataset(
    raw_panel_df: pd.DataFrame,
    config: EconometricConfig,
    fill_indicators: bool = True
) -> PanelBundle:
    """Clean, filter, and structure raw country-year data into an econometric PanelBundle.

    Args:
        raw_panel_df: Raw DataFrame containing country-year metrics.
        config: Econometric hyperparameter specifications.
        fill_indicators: Whether to interpolate slow-moving structural metrics per entity.

    Returns:
        PanelBundle containing multi-indexed [iso3, year] DataFrame and metadata.
    """
    if raw_panel_df.empty:
        raise ValueError("Cannot build panel from an empty DataFrame.")

    # Guard clause: require target dependent variable
    base_target = "firearm_homicide_rate"
    if base_target not in raw_panel_df.columns:
        raise KeyError(f"Required target '{base_target}' absent from panel data.")

    df = raw_panel_df.dropna(subset=[base_target]).copy()
    if df.empty:
        raise ValueError("No valid observations for firearm homicide rate.")

    if fill_indicators:
        structural_vars = ["gini_index", "unemployment_pct", "military_spending_pct_gdp"]
        df = _impute_structural_indicators(df, structural_vars)

    df = _apply_log_transforms(df)

    # Filter countries having at least min_years_per_country observations
    country_counts = df.groupby("iso3")["year"].count()
    valid_countries = country_counts[country_counts >= config.min_years_per_country].index
    df = df[df["iso3"].isin(valid_countries)].copy()

    # Establish MultiIndex (Entity, Time) for PanelOLS
    df = df.sort_values(by=["iso3", "year"]).reset_index(drop=True)
    panel_multi = df.set_index(["iso3", "year"])

    years = df["year"].dropna().astype(int)
    year_min = int(years.min()) if not years.empty else 0
    year_max = int(years.max()) if not years.empty else 0

    return PanelBundle(
        panel_df=panel_multi,
        countries_count=len(valid_countries),
        total_observations=len(df),
        year_min=year_min,
        year_max=year_max,
    )
