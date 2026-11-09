"""Centralized configuration objects for the GGVCF econometric and spatial analysis pipeline.

Respects ElmasCore guidelines:
- Strict immutability (frozen dataclasses)
- Explicit default values grounded in domain literature (UNODC / Small Arms Survey)
- No hidden global variables
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class PathConfig:
    """Immutable filesystem paths for inputs, processed data, and outputs."""
    root_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "data")
    artifacts_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "artifacts")
    
    @property
    def civilian_firearms_path(self) -> Path:
        return self.data_dir / "civilian_firearms_2017.csv"
        
    @property
    def country_gun_profile_path(self) -> Path:
        return self.data_dir / "country_gun_profile.csv"
        
    @property
    def panel_timeseries_path(self) -> Path:
        return self.data_dir / "firearm_homicides_country_year.csv"
        
    @property
    def dashboard_output_path(self) -> Path:
        return self.artifacts_dir / "dashboard.html"


@dataclass(frozen=True)
class EconometricConfig:
    """Hyperparameters and variable specifications for panel regression."""
    dependent_var: str = "log_firearm_homicide_rate"
    exogenous_vars: List[str] = field(
        default_factory=lambda: [
            "log_gdp_pc",
            "gini_index",
            "urban_population_pct",
            "male_15_64_share_pct",
            "unemployment_pct",
            "military_spending_pct_gdp",
        ]
    )
    entity_effects: bool = True
    time_effects: bool = True
    cluster_entity: bool = True
    min_years_per_country: int = 5


@dataclass(frozen=True)
class ClusterConfig:
    """Configuration for cross-sectional country archetype segmentation."""
    n_clusters: int = 4
    random_state: int = 42
    pca_components: int = 2
    feature_columns: List[str] = field(
        default_factory=lambda: [
            "firearms_per_100_people",
            "avg_2013_2022_firearm_homicide_rate",
            "avg_2013_2022_firearm_share_of_homicides_pct",
            "gdp_per_capita_usd_2017",
            "gini_index_2017",
            "urban_population_pct_2017",
        ]
    )


@dataclass(frozen=True)
class AnalysisConfig:
    """Master orchestrator configuration."""
    paths: PathConfig = field(default_factory=PathConfig)
    econometrics: EconometricConfig = field(default_factory=EconometricConfig)
    clustering: ClusterConfig = field(default_factory=ClusterConfig)

