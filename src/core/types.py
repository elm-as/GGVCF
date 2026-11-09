"""Value objects and result containers for GGVCF.

Respects ElmasCore guidelines:
- Strictly typed structures
- Clean separation between models, metrics, and tabular reports
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd


@dataclass(frozen=True)
class PanelBundle:
    """Cleaned multi-index panel ready for econometric estimation."""
    panel_df: pd.DataFrame
    countries_count: int
    total_observations: int
    year_min: int
    year_max: int


@dataclass(frozen=True)
class RegressionReport:
    """Summary metrics and statistical parameters from Two-Way Fixed Effects PanelOLS."""
    model_name: str
    coefficients: Dict[str, float]
    std_errors: Dict[str, float]
    t_stats: Dict[str, float]
    p_values: Dict[str, float]
    conf_int_lower: Dict[str, float]
    conf_int_upper: Dict[str, float]
    r_squared_within: float
    r_squared_between: float
    r_squared_overall: float
    f_statistic: float
    f_pvalue: float
    n_observations: int
    n_entities: int
    summary_text: str


@dataclass(frozen=True)
class ClusterReport:
    """Archetype segmentation results and dimensional reduction coordinates."""
    profile_df: pd.DataFrame
    silhouette_score: float
    archetype_labels: Dict[int, str]
    cluster_means: pd.DataFrame
    explained_variance_ratio: List[float]


@dataclass(frozen=True)
class RiskFactorReport:
    """Feature importance ranking and directional analysis from interpretable estimators."""
    feature_names: List[str]
    importances: List[float]
    spearman_correlations: Dict[str, float]
