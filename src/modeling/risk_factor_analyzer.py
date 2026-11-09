"""Interpretable non-linear risk factor analysis module.

Quantifies the relative explanatory power of socio-demographic indicators
and firearms penetration using ensemble decision trees and rank correlation.
"""

from typing import Dict, List
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor

from ..core.types import RiskFactorReport


def analyze_risk_factors(
    profile_df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str]
) -> RiskFactorReport:
    """Evaluate non-linear feature importances and rank correlations on country profiles.

    Args:
        profile_df: DataFrame with country cross-sectional features and target.
        target_col: Dependent indicator (e.g. 'avg_2013_2022_firearm_homicide_rate').
        feature_cols: Explanatory candidate indicators.

    Returns:
        RiskFactorReport with ranked importances and Spearman correlation coefficients.
    """
    valid_features = [c for c in feature_cols if c in profile_df.columns]
    data = profile_df.dropna(subset=[target_col] + valid_features).copy()
    if len(data) < 10:
        raise ValueError("Insufficient sample size to compute reliable risk factor importances.")

    x = data[valid_features]
    y = data[target_col]

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        min_samples_leaf=2
    )
    model.fit(x, y)

    importances = [float(v) for v in model.feature_importances_]

    # Compute individual non-parametric Spearman rank correlations
    spearman_dict: Dict[str, float] = {}
    for feat in valid_features:
        rho, _ = spearmanr(data[feat], y)
        spearman_dict[feat] = float(rho)

    # Sort in descending order of feature importance
    sorted_pairs = sorted(zip(valid_features, importances), key=lambda p: p[1], reverse=True)
    sorted_features = [p[0] for p in sorted_pairs]
    sorted_importances = [p[1] for p in sorted_pairs]

    return RiskFactorReport(
        feature_names=sorted_features,
        importances=sorted_importances,
        spearman_correlations=spearman_dict,
    )

