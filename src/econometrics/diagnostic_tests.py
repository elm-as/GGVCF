"""Diagnostic and specification tests for econometric validity.

Includes:
- Variance Inflation Factor (VIF) to detect multicollinearity among socio-economic controls
- Spearman non-parametric rank correlation matrix
- Publication-quality formatted summary tables
"""

from typing import Dict, List, Tuple
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor

from ..core.types import RegressionReport


def compute_vif_scores(df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, float]:
    """Calculate the Variance Inflation Factor (VIF) for each explanatory regressor.

    Rule of thumb: VIF > 5 indicates moderate multicollinearity, VIF > 10 severe.
    """
    clean_df = df[feature_cols].dropna().astype(float)
    if clean_df.empty or len(feature_cols) < 2:
        return {col: 1.0 for col in feature_cols}

    vif_dict: Dict[str, float] = {}
    values = clean_df.values
    for idx, col in enumerate(feature_cols):
        score = variance_inflation_factor(values, idx)
        vif_dict[col] = float(score)

    return vif_dict


def compute_spearman_rank_correlation(
    df: pd.DataFrame,
    col_x: str,
    col_y: str
) -> Tuple[float, float]:
    """Calculate non-parametric Spearman rank correlation and associated p-value.

    Robust to extreme outliers and highly skewed distributions.
    """
    sub = df[[col_x, col_y]].dropna()
    if len(sub) < 5:
        return 0.0, 1.0

    rho, pval = spearmanr(sub[col_x], sub[col_y])
    return float(rho), float(pval)


def format_regression_table(report: RegressionReport) -> pd.DataFrame:
    """Format regression estimates into an ElmasCore publication table."""
    records = []
    for var, coef in report.coefficients.items():
        std = report.std_errors.get(var, 0.0)
        t_stat = report.t_stats.get(var, 0.0)
        p_val = report.p_values.get(var, 1.0)

        # Standard significance asterisks
        stars = ""
        if p_val < 0.001:
            stars = "***"
        elif p_val < 0.01:
            stars = "**"
        elif p_val < 0.05:
            stars = "*"
        elif p_val < 0.1:
            stars = "†"

        records.append({
            "Variable": var,
            "Coefficient": f"{coef:+.4f}{stars}",
            "Std. Error": f"{std:.4f}",
            "t-Statistic": f"{t_stat:+.3f}",
            "p-Value": f"{p_val:.4f}",
            "95% CI": f"[{report.conf_int_lower.get(var, 0.0):+.3f}, {report.conf_int_upper.get(var, 0.0):+.3f}]"
        })

    return pd.DataFrame(records)
