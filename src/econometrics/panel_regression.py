"""Two-Way Fixed Effects Panel econometric modeling module.

Estimates econometric equations isolating within-country temporal shocks from
unobserved time-invariant institutional heterogeneity (Entity Effects) and
global macroeconomic shocks (Time Effects).
"""

from typing import List
from linearmodels.panel import PanelOLS
import statsmodels.api as sm
import pandas as pd

from ..core.config import EconometricConfig
from ..core.types import PanelBundle, RegressionReport


def estimate_panel_ols(bundle: PanelBundle, config: EconometricConfig) -> RegressionReport:
    """Estimate a Two-Way Fixed Effects PanelOLS model with clustered standard errors.

    Args:
        bundle: Multi-indexed PanelBundle.
        config: Econometric specifications.

    Returns:
        Structured RegressionReport value object.
    """
    df = bundle.panel_df
    target = config.dependent_var
    covariates = [c for c in config.exogenous_vars if c in df.columns]

    # Filter complete cases for the specific equation
    regression_df = df.dropna(subset=[target] + covariates).copy()
    if regression_df.empty:
        raise ValueError("No complete observations remaining for panel regression.")

    y = regression_df[target]
    x = regression_df[covariates]

    model = PanelOLS(
        y,
        x,
        entity_effects=config.entity_effects,
        time_effects=config.time_effects,
        check_rank=False
    )

    cov_kwargs = {"cov_type": "clustered", "cluster_entity": config.cluster_entity} if config.cluster_entity else {"cov_type": "robust"}
    results = model.fit(**cov_kwargs)

    conf_int = results.conf_int()
    conf_lower = conf_int.iloc[:, 0].to_dict()
    conf_upper = conf_int.iloc[:, 1].to_dict()

    return RegressionReport(
        model_name="Two-Way Fixed Effects PanelOLS (Entity + Time)",
        coefficients={k: float(v) for k, v in results.params.items()},
        std_errors={k: float(v) for k, v in results.std_errors.items()},
        t_stats={k: float(v) for k, v in results.tstats.items()},
        p_values={k: float(v) for k, v in results.pvalues.items()},
        conf_int_lower={k: float(v) for k, v in conf_lower.items()},
        conf_int_upper={k: float(v) for k, v in conf_upper.items()},
        r_squared_within=float(results.rsquared_within),
        r_squared_between=float(results.rsquared_between),
        r_squared_overall=float(results.rsquared_overall),
        f_statistic=float(results.f_statistic.stat),
        f_pvalue=float(results.f_statistic.pval),
        n_observations=int(results.nobs),
        n_entities=int(results.entity_info.total),
        summary_text=str(results.summary),
    )


def estimate_cross_sectional_ols(
    profile_df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str]
) -> RegressionReport:
    """Estimate a robust OLS regression on cross-sectional country profiles.

    Tests the empirical significance of civilian gun ownership rates relative to
    structural inequality (Gini) and GDP per capita.
    """
    valid_features = [c for c in feature_cols if c in profile_df.columns]
    data = profile_df.dropna(subset=[target_col] + valid_features).copy()
    if data.empty:
        raise ValueError("No valid cross-sectional data available for OLS.")

    y = data[target_col]
    x = sm.add_constant(data[valid_features])

    model = sm.OLS(y, x)
    res = model.fit(cov_type="HC3")

    conf_int = res.conf_int()
    conf_lower = {k: float(conf_int.loc[k, 0]) for k in res.params.index}
    conf_upper = {k: float(conf_int.loc[k, 1]) for k in res.params.index}

    return RegressionReport(
        model_name="Cross-Sectional Robust OLS (HC3 Heteroskedasticity-Consistent)",
        coefficients={k: float(v) for k, v in res.params.items()},
        std_errors={k: float(v) for k, v in res.bse.items()},
        t_stats={k: float(v) for k, v in res.tvalues.items()},
        p_values={k: float(v) for k, v in res.pvalues.items()},
        conf_int_lower=conf_lower,
        conf_int_upper=conf_upper,
        r_squared_within=0.0,
        r_squared_between=0.0,
        r_squared_overall=float(res.rsquared),
        f_statistic=float(res.fvalue) if res.fvalue is not None else 0.0,
        f_pvalue=float(res.f_pvalue) if res.f_pvalue is not None else 1.0,
        n_observations=int(res.nobs),
        n_entities=int(res.nobs),
        summary_text=str(res.summary()),
    )
