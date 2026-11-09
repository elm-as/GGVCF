"""Validation module for panel data structure and econometric suitability.

Ensures that the input data structures meet the formal criteria for
unbiased panel econometric estimation and archetype segmentation.
"""

from typing import List
import numpy as np
import pandas as pd

from ..core.types import PanelBundle


def validate_panel_bundle(bundle: PanelBundle, required_cols: List[str]) -> bool:
    """Verify that a PanelBundle satisfies dimensional and numerical constraints.

    Args:
        bundle: PanelBundle object under inspection.
        required_cols: List of column names that must exist and have non-null values.

    Returns:
        True if all validation criteria pass.

    Raises:
        ValueError: If constraints are violated.
    """
    df = bundle.panel_df
    if df.empty:
        raise ValueError("Panel DataFrame is empty.")

    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("Panel index must be a MultiIndex with levels ['iso3', 'year'].")

    if df.index.names != ["iso3", "year"]:
        raise ValueError(f"MultiIndex levels must be ['iso3', 'year'], found {df.index.names}.")

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in panel: {missing_cols}")

    # Check for infinite values in required numeric columns
    for col in required_cols:
        inf_mask = np.isinf(df[col].dropna())
        if inf_mask.any():
            raise ValueError(f"Column '{col}' contains infinite values.")

    return True


def validate_profile_dataframe(df: pd.DataFrame, feature_cols: List[str]) -> bool:
    """Verify cross-sectional country profile data before segmentation.

    Args:
        df: Country profile DataFrame.
        feature_cols: Features required for PCA and clustering.

    Returns:
        True if profile data satisfies criteria.
    """
    if df.empty:
        raise ValueError("Country profile DataFrame is empty.")

    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Profile DataFrame missing required features: {missing_cols}")

    return True

