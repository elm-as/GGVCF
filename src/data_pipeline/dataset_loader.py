"""Data loader for GGVCF raw tabular sources.

Adheres strictly to ElmasCore guidelines:
- Explicit path resolution and guard clauses
- Pure loading functions returning immutable or clean DataFrames
- Graceful error handling with descriptive exceptions
"""

from pathlib import Path
from typing import Tuple
import pandas as pd

from ..core.config import PathConfig


def load_raw_datasets(paths: PathConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the 3 foundational GGVCF CSV tables.

    Returns:
        Tuple of (civilian_firearms_df, country_profile_df, panel_timeseries_df)
    """
    if not paths.civilian_firearms_path.is_file():
        raise FileNotFoundError(f"Missing civilian firearms data at: {paths.civilian_firearms_path}")
    if not paths.country_gun_profile_path.is_file():
        raise FileNotFoundError(f"Missing country gun profile data at: {paths.country_gun_profile_path}")
    if not paths.panel_timeseries_path.is_file():
        raise FileNotFoundError(f"Missing panel timeseries data at: {paths.panel_timeseries_path}")

    civilian_df = pd.read_csv(paths.civilian_firearms_path)
    profile_df = pd.read_csv(paths.country_gun_profile_path)
    panel_df = pd.read_csv(paths.panel_timeseries_path)

    civilian_df.columns = [c.strip() for c in civilian_df.columns]
    profile_df.columns = [c.strip() for c in profile_df.columns]
    panel_df.columns = [c.strip() for c in panel_df.columns]

    return civilian_df, profile_df, panel_df
