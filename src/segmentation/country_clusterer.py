"""Country archetype segmentation module using PCA and K-Means.

Discovers global structural archetypes combining civilian firearms penetration,
empirical firearm lethality, income per capita, and Gini inequality.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from ..core.config import ClusterConfig
from ..core.types import ClusterReport


def _impute_profile_features(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Impute missing numerical features using regional median with global fallback."""
    out = df.copy()
    for feat in features:
        if feat not in out.columns:
            continue
        # Regional median imputation
        if "region" in out.columns:
            out[feat] = out.groupby("region")[feat].transform(lambda s: s.fillna(s.median()))
        # Fallback to global median
        global_med = out[feat].median()
        out[feat] = out[feat].fillna(global_med)
    return out


def _assign_archetype_names(cluster_means: pd.DataFrame) -> Dict[int, str]:
    """Assign domain archetypes based on cluster mean traits."""
    labels: Dict[int, str] = {}
    gun_col = "firearms_per_100_people"
    fhr_col = "avg_2013_2022_firearm_homicide_rate"

    for cluster_id in cluster_means.index:
        mean_guns = cluster_means.loc[cluster_id, gun_col]
        mean_fhr = cluster_means.loc[cluster_id, fhr_col]

        if mean_fhr >= 10.0:
            labels[cluster_id] = "High-Lethality Crisis (Critical Gun Violence)"
        elif mean_guns >= 25.0:
            labels[cluster_id] = "High Ownership / Low Lethality (Saturated & Stable)"
        elif mean_fhr < 2.0:
            labels[cluster_id] = "Strictly Regulated / Institutional Shield (Low Lethality)"
        else:
            labels[cluster_id] = "Intermediate Vulnerability Profile"

    return labels


def segment_country_archetypes(
    profile_df: pd.DataFrame,
    config: ClusterConfig
) -> ClusterReport:
    """Perform PCA dimensionality reduction and K-Means archetype segmentation.

    Args:
        profile_df: Cross-sectional DataFrame with country firearm and demographic indicators.
        config: Clustering specifications.

    Returns:
        ClusterReport with segmented DataFrame, silhouette score, and archetypes.
    """
    # Guard clause: filter countries having actual gun ownership and homicide data
    base_mask = profile_df["firearms_per_100_people"].notnull() & profile_df["avg_2013_2022_firearm_homicide_rate"].notnull()
    valid_df = profile_df[base_mask].copy()
    if len(valid_df) < config.n_clusters:
        raise ValueError("Insufficient country observations for clustering.")

    imputed_df = _impute_profile_features(valid_df, config.feature_columns)
    x_matrix = imputed_df[config.feature_columns].values

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_matrix)

    kmeans = KMeans(
        n_clusters=config.n_clusters,
        random_state=config.random_state,
        n_init=10
    )
    cluster_labels = kmeans.fit_predict(x_scaled)
    imputed_df["cluster_id"] = cluster_labels

    pca = PCA(n_components=config.pca_components, random_state=config.random_state)
    pca_coords = pca.fit_transform(x_scaled)
    imputed_df["pca_dim_1"] = pca_coords[:, 0]
    imputed_df["pca_dim_2"] = pca_coords[:, 1]

    sil_score = float(silhouette_score(x_scaled, cluster_labels))
    cluster_means = imputed_df.groupby("cluster_id")[config.feature_columns].mean()
    archetype_map = _assign_archetype_names(cluster_means)
    imputed_df["archetype_name"] = imputed_df["cluster_id"].map(archetype_map)

    return ClusterReport(
        profile_df=imputed_df,
        silhouette_score=sil_score,
        archetype_labels=archetype_map,
        cluster_means=cluster_means,
        explained_variance_ratio=[float(v) for v in pca.explained_variance_ratio_],
    )
