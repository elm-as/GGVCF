"""Interactive quadrant scatter plot renderer for civilian guns vs firearm homicides.

Highlights the empirical independence between gun saturation and lethal violence,
with explicit callouts for extreme outliers (USA, Switzerland, El Salvador, Jamaica).
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def render_quadrant_scatter(df: pd.DataFrame) -> go.Figure:
    """Construct an annotated 4-quadrant scatter plot.

    Args:
        df: Cross-sectional DataFrame with 'firearms_per_100_people' and
            'avg_2013_2022_firearm_homicide_rate'.

    Returns:
        Plotly Figure object ready for HTML dashboard embedding.
    """
    clean_df = df.dropna(subset=["firearms_per_100_people", "avg_2013_2022_firearm_homicide_rate"]).copy()

    x_median = float(clean_df["firearms_per_100_people"].median())
    y_median = float(clean_df["avg_2013_2022_firearm_homicide_rate"].median())

    fig = go.Figure()

    # Base points colored by region or archetype
    colors = {
        "High-Lethality Crisis (Critical Gun Violence)": "#ef4444",
        "High Ownership / Low Lethality (Saturated & Stable)": "#3b82f6",
        "Strictly Regulated / Institutional Shield (Low Lethality)": "#10b981",
        "Intermediate Vulnerability Profile": "#f59e0b",
    }

    if "archetype_name" in clean_df.columns:
        for arch, grp in clean_df.groupby("archetype_name"):
            color = colors.get(str(arch), "#64748b")
            fig.add_trace(go.Scatter(
                x=grp["firearms_per_100_people"],
                y=grp["avg_2013_2022_firearm_homicide_rate"],
                mode="markers",
                name=str(arch),
                text=grp["country"],
                customdata=np.stack((
                    grp["country"],
                    grp["region"] if "region" in grp.columns else grp["country"],
                    grp["gdp_per_capita_usd_2017"].fillna(0),
                    grp["gini_index_2017"].fillna(0)
                ), axis=-1),
                hovertemplate=(
                    "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                    "Civilian Guns / 100: <b>%{x:.1f}</b><br>"
                    "Firearm Homicides / 100k: <b>%{y:.2f}</b><br>"
                    "GDP per capita: $%{customdata[2]:,.0f}<br>"
                    "Gini Index: %{customdata[3]:.1f}<extra></extra>"
                ),
                marker=dict(size=9, color=color, opacity=0.85, line=dict(width=1, color="rgba(255, 255, 255, 0.2)")),
            ))
    else:
        fig.add_trace(go.Scatter(
            x=clean_df["firearms_per_100_people"],
            y=clean_df["avg_2013_2022_firearm_homicide_rate"],
            mode="markers+text",
            text=clean_df["country"],
            marker=dict(size=9, color="#38bdf8", opacity=0.8),
        ))

    # Median reference threshold lines
    fig.add_vline(x=x_median, line_dash="dash", line_color="#475569", annotation_text=f"Median Guns: {x_median:.1f}")
    fig.add_hline(y=y_median, line_dash="dash", line_color="#475569", annotation_text=f"Median Homicide: {y_median:.2f}")

    # Annotated Outliers
    key_outliers = ["United States", "Switzerland", "Serbia", "Finland", "El Salvador", "Jamaica", "Honduras"]
    for country in key_outliers:
        row = clean_df[clean_df["country"] == country]
        if row.empty:
            continue
        cx = float(row["firearms_per_100_people"].iloc[0])
        cy = float(row["avg_2013_2022_firearm_homicide_rate"].iloc[0])
        fig.add_annotation(
            x=cx,
            y=cy,
            text=f"<b>{country}</b>",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.5,
            arrowcolor="#f8fafc",
            font=dict(color="#f8fafc", size=11),
            bgcolor="#1e293b",
            bordercolor="#475569",
            borderwidth=1,
            borderpad=4,
            ax=20 if cx < 80 else -40,
            ay=-30 if cy > 10 else 30,
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        title="Civilian Gun Saturation vs. Firearm Homicides (The Empirical Paradox)",
        xaxis=dict(title="Civilian Firearms per 100 People", gridcolor="#1e293b", zerolinecolor="#334155"),
        yaxis=dict(title="Firearm Homicides per 100k (2013–2022 Avg)", gridcolor="#1e293b", zerolinecolor="#334155"),
        margin=dict(l=50, r=20, t=50, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif", color="#e2e8f0"),
    )

    return fig
