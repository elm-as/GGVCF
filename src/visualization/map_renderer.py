"""Geospatial choropleth map renderer for GGVCF global indicators.

Renders high-definition world maps displaying firearm homicide rates
and civilian firearm ownership with custom ElmasCore dark palette.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def render_world_choropleth(
    df: pd.DataFrame,
    value_col: str = "avg_2013_2022_firearm_homicide_rate",
    title: str = "Global Firearm Homicide Rate (per 100,000 population)"
) -> go.Figure:
    """Create a high-resolution world choropleth map.

    Args:
        df: DataFrame with 'iso3', 'country', and target metrics.
        value_col: Quantitative column used for color shading.
        title: Plot title.

    Returns:
        Plotly Figure object.
    """
    clean_df = df.dropna(subset=["iso3", value_col]).copy()

    fig = px.choropleth(
        clean_df,
        locations="iso3",
        color=value_col,
        hover_name="country",
        hover_data={
            "iso3": True,
            value_col: ":.2f",
            "firearms_per_100_people": ":.1f",
            "gdp_per_capita_usd_2017": ":$,.0f",
        } if "firearms_per_100_people" in clean_df.columns else {"iso3": True, value_col: ":.2f"},
        color_continuous_scale="Reds",
        projection="natural earth",
        title=title,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        margin=dict(l=10, r=10, t=50, b=10),
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif", color="#e2e8f0"),
        coloraxis_colorbar=dict(
            title=dict(text="Rate / 100k", font=dict(color="#94a3b8", size=12)),
            tickfont=dict(color="#cbd5e1"),
            thickness=14,
            len=0.75,
        ),
        geo=dict(
            bgcolor="#0f172a",
            showocean=True,
            oceancolor="#090d16",
            showlakes=True,
            lakecolor="#090d16",
            showland=True,
            landcolor="#1e293b",
            showcoastlines=True,
            coastlinecolor="#334155",
            showframe=False,
        )
    )

    return fig

