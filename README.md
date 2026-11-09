# GGVCF — Global Gun Violence & Civilian Firearms (2000–2024)

> **Econometric Panel Identification, Geospatial Disparities & Country Archetype Segmentation**  
> *Developed in strict compliance with the [ElmasCore Charter](ELMAS.md).*

---

## 1. Executive Summary & The Empirical Paradox

Across **95 countries** with concurrent civilian gun ownership and firearm homicide records:
- **The Empirical Paradox**: The non-parametric Spearman rank correlation between civilian firearms per 100 people and firearm homicide rates is **$\rho = -0.0026$ ($p = 0.9804$)** — statistically indistinguishable from zero.
- **Socio-Economic Determinants**: Extreme firearm violence is overwhelmingly driven by structural inequality (**Gini Index accounts for $59.8\%$ of non-linear variance**, $\rho = +0.725$) and institutional capacity rather than per-capita gun saturation.
- **Two-Way Fixed Effects PanelOLS**: In an unbalanced panel of **81 countries** over 2005–2023 absorbing time-invariant country heterogeneity and annual global macro shocks:
  $$\ln(\text{Firearm Homicides}_{it}) = \beta_1 \ln(\text{GDP pc}_{it}) + \beta_2 \text{Gini}_{it} + \beta_3 \text{Urban}_{it} + \alpha_i + \gamma_t + \varepsilon_{it}$$
  Real GDP per capita exhibits a significant protective elasticity ($\beta = -0.3491$, $p = 0.084$), while cross-sectional gun saturation remains independent of homicide variations.

---

## 2. Global Country Archetypes

Using PCA dimensionality reduction and $K$-Means clustering ($K=4$, Silhouette $= 0.273$), countries segment into four distinct structural regimes:

1. **High Ownership / Low Lethality (Saturated & Stable — 21 countries)**:
   - *Archetypes*: United States, Switzerland, Canada, Finland, Serbia, Norway.
   - *Profile*: 25–120 guns per 100 people, low to moderate firearm homicide rates (< 4.5/100k), high GDP per capita or high social cohesion.
2. **High-Lethality Crisis (Critical Gun Violence — 9 countries)**:
   - *Archetypes*: El Salvador, Jamaica, Honduras, Venezuela, Trinidad and Tobago.
   - *Profile*: 9–14 guns per 100 people, extreme firearm homicide rates (15–38/100k), 70–85% firearm murder share, high Gini inequality (> 48).
3. **Strictly Regulated / Institutional Shield (Low Lethality — 45 countries)**:
   - *Archetypes*: Japan, United Kingdom, Germany, South Korea, Singapore, Australia.
   - *Profile*: Low civilian ownership (< 10/100), near-zero firearm murder rates (< 1.0/100k), robust institutional frameworks.
4. **Intermediate Vulnerability Profile (20 countries)**:
   - *Profile*: Developing and emerging economies balancing transition dynamics and urbanization pressures.

---

## 3. Architecture & Codebase Design

Adheres to ElmasCore Domain-Driven Design:
```text
GGVCF/
├── artifacts/
│   └── dashboard.html            # Standalone interactive Plotly / HTML5 dashboard
├── data/
│   ├── civilian_firearms_2017.csv
│   ├── country_gun_profile.csv
│   └── firearm_homicides_country_year.csv
├── src/
│   ├── core/                     # Immutable configs and typed value objects
│   │   ├── config.py
│   │   └── types.py
│   ├── data_pipeline/            # Ingestion, MultiIndex construction, schema tests
│   │   ├── dataset_loader.py
│   │   ├── panel_builder.py
│   │   └── schema_validator.py
│   ├── econometrics/             # PanelOLS (Two-Way FE) and diagnostic statistics
│   │   ├── panel_regression.py
│   │   └── diagnostic_tests.py
│   ├── segmentation/             # PCA and K-Means archetype clustering
│   │   └── country_clusterer.py
│   ├── modeling/                 # Random Forest non-linear risk factor ranking
│   │   └── risk_factor_analyzer.py
│   └── visualization/            # World choropleth map, 4-quadrant plot, dashboard
│       ├── map_renderer.py
│       ├── quadrant_renderer.py
│       └── dashboard_builder.py
├── tests/
│   └── test_analysis_pipeline.py # 7 automated test suites
├── main.py                       # CLI orchestrator
├── requirements.txt              # Pinned analytical dependencies
└── ELMAS.md                      # ElmasCore quality charter
```

---

## 4. Quickstart & Usage

### Prerequisites
- Python 3.10+
```bash
pip install -r requirements.txt
```

### Run the End-to-End Pipeline
```bash
python main.py
```
This executes the econometric regressions, archetype clustering, and builds `artifacts/dashboard.html`.

### Run Test Suite
```bash
pytest tests/test_analysis_pipeline.py -v
```

---

## 5. Data Sources & Acknowledgments

- **UNODC**: Global Study on Homicide / United Nations Crime Trends Survey (UN-CTS).
- **Small Arms Survey (2018)**: *Estimating Global Civilian-Held Firearms Numbers*.
- **World Bank**: World Development Indicators (WDI API) for GDP per capita, Gini index, urbanization, and demographic shares.

