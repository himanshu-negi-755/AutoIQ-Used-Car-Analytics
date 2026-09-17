"""
AutoIQ - Visualization Module
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PALETTE = px.colors.qualitative.Safe
TEMPLATE = "plotly_white"


def _style(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(template=TEMPLATE, height=height, margin=dict(l=20, r=20, t=50, b=20),
                      legend_title_text="")
    return fig


# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #

def price_distribution(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(df, x="price_lakh", nbins=60, color_discrete_sequence=[PALETTE[0]],
                       labels={"price_lakh": "Selling price (₹ lakh)"},
                       title="Distribution of listing prices")
    fig.add_vline(x=df["price_lakh"].median(), line_dash="dash", line_color="firebrick",
                  annotation_text=f"Median ₹{df['price_lakh'].median():.1f} L", annotation_position="top right")
    return _style(fig)


def top_brands(df: pd.DataFrame, n: int = 12) -> go.Figure:
    counts = df["brand"].value_counts().head(n).reset_index()
    counts.columns = ["brand", "listings"]
    fig = px.bar(counts, x="listings", y="brand", orientation="h", color="listings",
                 color_continuous_scale="Blues", title=f"Top {n} brands by number of listings",
                 labels={"listings": "Listings", "brand": ""})
    fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
    return _style(fig)


def fuel_transmission_share(df: pd.DataFrame) -> go.Figure:
    fig = px.sunburst(df, path=["fuel", "transmission"], title="Market mix: fuel type → transmission",
                      color_discrete_sequence=PALETTE)
    return _style(fig)


# --------------------------------------------------------------------------- #
# Market trends
# --------------------------------------------------------------------------- #

def price_by_year(df: pd.DataFrame) -> go.Figure:
    grp = df.groupby("year")["price_lakh"].agg(["median", "count"]).reset_index()
    grp = grp[grp["count"] >= 10]
    fig = px.line(grp, x="year", y="median", markers=True, title="Median price by manufacture year",
                  labels={"median": "Median price (₹ lakh)", "year": "Year"},
                  hover_data={"count": True}, color_discrete_sequence=[PALETTE[1]])
    return _style(fig)


def depreciation_curve(df: pd.DataFrame, brands: list[str] | None = None) -> go.Figure:
    data = df if not brands else df[df["brand"].isin(brands)]
    grp = data.groupby(["brand", "car_age"])["price_lakh"].median().reset_index()
    counts = data.groupby(["brand", "car_age"]).size().reset_index(name="n")
    grp = grp.merge(counts).query("n >= 5 and car_age <= 15")
    fig = px.line(grp, x="car_age", y="price_lakh", color="brand", markers=True,
                  title="Depreciation curve: median price vs car age",
                  labels={"car_age": "Car age (years)", "price_lakh": "Median price (₹ lakh)"},
                  color_discrete_sequence=PALETTE)
    return _style(fig)


def brand_price_box(df: pd.DataFrame, n: int = 12) -> go.Figure:
    top = df["brand"].value_counts().head(n).index
    data = df[df["brand"].isin(top)]
    order = data.groupby("brand")["price_lakh"].median().sort_values(ascending=False).index
    fig = px.box(data, x="brand", y="price_lakh", category_orders={"brand": list(order)},
                 title="Price spread by brand (top brands by volume)", color="brand",
                 labels={"price_lakh": "Price (₹ lakh)", "brand": ""}, color_discrete_sequence=PALETTE)
    fig.update_layout(showlegend=False)
    return _style(fig)


def km_vs_price(df: pd.DataFrame, sample: int = 3000) -> go.Figure:
    data = df.sample(min(sample, len(df)), random_state=1)
    fig = px.scatter(data, x="km_driven", y="price_lakh", color="fuel", opacity=0.55,
                     hover_data=["name", "year"], title="Odometer reading vs price",
                     labels={"km_driven": "Kilometres driven", "price_lakh": "Price (₹ lakh)"},
                     color_discrete_sequence=PALETTE)
    return _style(fig)


def seller_type_premium(df: pd.DataFrame) -> go.Figure:
    grp = df.groupby(["seller_type", "owner"])["price_lakh"].median().reset_index()
    fig = px.bar(grp, x="owner", y="price_lakh", color="seller_type", barmode="group",
                 title="Median price by ownership history and seller type",
                 labels={"price_lakh": "Median price (₹ lakh)", "owner": ""},
                 color_discrete_sequence=PALETTE)
    return _style(fig)


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    cols = ["selling_price", "car_age", "km_driven", "mileage_kmpl", "engine_cc", "max_power_bhp", "torque_nm", "seats"]
    corr = df[cols].corr().round(2)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    title="Correlation between numeric features and price")
    return _style(fig, height=480)


# --------------------------------------------------------------------------- #
# Model insights
# --------------------------------------------------------------------------- #

def actual_vs_predicted(actual: list[float], predicted: list[float]) -> go.Figure:
    df = pd.DataFrame({"actual": actual, "pred": predicted}) / 1e5
    lim = [0, max(df.max()) * 1.02]
    fig = px.scatter(df, x="actual", y="pred", opacity=0.5, color_discrete_sequence=[PALETTE[2]],
                     title="Hold-out test set: actual vs predicted price",
                     labels={"actual": "Actual (₹ lakh)", "pred": "Predicted (₹ lakh)"})
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="Perfect fit", line=dict(dash="dash", color="gray")))
    return _style(fig)


def residual_distribution(actual: list[float], predicted: list[float]) -> go.Figure:
    df = pd.DataFrame({"actual": actual, "pred": predicted})
    df["error_pct"] = (df["pred"] - df["actual"]) / df["actual"] * 100
    fig = px.histogram(df, x="error_pct", nbins=60, color_discrete_sequence=[PALETTE[3]],
                       title="Prediction error distribution (% of actual price)",
                       labels={"error_pct": "Error (%)"})
    fig.add_vline(x=0, line_color="black")
    fig.update_xaxes(range=[-80, 80])
    return _style(fig)


def feature_importance_chart(imp: pd.DataFrame) -> go.Figure:
    fig = px.bar(imp, x="importance", y="feature", orientation="h", color="importance",
                 color_continuous_scale="Viridis", title="What drives price? (feature importance)",
                 labels={"importance": "Relative importance", "feature": ""})
    fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
    return _style(fig)


def model_comparison_chart(results: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=results["Model"], y=results["MAE (₹)"] / 1e3, name="Test MAE (₹ thousand)",
                         marker_color=PALETTE[0]))
    fig.add_trace(go.Scatter(x=results["Model"], y=results["CV R² mean"], name="CV R²", yaxis="y2",
                             mode="lines+markers", marker=dict(size=10, color=PALETTE[1])))
    fig.update_layout(title="Model comparison: lower MAE and higher R² is better",
                      yaxis=dict(title="MAE (₹ thousand)"),
                      yaxis2=dict(title="R²", overlaying="y", side="right", range=[0.8, 1.0]))
    return _style(fig)
