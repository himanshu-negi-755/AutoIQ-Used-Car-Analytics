"""
AutoIQ - Used Car Market Analytics Dashboard
============================================
Streamlit entry point. All heavy lifting (cleaning, modelling, charts,
insights) lives in separate modules; this file only wires UI to logic.

Run:  streamlit run app.py
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

import insights as ins
import visualizations as viz
from data_processing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    clean_data,
    data_quality_report,
    get_clean_data,
    load_raw,
    validate_schema,
)
from model import EXPERIMENTS_PATH, feature_importance, load_model, predict_price

st.set_page_config(page_title="AutoIQ · Used Car Market Analytics", page_icon="🚗", layout="wide",
                   menu_items={})

# Hide Streamlit's top-right toolbar (Deploy button + hamburger menu) and footer
# so the app looks like a finished product in screenshots and the video.
st.markdown(
    """
    <style>
      [data-testid="stToolbar"] {visibility: hidden; height: 0; position: fixed;}
      [data-testid="stDecoration"] {display: none;}
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
      header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner="Loading and cleaning listings…")
def _load_default() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = load_raw()
    return raw, get_clean_data()


@st.cache_data(show_spinner="Cleaning uploaded file…")
def _load_uploaded(file_bytes: bytes) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(io.BytesIO(file_bytes))
    validate_schema(raw)
    return raw, clean_data(raw)


@st.cache_resource(show_spinner="Loading price model…")
def _load_model():
    return load_model()


def _fmt_lakh(v: float) -> str:
    return f"₹{v:,.2f} L"


def _fmt_rs(v: float) -> str:
    return f"₹{v:,.0f}"


# --------------------------------------------------------------------------- #
# Sidebar: data source + global filters
# --------------------------------------------------------------------------- #

st.sidebar.title("🚗 AutoIQ")
st.sidebar.caption("Used Car Market Analytics")

st.sidebar.subheader("Data source")
uploaded = st.sidebar.file_uploader("Upload dealer CSV (same schema)", type="csv",
                                    help="Must contain the CarDekho columns: name, year, selling_price, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, torque, seats")
try:
    if uploaded is not None:
        raw_df, df_all = _load_uploaded(uploaded.getvalue())
        st.sidebar.success(f"Using uploaded file · {len(df_all):,} clean rows")
    else:
        raw_df, df_all = _load_default()
except ValueError as exc:
    st.sidebar.error(str(exc))
    raw_df, df_all = _load_default()

st.sidebar.subheader("Filters")
brands = sorted(df_all["brand"].unique())
sel_brands = st.sidebar.multiselect("Brand", brands, default=[])
yr_min, yr_max = int(df_all["year"].min()), int(df_all["year"].max())
sel_year = st.sidebar.slider("Manufacture year", yr_min, yr_max, (max(yr_min, 2008), yr_max))
sel_fuel = st.sidebar.multiselect("Fuel", sorted(df_all["fuel"].unique()), default=[])
sel_trans = st.sidebar.multiselect("Transmission", sorted(df_all["transmission"].unique()), default=[])
sel_seller = st.sidebar.multiselect("Seller type", sorted(df_all["seller_type"].unique()), default=[])
p_lo, p_hi = float(df_all["price_lakh"].min()), float(df_all["price_lakh"].max())
sel_price = st.sidebar.slider("Price range (₹ lakh)", p_lo, p_hi, (p_lo, p_hi))

mask = df_all["year"].between(*sel_year) & df_all["price_lakh"].between(*sel_price)
if sel_brands:
    mask &= df_all["brand"].isin(sel_brands)
if sel_fuel:
    mask &= df_all["fuel"].isin(sel_fuel)
if sel_trans:
    mask &= df_all["transmission"].isin(sel_trans)
if sel_seller:
    mask &= df_all["seller_type"].isin(sel_seller)
df = df_all[mask]

st.sidebar.markdown(f"**{len(df):,}** of {len(df_all):,} listings selected")
st.sidebar.download_button("⬇ Download filtered data (CSV)", df.to_csv(index=False).encode(),
                           "autoiq_filtered.csv", "text/csv")

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #

st.title("AutoIQ — Used Car Market Analytics Dashboard")
st.markdown(
    "Decision support for dealership pricing, procurement and inventory strategy, "
    "built on CarDekho listings. Use the sidebar to slice the market; every chart, "
    "table and insight updates dynamically as you filter. Data is historical CarDekho listings, not a live market feed."
)

if df.empty:
    st.warning("No listings match the current filters.")
    st.stop()

tab_overview, tab_trends, tab_segments, tab_predict, tab_model, tab_quality = st.tabs(
    ["📊 Overview", "📈 Market Trends", "🔍 Segment Explorer", "💰 Price Predictor", "🧠 Model Insights", "🧹 Data Quality"]
)

# --------------------------------------------------------------------------- #
# Tab 1: Overview
# --------------------------------------------------------------------------- #

with tab_overview:
    k = ins.headline_kpis(df)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Listings", f"{k['listings']:,}")
    c2.metric("Median price", _fmt_lakh(k["median_price_lakh"]))
    c3.metric("Median odometer", f"{k['median_km']:,.0f} km")
    c4.metric("Median age", f"{k['median_age']:.0f} yrs")
    c5.metric("First-owner share", f"{k['first_owner_share']:.0f}%")

    st.subheader("Key findings")
    for f in ins.auto_findings(df):
        st.markdown(f"- {f}")

    l, r = st.columns([3, 2])
    l.plotly_chart(viz.price_distribution(df), width="stretch")
    r.plotly_chart(viz.fuel_transmission_share(df), width="stretch")
    st.plotly_chart(viz.top_brands(df), width="stretch")

# --------------------------------------------------------------------------- #
# Tab 2: Market trends
# --------------------------------------------------------------------------- #

with tab_trends:
    l, r = st.columns(2)
    l.plotly_chart(viz.price_by_year(df), width="stretch")
    r.plotly_chart(viz.seller_type_premium(df), width="stretch")

    st.subheader("Depreciation by brand")
    top5 = df["brand"].value_counts().head(5).index.tolist()
    dep_brands = st.multiselect("Brands to compare", sorted(df["brand"].unique()), default=top5, key="dep")
    st.plotly_chart(viz.depreciation_curve(df, dep_brands), width="stretch")

    l, r = st.columns(2)
    l.plotly_chart(viz.brand_price_box(df), width="stretch")
    r.plotly_chart(viz.km_vs_price(df), width="stretch")
    st.plotly_chart(viz.correlation_heatmap(df), width="stretch")

    st.subheader("Value retention leaderboard")
    st.caption("Ratio of median listing price at 8–10 years to 3–5 years, per brand. "
               "A higher ratio means older cars of that brand are listed relatively closer to newer ones "
               "in this historical data. It is an observed association, not a measure of future resale value or inventory risk.")
    st.dataframe(ins.value_retention(df), width="stretch", hide_index=True)

# --------------------------------------------------------------------------- #
# Tab 3: Segment explorer
# --------------------------------------------------------------------------- #

with tab_segments:
    st.subheader("Brand summary")
    st.dataframe(
        ins.brand_summary(df).style.format({"median_price_lakh": "₹{:.2f} L", "median_km": "{:,.0f}",
                                             "avg_mileage": "{:.1f}", "automatic_pct": "{:.0f}%"}),
        width="stretch", hide_index=True,
    )
    st.subheader("Fuel × transmission segments")
    st.dataframe(ins.segment_table(df), width="stretch", hide_index=True)

    st.subheader("Listing browser")
    cols = ["name", "year", "price_lakh", "km_driven", "fuel", "transmission", "owner", "seller_type",
            "mileage_kmpl", "engine_cc", "max_power_bhp", "seats"]
    search = st.text_input("Search model name", placeholder="e.g. Swift, Creta, Fortuner")
    view = df[df["name"].str.contains(search, case=False, na=False)] if search else df
    st.dataframe(view[cols].sort_values("price_lakh", ascending=False).head(500), width="stretch", hide_index=True)

# --------------------------------------------------------------------------- #
# Tab 4: Price predictor
# --------------------------------------------------------------------------- #

with tab_predict:
    st.subheader("Historical-data price estimator")
    st.caption("Enter a car's details to get a price estimate based on historical CarDekho listings, "
               "shown with an indicative range. This is a data-based guide, not a formal valuation.")
    try:
        pipe, metrics = _load_model()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    with st.form("predict"):
        c1, c2, c3 = st.columns(3)
        brand = c1.selectbox("Brand", brands, index=brands.index("Maruti") if "Maruti" in brands else 0)
        year = c1.number_input("Manufacture year", 1995, 2021, 2015)
        km = c1.number_input("Kilometres driven", 0, 500000, 60000, step=5000)
        fuels = sorted(df_all["fuel"].unique())
        transes = sorted(df_all["transmission"].unique())
        sellers = sorted(df_all["seller_type"].unique())
        fuel = c2.selectbox("Fuel", fuels, index=fuels.index("Petrol") if "Petrol" in fuels else 0)
        trans = c2.selectbox("Transmission", transes, index=transes.index("Manual") if "Manual" in transes else 0)
        seller = c2.selectbox("Seller type", sellers, index=sellers.index("Dealer") if "Dealer" in sellers else 0)
        owner = c2.selectbox("Owner", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"])
        engine = c3.number_input("Engine (cc)", 600, 6000, 1200, step=50)
        power = c3.number_input("Max power (bhp)", 30.0, 600.0, 82.0, step=1.0)
        mileage = c3.number_input("Mileage (kmpl)", 5.0, 40.0, 20.0, step=0.5)
        torque = c3.number_input("Torque (Nm)", 40.0, 800.0, 115.0, step=5.0)
        seats = c3.selectbox("Seats", [2, 4, 5, 6, 7, 8, 9, 10], index=2)
        submitted = st.form_submit_button("Estimate price", type="primary")

    if submitted:
        from data_processing import OWNER_ORDER, REFERENCE_YEAR
        car = {
            "car_age": REFERENCE_YEAR - year, "km_driven": km, "mileage_kmpl": mileage, "engine_cc": engine,
            "max_power_bhp": power, "torque_nm": torque, "seats": seats, "owner_rank": OWNER_ORDER[owner],
            "brand": brand, "fuel": fuel, "seller_type": seller, "transmission": trans,
        }
        res = predict_price(pipe, car, mape=metrics["MAPE (%)"])
        a, b, c = st.columns(3)
        a.metric("Estimated price (historical)", _fmt_rs(res["predicted_price"]))
        b.metric("Lower bound", _fmt_rs(res["low"]))
        c.metric("Upper bound", _fmt_rs(res["high"]))
        st.caption(f"Indicative range = estimate ± model test-set MAPE ({metrics['MAPE (%)']:.1f}%). "
                   f"This is a rough error guide, not a statistical confidence interval. "
                   f"Model: {metrics['model']}, test R² = {metrics['R² (log)']:.3f}.")

        comps = df_all[(df_all["brand"] == brand) & (df_all["fuel"] == fuel)
                       & (df_all["year"].between(year - 1, year + 1))]
        if len(comps):
            st.markdown(f"**{len(comps)} comparable listings** ({brand}, {fuel}, {year}±1): "
                        f"median ₹{comps['selling_price'].median():,.0f}, "
                        f"range ₹{comps['selling_price'].quantile(.1):,.0f} – ₹{comps['selling_price'].quantile(.9):,.0f}")
            st.dataframe(comps[["name", "year", "selling_price", "km_driven", "transmission", "owner"]]
                         .sort_values("selling_price").head(15), width="stretch", hide_index=True)

# --------------------------------------------------------------------------- #
# Tab 5: Model insights
# --------------------------------------------------------------------------- #

with tab_model:
    try:
        pipe, metrics = _load_model()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    st.subheader(f"Production model: {metrics['model']}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test MAE", _fmt_rs(metrics["MAE (₹)"]))
    m2.metric("Test RMSE", _fmt_rs(metrics["RMSE (₹)"]))
    m3.metric("Test MAPE", f"{metrics['MAPE (%)']:.1f}%")
    m4.metric("Test R²", f"{metrics['R² (log)']:.3f}")
    st.caption(f"Trained on {metrics['n_train']:,} listings, evaluated on {metrics['n_test']:,} held-out listings.")

    l, r = st.columns(2)
    l.plotly_chart(viz.actual_vs_predicted(metrics["_test_actual"], metrics["_test_pred"]), width="stretch")
    r.plotly_chart(viz.residual_distribution(metrics["_test_actual"], metrics["_test_pred"]), width="stretch")
    st.plotly_chart(viz.feature_importance_chart(feature_importance(pipe)), width="stretch")

    st.subheader("Model selection experiment")
    if Path(EXPERIMENTS_PATH).exists():
        results = pd.read_csv(EXPERIMENTS_PATH)
        st.plotly_chart(viz.model_comparison_chart(results), width="stretch")
        st.dataframe(results.style.format({"MAE (₹)": "{:,.0f}", "RMSE (₹)": "{:,.0f}", "MAPE (%)": "{:.2f}",
                                           "CV R² mean": "{:.3f}", "CV R² std": "{:.3f}", "R² (log)": "{:.3f}",
                                           "R² (₹)": "{:.3f}", "Fit time (s)": "{:.2f}"}),
                     width="stretch", hide_index=True)
        st.markdown(
            "Four model families were compared with 5-fold cross-validation on the training split and a "
            "20% hold-out test. Tree ensembles clearly outperform linear models because price depends on "
            "interactions (e.g. age matters far more for luxury brands). Gradient Boosting was selected as "
            "it achieves the lowest error at a fit time suitable for periodic retraining."
        )
    else:
        st.info("Run `python scripts/run_experiments.py` to generate the comparison table.")

# --------------------------------------------------------------------------- #
# Tab 6: Data quality
# --------------------------------------------------------------------------- #

with tab_quality:
    st.subheader("Cleaning pipeline summary")
    st.dataframe(data_quality_report(raw_df, df_all), width="stretch", hide_index=True)
    st.markdown("""
**Steps applied (see `data_processing.py`):**
1. **Feature parsing** – free-text specs such as `23.4 kmpl`, `1248 CC`, `74 bhp`, `190Nm@ 2000rpm` are converted to numbers; torque in `kgm` is converted to Nm.
2. **Brand extraction** – first token of the model name (Land Rover handled as two words).
3. **De-duplication** – exact duplicate listings (re-posts) are dropped to prevent train/test leakage.
4. **Imputation** – missing specs filled with the brand median, then the global median.
5. **Outlier removal** – listings above the 99.5th percentile of price or odometer, or with 0 km, are excluded.
6. **Derived features** – car age (from 2021 reference), price in lakh, km per year, ownership rank.
""")
    st.subheader("Raw sample")
    st.dataframe(raw_df.head(20), width="stretch", hide_index=True)
    st.subheader("Clean sample")
    st.dataframe(df_all[["name", "brand", "year", "car_age", "selling_price", "km_driven"] + NUMERIC_FEATURES[2:6]].head(20),
                 width="stretch", hide_index=True)
