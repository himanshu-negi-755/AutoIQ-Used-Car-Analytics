"""
AutoIQ - Business Insights Module
"""

from __future__ import annotations

import pandas as pd


def headline_kpis(df: pd.DataFrame) -> dict:
    return {
        "listings": int(len(df)),
        "median_price_lakh": float(df["price_lakh"].median()),
        "mean_price_lakh": float(df["price_lakh"].mean()),
        "median_km": float(df["km_driven"].median()),
        "median_age": float(df["car_age"].median()),
        "brands": int(df["brand"].nunique()),
        "diesel_share": float((df["fuel"] == "Diesel").mean() * 100),
        "automatic_share": float((df["transmission"] == "Automatic").mean() * 100),
        "first_owner_share": float((df["owner"] == "First Owner").mean() * 100),
    }


def brand_summary(df: pd.DataFrame, min_listings: int = 20) -> pd.DataFrame:
    grp = (
        df.groupby("brand")
        .agg(listings=("name", "size"),
             median_price_lakh=("price_lakh", "median"),
             median_age=("car_age", "median"),
             median_km=("km_driven", "median"),
             avg_mileage=("mileage_kmpl", "mean"),
             automatic_pct=("transmission", lambda s: (s == "Automatic").mean() * 100))
        .query("listings >= @min_listings")
        .sort_values("listings", ascending=False)
        .round(2)
        .reset_index()
    )
    return grp


def value_retention(df: pd.DataFrame, min_listings: int = 30) -> pd.DataFrame:
    """Median listing price of 3-5 year-old cars vs 8-10 year-old cars per
    brand. A higher ratio means older cars of that brand are listed closer to
    the price of newer ones in this dataset. This is an observed association
    in historical listings, not a guarantee of future resale value or an
    inventory-risk measure (the data has no sales-duration information)."""
    young = df[df["car_age"].between(3, 5)].groupby("brand")["price_lakh"].median()
    old = df[df["car_age"].between(8, 10)].groupby("brand")["price_lakh"].median()
    counts = df["brand"].value_counts()
    out = pd.DataFrame({"price_3to5yr": young, "price_8to10yr": old, "listings": counts}).dropna()
    out = out[out["listings"] >= min_listings]
    out["retention_ratio"] = (out["price_8to10yr"] / out["price_3to5yr"]).round(2)
    return out.sort_values("retention_ratio", ascending=False).round(2).reset_index().rename(columns={"index": "brand"})


def segment_table(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["fuel", "transmission"])
        .agg(listings=("name", "size"), median_price_lakh=("price_lakh", "median"),
             median_km=("km_driven", "median"), avg_power_bhp=("max_power_bhp", "mean"))
        .round(2)
        .reset_index()
        .sort_values("listings", ascending=False)
    )


def auto_findings(df: pd.DataFrame) -> list[str]:
    """Plain-English findings for the Overview page, recomputed from the
    filtered data so they always match what the user is looking at."""
    if len(df) < 30:
        return ["Not enough listings in the current filter to generate reliable findings."]
    k = headline_kpis(df)
    findings = []

    top_brand = df["brand"].value_counts().idxmax()
    top_share = df["brand"].value_counts(normalize=True).max() * 100
    findings.append(f"**{top_brand}** is the largest brand in the data with {top_share:.0f}% of observed listings — the biggest single segment by supply.")

    fuel_med = df.groupby("fuel")["price_lakh"].median()
    if {"Diesel", "Petrol"} <= set(fuel_med.index):
        diff = (fuel_med["Diesel"] / fuel_med["Petrol"] - 1) * 100
        findings.append(f"Diesel cars list at a **{diff:+.0f}%** median premium over petrol (₹{fuel_med['Diesel']:.1f} L vs ₹{fuel_med['Petrol']:.1f} L).")

    tr_med = df.groupby("transmission")["price_lakh"].median()
    if {"Automatic", "Manual"} <= set(tr_med.index):
        ratio = tr_med["Automatic"] / tr_med["Manual"]
        findings.append(f"Automatics show **{ratio:.1f}×** the median listing price of manuals while making up only {k['automatic_share']:.0f}% of listings in the data.")

    own = df.groupby("owner")["price_lakh"].median()
    if {"First Owner", "Second Owner"} <= set(own.index):
        drop = (1 - own["Second Owner"] / own["First Owner"]) * 100
        findings.append(f"Second-owner cars list about **{drop:.0f}%** below first-owner cars on median price — ownership history is strongly associated with listing price.")

    corr = df[["selling_price", "car_age", "km_driven", "max_power_bhp"]].corr()["selling_price"]
    findings.append(f"Engine power is the strongest single price driver (r = {corr['max_power_bhp']:.2f}); age (r = {corr['car_age']:.2f}) matters more than odometer (r = {corr['km_driven']:.2f}).")
    return findings
