"""
AutoIQ - Data Processing Module
================================
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

RAW_DATA_PATH = Path(__file__).resolve().parent / "Car_details_v3.csv"
PROCESSED_DATA_PATH = Path(__file__).resolve().parent / "cars_clean.csv"


REFERENCE_YEAR = 2021

# 1 kgm (kilogram-metre) = 9.80665 Nm
KGM_TO_NM = 9.80665

REQUIRED_COLUMNS = [
    "name", "year", "selling_price", "km_driven", "fuel",
    "seller_type", "transmission", "owner", "mileage", "engine",
    "max_power", "torque", "seats",
]

OWNER_ORDER = {
    "Test Drive Car": 0,
    "First Owner": 1,
    "Second Owner": 2,
    "Third Owner": 3,
    "Fourth & Above Owner": 4,
}

NUMERIC_FEATURES = [
    "car_age", "km_driven", "mileage_kmpl", "engine_cc",
    "max_power_bhp", "torque_nm", "seats", "owner_rank",
]
CATEGORICAL_FEATURES = ["brand", "fuel", "seller_type", "transmission"]
TARGET = "selling_price"


# --------------------------------------------------------------------------- #
# Low-level parsers (pure functions)
# --------------------------------------------------------------------------- #

def _first_number(text: str | float | None) -> float:
    """Return the first numeric token in a string, or NaN."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return np.nan
    match = re.search(r"[-+]?\d*\.?\d+", str(text).replace(",", ""))
    return float(match.group()) if match else np.nan


def parse_mileage(value) -> float:
   
    return _first_number(value)


def parse_engine(value) -> float:
    """'1248 CC' -> 1248.0"""
    return _first_number(value)


def parse_power(value) -> float:
    """'74 bhp' -> 74.0 ; ' bhp' or '0' -> NaN (zero power is a data error)."""
    num = _first_number(value)
    return np.nan if (np.isnan(num) or num <= 0) else num


def parse_torque(value) -> tuple[float, float]:
   
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan, np.nan

    text = str(value).replace(",", "").lower()
    numbers = [float(n) for n in re.findall(r"\d*\.?\d+", text)]
    if not numbers:
        return np.nan, np.nan

    torque = numbers[0]
   
    nm_pos = text.find("nm")
    kgm_pos = text.find("kgm")
    first_unit_is_kgm = kgm_pos != -1 and (nm_pos == -1 or kgm_pos < nm_pos)
    if first_unit_is_kgm and torque <= 80:
        torque *= KGM_TO_NM

    
    rest = re.sub(r"\+/-\s*\d+", "", text)
    rest = re.sub(r"\(\s*\d*\.?\d+\s*kgm\)", "", rest)
    rest_numbers = [float(n) for n in re.findall(r"\d*\.?\d+", rest)][1:]
    rpm = max(rest_numbers) if rest_numbers else np.nan
    if not np.isnan(rpm) and not (500 <= rpm <= 9000):   # outside physical range
        rpm = np.nan
    return round(torque, 2), rpm


def extract_brand(name) -> str:
    """'Maruti Swift Dzire VDI' -> 'Maruti'. Land Rover is a two-word brand."""
    if not isinstance(name, str) or not name.strip():
        return "Unknown"
    tokens = name.strip().split()
    if tokens[0].lower() == "land" and len(tokens) > 1:
        return "Land Rover"
    return tokens[0].title()


# --------------------------------------------------------------------------- #
# Pipeline steps
# --------------------------------------------------------------------------- #

def load_raw(path: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV and validate that the expected schema is present."""
    df = pd.read_csv(path)
    validate_schema(df)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert free-text specs into numeric features and add derived columns."""
    out = df.copy()

    out["brand"] = out["name"].apply(extract_brand)
    out["mileage_kmpl"] = out["mileage"].apply(parse_mileage)
    out["engine_cc"] = out["engine"].apply(parse_engine)
    out["max_power_bhp"] = out["max_power"].apply(parse_power)

    torque_parsed = out["torque"].apply(parse_torque)
    out["torque_nm"] = torque_parsed.apply(lambda t: t[0])
    out["torque_rpm"] = torque_parsed.apply(lambda t: t[1])

    out["owner_rank"] = out["owner"].map(OWNER_ORDER).fillna(1).astype(int)
    out["car_age"] = REFERENCE_YEAR - out["year"]
    out["price_lakh"] = out["selling_price"] / 1e5
    out["km_per_year"] = out["km_driven"] / out["car_age"].clip(lower=1)
    return out


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Exact duplicate rows are re-posted listings; they would leak into the
    test set and inflate model scores, so they are dropped."""
    return df.drop_duplicates().reset_index(drop=True)


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
   
    out = df.copy()
    # A few small brands have all-NaN for a spec; np.median on an empty slice
    # emits a harmless RuntimeWarning. The global-median fallback on the next
    # line fills those, so the warning is suppressed to keep output clean.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for col in ["mileage_kmpl", "engine_cc", "max_power_bhp", "torque_nm", "torque_rpm", "seats"]:
            out[col] = out.groupby("brand")[col].transform(lambda s: s.fillna(s.median()))
            out[col] = out[col].fillna(out[col].median())
    return out


def remove_outliers(df: pd.DataFrame, price_q: float = 0.995, km_q: float = 0.995) -> pd.DataFrame:
   
    price_cap = df["selling_price"].quantile(price_q)
    km_cap = df["km_driven"].quantile(km_q)
    mask = (df["selling_price"] <= price_cap) & (df["km_driven"] <= km_cap) & (df["km_driven"] > 0)
    return df[mask].reset_index(drop=True)


def clean_data(df: pd.DataFrame, drop_outliers: bool = True) -> pd.DataFrame:
    
    out = engineer_features(df)
    out = remove_duplicates(out)
    out = handle_missing(out)
    if drop_outliers:
        out = remove_outliers(out)
    return out


def prepare_for_model(df: pd.DataFrame, drop_outliers: bool = True) -> pd.DataFrame:
    
    out = engineer_features(df)
    out = remove_duplicates(out)
    if drop_outliers:
        out = remove_outliers(out)
    return out


def get_clean_data(force_rebuild: bool = False) -> pd.DataFrame:
    """Return the processed dataset, building and caching it on first use."""
    if PROCESSED_DATA_PATH.exists() and not force_rebuild:
        return pd.read_csv(PROCESSED_DATA_PATH)
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean = clean_data(load_raw())
    clean.to_csv(PROCESSED_DATA_PATH, index=False)
    return clean


def data_quality_report(raw: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    """Summary table used in the dashboard's Data Quality page."""
    rows = [
        ("Raw listings", len(raw)),
        ("Exact duplicates removed", int(raw.duplicated().sum())),
        ("Rows with missing specs (raw)", int(raw[["mileage", "engine", "max_power", "seats"]].isna().any(axis=1).sum())),
        ("Outliers removed (price/km > 99.5th pct)", len(remove_duplicates(engineer_features(raw))) - len(clean)),
        ("Final clean listings", len(clean)),
        ("Unique brands", clean["brand"].nunique()),
        ("Year range", f"{int(clean['year'].min())} – {int(clean['year'].max())}"),
    ]
    table = pd.DataFrame(rows, columns=["Metric", "Value"])
    table["Value"] = table["Value"].astype(str)   # mixed int/str -> Arrow-safe
    return table


if __name__ == "__main__":
    df = get_clean_data(force_rebuild=True)
    print(df.shape)
    print(df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]].describe(include="all").T)
