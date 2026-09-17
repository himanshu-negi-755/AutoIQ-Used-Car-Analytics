"""Unit tests for the data-processing pipeline."""
import numpy as np
import pandas as pd
import pytest

from data_processing import (
    REQUIRED_COLUMNS, clean_data, extract_brand, handle_missing, parse_engine,
    parse_mileage, parse_power, parse_torque, remove_duplicates, remove_outliers, validate_schema,
)


# ---- parsers ---------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [("23.4 kmpl", 23.4), ("26.6 km/kg", 26.6), ("", np.nan), (None, np.nan)])
def test_parse_mileage(raw, expected):
    result = parse_mileage(raw)
    assert (np.isnan(result) and np.isnan(expected)) or result == expected


def test_parse_engine():
    assert parse_engine("1248 CC") == 1248.0
    assert np.isnan(parse_engine(np.nan))


@pytest.mark.parametrize("raw, expected", [("74 bhp", 74.0), ("88.5 bhp", 88.5), (" bhp", np.nan), ("0", np.nan)])
def test_parse_power(raw, expected):
    result = parse_power(raw)
    assert (np.isnan(result) and np.isnan(expected)) or result == expected


@pytest.mark.parametrize("raw, nm, rpm", [
    ("190Nm@ 2000rpm", 190.0, 2000.0),
    ("22.4 kgm at 1750-2750rpm", 219.67, 2750.0),
    ("12.7@ 2,700(kgm@ rpm)", 124.54, 2700.0),
    ("51Nm@ 4000+/-500rpm", 51.0, 4000.0),
    ("380Nm(38.7kgm)@ 2500rpm", 380.0, 2500.0),   # first unit is Nm -> no conversion
    ("115@ 2,500(kgm@ rpm)", 115.0, 2500.0),      # >80 kgm impossible -> treated as Nm
])
def test_parse_torque(raw, nm, rpm):
    t, r = parse_torque(raw)
    assert t == pytest.approx(nm, abs=0.01)
    assert r == rpm


def test_parse_torque_invalid_rpm_becomes_nan():
    t, r = parse_torque("190@ 21,800(kgm@ rpm)")
    assert t == 190.0 and np.isnan(r)


def test_parse_torque_missing():
    t, r = parse_torque(np.nan)
    assert np.isnan(t) and np.isnan(r)


@pytest.mark.parametrize("name, brand", [
    ("Maruti Swift Dzire VDI", "Maruti"), ("Land Rover Freelander 2 TD4 SE", "Land Rover"),
    ("Mercedes-Benz E-Class E250", "Mercedes-Benz"), ("", "Unknown"), (None, "Unknown"),
])
def test_extract_brand(name, brand):
    assert extract_brand(name) == brand


# ---- pipeline steps --------------------------------------------------------

@pytest.fixture
def sample_raw() -> pd.DataFrame:
    rows = [
        ["Maruti Swift VDI", 2014, 450000, 145500, "Diesel", "Individual", "Manual", "First Owner", "23.4 kmpl", "1248 CC", "74 bhp", "190Nm@ 2000rpm", 5],
        ["Maruti Swift VDI", 2014, 450000, 145500, "Diesel", "Individual", "Manual", "First Owner", "23.4 kmpl", "1248 CC", "74 bhp", "190Nm@ 2000rpm", 5],  # duplicate
        ["Maruti Alto LXi", 2012, 200000, 60000, "Petrol", "Individual", "Manual", "Second Owner", None, None, None, None, None],  # missing specs
        ["Hyundai i20 Asta", 2016, 550000, 40000, "Petrol", "Dealer", "Manual", "First Owner", "18.6 kmpl", "1197 CC", "81.8 bhp", "114.7Nm@ 4000rpm", 5],
        ["Toyota Fortuner 4x4", 2018, 3000000, 30000, "Diesel", "Dealer", "Automatic", "First Owner", "14.2 kmpl", "2755 CC", "174.5 bhp", "420Nm@ 1600-2400rpm", 7],
        ["Audi A8 L", 2019, 99999999, 0, "Petrol", "Dealer", "Automatic", "First Owner", "10 kmpl", "2995 CC", "335 bhp", "500Nm@ 1370-4500rpm", 5],  # outlier + 0 km
    ]
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def test_validate_schema_raises_on_missing_column(sample_raw):
    with pytest.raises(ValueError, match="missing required columns"):
        validate_schema(sample_raw.drop(columns=["torque"]))


def test_remove_duplicates(sample_raw):
    assert len(remove_duplicates(sample_raw)) == len(sample_raw) - 1


def test_handle_missing_uses_brand_median(sample_raw):
    from data_processing import engineer_features
    df = handle_missing(engineer_features(sample_raw))
    alto = df[df["name"] == "Maruti Alto LXi"].iloc[0]
    assert alto["engine_cc"] == 1248.0          # Maruti median
    assert alto["max_power_bhp"] == 74.0
    assert df[["mileage_kmpl", "engine_cc", "max_power_bhp", "torque_nm", "seats"]].isna().sum().sum() == 0


def test_remove_outliers_drops_zero_km(sample_raw):
    from data_processing import engineer_features
    df = remove_outliers(engineer_features(sample_raw), price_q=1.0, km_q=1.0)
    assert (df["km_driven"] > 0).all()
    assert "Audi A8 L" not in df["name"].values


def test_clean_data_end_to_end(sample_raw):
    df = clean_data(sample_raw)
    assert "brand" in df and "car_age" in df and "price_lakh" in df
    assert df["car_age"].min() >= 1
    assert df.isna()[["mileage_kmpl", "engine_cc", "max_power_bhp", "torque_nm", "seats"]].sum().sum() == 0
    assert len(df) < len(sample_raw)
