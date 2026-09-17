"""Tests for business-insight computations."""
import pandas as pd

import insights as ins
from data_processing import get_clean_data


def test_headline_kpis_keys_and_ranges():
    k = ins.headline_kpis(get_clean_data())
    assert k["listings"] > 5000
    assert 0 <= k["diesel_share"] <= 100 and 0 <= k["automatic_share"] <= 100


def test_brand_summary_respects_min_listings():
    s = ins.brand_summary(get_clean_data(), min_listings=100)
    assert (s["listings"] >= 100).all()


def test_auto_findings_small_sample_message():
    small = get_clean_data().head(10)
    assert "Not enough" in ins.auto_findings(small)[0]


def test_auto_findings_full_data():
    findings = ins.auto_findings(get_clean_data())
    assert len(findings) == 5 and all(isinstance(f, str) for f in findings)
