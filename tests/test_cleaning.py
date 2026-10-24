import pandas as pd
import pytest

from src.cleaning import clean_data
from src.features import add_features


def test_clean_data_encodes_values(sample_customer):
    cleaned = clean_data(pd.DataFrame([sample_customer]))
    row = cleaned.iloc[0]
    assert "customerID" not in cleaned.columns
    assert row["gender"] == 0
    assert row["Contract"] == 0
    assert row["Partner"] == 1
    assert row["InternetService_Fiber optic"] == 0
    assert row["InternetService_No"] == 0
    assert row["PaymentMethod_Electronic check"] == 1
    assert not cleaned.isna().any().any()


def test_zero_tenure_blank_total_charges_becomes_zero(sample_customer):
    sample_customer["tenure"] = 0
    sample_customer["TotalCharges"] = " "
    cleaned = clean_data(pd.DataFrame([sample_customer]))
    assert cleaned.loc[0, "TotalCharges"] == 0.0


def test_missing_total_charges_is_estimated(sample_customer):
    sample_customer["TotalCharges"] = None
    cleaned = clean_data(pd.DataFrame([sample_customer]))
    assert cleaned.loc[0, "TotalCharges"] == pytest.approx(70.5 * 12)


def test_target_is_encoded_when_present(sample_customer):
    sample_customer["Churn"] = "Yes"
    cleaned = clean_data(pd.DataFrame([sample_customer]))
    assert cleaned.loc[0, "Churn"] == 1


def test_missing_required_column_raises(sample_customer):
    del sample_customer["Contract"]
    with pytest.raises(ValueError):
        clean_data(pd.DataFrame([sample_customer]))


def test_unknown_category_raises(sample_customer):
    sample_customer["Partner"] = "Maybe"
    with pytest.raises(ValueError):
        clean_data(pd.DataFrame([sample_customer]))


def test_add_features(sample_customer):
    featured = add_features(clean_data(pd.DataFrame([sample_customer])))
    row = featured.iloc[0]
    assert row["services_count"] == 2
    assert row["tenure_group"] == "0-12"
    assert row["avg_monthly_charge_per_tenure"] == pytest.approx(850.5 / 12)
