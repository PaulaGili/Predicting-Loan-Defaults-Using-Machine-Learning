import pandas as pd
import pytest

from src.data_preprocessing import (
    create_target,
    drop_explicit_columns,
    drop_high_missing_columns,
    filter_individual_applications,
    filter_loan_status,
    remove_residual_nulls,
)


def test_filter_keeps_individual_only(raw_df):
    result = filter_individual_applications(raw_df)
    assert len(result) == len(raw_df) - 2
    assert "application_type" not in result.columns


def test_filter_loan_status_keeps_valid_only(raw_df):
    result = filter_loan_status(raw_df)
    assert set(result["loan_status"].unique()) <= {"Fully Paid", "Charged Off"}
    assert len(result) == 28


def test_drop_high_missing_drops_column():
    df = pd.DataFrame({
        "good_col": [1.0, 2.0, 3.0, 4.0, 5.0],
        "bad_col":  [1.0, None, None, None, None],
    })
    result = drop_high_missing_columns(df, threshold=0.5)
    assert "good_col" in result.columns
    assert "bad_col" not in result.columns


def test_drop_high_missing_keeps_column():
    df = pd.DataFrame({"barely_ok": [1.0, None, 3.0, 4.0, 5.0]})
    result = drop_high_missing_columns(df, threshold=0.20)
    assert "barely_ok" in result.columns


def test_drop_explicit_columns_removes_leakage(raw_df):
    result = drop_explicit_columns(raw_df, leakage_cols=["last_pymnt_amnt"], id_cols=[])
    assert "last_pymnt_amnt" not in result.columns
    assert "loan_amnt" in result.columns


def test_create_target_values(raw_df):
    df = filter_loan_status(raw_df)
    X, y = create_target(df)
    assert "loan_status" not in X.columns
    assert y[df["loan_status"] == "Charged Off"].eq(1).all()
    assert y[df["loan_status"] == "Fully Paid"].eq(0).all()


def test_remove_residual_nulls_leaves_no_nans():
    df = pd.DataFrame({"a": [1.0, None, 3.0], "b": [4.0, 5.0, None]})
    result = remove_residual_nulls(df)
    assert result.isna().sum().sum() == 0
    assert len(result) == 1
