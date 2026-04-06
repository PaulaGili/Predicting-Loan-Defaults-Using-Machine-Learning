import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    CreditBehaviorFeatures,
    EmpLengthEncoder,
    RatioFeatureCreator,
    SubGradeEncoder,
    TermEncoder,
    VerificationStatusEncoder,
    build_feature_pipeline,
)


def _make_df(**kwargs) -> pd.DataFrame:
    return pd.DataFrame({k: [v] for k, v in kwargs.items()})


def test_ratio_no_division_by_zero():
    df = _make_df(loan_amnt=5000.0, annual_inc=0.0, installment=100.0,
                  int_rate=10.0, term=36.0, revol_bal=1000.0)
    result = RatioFeatureCreator().fit_transform(df)
    assert result.isna().sum().sum() == 0
    assert not np.isinf(result.values).any()


def test_ratio_installment_pct_is_monthly_scale():
    df = _make_df(loan_amnt=10000.0, annual_inc=60000.0, installment=500.0,
                  int_rate=10.0, term=36.0, revol_bal=0.0)
    result = RatioFeatureCreator().fit_transform(df)
    assert result["installment_pct_income"].iloc[0] < 1.0


def test_sub_grade_encoder_ordering(minimal_config):
    grades = minimal_config["preprocessing"]["sub_grade_order"]
    enc = SubGradeEncoder(grades).fit(pd.DataFrame({"sub_grade": grades}))
    result = enc.transform(pd.DataFrame({"sub_grade": ["A1", "B3", "G5"]}))
    assert result["sub_grade"].tolist() == [0, 7, 34]


def test_sub_grade_encoder_unknown_becomes_nan(minimal_config):
    grades = minimal_config["preprocessing"]["sub_grade_order"]
    enc = SubGradeEncoder(grades).fit(pd.DataFrame({"sub_grade": grades}))
    result = enc.transform(pd.DataFrame({"sub_grade": ["Z9"]}))
    assert result["sub_grade"].isna().all()


def test_term_encoder_extracts_integer():
    df = pd.DataFrame({"term": [" 36 months", " 60 months"]})
    result = TermEncoder().fit_transform(df)
    assert result["term"].tolist() == [36.0, 60.0]


def test_verification_encoder_binary():
    df = pd.DataFrame({"verification_status": ["Not Verified", "Verified", "Source Verified"]})
    result = VerificationStatusEncoder().fit_transform(df)
    assert result["verification_status"].tolist() == [0, 1, 1]


def test_emp_length_encoder():
    df = pd.DataFrame({"emp_length": ["10+ years", "< 1 year", "5 years", "3 years"]})
    result = EmpLengthEncoder().fit(df).transform(df)
    assert result["emp_length"].tolist() == [10.0, 0.0, 5.0, 3.0]


def test_emp_length_encoder_handles_na():
    df_fit = pd.DataFrame({"emp_length": ["5 years", "3 years", "2 years"]})
    enc = EmpLengthEncoder().fit(df_fit)
    df_test = pd.DataFrame({"emp_length": ["n/a", "< 1 year"]})
    result = enc.transform(df_test)
    assert not result["emp_length"].isna().any()
    assert result["emp_length"].iloc[1] == 0.0


def test_credit_behavior_creates_columns():
    df = pd.DataFrame({
        "delinq_2yrs": [2.0], "open_acc": [10.0],
        "inq_last_6mths": [3.0], "total_acc": [20.0],
        "revol_util": [75.0], "sub_grade": [10.0],
        "dti": [20.0], "int_rate": [15.0],
    })
    result = CreditBehaviorFeatures().fit_transform(df)
    for col in ["delinq_rate", "inq_to_total_acc", "util_x_subgrade", "dti_x_rate"]:
        assert col in result.columns


def test_pipeline_all_numeric(raw_df, minimal_config):
    from src.data_preprocessing import (
        create_target, drop_explicit_columns, filter_individual_applications,
        filter_loan_status, remove_residual_nulls,
    )
    df = filter_individual_applications(raw_df)
    df = filter_loan_status(df)
    df = drop_explicit_columns(
        df,
        leakage_cols=minimal_config["preprocessing"]["leakage_columns"],
        id_cols=minimal_config["preprocessing"]["id_columns"],
    )
    df = remove_residual_nulls(df)
    X, _ = create_target(df)

    result = build_feature_pipeline(minimal_config).fit_transform(X)
    assert result.select_dtypes(include="object").empty
    assert result.isna().sum().sum() == 0
