import numpy as np
import pandas as pd
import pytest

ALL_GRADES = [
    "A1", "A2", "A3", "A4", "A5",
    "B1", "B2", "B3", "B4", "B5",
    "C1", "C2", "C3", "C4", "C5",
    "D1", "D2", "D3", "D4", "D5",
    "E1", "E2", "E3", "E4", "E5",
    "F1", "F2", "F3", "F4", "F5",
    "G1", "G2", "G3", "G4", "G5",
]


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Small DataFrame that mimics the structure of the real loan CSV."""
    rng = np.random.default_rng(42)
    n = 30

    return pd.DataFrame({
        "loan_status":        ["Fully Paid"] * 20 + ["Charged Off"] * 8 + ["Current"] * 2,
        "application_type":   ["INDIVIDUAL"] * 28 + ["JOINT"] * 2,
        "loan_amnt":          rng.integers(1000, 40000, n).astype(float),
        "term":               [" 36 months", " 60 months"] * (n // 2),
        "int_rate":           rng.uniform(5, 30, n),
        "installment":        rng.uniform(50, 1500, n),
        "sub_grade":          (["A1", "B2", "C3", "D4", "E5", "F1", "G5"] * 5)[:n],
        "home_ownership":     (["RENT", "OWN", "MORTGAGE"] * 10)[:n],
        "annual_inc":         rng.integers(20000, 200000, n).astype(float),
        "verification_status":(["Not Verified", "Verified", "Source Verified"] * 10)[:n],
        "purpose":            (["debt_consolidation", "credit_card", "home_improvement"] * 10)[:n],
        "dti":                rng.uniform(0, 40, n),
        "delinq_2yrs":        rng.integers(0, 5, n).astype(float),
        "inq_last_6mths":     rng.integers(0, 10, n).astype(float),
        "open_acc":           rng.integers(1, 30, n).astype(float),
        "pub_rec":            rng.integers(0, 3, n).astype(float),
        "revol_bal":          rng.integers(0, 50000, n).astype(float),
        "revol_util":         rng.uniform(0, 100, n),
        "total_acc":          rng.integers(5, 50, n).astype(float),
        "initial_list_status":(["f", "w"] * (n // 2))[:n],
        "collections_12_mths_ex_med": rng.integers(0, 2, n).astype(float),
        "chargeoff_within_12_mths":   rng.integers(0, 2, n).astype(float),
        "pub_rec_bankruptcies":       rng.integers(0, 2, n).astype(float),
        "emp_length":         (["10+ years", "< 1 year", "5 years", "3 years", "2 years"] * 6)[:n],
        # columns that should get dropped
        "id":                 range(n),
        "member_id":          range(n),
        "grade":              (["A", "B", "C"] * 10)[:n],
        "last_pymnt_amnt":    rng.uniform(0, 1000, n),  # leakage column
    })


@pytest.fixture
def minimal_config() -> dict:
    return {
        "preprocessing": {
            "missing_threshold": 0.20,
            "leakage_columns": ["last_pymnt_amnt"],
            "id_columns": ["id", "member_id", "grade"],
            "sub_grade_order": ALL_GRADES,
        },
        "training": {
            "random_seed": 42,
            "n_splits": 3,
            "test_size": 0.2,
        },
        "optuna": {
            "n_trials": 3,
            "timeout": None,
        },
        "models": {
            "lightgbm": {"enabled": True},
            "logistic_regression": {"enabled": False},
            "random_forest": {"enabled": False},
            "xgboost": {"enabled": False},
            "hist_gradient_boosting": {"enabled": False},
        },
    }
