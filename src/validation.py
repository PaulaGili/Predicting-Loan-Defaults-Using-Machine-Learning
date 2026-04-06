
import logging

import pandas as pd

log = logging.getLogger("loan_default.validation")


class ValidationError(ValueError):
    pass


# these are the columns expected after preprocessing, before feature engineering
REQUIRED_COLUMNS = [
    "loan_amnt", "term", "int_rate", "installment", "sub_grade",
    "home_ownership", "annual_inc", "verification_status", "purpose",
    "dti", "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "initial_list_status",
    "collections_12_mths_ex_med", "chargeoff_within_12_mths",
    "pub_rec_bankruptcies", "emp_length",
]

NUMERIC_COLUMNS = [
    "loan_amnt", "int_rate", "installment", "annual_inc", "dti",
    "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc",
    "collections_12_mths_ex_med", "chargeoff_within_12_mths", "pub_rec_bankruptcies",
]


def validate_schema(df: pd.DataFrame) -> list[str]:
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        return [f"Missing required columns: {sorted(missing)}"]
    return []


def validate_types(df: pd.DataFrame) -> list[str]:
    errors = []
    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            continue
        before_nans = df[col].isna().sum()
        after_nans = pd.to_numeric(df[col], errors="coerce").isna().sum()
        new_nans = after_nans - before_nans
        if new_nans > 0:
            errors.append(f"'{col}' has {new_nans} non-numeric values")
    return errors


def validate_ranges(df: pd.DataFrame) -> list[str]:
    warnings = []
    checks = {
        "loan_amnt":  lambda s: s > 0,
        "int_rate":   lambda s: s.between(0, 100),
        "dti":        lambda s: s >= 0,
        "annual_inc": lambda s: s >= 0,
        "revol_util": lambda s: s.between(0, 200),  # some edge cases go above 100
    }
    for col, check in checks.items():
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        bad = (~check(series)).sum()
        if bad > 0:
            warnings.append(f"'{col}' has {bad} out-of-range values")
    return warnings


def validate_input(df: pd.DataFrame, raise_on_error: bool = True) -> list[str]:
    """Run all checks. Returns list of issues found; raises if raise_on_error=True."""
    issues = validate_schema(df)
    issues += validate_types(df)
    issues += validate_ranges(df)
    for msg in issues:
        log.warning(msg)
    if raise_on_error and issues:
        raise ValidationError("\n".join(issues))
    return issues
