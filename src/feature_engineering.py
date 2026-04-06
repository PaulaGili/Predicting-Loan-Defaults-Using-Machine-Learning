
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

log = logging.getLogger("loan_default.features")


class RatioFeatureCreator(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        annual_inc = X.get("annual_inc", pd.Series(np.nan, index=X.index))
        monthly_inc = (annual_inc + 1) / 12  # divide first, then shift to avoid div-by-zero

        if "loan_amnt" in X.columns:
            X["loan_to_income"] = X["loan_amnt"] / (annual_inc + 1)

        if "installment" in X.columns:
            X["installment_pct_income"] = X["installment"] / monthly_inc

        if "loan_amnt" in X.columns and "int_rate" in X.columns and "term" in X.columns:
            X["total_interest_cost"] = (X["installment"] * X["term"]) - X["loan_amnt"]

        if "revol_bal" in X.columns and "annual_inc" in X.columns:
            X["revol_to_income"] = X["revol_bal"] / (annual_inc + 1)

        return X


class SubGradeEncoder(BaseEstimator, TransformerMixin):
    """Ordinal encode sub_grade: A1=0, A2=1, ... G5=34."""

    def __init__(self, sub_grade_order: list[str]) -> None:
        self.sub_grade_order = sub_grade_order

    def fit(self, X, y=None):
        self.mapping_ = {g: i for i, g in enumerate(self.sub_grade_order)}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if "sub_grade" in X.columns:
            X["sub_grade"] = X["sub_grade"].map(self.mapping_)
        return X


class VerificationStatusEncoder(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if "verification_status" in X.columns:
            X["verification_status"] = (
                X["verification_status"].astype(str).str.strip() != "Not Verified"
            ).astype(int)
        return X


class TermEncoder(BaseEstimator, TransformerMixin):
    """Extract integer months from strings like ' 36 months'."""

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if "term" in X.columns:
            X["term"] = X["term"].astype(str).str.extract(r"(\d+)", expand=False).astype(float)
        return X


class InitialListStatusEncoder(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if "initial_list_status" in X.columns:
            X["initial_list_status"] = (
                X["initial_list_status"].astype(str).str.strip() == "w"
            ).astype(int)
        return X


class EmpLengthEncoder(BaseEstimator, TransformerMixin):
    """Convert employment length strings to float years.
    '10+ years' -> 10, '< 1 year' -> 0, 'n/a' -> training median
    """

    def fit(self, X, y=None):
        if "emp_length" in X.columns:
            self.median_ = self._parse(X["emp_length"]).median()
        else:
            self.median_ = 5.0
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if "emp_length" not in X.columns:
            return X
        X["emp_length"] = self._parse(X["emp_length"]).fillna(self.median_)
        return X

    @staticmethod
    def _parse(s: pd.Series) -> pd.Series:
        return (
            s.astype(str).str.strip()
             .str.replace("10+ years", "10", regex=False)
             .str.replace("< 1 year", "0", regex=False)
             .str.extract(r"(\d+)", expand=False)
             .astype(float)
        )


class CreditBehaviorFeatures(BaseEstimator, TransformerMixin):
    """Interaction features that capture credit risk patterns.
    Tree models can find these on their own but having them explicit saves depth.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        if "delinq_2yrs" in X.columns and "open_acc" in X.columns:
            X["delinq_rate"] = X["delinq_2yrs"] / (X["open_acc"] + 1)

        if "inq_last_6mths" in X.columns and "total_acc" in X.columns:
            X["inq_to_total_acc"] = X["inq_last_6mths"] / (X["total_acc"] + 1)

        if "revol_util" in X.columns and "sub_grade" in X.columns:
            # sub_grade is already numeric at this point
            X["util_x_subgrade"] = X["revol_util"] * X["sub_grade"]

        if "dti" in X.columns and "int_rate" in X.columns:
            X["dti_x_rate"] = X["dti"] * X["int_rate"]

        return X


class OneHotEncoderWrapper(BaseEstimator, TransformerMixin):

    def __init__(self, drop: str = "first", handle_unknown: str = "ignore") -> None:
        self.drop = drop
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        self.cat_cols_ = X.select_dtypes(include=["object", "category"]).columns.tolist()
        if self.cat_cols_:
            self.ohe_ = OneHotEncoder(
                drop=self.drop,
                handle_unknown=self.handle_unknown,
                sparse_output=False,
                dtype=int,
            )
            self.ohe_.fit(X[self.cat_cols_])
            self.ohe_names_ = list(self.ohe_.get_feature_names_out(self.cat_cols_))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.cat_cols_:
            encoded = pd.DataFrame(
                self.ohe_.transform(X[self.cat_cols_]),
                columns=self.ohe_names_,
                index=X.index,
            )
            X = pd.concat([X.drop(columns=self.cat_cols_), encoded], axis=1)
        return X


def build_feature_pipeline(config: dict[str, Any]) -> Pipeline:
    sub_grade_order = config["preprocessing"]["sub_grade_order"]

    pipeline = Pipeline([
        ("sub_grade",       SubGradeEncoder(sub_grade_order)),
        ("verification",    VerificationStatusEncoder()),
        ("term",            TermEncoder()),
        ("list_status",     InitialListStatusEncoder()),
        ("emp_length",      EmpLengthEncoder()),
        ("ratios",          RatioFeatureCreator()),
        ("credit_behavior", CreditBehaviorFeatures()),
        ("ohe",             OneHotEncoderWrapper(drop="first", handle_unknown="ignore")),
    ])

    log.info("Feature pipeline built")
    return pipeline
