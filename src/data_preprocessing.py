
import logging
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger("loan_default.preprocessing")


def load_data(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Dataset not found at {p}\n"
            "Place loan.csv in the data/ folder before running the pipeline."
        )
    log.info(f"Loading {p}")
    df = pd.read_csv(p, low_memory=False)
    log.info(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} cols")
    return df


def filter_individual_applications(df: pd.DataFrame) -> pd.DataFrame:
    # JOINT loans have a different missing-data structure, keep them separate
    before = len(df)
    df = df[df["application_type"] != "JOINT"].copy()
    joint_cols = [c for c in df.columns if "joint" in c.lower()] + ["application_type"]
    df.drop(columns=[c for c in joint_cols if c in df.columns], inplace=True)
    log.info(f"Kept INDIVIDUAL only: {before:,} -> {len(df):,} rows")
    return df


def filter_loan_status(df: pd.DataFrame) -> pd.DataFrame:
    # only keep loans with known outcomes, skip anything still in progress
    valid = {"Fully Paid", "Charged Off"}
    before = len(df)
    df = df[df["loan_status"].isin(valid)].copy()
    log.info(f"Filtered to {valid}: {before:,} -> {len(df):,} rows")
    return df


def drop_high_missing_columns(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    fracs = df.isnull().mean()
    to_drop = fracs[fracs > threshold].index.tolist()
    log.info(f"Dropping {len(to_drop)} columns with >{threshold:.0%} missing: {to_drop}")
    return df.drop(columns=to_drop)


def drop_explicit_columns(df: pd.DataFrame, leakage_cols: list[str], id_cols: list[str]) -> pd.DataFrame:
    requested = set(leakage_cols + id_cols)
    existing = [c for c in requested if c in df.columns]
    log.info(f"Dropping {len(existing)} leakage/identifier columns")
    return df.drop(columns=existing)


def remove_residual_nulls(df: pd.DataFrame) -> pd.DataFrame:
    # at this point remaining nulls are <1%, row deletion is cleaner than imputation
    before = len(df)
    df = df.dropna()
    log.info(f"Dropped rows with nulls: {before:,} -> {len(df):,}")
    return df


def create_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = (df["loan_status"] == "Charged Off").astype(int)
    X = df.drop(columns=["loan_status"])
    pct = y.mean()
    log.info(f"Target: {y.sum():,} defaults ({pct:.1%}) / {(~y.astype(bool)).sum():,} paid ({1-pct:.1%})")
    return X, y


def preprocess(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    pp = config["preprocessing"]
    df = filter_individual_applications(df)
    df = filter_loan_status(df)
    df = drop_high_missing_columns(df, threshold=pp["missing_threshold"])
    df = drop_explicit_columns(
        df,
        leakage_cols=pp.get("leakage_columns", []),
        id_cols=pp.get("id_columns", []),
    )
    df = remove_residual_nulls(df)
    X, y = create_target(df)
    log.info(f"Preprocessing done — X: {X.shape}, y: {y.shape}")
    return X, y
