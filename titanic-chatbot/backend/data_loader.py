from __future__ import annotations

from pathlib import Path

import pandas as pd


EXPECTED_COLUMNS = {
    "passengerid",
    "survived",
    "pclass",
    "name",
    "sex",
    "age",
    "sibsp",
    "parch",
    "ticket",
    "fare",
    "cabin",
    "embarked",
}


def load_titanic_dataframe(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Titanic dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [col.strip().lower() for col in df.columns]

    missing_cols = EXPECTED_COLUMNS.difference(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {sorted(missing_cols)}")

    # Safe missing value handling for analytics.
    for numeric_col in ["age", "fare"]:
        df[numeric_col] = pd.to_numeric(df[numeric_col], errors="coerce")
        df[numeric_col] = df[numeric_col].fillna(df[numeric_col].median())

    for cat_col in ["embarked", "cabin", "sex"]:
        df[cat_col] = df[cat_col].fillna("Unknown")

    return df
