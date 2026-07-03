"""
Production-ready data preprocessing utilities for the churn prediction project.

This module focuses on loading, inspecting, cleaning, encoding, and splitting the
IBM Telco Customer Churn dataset (or compatible schema).

No model training happens here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


@dataclass(frozen=True)
class DatasetInspection:
    """A structured inspection summary for a pandas DataFrame."""

    rows: int
    columns: int
    missing_values: Dict[str, int]
    duplicate_rows: int
    numeric_columns: List[str]
    categorical_columns: List[str]

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "rows": self.rows,
            "columns": self.columns,
            "missing_values": dict(self.missing_values),
            "duplicate_rows": self.duplicate_rows,
            "numeric_columns": list(self.numeric_columns),
            "categorical_columns": list(self.categorical_columns),
        }


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """
    Load a churn dataset CSV into a pandas DataFrame.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        A pandas DataFrame containing the loaded data.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If the CSV cannot be parsed into a DataFrame.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Place the file at "
            f"'data/Telco-Customer-Churn.csv' or pass the correct csv_path."
        )
    if not path.is_file():
        raise FileNotFoundError(f"Provided csv_path is not a file: '{path}'")

    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"Failed to read CSV at '{path}': {exc}") from exc


def inspect_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Inspect the dataset and return summary statistics for quick validation.

    The returned dictionary contains:
      - rows
      - columns
      - missing values (per column)
      - duplicate rows
      - numeric columns
      - categorical columns

    Args:
        df: Input DataFrame.

    Returns:
        A dictionary with inspection details.

    Raises:
        TypeError: If df is not a pandas DataFrame.
        ValueError: If df is empty.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("inspect_dataset expects a pandas DataFrame.")
    if df.empty:
        raise ValueError("Dataset is empty; cannot inspect.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [
        c for c in df.columns if c not in numeric_cols
    ]

    inspection = DatasetInspection(
        rows=int(df.shape[0]),
        columns=int(df.shape[1]),
        missing_values=df.isna().sum().astype(int).to_dict(),
        duplicate_rows=int(df.duplicated().sum()),
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
    )
    return inspection.as_dict()


def _strip_and_normalize_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip leading/trailing whitespace for all string-like columns and normalize blanks.

    Blank/whitespace-only strings are replaced with NaN.
    """
    out = df.copy()

    obj_cols = out.select_dtypes(include=["object", "string"]).columns
    if len(obj_cols) == 0:
        return out

    for col in obj_cols:
        # Ensure values are strings for stripping, while preserving NaN
        series = out[col]
        series = series.astype("string")
        series = series.str.strip()
        series = series.replace("", pd.NA)
        out[col] = series

    return out


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataset safely for downstream ML steps.

    Operations performed:
    - Remove duplicate rows.
    - Strip leading/trailing whitespace from string columns.
    - Replace blank strings with NaN.
    - Convert `TotalCharges` to numeric (coerce errors to NaN).
    - Handle missing values in a conservative, production-safe way.
    - Remove rows where the target (`Churn`) is missing.

    Notes:
    - This function does not perform scaling or feature selection.
    - It does not create synthetic data.

    Args:
        df: Raw input DataFrame.

    Returns:
        Cleaned DataFrame.

    Raises:
        TypeError: If df is not a pandas DataFrame.
        ValueError: If required columns are missing or result becomes empty.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("clean_dataset expects a pandas DataFrame.")
    if df.empty:
        raise ValueError("Dataset is empty; cannot clean.")

    if "Churn" not in df.columns:
        raise ValueError("Missing required target column: 'Churn'.")

    out = df.copy()

    # Remove duplicates early to prevent skew in missing-value summaries.
    out = out.drop_duplicates().reset_index(drop=True)

    # Strip strings and normalize blanks to NaN
    out = _strip_and_normalize_strings(out)

    # Convert TotalCharges to numeric if present (common issue: stored as string)
    if "TotalCharges" in out.columns:
        out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")

    # Remove rows where target is missing / blank
    out = out.dropna(subset=["Churn"]).reset_index(drop=True)
    if out.empty:
        raise ValueError("All rows removed after dropping missing target 'Churn'.")

    # Handle remaining missing values safely:
    # - numeric: fill with median (robust to outliers)
    # - categorical: fill with 'Unknown'
    numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = out.select_dtypes(include=["object", "string"]).columns.tolist()

    # Exclude target from feature filling to avoid accidental modifications
    if "Churn" in categorical_cols:
        categorical_cols.remove("Churn")

    for col in numeric_cols:
        if out[col].isna().any():
            median = out[col].median()
            # If column is entirely NaN, median is NaN; fill with 0.0 as fallback.
            if pd.isna(median):
                median = 0.0
            out[col] = out[col].fillna(median)

    for col in categorical_cols:
        if out[col].isna().any():
            out[col] = out[col].fillna("Unknown")

    return out


def encode_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Encode the target and categorical features.

    - Target column `Churn` is converted:
        Yes -> 1
        No  -> 0
    - Categorical feature columns are label-encoded using scikit-learn's LabelEncoder.
      Only columns with non-numeric dtypes are encoded.

    Args:
        df: Cleaned DataFrame containing a `Churn` column.

    Returns:
        A tuple of:
          - processed DataFrame
          - dict of fitted encoders keyed by column name

    Raises:
        TypeError: If df is not a pandas DataFrame.
        ValueError: If `Churn` is missing or contains unexpected values.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("encode_features expects a pandas DataFrame.")
    if df.empty:
        raise ValueError("Dataset is empty; cannot encode.")
    if "Churn" not in df.columns:
        raise ValueError("Missing required target column: 'Churn'.")

    out = df.copy()

    churn_series = out["Churn"].astype("string").str.strip()
    mapping: Mapping[str, int] = {"Yes": 1, "No": 0}

    unmapped = set(churn_series.dropna().unique()) - set(mapping.keys())
    if unmapped:
        raise ValueError(
            "Unexpected values found in 'Churn'. Expected only 'Yes'/'No'. "
            f"Found: {sorted(unmapped)}"
        )

    out["Churn"] = churn_series.map(mapping).astype("Int64")
    out = out.dropna(subset=["Churn"]).reset_index(drop=True)

    encoders: Dict[str, LabelEncoder] = {}

    categorical_cols = out.select_dtypes(include=["object", "string"]).columns.tolist()
    if "Churn" in categorical_cols:
        categorical_cols.remove("Churn")

    for col in categorical_cols:
        le = LabelEncoder()
        # Ensure label encoder sees strings consistently (including "Unknown")
        values = out[col].astype("string").fillna("Unknown")
        out[col] = le.fit_transform(values).astype(int)
        encoders[col] = le

    return out, encoders


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split a processed DataFrame into features (X) and target (y).

    Args:
        df: DataFrame containing encoded `Churn` column.

    Returns:
        X: DataFrame of features (all columns except `Churn`)
        y: Series of target labels (`Churn`)

    Raises:
        TypeError: If df is not a pandas DataFrame.
        ValueError: If `Churn` is missing or df is empty after validation.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("split_features_target expects a pandas DataFrame.")
    if df.empty:
        raise ValueError("Dataset is empty; cannot split features/target.")
    if "Churn" not in df.columns:
        raise ValueError("Missing required target column: 'Churn'.")

    y = df["Churn"]
    X = df.drop(columns=["Churn"])

    if X.empty:
        raise ValueError("No feature columns available after dropping target.")

    return X, y


def train_test_split_data(
    X: pd.DataFrame,
    y: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets with an 80/20 ratio.

    Uses `random_state=42` and `stratify=y` for reproducibility and consistent
    class distribution across splits.

    Args:
        X: Feature matrix.
        y: Target vector.

    Returns:
        X_train, X_test, y_train, y_test

    Raises:
        ValueError: If input shapes are incompatible or y is empty.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")
    if not isinstance(y, (pd.Series, pd.DataFrame)):
        raise TypeError("y must be a pandas Series.")
    if len(X) == 0:
        raise ValueError("X is empty; cannot split.")
    if len(y) == 0:
        raise ValueError("y is empty; cannot split.")
    if len(X) != len(y):
        raise ValueError(
            f"X and y must have the same number of rows. Got {len(X)} and {len(y)}."
        )

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )


