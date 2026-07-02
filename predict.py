"""
Prediction utilities for the Customer Churn Prediction project.

This module loads trained artifacts and provides reusable functions for
preparing customer input, generating churn predictions, and returning
structured probability outputs.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "churn_model.pkl"
ENCODERS_PATH = MODEL_DIR / "label_encoders.pkl"

EXCLUDED_COLUMNS: Tuple[str, ...] = ("Churn", "customerID")

CustomerInput = Dict[str, Any]
PredictionResult = Dict[str, Union[str, float, int]]


@lru_cache(maxsize=1)
def load_model() -> Tuple[Any, Dict[str, LabelEncoder]]:
    """
    Load the trained churn model and fitted label encoders from disk.

    Returns:
        A tuple containing:
          - trained scikit-learn classifier
          - dictionary of label encoders keyed by column name

    Raises:
        FileNotFoundError: If model or encoder artifacts are missing.
        ValueError: If artifacts cannot be deserialized.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at '{MODEL_PATH}'. "
            "Run train_model.py before making predictions."
        )
    if not ENCODERS_PATH.exists():
        raise FileNotFoundError(
            f"Label encoders not found at '{ENCODERS_PATH}'. "
            "Run train_model.py before making predictions."
        )

    try:
        model = joblib.load(MODEL_PATH)
        encoders = joblib.load(ENCODERS_PATH)
    except Exception as exc:
        raise ValueError(f"Failed to load prediction artifacts: {exc}") from exc

    if not isinstance(encoders, dict):
        raise ValueError("Invalid encoder artifact format. Expected a dictionary.")

    return model, encoders


def _normalize_categorical_value(value: Any) -> str:
    """
    Normalize a categorical input value for encoding.

    Args:
        value: Raw categorical value from user input.

    Returns:
        Normalized string value.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "Unknown"
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else "Unknown"
    return str(value)


def _encode_categorical_value(value: Any, encoder: LabelEncoder) -> int:
    """
    Encode a categorical value, handling unseen categories gracefully.

    If the value was not seen during training, the function falls back to
    ``Unknown`` when available, otherwise to the first known class.

    Args:
        value: Raw categorical value.
        encoder: Fitted label encoder for the column.

    Returns:
        Encoded integer value.
    """
    normalized_value = _normalize_categorical_value(value)
    known_classes = set(encoder.classes_)

    if normalized_value in known_classes:
        return int(encoder.transform([normalized_value])[0])
    if "Unknown" in known_classes:
        return int(encoder.transform(["Unknown"])[0])

    return int(encoder.transform([encoder.classes_[0]])[0])


def prepare_input(
    input_data: CustomerInput,
    encoders: Dict[str, LabelEncoder],
) -> pd.DataFrame:
    """
    Convert raw customer input into a model-ready DataFrame.

    The function:
      - Accepts a dictionary of customer attributes
      - Converts it into a single-row DataFrame
      - Applies label encoding to categorical features
      - Converts numeric features to numeric dtype
      - Aligns column order with the trained model feature names

    Args:
        input_data: Dictionary containing customer feature values.
        encoders: Dictionary of fitted label encoders keyed by column name.

    Returns:
        Processed single-row DataFrame aligned to model feature order.

    Raises:
        TypeError: If ``input_data`` is not a dictionary.
        ValueError: If required model features are missing.
        FileNotFoundError: If model artifacts are unavailable for feature alignment.
    """
    if not isinstance(input_data, dict):
        raise TypeError("input_data must be a dictionary of customer attributes.")
    if not input_data:
        raise ValueError("input_data cannot be empty.")

    model, _ = load_model()
    if not hasattr(model, "feature_names_in_"):
        raise ValueError(
            "Loaded model does not expose feature_names_in_. "
            "Retrain the model before running predictions."
        )

    feature_names: List[str] = list(model.feature_names_in_)
    missing_features = [
        feature for feature in feature_names if feature not in input_data
    ]
    if missing_features:
        raise ValueError(
            "Missing required features for prediction: "
            f"{', '.join(missing_features)}"
        )

    filtered_input = {
        key: value
        for key, value in input_data.items()
        if key not in EXCLUDED_COLUMNS
    }
    processed_df = pd.DataFrame([filtered_input])

    for column in feature_names:
        if column in encoders:
            processed_df[column] = _encode_categorical_value(
                processed_df[column].iloc[0],
                encoders[column],
            )
        else:
            processed_df[column] = pd.to_numeric(
                processed_df[column],
                errors="coerce",
            )
            if processed_df[column].isna().any():
                raise ValueError(
                    f"Feature '{column}' must be numeric and cannot be null."
                )

    return processed_df[feature_names]


def predict_churn(
    model: Any,
    processed_input: pd.DataFrame,
) -> Dict[str, Union[int, float]]:
    """
    Generate a churn prediction and class probabilities.

    Args:
        model: Trained scikit-learn classifier.
        processed_input: Model-ready single-row feature DataFrame.

    Returns:
        Dictionary containing:
          - prediction: ``0`` for No Churn, ``1`` for Churn
          - probability_churn: probability of churn
          - probability_not_churn: probability of staying

    Raises:
        TypeError: If processed_input is not a DataFrame.
        ValueError: If prediction fails.
    """
    if not isinstance(processed_input, pd.DataFrame):
        raise TypeError("processed_input must be a pandas DataFrame.")
    if processed_input.empty:
        raise ValueError("processed_input cannot be empty.")

    try:
        prediction = int(model.predict(processed_input)[0])
        probabilities = model.predict_proba(processed_input)[0]
        class_to_index = {
            int(class_label): index
            for index, class_label in enumerate(model.classes_)
        }
    except Exception as exc:
        raise ValueError(f"Prediction failed: {exc}") from exc

    if 0 not in class_to_index or 1 not in class_to_index:
        raise ValueError(
            "Model classes must include 0 (No Churn) and 1 (Churn)."
        )

    return {
        "prediction": prediction,
        "probability_churn": float(probabilities[class_to_index[1]]),
        "probability_not_churn": float(probabilities[class_to_index[0]]),
    }


def predict_customer(input_data: CustomerInput) -> PredictionResult:
    """
    Run the full prediction workflow for a single customer.

    Steps:
      1. Load model and encoders
      2. Prepare and encode input features
      3. Predict churn outcome and probabilities

    Args:
        input_data: Dictionary containing customer feature values.

    Returns:
        Dictionary containing:
          - prediction: ``\"Churn\"`` or ``\"No Churn\"``
          - churn_probability: probability of churn
          - stay_probability: probability of not churning

    Raises:
        FileNotFoundError: If model artifacts are missing.
        ValueError: If input validation or prediction fails.
    """
    try:
        model, encoders = load_model()
        processed_input = prepare_input(input_data, encoders)
        result = predict_churn(model, processed_input)

        return {
            "prediction": "Churn" if result["prediction"] == 1 else "No Churn",
            "churn_probability": round(float(result["probability_churn"]), 4),
            "stay_probability": round(float(result["probability_not_churn"]), 4),
        }
    except (FileNotFoundError, TypeError, ValueError):
        raise
    except Exception as exc:
        raise ValueError(f"Customer prediction failed: {exc}") from exc


def main() -> PredictionResult:
    """
    Example entry point for manual prediction testing.

    Returns:
        Prediction result for a sample customer profile.

    Raises:
        FileNotFoundError: If trained artifacts are not available.
        ValueError: If prediction fails.
    """
    sample_customer: CustomerInput = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 844.2,
    }

    result = predict_customer(sample_customer)
    print("Prediction Result:")
    print(result)
    return result


if __name__ == "__main__":
    main()
