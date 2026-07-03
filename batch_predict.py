"""
Batch prediction utilities for the Customer Churn Prediction platform.

Handles CSV validation, multi-row inference, summary metrics, and chart data
without Streamlit dependencies.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from predict import load_model, predict_churn, prepare_input

OPTIONAL_BATCH_COLUMNS: Tuple[str, ...] = ("customerID", "Churn")


def map_risk_level(churn_probability: float) -> str:
    """
    Map churn probability to a business risk level label.

    Mirrors the risk meter thresholds used in the Streamlit application.

    Args:
        churn_probability: Model-predicted churn probability (0.0 to 1.0).

    Returns:
        Risk level label such as ``\"High Risk\"``.
    """
    probability = max(0.0, min(float(churn_probability), 1.0))
    percentage = probability * 100

    if percentage <= 20:
        return "Very Low Risk"
    if percentage <= 40:
        return "Low Risk"
    if percentage <= 60:
        return "Moderate Risk"
    if percentage <= 80:
        return "High Risk"
    return "Critical Risk"


def get_required_batch_columns() -> List[str]:
    """
    Return feature column names required for batch prediction.

    Returns:
        List of required column names from the trained model.

    Raises:
        FileNotFoundError: If model artifacts are missing.
        ValueError: If the model does not expose feature names.
    """
    model, _ = load_model()
    if not hasattr(model, "feature_names_in_"):
        raise ValueError(
            "Loaded model does not expose feature_names_in_. "
            "Retrain the model before running batch predictions."
        )
    return list(model.feature_names_in_)


def validate_batch_dataset(dataframe: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate that an uploaded batch dataset contains required model columns.

    Args:
        dataframe: Uploaded customer dataset.

    Returns:
        Dictionary with ``valid`` flag, required columns, missing columns,
        and optional human-readable error message.
    """
    if dataframe.empty:
        return {
            "valid": False,
            "required_columns": [],
            "missing_columns": [],
            "message": "The uploaded file contains no records.",
        }

    normalized_columns = [str(column).strip() for column in dataframe.columns]
    working_df = dataframe.copy()
    working_df.columns = normalized_columns

    try:
        required_columns = get_required_batch_columns()
    except (FileNotFoundError, ValueError) as exc:
        return {
            "valid": False,
            "required_columns": [],
            "missing_columns": [],
            "message": str(exc),
        }

    missing_columns = [
        column for column in required_columns if column not in working_df.columns
    ]

    if missing_columns:
        return {
            "valid": False,
            "required_columns": required_columns,
            "missing_columns": missing_columns,
            "message": (
                "The uploaded CSV is missing required columns: "
                f"{', '.join(missing_columns)}."
            ),
        }

    return {
        "valid": True,
        "required_columns": required_columns,
        "missing_columns": [],
        "message": None,
        "normalized_dataframe": working_df,
    }


def _normalize_batch_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize batch input values for consistent model preprocessing.

    Args:
        dataframe: Validated batch dataset with required columns.

    Returns:
        Copy of the dataframe with normalized numeric fields.
    """
    normalized = dataframe.copy()

    if "TotalCharges" in normalized.columns:
        normalized["TotalCharges"] = pd.to_numeric(
            normalized["TotalCharges"],
            errors="coerce",
        )
        if {"tenure", "MonthlyCharges"}.issubset(normalized.columns):
            fallback = pd.to_numeric(normalized["tenure"], errors="coerce") * pd.to_numeric(
                normalized["MonthlyCharges"],
                errors="coerce",
            )
            normalized["TotalCharges"] = normalized["TotalCharges"].fillna(fallback)
        normalized["TotalCharges"] = normalized["TotalCharges"].fillna(0)

    if "MonthlyCharges" in normalized.columns:
        normalized["MonthlyCharges"] = pd.to_numeric(
            normalized["MonthlyCharges"],
            errors="coerce",
        )

    if "tenure" in normalized.columns:
        normalized["tenure"] = pd.to_numeric(normalized["tenure"], errors="coerce")

    if "SeniorCitizen" in normalized.columns:
        normalized["SeniorCitizen"] = pd.to_numeric(
            normalized["SeniorCitizen"],
            errors="coerce",
        ).fillna(0).astype(int)

    return normalized


def predict_batch(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Predict churn outcomes for every row in a validated batch dataset.

    Args:
        dataframe: Batch dataset containing all required model feature columns.

    Returns:
        DataFrame with prediction outputs for each customer row.

    Raises:
        ValueError: If validation fails or a row cannot be processed.
        FileNotFoundError: If model artifacts are missing.
    """
    validation = validate_batch_dataset(dataframe)
    if not validation["valid"]:
        raise ValueError(validation.get("message") or "Invalid batch dataset.")

    working_df = _normalize_batch_values(validation["normalized_dataframe"])
    model, encoders = load_model()

    output_rows: List[Dict[str, Any]] = []
    for row_index, row in working_df.iterrows():
        customer_data = row.to_dict()
        processed_input = prepare_input(customer_data, encoders)
        result = predict_churn(model, processed_input)

        churn_probability = round(float(result["probability_churn"]), 4)
        stay_probability = round(float(result["probability_not_churn"]), 4)
        prediction_label = "Churn" if int(result["prediction"]) == 1 else "No Churn"

        output_row: Dict[str, Any] = {
            "Prediction": prediction_label,
            "Churn Probability": churn_probability,
            "Stay Probability": stay_probability,
            "Probability": churn_probability,
            "Risk Level": map_risk_level(churn_probability),
        }

        if "customerID" in working_df.columns:
            output_row["customerID"] = customer_data.get("customerID")

        output_rows.append(output_row)

    results = pd.DataFrame(output_rows)

    if "customerID" in results.columns:
        column_order = [
            "customerID",
            "Prediction",
            "Probability",
            "Churn Probability",
            "Stay Probability",
            "Risk Level",
        ]
        existing_columns = [column for column in column_order if column in results.columns]
        remaining_columns = [
            column for column in results.columns if column not in existing_columns
        ]
        results = results[existing_columns + remaining_columns]

    return results


def build_batch_summary(results: pd.DataFrame) -> Dict[str, Any]:
    """
    Build KPI summary metrics from batch prediction results.

    Args:
        results: Output dataframe from ``predict_batch``.

    Returns:
        Dictionary with total, churn, safe, and average probability metrics.
    """
    if results.empty:
        return {
            "total_customers": 0,
            "predicted_churn": 0,
            "predicted_safe": 0,
            "average_churn_probability": 0.0,
        }

    total_customers = int(len(results))
    predicted_churn = int((results["Prediction"] == "Churn").sum())
    predicted_safe = int((results["Prediction"] == "No Churn").sum())
    average_churn_probability = float(results["Churn Probability"].mean())

    return {
        "total_customers": total_customers,
        "predicted_churn": predicted_churn,
        "predicted_safe": predicted_safe,
        "average_churn_probability": average_churn_probability,
    }


def create_batch_visualizations(results: pd.DataFrame) -> Dict[str, go.Figure]:
    """
    Build Plotly charts for batch prediction analytics.

    Args:
        results: Output dataframe from ``predict_batch``.

    Returns:
        Dictionary of Plotly figures keyed by chart name.
    """
    if results.empty:
        empty_figure = go.Figure()
        empty_figure.update_layout(
            title="No data available",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return {
            "prediction_distribution": empty_figure,
            "risk_distribution": empty_figure,
            "probability_histogram": empty_figure,
        }

    prediction_counts = (
        results["Prediction"]
        .value_counts()
        .rename_axis("Prediction")
        .reset_index(name="Count")
    )
    prediction_distribution = px.bar(
        prediction_counts,
        x="Prediction",
        y="Count",
        color="Prediction",
        title="Prediction Distribution",
        color_discrete_map={"Churn": "#ef4444", "No Churn": "#10b981"},
    )

    risk_counts = (
        results["Risk Level"]
        .value_counts()
        .rename_axis("Risk Level")
        .reset_index(name="Count")
    )
    risk_order = [
        "Very Low Risk",
        "Low Risk",
        "Moderate Risk",
        "High Risk",
        "Critical Risk",
    ]
    risk_counts["Risk Level"] = pd.Categorical(
        risk_counts["Risk Level"],
        categories=risk_order,
        ordered=True,
    )
    risk_counts = risk_counts.sort_values("Risk Level")
    risk_distribution = px.bar(
        risk_counts,
        x="Risk Level",
        y="Count",
        color="Risk Level",
        title="Risk Level Distribution",
        color_discrete_sequence=px.colors.sequential.Blues_r,
    )

    probability_histogram = px.histogram(
        results,
        x="Churn Probability",
        nbins=20,
        title="Churn Probability Histogram",
        color_discrete_sequence=["#2563eb"],
    )
    probability_histogram.update_layout(
        xaxis_title="Churn Probability",
        yaxis_title="Number of Customers",
    )

    chart_layout = {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": "#374151"},
        "margin": {"l": 24, "r": 24, "t": 56, "b": 24},
        "height": 360,
    }

    for figure in (prediction_distribution, risk_distribution, probability_histogram):
        figure.update_layout(**chart_layout)
        figure.update_xaxes(gridcolor="#e5e7eb")
        figure.update_yaxes(gridcolor="#e5e7eb")

    return {
        "prediction_distribution": prediction_distribution,
        "risk_distribution": risk_distribution,
        "probability_histogram": probability_histogram,
    }


def results_to_csv_bytes(results: pd.DataFrame) -> bytes:
    """
    Serialize batch prediction results to CSV bytes for download.

    Args:
        results: Output dataframe from ``predict_batch``.

    Returns:
        UTF-8 encoded CSV content as bytes.
    """
    return results.to_csv(index=False).encode("utf-8")
