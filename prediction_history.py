"""
Prediction history utilities for the Customer Churn Analytics platform.

Session-scoped history storage logic kept separate from Streamlit UI rendering.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pandas as pd

PREDICTION_HISTORY_KEY = "prediction_history"
HISTORY_CONFIRM_CLEAR_KEY = "history_confirm_clear"
HISTORY_SELECTED_KEY = "history_selected_record_id"

RISK_LEVEL_OPTIONS: List[str] = [
    "Very Low Risk",
    "Low Risk",
    "Moderate Risk",
    "High Risk",
    "Critical Risk",
]


def initialize_prediction_history(existing_history: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Ensure a prediction history store exists and return it.

    Args:
        existing_history: Current history list from session state, if any.

    Returns:
        Initialized prediction history list.
    """
    if existing_history is None:
        return []
    return list(existing_history)


def format_customer_information(customer_data: Dict[str, Any]) -> str:
    """
    Build a compact customer information label for tables and search.

    Args:
        customer_data: Customer feature dictionary.

    Returns:
        Human-readable summary string.
    """
    senior_label = "Yes" if int(customer_data.get("SeniorCitizen", 0)) == 1 else "No"
    monthly_charges = float(customer_data.get("MonthlyCharges", 0))
    tenure = int(customer_data.get("tenure", 0))

    parts = [
        f"Gender: {str(customer_data.get('gender', 'N/A')).title()}",
        f"Senior: {senior_label}",
        f"Contract: {customer_data.get('Contract', 'N/A')}",
        f"Internet: {customer_data.get('InternetService', 'N/A')}",
        f"Payment: {customer_data.get('PaymentMethod', 'N/A')}",
        f"Charges: ${monthly_charges:,.2f}",
        f"Tenure: {tenure} mo",
    ]
    return " | ".join(parts)


def build_history_record(
    customer_data: Dict[str, Any],
    result: Dict[str, Any],
    confidence: float,
    churn_probability: float,
    stay_probability: float,
    risk_level: str,
    recommendations: List[Dict[str, str]],
    explanation_factors: List[str],
    business_impact: Dict[str, Any],
    timestamp: datetime | None = None,
) -> Dict[str, Any]:
    """
    Build a structured prediction history record.

    Args:
        customer_data: Customer feature dictionary.
        result: Raw prediction result from ``predict_customer``.
        confidence: Model confidence score.
        churn_probability: Predicted churn probability.
        stay_probability: Predicted stay probability.
        risk_level: Assigned risk level label.
        recommendations: Generated retention recommendations.
        explanation_factors: Explanation factor strings.
        business_impact: Estimated business impact metrics.
        timestamp: Prediction timestamp.

    Returns:
        Serializable history record dictionary.
    """
    prediction_time = timestamp or datetime.now()
    customer_id = customer_data.get("customerID") or customer_data.get("customer_id")

    return {
        "record_id": str(uuid4()),
        "prediction_timestamp": prediction_time.isoformat(timespec="seconds"),
        "prediction_date": prediction_time.date().isoformat(),
        "prediction": str(result["prediction"]),
        "churn_probability": float(churn_probability),
        "stay_probability": float(stay_probability),
        "risk_level": risk_level,
        "confidence": float(confidence),
        "customer_id": str(customer_id) if customer_id else "",
        "customer_data": dict(customer_data),
        "customer_information": format_customer_information(customer_data),
        "retention_recommendation_count": len(recommendations),
        "recommendations": list(recommendations),
        "explanation_factors": list(explanation_factors),
        "business_impact": dict(business_impact),
        "result": dict(result),
    }


def save_prediction_history(
    history: List[Dict[str, Any]],
    record: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Append a prediction record to the in-memory history store.

    Args:
        history: Existing prediction history list.
        record: New history record from ``build_history_record``.

    Returns:
        Updated history list with the new record appended.
    """
    updated_history = list(history)
    updated_history.insert(0, record)
    return updated_history


def filter_prediction_history(
    history: List[Dict[str, Any]],
    customer_id: str = "",
    prediction: str = "All",
    risk_level: str = "All",
    filter_date: date | None = None,
    search_text: str = "",
) -> List[Dict[str, Any]]:
    """
    Filter prediction history records using search criteria.

    Args:
        history: Full prediction history list.
        customer_id: Optional customer ID filter (partial match).
        prediction: Prediction label filter or ``\"All\"``.
        risk_level: Risk level filter or ``\"All\"``.
        filter_date: Optional date filter matched against prediction date.
        search_text: Free-text search across customer fields and summary text.

    Returns:
        Filtered list of history records.
    """
    filtered = list(history)

    if customer_id.strip():
        needle = customer_id.strip().lower()
        filtered = [
            record
            for record in filtered
            if needle in str(record.get("customer_id", "")).lower()
        ]

    if prediction != "All":
        filtered = [record for record in filtered if record.get("prediction") == prediction]

    if risk_level != "All":
        filtered = [record for record in filtered if record.get("risk_level") == risk_level]

    if filter_date is not None:
        target_date = filter_date.isoformat()
        filtered = [
            record for record in filtered if record.get("prediction_date") == target_date
        ]

    if search_text.strip():
        needle = search_text.strip().lower()
        filtered = [
            record
            for record in filtered
            if needle in str(record.get("customer_information", "")).lower()
            or needle in str(record.get("prediction", "")).lower()
            or needle in str(record.get("risk_level", "")).lower()
            or needle in str(record.get("customer_id", "")).lower()
            or any(
                needle in str(value).lower()
                for value in record.get("customer_data", {}).values()
            )
        ]

    return filtered


def build_history_summary(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build KPI summary metrics from prediction history.

    Args:
        history: Prediction history list.

    Returns:
        Dictionary with totals and average churn probability.
    """
    if not history:
        return {
            "total_predictions": 0,
            "high_risk_customers": 0,
            "low_risk_customers": 0,
            "average_churn_probability": 0.0,
        }

    total_predictions = len(history)
    high_risk_customers = sum(1 for record in history if record.get("prediction") == "Churn")
    low_risk_customers = sum(1 for record in history if record.get("prediction") == "No Churn")
    average_churn_probability = sum(
        float(record.get("churn_probability", 0.0)) for record in history
    ) / total_predictions

    return {
        "total_predictions": total_predictions,
        "high_risk_customers": high_risk_customers,
        "low_risk_customers": low_risk_customers,
        "average_churn_probability": average_churn_probability,
    }


def history_to_dataframe(history: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert history records to a display-ready dataframe.

    Args:
        history: Prediction history list.

    Returns:
        DataFrame for table display and CSV export.
    """
    if not history:
        return pd.DataFrame(
            columns=[
                "Prediction Time",
                "Customer ID",
                "Prediction",
                "Risk Level",
                "Probability",
                "Confidence",
                "Customer Information",
                "Retention Recommendations",
            ]
        )

    rows = []
    for record in history:
        rows.append(
            {
                "record_id": record.get("record_id", ""),
                "Prediction Time": record.get("prediction_timestamp", ""),
                "Customer ID": record.get("customer_id") or "—",
                "Prediction": record.get("prediction", ""),
                "Risk Level": record.get("risk_level", ""),
                "Probability": float(record.get("churn_probability", 0.0)),
                "Confidence": float(record.get("confidence", 0.0)),
                "Customer Information": record.get("customer_information", ""),
                "Retention Recommendations": int(
                    record.get("retention_recommendation_count", 0)
                ),
            }
        )

    return pd.DataFrame(rows)


def download_prediction_history(history: List[Dict[str, Any]]) -> bytes:
    """
    Serialize prediction history to CSV bytes for download.

    Args:
        history: Prediction history list.

    Returns:
        UTF-8 encoded CSV content as bytes.
    """
    export_df = history_to_dataframe(history)
    display_columns = [
        column for column in export_df.columns if column != "record_id"
    ]
    return export_df[display_columns].to_csv(index=False).encode("utf-8")


def find_history_record(
    history: List[Dict[str, Any]],
    record_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Find a history record by its unique identifier.

    Args:
        history: Prediction history list.
        record_id: Record identifier.

    Returns:
        Matching record or ``None`` if not found.
    """
    for record in history:
        if record.get("record_id") == record_id:
            return record
    return None


def clear_prediction_history() -> List[Dict[str, Any]]:
    """
    Return an empty history list for session reset.

    Returns:
        Empty prediction history list.
    """
    return []
