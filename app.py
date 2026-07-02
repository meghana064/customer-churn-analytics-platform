"""
Streamlit web application for Customer Churn Prediction.

Provides an interactive interface to collect customer details and generate
churn predictions using the trained model artifacts and prediction pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    roc_curve,
)

from predict import load_model, predict_customer
from preprocess import (
    clean_dataset,
    encode_features,
    load_dataset,
    split_features_target,
    train_test_split_data,
)
from train_model import (
    DEFAULT_DATA_PATH,
    _drop_identifier_columns,
    display_comparison_table,
    select_best_model,
    train_and_evaluate_models,
)

MODEL_NAME_MAP: Dict[str, str] = {
    "LogisticRegression": "Logistic Regression",
    "DecisionTreeClassifier": "Decision Tree",
    "RandomForestClassifier": "Random Forest",
}

YES_NO_OPTIONS: List[str] = ["No", "Yes"]
SENIOR_CITIZEN_OPTIONS: Dict[str, int] = {"No": 0, "Yes": 1}
INTERNET_SERVICE_OPTIONS: List[str] = ["No", "DSL", "Fiber optic"]
SERVICE_ADDON_OPTIONS: List[str] = ["No", "Yes", "No internet service"]
MULTIPLE_LINES_OPTIONS: List[str] = ["No", "Yes", "No phone service"]
CONTRACT_OPTIONS: List[str] = ["Month-to-month", "One year", "Two year"]
PAYMENT_METHOD_OPTIONS: List[str] = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]


def inject_custom_styles() -> None:
    """Inject custom CSS for a modern, professional UI."""
    st.markdown(
        """
        <style>
            .main-header {
                font-size: 2.4rem;
                font-weight: 700;
                margin-bottom: 0.25rem;
                color: #1f2937;
            }
            .sub-header {
                font-size: 1.05rem;
                color: #4b5563;
                margin-bottom: 1.5rem;
                line-height: 1.6;
            }
            .card {
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 16px;
                padding: 1.25rem 1.5rem;
                margin-bottom: 1rem;
                box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
            }
            .kpi-card {
                background: #ffffff;
                border: 1px solid #dbeafe;
                border-radius: 16px;
                padding: 1rem 1.1rem;
                min-height: 110px;
                box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
            }
            .kpi-title {
                font-size: 0.9rem;
                color: #6b7280;
                margin-bottom: 0.35rem;
            }
            .kpi-value {
                font-size: 1.45rem;
                font-weight: 700;
                color: #111827;
            }
            .result-card-high {
                background: linear-gradient(135deg, #fff5f5 0%, #ffffff 100%);
                border: 1px solid #fecaca;
                border-radius: 18px;
                padding: 1.5rem;
                margin-top: 1rem;
            }
            .result-card-low {
                background: linear-gradient(135deg, #f0fdf4 0%, #ffffff 100%);
                border: 1px solid #bbf7d0;
                border-radius: 18px;
                padding: 1.5rem;
                margin-top: 1rem;
            }
            .insight-card {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 16px;
                padding: 1.25rem 1.5rem;
                margin-top: 1rem;
            }
            .footer-text {
                text-align: center;
                color: #6b7280;
                font-size: 0.95rem;
                margin-top: 2rem;
                padding-top: 1rem;
                border-top: 1px solid #e5e7eb;
            }
            .metric-label {
                font-size: 0.95rem;
                color: #6b7280;
                margin-bottom: 0.25rem;
            }
            .metric-value {
                font-size: 1.6rem;
                font-weight: 700;
                color: #111827;
            }
            div[data-testid="stSidebar"] {
                background-color: #f8fafc;
            }
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
            .executive-header {
                font-size: 2.4rem;
                font-weight: 700;
                margin-bottom: 0.25rem;
                color: #1f2937;
            }
            .executive-sub-header {
                font-size: 1.05rem;
                color: #4b5563;
                margin-bottom: 1.5rem;
                line-height: 1.6;
            }
            .executive-kpi-card {
                background: #ffffff;
                border: 1px solid #dbeafe;
                border-radius: 16px;
                padding: 1rem 1.1rem;
                min-height: 118px;
                box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
            }
            .executive-kpi-icon {
                font-size: 1.35rem;
                margin-bottom: 0.35rem;
            }
            .executive-kpi-title {
                font-size: 0.85rem;
                color: #6b7280;
                margin-bottom: 0.35rem;
            }
            .executive-kpi-value {
                font-size: 1.35rem;
                font-weight: 700;
                color: #111827;
            }
            .executive-chart-card {
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 16px;
                padding: 0.75rem 1rem 0.25rem 1rem;
                margin-bottom: 1rem;
                box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
            }
            .executive-summary-card {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 16px;
                padding: 1.25rem 1.5rem;
                margin: 1rem 0 1.5rem 0;
            }
            .executive-summary-item {
                font-size: 0.98rem;
                color: #374151;
                margin-bottom: 0.45rem;
                line-height: 1.5;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def get_model_metadata() -> Tuple[Optional[Any], Optional[Dict[str, Any]], str, bool, Optional[str]]:
    """
    Load model artifacts and return metadata for the sidebar and predictions.

    Returns:
        Tuple of model, encoders, algorithm name, loaded flag, and error message.
    """
    try:
        model, encoders = load_model()
        algorithm = MODEL_NAME_MAP.get(
            type(model).__name__,
            type(model).__name__,
        )
        return model, encoders, algorithm, True, None
    except FileNotFoundError as exc:
        return None, None, "Not Available", False, str(exc)
    except ValueError as exc:
        return None, None, "Not Available", False, str(exc)
    except Exception as exc:
        return None, None, "Not Available", False, f"Unexpected error: {exc}"


@st.cache_data(show_spinner="Preparing model evaluation dashboard...")
def load_evaluation_data(
    csv_path: str = str(DEFAULT_DATA_PATH),
) -> Dict[str, Any]:
    """
    Prepare evaluation artifacts for the dashboard using the existing pipeline.

    Args:
        csv_path: Path to the churn dataset CSV.

    Returns:
        Dictionary containing metrics, predictions, and model comparison data.
    """
    raw_df = load_dataset(csv_path)
    cleaned_df = clean_dataset(raw_df)
    encoded_df, _ = encode_features(cleaned_df)
    features, target = split_features_target(encoded_df)
    features = _drop_identifier_columns(features)

    x_train, x_test, y_train, y_test = train_test_split_data(features, target)
    metrics, fitted_models = train_and_evaluate_models(
        x_train,
        x_test,
        y_train,
        y_test,
    )

    best_model_name = select_best_model(metrics)
    best_model = fitted_models[best_model_name]
    saved_model, _ = load_model()

    y_pred = saved_model.predict(x_test)
    y_proba = saved_model.predict_proba(x_test)[:, 1]

    return {
        "metrics": metrics,
        "best_model_name": best_model_name,
        "comparison_table": display_comparison_table(metrics),
        "feature_names": list(x_test.columns),
        "x_test": x_test,
        "y_test": y_test.to_numpy(),
        "y_pred": y_pred,
        "y_proba": y_proba,
        "saved_model": saved_model,
        "best_model": best_model,
    }


def render_sidebar(
    algorithm: str,
    model_loaded: bool,
    error_message: Optional[str],
) -> str:
    """Render the application sidebar and return the selected page."""
    with st.sidebar:
        st.markdown("### Navigation")
        page = st.radio(
            "Navigation",
            [
                "🔮 Churn Prediction",
                "📊 Model Evaluation",
                "📊 Executive Dashboard",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("### Project Status")
        if model_loaded:
            st.success("🟢 Model Loaded")
        else:
            st.error("🔴 Model Not Loaded")
            if error_message:
                st.caption(error_message)

        st.markdown("---")
        st.markdown("**Dataset:**  \nIBM Telco Customer Churn")
        st.markdown(f"**Algorithm:**  \n{algorithm}")
        st.markdown("**Developer:**  \nMeghana Gowda M")

    return str(page)


def render_landing_section() -> None:
    """Render the landing page hero section."""
    st.markdown(
        '<p class="main-header">🤖 Customer Churn Prediction</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <p class="sub-header">
        Predict whether a customer is likely to leave using a trained Machine Learning model.
        Churn prediction helps businesses identify at-risk customers early, reduce revenue loss,
        and prioritize retention efforts such as loyalty offers, proactive support, and contract upgrades.
        </p>
        """,
        unsafe_allow_html=True,
    )


def render_customer_form() -> Dict[str, Any]:
    """
    Render the customer information form and collect user input.

    Returns:
        Dictionary of customer feature values aligned with the prediction pipeline.
    """
    st.markdown("### Customer Information")
    st.markdown(
        '<div class="card">Enter customer details below to estimate churn risk.</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", list(SENIOR_CITIZEN_OPTIONS.keys()))
        partner = st.selectbox("Partner", YES_NO_OPTIONS)
        dependents = st.selectbox("Dependents", YES_NO_OPTIONS)
        tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)
        phone_service = st.selectbox("Phone Service", YES_NO_OPTIONS)
        multiple_lines = st.selectbox("Multiple Lines", MULTIPLE_LINES_OPTIONS)

    with col2:
        internet_service = st.selectbox("Internet Service", INTERNET_SERVICE_OPTIONS)
        online_security = st.selectbox("Online Security", SERVICE_ADDON_OPTIONS)
        online_backup = st.selectbox("Online Backup", SERVICE_ADDON_OPTIONS)
        device_protection = st.selectbox("Device Protection", SERVICE_ADDON_OPTIONS)
        tech_support = st.selectbox("Tech Support", SERVICE_ADDON_OPTIONS)
        streaming_tv = st.selectbox("Streaming TV", SERVICE_ADDON_OPTIONS)
        streaming_movies = st.selectbox("Streaming Movies", SERVICE_ADDON_OPTIONS)

    with col3:
        contract = st.selectbox("Contract", CONTRACT_OPTIONS)
        paperless_billing = st.selectbox("Paperless Billing", YES_NO_OPTIONS)
        payment_method = st.selectbox("Payment Method", PAYMENT_METHOD_OPTIONS)
        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=200.0,
            value=70.0,
            step=0.5,
        )
        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=10000.0,
            value=840.0,
            step=1.0,
        )

    return {
        "gender": gender,
        "SeniorCitizen": SENIOR_CITIZEN_OPTIONS[senior_citizen],
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }


def generate_business_insights(
    customer_data: Dict[str, Any],
    prediction: str,
) -> Tuple[List[str], List[str]]:
    """
    Generate business insights based on customer profile and prediction.

    Args:
        customer_data: Customer feature dictionary.
        prediction: Predicted churn outcome.

    Returns:
        Tuple of churn risk factors and recommended business actions.
    """
    risk_factors: List[str] = []
    actions: List[str] = []

    if customer_data["Contract"] == "Month-to-month":
        risk_factors.append("Month-to-month contracts are associated with higher churn risk.")
    if customer_data["tenure"] < 12:
        risk_factors.append("Short customer tenure often indicates lower loyalty.")
    if customer_data["PaymentMethod"] == "Electronic check":
        risk_factors.append("Electronic check payments are commonly linked to higher churn.")
    if customer_data["InternetService"] == "Fiber optic" and customer_data["MonthlyCharges"] > 80:
        risk_factors.append("High monthly charges on premium internet may reduce satisfaction.")
    if customer_data["OnlineSecurity"] == "No" and customer_data["InternetService"] != "No":
        risk_factors.append("Lack of online security add-ons can increase dissatisfaction.")
    if customer_data["TechSupport"] == "No" and customer_data["InternetService"] != "No":
        risk_factors.append("Customers without tech support may struggle to resolve service issues.")
    if customer_data["Partner"] == "No" and customer_data["Dependents"] == "No":
        risk_factors.append("Single-household customers may switch providers more easily.")

    if prediction == "Churn":
        actions.extend(
            [
                "Offer targeted discounts or loyalty rewards to improve retention.",
                "Provide proactive customer support and service quality reviews.",
                "Recommend contract upgrades with long-term pricing benefits.",
                "Follow up on billing concerns and payment method flexibility.",
            ]
        )
    else:
        actions.extend(
            [
                "Maintain engagement through satisfaction surveys and service updates.",
                "Promote value-added services that strengthen customer stickiness.",
                "Recognize loyal customers with referral or renewal incentives.",
                "Continue monitoring usage patterns for early signs of dissatisfaction.",
            ]
        )

    if not risk_factors:
        risk_factors.append(
            "No major high-risk indicators were detected in the selected profile."
        )

    return risk_factors, actions


def render_prediction_result(result: Dict[str, Any], customer_data: Dict[str, Any]) -> None:
    """
    Render prediction output, confidence, and business insights.

    Args:
        result: Prediction result from ``predict_customer``.
        customer_data: Original customer input dictionary.
    """
    prediction = str(result["prediction"])
    churn_probability = float(result["churn_probability"])
    stay_probability = float(result["stay_probability"])
    confidence = max(churn_probability, stay_probability)

    st.markdown("### Prediction Result")

    if prediction == "Churn":
        st.markdown(
            f"""
            <div class="result-card-high">
                <h3>🔴 High Risk Customer</h3>
                <p class="metric-label">Probability of Churn</p>
                <p class="metric-value">{churn_probability:.1%}</p>
                <p><strong>Recommended Action:</strong> Offer discounts, loyalty plans or customer support.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="result-card-low">
                <h3>🟢 Low Risk Customer</h3>
                <p class="metric-label">Probability of Staying</p>
                <p class="metric-value">{stay_probability:.1%}</p>
                <p><strong>Recommended Action:</strong> Maintain engagement and customer satisfaction.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Prediction Confidence")
    st.progress(confidence, text=f"Confidence: {confidence:.1%}")

    risk_factors, actions = generate_business_insights(customer_data, prediction)

    st.markdown("### Business Insights")
    st.markdown('<div class="insight-card">', unsafe_allow_html=True)
    st.markdown("**Why the customer may churn:**")
    for factor in risk_factors:
        st.markdown(f"- {factor}")

    st.markdown("**Business actions to reduce churn:**")
    for action in actions:
        st.markdown(f"- {action}")
    st.markdown("</div>", unsafe_allow_html=True)


def render_kpi_card(title: str, value: str) -> None:
    """Render a single KPI card."""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray) -> None:
    """Render a confusion matrix chart."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Churn", "Churn"],
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_roc_curve(y_test: np.ndarray, y_proba: np.ndarray) -> float:
    """Render an ROC curve and return the AUC score."""
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(fpr, tpr, color="#2563eb", linewidth=2, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#9ca3af", linewidth=1)
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    return float(roc_auc)


def render_feature_importance(model: Any, feature_names: List[str]) -> None:
    """Render top feature importances when supported by the model."""
    if not hasattr(model, "feature_importances_"):
        st.info(
            "Feature importance is not available for the selected model. "
            "This visualization is supported for tree-based models such as "
            "Random Forest and Decision Tree."
        )
        return

    importances = model.feature_importances_
    importance_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .head(10)
        .sort_values("Importance", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(importance_df["Feature"], importance_df["Importance"], color="#10b981")
    ax.set_title("Top 10 Feature Importances")
    ax.set_xlabel("Importance Score")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def highlight_best_model_row(row: pd.Series, best_model_name: str) -> List[str]:
    """Apply highlight styling to the best model row."""
    if row["Model"] == best_model_name:
        return ["background-color: #dbeafe; font-weight: 600;"] * len(row)
    return [""] * len(row)


def render_business_interpretation(best_model_name: str, best_metrics: Dict[str, float]) -> None:
    """Render business interpretation for the selected model."""
    st.markdown("### 🧠 Business Interpretation")
    st.markdown('<div class="insight-card">', unsafe_allow_html=True)

    st.markdown(
        f"""
        **Why the selected model performed best:**  
        **{best_model_name}** achieved the strongest ROC-AUC of **{best_metrics['roc_auc']:.3f}**
        and an F1 Score of **{best_metrics['f1_score']:.3f}**, indicating a strong balance between
        identifying churners correctly and limiting false alarms. Tree-based models often perform well
        on churn datasets because customer behavior is influenced by non-linear patterns and feature
        interactions such as contract type, tenure, and monthly charges.
        """
    )

    st.markdown(
        """
        **How businesses can use this model:**  
        The model can score customers daily or weekly, rank them by churn probability, and help teams
        prioritize retention campaigns. Customer success, marketing, and support teams can use these
        scores to offer contract upgrades, loyalty incentives, or proactive service outreach to
        high-risk customers.
        """
    )

    st.markdown(
        """
        **Benefits of predicting churn early:**  
        Early churn prediction protects recurring revenue, reduces customer acquisition costs, and
        improves customer lifetime value. Instead of reacting after cancellation, businesses can
        intervene while the customer relationship is still recoverable.
        """
    )

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Executive Dashboard — business analytics powered by the raw Telco dataset
# ---------------------------------------------------------------------------

EXECUTIVE_DATA_PATH = Path("data") / "Telco-Customer-Churn.csv"

# Shared Plotly styling aligned with the application's light theme.
PLOTLY_EXECUTIVE_LAYOUT: Dict[str, Any] = {
    "template": "plotly_white",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "#374151", "size": 12},
    "margin": {"l": 24, "r": 24, "t": 56, "b": 24},
    "height": 380,
    "colorway": ["#2563eb", "#10b981", "#ef4444", "#f59e0b", "#8b5cf6"],
}


@st.cache_data(show_spinner="Loading executive dashboard data...")
def load_executive_dashboard_data(
    csv_path: str = str(EXECUTIVE_DATA_PATH),
) -> pd.DataFrame:
    """
    Load and prepare the Telco dataset for executive-level business analytics.

    Uses ``load_dataset`` for file I/O only; analytics are computed in-app
    without altering the preprocessing or training pipelines.

    Args:
        csv_path: Path to the IBM Telco Customer Churn CSV.

    Returns:
        DataFrame ready for KPI and chart calculations.
    """
    dataframe = load_dataset(csv_path).copy()

    # Normalize numeric fields required for revenue and tenure KPIs.
    dataframe["MonthlyCharges"] = pd.to_numeric(dataframe["MonthlyCharges"], errors="coerce")
    dataframe["TotalCharges"] = pd.to_numeric(dataframe["TotalCharges"], errors="coerce")
    dataframe["tenure"] = pd.to_numeric(dataframe["tenure"], errors="coerce")

    # Senior citizen flag is stored as 0/1 in the source dataset.
    dataframe["SeniorCitizenLabel"] = dataframe["SeniorCitizen"].map(
        {0: "No", 1: "Yes"}
    )

    # Composite segment used for the top-10 churn segment chart.
    dataframe["CustomerSegment"] = (
        dataframe["Contract"].astype(str)
        + " | "
        + dataframe["InternetService"].astype(str)
        + " | "
        + dataframe["PaymentMethod"].astype(str)
    )

    return dataframe.dropna(subset=["Churn", "MonthlyCharges", "tenure"])


def compute_executive_kpis(dataframe: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate headline KPI metrics for the executive dashboard.

    Args:
        dataframe: Prepared Telco customer dataset.

    Returns:
        Dictionary of KPI names and numeric values.
    """
    churn_mask = dataframe["Churn"] == "Yes"
    total_customers = len(dataframe)
    churn_customers = int(churn_mask.sum())

    return {
        "total_customers": float(total_customers),
        "churn_customers": float(churn_customers),
        "churn_rate": (churn_customers / total_customers * 100) if total_customers else 0.0,
        "avg_monthly_charges": float(dataframe["MonthlyCharges"].mean()),
        "avg_tenure": float(dataframe["tenure"].mean()),
        "revenue_at_risk": float(dataframe.loc[churn_mask, "MonthlyCharges"].sum()),
    }


def _build_churn_rate_dataframe(dataframe: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Aggregate churn rate by a single categorical dimension.

    Args:
        dataframe: Prepared customer dataset.
        column: Categorical column to group by.

    Returns:
        Summary DataFrame with customer counts and churn rates.
    """
    summary = (
        dataframe.groupby(column, dropna=False)
        .agg(
            customers=("Churn", "size"),
            churn_count=("Churn", lambda values: (values == "Yes").sum()),
        )
        .reset_index()
    )
    summary["churn_rate"] = (summary["churn_count"] / summary["customers"]) * 100
    return summary.sort_values("churn_rate", ascending=False)


def get_highest_churn_category(
    dataframe: pd.DataFrame,
    column: str,
) -> Tuple[str, float]:
    """
    Return the category with the highest churn rate for a given column.

    Args:
        dataframe: Prepared customer dataset.
        column: Categorical feature column.

    Returns:
        Tuple of category name and churn rate percentage.
    """
    summary = _build_churn_rate_dataframe(dataframe, column)
    top_row = summary.iloc[0]
    return str(top_row[column]), float(top_row["churn_rate"])


@st.cache_data(show_spinner="Calculating average churn probability...")
def compute_average_churn_probability(
    csv_path: str = str(EXECUTIVE_DATA_PATH),
) -> Optional[float]:
    """
    Compute the average model-predicted churn probability across all customers.

    Uses the saved model and existing preprocessing helpers without retraining
    or modifying the ML pipeline modules.

    Args:
        csv_path: Path to the Telco customer dataset.

    Returns:
        Mean churn probability, or ``None`` if the model is unavailable.
    """
    try:
        model, _ = load_model()
        raw_df = load_dataset(csv_path)
        cleaned_df = clean_dataset(raw_df)
        encoded_df, _ = encode_features(cleaned_df)
        features, _ = split_features_target(encoded_df)
        features = _drop_identifier_columns(features)
        probabilities = model.predict_proba(features)[:, 1]
        return float(np.mean(probabilities))
    except Exception:
        return None


def compute_business_summary(
    dataframe: pd.DataFrame,
    kpis: Dict[str, float],
    avg_churn_probability: Optional[float],
) -> Dict[str, Any]:
    """
    Build executive business summary insights from dataset KPIs.

    Args:
        dataframe: Prepared customer dataset.
        kpis: Precomputed executive KPI dictionary.
        avg_churn_probability: Optional average predicted churn probability.

    Returns:
        Dictionary of business summary fields for display.
    """
    highest_contract, contract_rate = get_highest_churn_category(dataframe, "Contract")
    highest_payment, payment_rate = get_highest_churn_category(dataframe, "PaymentMethod")
    highest_internet, internet_rate = get_highest_churn_category(dataframe, "InternetService")

    summary: Dict[str, Any] = {
        "highest_churn_contract": highest_contract,
        "highest_churn_contract_rate": contract_rate,
        "highest_churn_payment_method": highest_payment,
        "highest_churn_payment_rate": payment_rate,
        "highest_churn_internet_service": highest_internet,
        "highest_churn_internet_rate": internet_rate,
        "average_tenure": kpis["avg_tenure"],
        "revenue_at_risk": kpis["revenue_at_risk"],
        "average_churn_probability": avg_churn_probability,
    }
    return summary


def _apply_plotly_layout(figure: Any, height: int = 380) -> Any:
    """Apply consistent executive dashboard styling to a Plotly figure."""
    layout = dict(PLOTLY_EXECUTIVE_LAYOUT)
    layout["height"] = height
    figure.update_layout(**layout)
    return figure


def create_churn_bar_chart(
    dataframe: pd.DataFrame,
    column: str,
    title: str,
    color: str = "#2563eb",
    orientation: str = "v",
) -> Any:
    """
    Build a Plotly bar chart showing churn rate by category.

    Args:
        dataframe: Prepared customer dataset.
        column: Feature column to analyze.
        title: Chart title.
        color: Bar color hex code.
        orientation: ``'v'`` for vertical bars, ``'h'`` for horizontal bars.

    Returns:
        Plotly figure object.
    """
    summary = _build_churn_rate_dataframe(dataframe, column)

    if orientation == "h":
        figure = px.bar(
            summary.sort_values("churn_rate", ascending=True),
            x="churn_rate",
            y=column,
            text=summary.sort_values("churn_rate", ascending=True)["churn_rate"].map(
                lambda value: f"{value:.1f}%"
            ),
            orientation="h",
            labels={column: column, "churn_rate": "Churn Rate (%)"},
            title=title,
        )
    else:
        figure = px.bar(
            summary,
            x=column,
            y="churn_rate",
            text=summary["churn_rate"].map(lambda value: f"{value:.1f}%"),
            labels={column: column, "churn_rate": "Churn Rate (%)"},
            title=title,
        )

    figure.update_traces(marker_color=color, textposition="outside")
    figure.update_yaxes(range=[0, max(100, summary["churn_rate"].max() + 5)])
    return _apply_plotly_layout(figure)


def create_churn_donut_chart(
    dataframe: pd.DataFrame,
    column: str,
    title: str,
) -> Any:
    """
    Build a Plotly donut chart of churned customers by category.

    Args:
        dataframe: Prepared customer dataset.
        column: Feature column to analyze.
        title: Chart title.

    Returns:
        Plotly donut chart figure.
    """
    churned_customers = dataframe[dataframe["Churn"] == "Yes"]
    summary = (
        churned_customers.groupby(column, dropna=False)
        .size()
        .reset_index(name="churn_count")
        .sort_values("churn_count", ascending=False)
    )

    figure = px.pie(
        summary,
        names=column,
        values="churn_count",
        hole=0.45,
        title=title,
        labels={column: column, "churn_count": "Churned Customers"},
    )
    figure.update_traces(textposition="inside", textinfo="percent+label")
    return _apply_plotly_layout(figure)


def create_churn_pie_chart(
    dataframe: pd.DataFrame,
    column: str,
    title: str,
) -> Any:
    """
    Build a Plotly pie chart of churned customers by category.

    Args:
        dataframe: Prepared customer dataset.
        column: Feature column to analyze.
        title: Chart title.

    Returns:
        Plotly pie chart figure.
    """
    churned_customers = dataframe[dataframe["Churn"] == "Yes"]
    summary = (
        churned_customers.groupby(column, dropna=False)
        .size()
        .reset_index(name="churn_count")
        .sort_values("churn_count", ascending=False)
    )

    figure = px.pie(
        summary,
        names=column,
        values="churn_count",
        title=title,
        labels={column: column, "churn_count": "Churned Customers"},
    )
    figure.update_traces(textposition="inside", textinfo="percent+label")
    return _apply_plotly_layout(figure)


def create_top_segment_chart(dataframe: pd.DataFrame) -> Any:
    """
    Build a Plotly chart for the top 10 customer segments with highest churn.

    Args:
        dataframe: Prepared customer dataset.

    Returns:
        Plotly figure showing the highest-churn composite segments.
    """
    segment_summary = (
        dataframe.groupby("CustomerSegment")
        .agg(
            customers=("Churn", "size"),
            churn_count=("Churn", lambda values: (values == "Yes").sum()),
        )
        .reset_index()
    )
    segment_summary["churn_rate"] = (
        segment_summary["churn_count"] / segment_summary["customers"]
    ) * 100

    # Focus on segments with enough volume to be actionable for executives.
    segment_summary = segment_summary[segment_summary["customers"] >= 20]
    top_segments = segment_summary.sort_values(
        ["churn_rate", "churn_count"],
        ascending=[False, False],
    ).head(10)

    sorted_segments = top_segments.sort_values("churn_rate", ascending=True)
    figure = px.bar(
        sorted_segments,
        x="churn_rate",
        y="CustomerSegment",
        orientation="h",
        text=sorted_segments["churn_rate"].map(lambda value: f"{value:.1f}%"),
        labels={"CustomerSegment": "Customer Segment", "churn_rate": "Churn Rate (%)"},
        title="Top 10 Customer Segments with Highest Churn",
    )
    figure.update_traces(marker_color="#ef4444", textposition="outside")
    return _apply_plotly_layout(figure, height=460)


@st.cache_data(show_spinner="Loading executive dashboard metrics...")
def get_executive_dashboard_bundle(
    csv_path: str = str(EXECUTIVE_DATA_PATH),
) -> Dict[str, Any]:
    """
    Load and cache all executive dashboard metrics in a single bundle.

    Combines dataset preparation, KPI calculation, and business summary generation
    to avoid recomputation when switching between app pages.

    Args:
        csv_path: Path to the IBM Telco Customer Churn CSV.

    Returns:
        Dictionary containing ``kpis`` and ``business_summary``.
    """
    executive_data = load_executive_dashboard_data(csv_path)
    kpis = compute_executive_kpis(executive_data)
    avg_churn_probability = compute_average_churn_probability(csv_path)
    business_summary = compute_business_summary(
        executive_data,
        kpis,
        avg_churn_probability,
    )
    return {
        "kpis": kpis,
        "business_summary": business_summary,
    }


@st.cache_data(show_spinner="Building executive dashboard charts...")
def build_executive_dashboard_charts(
    csv_path: str = str(EXECUTIVE_DATA_PATH),
) -> Dict[str, Any]:
    """
    Build and cache all Plotly charts for the executive dashboard.

    Args:
        csv_path: Path to the IBM Telco Customer Churn CSV.

    Returns:
        Dictionary of cached Plotly figure objects keyed by chart name.
    """
    executive_data = load_executive_dashboard_data(csv_path)
    return {
        "contract": create_churn_bar_chart(
            executive_data, "Contract", "Churn by Contract"
        ),
        "payment_method": create_churn_bar_chart(
            executive_data,
            "PaymentMethod",
            "Churn by Payment Method",
            color="#10b981",
        ),
        "internet_service": create_churn_donut_chart(
            executive_data,
            "InternetService",
            "Churn by Internet Service",
        ),
        "gender": create_churn_pie_chart(
            executive_data, "gender", "Churn by Gender"
        ),
        "senior_citizen": create_churn_bar_chart(
            executive_data,
            "SeniorCitizenLabel",
            "Churn by Senior Citizen",
            color="#f59e0b",
        ),
        "top_segments": create_top_segment_chart(executive_data),
    }


def render_executive_kpi_card(icon: str, title: str, value: str) -> None:
    """Render a KPI card with icon for the executive dashboard."""
    st.markdown(
        f"""
        <div class="executive-kpi-card">
            <div class="executive-kpi-icon">{icon}</div>
            <div class="executive-kpi-title">{title}</div>
            <div class="executive-kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_business_summary_panel(summary: Dict[str, Any]) -> None:
    """
    Render the automated business summary insight panel.

    Args:
        summary: Business summary dictionary produced by ``compute_business_summary``.
    """
    st.markdown("### Business Summary")
    st.markdown('<div class="executive-summary-card">', unsafe_allow_html=True)

    summary_lines = [
        (
            "**Highest churn contract:** "
            f"{summary['highest_churn_contract']} "
            f"({summary['highest_churn_contract_rate']:.1f}% churn rate)"
        ),
        (
            "**Highest churn payment method:** "
            f"{summary['highest_churn_payment_method']} "
            f"({summary['highest_churn_payment_rate']:.1f}% churn rate)"
        ),
        (
            "**Highest churn internet service:** "
            f"{summary['highest_churn_internet_service']} "
            f"({summary['highest_churn_internet_rate']:.1f}% churn rate)"
        ),
        f"**Average customer tenure:** {summary['average_tenure']:.1f} months",
        f"**Estimated revenue at risk:** ${summary['revenue_at_risk']:,.2f}",
    ]

    if summary["average_churn_probability"] is not None:
        summary_lines.append(
            "**Average churn probability (model):** "
            f"{summary['average_churn_probability']:.1%}"
        )

    for line in summary_lines:
        st.markdown(f'<p class="executive-summary-item">{line}</p>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def render_executive_chart(figure: Any) -> None:
    """Render a Plotly chart inside a styled executive container."""
    st.markdown('<div class="executive-chart-card">', unsafe_allow_html=True)
    st.plotly_chart(figure, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_executive_dashboard_page() -> None:
    """Render the executive business intelligence dashboard."""
    st.markdown(
        '<p class="executive-header">📊 Executive Dashboard</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <p class="executive-sub-header">
        Executive-level view of customer churn, revenue exposure, and the highest-risk
        customer segments across the IBM Telco customer base.
        </p>
        """,
        unsafe_allow_html=True,
    )

    if not EXECUTIVE_DATA_PATH.exists():
        st.error(
            f"Dataset not found at `{EXECUTIVE_DATA_PATH}`. "
            "Add `Telco-Customer-Churn.csv` to the `data/` folder."
        )
        return

    try:
        dashboard_bundle = get_executive_dashboard_bundle()
        charts = build_executive_dashboard_charts()
        kpis = dashboard_bundle["kpis"]
        business_summary = dashboard_bundle["business_summary"]
    except FileNotFoundError as exc:
        st.error("The executive dashboard could not find the dataset file.")
        st.warning(str(exc))
        return
    except Exception as exc:
        st.error("Unable to build the executive dashboard.")
        st.warning(str(exc))
        return

    # KPI cards in a single responsive row
    st.markdown("### Key Business Metrics")
    kpi_columns = st.columns(6)
    kpi_values = [
        ("👥", "Total Customers", f"{kpis['total_customers']:,.0f}"),
        ("📉", "Total Churn Customers", f"{kpis['churn_customers']:,.0f}"),
        ("⚠️", "Churn Rate (%)", f"{kpis['churn_rate']:.2f}%"),
        ("💳", "Average Monthly Charges", f"${kpis['avg_monthly_charges']:.2f}"),
        ("⏳", "Average Customer Tenure", f"{kpis['avg_tenure']:.1f} mo"),
        ("💰", "Monthly Revenue at Risk", f"${kpis['revenue_at_risk']:,.2f}"),
    ]

    for column, (icon, title, value) in zip(kpi_columns, kpi_values):
        with column:
            render_executive_kpi_card(icon, title, value)

    render_business_summary_panel(business_summary)

    st.markdown("---")
    st.markdown("### Churn Analytics")

    # Two-column chart grid with required chart types
    chart_pairs = [
        (charts["contract"], charts["payment_method"]),
        (charts["internet_service"], charts["gender"]),
        (charts["senior_citizen"], charts["top_segments"]),
    ]

    for left_chart, right_chart in chart_pairs:
        left_column, right_column = st.columns(2)
        with left_column:
            render_executive_chart(left_chart)
        with right_column:
            render_executive_chart(right_chart)


def render_model_evaluation_page(model_loaded: bool) -> None:
    """Render the model evaluation dashboard."""
    st.markdown(
        '<p class="main-header">📊 Model Evaluation</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <p class="sub-header">
        Review model performance, compare algorithms, and understand how the selected model supports
        business decisions for customer retention.
        </p>
        """,
        unsafe_allow_html=True,
    )

    if not model_loaded:
        st.error(
            "Model evaluation is unavailable because trained artifacts were not found. "
            "Run `python train_model.py` after adding the dataset."
        )
        return

    if not Path(DEFAULT_DATA_PATH).exists():
        st.error(
            f"Dataset not found at `{DEFAULT_DATA_PATH}`. "
            "Add `Telco-Customer-Churn.csv` to the `data/` folder to view evaluation metrics."
        )
        return

    try:
        evaluation = load_evaluation_data()
    except FileNotFoundError as exc:
        st.error("Required files for evaluation were not found.")
        st.warning(str(exc))
        return
    except ValueError as exc:
        st.error("Evaluation could not be completed due to a data or model issue.")
        st.warning(str(exc))
        return
    except Exception as exc:
        st.error("An unexpected error occurred while building the evaluation dashboard.")
        st.warning(str(exc))
        return

    best_model_name = evaluation["best_model_name"]
    best_metrics = evaluation["metrics"][best_model_name]
    comparison_table = evaluation["comparison_table"].copy()

    st.markdown("### Key Performance Indicators")
    kpi_cols = st.columns(6)
    kpi_values = [
        ("Best Model", best_model_name),
        ("Accuracy", f"{best_metrics['accuracy']:.3f}"),
        ("Precision", f"{best_metrics['precision']:.3f}"),
        ("Recall", f"{best_metrics['recall']:.3f}"),
        ("F1 Score", f"{best_metrics['f1_score']:.3f}"),
        ("ROC-AUC", f"{best_metrics['roc_auc']:.3f}"),
    ]

    for column, (title, value) in zip(kpi_cols, kpi_values):
        with column:
            render_kpi_card(title, value)

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("### Confusion Matrix")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        render_confusion_matrix(evaluation["y_test"], evaluation["y_pred"])
        st.markdown("</div>", unsafe_allow_html=True)

    with chart_col2:
        st.markdown("### ROC Curve")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        auc_score = render_roc_curve(evaluation["y_test"], evaluation["y_proba"])
        st.caption(f"AUC Score: **{auc_score:.3f}**")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Feature Importance")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    render_feature_importance(
        evaluation["saved_model"],
        evaluation["feature_names"],
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Model Comparison")
    styled_table = comparison_table.style.apply(
        lambda row: highlight_best_model_row(row, best_model_name),
        axis=1,
    )
    st.dataframe(styled_table, use_container_width=True, hide_index=True)

    render_business_interpretation(best_model_name, best_metrics)


def render_prediction_page(model_loaded: bool, error_message: Optional[str]) -> None:
    """Render the churn prediction page."""
    render_landing_section()
    customer_data = render_customer_form()

    st.markdown("---")
    predict_clicked = st.button("🔮 Predict Customer Churn", type="primary", use_container_width=True)

    if predict_clicked:
        if not model_loaded:
            st.error(
                "Prediction unavailable. Please train the model first by running "
                "`python train_model.py` and ensure artifacts exist in the `model/` folder."
            )
            if error_message:
                st.warning(error_message)
            return

        try:
            result = predict_customer(customer_data)
            render_prediction_result(result, customer_data)
        except FileNotFoundError as exc:
            st.error("Required model files were not found.")
            st.warning(str(exc))
        except ValueError as exc:
            st.error("Invalid input or prediction error.")
            st.warning(str(exc))
        except Exception as exc:
            st.error("An unexpected error occurred during prediction.")
            st.warning(str(exc))


def render_footer() -> None:
    """Render the application footer."""
    st.markdown(
        """
        <div class="footer-text">
            <strong>Built using:</strong> Streamlit · Scikit-learn · Pandas · Python<br>
            <strong>Author:</strong> Meghana Gowda M
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Run the Streamlit churn prediction application."""
    st.set_page_config(
        page_title="Customer Churn Prediction",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_custom_styles()
    _, _, algorithm, model_loaded, error_message = get_model_metadata()
    page = render_sidebar(algorithm, model_loaded, error_message)

    if page == "📊 Model Evaluation":
        render_model_evaluation_page(model_loaded)
    elif page == "📊 Executive Dashboard":
        render_executive_dashboard_page()
    else:
        render_prediction_page(model_loaded, error_message)

    render_footer()


if __name__ == "__main__":
    main()
