"""
Streamlit web application for Customer Churn Prediction.

Provides an interactive interface to collect customer details and generate
churn predictions using the trained model artifacts and prediction pipeline.
"""

from __future__ import annotations

from datetime import datetime
from hashlib import md5
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    roc_curve,
)

from predict import load_model, predict_customer
from batch_predict import (
    build_batch_summary,
    create_batch_visualizations,
    predict_batch,
    results_to_csv_bytes,
    validate_batch_dataset,
)
from executive_insights import (
    compute_executive_insights,
    load_insights_dataframe,
)
from prediction_history import (
    HISTORY_CONFIRM_CLEAR_KEY,
    HISTORY_SELECTED_KEY,
    PREDICTION_HISTORY_KEY,
    build_history_record,
    build_history_summary,
    clear_prediction_history,
    download_prediction_history,
    filter_prediction_history,
    find_history_record,
    initialize_prediction_history,
    save_prediction_history,
)
from pdf_report import build_prediction_summary, generate_prediction_pdf
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
from ui_theme import (
    apply_enterprise_plotly_theme,
    close_chart_card,
    format_history_record_label,
    inject_app_styles,
    inject_history_page_styles,
    inject_prediction_page_styles,
    NAVIGATION_KEY,
    open_chart_card,
    render_app_header,
    render_confirm_banner,
    render_empty_state,
    render_history_filter_header,
    render_history_table,
    render_html_card,
    render_kpi_card as render_premium_kpi_card,
    render_page_header,
    render_plotly_chart,
    render_premium_footer,
    render_section_header,
    render_sidebar_brand,
    render_sidebar_collapse_toggle,
    render_sidebar_metadata,
    render_sidebar_navigation,
    render_sidebar_section_label,
    is_sidebar_collapsed,
    render_status_badge_html,
    render_structured_customer_info,
    render_validation_error_card,
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

# Thresholds used by prediction explanation business rules.
HIGH_MONTHLY_CHARGE_THRESHOLD = 70.0
LOW_MONTHLY_CHARGE_THRESHOLD = 50.0
SHORT_TENURE_THRESHOLD = 12
LONG_TENURE_THRESHOLD = 24
AUTOMATIC_PAYMENT_METHODS = {
    "Bank transfer (automatic)",
    "Credit card (automatic)",
}

HISTORY_APPLIED_FILTERS_KEY = "history_applied_filters"
HISTORY_FILTER_WIDGET_KEYS = {
    "customer_id": "hf_customer_id",
    "prediction": "hf_prediction",
    "risk_level": "hf_risk_level",
    "use_date": "hf_use_date",
    "filter_date": "hf_filter_date",
    "search_text": "hf_search_text",
}


def inject_custom_styles() -> None:
    """Inject custom CSS for a modern, professional UI."""
    inject_app_styles()


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

    model_path = Path("model") / "churn_model.pkl"
    training_date = None
    if model_path.exists():
        training_date = datetime.fromtimestamp(model_path.stat().st_mtime).strftime(
            "%Y-%m-%d %H:%M"
        )

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
        "training_samples": int(len(x_train)),
        "testing_samples": int(len(x_test)),
        "dataset_size": int(len(cleaned_df)),
        "num_features": int(features.shape[1]),
        "training_date": training_date,
    }


def render_sidebar(
    algorithm: str,
    model_loaded: bool,
    error_message: Optional[str],
    prediction_count: int = 0,
) -> str:
    """Render the application sidebar and return the selected page."""
    with st.sidebar:
        render_sidebar_collapse_toggle()
        render_sidebar_brand()
        if not is_sidebar_collapsed():
            render_sidebar_section_label("Navigation")
        page = render_sidebar_navigation()
        if not is_sidebar_collapsed():
            render_sidebar_metadata(algorithm, model_loaded, prediction_count)
            if not model_loaded and error_message:
                st.caption(error_message)

    return str(page)


def render_landing_section() -> None:
    """Render the landing page hero section."""
    render_page_header(
        "🔮",
        "Customer Churn Prediction",
        "Predict whether a customer is likely to leave using a trained machine learning model. "
        "Churn prediction helps businesses identify at-risk customers early, reduce revenue loss, "
        "and prioritize retention efforts.",
    )


def _render_history_filters_panel() -> None:
    """Render Search & Filters with a clean responsive grid."""
    render_history_filter_header()

    filter_row = st.columns([1.4, 1, 1, 1.4], gap="medium")
    with filter_row[0]:
        st.text_input(
            "Customer ID",
            placeholder="e.g. 7590-VHVEG",
            key=HISTORY_FILTER_WIDGET_KEYS["customer_id"],
        )
    with filter_row[1]:
        st.selectbox(
            "Prediction",
            ["All", "Churn", "No Churn"],
            key=HISTORY_FILTER_WIDGET_KEYS["prediction"],
        )
    with filter_row[2]:
        st.selectbox(
            "Risk Level",
            [
                "All",
                "Very Low Risk",
                "Low Risk",
                "Moderate Risk",
                "High Risk",
                "Critical Risk",
            ],
            key=HISTORY_FILTER_WIDGET_KEYS["risk_level"],
        )
    with filter_row[3]:
        st.text_input(
            "Keyword",
            placeholder="Contract, payment, internet, tenure...",
            key=HISTORY_FILTER_WIDGET_KEYS["search_text"],
        )

    date_row = st.columns([1.4, 1, 1, 1.4], gap="medium")
    with date_row[0]:
        st.date_input(
            "Prediction Date",
            key=HISTORY_FILTER_WIDGET_KEYS["filter_date"],
        )
    with date_row[1]:
        if st.button("Apply Filters", type="primary", width="stretch", key="history_apply_filters"):
            applied = _capture_history_filters_from_widgets()
            applied["use_date"] = True
            st.session_state[HISTORY_APPLIED_FILTERS_KEY] = applied
            st.rerun()
    with date_row[2]:
        if st.button("Reset Filters", type="secondary", width="stretch", key="history_reset_filters"):
            _reset_history_filter_widgets()
            st.rerun()


def _history_filter_defaults() -> Dict[str, Any]:
    """Return default UI filter values for the history page."""
    return {
        "customer_id": "",
        "prediction": "All",
        "risk_level": "All",
        "use_date": False,
        "filter_date": datetime.now().date(),
        "search_text": "",
    }


def _ensure_history_applied_filters() -> Dict[str, Any]:
    """Initialize applied history filters in session state if missing."""
    if HISTORY_APPLIED_FILTERS_KEY not in st.session_state:
        st.session_state[HISTORY_APPLIED_FILTERS_KEY] = _history_filter_defaults()
    return dict(st.session_state[HISTORY_APPLIED_FILTERS_KEY])


def _reset_history_filter_widgets() -> None:
    """Reset history filter widgets and applied values to defaults."""
    defaults = _history_filter_defaults()
    st.session_state[HISTORY_APPLIED_FILTERS_KEY] = defaults
    for field, widget_key in HISTORY_FILTER_WIDGET_KEYS.items():
        st.session_state[widget_key] = defaults[field]


def _sync_history_widget_defaults() -> None:
    """Ensure filter widgets start from applied filter values."""
    applied = _ensure_history_applied_filters()
    for field, widget_key in HISTORY_FILTER_WIDGET_KEYS.items():
        if widget_key not in st.session_state:
            st.session_state[widget_key] = applied[field]


def _capture_history_filters_from_widgets() -> Dict[str, Any]:
    """Read current widget values for Apply Filters."""
    defaults = _history_filter_defaults()
    return {
        "customer_id": st.session_state.get(
            HISTORY_FILTER_WIDGET_KEYS["customer_id"], defaults["customer_id"]
        ),
        "prediction": st.session_state.get(
            HISTORY_FILTER_WIDGET_KEYS["prediction"], defaults["prediction"]
        ),
        "risk_level": st.session_state.get(
            HISTORY_FILTER_WIDGET_KEYS["risk_level"], defaults["risk_level"]
        ),
        "use_date": st.session_state.get(
            HISTORY_FILTER_WIDGET_KEYS["use_date"], defaults["use_date"]
        ),
        "filter_date": st.session_state.get(
            HISTORY_FILTER_WIDGET_KEYS["filter_date"], defaults["filter_date"]
        ),
        "search_text": st.session_state.get(
            HISTORY_FILTER_WIDGET_KEYS["search_text"], defaults["search_text"]
        ),
    }


def _yes_no_toggle(
    label: str,
    key: str,
    help_text: str = "",
    default: bool = False,
) -> str:
    """Render a compact No/Yes pill control and return Yes/No for the pipeline."""
    label_col, control_col = st.columns([1.55, 1])
    with label_col:
        st.markdown(f'<p class="bool-field-label">{label}</p>', unsafe_allow_html=True)
        if help_text:
            st.markdown(f'<p class="bool-field-caption">{help_text}</p>', unsafe_allow_html=True)
    with control_col:
        choice = st.segmented_control(
            f"{key}_control",
            options=["No", "Yes"],
            default="Yes" if default else "No",
            key=key,
            label_visibility="collapsed",
            width="stretch",
        )
    return str(choice) if choice else "No"


def _addon_toggle(
    label: str,
    key: str,
    help_text: str = "",
    default: bool = False,
) -> str:
    """Render a compact service add-on No/Yes control."""
    label_col, control_col = st.columns([1.55, 1])
    with label_col:
        st.markdown(f'<p class="bool-field-label">{label}</p>', unsafe_allow_html=True)
        if help_text:
            st.markdown(f'<p class="bool-field-caption">{help_text}</p>', unsafe_allow_html=True)
    with control_col:
        choice = st.segmented_control(
            f"{key}_control",
            options=["No", "Yes"],
            default="Yes" if default else "No",
            key=key,
            label_visibility="collapsed",
            width="stretch",
        )
    return str(choice) if choice else "No"


def _render_currency_hint(amount: float) -> None:
    """Show formatted currency below a numeric input."""
    st.markdown(
        f'<p class="field-format-hint">${amount:,.2f}</p>',
        unsafe_allow_html=True,
    )


def _render_tenure_hint(months: int) -> None:
    """Show formatted tenure below the slider."""
    label = "Month" if months == 1 else "Months"
    st.markdown(
        f'<p class="field-format-hint">{months} {label}</p>',
        unsafe_allow_html=True,
    )


def render_customer_form() -> Dict[str, Any]:
    """
    Render the customer information form and collect user input.

    Returns:
        Dictionary of customer feature values aligned with the prediction pipeline.
    """
    st.markdown('<div class="prediction-form-active"></div>', unsafe_allow_html=True)

    with st.expander("👤  Customer Profile", expanded=True):
        st.markdown(
            '<p class="form-section-caption">Personal and household information</p>',
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox(
                "Gender",
                ["Female", "Male"],
                help="Customer gender as recorded in the account profile.",
            )
            partner = _yes_no_toggle(
                "Partner",
                "form_partner",
                help_text="Customer has a partner on the account.",
            )
        with col2:
            senior_citizen = _yes_no_toggle(
                "Senior Citizen",
                "form_senior",
                help_text="Customer is classified as a senior citizen (65+).",
            )
            dependents = _yes_no_toggle(
                "Dependents",
                "form_dependents",
                help_text="Customer has dependents on the account.",
            )

    with st.expander("📄  Contract Details", expanded=True):
        st.markdown(
            '<p class="form-section-caption">Billing, contract, and payment terms</p>',
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        with col1:
            contract = st.selectbox(
                "Contract",
                CONTRACT_OPTIONS,
                help="Length and type of service agreement.",
            )
            paperless_billing = _yes_no_toggle(
                "Paperless Billing",
                "form_paperless",
                help_text="Customer receives bills electronically.",
            )
            tenure = st.slider(
                "Tenure",
                min_value=0,
                max_value=72,
                value=12,
                format="%d Months",
                help="Total months the customer has been with the company.",
            )
            _render_tenure_hint(tenure)
        with col2:
            payment_method = st.selectbox(
                "Payment Method",
                PAYMENT_METHOD_OPTIONS,
                help="Primary method used for monthly payments.",
            )
            monthly_charges = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                max_value=200.0,
                value=70.0,
                step=0.5,
                format="%.2f",
                help="Average monthly bill amount.",
            )
            _render_currency_hint(monthly_charges)
            total_charges = st.number_input(
                "Total Charges",
                min_value=0.0,
                max_value=10000.0,
                value=840.0,
                step=1.0,
                format="%.2f",
                help="Lifetime charges billed to the customer.",
            )
            _render_currency_hint(total_charges)

    with st.expander("🌐  Internet Services", expanded=True):
        st.markdown(
            '<p class="form-section-caption">Phone and internet connectivity options</p>',
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        with col1:
            internet_service = st.selectbox(
                "Internet Service",
                INTERNET_SERVICE_OPTIONS,
                help="Type of internet connection, if any.",
            )
            phone_service = _yes_no_toggle(
                "Phone Service",
                "form_phone",
                help_text="Customer subscribes to home phone service.",
            )
        with col2:
            if phone_service == "No":
                st.markdown(
                    '<p class="field-na-hint">Multiple Lines — not applicable without phone service</p>',
                    unsafe_allow_html=True,
                )
                multiple_lines = "No phone service"
            else:
                multiple_lines = _yes_no_toggle(
                    "Multiple Lines",
                    "form_multiple_lines",
                    help_text="Customer has more than one phone line.",
                )

    with st.expander("🛡  Security Services", expanded=False):
        st.markdown(
            '<p class="form-section-caption">Protection and support add-ons for connected customers</p>',
            unsafe_allow_html=True,
        )
        if internet_service == "No":
            st.markdown(
                '<p class="field-na-hint">Security add-ons are not available without an internet subscription.</p>',
                unsafe_allow_html=True,
            )
            online_security = "No internet service"
            online_backup = "No internet service"
            device_protection = "No internet service"
            tech_support = "No internet service"
        else:
            col1, col2 = st.columns(2)
            with col1:
                online_security = _addon_toggle(
                    "Online Security",
                    "form_online_security",
                    help_text="Malware and virus protection for internet users.",
                )
                online_backup = _addon_toggle(
                    "Online Backup",
                    "form_online_backup",
                    help_text="Cloud backup for customer files and data.",
                )
            with col2:
                device_protection = _addon_toggle(
                    "Device Protection",
                    "form_device_protection",
                    help_text="Technical support for connected devices.",
                )
                tech_support = _addon_toggle(
                    "Tech Support",
                    "form_tech_support",
                    help_text="Priority technical assistance for service issues.",
                )

    with st.expander("🎬  Entertainment", expanded=False):
        st.markdown(
            '<p class="form-section-caption">Streaming and entertainment package options</p>',
            unsafe_allow_html=True,
        )
        if internet_service == "No":
            st.markdown(
                '<p class="field-na-hint">Entertainment add-ons require an active internet subscription.</p>',
                unsafe_allow_html=True,
            )
            streaming_tv = "No internet service"
            streaming_movies = "No internet service"
        else:
            col1, col2 = st.columns(2)
            with col1:
                streaming_tv = _addon_toggle(
                    "Streaming TV",
                    "form_streaming_tv",
                    help_text="Television streaming package add-on.",
                )
            with col2:
                streaming_movies = _addon_toggle(
                    "Streaming Movies",
                    "form_streaming_movies",
                    help_text="Movie streaming package add-on.",
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


# ---------------------------------------------------------------------------
# Prediction explanation — business rules separated from UI rendering
# ---------------------------------------------------------------------------

def identify_risk_factors(customer_data: Dict[str, Any]) -> List[str]:
    """
    Identify applicable business risk factors for the current customer profile.

    Args:
        customer_data: Dictionary of customer feature values from the form.

    Returns:
        List of human-readable risk factor labels applicable to this customer.
    """
    factors: List[str] = []

    if customer_data["Contract"] == "Month-to-month":
        factors.append("Month-to-Month Contract")
    if customer_data["InternetService"] == "Fiber optic":
        factors.append("Fiber Optic Internet")
    if customer_data["PaymentMethod"] == "Electronic check":
        factors.append("Electronic Check Payment")
    if customer_data["SeniorCitizen"] == 1:
        factors.append("Senior Citizen")
    if float(customer_data["MonthlyCharges"]) >= HIGH_MONTHLY_CHARGE_THRESHOLD:
        factors.append("High Monthly Charges")
    if int(customer_data["tenure"]) < SHORT_TENURE_THRESHOLD:
        factors.append("Short Customer Tenure")
    if customer_data["PaperlessBilling"] == "Yes":
        factors.append("Paperless Billing")
    if (
        customer_data["InternetService"] != "No"
        and customer_data["TechSupport"] == "No"
    ):
        factors.append("No Tech Support")
    if (
        customer_data["InternetService"] != "No"
        and customer_data["OnlineSecurity"] == "No"
    ):
        factors.append("No Online Security")

    return factors


def identify_protective_factors(customer_data: Dict[str, Any]) -> List[str]:
    """
    Identify applicable protective factors for low-risk customer profiles.

    Args:
        customer_data: Dictionary of customer feature values from the form.

    Returns:
        List of human-readable protective factor labels applicable to this customer.
    """
    factors: List[str] = []

    if int(customer_data["tenure"]) >= LONG_TENURE_THRESHOLD:
        factors.append("Long Customer Tenure")
    if customer_data["Contract"] == "One year":
        factors.append("One Year Contract")
    if customer_data["Contract"] == "Two year":
        factors.append("Two Year Contract")
    if float(customer_data["MonthlyCharges"]) < LOW_MONTHLY_CHARGE_THRESHOLD:
        factors.append("Low Monthly Charges")
    if (
        customer_data["InternetService"] != "No"
        and customer_data["TechSupport"] == "Yes"
    ):
        factors.append("Tech Support Enabled")
    if (
        customer_data["InternetService"] != "No"
        and customer_data["OnlineSecurity"] == "Yes"
    ):
        factors.append("Online Security Enabled")
    if customer_data["PaymentMethod"] in AUTOMATIC_PAYMENT_METHODS:
        factors.append("Automatic Payment")

    return factors


def get_prediction_explanation_factors(
    customer_data: Dict[str, Any],
    prediction: str,
) -> List[str]:
    """
    Return explanation factors aligned with the model prediction outcome.

    High-risk predictions surface risk factors; low-risk predictions surface
    protective factors. When no rule matches, a fallback message is returned.

    Args:
        customer_data: Dictionary of customer feature values.
        prediction: Predicted label (``\"Churn\"`` or ``\"No Churn\"``).

    Returns:
        List of explanation factor labels to display to the user.
    """
    if prediction == "Churn":
        factors = identify_risk_factors(customer_data)
        if not factors:
            return ["Multiple moderate risk signals in the customer profile"]
        return factors

    factors = identify_protective_factors(customer_data)
    if not factors:
        return ["Stable customer profile with no major churn indicators"]
    return factors


def build_confidence_summary(result: Dict[str, Any]) -> Dict[str, float]:
    """
    Build confidence metrics from a prediction result produced by the model.

    Args:
        result: Output dictionary from ``predict_customer``.

    Returns:
        Dictionary containing confidence, churn probability, and stay probability.
    """
    churn_probability = float(result["churn_probability"])
    stay_probability = float(result["stay_probability"])
    return {
        "confidence": max(churn_probability, stay_probability),
        "churn_probability": churn_probability,
        "stay_probability": stay_probability,
    }


# ---------------------------------------------------------------------------
# Risk meter — business logic separated from UI rendering
# ---------------------------------------------------------------------------

def calculate_risk_level(churn_probability: float) -> str:
    """
    Map churn probability to an executive risk level label.

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


def calculate_risk_color(risk_level: str) -> str:
    """
    Return the hex color associated with a risk level.

    Args:
        risk_level: Risk level label.

    Returns:
        Hex color code for UI styling.
    """
    color_map = {
        "Very Low Risk": "#10b981",
        "Low Risk": "#86efac",
        "Moderate Risk": "#fbbf24",
        "High Risk": "#f97316",
        "Critical Risk": "#ef4444",
    }
    return color_map.get(risk_level, "#6b7280")


def calculate_risk_icon(risk_level: str) -> str:
    """Return the icon associated with a risk level."""
    icon_map = {
        "Very Low Risk": "🟢",
        "Low Risk": "🟢",
        "Moderate Risk": "🟡",
        "High Risk": "🟠",
        "Critical Risk": "🔴",
    }
    return icon_map.get(risk_level, "⚪")


def calculate_risk_description(risk_level: str) -> str:
    """
    Return a short business explanation for the assigned risk level.

    Args:
        risk_level: Risk level label.

    Returns:
        Human-readable risk description.
    """
    descriptions = {
        "Very Low Risk": "This customer shows strong retention characteristics.",
        "Low Risk": "Customer appears stable with only minor churn indicators.",
        "Moderate Risk": "Customer should be monitored for behavioral changes.",
        "High Risk": "Customer should receive proactive retention offers.",
        "Critical Risk": "Immediate intervention is recommended to reduce churn probability.",
    }
    return descriptions.get(risk_level, "Review this customer profile for retention actions.")


def build_risk_profile(churn_probability: float) -> Dict[str, Any]:
    """
    Build a complete risk profile from churn probability.

    Args:
        churn_probability: Model-predicted churn probability.

    Returns:
        Dictionary containing level, score, percentage, color, icon, and description.
    """
    risk_level = calculate_risk_level(churn_probability)
    return {
        "level": risk_level,
        "score": float(churn_probability),
        "percentage": float(churn_probability) * 100,
        "color": calculate_risk_color(risk_level),
        "icon": calculate_risk_icon(risk_level),
        "description": calculate_risk_description(risk_level),
    }


def create_risk_gauge_figure(risk_profile: Dict[str, Any]) -> go.Figure:
    """
    Build a Plotly indicator gauge for the customer risk score.

    Args:
        risk_profile: Risk profile from ``build_risk_profile``.

    Returns:
        Plotly figure configured as an executive-style gauge.
    """
    percentage = risk_profile["percentage"]
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=percentage,
            number={"suffix": "%", "font": {"size": 28, "color": "#111827"}},
            title={"text": "Churn Risk Score", "font": {"size": 16, "color": "#374151"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#9ca3af"},
                "bar": {"color": risk_profile["color"], "thickness": 0.28},
                "bgcolor": "#f9fafb",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 20], "color": "#d1fae5"},
                    {"range": [20, 40], "color": "#ecfdf5"},
                    {"range": [40, 60], "color": "#fef3c7"},
                    {"range": [60, 80], "color": "#ffedd5"},
                    {"range": [80, 100], "color": "#fee2e2"},
                ],
            },
        )
    )
    figure.update_layout(
        height=260,
        margin={"l": 40, "r": 40, "t": 56, "b": 24},
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font={"color": "#1F2937", "family": "Inter, sans-serif"},
    )
    return figure


def render_risk_meter(churn_probability: float) -> None:
    """
    Render the executive-style customer risk meter.

    Args:
        churn_probability: Model-predicted churn probability.
    """
    risk_profile = build_risk_profile(churn_probability)
    marker_position = max(0.0, min(risk_profile["percentage"], 100.0))

    render_section_header("📊", "Customer Risk Meter", "Visual risk score based on predicted churn probability.")
    st.markdown(
        f"""
        <div class="risk-meter-card">
            <p class="risk-meter-title">Risk Level</p>
            <p class="risk-meter-level" style="color: {risk_profile["color"]};">
                {risk_profile["icon"]} {risk_profile["level"]}
            </p>
            <p class="risk-meter-score-label">Risk Score</p>
            <p class="risk-meter-score-value" style="color: {risk_profile["color"]};">
                {risk_profile["percentage"]:.0f}%
            </p>
            <div class="risk-meter-labels">
                <span>Very Low</span><span>Low</span><span>Moderate</span>
                <span>High</span><span>Critical</span>
            </div>
            <div class="risk-meter-track">
                <div class="risk-meter-marker" style="left: {marker_position:.1f}%;"></div>
            </div>
            <p class="risk-meter-description">{risk_profile["description"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_plotly_chart(
        create_risk_gauge_figure(risk_profile),
        height=260,
        hide_title=True,
        key="risk_gauge_chart",
    )


def render_confidence_box(
    prediction: str,
    confidence_summary: Dict[str, float],
) -> None:
    """
    Render a styled confidence summary box below the prediction result.

    Args:
        prediction: Predicted label (``\"Churn\"`` or ``\"No Churn\"``).
        confidence_summary: Confidence metrics from ``build_confidence_summary``.
    """
    is_high_risk = prediction == "Churn"
    box_class = "confidence-box-high" if is_high_risk else "confidence-box-low"
    fill_class = "confidence-bar-fill-high" if is_high_risk else "confidence-bar-fill-low"
    confidence_pct = confidence_summary["confidence"] * 100
    bar_width = max(0, min(confidence_pct, 100))

    render_section_header("🎯", "Prediction Confidence", "Model certainty and probability breakdown.")
    st.markdown(
        f"""
        <div class="confidence-box {box_class}">
            <p class="confidence-stat"><strong>Confidence</strong></p>
            <p class="confidence-value-large">{confidence_pct:.1f}%</p>
            <div class="confidence-bar-track">
                <div class="{fill_class}" style="width: {bar_width:.1f}%;"></div>
            </div>
            <p class="confidence-stat"><strong>Churn Probability:</strong>
            {confidence_summary["churn_probability"]:.1%}</p>
            <p class="confidence-stat"><strong>Stay Probability:</strong>
            {confidence_summary["stay_probability"]:.1%}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_explanation(
    customer_data: Dict[str, Any],
    prediction: str,
) -> None:
    """
    Render the post-prediction explanation section.

    Args:
        customer_data: Dictionary of customer feature values.
        prediction: Predicted label from the trained model.
    """
    explanation_factors = get_prediction_explanation_factors(customer_data, prediction)
    is_high_risk = prediction == "Churn"
    card_class = "explain-card-high" if is_high_risk else "explain-card-low"
    heading = "Main Reasons" if is_high_risk else "Reasons"
    icon = "🔴" if is_high_risk else "🟢"
    customer_label = "High Risk Customer" if is_high_risk else "Low Risk Customer"

    render_section_header(
        "🔍",
        "Why was this prediction made?",
        "Key factors influencing the model outcome.",
    )
    factors_html = "".join(
        f'<p class="explain-factor-item">✔ {factor}</p>' for factor in explanation_factors
    )
    render_html_card(
        card_class,
        f"<p><strong>{icon} {customer_label}</strong></p>"
        f"<p><strong>{heading}</strong></p>{factors_html}",
    )


# ---------------------------------------------------------------------------
# AI retention recommendations — business logic separated from UI rendering
# ---------------------------------------------------------------------------

PRIORITY_ORDER: Dict[str, int] = {"High": 0, "Medium": 1, "Low": 2}
PRIORITY_BADGE_CLASS: Dict[str, str] = {
    "High": "priority-badge-high",
    "Medium": "priority-badge-medium",
    "Low": "priority-badge-low",
}
PRIORITY_BADGE_LABEL: Dict[str, str] = {
    "High": "🔥 HIGH",
    "Medium": "🟠 MEDIUM",
    "Low": "🔵 LOW",
}


def _adjust_recommendation_priority(
    base_priority: str,
    prediction: str,
    risk_level: str,
) -> str:
    """
    Elevate or reduce recommendation priority based on prediction and risk.

    Args:
        base_priority: Initial priority assigned by the business rule.
        prediction: Model prediction label.
        risk_level: Risk level from the risk meter.

    Returns:
        Adjusted priority label.
    """
    priority = base_priority

    if risk_level in {"Critical Risk", "High Risk"}:
        if priority == "Low":
            priority = "Medium"
        elif priority == "Medium":
            priority = "High"

    if prediction == "No Churn" and priority == "High":
        priority = "Medium"
    elif prediction == "No Churn" and risk_level in {"Very Low Risk", "Low Risk"}:
        if priority == "Medium":
            priority = "Low"

    return priority


def generate_retention_recommendations(
    customer_data: Dict[str, Any],
    prediction: str,
    risk_level: str,
) -> List[Dict[str, str]]:
    """
    Generate personalized retention recommendations from profile and risk context.

    Args:
        customer_data: Customer feature dictionary from the prediction form.
        prediction: Model prediction label.
        risk_level: Assigned customer risk level.

    Returns:
        Sorted list of recommendation dictionaries.
    """
    recommendations: List[Dict[str, str]] = []

    if customer_data["Contract"] == "Month-to-month":
        recommendations.append(
            {
                "priority": _adjust_recommendation_priority("High", prediction, risk_level),
                "title": "Offer a 12-month or 24-month contract with a discount.",
                "reason": "Customer is on a Month-to-Month contract.",
                "expected_impact": "Reduce churn probability.",
            }
        )

    if customer_data["PaymentMethod"] == "Electronic check":
        recommendations.append(
            {
                "priority": _adjust_recommendation_priority("Medium", prediction, risk_level),
                "title": "Encourage Auto Pay using Credit Card or Bank Transfer.",
                "reason": "Electronic Check customers show higher churn.",
                "expected_impact": "Improve payment retention.",
            }
        )

    if (
        customer_data["InternetService"] != "No"
        and customer_data["TechSupport"] == "No"
    ):
        recommendations.append(
            {
                "priority": _adjust_recommendation_priority("Medium", prediction, risk_level),
                "title": "Offer free Tech Support for three months.",
                "reason": "Customer does not currently have Tech Support.",
                "expected_impact": "Increase service satisfaction and issue resolution.",
            }
        )

    if (
        customer_data["InternetService"] != "No"
        and customer_data["OnlineSecurity"] == "No"
    ):
        recommendations.append(
            {
                "priority": _adjust_recommendation_priority("Medium", prediction, risk_level),
                "title": "Offer an Online Security bundle.",
                "reason": "Customer lacks Online Security protection.",
                "expected_impact": "Strengthen product stickiness and perceived value.",
            }
        )

    if int(customer_data["tenure"]) < SHORT_TENURE_THRESHOLD:
        recommendations.append(
            {
                "priority": _adjust_recommendation_priority("Medium", prediction, risk_level),
                "title": "Provide a loyalty welcome offer.",
                "reason": "Customer tenure is under 12 months.",
                "expected_impact": "Build early loyalty during the onboarding phase.",
            }
        )

    if float(customer_data["MonthlyCharges"]) > 80:
        recommendations.append(
            {
                "priority": _adjust_recommendation_priority("High", prediction, risk_level),
                "title": "Offer a personalized discount or lower-cost plan.",
                "reason": "Monthly charges exceed $80.",
                "expected_impact": "Reduce price sensitivity and improve retention.",
            }
        )

    if customer_data["SeniorCitizen"] == 1:
        recommendations.append(
            {
                "priority": _adjust_recommendation_priority("Medium", prediction, risk_level),
                "title": "Provide priority customer support.",
                "reason": "Customer is a senior citizen.",
                "expected_impact": "Improve service experience and trust.",
            }
        )

    if not recommendations:
        recommendations.append(
            {
                "priority": "Low",
                "title": "Maintain proactive customer engagement.",
                "reason": "No major retention triggers detected in the current profile.",
                "expected_impact": "Sustain customer satisfaction and long-term loyalty.",
            }
        )

    return sorted(recommendations, key=lambda item: PRIORITY_ORDER[item["priority"]])


def calculate_business_impact(
    customer_data: Dict[str, Any],
    churn_probability: float,
    risk_level: str,
    recommendations: List[Dict[str, str]],
) -> Dict[str, Any]:
    """
    Estimate retention improvement and revenue protection for recommendations.

    Args:
        customer_data: Customer feature dictionary.
        churn_probability: Model-predicted churn probability.
        risk_level: Assigned customer risk level.
        recommendations: Generated recommendation list.

    Returns:
        Dictionary with retention improvement label and revenue protection value.
    """
    monthly_charges = float(customer_data["MonthlyCharges"])
    high_priority_count = sum(1 for item in recommendations if item["priority"] == "High")

    if risk_level in {"Critical Risk", "High Risk"} or high_priority_count >= 2:
        retention_improvement = "High"
    elif risk_level == "Moderate Risk" or high_priority_count == 1:
        retention_improvement = "Medium"
    else:
        retention_improvement = "Low"

    revenue_protection = monthly_charges * churn_probability

    return {
        "retention_improvement": retention_improvement,
        "revenue_protection": revenue_protection,
    }


def render_recommendation_card(recommendation: Dict[str, str]) -> None:
    """Render a single retention recommendation card."""
    priority = recommendation["priority"]
    badge_class = PRIORITY_BADGE_CLASS[priority]
    badge_label = PRIORITY_BADGE_LABEL[priority]
    card_class = {
        "High": "retention-card retention-card-high",
        "Medium": "retention-card retention-card-medium",
        "Low": "retention-card retention-card-low",
    }.get(priority, "retention-card")

    st.markdown(
        f"""
        <div class="{card_class}">
            <span class="{badge_class}">{badge_label}</span>
            <p class="retention-title">{recommendation["title"]}</p>
            <p class="retention-meta"><strong>Reason:</strong> {recommendation["reason"]}</p>
            <p class="retention-meta"><strong>Expected Impact:</strong> {recommendation["expected_impact"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_recommendation_summary(recommendations: List[Dict[str, str]]) -> None:
    """Render the recommended actions summary box."""
    high_count = sum(1 for item in recommendations if item["priority"] == "High")
    medium_count = sum(1 for item in recommendations if item["priority"] == "Medium")
    low_count = sum(1 for item in recommendations if item["priority"] == "Low")

    render_html_card(
        "retention-summary-card",
        "<p><strong>Recommended Actions</strong></p>"
        f"<p>- <strong>High Priority Actions:</strong> {high_count}</p>"
        f"<p>- <strong>Medium Priority Actions:</strong> {medium_count}</p>"
        f"<p>- <strong>Low Priority Actions:</strong> {low_count}</p>",
    )


def render_business_impact_box(business_impact: Dict[str, Any]) -> None:
    """Render estimated business impact metrics."""
    render_html_card(
        "retention-impact-card",
        "<p><strong>Business Impact</strong></p>"
        f'<p class="retention-meta"><strong>Estimated Retention Improvement:</strong> '
        f'↑ {business_impact["retention_improvement"]}</p>'
        f'<p class="retention-meta"><strong>Estimated Revenue Protection:</strong> '
        f'${business_impact["revenue_protection"]:,.2f} / month</p>',
    )


def render_retention_recommendations(
    customer_data: Dict[str, Any],
    prediction: str,
    churn_probability: float,
) -> None:
    """
    Render the AI-powered retention recommendations section.

    Args:
        customer_data: Customer feature dictionary.
        prediction: Model prediction label.
        churn_probability: Model-predicted churn probability.
    """
    risk_profile = build_risk_profile(churn_probability)
    recommendations = generate_retention_recommendations(
        customer_data,
        prediction,
        risk_profile["level"],
    )
    business_impact = calculate_business_impact(
        customer_data,
        churn_probability,
        risk_profile["level"],
        recommendations,
    )

    render_section_header(
        "🎯",
        "AI Retention Recommendations",
        "Prioritized actions to improve customer retention.",
    )
    for recommendation in recommendations:
        render_recommendation_card(recommendation)

    render_recommendation_summary(recommendations)
    render_business_impact_box(business_impact)


def create_pdf_download_button(
    pdf_bytes: bytes,
    generated_at: datetime,
) -> None:
    """
    Render the Streamlit download button for a generated prediction PDF.

    Args:
        pdf_bytes: PDF document bytes from ``generate_prediction_pdf``.
        generated_at: Timestamp used for the download filename.
    """
    file_name = f"churn_prediction_report_{generated_at.strftime('%Y%m%d_%H%M%S')}.pdf"
    st.download_button(
        label="📥 Download PDF Report",
        data=pdf_bytes,
        file_name=file_name,
        mime="application/pdf",
        type="primary",
        width="stretch",
        key="download_pdf_report",
    )


def render_pdf_report_section(
    customer_data: Dict[str, Any],
    result: Dict[str, Any],
    prediction: str,
) -> None:
    """
    Render the PDF report download section below retention recommendations.

    Args:
        customer_data: Customer feature dictionary.
        result: Prediction result from ``predict_customer``.
        prediction: Model prediction label.
    """
    confidence_summary = build_confidence_summary(result)
    churn_probability = confidence_summary["churn_probability"]
    risk_profile = build_risk_profile(churn_probability)
    recommendations = generate_retention_recommendations(
        customer_data,
        prediction,
        risk_profile["level"],
    )
    business_impact = calculate_business_impact(
        customer_data,
        churn_probability,
        risk_profile["level"],
        recommendations,
    )
    explanation_factors = get_prediction_explanation_factors(customer_data, prediction)
    generated_at = datetime.now()

    report_data = build_prediction_summary(
        customer_data=customer_data,
        prediction=prediction,
        confidence=confidence_summary["confidence"],
        churn_probability=confidence_summary["churn_probability"],
        stay_probability=confidence_summary["stay_probability"],
        risk_level=risk_profile["level"],
        explanation_factors=explanation_factors,
        recommendations=recommendations,
        business_impact=business_impact,
        generated_at=generated_at,
    )

    try:
        pdf_bytes = generate_prediction_pdf(report_data)
    except Exception as exc:
        render_section_header("📄", "Download Prediction Report", "Export a professional PDF summary.")
        st.error("Unable to generate the PDF report.")
        st.warning(str(exc))
        return

    st.markdown("---")
    render_section_header(
        "📄",
        "Download Prediction Report",
        "Download a professional PDF summarizing prediction, explanation factors, retention recommendations, and business impact.",
    )
    create_pdf_download_button(pdf_bytes, generated_at)


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
    risk_factor_labels = identify_risk_factors(customer_data)
    risk_factors = [
        f"{label} increases likelihood of churn based on historical customer patterns."
        for label in risk_factor_labels
    ]
    actions: List[str] = []

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


def record_prediction_history(
    result: Dict[str, Any],
    customer_data: Dict[str, Any],
) -> None:
    """
    Persist a successful single-customer prediction to session history.

    Args:
        result: Prediction result from ``predict_customer``.
        customer_data: Original customer input dictionary.
    """
    if PREDICTION_HISTORY_KEY not in st.session_state:
        st.session_state[PREDICTION_HISTORY_KEY] = initialize_prediction_history(None)

    prediction = str(result["prediction"])
    confidence_summary = build_confidence_summary(result)
    churn_probability = confidence_summary["churn_probability"]
    risk_profile = build_risk_profile(churn_probability)
    recommendations = generate_retention_recommendations(
        customer_data,
        prediction,
        risk_profile["level"],
    )
    business_impact = calculate_business_impact(
        customer_data,
        churn_probability,
        risk_profile["level"],
        recommendations,
    )
    explanation_factors = get_prediction_explanation_factors(customer_data, prediction)

    record = build_history_record(
        customer_data=customer_data,
        result=result,
        confidence=confidence_summary["confidence"],
        churn_probability=confidence_summary["churn_probability"],
        stay_probability=confidence_summary["stay_probability"],
        risk_level=risk_profile["level"],
        recommendations=recommendations,
        explanation_factors=explanation_factors,
        business_impact=business_impact,
    )
    st.session_state[PREDICTION_HISTORY_KEY] = save_prediction_history(
        st.session_state[PREDICTION_HISTORY_KEY],
        record,
    )


def render_prediction_result(result: Dict[str, Any], customer_data: Dict[str, Any]) -> None:
    """
    Render prediction output, confidence, and business insights.

    Args:
        result: Prediction result from ``predict_customer``.
        customer_data: Original customer input dictionary.
    """
    prediction = str(result["prediction"])
    confidence_summary = build_confidence_summary(result)

    record_prediction_history(result, customer_data)

    render_section_header("🎯", "Prediction Result", "Model outcome and recommended next steps.")

    if prediction == "Churn":
        st.markdown(
            f"""
            <div class="result-card-high">
                <h3>🔴 High Risk Customer</h3>
                <p class="metric-label">Probability of Churn</p>
                <p class="metric-value">{confidence_summary['churn_probability']:.1%}</p>
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
                <p class="metric-value">{confidence_summary['stay_probability']:.1%}</p>
                <p><strong>Recommended Action:</strong> Maintain engagement and customer satisfaction.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_confidence_box(prediction, confidence_summary)
    render_risk_meter(confidence_summary["churn_probability"])
    render_prediction_explanation(customer_data, prediction)
    render_retention_recommendations(
        customer_data,
        prediction,
        confidence_summary["churn_probability"],
    )
    render_pdf_report_section(customer_data, result, prediction)

    risk_factors, actions = generate_business_insights(customer_data, prediction)

    render_section_header("💡", "Business Insights", "Risk factors and recommended retention actions.")
    factors_html = "".join(f"<p>- {factor}</p>" for factor in risk_factors)
    actions_html = "".join(f"<p>- {action}</p>" for action in actions)
    render_html_card(
        "insight-card",
        "<p><strong>Why the customer may churn:</strong></p>"
        f"{factors_html}"
        "<p><strong>Business actions to reduce churn:</strong></p>"
        f"{actions_html}",
    )


def render_kpi_card(
    title: str,
    value: str,
    icon: str = "",
    description: str = "",
    trend: str = "",
    color_index: int = 0,
) -> None:
    """Render a single standardized KPI card."""
    render_premium_kpi_card(
        title,
        value,
        icon=icon,
        description=description,
        trend=trend,
        color_index=color_index,
    )


def render_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray) -> None:
    """Render a confusion matrix chart."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5), facecolor="#FFFFFF")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Churn", "Churn"],
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_facecolor("#FFFFFF")
    ax.set_title("Confusion Matrix", color="#1F2937")
    ax.set_xlabel("Predicted Label", color="#1F2937")
    ax.set_ylabel("Actual Label", color="#1F2937")
    ax.tick_params(colors="#6B7280")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_roc_curve(y_test: np.ndarray, y_proba: np.ndarray) -> float:
    """Render an ROC curve and return the AUC score."""
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5.5, 4.5), facecolor="#FFFFFF")
    ax.plot(fpr, tpr, color="#2563EB", linewidth=2, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#9CA3AF", linewidth=1)
    ax.set_facecolor("#FFFFFF")
    ax.set_title("ROC Curve", color="#1F2937")
    ax.set_xlabel("False Positive Rate", color="#1F2937")
    ax.set_ylabel("True Positive Rate", color="#1F2937")
    ax.tick_params(colors="#6B7280")
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

    fig, ax = plt.subplots(figsize=(7, 5), facecolor="#FFFFFF")
    ax.barh(importance_df["Feature"], importance_df["Importance"], color="#10B981")
    ax.set_facecolor("#FFFFFF")
    ax.set_title("Top 10 Feature Importances", color="#1F2937")
    ax.set_xlabel("Importance Score", color="#1F2937")
    ax.tick_params(colors="#6B7280")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def highlight_best_model_row(row: pd.Series, best_model_name: str) -> List[str]:
    """Apply highlight styling to the best model row."""
    if row["Model"] == best_model_name:
        return [
            "background-color: #DBEAFE; color: #1F2937; font-weight: 600;"
        ] * len(row)
    return ["color: #1F2937; background-color: #FFFFFF;"] * len(row)


def render_business_interpretation(best_model_name: str, best_metrics: Dict[str, float]) -> None:
    """Render business interpretation for the selected model."""
    render_section_header(
        "🧠",
        "Business Interpretation",
        "How model performance translates to retention strategy.",
    )
    render_html_card(
        "insight-card",
        f"""
        <p><strong>Why the selected model performed best:</strong><br/>
        <strong>{best_model_name}</strong> achieved the strongest ROC-AUC of
        <strong>{best_metrics['roc_auc']:.3f}</strong> and an F1 Score of
        <strong>{best_metrics['f1_score']:.3f}</strong>, indicating a strong balance between
        identifying churners correctly and limiting false alarms.</p>
        <p><strong>How businesses can use this model:</strong><br/>
        The model can score customers daily or weekly, rank them by churn probability, and help teams
        prioritize retention campaigns.</p>
        <p><strong>Benefits of predicting churn early:</strong><br/>
        Early churn prediction protects recurring revenue, reduces customer acquisition costs, and
        improves customer lifetime value.</p>
        """,
    )


# ---------------------------------------------------------------------------
# Executive Dashboard — business analytics powered by the raw Telco dataset
# ---------------------------------------------------------------------------

EXECUTIVE_DATA_PATH = Path("data") / "Telco-Customer-Churn.csv"

# Shared Plotly styling aligned with the enterprise design system.
PLOTLY_EXECUTIVE_LAYOUT: Dict[str, Any] = {
    "template": "plotly_white",
    "paper_bgcolor": "#FFFFFF",
    "plot_bgcolor": "#FFFFFF",
    "font": {"color": "#1F2937", "size": 13, "family": "Inter, sans-serif"},
    "margin": {"l": 56, "r": 28, "t": 72, "b": 56},
    "height": 380,
    "colorway": ["#2563EB", "#10B981", "#EF4444", "#F59E0B", "#8B5CF6"],
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


def _apply_plotly_layout(figure: Any, height: int = 400) -> Any:
    """Apply consistent executive dashboard styling to a Plotly figure."""
    return apply_enterprise_plotly_theme(figure, height=height)


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


def render_executive_kpi_card(icon: str, title: str, value: str, color_index: int = 0) -> None:
    """Render a KPI card with icon for the executive dashboard."""
    render_premium_kpi_card(title, value, icon=icon, color_index=color_index)


def render_business_summary_panel(summary: Dict[str, Any]) -> None:
    """
    Render the automated business summary insight panel.

    Args:
        summary: Business summary dictionary produced by ``compute_business_summary``.
    """
    render_section_header("📋", "Business Summary", "Key portfolio insights and revenue exposure.")

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

    lines_html = "".join(
        f'<p class="executive-summary-item">{line}</p>' for line in summary_lines
    )
    render_html_card("executive-summary-card", lines_html)


def render_executive_chart(
    figure: Any,
    title: str = "",
    description: str = "",
) -> None:
    """Render a Plotly chart with enterprise card styling."""
    chart_title = title
    if not chart_title and hasattr(figure, "layout") and figure.layout.title:
        chart_title = str(figure.layout.title.text or "")
    open_chart_card(chart_title, description)
    chart_height = getattr(figure.layout, "height", None) or 380
    render_plotly_chart(figure, height=int(chart_height), hide_title=bool(chart_title))
    close_chart_card()


# ---------------------------------------------------------------------------
# Explainable AI — model feature importance and business interpretation
# ---------------------------------------------------------------------------

FEATURE_BUSINESS_EXPLANATIONS: Dict[str, str] = {
    "MonthlyCharges": (
        "**High Monthly Charges** → Customers paying higher bills tend to churn more frequently."
    ),
    "tenure": (
        "**Short Tenure** → New customers are less loyal and more likely to leave early."
    ),
    "Contract": (
        "**Month-to-Month Contract** → Customers without long-term contracts have higher churn."
    ),
    "PaymentMethod": (
        "**Electronic Check** → Customers using electronic check historically churn more."
    ),
    "InternetService": (
        "**Fiber Optic** → Fiber customers show higher churn in this dataset."
    ),
    "OnlineSecurity": (
        "**No Online Security** → Missing security add-ons can increase dissatisfaction."
    ),
    "TechSupport": (
        "**No Tech Support** → Customers without support access may churn after service issues."
    ),
    "PaperlessBilling": (
        "**Paperless Billing** → Paperless billing profiles can correlate with churn behavior."
    ),
    "SeniorCitizen": (
        "**Senior Citizen Status** → Senior customer segments may show distinct churn patterns."
    ),
    "gender": (
        "**Gender** → Gender can contribute modestly to churn segmentation in this model."
    ),
    "Partner": (
        "**Partner Status** → Household composition influences customer stability."
    ),
    "Dependents": (
        "**Dependents** → Family household structure affects switching behavior."
    ),
    "TotalCharges": (
        "**Total Charges** → Lifetime billing volume can signal loyalty or dissatisfaction."
    ),
    "PhoneService": (
        "**Phone Service** → Core service subscription patterns influence retention."
    ),
    "MultipleLines": (
        "**Multiple Lines** → Additional phone services affect overall customer value."
    ),
}


@st.cache_data(show_spinner="Loading feature importance...")
def load_feature_importance() -> Dict[str, Any]:
    """
    Load feature importance scores from the saved model without retraining.

    Supports tree-based ``feature_importances_`` and Logistic Regression
    absolute coefficient values.

    Returns:
        Dictionary with availability flag, method, model type, and importance table.
    """
    try:
        model, _ = load_model()
    except Exception as exc:
        return {
            "available": False,
            "method": None,
            "model_type": None,
            "dataframe": pd.DataFrame(),
            "error": str(exc),
        }

    if not hasattr(model, "feature_names_in_"):
        return {
            "available": False,
            "method": None,
            "model_type": type(model).__name__,
            "dataframe": pd.DataFrame(),
            "error": "Model does not expose feature names.",
        }

    feature_names = list(model.feature_names_in_)

    if hasattr(model, "feature_importances_"):
        scores = np.asarray(model.feature_importances_, dtype=float)
        method = "feature_importances"
    elif hasattr(model, "coef_"):
        coefficients = np.asarray(model.coef_, dtype=float).reshape(-1)
        if coefficients.shape[0] != len(feature_names):
            return {
                "available": False,
                "method": None,
                "model_type": type(model).__name__,
                "dataframe": pd.DataFrame(),
                "error": "Coefficient shape does not match feature names.",
            }
        scores = np.abs(coefficients)
        method = "coefficients"
    else:
        return {
            "available": False,
            "method": None,
            "model_type": type(model).__name__,
            "dataframe": pd.DataFrame(),
            "error": "Feature importance is not supported for this model type.",
        }

    importance_df = (
        pd.DataFrame(
            {
                "Feature Name": feature_names,
                "Importance Score": scores,
            }
        )
        .sort_values("Importance Score", ascending=False)
        .reset_index(drop=True)
    )
    importance_df["Rank"] = importance_df.index + 1
    importance_df["Importance Score"] = importance_df["Importance Score"].round(6)

    return {
        "available": True,
        "method": method,
        "model_type": type(model).__name__,
        "dataframe": importance_df,
        "error": None,
    }


def build_feature_importance_table(importance_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the feature importance table for display.

    Args:
        importance_df: Full importance dataframe from ``load_feature_importance``.

    Returns:
        Display-ready dataframe with rank, feature name, and importance score.
    """
    return importance_df[["Rank", "Feature Name", "Importance Score"]].copy()


def create_feature_importance_chart(importance_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """
    Build a Plotly horizontal bar chart for the top N important features.

    Args:
        importance_df: Full importance dataframe.
        top_n: Number of top features to include.

    Returns:
        Plotly figure sorted from highest to lowest importance.
    """
    top_features = importance_df.head(top_n).sort_values("Importance Score", ascending=True)
    figure = px.bar(
        top_features,
        x="Importance Score",
        y="Feature Name",
        orientation="h",
        title=f"Top {top_n} Most Important Features",
        labels={"Importance Score": "Importance Score", "Feature Name": "Feature"},
        color="Importance Score",
        color_continuous_scale=["#93c5fd", "#2563eb"],
    )
    figure.update_layout(
        height=max(420, top_n * 28),
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        coloraxis_showscale=False,
    )
    return apply_enterprise_plotly_theme(figure, height=max(420, top_n * 28))


def generate_business_feature_explanations(
    importance_df: pd.DataFrame,
    top_n: int = 8,
) -> List[str]:
    """
    Generate business-friendly explanations for the most influential features.

    Args:
        importance_df: Full importance dataframe.
        top_n: Number of top features to interpret.

    Returns:
        List of business explanation strings.
    """
    explanations: List[str] = []
    seen: set[str] = set()

    for feature_name in importance_df.head(top_n)["Feature Name"]:
        if feature_name in FEATURE_BUSINESS_EXPLANATIONS:
            explanation = FEATURE_BUSINESS_EXPLANATIONS[feature_name]
            if explanation not in seen:
                explanations.append(explanation)
                seen.add(explanation)
        else:
            generic = (
                f"**{feature_name}** → This feature materially influences churn predictions "
                "in the trained model."
            )
            if generic not in seen:
                explanations.append(generic)
                seen.add(generic)

    return explanations


def render_explainable_ai_page(model_loaded: bool) -> None:
    """Render the Explainable AI page for model transparency."""
    render_page_header(
        "🧠",
        "Explainable AI",
        "Understand which customer features most influence churn predictions and what they "
        "mean in business terms — without retraining the model.",
    )

    if not model_loaded:
        render_empty_state(
            "🧠",
            "Explainable AI unavailable",
            "Trained model artifacts were not found. Run `python train_model.py` first to "
            "explore feature importance and business interpretations.",
        )
        return

    importance_payload = load_feature_importance()

    if not importance_payload["available"]:
        st.warning(
            importance_payload.get("error")
            or "Feature importance is not available for the selected model."
        )
        st.info(
            "This page supports tree-based models with `feature_importances_` and "
            "Logistic Regression models using absolute coefficient values."
        )
        return

    importance_df = importance_payload["dataframe"]
    method_label = (
        "Feature Importances"
        if importance_payload["method"] == "feature_importances"
        else "Absolute Coefficient Values"
    )

    render_section_header(
        "📊",
        "Feature Importance",
        f"Model: {MODEL_NAME_MAP.get(importance_payload['model_type'], importance_payload['model_type'])} · Method: {method_label}",
    )
    render_plotly_chart(
        create_feature_importance_chart(importance_df, top_n=15),
        height=max(420, 15 * 28),
    )

    render_section_header("📋", "Feature Importance Table", "Ranked features with importance scores.")
    st.dataframe(
        build_feature_importance_table(importance_df),
        width="stretch",
        hide_index=True,
    )

    render_section_header("💡", "Business Interpretation", "What the top features mean for retention strategy.")
    explanations_html = "".join(
        f'<p class="xai-explanation-item">{explanation}</p>'
        for explanation in generate_business_feature_explanations(importance_df)
    )
    render_html_card("insight-card", explanations_html)


CONFUSION_CELL_METADATA: Dict[str, tuple[str, str]] = {
    "TP": (
        "True Positive",
        "Correctly flagged customers who actually churned — enables timely retention outreach.",
    ),
    "TN": (
        "True Negative",
        "Correctly identified loyal customers — avoids unnecessary retention spend.",
    ),
    "FP": (
        "False Positive",
        "Predicted churn but the customer stayed — may trigger extra retention cost.",
    ),
    "FN": (
        "False Negative",
        "Missed a customer who churned — highest business risk and lost revenue opportunity.",
    ),
}


def compute_confusion_matrix_summary(
    y_test: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, int]:
    """
    Compute confusion matrix cell counts for binary churn classification.

    Args:
        y_test: Actual labels.
        y_pred: Predicted labels.

    Returns:
        Dictionary with TP, TN, FP, and FN counts.
    """
    matrix = confusion_matrix(y_test, y_pred)
    if matrix.shape != (2, 2):
        return {"TN": 0, "FP": 0, "FN": 0, "TP": 0}
    tn, fp, fn, tp = matrix.ravel()
    return {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}


def create_confusion_matrix_plotly(y_test: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """
    Build a Plotly heatmap for the confusion matrix.

    Args:
        y_test: Actual labels.
        y_pred: Predicted labels.

    Returns:
        Plotly figure for the confusion matrix.
    """
    matrix = confusion_matrix(y_test, y_pred)
    labels = ["No Churn", "Churn"]
    figure = px.imshow(
        matrix,
        text_auto=True,
        x=[f"Predicted {label}" for label in labels],
        y=[f"Actual {label}" for label in labels],
        color_continuous_scale=["#eff6ff", "#2563eb"],
        aspect="auto",
    )
    figure.update_traces(textfont={"size": 16, "color": "#1F2937"})
    figure.update_layout(
        title="Confusion Matrix",
        height=380,
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        coloraxis_showscale=False,
    )
    return apply_enterprise_plotly_theme(figure, height=380)


def create_roc_curve_plotly(y_test: np.ndarray, y_proba: np.ndarray) -> tuple[go.Figure, float]:
    """
    Build a Plotly ROC curve and return the figure with AUC score.

    Args:
        y_test: Actual labels.
        y_proba: Predicted positive-class probabilities.

    Returns:
        Tuple of Plotly figure and ROC-AUC score.
    """
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = float(auc(fpr, tpr))

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"ROC Curve (AUC = {roc_auc:.3f})",
            line={"color": "#2563eb", "width": 3},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Classifier",
            line={"color": "#9ca3af", "width": 1, "dash": "dash"},
        )
    )
    figure.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=380,
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    figure.update_xaxes(range=[0, 1])
    figure.update_yaxes(range=[0, 1])
    return apply_enterprise_plotly_theme(figure, height=380), roc_auc


def render_model_summary(
    evaluation: Dict[str, Any],
    best_model_name: str,
    algorithm: str,
) -> None:
    """
    Render a professional model summary card for the evaluation dashboard.

    Args:
        evaluation: Cached evaluation payload from ``load_evaluation_data``.
        best_model_name: Name of the best-performing model.
        algorithm: Human-readable algorithm label for the saved model.
    """
    training_date = evaluation.get("training_date") or "Not available"
    render_section_header("📋", "Model Summary", "Training configuration and dataset overview.")
    summary_items = [
        ("Model Name", best_model_name),
        ("Algorithm", algorithm),
        ("Training Date", training_date),
        ("Dataset Size", f"{evaluation['dataset_size']:,} customers"),
        ("Number of Features", str(evaluation["num_features"])),
        ("Prediction Classes", "No Churn (0), Churn (1)"),
    ]
    items_html = "".join(
        f'<p class="eval-summary-item"><strong>{label}:</strong> {value}</p>'
        for label, value in summary_items
    )
    render_html_card("eval-summary-card", items_html)


def render_metric_cards(best_metrics: Dict[str, float], evaluation: Dict[str, Any]) -> None:
    """
    Render KPI metric cards for model evaluation.

    Args:
        best_metrics: Metric dictionary for the best model.
        evaluation: Cached evaluation payload with sample counts.
    """
    render_section_header("📈", "Key Performance Indicators", "Core evaluation metrics on the hold-out test set.")
    row_one = st.columns(4)
    row_one_metrics = [
        ("🎯", "Accuracy", f"{best_metrics['accuracy']:.3f}", "Overall correct predictions"),
        ("🎯", "Precision", f"{best_metrics['precision']:.3f}", "Accuracy among predicted churners"),
        ("📈", "Recall", f"{best_metrics['recall']:.3f}", "Share of churners detected"),
        ("⚖️", "F1 Score", f"{best_metrics['f1_score']:.3f}", "Precision-recall balance"),
    ]
    for column, (index, (icon, title, value, description)) in zip(row_one, enumerate(row_one_metrics)):
        with column:
            render_kpi_card(title, value, icon=icon, description=description, color_index=index)

    row_two = st.columns(3)
    row_two_metrics = [
        ("📊", "ROC-AUC", f"{best_metrics['roc_auc']:.3f}", "Ranking discrimination strength"),
        ("🧪", "Training Samples", f"{evaluation['training_samples']:,}", "Rows used for training"),
        ("🧪", "Testing Samples", f"{evaluation['testing_samples']:,}", "Hold-out evaluation rows"),
    ]
    for column, (index, (icon, title, value, description)) in zip(row_two, enumerate(row_two_metrics)):
        with column:
            render_kpi_card(title, value, icon=icon, description=description, color_index=index + 4)


def render_confusion_matrix_summary(y_test: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Render confusion matrix heatmap and business-friendly cell explanations.

    Args:
        y_test: Actual labels.
        y_pred: Predicted labels.
    """
    cm_summary = compute_confusion_matrix_summary(y_test, y_pred)
    render_plotly_chart(
        create_confusion_matrix_plotly(y_test, y_pred),
        height=380,
        hide_title=True,
    )

    st.markdown("#### Confusion Matrix Breakdown")
    metric_columns = st.columns(2)
    cell_order = ["TP", "TN", "FP", "FN"]
    for index, cell_key in enumerate(cell_order):
        title, explanation = CONFUSION_CELL_METADATA[cell_key]
        with metric_columns[index % 2]:
            st.markdown(
                f"""
                <div class="cm-metric-card">
                    <strong>{title}</strong> ({cell_key}): {cm_summary[cell_key]:,}<br/>
                    <span class="section-desc">{explanation}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_roc_curve_section(
    y_test: np.ndarray,
    y_proba: np.ndarray,
    roc_auc_score: float,
) -> None:
    """
    Render an enhanced ROC curve with AUC score and interpretation.

    Args:
        y_test: Actual labels.
        y_proba: Predicted positive-class probabilities.
        roc_auc_score: ROC-AUC score from evaluation metrics.
    """
    roc_figure, computed_auc = create_roc_curve_plotly(y_test, y_proba)
    render_plotly_chart(roc_figure, height=380, hide_title=True)
    st.markdown(
        f"**ROC-AUC Score:** {roc_auc_score:.3f} "
        f"(computed on hold-out test set: {computed_auc:.3f})"
    )
    if roc_auc_score >= 0.90:
        interpretation = (
            "ROC-AUC above **0.90** indicates **excellent discrimination** — the model "
            "separates churners from loyal customers very effectively."
        )
    elif roc_auc_score >= 0.80:
        interpretation = (
            "ROC-AUC above **0.80** indicates **strong discrimination** — the model "
            "reliably ranks at-risk customers higher than retained customers."
        )
    elif roc_auc_score >= 0.70:
        interpretation = (
            "ROC-AUC in the **0.70–0.80** range indicates **moderate discrimination** — "
            "useful for prioritization but may benefit from feature or threshold tuning."
        )
    else:
        interpretation = (
            "ROC-AUC below **0.70** suggests **limited separation** — review data quality "
            "and consider alternative models or features."
        )
    st.markdown(
        f'<p class="eval-summary-item">{interpretation} '
        "A higher curve above the diagonal means better true-positive detection at each "
        "false-positive rate.</p>",
        unsafe_allow_html=True,
    )


def generate_model_interpretation_observations(
    best_metrics: Dict[str, float],
    cm_summary: Dict[str, int],
) -> List[str]:
    """
    Generate automatic performance observations for the evaluation dashboard.

    Args:
        best_metrics: Metric dictionary for the best model.
        cm_summary: Confusion matrix cell counts.

    Returns:
        List of observation strings.
    """
    observations: List[str] = []
    recall = best_metrics["recall"]
    precision = best_metrics["precision"]
    roc_auc_score = best_metrics["roc_auc"]
    accuracy = best_metrics["accuracy"]
    f1_score = best_metrics["f1_score"]
    total_errors = cm_summary["FP"] + cm_summary["FN"]

    if recall >= 0.85:
        observations.append(
            "Excellent recall means most churn customers are detected before they leave."
        )
    elif recall >= 0.70:
        observations.append(
            "Solid recall — the model captures a majority of at-risk churners."
        )
    else:
        observations.append(
            "Recall could be improved — some churners may be missed (false negatives)."
        )

    if precision >= 0.80:
        observations.append(
            "High precision reduces unnecessary retention campaigns on loyal customers."
        )
    elif precision >= 0.65:
        observations.append(
            "Moderate precision — some retention outreach may target customers who would stay."
        )
    else:
        observations.append(
            "Lower precision may increase retention cost through false-positive alerts."
        )

    if roc_auc_score >= 0.90:
        observations.append(
            "ROC-AUC above 0.90 indicates excellent discrimination between churn and retention."
        )
    elif roc_auc_score >= 0.80:
        observations.append(
            "ROC-AUC above 0.80 indicates strong ranking ability for prioritizing outreach."
        )

    if f1_score >= 0.75:
        observations.append(
            f"F1 Score of {f1_score:.3f} reflects a healthy balance between precision and recall."
        )

    if accuracy >= 0.80:
        observations.append(
            f"Overall accuracy of {accuracy:.1%} supports reliable day-to-day scoring workflows."
        )

    if cm_summary["FN"] < cm_summary["FP"]:
        observations.append(
            "False negatives are lower than false positives — the model favors catching churners."
        )
    elif total_errors > 0:
        observations.append(
            f"The model produced {total_errors:,} total misclassifications on the test set."
        )

    return observations


def render_model_interpretation(
    best_metrics: Dict[str, float],
    cm_summary: Dict[str, int],
) -> None:
    """
    Render automatically generated model performance observations.

    Args:
        best_metrics: Metric dictionary for the best model.
        cm_summary: Confusion matrix cell counts.
    """
    render_section_header(
        "💡",
        "Model Performance Interpretation",
        "Automated observations from evaluation metrics.",
    )
    observations_html = "".join(
        f"<p>- {observation}</p>"
        for observation in generate_model_interpretation_observations(best_metrics, cm_summary)
    )
    render_html_card("insight-card", observations_html)


def _collect_model_strengths(
    best_metrics: Dict[str, float],
    cm_summary: Dict[str, int],
) -> List[str]:
    """Collect model strength bullet points based on evaluation results."""
    strengths: List[str] = []

    if best_metrics["accuracy"] >= 0.75:
        strengths.append("Detects churn accurately on the hold-out test set")
    if best_metrics["recall"] >= 0.70:
        strengths.append("Captures most actual churners (strong recall)")
    if cm_summary["FN"] <= cm_summary["FP"]:
        strengths.append("Low false negatives relative to false positives")
    if best_metrics["roc_auc"] >= 0.80:
        strengths.append("Strong ROC performance for ranking at-risk customers")
    if best_metrics["f1_score"] >= 0.70:
        strengths.append("Balanced precision–recall trade-off (F1 Score)")

    if not strengths:
        strengths = [
            "Provides a structured baseline for churn scoring",
            "Supports side-by-side comparison of multiple algorithms",
        ]

    return strengths


def render_model_strengths(
    best_metrics: Dict[str, float],
    cm_summary: Dict[str, int],
) -> None:
    """Render model strength bullet points based on evaluation results."""
    strengths = _collect_model_strengths(best_metrics, cm_summary)
    strengths_html = "".join(f"<p>✔ {strength}</p>" for strength in strengths)
    render_html_card("insight-card", strengths_html)


def _collect_model_limitations() -> List[str]:
    """Return model limitation bullet points."""
    return [
        "Trained on a single IBM Telco dataset — patterns may differ in other industries",
        "May require retraining for new business environments or product lines",
        "Performance depends on data quality, feature completeness, and label accuracy",
        "Threshold and campaign rules should be validated with business stakeholders",
    ]


def render_model_limitations() -> None:
    """Render model limitation bullet points for transparent ML reporting."""
    limitations_html = "".join(f"<p>• {limitation}</p>" for limitation in _collect_model_limitations())
    render_html_card("insight-card", limitations_html)


def render_feature_importance_preview() -> None:
    """Render a top-5 feature importance preview with navigation to Explainable AI."""
    render_section_header("🔍", "Feature Importance Preview", "Top five drivers with link to full analysis.")
    importance_payload = load_feature_importance()

    if not importance_payload["available"]:
        st.info(
            importance_payload.get("error")
            or "Feature importance preview is unavailable for the saved model."
        )
        st.caption("Visit **Explainable AI** in the sidebar when a compatible model is loaded.")
        return

    importance_df = importance_payload["dataframe"]
    render_plotly_chart(
        create_feature_importance_chart(importance_df, top_n=5),
        height=max(420, 5 * 28),
    )

    if st.button("View Full Explainable AI Analysis", key="eval_open_xai", type="primary"):
        st.session_state[NAVIGATION_KEY] = "🧠 Explainable AI"
        st.rerun()

    st.caption(
        "Explore the full ranked table, search features, and business interpretations "
        "on the Explainable AI page."
    )


@st.cache_data(show_spinner="Building executive business insights...")
def get_executive_insights_bundle(
    csv_path: str = str(EXECUTIVE_DATA_PATH),
) -> Dict[str, Any]:
    """
    Load and cache the full executive business insights payload.

    Args:
        csv_path: Path to the IBM Telco Customer Churn CSV.

    Returns:
        Insights bundle from ``compute_executive_insights``.
    """
    dataframe = load_insights_dataframe(csv_path)
    avg_churn_probability = compute_average_churn_probability(csv_path)
    return compute_executive_insights(dataframe, avg_churn_probability)


def render_risk_insight_card(risk_item: Dict[str, Any]) -> None:
    """Render a single top business risk insight card."""
    badge_type = "risk-high" if risk_item.get("risk_percentage", 0) >= 40 else "risk-low"
    st.markdown(
        f"""
        <div class="card">
            <p class="retention-title">{risk_item.get("label", "Business Risk")}</p>
            <p class="retention-meta"><strong>Value:</strong> {risk_item.get("value", "—")}</p>
            <p class="retention-meta">
                <strong>Risk:</strong>
                <span class="status-badge status-badge-{badge_type}">
                    {risk_item.get("risk_percentage", 0):.1f}%
                </span>
            </p>
            <p class="retention-meta">{risk_item.get("explanation", "")}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_retention_strategy_section(strategy: Dict[str, List[Dict[str, str]]]) -> None:
    """Render prioritized retention strategy recommendations."""
    priority_badges = {
        "High Priority": ("priority-high", "🔥 HIGH"),
        "Medium Priority": ("priority-medium", "🟠 MEDIUM"),
        "Low Priority": ("priority-low", "🔵 LOW"),
    }

    for priority_level, recommendations in strategy.items():
        badge_class, badge_label = priority_badges.get(
            priority_level,
            ("model", priority_level),
        )
        st.markdown(f"#### {priority_level}")
        for recommendation in recommendations:
            priority = recommendation.get("priority", "Low")
            card_class = {
                "High": "retention-card retention-card-high",
                "Medium": "retention-card retention-card-medium",
                "Low": "retention-card retention-card-low",
            }.get(priority, "retention-card")
            st.markdown(
                f"""
                <div class="{card_class}">
                    <span class="status-badge status-badge-{badge_class}">{badge_label}</span>
                    <p class="retention-title">{recommendation["action"]}</p>
                    <p class="retention-meta"><strong>Expected Business Impact:</strong>
                    {recommendation["impact"]}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_management_takeaways(takeaways: Dict[str, List[str]]) -> None:
    """Render management takeaway cards for strengths, risks, and actions."""
    sections = [
        ("Strengths", "success", takeaways.get("strengths", [])),
        ("Risks", "warning", takeaways.get("risks", [])),
        ("Immediate Actions", "priority-high", takeaways.get("immediate_actions", [])),
        ("Long-term Strategy", "model", takeaways.get("long_term_strategy", [])),
    ]

    col_left, col_right = st.columns(2)
    for index, (title, badge_type, items) in enumerate(sections):
        column = col_left if index % 2 == 0 else col_right
        with column:
            items_html = "".join(f"<p>- {item}</p>" for item in items)
            render_html_card(
                "insight-card",
                f"{render_status_badge_html(badge_type, title)}"
                f"<p><strong>{title}</strong></p>{items_html}",
            )


def render_executive_insights_page() -> None:
    """Render the executive business insights page for management decision support."""
    render_page_header(
        "📈",
        "Executive Business Insights",
        "Business-focused summary of churn trends, revenue exposure, retention priorities, "
        "and management actions — powered by the IBM Telco customer dataset.",
    )

    if not EXECUTIVE_DATA_PATH.exists():
        render_empty_state(
            "📈",
            "Executive insights unavailable",
            f"Dataset not found at `{EXECUTIVE_DATA_PATH}`. Add `Telco-Customer-Churn.csv` "
            "to the `data/` folder to enable business insights.",
        )
        return

    try:
        insights = get_executive_insights_bundle()
    except FileNotFoundError as exc:
        render_empty_state(
            "📈",
            "Executive insights unavailable",
            "The dataset file could not be found. Verify the Telco CSV is present in `data/`.",
        )
        st.warning(str(exc))
        return
    except Exception as exc:
        st.error("Unable to build executive business insights.")
        st.warning(str(exc))
        return

    kpis = insights["kpis"]

    render_section_header("📊", "Executive Summary", "Portfolio KPIs and high-level business overview.")
    kpi_columns = st.columns(6)
    kpi_values = [
        ("👥", "Total Customers", f"{kpis['total_customers']:,.0f}", "Active accounts analyzed"),
        ("📉", "Churn Customers", f"{kpis['churn_customers']:,.0f}", "Customers who churned"),
        ("⚠️", "Churn Rate", f"{kpis['churn_rate']:.2f}%", "Portfolio churn percentage"),
        ("💳", "Average Monthly Charges", f"${kpis['avg_monthly_charges']:.2f}", "Mean monthly ARPU"),
        ("⏳", "Average Tenure", f"{kpis['avg_tenure']:.1f} mo", "Mean customer lifetime"),
        ("💰", "Estimated Revenue at Risk", f"${kpis['revenue_at_risk']:,.2f}", "Monthly churn revenue"),
    ]
    for column, (index, (icon, title, value, description)) in zip(
        kpi_columns,
        enumerate(kpi_values),
    ):
        with column:
            render_kpi_card(
                title,
                value,
                icon=icon,
                description=description,
                color_index=index,
            )

    bullets_html = "".join(f"<p>- {bullet}</p>" for bullet in insights["summary_bullets"])
    render_html_card("executive-summary-card", f"<p><strong>Executive Summary</strong></p>{bullets_html}")

    st.markdown("---")
    render_section_header("⚠️", "Top Business Risks", "Highest-impact churn drivers requiring attention.")
    risk_columns = st.columns(2)
    for index, risk_item in enumerate(insights["top_risks"]):
        with risk_columns[index % 2]:
            render_risk_insight_card(risk_item)

    st.markdown("---")
    render_section_header("🎯", "Retention Strategy", "Recommended actions to reduce churn and protect revenue.")
    render_retention_strategy_section(insights["retention_strategy"])

    st.markdown("---")
    render_section_header("📉", "Executive Visuals", "Charts highlighting churn patterns and revenue exposure.")
    charts = insights["visuals"]
    chart_pairs = [
        (charts["revenue_at_risk_by_contract"], charts["churn_by_payment_method"]),
        (charts["churn_by_internet_service"], charts["monthly_charges_vs_churn"]),
    ]
    for left_chart, right_chart in chart_pairs:
        left_column, right_column = st.columns(2)
        with left_column:
            render_executive_chart(left_chart)
        with right_column:
            render_executive_chart(right_chart)
    render_executive_chart(charts["top_risk_segments"])

    st.markdown("---")
    render_section_header("✅", "Management Takeaways", "Key decisions and priorities for leadership.")
    render_management_takeaways(insights["management_takeaways"])


def render_executive_dashboard_page() -> None:
    render_page_header(
        "📊",
        "Executive Dashboard",
        "Executive-level view of customer churn, revenue exposure, and the highest-risk "
        "customer segments across the IBM Telco customer base.",
    )

    if not EXECUTIVE_DATA_PATH.exists():
        render_empty_state(
            "📊",
            "Dataset not found",
            "Add `Telco-Customer-Churn.csv` to the `data/` folder to load the executive dashboard.",
        )
        return

    try:
        dashboard_bundle = get_executive_dashboard_bundle()
        charts = build_executive_dashboard_charts()
        kpis = dashboard_bundle["kpis"]
        business_summary = dashboard_bundle["business_summary"]
    except FileNotFoundError:
        render_empty_state(
            "📊",
            "Dataset unavailable",
            "The executive dashboard could not find the dataset file. Verify the data path and try again.",
        )
        return
    except Exception as exc:
        render_empty_state(
            "📊",
            "Dashboard unavailable",
            f"Unable to build the executive dashboard. {exc}",
        )
        return

    # KPI cards in a single responsive row
    render_section_header("📈", "Key Business Metrics", "Portfolio-level KPIs across the customer base.")
    kpi_columns = st.columns(6)
    kpi_values = [
        ("👥", "Total Customers", f"{kpis['total_customers']:,.0f}"),
        ("📉", "Total Churn Customers", f"{kpis['churn_customers']:,.0f}"),
        ("⚠️", "Churn Rate (%)", f"{kpis['churn_rate']:.2f}%"),
        ("💳", "Average Monthly Charges", f"${kpis['avg_monthly_charges']:.2f}"),
        ("⏳", "Average Customer Tenure", f"{kpis['avg_tenure']:.1f} mo"),
        ("💰", "Monthly Revenue at Risk", f"${kpis['revenue_at_risk']:,.2f}"),
    ]

    for column, (index, (icon, title, value)) in zip(
        kpi_columns,
        enumerate(kpi_values),
    ):
        with column:
            render_executive_kpi_card(icon, title, value, color_index=index)

    render_business_summary_panel(business_summary)

    st.markdown("---")
    render_section_header("📉", "Churn Analytics", "Segment-level churn patterns and risk distribution.")

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
    render_page_header(
        "📊",
        "Model Evaluation",
        "Review model performance, compare algorithms, and understand how the selected model supports "
        "business decisions for customer retention.",
    )

    if not model_loaded:
        render_empty_state(
            "📈",
            "No evaluation available",
            "Model evaluation requires trained artifacts. Run `python train_model.py` after adding "
            "the dataset to enable performance metrics and comparison dashboards.",
        )
        return

    if not Path(DEFAULT_DATA_PATH).exists():
        render_empty_state(
            "📈",
            "No evaluation available",
            f"Dataset not found at `{DEFAULT_DATA_PATH}`. Add `Telco-Customer-Churn.csv` to the "
            "`data/` folder to view evaluation metrics.",
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
    cm_summary = compute_confusion_matrix_summary(
        evaluation["y_test"],
        evaluation["y_pred"],
    )
    saved_algorithm = MODEL_NAME_MAP.get(
        type(evaluation["saved_model"]).__name__,
        type(evaluation["saved_model"]).__name__,
    )

    render_model_summary(evaluation, best_model_name, saved_algorithm)
    render_metric_cards(best_metrics, evaluation)

    st.markdown("---")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        render_section_header("🎯", "Confusion Matrix", "Classification outcomes on the test set.")
        render_confusion_matrix_summary(evaluation["y_test"], evaluation["y_pred"])

    with chart_col2:
        render_section_header("📈", "ROC Curve", "Discrimination strength across probability thresholds.")
        render_roc_curve_section(
            evaluation["y_test"],
            evaluation["y_proba"],
            best_metrics["roc_auc"],
        )

    render_feature_importance_preview()

    render_model_interpretation(best_metrics, cm_summary)

    strengths_col, limitations_col = st.columns(2)
    with strengths_col:
        render_section_header("✅", "Model Strengths", "Where the model performs well.")
        render_model_strengths(best_metrics, cm_summary)
    with limitations_col:
        render_section_header("⚠️", "Model Limitations", "Constraints and considerations for deployment.")
        render_model_limitations()

    st.markdown("---")
    render_section_header("📊", "Feature Importance", "Top drivers of churn in the trained model.")
    render_feature_importance(
        evaluation["saved_model"],
        evaluation["feature_names"],
    )

    render_section_header("📋", "Model Comparison", "Side-by-side algorithm performance on the test set.")
    styled_table = comparison_table.style.apply(
        lambda row: highlight_best_model_row(row, best_model_name),
        axis=1,
    )
    st.dataframe(styled_table, width="stretch", hide_index=True)

    render_business_interpretation(best_model_name, best_metrics)


def render_prediction_page(model_loaded: bool, error_message: Optional[str]) -> None:
    """Render the churn prediction page."""
    inject_prediction_page_styles()
    render_landing_section()
    customer_data = render_customer_form()

    st.markdown('<div class="prediction-submit-marker"></div>', unsafe_allow_html=True)
    predict_clicked = st.button(
        "Predict Customer Churn",
        type="primary",
        width="stretch",
        key="prediction_submit_btn",
    )

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
    render_premium_footer()


@st.cache_data(show_spinner="Running batch predictions...")
def run_cached_batch_prediction(file_signature: str, file_bytes: bytes) -> pd.DataFrame:
    """
    Run batch predictions on uploaded CSV bytes with caching.

    Args:
        file_signature: Stable hash identifier for the uploaded file contents.
        file_bytes: Raw CSV file bytes.

    Returns:
        Batch prediction results dataframe.
    """
    _ = file_signature
    uploaded_df = pd.read_csv(BytesIO(file_bytes))
    return predict_batch(uploaded_df)


def render_batch_prediction_page(model_loaded: bool) -> None:
    """Render the batch prediction page for CSV uploads."""
    render_page_header(
        "📂",
        "Batch Prediction",
        "Upload a CSV file containing multiple customer records to score churn risk for "
        "every customer using the trained model — without retraining.",
    )

    if not model_loaded:
        render_empty_state(
            "📂",
            "Batch prediction unavailable",
            "Trained model artifacts were not found. Run `python train_model.py` first to "
            "score customer lists from CSV uploads.",
        )
        return

    uploaded_file = st.file_uploader(
        "Upload customer CSV",
        type=["csv"],
        help="CSV must include all model feature columns (same schema as the Telco dataset).",
    )

    if uploaded_file is None:
        render_empty_state(
            "📂",
            "No batch predictions yet",
            "Upload a CSV file containing customer records to generate churn predictions, "
            "summary KPIs, analytics charts, and downloadable results.",
        )
        return

    file_bytes = uploaded_file.getvalue()

    try:
        raw_df = pd.read_csv(BytesIO(file_bytes))
    except Exception as exc:
        st.error("Unable to read the uploaded CSV file.")
        st.warning(str(exc))
        return

    render_section_header("📄", "Upload Summary", "File overview before batch scoring.")
    summary_cols = st.columns(2)
    with summary_cols[0]:
        render_kpi_card("Total Records Uploaded", f"{len(raw_df):,}", icon="📄")
    with summary_cols[1]:
        render_kpi_card("Number of Columns", str(len(raw_df.columns)), icon="🧮")

    render_section_header("👁️", "Dataset Preview", "First 10 rows of the uploaded file.")
    st.dataframe(raw_df.head(10), width="stretch", hide_index=True)

    validation = validate_batch_dataset(raw_df)
    if not validation["valid"]:
        render_validation_error_card(
            validation.get("message") or "The uploaded file cannot be processed because required columns are missing or invalid.",
            missing_columns=validation.get("missing_columns"),
            required_columns=validation.get("required_columns"),
        )
        return

    file_signature = md5(file_bytes).hexdigest()

    try:
        results_df = run_cached_batch_prediction(file_signature, file_bytes)
    except ValueError as exc:
        st.error("Batch prediction could not be completed.")
        st.warning(str(exc))
        return
    except FileNotFoundError as exc:
        st.error("Required model artifacts were not found.")
        st.warning(str(exc))
        return
    except Exception as exc:
        st.error("An unexpected error occurred during batch prediction.")
        st.warning(str(exc))
        return

    summary = build_batch_summary(results_df)
    charts = create_batch_visualizations(results_df)

    st.markdown("---")
    render_section_header("📈", "Summary KPIs", "Batch scoring results at a glance.")
    kpi_columns = st.columns(4)
    kpi_values = [
        ("👥", "Total Customers", f"{summary['total_customers']:,}", "Rows scored in this batch"),
        ("🔴", "Predicted Churn Customers", f"{summary['predicted_churn']:,}", "High-risk predictions"),
        ("🟢", "Predicted Safe Customers", f"{summary['predicted_safe']:,}", "Retained customer predictions"),
        ("📉", "Average Churn Probability", f"{summary['average_churn_probability']:.1%}", "Mean model score"),
    ]
    for column, (index, (icon, title, value, description)) in zip(
        kpi_columns,
        enumerate(kpi_values),
    ):
        with column:
            render_kpi_card(
                title,
                value,
                icon=icon,
                description=description,
                color_index=index,
            )

    render_section_header("📉", "Batch Analytics", "Distribution of predictions and risk levels.")
    chart_left, chart_right = st.columns(2)
    with chart_left:
        render_executive_chart(charts["prediction_distribution"])
    with chart_right:
        render_executive_chart(charts["risk_distribution"])
    render_executive_chart(charts["probability_histogram"])

    render_section_header("📋", "Prediction Results", "Search and review scored customer records.")
    search_term = st.text_input(
        "Search results",
        placeholder="Filter by prediction, risk level, or customer ID...",
    )

    display_columns = [
        column
        for column in ["customerID", "Prediction", "Probability", "Risk Level"]
        if column in results_df.columns
    ]
    display_df = results_df[display_columns].copy()

    if search_term:
        mask = display_df.astype(str).apply(
            lambda column: column.str.contains(search_term, case=False, na=False),
        ).any(axis=1)
        display_df = display_df[mask]

    st.dataframe(display_df, width="stretch", hide_index=True)

    render_section_header("📥", "Download Results", "Export batch predictions as CSV.")
    st.download_button(
        label="📥 Download Prediction Results",
        data=results_to_csv_bytes(results_df),
        file_name=f"batch_prediction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        type="primary",
        width="stretch",
        key="download_batch_results",
    )


def render_prediction_detail(record: Dict[str, Any]) -> None:
    """
    Render the detailed view for a selected prediction history record.

    Args:
        record: History record from session storage.
    """
    prediction = str(record.get("prediction", ""))
    churn_probability = float(record.get("churn_probability", 0.0))
    stay_probability = float(record.get("stay_probability", 0.0))
    confidence = float(record.get("confidence", 0.0))
    is_high_risk = prediction == "Churn"
    card_class = "result-card-high" if is_high_risk else "result-card-low"

    st.markdown("---")
    render_section_header("🔍", "Selected Prediction Detail", "Full record for the chosen history entry.")

    st.markdown(
        f"""
        <div class="{card_class}">
            <p><strong>Prediction Time:</strong> {record.get("prediction_timestamp", "—")}</p>
            <p><strong>Prediction:</strong> {prediction}</p>
            <p><strong>Churn Probability:</strong> {churn_probability:.1%}</p>
            <p><strong>Stay Probability:</strong> {stay_probability:.1%}</p>
            <p><strong>Confidence:</strong> {confidence:.1%}</p>
            <p><strong>Risk Level:</strong> {record.get("risk_level", "—")}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_section_header("👤", "Customer Information", "Profile details from the saved prediction.")
    customer_data = record.get("customer_data") or {}
    if customer_data:
        render_structured_customer_info(customer_data)
    else:
        st.markdown(
            f'<div class="card">{record.get("customer_information", "—")}</div>',
            unsafe_allow_html=True,
        )

    render_risk_meter(churn_probability)

    explanation_factors = record.get("explanation_factors", [])
    explain_class = "explain-card-high" if is_high_risk else "explain-card-low"
    render_section_header("🔍", "Prediction Explanation", "Key factors behind this prediction.")
    factors_html = "".join(
        f'<p class="explain-factor-item">✔ {factor}</p>' for factor in explanation_factors
    )
    render_html_card(explain_class, factors_html)

    render_section_header("🎯", "Retention Recommendations", "Suggested actions for this customer.")
    recommendations = record.get("recommendations", [])
    for recommendation in recommendations:
        render_recommendation_card(recommendation)
    render_recommendation_summary(recommendations)
    render_business_impact_box(record.get("business_impact", {}))


def render_prediction_history_page() -> None:
    """Render the customer search and prediction history page."""
    inject_history_page_styles()
    st.markdown('<div class="history-page-active"></div>', unsafe_allow_html=True)
    render_page_header(
        "👥",
        "Customer Search & History",
        "Search and review single-customer predictions made during this session — "
        "without re-running the model.",
    )

    history: List[Dict[str, Any]] = st.session_state.get(PREDICTION_HISTORY_KEY, [])

    if not history:
        render_empty_state(
            "👥",
            "No prediction history",
            "Run a prediction on the Churn Prediction page and records will appear here "
            "automatically for search, review, and export during this session.",
            action_label="Go to Churn Prediction",
            action_page="🔮 Churn Prediction",
        )
        return

    summary = build_history_summary(history)
    average_confidence = sum(float(record.get("confidence", 0.0)) for record in history) / len(history)

    render_section_header("📈", "Session KPIs", "Prediction analytics for the current session.")
    st.markdown('<div class="history-kpi-scope">', unsafe_allow_html=True)
    kpi_row1 = st.columns(3)
    kpi_row2 = st.columns(2)
    kpi_values = [
        ("🗂️", "Total Predictions", f"{summary['total_predictions']:,}", "Saved this session"),
        ("🔴", "High Risk Customers", f"{summary['high_risk_customers']:,}", "Predicted churn"),
        ("🟢", "Low Risk Customers", f"{summary['low_risk_customers']:,}", "Predicted retention"),
        ("📉", "Avg Churn Probability", f"{summary['average_churn_probability']:.1%}", "Session average"),
        ("🎯", "Avg Confidence", f"{average_confidence:.1%}", "Model certainty"),
    ]
    for index, (icon, title, value, description) in enumerate(kpi_values[:3]):
        with kpi_row1[index]:
            render_premium_kpi_card(title, value, icon=icon, description=description, color_index=index)
    for offset, (icon, title, value, description) in enumerate(kpi_values[3:]):
        with kpi_row2[offset]:
            render_premium_kpi_card(
                title, value, icon=icon, description=description, color_index=3 + offset
            )
    st.markdown("</div>", unsafe_allow_html=True)

    applied_filters = _ensure_history_applied_filters()
    _sync_history_widget_defaults()

    st.markdown('<div class="history-filter-scope">', unsafe_allow_html=True)
    with st.container(border=True):
        _render_history_filters_panel()
    st.markdown("</div>", unsafe_allow_html=True)

    filter_date = applied_filters["filter_date"] if applied_filters.get("use_date") else None
    filtered_history = filter_prediction_history(
        history,
        customer_id=str(applied_filters.get("customer_id", "")),
        prediction=str(applied_filters.get("prediction", "All")),
        risk_level=str(applied_filters.get("risk_level", "All")),
        filter_date=filter_date,
        search_text=str(applied_filters.get("search_text", "")),
    )

    render_section_header(
        "🗂️",
        "Prediction History",
        f"{len(filtered_history):,} record(s) matching the applied filters.",
    )
    if not filtered_history:
        render_empty_state(
            "🔍",
            "No matching predictions",
            "No records match the current filters. Adjust your criteria and click Apply Filters, "
            "or reset to view all session predictions.",
        )
    else:
        render_history_table(filtered_history)

        selected_record_id = st.selectbox(
            "View prediction detail",
            options=[record["record_id"] for record in filtered_history],
            format_func=lambda record_id: format_history_record_label(
                find_history_record(filtered_history, record_id) or {}
            ),
            key=HISTORY_SELECTED_KEY,
        )
        selected_record = find_history_record(filtered_history, selected_record_id)
        if selected_record:
            render_prediction_detail(selected_record)

    st.markdown("---")
    render_section_header("📤", "Export & Maintenance", "Download history or clear session records.")
    action_col1, action_col2 = st.columns(2)

    with action_col1:
        st.download_button(
            label="📥 Download Prediction History",
            data=download_prediction_history(history),
            file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            type="primary",
            width="stretch",
            key="download_prediction_history",
        )

    with action_col2:
        st.markdown('<div class="danger-btn-marker"></div>', unsafe_allow_html=True)
        if st.button(
            "🗑 Clear Prediction History",
            type="secondary",
            width="stretch",
            key="clear_prediction_history_btn",
        ):
            st.session_state[HISTORY_CONFIRM_CLEAR_KEY] = True

    if st.session_state.get(HISTORY_CONFIRM_CLEAR_KEY):
        render_confirm_banner(
            "Are you sure you want to clear all prediction history for this session?"
        )
        confirm_col1, confirm_col2 = st.columns(2)
        with confirm_col1:
            if st.button("Yes, clear history", type="primary", width="stretch"):
                st.session_state[PREDICTION_HISTORY_KEY] = clear_prediction_history()
                st.session_state[HISTORY_CONFIRM_CLEAR_KEY] = False
                st.session_state.pop(HISTORY_SELECTED_KEY, None)
                st.session_state.pop(HISTORY_APPLIED_FILTERS_KEY, None)
                _reset_history_filter_widgets()
                st.success("Prediction history cleared.")
                st.rerun()
        with confirm_col2:
            if st.button("Cancel", type="secondary", width="stretch", key="cancel_clear_history"):
                st.session_state[HISTORY_CONFIRM_CLEAR_KEY] = False
                st.rerun()


def main() -> None:
    """Run the Streamlit churn prediction application."""
    st.set_page_config(
        page_title="Customer Churn Analytics Platform",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_custom_styles()
    st.session_state[PREDICTION_HISTORY_KEY] = initialize_prediction_history(
        st.session_state.get(PREDICTION_HISTORY_KEY),
    )
    prediction_count = len(st.session_state.get(PREDICTION_HISTORY_KEY, []))
    _, _, algorithm, model_loaded, error_message = get_model_metadata()
    page = render_sidebar(algorithm, model_loaded, error_message, prediction_count)

    render_app_header(page, model_loaded)

    if page == "📊 Model Evaluation":
        render_model_evaluation_page(model_loaded)
    elif page == "📂 Batch Prediction":
        render_batch_prediction_page(model_loaded)
    elif page == "👥 Customer Search & History":
        render_prediction_history_page()
    elif page == "📊 Executive Dashboard":
        render_executive_dashboard_page()
    elif page == "📈 Executive Business Insights":
        render_executive_insights_page()
    elif page == "🧠 Explainable AI":
        render_explainable_ai_page(model_loaded)
    else:
        render_prediction_page(model_loaded, error_message)

    render_footer()


if __name__ == "__main__":
    main()
