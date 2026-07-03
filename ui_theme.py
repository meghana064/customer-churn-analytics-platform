"""
Enterprise UI theme — centralized design system for the Streamlit application.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

# ── Platform metadata ──────────────────────────────────────────────────────
PLATFORM_NAME = "Customer Churn Analytics & Retention Intelligence Platform"
HEADER_PLATFORM_NAME = "Customer Churn Analytics Platform"
PLATFORM_SHORT_NAME = "Customer Churn Analytics"
PLATFORM_TAGLINE = "Enterprise retention intelligence powered by machine learning."
APP_VERSION = "2.0.0"
DEVELOPER_NAME = "Meghana Gowda M"
DATASET_NAME = "IBM Telco Customer Churn"
APP_LOGO = "📊"
CURRENT_YEAR = datetime.now().year
NAVIGATION_KEY = "navigation"
SIDEBAR_COLLAPSED_KEY = "sidebar_collapsed"

PAGE_LABELS = {
    "📊 Executive Dashboard": "Dashboard",
    "🔮 Churn Prediction": "Prediction",
    "📂 Batch Prediction": "Batch Prediction",
    "👥 Customer Search & History": "History",
    "📈 Executive Business Insights": "Executive Insights",
    "📊 Model Evaluation": "Model Evaluation",
    "🧠 Explainable AI": "Explainable AI",
}

NAV_ITEMS: List[Tuple[str, str, str]] = [
    ("📊 Executive Dashboard", "🏠", "Dashboard"),
    ("🔮 Churn Prediction", "🔮", "Prediction"),
    ("📂 Batch Prediction", "📂", "Batch Prediction"),
    ("👥 Customer Search & History", "👥", "History"),
    ("📈 Executive Business Insights", "📈", "Executive Insights"),
    ("📊 Model Evaluation", "📊", "Model Evaluation"),
    ("🧠 Explainable AI", "🧠", "Explainable AI"),
]

KPI_ACCENTS = ("blue", "green", "amber", "red", "violet", "slate")

DESIGN_TOKENS: Dict[str, str] = {
    "primary": "#2563EB",
    "secondary": "#0EA5E9",
    "accent": "#2563EB",
    "background": "#F8FAFC",
    "surface": "#FFFFFF",
    "card": "#FFFFFF",
    "border": "#E5E7EB",
    "text_primary": "#1F2937",
    "text_secondary": "#6B7280",
    "success": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "info": "#3B82F6",
    "input_bg": "#F8FAFC",
    "input_border": "#CBD5E1",
}

ENTERPRISE_PLOTLY_LAYOUT: Dict[str, Any] = {
    "template": "plotly_white",
    "paper_bgcolor": "#FFFFFF",
    "plot_bgcolor": "#FFFFFF",
    "font": {"family": "Inter, sans-serif", "color": "#111827", "size": 14},
    "margin": {"l": 60, "r": 30, "t": 60, "b": 50},
    "colorway": ["#2563EB", "#16A34A", "#DC2626", "#F59E0B", "#8B5CF6", "#0EA5E9"],
    "legend": {
        "orientation": "h",
        "yanchor": "bottom",
        "y": 1.02,
        "xanchor": "right",
        "x": 1,
        "font": {"color": "#111827", "size": 13},
        "bgcolor": "rgba(255,255,255,0.95)",
        "bordercolor": "#E5E7EB",
        "borderwidth": 1,
    },
    "hoverlabel": {
        "bgcolor": "#FFFFFF",
        "bordercolor": "#E5E7EB",
        "font": {"color": "#111827", "size": 14, "family": "Inter, sans-serif"},
    },
    "title": {"font": {"size": 20, "color": "#111827"}, "x": 0.02, "xanchor": "left"},
}

PLOTLY_CHART_CONFIG: Dict[str, Any] = {
    "displayModeBar": True,
    "responsive": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


def apply_enterprise_plotly_theme(
    figure: Any,
    height: Optional[int] = None,
    hide_title: bool = False,
) -> Any:
    """Apply consistent enterprise Plotly styling for readable, visible charts."""
    layout = dict(ENTERPRISE_PLOTLY_LAYOUT)
    if height is not None:
        layout["height"] = max(380, min(450, int(height)))
    elif getattr(figure.layout, "height", None):
        layout["height"] = max(380, min(450, int(figure.layout.height)))
    else:
        layout["height"] = 400

    if hide_title:
        layout["title"] = None

    figure.update_layout(**layout)
    figure.update_xaxes(
        showgrid=True,
        showline=True,
        gridcolor="#E5E7EB",
        zerolinecolor="#E5E7EB",
        linecolor="#CBD5E1",
        tickfont={"color": "#6B7280", "size": 13},
        title_font={"color": "#111827", "size": 14},
        automargin=True,
    )
    figure.update_yaxes(
        showgrid=True,
        showline=True,
        gridcolor="#E5E7EB",
        zerolinecolor="#E5E7EB",
        linecolor="#CBD5E1",
        tickfont={"color": "#6B7280", "size": 13},
        title_font={"color": "#111827", "size": 14},
        automargin=True,
    )
    figure.update_traces(
        textfont={"color": "#1F2937", "size": 12},
        selector={"type": "bar"},
    )
    figure.update_traces(
        textfont={"color": "#1F2937", "size": 12},
        selector={"type": "pie"},
    )
    figure.update_traces(
        textfont={"color": "#1F2937", "size": 14},
        selector={"type": "heatmap"},
    )
    return figure


def render_plotly_chart(
    figure: Any,
    use_container_width: bool = True,
    height: Optional[int] = None,
    hide_title: bool = False,
    key: Optional[str] = None,
) -> None:
    """Render a Plotly chart with enterprise theme applied."""
    themed = apply_enterprise_plotly_theme(figure, height=height, hide_title=hide_title)
    kwargs: Dict[str, Any] = {
        "use_container_width": use_container_width,
        "config": PLOTLY_CHART_CONFIG,
    }
    if key is not None:
        kwargs["key"] = key
    st.plotly_chart(themed, **kwargs)


def render_validation_error_card(
    message: str,
    missing_columns: Optional[List[str]] = None,
    required_columns: Optional[List[str]] = None,
) -> None:
    """Render a structured validation error card with column chips."""
    chips_html = ""
    if missing_columns:
        chips = "".join(f'<span class="validation-chip">{col}</span>' for col in missing_columns)
        chips_html = (
            '<p class="validation-error-message"><strong>Missing columns:</strong></p>'
            f'<div class="validation-chip-row">{chips}</div>'
        )
    st.markdown(
        f"""
        <div class="validation-error-card">
            <p class="validation-error-title">⚠️ Missing Columns</p>
            <p class="validation-error-message">{message}</p>
            {chips_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
    if required_columns:
        with st.expander("View all required columns"):
            st.code(", ".join(required_columns))


def form_expander_label(icon: str, title: str, field_count: int) -> str:
    """Build a professional expander label with icon and field count."""
    return f"{icon}  {title}  ·  {field_count} fields"


def get_prediction_page_styles() -> str:
    """Return CSS scoped to the Prediction page form and actions."""
    return """
        <style>
            .prediction-section-header {
                display: flex; align-items: flex-start; gap: 0.75rem;
                margin-bottom: 0.85rem; padding-bottom: 0.65rem;
                border-bottom: 1px solid #E5E7EB;
            }

            .prediction-section-icon {
                width: 40px; height: 40px; border-radius: 10px;
                background: #EFF6FF;
                border: 1px solid #BFDBFE;
                display: flex; align-items: center; justify-content: center;
                font-size: 1.15rem; flex-shrink: 0;
            }

            .prediction-section-title {
                font-size: 16px; font-weight: 700; color: #1F2937 !important;
                margin: 0; line-height: 1.3;
            }

            .prediction-section-subtitle {
                font-size: 13px; color: #6B7280 !important;
                margin: 0.2rem 0 0 0; line-height: 1.4;
            }

            [data-testid="stVerticalBlockBorderWrapper"] {
                background: #FFFFFF !important;
                border: 1px solid #E5E7EB !important;
                border-radius: 16px !important;
                box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06), 0 1px 2px rgba(15, 23, 42, 0.04) !important;
                padding: 24px !important;
                margin-bottom: 20px !important;
            }

            .prediction-form-active ~ div [data-testid="stExpander"],
            .main .prediction-form-active ~ [data-testid="stExpander"] {
                background: #FFFFFF !important;
                border: 1px solid #E5E7EB !important;
                border-radius: 16px !important;
                box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06) !important;
                margin-bottom: 14px !important;
            }

            .prediction-form-active ~ div [data-testid="stExpander"] summary,
            .main .prediction-form-active ~ [data-testid="stExpander"] summary {
                font-size: 15px !important;
                font-weight: 700 !important;
                color: #111827 !important;
                background: #F8FAFC !important;
                border-radius: 16px 16px 0 0;
                padding: 0.75rem 1rem !important;
            }

            .prediction-form-active ~ div [data-testid="stExpander"] [data-testid="stExpanderDetails"],
            .main .prediction-form-active ~ [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
                padding: 0.5rem 1rem 1rem 1rem !important;
            }

            .form-section-caption {
                font-size: 13px !important;
                color: #6B7280 !important;
                margin: 0 0 0.65rem 0 !important;
                line-height: 1.4 !important;
            }

            [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"] {
                gap: 0.45rem !important;
            }

            [data-testid="stVerticalBlockBorderWrapper"] [data-testid="column"] [data-testid="stVerticalBlock"] {
                gap: 0.4rem !important;
            }

            [data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stWidgetLabel"] p,
            [data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stWidgetLabel"] label {
                color: #374151 !important;
                font-weight: 600 !important;
                font-size: 13px !important;
                margin-bottom: 0.1rem !important;
            }

            .field-format-hint {
                font-size: 12px; font-weight: 600; color: #2563EB !important;
                margin: -0.15rem 0 0.15rem 0; line-height: 1.2;
            }

            .field-na-hint {
                font-size: 12px; color: #9CA3AF !important;
                font-style: italic; margin: 0.15rem 0 0.35rem 0;
            }

            /* ── Boolean controls — compact inline No/Yes pills ── */
            .bool-field-label {
                color: #374151 !important;
                font-weight: 600 !important;
                font-size: 13px !important;
                margin: 0.35rem 0 0.1rem 0 !important;
                line-height: 1.3 !important;
            }

            .bool-field-caption {
                color: #6B7280 !important;
                font-size: 11px !important;
                margin: 0 0 0.25rem 0 !important;
                line-height: 1.35 !important;
            }

            [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stSegmentedControl"],
            .main [data-testid="stSegmentedControl"] {
                margin: 0.15rem 0 0.5rem 0 !important;
            }

            [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stSegmentedControl"] > div,
            .main [data-testid="stSegmentedControl"] > div {
                gap: 0.25rem !important;
                background: #E5E7EB !important;
                border-radius: 999px !important;
                padding: 3px !important;
                border: 1px solid #D1D5DB !important;
            }

            [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stSegmentedControl"] button,
            .main [data-testid="stSegmentedControl"] button {
                background: transparent !important;
                color: #6B7280 !important;
                border: none !important;
                font-weight: 600 !important;
                font-size: 12px !important;
                min-height: 30px !important;
                border-radius: 999px !important;
                padding: 0.2rem 0.65rem !important;
                box-shadow: none !important;
                transition: all 0.18s ease !important;
            }

            [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stSegmentedControl"] button[aria-pressed="true"],
            .main [data-testid="stSegmentedControl"] button[aria-pressed="true"] {
                background: #FFFFFF !important;
                color: #2563EB !important;
                border: none !important;
                box-shadow: 0 1px 3px rgba(37, 99, 235, 0.22) !important;
            }

            [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stSegmentedControl"] button[aria-pressed="true"]:first-child,
            .main [data-testid="stSegmentedControl"] button[aria-pressed="true"]:first-child {
                color: #64748B !important;
            }

            [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stSegmentedControl"] button[aria-pressed="true"]:last-child,
            .main [data-testid="stSegmentedControl"] button[aria-pressed="true"]:last-child {
                color: #2563EB !important;
            }

            [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stSegmentedControl"] button p,
            [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stSegmentedControl"] button span,
            .main [data-testid="stSegmentedControl"] button p,
            .main [data-testid="stSegmentedControl"] button span {
                color: inherit !important;
                font-size: 12px !important;
            }

            [data-testid="stVerticalBlockBorderWrapper"] [data-testid="column"] {
                overflow: visible !important;
            }

            /* Equal input heights */
            [data-testid="stVerticalBlockBorderWrapper"] .stNumberInput input,
            [data-testid="stVerticalBlockBorderWrapper"] .stSelectbox [data-baseweb="select"],
            [data-testid="stVerticalBlockBorderWrapper"] .stSelectbox div[data-baseweb="select"] > div {
                min-height: 40px !important;
            }

            [data-testid="stVerticalBlockBorderWrapper"] .stSlider {
                padding-top: 0.15rem !important;
                padding-bottom: 0.15rem !important;
            }

            /* Primary predict action */
            .prediction-submit-marker + div[data-testid="stButton"] > button,
            .prediction-submit-marker + div[data-testid="stButton"] button {
                border-radius: 14px !important;
                min-height: 52px !important;
                font-size: 16px !important;
                font-weight: 700 !important;
                letter-spacing: 0.01em;
                background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
                color: #FFFFFF !important;
                border: none !important;
                box-shadow: 0 4px 14px rgba(37, 99, 235, 0.32) !important;
                transition: transform 0.18s ease, box-shadow 0.18s ease !important;
            }

            .prediction-submit-marker + div[data-testid="stButton"] > button:hover,
            .prediction-submit-marker + div[data-testid="stButton"] button:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 8px 22px rgba(37, 99, 235, 0.38) !important;
                color: #FFFFFF !important;
            }

            .prediction-submit-marker + div[data-testid="stButton"] > button:focus-visible,
            .prediction-submit-marker + div[data-testid="stButton"] button:focus-visible {
                outline: 2px solid #93C5FD !important;
                outline-offset: 3px !important;
            }

            .prediction-submit-marker + div[data-testid="stButton"] > button p,
            .prediction-submit-marker + div[data-testid="stButton"] button p {
                color: #FFFFFF !important;
            }

            @media (max-width: 768px) {
                [data-testid="stVerticalBlockBorderWrapper"] {
                    padding: 18px !important;
                    margin-bottom: 14px !important;
                }
            }
        </style>
    """


def get_history_page_styles() -> str:
    """Return CSS for the Customer Search & History page."""
    return """
        <style>
            .history-filter-scope [data-testid="stDateInput"] label,
            .history-filter-scope [data-testid="stDateInput"] [data-testid="stWidgetLabel"] {
                white-space: normal !important;
            }

            .history-filter-scope [data-testid="stDateInput"] > div {
                flex-wrap: wrap !important;
            }

            .history-filter-scope [data-testid="stDateInput"] input {
                min-width: 9.5rem !important;
            }

            .history-filter-card-header {
                display: flex; align-items: center; gap: 0.6rem;
                margin-bottom: 0.85rem; padding-bottom: 0.65rem;
                border-bottom: 1px solid #E5E7EB;
            }

            .history-filter-card-header .history-filter-icon {
                width: 36px; height: 36px; border-radius: 10px;
                background: #EFF6FF; border: 1px solid #BFDBFE;
                display: flex; align-items: center; justify-content: center;
                font-size: 1rem;
            }

            .history-filter-card-header .history-filter-title {
                font-size: 15px; font-weight: 700; color: #1F2937 !important; margin: 0;
            }

            .history-filter-card-header .history-filter-subtitle {
                font-size: 12px; color: #6B7280 !important; margin: 0.15rem 0 0 0;
            }

            .history-table-wrap {
                border: 1px solid #E5E7EB;
                border-radius: 16px;
                overflow: auto;
                background: #FFFFFF;
                box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
                margin-top: 0.35rem;
            }

            .history-table {
                width: 100%;
                border-collapse: separate;
                border-spacing: 0;
                font-size: 13px;
            }

            .history-table thead th {
                position: sticky; top: 0; z-index: 2;
                background: #F8FAFC !important;
                color: #374151 !important;
                font-weight: 700 !important;
                text-align: left;
                padding: 0.85rem 0.9rem;
                border-bottom: 1px solid #E5E7EB;
                white-space: nowrap;
            }

            .history-table tbody td {
                padding: 0.75rem 0.9rem;
                color: #1F2937 !important;
                border-bottom: 1px solid #F1F5F9;
                vertical-align: middle;
            }

            .history-table tbody tr:nth-child(even) td {
                background: #FAFBFC !important;
            }

            .history-table tbody tr:hover td {
                background: #EFF6FF !important;
            }

            .history-pred-badge {
                display: inline-block; font-size: 11px; font-weight: 700;
                padding: 0.2rem 0.55rem; border-radius: 999px;
            }

            .history-pred-churn { background: #FEE2E2; color: #B91C1C; border: 1px solid #FECACA; }
            .history-pred-safe  { background: #DCFCE7; color: #166534; border: 1px solid #86EFAC; }

            .history-risk-badge {
                display: inline-block; font-size: 11px; font-weight: 700;
                padding: 0.22rem 0.55rem; border-radius: 999px; white-space: nowrap;
            }

            .history-risk-very-low { background: #ECFDF5; color: #047857; border: 1px solid #A7F3D0; }
            .history-risk-low      { background: #F0FDF4; color: #15803D; border: 1px solid #BBF7D0; }
            .history-risk-moderate { background: #FFFBEB; color: #B45309; border: 1px solid #FDE68A; }
            .history-risk-high     { background: #FFF7ED; color: #C2410C; border: 1px solid #FED7AA; }
            .history-risk-critical { background: #FEF2F2; color: #B91C1C; border: 1px solid #FECACA; }
            .history-risk-default  { background: #F3F4F6; color: #374151; border: 1px solid #E5E7EB; }

            .history-mini-bar {
                position: relative; height: 22px; min-width: 88px;
                background: #E5E7EB; border-radius: 999px; overflow: hidden;
            }

            .history-mini-bar-fill {
                position: absolute; left: 0; top: 0; bottom: 0;
                border-radius: 999px; transition: width 0.2s ease;
            }

            .history-mini-bar-blue .history-mini-bar-fill { background: linear-gradient(90deg, #60A5FA, #2563EB); }
            .history-mini-bar-violet .history-mini-bar-fill { background: linear-gradient(90deg, #A78BFA, #7C3AED); }

            .history-mini-bar-label {
                position: relative; z-index: 1;
                display: flex; align-items: center; justify-content: center;
                height: 100%; font-size: 11px; font-weight: 700;
                color: #1F2937 !important;
            }

            .history-customer-cell {
                display: flex; flex-wrap: wrap; gap: 0.35rem 0.55rem;
                max-width: 280px; line-height: 1.35;
            }

            .history-customer-chip {
                display: inline-flex; align-items: center; gap: 0.2rem;
                font-size: 11px; color: #4B5563 !important;
                background: #F8FAFC; border: 1px solid #E5E7EB;
                border-radius: 6px; padding: 0.15rem 0.4rem;
            }

            .history-customer-profile {
                display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 0.55rem 0.85rem; margin-top: 0.25rem;
            }

            .history-customer-profile-item {
                display: flex; align-items: flex-start; gap: 0.45rem;
                font-size: 13px; color: #374151 !important;
                background: #F8FAFC; border: 1px solid #E5E7EB;
                border-radius: 10px; padding: 0.55rem 0.65rem;
            }

            .history-customer-profile-icon { font-size: 1rem; line-height: 1.2; }

            .history-customer-profile-label {
                font-size: 10px; font-weight: 700; text-transform: uppercase;
                letter-spacing: 0.04em; color: #6B7280 !important;
            }

            .history-customer-profile-value {
                font-size: 13px; font-weight: 600; color: #1F2937 !important;
            }

            .danger-btn-marker + div[data-testid="stButton"] > button,
            .danger-btn-marker + div[data-testid="stButton"] button {
                background: #FFFFFF !important;
                color: #DC2626 !important;
                border: 1.5px solid #DC2626 !important;
            }

            .danger-btn-marker + div[data-testid="stButton"] > button p,
            .danger-btn-marker + div[data-testid="stButton"] button p {
                color: #DC2626 !important;
            }

            .danger-btn-marker + div[data-testid="stButton"] > button:hover,
            .danger-btn-marker + div[data-testid="stButton"] button:hover {
                background: #FEF2F2 !important;
            }

            @media (max-width: 1100px) {
                .history-customer-profile { grid-template-columns: 1fr; }
            }

            @media (max-width: 768px) {
                .history-table { font-size: 12px; }
                .history-table thead th, .history-table tbody td { padding: 0.6rem 0.65rem; }
            }
        </style>
    """


def inject_history_page_styles() -> None:
    """Inject Customer Search & History page styles."""
    st.markdown(get_history_page_styles(), unsafe_allow_html=True)


def _history_risk_badge_class(risk_level: str) -> str:
    """Map risk level labels to CSS badge classes."""
    mapping = {
        "Very Low Risk": "history-risk-very-low",
        "Low Risk": "history-risk-low",
        "Moderate Risk": "history-risk-moderate",
        "High Risk": "history-risk-high",
        "Critical Risk": "history-risk-critical",
    }
    return mapping.get(str(risk_level), "history-risk-default")


def _history_mini_bar_html(value: float, tone: str = "blue") -> str:
    """Build a mini progress bar HTML snippet."""
    pct = max(0.0, min(1.0, float(value)))
    width = pct * 100.0
    return (
        f'<div class="history-mini-bar history-mini-bar-{tone}">'
        f'<div class="history-mini-bar-fill" style="width:{width:.1f}%"></div>'
        f'<span class="history-mini-bar-label">{pct:.0%}</span></div>'
    )


def render_history_filter_header() -> None:
    """Render the premium filter card header."""
    st.markdown(
        """
        <div class="history-filter-card-header">
            <div class="history-filter-icon">🔎</div>
            <div>
                <p class="history-filter-title">Search &amp; Filters</p>
                <p class="history-filter-subtitle">Refine session predictions by ID, risk, date, or keywords</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_history_table(records: List[Dict[str, Any]]) -> None:
    """Render a premium HTML history table with badges and progress bars."""
    if not records:
        return

    rows: List[str] = []
    for record in records:
        prediction = str(record.get("prediction", "—"))
        pred_class = "history-pred-churn" if prediction == "Churn" else "history-pred-safe"
        risk_level = str(record.get("risk_level", "—"))
        risk_class = _history_risk_badge_class(risk_level)
        probability = float(record.get("churn_probability", 0.0))
        confidence = float(record.get("confidence", 0.0))
        customer_data = record.get("customer_data") or {}
        customer_cell = _history_customer_cell_html(customer_data)
        timestamp = str(record.get("prediction_timestamp", "—")).replace("T", " ")
        customer_id = record.get("customer_id") or "—"
        rec_count = int(record.get("retention_recommendation_count", 0))

        rows.append(
            f"<tr>"
            f"<td>{timestamp}</td>"
            f"<td>{customer_id}</td>"
            f'<td><span class="history-pred-badge {pred_class}">{prediction}</span></td>'
            f'<td><span class="history-risk-badge {risk_class}">{risk_level}</span></td>'
            f"<td>{_history_mini_bar_html(probability, 'blue')}</td>"
            f"<td>{_history_mini_bar_html(confidence, 'violet')}</td>"
            f"<td>{customer_cell}</td>"
            f"<td>{rec_count}</td>"
            f"</tr>"
        )

    table_html = (
        '<div class="history-table-wrap"><table class="history-table">'
        "<thead><tr>"
        "<th>Prediction Time</th><th>Customer ID</th><th>Prediction</th>"
        "<th>Risk Level</th><th>Probability</th><th>Confidence</th>"
        "<th>Customer Information</th><th>Retention Recs</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)


def _history_customer_cell_html(customer_data: Dict[str, Any]) -> str:
    """Build compact customer information chips for the history table."""
    if not customer_data:
        return "—"
    senior = "Yes" if int(customer_data.get("SeniorCitizen", 0)) == 1 else "No"
    chips = [
        ("👤", str(customer_data.get("gender", "N/A")).title()),
        ("📄", str(customer_data.get("Contract", "N/A"))),
        ("🌐", str(customer_data.get("InternetService", "N/A"))),
        ("💳", str(customer_data.get("PaymentMethod", "N/A"))[:18]),
        ("💵", f"${float(customer_data.get('MonthlyCharges', 0)):,.0f}"),
        ("📅", f"{int(customer_data.get('tenure', 0))} mo"),
        ("🎂", f"Senior: {senior}"),
    ]
    chip_html = "".join(
        f'<span class="history-customer-chip"><span>{icon}</span>{text}</span>'
        for icon, text in chips
    )
    return f'<div class="history-customer-cell">{chip_html}</div>'


def render_structured_customer_info(customer_data: Dict[str, Any]) -> None:
    """Render structured customer profile details with icons."""
    if not customer_data:
        render_html_card("card", "<p>No customer profile details available.</p>")
        return

    senior = "Yes" if int(customer_data.get("SeniorCitizen", 0)) == 1 else "No"
    items = [
        ("👤", "Gender", str(customer_data.get("gender", "N/A")).title()),
        ("🎂", "Senior Citizen", senior),
        ("👥", "Partner", str(customer_data.get("Partner", "N/A"))),
        ("👨‍👩‍👧", "Dependents", str(customer_data.get("Dependents", "N/A"))),
        ("📄", "Contract", str(customer_data.get("Contract", "N/A"))),
        ("📅", "Tenure", f"{int(customer_data.get('tenure', 0))} months"),
        ("🌐", "Internet", str(customer_data.get("InternetService", "N/A"))),
        ("📞", "Phone", str(customer_data.get("PhoneService", "N/A"))),
        ("💳", "Payment", str(customer_data.get("PaymentMethod", "N/A"))),
        ("💵", "Monthly Charges", f"${float(customer_data.get('MonthlyCharges', 0)):,.2f}"),
        ("💰", "Total Charges", f"${float(customer_data.get('TotalCharges', 0)):,.2f}"),
    ]
    items_html = "".join(
        f'<div class="history-customer-profile-item">'
        f'<span class="history-customer-profile-icon">{icon}</span>'
        f"<div><div class='history-customer-profile-label'>{label}</div>"
        f"<div class='history-customer-profile-value'>{value}</div></div></div>"
        for icon, label, value in items
    )
    render_html_card("card", f'<div class="history-customer-profile">{items_html}</div>')


def format_history_record_label(record: Dict[str, Any]) -> str:
    """Build a clean dropdown label for a history record."""
    timestamp = str(record.get("prediction_timestamp", "—")).replace("T", " ")[:16]
    prediction = str(record.get("prediction", "—"))
    risk_level = str(record.get("risk_level", "—"))
    probability = float(record.get("churn_probability", 0.0))
    customer_id = record.get("customer_id") or "No ID"
    return (
        f"{timestamp}  •  {customer_id}  •  {prediction}  •  "
        f"{risk_level}  •  {probability:.0%} churn"
    )


def render_confirm_banner(message: str) -> None:
    """Render a visible confirmation banner with high-contrast text."""
    st.markdown(
        f"""
        <div class="confirm-banner">
            <span class="confirm-banner-icon">⚠️</span>
            <p class="confirm-banner-text">{message}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def inject_prediction_page_styles() -> None:
    """Inject Prediction page scoped styles."""
    st.markdown(get_prediction_page_styles(), unsafe_allow_html=True)


def render_prediction_section_header(icon: str, title: str, subtitle: str) -> None:
    """Render a premium section header for the prediction form."""
    st.markdown(
        f"""
        <div class="prediction-section-header">
            <div class="prediction-section-icon">{icon}</div>
            <div>
                <p class="prediction-section-title">{title}</p>
                <p class="prediction-section-subtitle">{subtitle}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_html_card(css_class: str, inner_html: str) -> None:
    """Render a self-contained HTML card (single block — safe with Streamlit)."""
    st.markdown(
        f'<div class="{css_class}">{inner_html}</div>',
        unsafe_allow_html=True,
    )


def get_app_styles() -> str:
    """Return the complete enterprise design-system CSS."""
    return """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

            :root {
                --color-primary: #2563EB;
                --color-primary-dark: #1D4ED8;
                --color-primary-soft: #EFF6FF;
                --color-secondary: #0EA5E9;
                --color-accent: #2563EB;
                --color-background: #F8FAFC;
                --color-surface: #FFFFFF;
                --color-card: #FFFFFF;
                --color-border: #E5E7EB;
                --color-input-bg: #FFFFFF;
                --color-input-border: #CBD5E1;
                --color-text-primary: #111827;
                --color-text-secondary: #6B7280;
                --color-success: #16A34A;
                --color-warning: #F59E0B;
                --color-danger: #DC2626;
                --color-info: #3B82F6;
                --color-success-bg: #ECFDF5;
                --color-success-text: #047857;
                --color-success-border: #A7F3D0;
                --color-warning-bg: #FFFBEB;
                --color-warning-text: #B45309;
                --color-warning-border: #FDE68A;
                --color-danger-bg: #FEF2F2;
                --color-danger-text: #B91C1C;
                --color-danger-border: #FECACA;
                --color-info-bg: #EFF6FF;
                --color-info-text: #1D4ED8;
                --color-info-border: #BFDBFE;
                --shadow-xs: 0 1px 2px rgba(15, 23, 42, 0.04);
                --shadow-sm: 0 1px 3px rgba(15, 23, 42, 0.06);
                --shadow-md: 0 4px 16px rgba(15, 23, 42, 0.08);
                --shadow-lg: 0 8px 24px rgba(15, 23, 42, 0.10);
                --shadow-glow: 0 0 0 3px rgba(37, 99, 235, 0.14);
                --radius-card: 16px;
                --radius-input: 12px;
                --card-padding: 28px;
                --card-gap: 24px;
                --header-height: 86px;
                --font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            }

            #MainMenu, footer, header[data-testid="stHeader"] {
                visibility: hidden !important;
                height: 0 !important;
                min-height: 0 !important;
            }

            .stDeployButton, [data-testid="stToolbar"], [data-testid="stDecoration"] {
                display: none !important;
            }

            .stApp, [data-testid="stAppViewContainer"], .main {
                background-color: var(--color-background) !important;
                color: var(--color-text-primary) !important;
                font-family: var(--font-family) !important;
                font-size: 15px;
                overflow-x: hidden !important;
            }

            .main .block-container {
                padding-top: 1rem;
                padding-bottom: 2rem;
                padding-left: 1.75rem;
                padding-right: 1.75rem;
                max-width: 100%;
                overflow-x: hidden !important;
            }

            div[data-testid="column"] { min-width: 0 !important; }

            div[data-testid="stWidgetLabel"] p,
            div[data-testid="stWidgetLabel"] label {
                color: #111827 !important;
                font-weight: 600 !important;
                font-size: 14px !important;
                white-space: normal !important;
                line-height: 1.35 !important;
            }

            /* ── Enterprise collapsible sidebar (expanded + icon rail) ── */
            [data-testid="stSidebar"] {
                transition: min-width 0.25s ease, max-width 0.25s ease, width 0.25s ease !important;
                z-index: 999900 !important;
            }

            [data-testid="stAppViewContainer"] > section[data-testid="stMain"],
            section.main {
                transition: margin-left 0.25s ease !important;
            }

            .stApp:has(.sidebar-is-expanded) [data-testid="stSidebar"] {
                min-width: 280px !important;
                max-width: 280px !important;
                width: 280px !important;
            }

            .stApp:has(.sidebar-is-collapsed) [data-testid="stSidebar"] {
                min-width: 72px !important;
                max-width: 72px !important;
                width: 72px !important;
                transform: none !important;
                visibility: visible !important;
                pointer-events: auto !important;
                overflow-x: hidden !important;
            }

            .stApp:has(.sidebar-is-expanded) [data-testid="stAppViewContainer"] > section[data-testid="stMain"] {
                margin-left: 280px !important;
            }

            .stApp:has(.sidebar-is-collapsed) [data-testid="stAppViewContainer"] > section[data-testid="stMain"] {
                margin-left: 72px !important;
            }

            .stApp:has(.sidebar-is-collapsed) .sidebar-brand-title,
            .stApp:has(.sidebar-is-collapsed) .sidebar-brand-version,
            .stApp:has(.sidebar-is-collapsed) .sidebar-section-label,
            .stApp:has(.sidebar-is-collapsed) .sidebar-meta-card,
            .stApp:has(.sidebar-is-collapsed) .sidebar-footer,
            .stApp:has(.sidebar-is-collapsed) .sidebar-nav-label-text {
                display: none !important;
            }

            .stApp:has(.sidebar-is-collapsed) .sidebar-brand {
                margin-bottom: 0.35rem !important;
                padding-bottom: 0.35rem !important;
            }

            .stApp:has(.sidebar-is-collapsed) div[data-testid="stSidebar"] .block-container {
                padding: 0.65rem 0.4rem 1rem 0.4rem !important;
            }

            .stApp:has(.sidebar-is-collapsed) div[data-testid="stSidebar"] .stButton {
                margin-bottom: 2px !important;
            }

            .stApp:has(.sidebar-is-collapsed) div[data-testid="stSidebar"] .stButton > button {
                min-height: 44px !important;
                padding: 0.4rem !important;
                justify-content: center !important;
            }

            .stApp:has(.sidebar-is-collapsed) div[data-testid="stSidebar"] .stButton > button p {
                font-size: 1.2rem !important;
                line-height: 1 !important;
            }

            .stApp:has(.sidebar-is-collapsed) div[data-testid="stSidebar"] .stButton > button[kind="primary"] {
                border-left: 3px solid #2563EB !important;
                background: #EFF6FF !important;
            }

            .sidebar-collapse-toggle-wrap + div[data-testid="stButton"] button,
            .sidebar-collapse-toggle-wrap ~ div[data-testid="stElementContainer"] button {
                min-height: 40px !important;
                margin-bottom: 0.5rem !important;
                border-radius: 10px !important;
                font-size: 18px !important;
                font-weight: 700 !important;
            }

            [data-testid="collapsedControl"],
            [data-testid="stSidebarCollapsedControl"],
            [data-testid="stSidebarCollapseButton"] {
                display: none !important;
            }

            /* Form / filter spacing */
            [data-testid="stVerticalBlock"] > div {
                gap: 0.5rem;
            }

            [data-testid="stTextInput"],
            [data-testid="stSelectbox"],
            [data-testid="stNumberInput"],
            [data-testid="stDateInput"],
            [data-testid="stCheckbox"],
            [data-testid="stSegmentedControl"] {
                margin-bottom: 0.35rem !important;
            }

            .history-filter-scope [data-testid="stVerticalBlockBorderWrapper"] {
                padding: 1.25rem !important;
            }

            .history-filter-scope [data-testid="column"] {
                min-width: 0 !important;
            }

            .history-filter-scope [data-testid="stWidgetLabel"] p,
            .history-filter-scope [data-testid="stWidgetLabel"] label {
                white-space: normal !important;
            }

            .history-kpi-scope [data-testid="column"] {
                min-width: 0 !important;
            }

            .history-filter-actions {
                margin-top: 0.35rem;
            }

            div[data-testid="stPlotlyChart"] {
                width: 100% !important;
                min-height: 380px;
                overflow: hidden !important;
            }

            div[data-testid="stPlotlyChart"] > div {
                width: 100% !important;
            }

            /* ── Header ── */
            .app-header {
                background: var(--color-card) !important;
                border: 1px solid var(--color-border) !important;
                border-radius: 14px !important;
                box-shadow: var(--shadow-sm) !important;
                padding: 0 20px !important;
                margin-bottom: var(--card-gap) !important;
                height: var(--header-height);
                max-height: 90px;
                display: flex;
                align-items: center;
            }

            .app-header-top {
                display: flex;
                flex-wrap: wrap;
                align-items: center;
                justify-content: space-between;
                gap: 0.5rem 1rem;
                width: 100%;
            }

            .app-header-brand { display: flex; align-items: center; gap: 0.7rem; min-width: 0; }

            .app-header-logo {
                width: 38px; height: 38px; border-radius: 10px;
                background: var(--color-primary-soft);
                display: flex; align-items: center; justify-content: center;
                font-size: 1.15rem; border: 1px solid var(--color-info-border); flex-shrink: 0;
            }

            .app-header-title {
                font-size: 16px; font-weight: 700;
                color: var(--color-text-primary) !important;
                margin: 0; line-height: 1.2;
            }

            .app-header-subtitle {
                font-size: 12px; color: var(--color-text-secondary) !important;
                margin: 2px 0 0 0; line-height: 1.3;
            }

            .app-header-meta { display: flex; align-items: center; gap: 10px; flex-shrink: 0; flex-wrap: wrap; justify-content: flex-end; }

            .header-status {
                display: inline-flex; align-items: center; gap: 5px;
                font-size: 11px; font-weight: 600; color: var(--color-text-secondary);
                padding: 3px 8px; border-radius: 999px;
                background: var(--color-input-bg); border: 1px solid var(--color-border);
            }

            .header-status-dot { width: 7px; height: 7px; border-radius: 50%; }
            .header-status-dot.online { background: var(--color-success); }
            .header-status-dot.offline { background: var(--color-danger); }

            .page-badge {
                font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 999px;
                letter-spacing: 0.04em; text-transform: uppercase;
                background: var(--color-primary-soft); color: var(--color-primary);
                border: 1px solid var(--color-info-border);
            }

            .header-datetime { font-size: 12px; color: var(--color-text-secondary) !important; }

            /* ── Light enterprise sidebar ── */
            div[data-testid="stSidebar"] {
                background: var(--color-card) !important;
                border-right: 1px solid var(--color-border);
                min-width: 260px !important;
                max-width: 280px !important;
            }

            div[data-testid="stSidebar"] > div:first-child { background: var(--color-card) !important; }

            div[data-testid="stSidebar"] .block-container {
                padding: 0.85rem 0.75rem 1rem 0.75rem;
            }

            div[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
                color: var(--color-text-secondary) !important;
                font-size: 11px !important;
            }

            .sidebar-brand {
                text-align: center; margin-bottom: 0.65rem;
                padding-bottom: 0.75rem; border-bottom: 1px solid var(--color-border);
            }

            .sidebar-logo-placeholder {
                width: 44px; height: 44px; border-radius: 12px;
                background: var(--color-primary-soft);
                display: flex; align-items: center; justify-content: center;
                font-size: 1.3rem; border: 1px solid var(--color-info-border); margin: 0 auto;
            }

            .sidebar-brand-title {
                font-size: 14px; font-weight: 700;
                color: var(--color-text-primary) !important; margin-top: 0.45rem;
            }

            .sidebar-brand-version { font-size: 11px; color: var(--color-text-secondary) !important; }

            .sidebar-section-label {
                font-size: 10px; font-weight: 700; text-transform: uppercase;
                letter-spacing: 0.1em; color: var(--color-text-secondary) !important;
                margin: 0.75rem 0 0.35rem 0.3rem;
            }

            div[data-testid="stSidebar"] .stButton { margin-bottom: 4px; }

            div[data-testid="stSidebar"] .stButton > button {
                width: 100%; border-radius: var(--radius-input) !important;
                min-height: 42px !important; font-weight: 600 !important;
                font-size: 13px !important; text-align: left !important;
                padding: 0.48rem 0.85rem !important;
                transition: all 0.18s ease !important;
                font-family: var(--font-family) !important;
            }

            div[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
                background: var(--color-card) !important;
                color: var(--color-text-secondary) !important;
                border: 1px solid var(--color-border) !important;
                border-left: 3px solid transparent !important;
                box-shadow: var(--shadow-xs) !important;
            }

            div[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
                background: var(--color-primary-soft) !important;
                color: var(--color-primary) !important;
                border-color: var(--color-info-border) !important;
                border-left-color: var(--color-primary) !important;
                transform: translateX(2px);
            }

            div[data-testid="stSidebar"] .stButton > button[kind="primary"] {
                background: var(--color-primary) !important;
                color: #FFFFFF !important;
                border: none !important;
                border-left: 3px solid var(--color-primary-dark) !important;
                box-shadow: 0 2px 10px rgba(37, 99, 235, 0.28) !important;
            }

            .sidebar-meta-card {
                background: var(--color-input-bg) !important;
                border: 1px solid var(--color-border) !important;
                border-radius: var(--radius-input) !important;
                padding: 0.75rem 0.85rem !important;
                margin-bottom: 0.45rem !important;
            }

            .sidebar-meta-row {
                display: flex; justify-content: space-between; gap: 0.5rem;
                font-size: 12px; color: var(--color-text-secondary) !important;
                margin-bottom: 0.3rem;
            }

            .sidebar-meta-row strong {
                color: var(--color-text-primary) !important;
                font-weight: 600; text-align: right;
            }

            .sidebar-footer {
                margin-top: 0.85rem; padding-top: 0.75rem;
                border-top: 1px solid var(--color-border); text-align: center;
            }

            .sidebar-footer-text {
                font-size: 10px; color: var(--color-text-secondary) !important; line-height: 1.5;
            }

            /* ── Forms ── */
            .stTextInput > div > div,
            .stNumberInput > div > div,
            .stDateInput > div > div,
            .stTextArea > div > div {
                background-color: transparent !important;
                border: none !important;
                box-shadow: none !important;
            }

            .stTextInput input, .stNumberInput input, .stDateInput input, .stTextArea textarea {
                background-color: #FFFFFF !important;
                color: #1F2937 !important;
                border: 1px solid #CBD5E1 !important;
                border-radius: 8px !important;
                min-height: 42px; font-size: 15px !important;
                box-shadow: none !important;
            }

            .stTextInput input:focus, .stNumberInput input:focus,
            .stDateInput input:focus, .stTextArea textarea:focus {
                border-color: var(--color-primary) !important;
                box-shadow: var(--shadow-glow) !important;
            }

            .stSelectbox [data-baseweb="select"],
            .stSelectbox div[data-baseweb="select"] > div,
            .stMultiSelect [data-baseweb="select"],
            .stMultiSelect div[data-baseweb="select"] > div {
                background-color: #FFFFFF !important;
                color: #1F2937 !important;
                border: 1px solid #CBD5E1 !important;
                border-radius: 8px !important;
                min-height: 42px;
                box-shadow: none !important;
            }

            /* ── Checkboxes — visible check marks ── */
            div[data-testid="stCheckbox"] label[data-baseweb="checkbox"],
            .stCheckbox label[data-baseweb="checkbox"] {
                background: transparent !important;
            }

            div[data-testid="stCheckbox"] label[data-baseweb="checkbox"] > div:first-of-type,
            .stCheckbox label[data-baseweb="checkbox"] > div:first-of-type {
                width: 18px !important;
                height: 18px !important;
                background: #FFFFFF !important;
                border: 2px solid #94A3B8 !important;
                border-radius: 4px !important;
            }

            div[data-testid="stCheckbox"] label[data-baseweb="checkbox"][aria-checked="true"] > div:first-of-type,
            .stCheckbox label[data-baseweb="checkbox"][aria-checked="true"] > div:first-of-type {
                background: #2563EB !important;
                border-color: #1D4ED8 !important;
            }

            div[data-testid="stCheckbox"] [data-testid="stWidgetLabel"] p,
            .stCheckbox [data-testid="stWidgetLabel"] p {
                color: #374151 !important;
                font-weight: 600 !important;
            }

            /* ── Segmented controls (No/Yes boolean fields) ── */
            [data-testid="stSegmentedControl"] > div {
                gap: 0.25rem !important;
                background: #E5E7EB !important;
                border-radius: 999px !important;
                padding: 3px !important;
                border: 1px solid #D1D5DB !important;
            }

            [data-testid="stSegmentedControl"] button {
                background: transparent !important;
                color: #6B7280 !important;
                border: none !important;
                font-weight: 600 !important;
                min-height: 30px !important;
                border-radius: 999px !important;
                box-shadow: none !important;
            }

            [data-testid="stSegmentedControl"] button[aria-pressed="true"] {
                background: #FFFFFF !important;
                box-shadow: 0 1px 3px rgba(37, 99, 235, 0.22) !important;
            }

            [data-testid="stSegmentedControl"] button p,
            [data-testid="stSegmentedControl"] button span {
                color: inherit !important;
                font-size: 12px !important;
            }

            /* ── Confirmation banner ── */
            .confirm-banner {
                display: flex; align-items: flex-start; gap: 0.65rem;
                background: #FFFBEB !important;
                border: 1px solid #F59E0B !important;
                border-radius: 16px !important;
                padding: 1rem 1.15rem !important;
                margin: 0.75rem 0 1rem 0 !important;
            }

            .confirm-banner-icon { font-size: 1.1rem; line-height: 1.4; }

            .confirm-banner-text {
                margin: 0 !important;
                color: #92400E !important;
                font-size: 14px !important;
                font-weight: 600 !important;
                line-height: 1.5 !important;
            }

            .stSelectbox [data-baseweb="select"] span,
            .stMultiSelect [data-baseweb="select"] span { color: var(--color-text-primary) !important; }

            .stSelectbox [data-baseweb="select"]:focus-within,
            .stMultiSelect [data-baseweb="select"]:focus-within {
                border-color: var(--color-primary) !important;
                box-shadow: var(--shadow-glow) !important;
            }

            div[data-baseweb="popover"] > div, ul[data-baseweb="menu"] {
                background: var(--color-card) !important;
                border: 1px solid var(--color-border) !important;
                border-radius: var(--radius-input) !important;
            }

            li[role="option"] {
                background: var(--color-card) !important;
                color: var(--color-text-primary) !important;
            }

            li[role="option"]:hover, li[role="option"][aria-selected="true"] {
                background: var(--color-primary-soft) !important;
                color: var(--color-primary) !important;
            }

            div[data-testid="stWidgetLabel"] p, div[data-testid="stWidgetLabel"] label {
                font-size: 14px !important; color: #374151 !important;
                font-weight: 600 !important; margin-bottom: 0.15rem !important;
            }

            div[data-testid="stRadio"] label p,
            div[data-testid="stRadio"] label span,
            div[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {
                color: #374151 !important;
                font-weight: 500 !important;
            }

            div[data-testid="stExpander"] {
                background: var(--color-card); border: 1px solid var(--color-border);
                border-radius: var(--radius-card); box-shadow: var(--shadow-xs);
                margin-bottom: 0.75rem !important;
            }

            div[data-testid="stExpander"] summary {
                font-size: 15px !important; font-weight: 700 !important;
                color: #111827 !important;
                background: #F8FAFC !important;
                border-radius: var(--radius-card) var(--radius-card) 0 0;
                padding: 0.7rem 1rem !important;
            }

            div[data-testid="stExpander"] summary:hover {
                background: var(--color-primary-soft) !important;
                color: var(--color-primary) !important;
            }

            div[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
                padding: 0.5rem 1rem 1rem 1rem !important;
            }

            .main [data-testid="stVerticalBlock"] > div { gap: 0.35rem; }

            .validation-error-card {
                background: var(--color-danger-bg) !important;
                border: 1px solid var(--color-danger-border) !important;
                border-radius: var(--radius-card) !important;
                padding: 1rem 1.15rem !important;
                margin-bottom: 1rem !important;
                color: var(--color-danger-text) !important;
            }

            .validation-error-title {
                font-size: 15px; font-weight: 700;
                color: var(--color-danger-text) !important;
                margin: 0 0 0.35rem 0;
            }

            .validation-error-message {
                font-size: 14px; color: #7F1D1D !important;
                margin: 0 0 0.65rem 0; line-height: 1.45;
            }

            .validation-chip-row {
                display: flex; flex-wrap: wrap; gap: 0.4rem;
                margin-top: 0.35rem;
            }

            .validation-chip {
                display: inline-block; font-size: 12px; font-weight: 600;
                padding: 0.25rem 0.55rem; border-radius: 999px;
                background: #FFFFFF; color: var(--color-danger-text);
                border: 1px solid var(--color-danger-border);
            }

            .stSlider [data-baseweb="slider"] [role="slider"] { background: var(--color-primary) !important; }
            .stSlider [data-testid="stThumbValue"] { color: #374151 !important; font-weight: 600 !important; }

            div[data-testid="stFileUploader"] {
                background: var(--color-card); border: 1px dashed var(--color-border);
                border-radius: var(--radius-card); padding: 0.5rem;
            }

            /* ── Cards ── */
            .page-card, .section-card, .card, .insight-card, .eval-summary-card,
            .executive-summary-card, .confidence-box, .risk-meter-card,
            .retention-card, .retention-summary-card, .retention-impact-card,
            .xai-card, .cm-metric-card, .result-card-high, .result-card-low,
            .explain-card-high, .explain-card-low, .empty-state, .form-card {
                background: var(--color-card) !important;
                border: 1px solid var(--color-border) !important;
                border-radius: var(--radius-card) !important;
                color: var(--color-text-primary) !important;
                box-shadow: var(--shadow-sm) !important;
                padding: var(--card-padding) !important;
                margin-bottom: var(--card-gap) !important;
            }

            .chart-card-header { margin-bottom: 0.35rem; }

            /* Hide orphan empty card shells from split markdown blocks */
            [data-testid="stMarkdownContainer"] > div.card:empty,
            [data-testid="stMarkdownContainer"] > div.xai-card:empty,
            [data-testid="stMarkdownContainer"] > div.form-card:empty,
            [data-testid="stMarkdownContainer"] > div.insight-card:empty,
            [data-testid="stMarkdownContainer"] > div.eval-summary-card:empty,
            [data-testid="stMarkdownContainer"] > div.executive-summary-card:empty,
            [data-testid="stMarkdownContainer"] > div.confidence-box:empty,
            [data-testid="stMarkdownContainer"] > div.risk-meter-card:empty,
            [data-testid="stMarkdownContainer"] > div.retention-card:empty {
                display: none !important;
                padding: 0 !important;
                margin: 0 !important;
                border: none !important;
                box-shadow: none !important;
                min-height: 0 !important;
            }

            .main div[data-testid="stPlotlyChart"] {
                background: var(--color-card) !important;
                border: 1px solid var(--color-border) !important;
                border-radius: var(--radius-card) !important;
                box-shadow: var(--shadow-sm) !important;
                padding: 8px 8px 4px 8px !important;
                margin-bottom: var(--card-gap) !important;
                overflow: visible !important;
            }

            .main div[data-testid="stPlotlyChart"] .js-plotly-plot,
            .main div[data-testid="stPlotlyChart"] .plot-container {
                width: 100% !important;
            }

            .main div[data-testid="stPyplot"] {
                background: var(--color-card) !important;
                border: 1px solid var(--color-border) !important;
                border-radius: var(--radius-card) !important;
                box-shadow: var(--shadow-sm) !important;
                padding: 12px !important;
                margin-bottom: var(--card-gap) !important;
            }

            /* ── KPI ── */
            .kpi-card {
                background: var(--color-card) !important;
                border: 1px solid var(--color-border) !important;
                border-radius: var(--radius-card) !important;
                padding: 20px 22px !important;
                min-height: 150px; height: 100%;
                box-shadow: var(--shadow-sm) !important;
                display: flex; flex-direction: column; gap: 0.3rem;
                margin-bottom: var(--card-gap) !important;
                transition: box-shadow 0.2s ease, transform 0.2s ease;
            }

            .kpi-card:hover { box-shadow: var(--shadow-lg) !important; transform: translateY(-2px); }

            .kpi-accent-blue  { border-left: 4px solid var(--color-primary) !important; }
            .kpi-accent-green { border-left: 4px solid var(--color-success) !important; }
            .kpi-accent-amber { border-left: 4px solid var(--color-warning) !important; }
            .kpi-accent-red   { border-left: 4px solid var(--color-danger) !important; }
            .kpi-accent-violet{ border-left: 4px solid #8B5CF6 !important; }
            .kpi-accent-slate { border-left: 4px solid #64748B !important; }

            .kpi-icon-circle {
                width: 42px; height: 42px; border-radius: 12px;
                display: flex; align-items: center; justify-content: center;
                font-size: 1.1rem; margin-bottom: 4px;
            }

            .kpi-icon-circle-blue   { background: var(--color-info-bg); }
            .kpi-icon-circle-green  { background: var(--color-success-bg); }
            .kpi-icon-circle-amber  { background: var(--color-warning-bg); }
            .kpi-icon-circle-red    { background: var(--color-danger-bg); }
            .kpi-icon-circle-violet { background: #F5F3FF; }
            .kpi-icon-circle-slate  { background: var(--color-input-bg); }

            .kpi-title {
                font-size: 11px; color: var(--color-text-secondary) !important;
                font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em;
            }

            .kpi-value {
                font-size: 1.75rem; font-weight: 700;
                color: var(--color-text-primary) !important; line-height: 1.1;
            }

            .kpi-description, .kpi-trend {
                font-size: 13px; color: var(--color-text-secondary) !important; line-height: 1.35;
            }

            .kpi-trend { margin-top: auto; padding-top: 4px; font-size: 11px; font-weight: 600; }

            /* ── Typography ── */
            .page-title { font-size: 32px !important; font-weight: 700 !important; color: var(--color-text-primary) !important; margin: 0 !important; }
            .page-subtitle { font-size: 16px !important; color: var(--color-text-secondary) !important; margin: 0.3rem 0 0 0 !important; }
            .section-title { font-size: 22px !important; font-weight: 700 !important; color: var(--color-text-primary) !important; margin: 0 !important; }
            .chart-title, .card-title { font-size: 18px; font-weight: 700; color: var(--color-text-primary) !important; margin: 0 0 0.15rem 0; }
            .section-desc, .chart-desc { font-size: 14px; color: var(--color-text-secondary) !important; margin: 0.2rem 0 0 0; line-height: 1.45; }
            .section-divider { height: 1px; background: var(--color-border); margin-top: 0.6rem; }
            .section-header { margin: 0 0 0.65rem 0; }
            .section-header-top { display: flex; align-items: center; gap: 0.5rem; }
            .section-icon { font-size: 1.2rem; }

            /* ── Buttons — force readable text on all variants ── */
            .main .stButton > button,
            .main .stButton button,
            .main div[data-testid="stButton"] > button,
            .main div[data-testid="stButton"] button {
                border-radius: var(--radius-input) !important;
                min-height: 44px !important; font-weight: 600 !important;
                font-size: 15px !important; transition: all 0.18s ease !important;
                outline: none !important;
            }

            .main .stButton > button:focus-visible,
            .main .stButton button:focus-visible,
            .main div[data-testid="stButton"] > button:focus-visible,
            .main div[data-testid="stButton"] button:focus-visible {
                box-shadow: var(--shadow-glow) !important;
            }

            /* Secondary / default buttons — white background, dark readable text */
            .main .stButton > button,
            .main .stButton button,
            .main div[data-testid="stButton"] > button,
            .main div[data-testid="stButton"] button,
            .main button[data-testid="stBaseButton-secondary"],
            .main button[data-testid="baseButton-secondary"] {
                border-radius: var(--radius-input) !important;
                min-height: 44px !important;
                font-weight: 600 !important;
                font-size: 15px !important;
            }

            .main .stButton > button[kind="secondary"],
            .main .stButton button[kind="secondary"],
            .main div[data-testid="stButton"] > button[kind="secondary"],
            .main div[data-testid="stButton"] button[kind="secondary"],
            .main .stButton > button:not([kind="primary"]),
            .main .stButton button:not([kind="primary"]),
            .main div[data-testid="stButton"] > button:not([kind="primary"]),
            .main div[data-testid="stButton"] button:not([kind="primary"]),
            .main button[data-testid="stBaseButton-secondary"],
            .main button[data-testid="baseButton-secondary"] {
                background: #FFFFFF !important;
                background-color: #FFFFFF !important;
                color: #1D4ED8 !important;
                border: 1.5px solid #2563EB !important;
            }

            .main .stButton > button[kind="secondary"] p,
            .main .stButton > button[kind="secondary"] span,
            .main .stButton button[kind="secondary"] p,
            .main .stButton button[kind="secondary"] span,
            .main div[data-testid="stButton"] > button[kind="secondary"] p,
            .main div[data-testid="stButton"] > button[kind="secondary"] span,
            .main div[data-testid="stButton"] button[kind="secondary"] p,
            .main div[data-testid="stButton"] button[kind="secondary"] span,
            .main .stButton > button:not([kind="primary"]) p,
            .main .stButton > button:not([kind="primary"]) span,
            .main .stButton button:not([kind="primary"]) p,
            .main .stButton button:not([kind="primary"]) span,
            .main div[data-testid="stButton"] > button:not([kind="primary"]) p,
            .main div[data-testid="stButton"] > button:not([kind="primary"]) span,
            .main div[data-testid="stButton"] button:not([kind="primary"]) p,
            .main div[data-testid="stButton"] button:not([kind="primary"]) span,
            .main button[data-testid="stBaseButton-secondary"] p,
            .main button[data-testid="baseButton-secondary"] p {
                color: #1D4ED8 !important;
            }

            .main .stButton > button[kind="secondary"]:hover,
            .main .stButton button[kind="secondary"]:hover,
            .main div[data-testid="stButton"] > button[kind="secondary"]:hover,
            .main div[data-testid="stButton"] button[kind="secondary"]:hover,
            .main .stButton > button:not([kind="primary"]):hover,
            .main .stButton button:not([kind="primary"]):hover,
            .main div[data-testid="stButton"] > button:not([kind="primary"]):hover,
            .main div[data-testid="stButton"] button:not([kind="primary"]):hover,
            .main button[data-testid="stBaseButton-secondary"]:hover,
            .main button[data-testid="baseButton-secondary"]:hover {
                background: #EFF6FF !important;
                background-color: #EFF6FF !important;
                color: #1D4ED8 !important;
                border-color: #2563EB !important;
            }

            .main .stButton > button[kind="primary"],
            .main .stButton button[kind="primary"],
            .main div[data-testid="stButton"] > button[kind="primary"],
            .main div[data-testid="stButton"] button[kind="primary"],
            .main button[data-testid="stBaseButton-primary"],
            .main button[data-testid="baseButton-primary"] {
                background: linear-gradient(135deg, #2563EB, #1D4ED8) !important;
                background-color: #2563EB !important;
                color: #FFFFFF !important;
                border: none !important;
                box-shadow: 0 2px 8px rgba(37, 99, 235, 0.28) !important;
            }

            .main .stButton > button[kind="primary"] p,
            .main .stButton > button[kind="primary"] span,
            .main .stButton button[kind="primary"] p,
            .main .stButton button[kind="primary"] span,
            .main div[data-testid="stButton"] > button[kind="primary"] p,
            .main div[data-testid="stButton"] > button[kind="primary"] span,
            .main div[data-testid="stButton"] button[kind="primary"] p,
            .main div[data-testid="stButton"] button[kind="primary"] span,
            .main button[data-testid="stBaseButton-primary"] p,
            .main button[data-testid="baseButton-primary"] p {
                color: #FFFFFF !important;
            }

            .stDownloadButton > button,
            .stDownloadButton button {
                border-radius: var(--radius-input) !important; min-height: 44px !important;
                background: linear-gradient(135deg, var(--color-primary), var(--color-primary-dark)) !important;
                color: #FFFFFF !important; border: none !important;
                font-weight: 600 !important;
            }

            .stDownloadButton > button p,
            .stDownloadButton button p {
                color: #FFFFFF !important;
            }

            /* ── Tables — light theme, WCAG-readable cells ── */
            div[data-testid="stDataFrame"], div[data-testid="stDataEditor"] {
                background: var(--color-card); border: 1px solid var(--color-border);
                border-radius: var(--radius-card); overflow: hidden; box-shadow: var(--shadow-sm);
            }

            div[data-testid="stDataFrame"] table,
            div[data-testid="stDataFrame"] [role="grid"],
            div[data-testid="stDataFrame"] [data-testid="stDataFrameResizable"],
            div[data-testid="stDataFrame"] [data-testid="glideDataEditor"] {
                background: #FFFFFF !important;
                color: #1F2937 !important;
            }

            div[data-testid="stDataFrame"] [data-testid="stEmptyTableSparkline"] {
                color: #6B7280 !important;
                background: #FFFFFF !important;
            }

            div[data-testid="stDataFrame"] thead tr th,
            div[data-testid="stDataFrame"] th {
                position: sticky; top: 0;
                background: #F3F4F6 !important;
                color: #374151 !important;
                font-weight: 700 !important; font-size: 14px !important; z-index: 1;
                border-bottom: 1px solid var(--color-border) !important;
            }

            div[data-testid="stDataFrame"] tbody tr td,
            div[data-testid="stDataFrame"] td {
                font-size: 14px !important;
                color: #1F2937 !important;
                background-color: #FFFFFF !important;
                border-color: var(--color-border) !important;
            }

            div[data-testid="stDataFrame"] tbody tr:nth-child(even) td {
                background-color: #F9FAFB !important;
                color: #1F2937 !important;
            }

            /* Pandas Styler highlighted rows — dark text on light blue */
            div[data-testid="stDataFrame"] td[style*="background-color: rgb(219, 234, 254)"],
            div[data-testid="stDataFrame"] td[style*="background-color: #dbeafe"],
            div[data-testid="stDataFrame"] td[style*="background-color: #DBEAFE"] {
                color: #1F2937 !important;
                font-weight: 600 !important;
            }

            div[data-testid="stCaptionContainer"] p,
            div[data-testid="stCaptionContainer"] small {
                color: #6B7280 !important;
                font-size: 13px !important;
            }

            /* ── Alerts — force readable dark text on all backgrounds ── */
            div[data-testid="stAlert"],
            div[data-testid="stException"],
            div[data-testid="stSuccess"],
            div[data-testid="stInfo"] {
                border-radius: var(--radius-card) !important;
                padding: 0.75rem 1rem !important;
            }

            div[data-testid="stAlert"] p,
            div[data-testid="stAlert"] [data-testid="stMarkdownContainer"],
            div[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p,
            div[data-testid="stAlert"] [data-testid="stMarkdownContainer"] span,
            div[data-testid="stException"] p,
            div[data-testid="stSuccess"] p,
            div[data-testid="stSuccess"] [data-testid="stMarkdownContainer"] p,
            div[data-testid="stInfo"] p {
                color: #422006 !important;
                font-size: 14px !important;
                line-height: 1.45 !important;
                font-weight: 600 !important;
            }

            div[data-testid="stException"] p {
                color: #991B1B !important;
            }

            div[data-testid="stSuccess"] p,
            div[data-testid="stSuccess"] [data-testid="stMarkdownContainer"] p {
                color: #166534 !important;
            }

            /* ── Horizontal radio — compact segmented look ── */
            div[data-testid="stRadio"] > div[role="radiogroup"] {
                gap: 0.35rem !important;
            }

            div[data-testid="stRadio"] label {
                background: var(--color-input-bg) !important;
                border: 1px solid var(--color-input-border) !important;
                border-radius: 8px !important;
                padding: 0.3rem 0.65rem !important;
                margin-right: 0.35rem !important;
                cursor: pointer !important;
            }

            div[data-testid="stRadio"] label:has(input:checked) {
                background: var(--color-primary-soft) !important;
                border-color: var(--color-primary) !important;
            }

            div[data-testid="stRadio"] label:has(input:checked) p,
            div[data-testid="stRadio"] label:has(input:checked) span {
                color: var(--color-primary) !important;
                font-weight: 600 !important;
            }

            div[data-testid="stRadio"] input:focus-visible + div {
                outline: 2px solid var(--color-primary) !important;
                outline-offset: 2px !important;
            }

            /* ── Keyboard focus — accessibility ── */
            .main input:focus-visible,
            .main textarea:focus-visible,
            .main select:focus-visible {
                outline: 2px solid var(--color-primary) !important;
                outline-offset: 2px !important;
            }

            div[data-testid="stSidebar"] .stButton > button p,
            div[data-testid="stSidebar"] .stButton > button span,
            div[data-testid="stSidebar"] .stButton button p {
                color: inherit !important;
            }

            div[data-testid="stSidebar"] .stButton > button:focus-visible,
            div[data-testid="stSidebar"] .stButton button:focus-visible {
                outline: 2px solid var(--color-primary) !important;
                outline-offset: 2px !important;
            }

            /* ── Badges & insight cards ── */
            .status-badge {
                display: inline-block; font-size: 10px; font-weight: 700;
                padding: 3px 9px; border-radius: 999px; text-transform: uppercase;
            }

            .status-badge-success { background: var(--color-success-bg); color: var(--color-success-text); border: 1px solid var(--color-success-border); }
            .status-badge-warning { background: var(--color-warning-bg); color: var(--color-warning-text); border: 1px solid var(--color-warning-border); }
            .status-badge-danger  { background: var(--color-danger-bg); color: var(--color-danger-text); border: 1px solid var(--color-danger-border); }
            .status-badge-info    { background: var(--color-info-bg); color: var(--color-info-text); border: 1px solid var(--color-info-border); }
            .status-badge-prediction-churn { background: var(--color-danger-bg); color: var(--color-danger-text); border: 1px solid var(--color-danger-border); }
            .status-badge-prediction-safe  { background: var(--color-success-bg); color: var(--color-success-text); border: 1px solid var(--color-success-border); }
            .status-badge-risk-high { background: var(--color-warning-bg); color: var(--color-warning-text); border: 1px solid var(--color-warning-border); }
            .status-badge-risk-low  { background: var(--color-success-bg); color: var(--color-success-text); border: 1px solid var(--color-success-border); }
            .status-badge-priority-high   { background: var(--color-danger-bg); color: var(--color-danger-text); }
            .status-badge-priority-medium { background: var(--color-warning-bg); color: var(--color-warning-text); }
            .status-badge-priority-low    { background: var(--color-info-bg); color: var(--color-info-text); }
            .status-badge-model { background: var(--color-info-bg); color: var(--color-info-text); border: 1px solid var(--color-info-border); }

            .empty-state { text-align: center; border-style: dashed !important; padding: 2.5rem var(--card-padding) !important; }
            .empty-state-icon { font-size: 2.75rem; margin-bottom: 0.5rem; }
            .empty-state-title { font-size: 18px; font-weight: 700; color: var(--color-text-primary) !important; }
            .empty-state-message { font-size: 14px; color: var(--color-text-secondary) !important; line-height: 1.55; }

            .result-card-high { border-left: 4px solid var(--color-danger) !important; }
            .result-card-low  { border-left: 4px solid var(--color-success) !important; }
            .explain-card-high{ border-left: 4px solid var(--color-warning) !important; }
            .explain-card-low { border-left: 4px solid var(--color-success) !important; }

            .metric-label { font-size: 14px; color: var(--color-text-secondary) !important; }
            .metric-value { font-size: 1.5rem; font-weight: 700; color: var(--color-text-primary) !important; }

            .risk-meter-title, .risk-meter-level, .risk-meter-score-value,
            .risk-meter-score-label, .risk-meter-description, .risk-meter-labels span {
                color: var(--color-text-primary) !important;
            }

            .risk-meter-level, .risk-meter-score-value { font-weight: 700; font-size: 1.25rem; }

            .confidence-bar-track { background: var(--color-border); border-radius: 999px; height: 10px; margin: 0.5rem 0 0.65rem 0; }
            .confidence-bar-fill-high, .confidence-bar-fill-low {
                background: linear-gradient(90deg, var(--color-primary), var(--color-primary-dark)) !important;
                height: 10px; border-radius: 999px;
            }

            .confidence-value-large { font-size: 1.75rem; font-weight: 700; color: var(--color-text-primary) !important; }

            .confidence-stat, .explain-factor-item, .retention-meta,
            .executive-summary-item, .eval-summary-item, .xai-explanation-item,
            .retention-title {
                color: var(--color-text-primary) !important; font-size: 15px; line-height: 1.5;
            }

            .risk-meter-track {
                height: 12px; border-radius: 999px; margin: 0.55rem 0 0.3rem 0;
                background: linear-gradient(90deg, var(--color-success), var(--color-warning), var(--color-danger));
            }

            .risk-meter-marker {
                position: absolute; top: -4px; width: 3px; height: 20px;
                background: var(--color-text-primary); border-radius: 3px;
                transform: translateX(-50%); box-shadow: 0 0 0 2px var(--color-card);
            }

            .risk-meter-score-label, .risk-meter-labels { color: var(--color-text-secondary) !important; font-size: 13px; }

            .retention-card-high   { background: var(--color-danger-bg) !important; border-color: var(--color-danger-border) !important; }
            .retention-card-medium { background: var(--color-warning-bg) !important; border-color: var(--color-warning-border) !important; }
            .retention-card-low    { background: var(--color-info-bg) !important; border-color: var(--color-info-border) !important; }

            .priority-badge-high   { background: var(--color-danger-bg); color: var(--color-danger-text); }
            .priority-badge-medium { background: var(--color-warning-bg); color: var(--color-warning-text); }
            .priority-badge-low    { background: var(--color-info-bg); color: var(--color-info-text); }

            .premium-footer {
                text-align: center; color: var(--color-text-secondary) !important;
                font-size: 13px; margin-top: 1.25rem; padding-top: 0.85rem;
                border-top: 1px solid var(--color-border);
            }

            .premium-footer-title { font-weight: 700; color: var(--color-text-primary) !important; }

            /* ── Plotly text visibility ── */
            .js-plotly-plot text, .js-plotly-plot .ytitle, .js-plotly-plot .xtitle {
                fill: var(--color-text-primary) !important;
                color: var(--color-text-primary) !important;
            }

            .js-plotly-plot .xtick text, .js-plotly-plot .ytick text {
                fill: var(--color-text-secondary) !important;
            }

            @media (max-width: 1280px) {
                .history-filter-scope [data-testid="stHorizontalBlock"] > div[data-testid="column"] {
                    flex: 1 1 48% !important;
                    min-width: 180px !important;
                }
            }

            @media (max-width: 1366px) {
                .main .block-container { padding-left: 1.25rem; padding-right: 1.25rem; }
                .page-title { font-size: 28px !important; }
                .kpi-value { font-size: 1.45rem; }
                .app-header-subtitle { display: none; }
            }

            @media (max-width: 1024px) {
                .main .block-container { padding-left: 1rem; padding-right: 1rem; }
                .kpi-card { min-height: 130px; padding: 16px 18px !important; }
                .history-table-wrap { overflow-x: auto; }
                .history-filter-scope [data-testid="stHorizontalBlock"] > div[data-testid="column"] {
                    flex: 1 1 100% !important;
                    min-width: 100% !important;
                }
            }

            @media (max-width: 992px) {
                .app-header { height: auto; max-height: none; padding: 12px 16px !important; }
                .page-title, .section-title { font-size: 20px !important; }
                .app-header-meta { width: 100%; justify-content: flex-start !important; }
            }
        </style>
    """


def inject_app_styles() -> None:
    """Inject the enterprise design-system styles."""
    st.markdown(get_app_styles(), unsafe_allow_html=True)
    inject_sidebar_collapse_styles()


def inject_sidebar_collapse_styles() -> None:
    """Inject expanded/collapsed layout marker for the enterprise sidebar."""
    collapsed = bool(st.session_state.get(SIDEBAR_COLLAPSED_KEY, False))
    marker_class = "sidebar-is-collapsed" if collapsed else "sidebar-is-expanded"
    st.markdown(f'<span class="{marker_class}" aria-hidden="true"></span>', unsafe_allow_html=True)


def is_sidebar_collapsed() -> bool:
    """Return whether the sidebar is in icon-rail mode."""
    return bool(st.session_state.get(SIDEBAR_COLLAPSED_KEY, False))


def render_sidebar_collapse_toggle() -> None:
    """Render the collapse/expand control at the top of the sidebar."""
    collapsed = is_sidebar_collapsed()
    toggle_label = "»" if collapsed else "«"
    toggle_help = "Expand sidebar" if collapsed else "Collapse to icon rail"
    st.markdown('<span class="sidebar-collapse-toggle-wrap"></span>', unsafe_allow_html=True)
    if st.button(
        toggle_label,
        key="sidebar_collapse_toggle",
        help=toggle_help,
        use_container_width=True,
    ):
        st.session_state[SIDEBAR_COLLAPSED_KEY] = not collapsed
        st.rerun()


def render_sidebar_navigation() -> str:
    """Render enterprise navigation menu buttons."""
    if NAVIGATION_KEY not in st.session_state:
        st.session_state[NAVIGATION_KEY] = "🔮 Churn Prediction"

    current_page = st.session_state[NAVIGATION_KEY]
    collapsed = is_sidebar_collapsed()

    for page_id, icon, label in NAV_ITEMS:
        is_active = current_page == page_id
        button_label = icon if collapsed else f"{icon}   {label}"
        if st.button(
            button_label,
            key=f"nav_btn_{page_id}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
            help=label if collapsed else None,
        ):
            st.session_state[NAVIGATION_KEY] = page_id
            current_page = page_id
            st.rerun()

    return str(current_page)


def render_app_header(current_page: str, model_loaded: bool = True) -> None:
    """Render the compact global header."""
    page_label = PAGE_LABELS.get(current_page, current_page)
    timestamp = datetime.now().strftime("%A, %B %d, %Y · %H:%M")
    status_class = "online" if model_loaded else "offline"
    status_label = "Model Ready" if model_loaded else "Model Offline"

    st.markdown(
        f"""
        <div class="app-header">
            <div class="app-header-top">
                <div class="app-header-brand">
                    <div class="app-header-logo">{APP_LOGO}</div>
                    <div>
                        <p class="app-header-title">{HEADER_PLATFORM_NAME}</p>
                        <p class="app-header-subtitle">{PLATFORM_TAGLINE}</p>
                    </div>
                </div>
                <div class="app-header-meta">
                    <span class="header-status">
                        <span class="header-status-dot {status_class}"></span>
                        {status_label}
                    </span>
                    <span class="page-badge">{page_label}</span>
                    <span class="header-datetime">{timestamp}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(icon: str, title: str, subtitle: str = "") -> None:
    """Render a page-level hero card."""
    subtitle_html = f'<p class="page-subtitle">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f"""
        <div class="page-card">
            <div class="section-header-top">
                <span class="section-icon">{icon}</span>
                <h1 class="page-title">{title}</h1>
            </div>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(icon: str, title: str, description: str = "") -> None:
    """Render a section header with divider."""
    description_html = f'<p class="section-desc">{description}</p>' if description else ""
    st.markdown(
        f"""
        <div class="section-header">
            <div class="section-header-top">
                <span class="section-icon">{icon}</span>
                <h3 class="section-title">{title}</h3>
            </div>
            {description_html}
            <div class="section-divider"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chart_header(title: str = "", description: str = "") -> None:
    """Render chart card header above a Plotly chart."""
    if not title and not description:
        return
    title_html = f'<p class="chart-title">{title}</p>' if title else ""
    desc_html = f'<p class="chart-desc">{description}</p>' if description else ""
    st.markdown(
        f'<div class="chart-card-header">{title_html}{desc_html}</div>',
        unsafe_allow_html=True,
    )


def open_chart_card(title: str = "", description: str = "") -> None:
    """Render chart header (card body styled via global Plotly CSS)."""
    render_chart_header(title, description)


def close_chart_card() -> None:
    """Legacy no-op — Streamlit cannot nest widgets inside HTML divs."""


def render_sidebar_brand() -> None:
    """Render sidebar brand block."""
    st.markdown(
        f"""
        <div class="sidebar-brand">
            <div class="sidebar-logo-placeholder">{APP_LOGO}</div>
            <div class="sidebar-brand-title">{PLATFORM_SHORT_NAME}</div>
            <div class="sidebar-brand-version">Version {APP_VERSION}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_section_label(label: str) -> None:
    """Render a sidebar section label."""
    st.markdown(f'<div class="sidebar-section-label">{label}</div>', unsafe_allow_html=True)


def render_sidebar_system_info(
    algorithm: str,
    model_loaded: bool,
    prediction_count: int,
) -> None:
    """Render system information in the sidebar."""
    status_badge = render_status_badge_html(
        "success" if model_loaded else "danger",
        "MODEL READY" if model_loaded else "NOT AVAILABLE",
    )
    st.markdown(
        f"""
        <div class="sidebar-meta-card">
            <div class="sidebar-meta-row"><span>Dataset</span><strong>{DATASET_NAME}</strong></div>
            <div class="sidebar-meta-row"><span>Model</span><strong>{algorithm}</strong></div>
            <div class="sidebar-meta-row"><span>Status</span><strong>{status_badge}</strong></div>
            <div class="sidebar-meta-row"><span>Predictions</span><strong>{prediction_count:,}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_metadata(
    algorithm: str,
    model_loaded: bool,
    prediction_count: int,
) -> None:
    """Render sidebar system info and footer."""
    render_sidebar_section_label("System Information")
    render_sidebar_system_info(algorithm, model_loaded, prediction_count)
    render_sidebar_footer()


def render_sidebar_footer() -> None:
    """Render sidebar footer."""
    st.markdown(
        f"""
        <div class="sidebar-footer">
            <div class="sidebar-footer-text">{DEVELOPER_NAME}</div>
            <div class="sidebar-footer-text">© {CURRENT_YEAR} · v{APP_VERSION}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_badge_html(badge_type: str, label: str) -> str:
    """Build HTML for a status badge."""
    return f'<span class="status-badge status-badge-{badge_type}">{label}</span>'


def render_status_badge(badge_type: str, label: str) -> None:
    """Render a status badge."""
    st.markdown(render_status_badge_html(badge_type, label), unsafe_allow_html=True)


def render_kpi_card(
    title: str,
    value: str,
    icon: str = "",
    description: str = "",
    trend: str = "",
    color_index: int = 0,
) -> None:
    """Render a premium KPI card."""
    accent = KPI_ACCENTS[color_index % len(KPI_ACCENTS)]
    icon_html = (
        f'<div class="kpi-icon-circle kpi-icon-circle-{accent}">{icon}</div>'
        if icon else ""
    )
    description_html = f'<div class="kpi-description">{description}</div>' if description else ""
    trend_html = f'<div class="kpi-trend">{trend}</div>' if trend else ""

    st.markdown(
        f"""
        <div class="kpi-card kpi-accent-{accent}">
            {icon_html}
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            {description_html}
            {trend_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state(
    icon: str,
    title: str,
    message: str,
    action_label: Optional[str] = None,
    action_page: Optional[str] = None,
) -> None:
    """Render a professional empty state with optional navigation action."""
    st.markdown(
        f"""
        <div class="empty-state">
            <div class="empty-state-icon">{icon}</div>
            <div class="empty-state-title">{title}</div>
            <div class="empty-state-message">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if action_label and action_page:
        if st.button(action_label, type="primary", use_container_width=True, key=f"empty_{action_page}"):
            st.session_state[NAVIGATION_KEY] = action_page
            st.rerun()


def render_premium_footer() -> None:
    """Render the main content footer."""
    st.markdown(
        f"""
        <div class="premium-footer">
            <div class="premium-footer-title">{PLATFORM_NAME}</div>
            <div>Developer: {DEVELOPER_NAME}</div>
            <div>Python · Streamlit · Scikit-learn · Pandas · Plotly · ReportLab</div>
            <div>Version {APP_VERSION} · © {CURRENT_YEAR} {DEVELOPER_NAME}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_page_label(current_page: str) -> str:
    """Return a clean page label."""
    return PAGE_LABELS.get(current_page, current_page)
