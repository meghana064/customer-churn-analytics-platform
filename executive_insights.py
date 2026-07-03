"""
Executive business insights analytics for the churn retention platform.

Computes KPIs, risk profiles, retention strategy, visuals, and management takeaways
from the Telco dataset without Streamlit or ML training dependencies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from preprocess import load_dataset

EXECUTIVE_DATA_PATH = Path("data") / "Telco-Customer-Churn.csv"

PLOTLY_INSIGHTS_LAYOUT: Dict[str, Any] = {
    "template": "plotly_white",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "#374151", "size": 12},
    "margin": {"l": 24, "r": 24, "t": 56, "b": 24},
    "height": 380,
    "colorway": ["#2563eb", "#10b981", "#ef4444", "#f59e0b", "#8b5cf6"],
}


def load_insights_dataframe(csv_path: str | Path = EXECUTIVE_DATA_PATH) -> pd.DataFrame:
    """
    Load and prepare the Telco dataset for executive business insights.

    Args:
        csv_path: Path to the IBM Telco Customer Churn CSV.

    Returns:
        Prepared dataframe ready for insight calculations.
    """
    dataframe = load_dataset(csv_path).copy()
    dataframe["MonthlyCharges"] = pd.to_numeric(dataframe["MonthlyCharges"], errors="coerce")
    dataframe["TotalCharges"] = pd.to_numeric(dataframe["TotalCharges"], errors="coerce")
    dataframe["tenure"] = pd.to_numeric(dataframe["tenure"], errors="coerce")
    dataframe["SeniorCitizenLabel"] = dataframe["SeniorCitizen"].map({0: "No", 1: "Yes"})
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
    Calculate headline KPI metrics for executive insights.

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


def _build_churn_rate_summary(dataframe: pd.DataFrame, column: str) -> pd.DataFrame:
    """Aggregate churn rate by a categorical dimension."""
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


def _top_category_by_churn_rate(
    dataframe: pd.DataFrame,
    column: str,
    explanation_template: str,
) -> Dict[str, Any]:
    """Return the highest churn-rate category for a given column."""
    summary = _build_churn_rate_summary(dataframe, column)
    top_row = summary.iloc[0]
    category = str(top_row[column])
    churn_rate = float(top_row["churn_rate"])
    return {
        "label": column.replace("_", " "),
        "value": category,
        "risk_percentage": churn_rate,
        "explanation": explanation_template.format(category=category, rate=churn_rate),
    }


def _top_revenue_loss_category(dataframe: pd.DataFrame) -> Dict[str, Any]:
    """Identify the contract type with the highest monthly revenue at risk."""
    churned = dataframe[dataframe["Churn"] == "Yes"]
    revenue_summary = (
        churned.groupby("Contract", dropna=False)["MonthlyCharges"]
        .sum()
        .reset_index(name="revenue_at_risk")
        .sort_values("revenue_at_risk", ascending=False)
    )
    top_row = revenue_summary.iloc[0]
    contract = str(top_row["Contract"])
    revenue = float(top_row["revenue_at_risk"])
    contract_churn = _build_churn_rate_summary(dataframe, "Contract")
    contract_rate = float(
        contract_churn.loc[contract_churn["Contract"] == contract, "churn_rate"].iloc[0]
    )
    return {
        "label": "Highest Monthly Revenue Loss Category",
        "value": contract,
        "risk_percentage": contract_rate,
        "revenue_at_risk": revenue,
        "explanation": (
            f"**{contract}** customers represent the largest monthly revenue exposure "
            f"at **${revenue:,.2f}**, with a **{contract_rate:.1f}%** churn rate."
        ),
    }


def _top_customer_segment(dataframe: pd.DataFrame) -> Dict[str, Any]:
    """Identify the highest-churn composite customer segment."""
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
    segment_summary = segment_summary[segment_summary["customers"] >= 20]
    if segment_summary.empty:
        return {
            "label": "Highest-Risk Customer Segment",
            "value": "Not available",
            "risk_percentage": 0.0,
            "explanation": "Insufficient segment volume to identify a reliable high-risk segment.",
        }

    top_row = segment_summary.sort_values("churn_rate", ascending=False).iloc[0]
    segment = str(top_row["CustomerSegment"])
    churn_rate = float(top_row["churn_rate"])
    return {
        "label": "Highest-Risk Customer Segment",
        "value": segment,
        "risk_percentage": churn_rate,
        "explanation": (
            f"The segment **{segment}** shows the highest churn rate at **{churn_rate:.1f}%**, "
            "making it a priority target for retention campaigns."
        ),
    }


def identify_top_business_risks(dataframe: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Identify top business risk categories across contract, payment, service, and segments.

    Args:
        dataframe: Prepared Telco customer dataset.

    Returns:
        List of risk dictionaries with value, percentage, and explanation.
    """
    return [
        _top_category_by_churn_rate(
            dataframe,
            "Contract",
            "**{category}** contracts carry the highest churn rate at **{rate:.1f}%**, "
            "indicating weak long-term commitment among these customers.",
        ),
        _top_category_by_churn_rate(
            dataframe,
            "PaymentMethod",
            "**{category}** is the highest-risk payment method with a **{rate:.1f}%** churn rate, "
            "suggesting billing friction or lower payment stickiness.",
        ),
        _top_category_by_churn_rate(
            dataframe,
            "InternetService",
            "**{category}** internet service shows the highest churn at **{rate:.1f}%**, "
            "signaling product satisfaction or pricing pressure in this tier.",
        ),
        _top_customer_segment(dataframe),
        _top_revenue_loss_category(dataframe),
    ]


def build_executive_summary_bullets(
    kpis: Dict[str, float],
    risks: List[Dict[str, Any]],
    avg_churn_probability: Optional[float] = None,
) -> List[str]:
    """
    Generate 3–5 executive summary bullet points from KPIs and risk insights.

    Args:
        kpis: Executive KPI dictionary.
        risks: Top business risk items.
        avg_churn_probability: Optional average model-predicted churn probability.

    Returns:
        List of executive summary bullet strings.
    """
    contract_risk = risks[0]
    payment_risk = risks[1]
    revenue_risk = risks[4]

    bullets = [
        (
            f"The customer base includes **{kpis['total_customers']:,.0f}** accounts with an "
            f"overall churn rate of **{kpis['churn_rate']:.1f}%**, representing "
            f"**{kpis['churn_customers']:,.0f}** departing customers."
        ),
        (
            f"Estimated monthly revenue at risk from churned customers is "
            f"**${kpis['revenue_at_risk']:,.2f}**, with **{revenue_risk['value']}** "
            "driving the largest revenue loss category."
        ),
        (
            f"**{contract_risk['value']}** contracts and **{payment_risk['value']}** payments "
            f"are the highest-risk commercial levers at **{contract_risk['risk_percentage']:.1f}%** "
            f"and **{payment_risk['risk_percentage']:.1f}%** churn respectively."
        ),
        (
            f"Average customer tenure is **{kpis['avg_tenure']:.1f} months** with mean monthly "
            f"charges of **${kpis['avg_monthly_charges']:.2f}**, highlighting both lifecycle "
            "and pricing dynamics in retention planning."
        ),
    ]

    if avg_churn_probability is not None:
        bullets.append(
            f"The trained model estimates an average churn probability of "
            f"**{avg_churn_probability:.1%}** across the portfolio, supporting proactive "
            "prioritization of at-risk accounts."
        )
    else:
        bullets.append(
            "Deploying predictive scoring alongside these descriptive insights enables "
            "proactive outreach before customers cancel."
        )

    return bullets[:5]


def build_retention_strategy(
    risks: List[Dict[str, Any]],
    kpis: Dict[str, float],
) -> Dict[str, List[Dict[str, str]]]:
    """
    Build prioritized retention strategy recommendations for executives.

    Args:
        risks: Top business risk items.
        kpis: Executive KPI dictionary.

    Returns:
        Dictionary keyed by priority level with action and impact items.
    """
    contract_risk = risks[0]["value"]
    payment_risk = risks[1]["value"]
    internet_risk = risks[2]["value"]
    segment_risk = risks[3]["value"]

    return {
        "High Priority": [
            {
                "action": f"Launch contract upgrade campaigns targeting **{contract_risk}** customers.",
                "impact": "Reduce short-term churn and stabilize recurring revenue.",
            },
            {
                "action": (
                    f"Introduce retention offers for **{internet_risk}** subscribers "
                    "with service quality reviews."
                ),
                "impact": "Address the highest churn product line and improve satisfaction.",
            },
            {
                "action": (
                    f"Prioritize outreach to the segment **{segment_risk}** "
                    "with tailored save offers."
                ),
                "impact": "Focus retention spend on the highest-risk customer cohort.",
            },
        ],
        "Medium Priority": [
            {
                "action": f"Promote automatic payment enrollment for **{payment_risk}** users.",
                "impact": "Improve payment retention and reduce involuntary churn.",
            },
            {
                "action": "Deploy loyalty incentives for customers under 12 months tenure.",
                "impact": "Strengthen early lifecycle retention during onboarding.",
            },
            {
                "action": "Review pricing for customers above average monthly charges.",
                "impact": "Mitigate price-driven churn among high-ARPU accounts.",
            },
        ],
        "Low Priority": [
            {
                "action": "Maintain quarterly satisfaction surveys for stable customer cohorts.",
                "impact": "Detect emerging dissatisfaction before cancellation.",
            },
            {
                "action": "Expand referral and renewal programs for low-risk segments.",
                "impact": "Increase lifetime value without heavy discounting.",
            },
            {
                "action": "Monitor KPI trends monthly through the executive dashboard.",
                "impact": f"Sustain visibility on churn rate (currently {kpis['churn_rate']:.1f}%).",
            },
        ],
    }


def build_management_takeaways(
    kpis: Dict[str, float],
    risks: List[Dict[str, Any]],
    strategy: Dict[str, List[Dict[str, str]]],
) -> Dict[str, List[str]]:
    """
    Generate management takeaway sections for strengths, risks, and actions.

    Args:
        kpis: Executive KPI dictionary.
        risks: Top business risk items.
        strategy: Retention strategy dictionary.

    Returns:
        Dictionary with strengths, risks, immediate_actions, and long_term_strategy lists.
    """
    retained_customers = kpis["total_customers"] - kpis["churn_customers"]
    retention_rate = 100 - kpis["churn_rate"]

    return {
        "strengths": [
            f"**{retained_customers:,.0f}** customers remain active ({retention_rate:.1f}% retention rate).",
            "Rich customer attributes enable targeted retention by contract, payment, and service type.",
            "Integrated prediction, batch scoring, and explainability support data-driven decisions.",
            "Executive dashboards provide real-time visibility into churn and revenue exposure.",
        ],
        "risks": [
            f"Overall churn rate of **{kpis['churn_rate']:.1f}%** creates **${kpis['revenue_at_risk']:,.2f}** monthly revenue at risk.",
            f"**{risks[0]['value']}** contracts remain the highest churn contract type.",
            f"**{risks[1]['value']}** customers show elevated payment-related churn risk.",
            f"Short average tenure (**{kpis['avg_tenure']:.1f} months**) increases vulnerability in early lifecycle.",
        ],
        "immediate_actions": [
            item["action"] for item in strategy["High Priority"][:3]
        ],
        "long_term_strategy": [
            "Build predictive retention playbooks tied to churn probability tiers.",
            "Invest in contract migration programs from month-to-month to annual plans.",
            "Expand value-added services that increase switching costs and satisfaction.",
            "Establish a recurring executive review cadence for churn KPIs and campaign ROI.",
        ],
    }


def _apply_plotly_layout(figure: go.Figure, height: int = 380) -> go.Figure:
    """Apply consistent styling to insight charts."""
    layout = dict(PLOTLY_INSIGHTS_LAYOUT)
    layout["height"] = height
    figure.update_layout(**layout)
    return figure


def create_executive_visuals(dataframe: pd.DataFrame) -> Dict[str, go.Figure]:
    """
    Build Plotly charts for the executive business insights page.

    Args:
        dataframe: Prepared Telco customer dataset.

    Returns:
        Dictionary of Plotly figures keyed by chart name.
    """
    churned = dataframe[dataframe["Churn"] == "Yes"]

    revenue_by_contract = (
        churned.groupby("Contract", dropna=False)["MonthlyCharges"]
        .sum()
        .reset_index(name="revenue_at_risk")
        .sort_values("revenue_at_risk", ascending=False)
    )
    revenue_chart = px.bar(
        revenue_by_contract,
        x="Contract",
        y="revenue_at_risk",
        text=revenue_by_contract["revenue_at_risk"].map(lambda value: f"${value:,.0f}"),
        title="Revenue at Risk by Contract",
        labels={"Contract": "Contract Type", "revenue_at_risk": "Monthly Revenue at Risk ($)"},
        color_discrete_sequence=["#ef4444"],
    )
    revenue_chart.update_traces(textposition="outside")

    payment_summary = _build_churn_rate_summary(dataframe, "PaymentMethod")
    payment_chart = px.bar(
        payment_summary,
        x="PaymentMethod",
        y="churn_rate",
        text=payment_summary["churn_rate"].map(lambda value: f"{value:.1f}%"),
        title="Churn by Payment Method",
        labels={"PaymentMethod": "Payment Method", "churn_rate": "Churn Rate (%)"},
        color_discrete_sequence=["#2563eb"],
    )
    payment_chart.update_traces(textposition="outside")
    payment_chart.update_yaxes(range=[0, max(100, payment_summary["churn_rate"].max() + 5)])

    internet_summary = _build_churn_rate_summary(dataframe, "InternetService")
    internet_chart = px.bar(
        internet_summary,
        x="InternetService",
        y="churn_rate",
        text=internet_summary["churn_rate"].map(lambda value: f"{value:.1f}%"),
        title="Churn by Internet Service",
        labels={"InternetService": "Internet Service", "churn_rate": "Churn Rate (%)"},
        color_discrete_sequence=["#10b981"],
    )
    internet_chart.update_traces(textposition="outside")
    internet_chart.update_yaxes(range=[0, max(100, internet_summary["churn_rate"].max() + 5)])

    charges_comparison = dataframe.copy()
    charges_comparison["Churn Label"] = charges_comparison["Churn"].map(
        {"Yes": "Churned", "No": "Retained"}
    )
    charges_chart = px.box(
        charges_comparison,
        x="Churn Label",
        y="MonthlyCharges",
        color="Churn Label",
        title="Monthly Charges vs Churn",
        labels={"Churn Label": "Customer Status", "MonthlyCharges": "Monthly Charges ($)"},
        color_discrete_map={"Churned": "#ef4444", "Retained": "#10b981"},
    )

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
    segment_summary = segment_summary[segment_summary["customers"] >= 20]
    top_segments = segment_summary.sort_values("churn_rate", ascending=False).head(10)
    top_segments_sorted = top_segments.sort_values("churn_rate", ascending=True)
    segments_chart = px.bar(
        top_segments_sorted,
        x="churn_rate",
        y="CustomerSegment",
        orientation="h",
        text=top_segments_sorted["churn_rate"].map(lambda value: f"{value:.1f}%"),
        title="Top Risk Segments",
        labels={"CustomerSegment": "Customer Segment", "churn_rate": "Churn Rate (%)"},
        color_discrete_sequence=["#f59e0b"],
    )
    segments_chart.update_traces(textposition="outside")

    return {
        "revenue_at_risk_by_contract": _apply_plotly_layout(revenue_chart),
        "churn_by_payment_method": _apply_plotly_layout(payment_chart),
        "churn_by_internet_service": _apply_plotly_layout(internet_chart),
        "monthly_charges_vs_churn": _apply_plotly_layout(charges_chart),
        "top_risk_segments": _apply_plotly_layout(segments_chart, height=460),
    }


def compute_executive_insights(
    dataframe: pd.DataFrame,
    avg_churn_probability: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute the full executive business insights payload.

    Args:
        dataframe: Prepared Telco customer dataset.
        avg_churn_probability: Optional average model-predicted churn probability.

    Returns:
        Dictionary containing KPIs, summary bullets, risks, strategy, takeaways, and charts.
    """
    kpis = compute_executive_kpis(dataframe)
    risks = identify_top_business_risks(dataframe)
    summary_bullets = build_executive_summary_bullets(kpis, risks, avg_churn_probability)
    strategy = build_retention_strategy(risks, kpis)
    takeaways = build_management_takeaways(kpis, risks, strategy)
    visuals = create_executive_visuals(dataframe)

    return {
        "kpis": kpis,
        "summary_bullets": summary_bullets,
        "top_risks": risks,
        "retention_strategy": strategy,
        "management_takeaways": takeaways,
        "visuals": visuals,
    }
