"""
PDF report generation for Customer Churn Prediction results.

Uses ReportLab to build professional prediction summary reports.
Kept separate from Streamlit UI rendering logic.
"""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PLATFORM_NAME = "Customer Churn Analytics & Retention Intelligence Platform"
REPORT_TITLE = "Customer Churn Prediction Report"


def build_prediction_summary(
    customer_data: Dict[str, Any],
    prediction: str,
    confidence: float,
    churn_probability: float,
    stay_probability: float,
    risk_level: str,
    explanation_factors: List[str],
    recommendations: List[Dict[str, str]],
    business_impact: Dict[str, Any],
    generated_at: datetime | None = None,
) -> Dict[str, Any]:
    """
    Assemble structured report data for PDF generation.

    Args:
        customer_data: Customer feature dictionary from the prediction form.
        prediction: Model prediction label.
        confidence: Model confidence score (0.0 to 1.0).
        churn_probability: Predicted churn probability.
        stay_probability: Predicted stay probability.
        risk_level: Assigned customer risk level label.
        explanation_factors: Explanation factor strings shown in the app.
        recommendations: Generated retention recommendation dictionaries.
        business_impact: Estimated business impact metrics.
        generated_at: Report generation timestamp.

    Returns:
        Dictionary payload consumed by ``generate_prediction_pdf``.
    """
    senior_label = "Yes" if int(customer_data.get("SeniorCitizen", 0)) == 1 else "No"

    return {
        "customer_info": {
            "Gender": str(customer_data.get("gender", "N/A")).title(),
            "Senior Citizen": senior_label,
            "Contract": str(customer_data.get("Contract", "N/A")),
            "Internet Service": str(customer_data.get("InternetService", "N/A")),
            "Payment Method": str(customer_data.get("PaymentMethod", "N/A")),
            "Monthly Charges": f"${float(customer_data.get('MonthlyCharges', 0)):,.2f}",
            "Tenure": f"{int(customer_data.get('tenure', 0))} months",
        },
        "prediction_summary": {
            "Prediction": prediction,
            "Churn Probability": f"{churn_probability:.1%}",
            "Stay Probability": f"{stay_probability:.1%}",
            "Confidence": f"{confidence:.1%}",
            "Risk Level": risk_level,
        },
        "explanation_factors": list(explanation_factors),
        "recommendations": list(recommendations),
        "business_impact": {
            "Estimated Revenue Protection": (
                f"${float(business_impact['revenue_protection']):,.2f} / month"
            ),
            "Estimated Retention Improvement": (
                f"↑ {business_impact['retention_improvement']}"
            ),
        },
        "generated_at": generated_at or datetime.now(),
        "platform_name": PLATFORM_NAME,
    }


def build_recommendation_table(
    recommendations: List[Dict[str, str]],
) -> List[List[str]]:
    """
    Build tabular rows for retention recommendations.

    Args:
        recommendations: Generated recommendation dictionaries.

    Returns:
        Table rows including a header row.
    """
    rows: List[List[str]] = [
        ["Priority", "Recommendation", "Reason", "Expected Impact"],
    ]
    for recommendation in recommendations:
        rows.append(
            [
                recommendation.get("priority", ""),
                recommendation.get("title", ""),
                recommendation.get("reason", ""),
                recommendation.get("expected_impact", ""),
            ]
        )
    return rows


def _add_page_number(canvas: Any, doc: Any) -> None:
    """Draw page numbers in the report footer area."""
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.HexColor("#6b7280"))
    canvas.drawRightString(
        letter[0] - 0.75 * inch,
        0.45 * inch,
        f"Page {doc.page}",
    )
    canvas.restoreState()


def _build_section_table(rows: List[List[str]], col_widths: List[float]) -> Table:
    """Create a styled two-column key-value or multi-column table."""
    table = Table(rows, colWidths=col_widths, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563eb")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 1), (-1, -1), 9),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#111827")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def _build_key_value_table(data: Dict[str, str]) -> Table:
    """Create a two-column table from a label-value dictionary."""
    rows = [["Field", "Value"]] + [[key, value] for key, value in data.items()]
    return _build_section_table(rows, [2.2 * inch, 4.3 * inch])


def generate_prediction_pdf(report_data: Dict[str, Any]) -> bytes:
    """
    Generate a professional PDF report from structured prediction data.

    Args:
        report_data: Payload from ``build_prediction_summary``.

    Returns:
        PDF document as bytes suitable for download.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.85 * inch,
        title=REPORT_TITLE,
        author=PLATFORM_NAME,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        textColor=colors.HexColor("#111827"),
        spaceAfter=6,
        alignment=TA_CENTER,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        textColor=colors.HexColor("#4b5563"),
        spaceAfter=18,
        alignment=TA_CENTER,
    )
    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        textColor=colors.HexColor("#1f2937"),
        spaceBefore=14,
        spaceAfter=8,
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        textColor=colors.HexColor("#374151"),
        leading=14,
        spaceAfter=6,
    )
    bullet_style = ParagraphStyle(
        "BulletItem",
        parent=body_style,
        leftIndent=12,
        spaceAfter=4,
    )
    footer_style = ParagraphStyle(
        "FooterText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        textColor=colors.HexColor("#6b7280"),
        alignment=TA_CENTER,
        spaceBefore=18,
    )

    generated_at = report_data.get("generated_at", datetime.now())
    if isinstance(generated_at, datetime):
        generated_label = generated_at.strftime("%Y-%m-%d %H:%M:%S")
    else:
        generated_label = str(generated_at)

    story: List[Any] = [
        Paragraph(REPORT_TITLE, title_style),
        Paragraph(PLATFORM_NAME, subtitle_style),
        Spacer(1, 0.1 * inch),
    ]

    story.append(Paragraph("Customer Information", heading_style))
    story.append(_build_key_value_table(report_data["customer_info"]))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Prediction Summary", heading_style))
    story.append(_build_key_value_table(report_data["prediction_summary"]))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Prediction Explanation", heading_style))
    explanation_items = [
        ListItem(Paragraph(factor, bullet_style), leftIndent=12)
        for factor in report_data.get("explanation_factors", [])
    ]
    if explanation_items:
        story.append(ListFlowable(explanation_items, bulletType="bullet", start="•"))
    else:
        story.append(Paragraph("No explanation factors available.", body_style))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Retention Recommendations", heading_style))
    recommendation_rows = build_recommendation_table(report_data.get("recommendations", []))
    recommendation_table = _build_section_table(
        recommendation_rows,
        [0.9 * inch, 2.2 * inch, 1.8 * inch, 1.6 * inch],
    )
    story.append(recommendation_table)
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Business Impact", heading_style))
    story.append(_build_key_value_table(report_data["business_impact"]))
    story.append(Spacer(1, 0.25 * inch))

    story.append(Paragraph("—" * 72, body_style))
    story.append(
        Paragraph(
            f"<b>Generated by</b><br/>{report_data.get('platform_name', PLATFORM_NAME)}",
            footer_style,
        )
    )
    story.append(
        Paragraph(
            f"<b>Generation Date &amp; Time:</b> {generated_label}",
            footer_style,
        )
    )

    doc.build(story, onFirstPage=_add_page_number, onLaterPages=_add_page_number)
    buffer.seek(0)
    return buffer.getvalue()
