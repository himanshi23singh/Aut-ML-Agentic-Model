# src/report/pdf_report.py

from datetime import datetime
from typing import Dict

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas

from config import REPORTS_DIR


def generate_pdf_report(
    dataset_name: str,
    target_col: str,
    problem_type: str,
    leaderboard: Dict[str, dict],
    best_model_name: str,
    pdf_name: str | None = None,
) -> str:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if pdf_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_name = f"auto_ml_report_{timestamp}.pdf"

    pdf_path = REPORTS_DIR / pdf_name
    c = canvas.Canvas(str(pdf_path), pagesize=A4)

    width, height = A4
    x_margin, y_margin = 2 * cm, 2 * cm
    y = height - y_margin

    c.setFont("Helvetica-Bold", 16)
    c.drawString(x_margin, y, "Agentic Auto-ML Report")
    y -= 1.5 * cm

    c.setFont("Helvetica", 11)
    c.drawString(x_margin, y, f"Dataset: {dataset_name}")
    y -= 0.7 * cm
    c.drawString(x_margin, y, f"Target Column: {target_col}")
    y -= 0.7 * cm
    c.drawString(x_margin, y, f"Problem Type: {problem_type.title()}")
    y -= 0.7 * cm
    c.drawString(x_margin, y, f"Best Model: {best_model_name}")
    y -= 1.2 * cm

    c.setFont("Helvetica-Bold", 12)
    c.drawString(x_margin, y, "Model Leaderboard:")
    y -= 0.8 * cm
    c.setFont("Helvetica", 10)

    for name, info in leaderboard.items():
        line = f"{name} -> main_metric: {info['main_metric']} = {info['main_metric_value']:.4f}"
        c.drawString(x_margin, y, line)
        y -= 0.6 * cm
        if y < y_margin:
            c.showPage()
            y = height - y_margin
            c.setFont("Helvetica", 10)

    c.showPage()
    c.save()
    return str(pdf_path)
