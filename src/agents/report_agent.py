# src/agents/report_agent.py

from dataclasses import dataclass
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

from config import REPORTS_DIR


@dataclass
class ReportOutput:
    report_path: str


class ReportAgent:
    def run(self, model_name: str, metrics_dict: dict):

        path = REPORTS_DIR / "final_report.pdf"
        styles = getSampleStyleSheet()
        doc = SimpleDocTemplate(str(path), pagesize=A4)

        story = []
        story.append(Paragraph("<b>AutoML Report</b>", styles["Heading1"]))
        story.append(Spacer(1, 20))

        story.append(Paragraph(f"<b>Best Model:</b> {model_name}", styles["BodyText"]))
        story.append(Spacer(1, 10))

        story.append(Paragraph("<b>Metrics:</b>", styles["Heading3"]))

        for m, v in metrics_dict.items():
            story.append(Paragraph(f"{m}: {v}", styles["BodyText"]))

        doc.build(story)

        return ReportOutput(report_path=str(path))
