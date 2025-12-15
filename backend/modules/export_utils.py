"""
Export Utilities Module
Provides JSON and PDF export functionality for analysis results.
"""
from typing import Dict, Any, Optional
from datetime import datetime
import json
import io

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.platypus import PageBreak, HRFlowable


class ExportManager:
    """
    Manages export of analysis results to various formats.
    Supports JSON and PDF exports with full traceability.
    """
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom PDF styles."""
        self.styles.add(ParagraphStyle(
            name='Title_Custom',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#1a1a2e')
        ))
        
        self.styles.add(ParagraphStyle(
            name='Heading_Custom',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.HexColor('#16213e')
        ))
        
        self.styles.add(ParagraphStyle(
            name='SubHeading',
            parent=self.styles['Heading2'],
            fontSize=12,
            spaceAfter=8,
            textColor=colors.HexColor('#1f4068')
        ))
        
        self.styles.add(ParagraphStyle(
            name='Body_Custom',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            leading=14
        ))
        
        self.styles.add(ParagraphStyle(
            name='Highlight',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=colors.HexColor('#e94560'),
            spaceAfter=10
        ))
    
    def export_json(self, result: Dict[str, Any], include_raw: bool = True) -> Dict[str, Any]:
        """
        Export analysis result as structured JSON.
        
        Args:
            result: The analysis result (draft or final output)
            include_raw: Whether to include raw data
        
        Returns:
            JSON-serializable dictionary
        """
        export_data = {
            "export_info": {
                "format": "json",
                "version": "1.0",
                "exported_at": datetime.utcnow().isoformat(),
                "generator": "AutoBench Glass-Box AI"
            },
            "session_id": result.get("session_id"),
            "task": result.get("task"),
            "status": result.get("status"),
            "generated_at": result.get("generated_at"),
            
            # Main Results
            "summary": result.get("summary"),
            "confidence": result.get("confidence"),
            
            # Formatted Content
            "content": self._extract_display_content(result.get("content", {})),
            
            # Quality Metrics
            "quality": result.get("quality_indicators", {}),
            
            # Traceability
            "traceability": result.get("traceability", []),
            "sources": result.get("sources", [])
        }
        
        if include_raw:
            export_data["raw_data"] = result.get("content", {}).get("raw_data", {})
        
        return export_data
    
    def export_json_string(self, result: Dict[str, Any], pretty: bool = True) -> str:
        """Export as JSON string."""
        data = self.export_json(result)
        if pretty:
            return json.dumps(data, indent=2, default=str)
        return json.dumps(data, default=str)
    
    def export_pdf(self, result: Dict[str, Any], session_state: Optional[Dict] = None) -> bytes:
        """
        Export analysis result as formatted PDF report.
        
        Args:
            result: The analysis result
            session_state: Optional session state for additional context
        
        Returns:
            PDF file as bytes
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        story = []
        
        # Title
        story.append(Paragraph("AutoBench Analysis Report", self.styles['Title_Custom']))
        story.append(Spacer(1, 12))
        
        # Report Info
        info_data = [
            ["Session ID:", result.get("session_id", "N/A")[:36]],
            ["Generated:", result.get("generated_at", datetime.utcnow().isoformat())[:19]],
            ["Task:", result.get("task", "Unknown").replace("_", " ").title()],
            ["Status:", result.get("status", "unknown").title()]
        ]
        
        info_table = Table(info_data, colWidths=[1.5*inch, 4*inch])
        info_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#1f4068')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(info_table)
        story.append(Spacer(1, 20))
        
        # Horizontal Rule
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e94560')))
        story.append(Spacer(1, 20))
        
        # Summary Section
        story.append(Paragraph("Summary", self.styles['Heading_Custom']))
        summary = result.get("summary", "No summary available")
        story.append(Paragraph(summary, self.styles['Body_Custom']))
        
        confidence = result.get("confidence", 0)
        story.append(Paragraph(
            f"Confidence: {confidence:.0%}",
            self.styles['Highlight']
        ))
        
        # Main Findings
        story.append(Paragraph("Key Findings", self.styles['Heading_Custom']))
        
        content = result.get("content", {})
        display = content.get("display", {})
        
        main_finding = display.get("main_finding", "Analysis complete")
        story.append(Paragraph(f"• {main_finding}", self.styles['Body_Custom']))
        
        # Task-specific content
        task = result.get("task", "")
        
        if task == "named_entity_recognition":
            story.extend(self._format_ner_content(display))
        elif task == "sentiment_analysis":
            story.extend(self._format_sentiment_content(display))
        elif task == "topic_modeling":
            story.extend(self._format_topic_content(display))
        elif task == "text_classification":
            story.extend(self._format_classification_content(display))
        else:
            story.extend(self._format_general_content(display))
        
        # Quality Indicators
        story.append(Paragraph("Quality Assessment", self.styles['Heading_Custom']))
        quality = result.get("quality_indicators", {})
        
        quality_data = [
            ["Overall Score:", f"{quality.get('overall_score', 0):.0%}"],
            ["Quality Level:", quality.get('quality_level', 'N/A')]
        ]
        
        for factor in quality.get('factors', [])[:3]:
            quality_data.append([factor['name'], factor['impact']])
        
        quality_table = Table(quality_data, colWidths=[2.5*inch, 2*inch])
        quality_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f0f0f0')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(quality_table)
        
        # Traceability
        story.append(Spacer(1, 20))
        story.append(Paragraph("Execution Trace", self.styles['Heading_Custom']))
        
        traces = result.get("traceability", [])
        if traces:
            # Handle both list and dict types
            if isinstance(traces, list):
                trace_list = traces[:5]
            elif isinstance(traces, dict):
                trace_list = list(traces.values())[:5] if traces else []
            else:
                trace_list = []
            
            for trace in trace_list:
                if isinstance(trace, dict):
                    trace_text = f"Step {trace.get('step', '?')}: {trace.get('tool', 'Unknown')} - {trace.get('output_summary', '')}"
                else:
                    trace_text = str(trace)
                story.append(Paragraph(f"• {trace_text}", self.styles['Body_Custom']))
        else:
            story.append(Paragraph("No trace data available", self.styles['Body_Custom']))
        
        # Footer
        story.append(Spacer(1, 30))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
        story.append(Spacer(1, 10))
        story.append(Paragraph(
            "Generated by AutoBench - Glass-Box Agentic AI System",
            ParagraphStyle(name='Footer', fontSize=8, textColor=colors.grey, alignment=1)
        ))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer.read()
    
    def _extract_display_content(self, content: Dict) -> Dict:
        """Extract display-friendly content."""
        display = content.get("display", {})
        return {
            "title": display.get("title", "Analysis Result"),
            "main_finding": display.get("main_finding", ""),
            "confidence": display.get("confidence_display", ""),
            "details": display
        }
    
    def _format_ner_content(self, display: Dict) -> list:
        """Format NER results for PDF."""
        elements = []
        elements.append(Spacer(1, 10))
        elements.append(Paragraph("Entities Found", self.styles['SubHeading']))
        
        entities_by_type = display.get("entities_by_type", {})
        for entity_type, entities in entities_by_type.items():
            entity_list = ", ".join(entities[:5])
            if len(entities) > 5:
                entity_list += f" (+{len(entities) - 5} more)"
            elements.append(Paragraph(
                f"<b>{entity_type}:</b> {entity_list}",
                self.styles['Body_Custom']
            ))
        
        return elements
    
    def _format_sentiment_content(self, display: Dict) -> list:
        """Format sentiment results for PDF."""
        elements = []
        elements.append(Spacer(1, 10))
        
        breakdown = display.get("breakdown", {})
        if breakdown:
            data = [[k, f"{v:.1%}" if isinstance(v, float) else str(v)] 
                    for k, v in breakdown.items()]
            if data:
                table = Table(data, colWidths=[2*inch, 2*inch])
                table.setStyle(TableStyle([
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('PADDING', (0, 0), (-1, -1), 6),
                ]))
                elements.append(table)
        
        return elements
    
    def _format_topic_content(self, display: Dict) -> list:
        """Format topic modeling results for PDF."""
        elements = []
        elements.append(Spacer(1, 10))
        elements.append(Paragraph("Topics Identified", self.styles['SubHeading']))
        
        topics = display.get("topics_summary", [])
        for topic in topics[:5]:
            keywords = ", ".join(topic.get("keywords", [])[:5])
            elements.append(Paragraph(
                f"<b>Topic {topic.get('topic_id', 0) + 1}:</b> {keywords}",
                self.styles['Body_Custom']
            ))
        
        return elements
    
    def _format_classification_content(self, display: Dict) -> list:
        """Format classification results for PDF."""
        elements = []
        elements.append(Spacer(1, 10))
        
        categories = display.get("all_categories", [])
        if categories:
            data = [[cat['label'], f"{cat['score']:.1%}"] for cat in categories[:5]]
            table = Table(data, colWidths=[3*inch, 1.5*inch])
            table.setStyle(TableStyle([
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('PADDING', (0, 0), (-1, -1), 6),
                ('BACKGROUND', (0, 0), (0, 0), colors.HexColor('#e8f4e8')),
            ]))
            elements.append(table)
        
        return elements
    
    def _format_general_content(self, display: Dict) -> list:
        """Format general analysis results for PDF."""
        elements = []
        
        details = display.get("details", {})
        if isinstance(details, dict):
            for key, value in list(details.items())[:5]:
                if key not in ['title', 'main_finding']:
                    elements.append(Paragraph(
                        f"<b>{key.replace('_', ' ').title()}:</b> {str(value)[:100]}",
                        self.styles['Body_Custom']
                    ))
        
        return elements


# Global instance
export_manager = ExportManager()
