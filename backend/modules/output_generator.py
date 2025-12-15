"""
Output Generator Module
Generates draft and final outputs with approval workflow.
"""
from typing import Dict, Any, Optional
from datetime import datetime
import json


class OutputGenerator:
    """
    Generates reviewable draft outputs and final approved outputs.
    Supports editing and regeneration requests.
    """
    
    def __init__(self):
        self.drafts = {}  # session_id -> draft
        self.finals = {}  # session_id -> final output
    
    async def generate_draft(
        self,
        session_id: str,
        aggregated_result: Dict[str, Any],
        task: str,
        request_metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Generate a draft output for review.
        
        Args:
            session_id: Current session ID
            aggregated_result: Result from aggregator
            task: The task type
            request_metadata: Original request info
        
        Returns:
            Draft output structure
        """
        draft = {
            "session_id": session_id,
            "draft_id": f"draft_{session_id}_{datetime.utcnow().strftime('%H%M%S')}",
            "generated_at": datetime.utcnow().isoformat(),
            "status": "pending_review",
            "task": task,
            "is_approved": False,
            
            # The actual output content
            "content": self._format_output(aggregated_result, task),
            
            # Summary for quick review
            "summary": aggregated_result.get("summary", ""),
            
            # Confidence and quality indicators
            "confidence": aggregated_result.get("confidence", 0.0),
            "quality_indicators": self._assess_quality(aggregated_result),
            
            # Traceability
            "traceability": aggregated_result.get("traceability", []),
            "sources": aggregated_result.get("sources", []),
            
            # Original request context
            "request_context": request_metadata or {},
            
            # Review metadata
            "review_notes": [],
            "edit_history": []
        }
        
        self.drafts[session_id] = draft
        return draft
    
    def _format_output(self, aggregated: Dict, task: str) -> Dict[str, Any]:
        """Format the aggregated result into a presentable output."""
        normalized = aggregated.get("normalized_output", {})
        primary = normalized.get("primary_result", {})
        
        formatted = {
            "type": task,
            "result": primary,
            "raw_data": aggregated.get("combined_output", {})
        }
        
        # Add task-specific formatting
        if task == "sentiment_analysis":
            formatted["display"] = {
                "title": "Sentiment Analysis Result",
                "main_finding": f"The text expresses a {primary.get('sentiment', 'neutral')} sentiment",
                "confidence_display": f"{primary.get('confidence', 0):.0%}",
                "breakdown": primary.get("scores", {})
            }
            
        elif task == "named_entity_recognition":
            entities = primary.get("entities", [])
            formatted["display"] = {
                "title": "Named Entity Recognition Result",
                "main_finding": f"Found {len(entities)} named entities",
                "entities_by_type": self._group_entities_by_type(entities),
                "entity_list": entities[:20]  # Limit for display
            }
            
        elif task == "topic_modeling":
            topics = primary.get("topics", [])
            formatted["display"] = {
                "title": "Topic Modeling Result",
                "main_finding": f"Identified {len(topics)} main topics",
                "topics_summary": [
                    {"topic_id": i, "keywords": t.get("keywords", [])[:5]}
                    for i, t in enumerate(topics)
                ]
            }
            
        elif task == "text_classification":
            top_cat = primary.get("top_category", {})
            formatted["display"] = {
                "title": "Text Classification Result",
                "main_finding": f"Classified as: {top_cat.get('label', 'Unknown')}",
                "confidence_display": f"{top_cat.get('score', 0):.0%}",
                "all_categories": primary.get("all_categories", [])
            }
            
        else:
            formatted["display"] = {
                "title": "Analysis Result",
                "main_finding": aggregated.get("summary", "Analysis complete"),
                "details": primary
            }
        
        return formatted
    
    def _group_entities_by_type(self, entities: list) -> Dict[str, list]:
        """Group entities by their type for display."""
        grouped = {}
        for entity in entities:
            etype = entity.get("type", "OTHER")
            if etype not in grouped:
                grouped[etype] = []
            grouped[etype].append(entity.get("text", entity.get("value", "")))
        return grouped
    
    def _assess_quality(self, aggregated: Dict) -> Dict[str, Any]:
        """Assess the quality of the output."""
        indicators = {
            "overall_score": 0.0,
            "factors": []
        }
        
        score = 0.0
        
        # Factor 1: Confidence
        confidence = aggregated.get("confidence", 0)
        if confidence >= 0.8:
            score += 0.3
            indicators["factors"].append({"name": "High confidence", "impact": "+30%"})
        elif confidence >= 0.5:
            score += 0.2
            indicators["factors"].append({"name": "Moderate confidence", "impact": "+20%"})
        else:
            score += 0.1
            indicators["factors"].append({"name": "Low confidence", "impact": "+10%"})
        
        # Factor 2: Successful tool executions
        sources = aggregated.get("sources", [])
        successful = len([s for s in sources if s.get("success")])
        if successful == len(sources) and sources:
            score += 0.3
            indicators["factors"].append({"name": "All tools succeeded", "impact": "+30%"})
        elif successful > 0:
            score += 0.15
            indicators["factors"].append({"name": "Partial success", "impact": "+15%"})
        
        # Factor 3: Traceability
        trace = aggregated.get("traceability", [])
        if trace:
            score += 0.2
            indicators["factors"].append({"name": "Full traceability", "impact": "+20%"})
        
        # Factor 4: Has intermediate outputs
        for source in sources:
            exec_record = source.get("execution_record", {})
            if exec_record.get("intermediate_outputs"):
                score += 0.2
                indicators["factors"].append({"name": "Explainable steps", "impact": "+20%"})
                break
        
        indicators["overall_score"] = min(score, 1.0)
        indicators["quality_level"] = (
            "High" if score >= 0.8 else
            "Medium" if score >= 0.5 else
            "Low"
        )
        
        return indicators
    
    async def approve_draft(
        self,
        session_id: str,
        reviewer_notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """Approve a draft and generate final output."""
        if session_id not in self.drafts:
            raise ValueError(f"No draft found for session: {session_id}")
        
        draft = self.drafts[session_id]
        
        # Update draft status
        draft["is_approved"] = True
        draft["status"] = "approved"
        draft["approved_at"] = datetime.utcnow().isoformat()
        if reviewer_notes:
            draft["review_notes"].append({
                "timestamp": datetime.utcnow().isoformat(),
                "note": reviewer_notes
            })
        
        # Generate final output
        final = {
            "session_id": session_id,
            "generated_at": datetime.utcnow().isoformat(),
            "status": "final",
            "task": draft["task"],
            
            # The approved content
            "content": draft["content"],
            "summary": draft["summary"],
            
            # Quality and confidence
            "confidence": draft["confidence"],
            "quality_score": draft["quality_indicators"]["overall_score"],
            
            # Full traceability chain
            "traceability": {
                "draft_id": draft["draft_id"],
                "generated_at": draft["generated_at"],
                "approved_at": draft["approved_at"],
                "sources": draft["sources"],
                "execution_trace": draft["traceability"]
            },
            
            # Export formats
            "exports": {
                "json": self._export_json(draft["content"]),
                "text": self._export_text(draft["content"], draft["task"])
            }
        }
        
        self.finals[session_id] = final
        return final
    
    async def edit_draft(
        self,
        session_id: str,
        edits: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply edits to a draft."""
        if session_id not in self.drafts:
            raise ValueError(f"No draft found for session: {session_id}")
        
        draft = self.drafts[session_id]
        
        # Record edit history
        draft["edit_history"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "edits": edits,
            "previous_content": draft["content"].copy()
        })
        
        # Apply edits
        if "summary" in edits:
            draft["summary"] = edits["summary"]
        if "content" in edits:
            draft["content"].update(edits["content"])
        
        draft["status"] = "edited"
        
        return draft
    
    def _export_json(self, content: Dict) -> str:
        """Export content as JSON string."""
        return json.dumps(content, indent=2, default=str)
    
    def _export_text(self, content: Dict, task: str) -> str:
        """Export content as human-readable text."""
        display = content.get("display", {})
        
        lines = [
            "=" * 50,
            display.get("title", "Analysis Result"),
            "=" * 50,
            "",
            display.get("main_finding", ""),
            ""
        ]
        
        if "confidence_display" in display:
            lines.append(f"Confidence: {display['confidence_display']}")
            lines.append("")
        
        if task == "named_entity_recognition":
            entities_by_type = display.get("entities_by_type", {})
            for etype, entities in entities_by_type.items():
                lines.append(f"\n{etype}:")
                for e in entities[:10]:
                    lines.append(f"  - {e}")
        
        elif task == "topic_modeling":
            topics = display.get("topics_summary", [])
            for topic in topics:
                keywords = ", ".join(topic.get("keywords", []))
                lines.append(f"\nTopic {topic['topic_id'] + 1}: {keywords}")
        
        lines.extend(["", "=" * 50])
        
        return "\n".join(lines)
    
    def get_draft(self, session_id: str) -> Optional[Dict]:
        """Get the draft for a session."""
        return self.drafts.get(session_id)
    
    def get_final(self, session_id: str) -> Optional[Dict]:
        """Get the final output for a session."""
        return self.finals.get(session_id)
