"""
Aggregator Module
Combines outputs from multiple tools into a unified schema.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime


class Aggregator:
    """
    Aggregates and normalizes results from multiple NLP tools.
    Maintains traceability between tools and their outputs.
    """
    
    def __init__(self):
        self.last_aggregation = None
    
    async def aggregate(
        self,
        results: List[Dict[str, Any]],
        task: str,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Aggregate results from one or more tool executions.
        
        Args:
            results: List of execution results
            task: The original task type
            session_id: Current session ID
        
        Returns:
            Aggregated result with full traceability
        """
        aggregation = {
            "session_id": session_id,
            "task": task,
            "aggregated_at": datetime.utcnow().isoformat(),
            "sources": [],
            "combined_output": {},
            "summary": "",
            "confidence": 0.0,
            "traceability": []
        }
        
        for i, result in enumerate(results):
            source_info = {
                "index": i,
                "tool_id": result.get("execution_record", {}).get("tool_id", "unknown"),
                "execution_id": result.get("execution_id"),
                "success": result.get("success", False),
                "execution_time_ms": self._calculate_execution_time(result)
            }
            aggregation["sources"].append(source_info)
            
            if result.get("success"):
                # Add to combined output
                tool_result = result.get("result", {})
                tool_id = source_info["tool_id"]
                aggregation["combined_output"][tool_id] = tool_result
                
                # Track confidence
                if "confidence" in tool_result:
                    aggregation["confidence"] = max(
                        aggregation["confidence"],
                        tool_result["confidence"]
                    )
                
                # Add traceability entry
                aggregation["traceability"].append({
                    "step": i + 1,
                    "tool": tool_id,
                    "input_summary": f"Text length: {result.get('execution_record', {}).get('parameters', {}).get('text_length', 'N/A')}",
                    "output_summary": self._summarize_output(tool_result),
                    "intermediate_steps": len(
                        result.get("execution_record", {}).get("intermediate_outputs", [])
                    )
                })
        
        # Generate overall summary
        aggregation["summary"] = self._generate_summary(aggregation, task)
        
        # Normalize the combined output
        aggregation["normalized_output"] = self._normalize_output(
            aggregation["combined_output"],
            task
        )
        
        self.last_aggregation = aggregation
        return aggregation
    
    def _calculate_execution_time(self, result: Dict) -> Optional[float]:
        """Calculate execution time from timestamps."""
        record = result.get("execution_record", {})
        started = record.get("started_at")
        completed = record.get("completed_at")
        
        if started and completed:
            try:
                start_dt = datetime.fromisoformat(started)
                end_dt = datetime.fromisoformat(completed)
                return (end_dt - start_dt).total_seconds() * 1000
            except:
                pass
        return None
    
    def _summarize_output(self, output: Dict) -> str:
        """Generate a brief summary of tool output."""
        if "label" in output:
            return f"Label: {output['label']} (conf: {output.get('confidence', 'N/A')})"
        elif "entities" in output:
            count = len(output["entities"])
            return f"Found {count} entities"
        elif "topics" in output:
            count = len(output["topics"])
            return f"Identified {count} topics"
        elif "categories" in output:
            return f"Top category: {output['categories'][0]['label'] if output['categories'] else 'None'}"
        else:
            return f"Output keys: {list(output.keys())}"
    
    def _generate_summary(self, aggregation: Dict, task: str) -> str:
        """Generate a human-readable summary of the aggregated results."""
        source_count = len([s for s in aggregation["sources"] if s["success"]])
        total_count = len(aggregation["sources"])
        
        combined = aggregation["combined_output"]
        
        if task == "sentiment_analysis" and combined:
            tool_output = list(combined.values())[0] if combined else {}
            label = tool_output.get("label", "Unknown")
            conf = tool_output.get("confidence", 0)
            return f"Sentiment: {label} ({conf:.0%} confidence)"
        
        elif task == "named_entity_recognition" and combined:
            tool_output = list(combined.values())[0] if combined else {}
            entities = tool_output.get("entities", [])
            entity_types = set(e.get("type") for e in entities)
            return f"Found {len(entities)} entities of types: {', '.join(entity_types)}"
        
        elif task == "topic_modeling" and combined:
            tool_output = list(combined.values())[0] if combined else {}
            topics = tool_output.get("topics", [])
            return f"Identified {len(topics)} main topics"
        
        elif task == "text_classification" and combined:
            tool_output = list(combined.values())[0] if combined else {}
            categories = tool_output.get("categories", [])
            if categories:
                top = categories[0]
                return f"Classification: {top['label']} ({top['score']:.0%})"
        
        return f"Completed {source_count}/{total_count} tool executions"
    
    def _normalize_output(self, combined: Dict, task: str) -> Dict[str, Any]:
        """Normalize output into a consistent schema."""
        normalized = {
            "task": task,
            "primary_result": None,
            "details": {},
            "metadata": {}
        }
        
        if not combined:
            return normalized
        
        # Get the primary tool's output
        primary_output = list(combined.values())[0] if combined else {}
        
        if task == "sentiment_analysis":
            normalized["primary_result"] = {
                "sentiment": primary_output.get("label"),
                "confidence": primary_output.get("confidence"),
                "scores": primary_output.get("scores", {})
            }
            
        elif task == "named_entity_recognition":
            normalized["primary_result"] = {
                "entities": primary_output.get("entities", []),
                "entity_count": len(primary_output.get("entities", []))
            }
            
        elif task == "topic_modeling":
            normalized["primary_result"] = {
                "topics": primary_output.get("topics", []),
                "topic_count": len(primary_output.get("topics", []))
            }
            
        elif task == "text_classification":
            categories = primary_output.get("categories", [])
            normalized["primary_result"] = {
                "top_category": categories[0] if categories else None,
                "all_categories": categories
            }
            
        else:
            normalized["primary_result"] = primary_output
        
        normalized["details"] = combined
        
        return normalized
    
    def get_aggregation_summary(self) -> Dict[str, Any]:
        """Get summary of last aggregation for dashboard."""
        if not self.last_aggregation:
            return {"status": "No aggregation performed yet"}
        
        agg = self.last_aggregation
        return {
            "task": agg.get("task"),
            "summary": agg.get("summary"),
            "confidence": agg.get("confidence"),
            "source_count": len(agg.get("sources", [])),
            "successful_sources": len([s for s in agg.get("sources", []) if s["success"]])
        }
