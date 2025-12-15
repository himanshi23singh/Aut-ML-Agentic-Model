"""
Tool Selector Module
Dynamically selects appropriate NLP tools/models based on detected task.
"""
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime


class ToolSelector:
    """
    Intelligent tool selection based on task type and text characteristics.
    Maintains a registry of available tools with their capabilities.
    """
    
    # Tool registry with task mappings and priorities
    TOOL_REGISTRY = {
        "sentiment_transformer": {
            "name": "Sentiment Transformer",
            "type": "sentiment",
            "model": "distilbert-base-uncased-finetuned-sst-2-english",
            "tasks": ["sentiment_analysis"],
            "priority": 1,
            "description": "DistilBERT fine-tuned for sentiment analysis",
            "requirements": ["transformers", "torch"],
            "max_length": 512
        },
        "spacy_ner": {
            "name": "SpaCy NER",
            "type": "ner",
            "model": "en_core_web_sm",
            "tasks": ["named_entity_recognition"],
            "priority": 1,
            "description": "SpaCy's named entity recognition",
            "requirements": ["spacy"],
            "max_length": None
        },
        "zero_shot_classifier": {
            "name": "Zero-Shot Classifier",
            "type": "classification",
            "model": "facebook/bart-large-mnli",
            "tasks": ["text_classification"],
            "priority": 1,
            "description": "BART-based zero-shot classification",
            "requirements": ["transformers", "torch"],
            "max_length": 1024
        },
        "lda_topic_model": {
            "name": "LDA Topic Model",
            "type": "topic",
            "model": "sklearn-lda",
            "tasks": ["topic_modeling"],
            "priority": 1,
            "description": "Latent Dirichlet Allocation for topic modeling",
            "requirements": ["scikit-learn", "nltk"],
            "max_length": None
        },
        "general_analyzer": {
            "name": "General Analyzer",
            "type": "general",
            "model": "multi-tool",
            "tasks": ["general_analysis"],
            "priority": 1,
            "description": "Combined analysis using multiple tools",
            "requirements": [],
            "max_length": None
        }
    }
    
    def __init__(self):
        self.last_selection_result = None
        self._available_tools = None
    
    async def select_tool(
        self, 
        task: str,
        text_length: int,
        force_tool: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Select the best tool for the given task.
        
        Args:
            task: The detected/selected task
            text_length: Length of the text to process
            force_tool: If provided, use this tool (override)
        
        Returns:
            Tuple of (tool_id, selection_metadata)
        """
        metadata = {
            "started_at": datetime.utcnow().isoformat(),
            "task": task,
            "text_length": text_length,
            "selection_method": "automatic" if not force_tool else "forced",
            "candidates": [],
            "reasoning": [],
            "override_applied": force_tool is not None
        }
        
        if force_tool:
            if force_tool in self.TOOL_REGISTRY:
                tool_info = self.TOOL_REGISTRY[force_tool]
                metadata["selected_tool"] = force_tool
                metadata["tool_info"] = tool_info
                metadata["reasoning"].append(f"Tool forced to: {force_tool}")
                self.last_selection_result = metadata
                return force_tool, metadata
            else:
                raise ValueError(f"Unknown tool: {force_tool}")
        
        # Find candidate tools for the task
        candidates = []
        for tool_id, tool_info in self.TOOL_REGISTRY.items():
            if task in tool_info["tasks"]:
                candidates.append({
                    "id": tool_id,
                    "info": tool_info,
                    "score": self._calculate_tool_score(tool_info, text_length)
                })
        
        metadata["candidates"] = [
            {"id": c["id"], "name": c["info"]["name"], "score": c["score"]}
            for c in candidates
        ]
        
        if not candidates:
            # Fallback to general analyzer
            metadata["reasoning"].append(f"No specific tool for task '{task}'")
            metadata["reasoning"].append("Falling back to general analyzer")
            selected = "general_analyzer"
        else:
            # Sort by score and select best
            candidates.sort(key=lambda x: x["score"], reverse=True)
            selected = candidates[0]["id"]
            metadata["reasoning"].append(
                f"Selected '{selected}' with score {candidates[0]['score']:.2f}"
            )
        
        tool_info = self.TOOL_REGISTRY[selected]
        
        # Check for text length constraints
        if tool_info.get("max_length") and text_length > tool_info["max_length"]:
            metadata["reasoning"].append(
                f"Warning: Text length ({text_length}) exceeds tool limit ({tool_info['max_length']})"
            )
            metadata["text_will_be_truncated"] = True
        
        metadata["selected_tool"] = selected
        metadata["tool_info"] = tool_info
        metadata["completed_at"] = datetime.utcnow().isoformat()
        
        self.last_selection_result = metadata
        return selected, metadata
    
    def _calculate_tool_score(self, tool_info: Dict, text_length: int) -> float:
        """Calculate a suitability score for a tool."""
        score = 1.0
        
        # Priority factor
        score *= (1 / tool_info.get("priority", 1))
        
        # Length suitability
        max_len = tool_info.get("max_length")
        if max_len:
            if text_length <= max_len:
                score *= 1.0
            elif text_length <= max_len * 2:
                score *= 0.8  # Slight penalty for truncation
            else:
                score *= 0.5  # Significant penalty for heavy truncation
        
        return round(score, 3)
    
    def get_available_tools(self, task: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get list of available tools, optionally filtered by task."""
        tools = []
        for tool_id, tool_info in self.TOOL_REGISTRY.items():
            if task is None or task in tool_info["tasks"]:
                tools.append({
                    "id": tool_id,
                    "name": tool_info["name"],
                    "type": tool_info["type"],
                    "model": tool_info["model"],
                    "description": tool_info["description"],
                    "tasks": tool_info["tasks"]
                })
        return tools
    
    def get_tool_info(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed info about a specific tool."""
        if tool_id in self.TOOL_REGISTRY:
            return {
                "id": tool_id,
                **self.TOOL_REGISTRY[tool_id]
            }
        return None
    
    def get_selection_summary(self) -> Dict[str, Any]:
        """Get summary of last selection for dashboard."""
        if not self.last_selection_result:
            return {"status": "No tool selected yet"}
        
        result = self.last_selection_result
        return {
            "selected_tool": result.get("selected_tool"),
            "tool_name": result.get("tool_info", {}).get("name"),
            "model": result.get("tool_info", {}).get("model"),
            "reasoning": result.get("reasoning"),
            "alternatives": [
                c for c in result.get("candidates", [])
                if c["id"] != result.get("selected_tool")
            ][:2]
        }
