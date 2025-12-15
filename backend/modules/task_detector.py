"""
Task Detector Module
Automatically infers the required NLP task from text content.
"""
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from collections import Counter


class TaskDetector:
    """
    Intelligent task detection using pattern matching and heuristics.
    Provides confidence scores and reasoning for glass-box transparency.
    """
    
    # Task definitions with keywords and patterns
    TASK_DEFINITIONS = {
        "sentiment_analysis": {
            "keywords": [
                "good", "bad", "love", "hate", "amazing", "terrible", "great",
                "awful", "excellent", "poor", "happy", "sad", "angry", "positive",
                "negative", "like", "dislike", "best", "worst", "beautiful", "ugly"
            ],
            "patterns": [
                r'\b(feel|feeling|felt)\b',
                r'!{2,}',  # Multiple exclamation marks
                r'\b(love|hate|adore|despise)\b',
                r'(:\)|:\(|😊|😢|👍|👎)',  # Emoticons and emojis
            ],
            "description": "Analyze the sentiment/emotion expressed in text"
        },
        "named_entity_recognition": {
            "keywords": [],  # We'll detect by patterns
            "patterns": [
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b',  # Multi-word proper nouns
                r'\b(?:Mr\.|Mrs\.|Dr\.|Prof\.)\s+[A-Z][a-z]+\b',  # Titles
                r'\$\d+(?:,\d{3})*(?:\.\d{2})?',  # Currency
                r'\b\d{4}\b',  # Years
                r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b',
                r'\b[A-Z]{2,}\b',  # Acronyms
            ],
            "description": "Extract named entities (people, places, organizations)"
        },
        "text_classification": {
            "keywords": [
                "category", "type", "classify", "class", "label", "topic",
                "about", "regarding", "concerning", "subject"
            ],
            "patterns": [
                r'\b(news|article|report|blog|post)\b',
                r'\b(sports|politics|technology|science|health|business)\b',
            ],
            "description": "Classify text into predefined categories"
        },
        "topic_modeling": {
            "keywords": [
                "topics", "themes", "subjects", "ideas", "concepts",
                "discussed", "mentioned", "covered", "addressed"
            ],
            "patterns": [
                r'\b(various|multiple|several|different)\s+(topics?|themes?|subjects?)\b',
            ],
            "description": "Discover main topics/themes in text"
        },
        "general_analysis": {
            "keywords": ["analyze", "summary", "overview", "examine"],
            "patterns": [],
            "description": "General text analysis and summarization"
        }
    }
    
    def __init__(self):
        self.last_detection_result = None
    
    async def detect_task(
        self, 
        text: str,
        force_task: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Detect the most appropriate NLP task for the given text.
        
        Args:
            text: Preprocessed text
            force_task: If provided, skip detection and use this task
        
        Returns:
            Tuple of (selected_task, detection_metadata)
        """
        metadata = {
            "started_at": datetime.utcnow().isoformat(),
            "text_preview": text[:200] + "..." if len(text) > 200 else text,
            "text_length": len(text),
            "detection_method": "hybrid" if not force_task else "forced",
            "task_scores": {},
            "signals": {},
            "reasoning": [],
            "override_applied": force_task is not None
        }
        
        if force_task:
            metadata["reasoning"].append(f"Task forced to: {force_task}")
            metadata["selected_task"] = force_task
            metadata["confidence"] = 1.0
            self.last_detection_result = metadata
            return force_task, metadata
        
        # Calculate scores for each task
        task_scores = {}
        task_signals = {}
        
        for task_name, task_def in self.TASK_DEFINITIONS.items():
            score, signals = await self._calculate_task_score(
                text, task_name, task_def
            )
            task_scores[task_name] = score
            task_signals[task_name] = signals
        
        metadata["task_scores"] = task_scores
        metadata["signals"] = task_signals
        
        # Select the best task
        selected_task = max(task_scores, key=task_scores.get)
        confidence = task_scores[selected_task]
        
        # Apply decision rules
        reasoning = []
        
        # Rule 1: If score is too low, default to general analysis
        if confidence < 0.2:
            reasoning.append(f"Low confidence ({confidence:.2f}) for {selected_task}")
            reasoning.append("Defaulting to general_analysis")
            selected_task = "general_analysis"
            confidence = 0.5
        
        # Rule 2: If sentiment and NER are close, prefer sentiment for short text
        if (abs(task_scores.get("sentiment_analysis", 0) - 
                task_scores.get("named_entity_recognition", 0)) < 0.1
            and len(text) < 500):
            reasoning.append("Short text with mixed signals - preferring sentiment")
            selected_task = "sentiment_analysis"
            confidence = task_scores["sentiment_analysis"]
        
        # Rule 3: Long documents likely need topic modeling
        if len(text) > 2000 and task_scores.get("topic_modeling", 0) > 0.3:
            reasoning.append("Long document detected - boosting topic modeling")
            task_scores["topic_modeling"] *= 1.3
            if task_scores["topic_modeling"] > confidence:
                selected_task = "topic_modeling"
                confidence = min(task_scores["topic_modeling"], 1.0)
        
        metadata["reasoning"] = reasoning
        metadata["selected_task"] = selected_task
        metadata["confidence"] = round(confidence, 3)
        metadata["completed_at"] = datetime.utcnow().isoformat()
        
        # Generate human-readable rationale
        metadata["rationale"] = self._generate_rationale(
            selected_task, confidence, task_signals.get(selected_task, [])
        )
        
        self.last_detection_result = metadata
        return selected_task, metadata
    
    async def _calculate_task_score(
        self, 
        text: str, 
        task_name: str, 
        task_def: Dict
    ) -> Tuple[float, List[str]]:
        """Calculate a confidence score for a specific task."""
        signals = []
        score = 0.0
        
        text_lower = text.lower()
        
        # Keyword matching
        keyword_matches = 0
        for keyword in task_def.get("keywords", []):
            count = text_lower.count(keyword.lower())
            if count > 0:
                keyword_matches += count
                signals.append(f"Keyword '{keyword}' found {count}x")
        
        # Normalize keyword score (max 0.4)
        keyword_score = min(keyword_matches / 10, 0.4)
        score += keyword_score
        
        # Pattern matching
        pattern_matches = 0
        for pattern in task_def.get("patterns", []):
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                pattern_matches += len(matches)
                signals.append(f"Pattern match: {pattern[:30]}... ({len(matches)}x)")
        
        # Normalize pattern score (max 0.4)
        pattern_score = min(pattern_matches / 5, 0.4)
        score += pattern_score
        
        # Task-specific heuristics
        if task_name == "named_entity_recognition":
            # Check for capitalized words
            caps = len(re.findall(r'\b[A-Z][a-z]{2,}\b', text))
            if caps > 5:
                score += 0.2
                signals.append(f"Found {caps} capitalized words")
        
        elif task_name == "sentiment_analysis":
            # Check for opinion indicators
            opinion_words = len(re.findall(
                r'\b(think|believe|feel|opinion|seems|appears)\b', 
                text_lower
            ))
            if opinion_words > 0:
                score += 0.1
                signals.append(f"Found {opinion_words} opinion indicators")
        
        elif task_name == "topic_modeling":
            # Longer texts are better for topic modeling
            if len(text) > 1000:
                score += 0.15
                signals.append("Long text suitable for topic modeling")
        
        return min(score, 1.0), signals
    
    def _generate_rationale(
        self, 
        task: str, 
        confidence: float, 
        signals: List[str]
    ) -> str:
        """Generate a human-readable rationale for the task selection."""
        task_descriptions = {
            "sentiment_analysis": "analyze sentiment and emotional tone",
            "named_entity_recognition": "extract named entities",
            "text_classification": "classify into categories",
            "topic_modeling": "discover main topics",
            "general_analysis": "perform general text analysis"
        }
        
        desc = task_descriptions.get(task, task)
        signal_summary = ", ".join(signals[:3]) if signals else "general text patterns"
        
        return (
            f"Selected '{task}' with {confidence:.0%} confidence. "
            f"This text appears to require {desc} based on: {signal_summary}."
        )
    
    def get_available_tasks(self) -> List[Dict[str, str]]:
        """Get list of available tasks for dashboard override menu."""
        return [
            {"id": task_name, "description": task_def["description"]}
            for task_name, task_def in self.TASK_DEFINITIONS.items()
        ]
    
    def get_detection_summary(self) -> Dict[str, Any]:
        """Get summary of last detection for dashboard."""
        if not self.last_detection_result:
            return {"status": "No detection performed yet"}
        
        result = self.last_detection_result
        return {
            "selected_task": result.get("selected_task"),
            "confidence": result.get("confidence"),
            "rationale": result.get("rationale"),
            "alternatives": [
                {"task": t, "score": s}
                for t, s in sorted(
                    result.get("task_scores", {}).items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
            ]
        }
