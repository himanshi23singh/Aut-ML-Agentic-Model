"""
Model Configuration Service
Manages custom configuration for NLP models and tools.
"""
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json


@dataclass
class ModelConfig:
    """Configuration for a specific NLP model/tool."""
    tool_id: str
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UserSettings:
    """User-specific settings and preferences."""
    
    # Sentiment Analysis
    sentiment_threshold: float = 0.5
    sentiment_include_neutral: bool = True
    
    # Named Entity Recognition
    ner_entity_types: List[str] = field(default_factory=lambda: [
        "PERSON", "ORG", "GPE", "LOC", "DATE", "MONEY", "PRODUCT"
    ])
    ner_min_confidence: float = 0.0
    
    # Topic Modeling
    topic_count: int = 5
    topic_min_words: int = 10
    
    # Text Classification
    classification_threshold: float = 0.3
    classification_categories: List[str] = field(default_factory=lambda: [
        "business", "technology", "politics", "sports", "entertainment"
    ])
    
    # General Settings
    max_text_length: int = 10000
    auto_detect_task: bool = True
    preferred_task: Optional[str] = None
    
    # Export Settings
    export_include_raw: bool = False
    export_format: str = "json"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserSettings':
        """Create UserSettings from dictionary."""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


class ModelConfigManager:
    """
    Manages model configurations and user settings.
    Provides session-based configuration storage.
    """
    
    def __init__(self):
        # Default settings
        self.default_settings = UserSettings()
        
        # Session-specific settings
        self.session_settings: Dict[str, UserSettings] = {}
        
        # Model-specific configurations
        self.model_configs: Dict[str, ModelConfig] = {
            "sentiment_transformer": ModelConfig(
                tool_id="sentiment_transformer",
                parameters={
                    "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
                    "max_length": 512
                }
            ),
            "spacy_ner": ModelConfig(
                tool_id="spacy_ner",
                parameters={
                    "model_name": "en_core_web_sm",
                    "disable_components": []
                }
            ),
            "zero_shot_classifier": ModelConfig(
                tool_id="zero_shot_classifier",
                parameters={
                    "model_name": "facebook/bart-large-mnli",
                    "multi_label": True
                }
            ),
            "lda_topic_model": ModelConfig(
                tool_id="lda_topic_model",
                parameters={
                    "n_topics": 5,
                    "max_features": 1000,
                    "max_iter": 10
                }
            ),
            "general_analyzer": ModelConfig(
                tool_id="general_analyzer",
                parameters={
                    "include_readability": True,
                    "include_statistics": True
                }
            )
        }
    
    def get_settings(self, session_id: Optional[str] = None) -> UserSettings:
        """Get settings for a session or default settings."""
        if session_id and session_id in self.session_settings:
            return self.session_settings[session_id]
        return self.default_settings
    
    def update_settings(
        self,
        session_id: str,
        updates: Dict[str, Any]
    ) -> UserSettings:
        """Update settings for a session."""
        current = self.get_settings(session_id)
        
        # Create new settings from current + updates
        current_dict = current.to_dict()
        current_dict.update(updates)
        
        new_settings = UserSettings.from_dict(current_dict)
        self.session_settings[session_id] = new_settings
        
        return new_settings
    
    def reset_settings(self, session_id: str) -> UserSettings:
        """Reset session settings to defaults."""
        if session_id in self.session_settings:
            del self.session_settings[session_id]
        return self.default_settings
    
    def get_model_config(self, tool_id: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model."""
        return self.model_configs.get(tool_id)
    
    def update_model_config(
        self,
        tool_id: str,
        updates: Dict[str, Any]
    ) -> Optional[ModelConfig]:
        """Update model configuration."""
        if tool_id not in self.model_configs:
            return None
        
        config = self.model_configs[tool_id]
        
        if "enabled" in updates:
            config.enabled = updates["enabled"]
        if "parameters" in updates:
            config.parameters.update(updates["parameters"])
        
        return config
    
    def get_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get all model configurations."""
        return {
            tool_id: config.to_dict()
            for tool_id, config in self.model_configs.items()
        }
    
    def get_tool_parameters(
        self,
        tool_id: str,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get merged parameters for a tool (model config + user settings)."""
        params = {}
        
        # Get model-level config
        model_config = self.get_model_config(tool_id)
        if model_config:
            params.update(model_config.parameters)
        
        # Apply user settings
        settings = self.get_settings(session_id)
        
        if tool_id == "sentiment_transformer":
            params["threshold"] = settings.sentiment_threshold
            params["include_neutral"] = settings.sentiment_include_neutral
            
        elif tool_id == "spacy_ner":
            params["entity_types"] = settings.ner_entity_types
            params["min_confidence"] = settings.ner_min_confidence
            
        elif tool_id == "lda_topic_model":
            params["n_topics"] = settings.topic_count
            params["min_words"] = settings.topic_min_words
            
        elif tool_id == "zero_shot_classifier":
            params["threshold"] = settings.classification_threshold
            params["candidate_labels"] = settings.classification_categories
        
        return params
    
    def export_settings(self, session_id: Optional[str] = None) -> str:
        """Export settings as JSON string."""
        settings = self.get_settings(session_id)
        return json.dumps(settings.to_dict(), indent=2)
    
    def import_settings(self, session_id: str, json_string: str) -> UserSettings:
        """Import settings from JSON string."""
        data = json.loads(json_string)
        new_settings = UserSettings.from_dict(data)
        self.session_settings[session_id] = new_settings
        return new_settings


# Global instance
model_config_manager = ModelConfigManager()
