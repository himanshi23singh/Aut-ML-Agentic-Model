"""
Preprocessor Module
Text cleaning, normalization, tokenization, and language detection.
"""
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime


class Preprocessor:
    """
    Preprocessing pipeline for text data.
    Provides step-by-step transparency for glass-box observability.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.last_preprocessing_result = None
        self._nlp = None
        self._nltk_ready = False
    
    def _ensure_nltk(self):
        """Ensure NLTK data is downloaded."""
        if self._nltk_ready:
            return
        
        try:
            import nltk
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            import nltk
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt_tab', quiet=True)
        
        self._nltk_ready = True
    
    async def preprocess(
        self, 
        text: str,
        steps: Optional[List[str]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Preprocess text with configurable steps.
        
        Args:
            text: Raw input text
            steps: List of preprocessing steps to apply
                   Default: ['clean', 'normalize', 'detect_language']
        
        Returns:
            Tuple of (processed_text, metadata_dict)
        """
        if steps is None:
            steps = ['clean', 'normalize', 'detect_language']
        
        metadata = {
            "started_at": datetime.utcnow().isoformat(),
            "original_length": len(text),
            "steps_applied": [],
            "steps_skipped": [],
            "language_detected": None,
            "token_count": None,
            "decisions": []
        }
        
        processed_text = text
        
        for step in steps:
            step_start = datetime.utcnow()
            
            if step == 'clean':
                processed_text, step_meta = await self._clean_text(processed_text)
                metadata["steps_applied"].append({
                    "step": "clean",
                    "duration_ms": (datetime.utcnow() - step_start).total_seconds() * 1000,
                    "details": step_meta
                })
                
            elif step == 'normalize':
                processed_text, step_meta = await self._normalize_text(processed_text)
                metadata["steps_applied"].append({
                    "step": "normalize",
                    "duration_ms": (datetime.utcnow() - step_start).total_seconds() * 1000,
                    "details": step_meta
                })
                
            elif step == 'detect_language':
                lang, step_meta = await self._detect_language(processed_text)
                metadata["language_detected"] = lang
                metadata["steps_applied"].append({
                    "step": "detect_language",
                    "duration_ms": (datetime.utcnow() - step_start).total_seconds() * 1000,
                    "details": step_meta
                })
                metadata["decisions"].append(f"Detected language: {lang}")
                
            elif step == 'tokenize':
                tokens, step_meta = await self._tokenize_text(processed_text)
                metadata["token_count"] = len(tokens)
                metadata["tokens_preview"] = tokens[:20]  # First 20 tokens
                metadata["steps_applied"].append({
                    "step": "tokenize",
                    "duration_ms": (datetime.utcnow() - step_start).total_seconds() * 1000,
                    "details": step_meta
                })
            else:
                metadata["steps_skipped"].append(step)
        
        metadata["completed_at"] = datetime.utcnow().isoformat()
        metadata["final_length"] = len(processed_text)
        metadata["length_reduction_pct"] = round(
            (1 - len(processed_text) / max(len(text), 1)) * 100, 2
        )
        
        self.last_preprocessing_result = metadata
        return processed_text, metadata
    
    async def _clean_text(self, text: str) -> Tuple[str, Dict]:
        """Clean text by removing noise."""
        meta = {
            "operations": [],
            "chars_removed": 0
        }
        
        original_len = len(text)
        
        # Remove URLs
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        text = re.sub(url_pattern, ' ', text)
        meta["operations"].append("Removed URLs")
        
        # Remove email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        text = re.sub(email_pattern, ' ', text)
        meta["operations"].append("Removed emails")
        
        # Remove HTML tags if present
        text = re.sub(r'<[^>]+>', ' ', text)
        meta["operations"].append("Removed HTML tags")
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        meta["operations"].append("Normalized whitespace")
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        meta["chars_removed"] = original_len - len(text)
        
        return text, meta
    
    async def _normalize_text(self, text: str) -> Tuple[str, Dict]:
        """Normalize text for consistent processing."""
        meta = {
            "operations": [],
            "case_changed": False
        }
        
        # Convert to lowercase (optional based on config)
        if self.config.get("lowercase", False):
            text = text.lower()
            meta["case_changed"] = True
            meta["operations"].append("Converted to lowercase")
        
        # Normalize unicode characters
        import unicodedata
        text = unicodedata.normalize('NFKC', text)
        meta["operations"].append("Unicode normalization (NFKC)")
        
        # Replace common contractions
        contractions = {
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'s": " is",
            "'d": " would",
            "'ll": " will",
            "'ve": " have",
            "'m": " am"
        }
        for contraction, expansion in contractions.items():
            if contraction in text.lower():
                text = re.sub(re.escape(contraction), expansion, text, flags=re.IGNORECASE)
                meta["operations"].append(f"Expanded contraction: {contraction}")
        
        return text, meta
    
    async def _detect_language(self, text: str) -> Tuple[str, Dict]:
        """Detect the language of the text."""
        meta = {
            "method": None,
            "confidence": None,
            "alternatives": []
        }
        
        try:
            from langdetect import detect, detect_langs
            
            # Get primary language
            lang = detect(text)
            meta["method"] = "langdetect"
            
            # Get all detected languages with probabilities
            try:
                all_langs = detect_langs(text)
                meta["confidence"] = float(all_langs[0].prob) if all_langs else None
                meta["alternatives"] = [
                    {"lang": str(l.lang), "prob": float(l.prob)}
                    for l in all_langs[:3]
                ]
            except:
                meta["confidence"] = 0.9  # Default confidence
            
            return lang, meta
            
        except ImportError:
            meta["method"] = "fallback"
            meta["confidence"] = 0.5
            return "en", meta
        except Exception as e:
            meta["method"] = "fallback"
            meta["error"] = str(e)
            meta["confidence"] = 0.3
            return "en", meta
    
    async def _tokenize_text(self, text: str) -> Tuple[List[str], Dict]:
        """Tokenize text into words."""
        meta = {
            "method": None,
            "token_count": 0
        }
        
        self._ensure_nltk()
        
        try:
            from nltk.tokenize import word_tokenize
            tokens = word_tokenize(text)
            meta["method"] = "nltk_word_tokenize"
        except:
            # Fallback to simple split
            tokens = text.split()
            meta["method"] = "simple_split"
        
        meta["token_count"] = len(tokens)
        
        return tokens, meta
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Get summary of last preprocessing for dashboard."""
        if not self.last_preprocessing_result:
            return {"status": "No preprocessing done yet"}
        
        meta = self.last_preprocessing_result
        return {
            "original_length": meta.get("original_length"),
            "final_length": meta.get("final_length"),
            "reduction_pct": meta.get("length_reduction_pct"),
            "language": meta.get("language_detected"),
            "steps_count": len(meta.get("steps_applied", [])),
            "decisions": meta.get("decisions", [])
        }
