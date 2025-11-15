# src/preprocess/text.py
import re
from sklearn.feature_extraction.text import TfidfVectorizer

class TextVectorizer:
    def __init__(self, text_col='text', max_features=5000, stop_words='english'):
        self.text_col = text_col
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words=stop_words)

    def _clean(self, s: str) -> str:
        s = s or ''
        s = s.lower()
        s = re.sub(r"[^a-z0-9\s]", ' ', s)
        s = re.sub(r"\s+", ' ', s).strip()
        return s

    def fit_transform(self, X_df):
        docs = X_df[self.text_col].fillna('').astype(str).apply(self._clean).values
        return self.vectorizer.fit_transform(docs)

    def transform(self, X_df):
        docs = X_df[self.text_col].fillna('').astype(str).apply(self._clean).values
        return self.vectorizer.transform(docs)
