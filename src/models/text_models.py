# src/models/text_models.py
from sklearn.feature_extraction.text import TfidfVectorizer

class SimpleTextVectorizer:
    def __init__(self, text_col='text', max_features=5000):
        self.text_col = text_col
        self.vectorizer = TfidfVectorizer(max_features=max_features)

    def fit_transform(self, df):
        docs = df[self.text_col].fillna('').astype(str).values
        return self.vectorizer.fit_transform(docs)

    def transform(self, df):
        docs = df[self.text_col].fillna('').astype(str).values
        return self.vectorizer.transform(docs)
