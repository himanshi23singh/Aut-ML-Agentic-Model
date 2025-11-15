# src/agents/transparent_model_agent.py
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

class TransparentModelAgent:
    def __init__(self):
        pass

    def train(self, X_train, y_train, task='classification'):
        if task in ('classification','text_classification'):
            model = DecisionTreeClassifier(max_depth=4)
            model.fit(X_train, y_train)
            return model
        else:
            model = DecisionTreeRegressor(max_depth=4)
            model.fit(X_train, y_train)
            return model
