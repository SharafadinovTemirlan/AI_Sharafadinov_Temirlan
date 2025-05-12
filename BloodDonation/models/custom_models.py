# models/custom_models.py

import numpy as np

class DecisionStump:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.left_value = None
        self.right_value = None

    def fit(self, X, y):
        m, n = X.shape
        min_error = float('inf')
        for feature_index in range(n):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_mask = X[:, feature_index] < threshold
                right_mask = ~left_mask
                left_value = np.mean(y[left_mask]) if np.any(left_mask) else 0
                right_value = np.mean(y[right_mask]) if np.any(right_mask) else 0
                y_pred = np.where(left_mask, left_value, right_value)
                error = np.mean((y - y_pred) ** 2)
                if error < min_error:
                    min_error = error
                    self.feature_index = feature_index
                    self.threshold = threshold
                    self.left_value = left_value
                    self.right_value = right_value

    def predict(self, X):
        left_mask = X[:, self.feature_index] < self.threshold
        return np.where(left_mask, self.left_value, self.right_value)

class GradientBoosting:
    def __init__(self, n_estimators=10, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []

    def fit(self, X, y):
        y_pred = np.zeros(len(y))
        for _ in range(self.n_estimators):
            residual = y - y_pred
            stump = DecisionStump()
            stump.fit(X, residual)
            update = stump.predict(X)
            y_pred += self.learning_rate * update
            self.models.append(stump)

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for stump in self.models:
            y_pred += self.learning_rate * stump.predict(X)
        return np.round(y_pred)
