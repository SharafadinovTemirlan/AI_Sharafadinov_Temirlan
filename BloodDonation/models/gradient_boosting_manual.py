import numpy as np
import os
import pickle

def load_data(filepath):
    filepath = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'blood_data.txt')
    data = np.genfromtxt(filepath, delimiter=",", skip_header=1)
    X = data[:, :-1]
    y = data[:, -1]
    return X, y

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
        y_pred = np.zeros(X.shape[0])
        for stump in self.models:
            y_pred += self.learning_rate * stump.predict(X)
        return np.round(y_pred)


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

# === Main ===
X, y = load_data("../dataset/blood_data.txt")
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

gb = GradientBoosting()
gb.fit(X_train, y_train)
predictions = gb.predict(X_test)
accuracy = np.mean(predictions == y_test)
print("Gradient Boosting Accuracy:", accuracy)


with open('gradient_boosting_model.pkl', 'wb') as f:
    pickle.dump(gb, f)

def run():
    return "Градиентный бустинг выполнен успешно!"
