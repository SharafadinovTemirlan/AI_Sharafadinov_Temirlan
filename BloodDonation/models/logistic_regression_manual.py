import numpy as np
import os
import pickle

def load_data():
    filepath = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'blood_data.txt')
    data = np.genfromtxt(filepath, delimiter=",", skip_header=1)
    X = data[:, :-1]
    y = data[:, -1]
    return X, y

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_proba(X, weights):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    return sigmoid(X_b @ weights)

def predict(X, weights):
    return predict_proba(X, weights) >= 0.5

def train_logistic_regression(X, y, lr=0.01, epochs=1000):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    weights = np.zeros(X_b.shape[1])

    for _ in range(epochs):
        predictions = sigmoid(X_b @ weights)
        gradient = X_b.T @ (predictions - y) / len(y)
        weights -= lr * gradient
    return weights

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# === Main ===
X, y = load_data()

# Разделим данные (80% train / 20% test)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

weights = train_logistic_regression(X_train, y_train)
y_pred = predict(X_test, weights)

print("Logistic Regression Accuracy:", accuracy(y_test, y_pred))


with open('logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(weights, f)


def run():
    return "Логистическая регрессия выполнена успешно!"
with open('logistic_model.pkl', 'wb') as f:
    pickle.dump(weights, f)