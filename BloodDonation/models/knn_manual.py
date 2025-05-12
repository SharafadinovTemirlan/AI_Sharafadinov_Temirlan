import numpy as np
import os
import pickle

def load_data(filepath):
    filepath = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'blood_data.txt')
    data = np.genfromtxt(filepath, delimiter=",", skip_header=1)
    X = data[:, :-1]
    y = data[:, -1].astype(int)
    return X, y

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def predict(X_train, y_train, row, k=3):
    distances = [euclidean_distance(row, x) for x in X_train]
    k_indices = np.argsort(distances)[:k]
    k_labels = [y_train[i] for i in k_indices]
    return max(set(k_labels), key=k_labels.count)

def evaluate(X_train, y_train, X_test, y_test, k=3):
    predictions = [predict(X_train, y_train, row, k) for row in X_test]
    return np.mean(predictions == y_test)

# === Main ===
X, y = load_data("../dataset/blood_data.txt")
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

acc = evaluate(X_train, y_train, X_test, y_test, k=3)
print("KNN Accuracy (k=3):", acc)

with open('knn_model.pkl', 'wb') as f:
    pickle.dump({'X_train': X_train, 'y_train': y_train}, f)

def run():
    return "Метод k-ближайших соседей выполнен успешно!"
