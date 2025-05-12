import numpy as np
import os

def load_data(filepath):
    filepath = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'blood_data.txt')
    data = np.genfromtxt(filepath, delimiter=",", skip_header=1)
    X = data[:, :-1]
    return X

class KMeans:
    def __init__(self, k=2, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None

    def fit(self, X):
        n_samples, n_features = X.shape
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_indices]

        for _ in range(self.max_iters):
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            new_centroids = np.array([X[labels == i].mean(axis=0) if np.any(labels == i) else self.centroids[i] for i in range(self.k)])
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

# === Main ===
X = load_data("../dataset/blood_data.txt")
kmeans = KMeans(k=2)
kmeans.fit(X)
labels = kmeans.predict(X)
print("K-Means cluster assignments:", labels)


def run():
    return "K-средних кластеризация выполнена успешно!"
