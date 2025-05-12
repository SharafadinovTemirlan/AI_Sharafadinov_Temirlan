import numpy as np
import matplotlib.pyplot as plt
import os

def load_data(filepath):
    filepath = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'blood_data.txt')
    data = np.genfromtxt(filepath, delimiter=",", skip_header=1)
    X = data[:, :-1]  # Все столбцы, кроме последнего (label)
    return X

def pca(X, n_components=2):
    X_meaned = X - np.mean(X, axis=0)
    cov_matrix = np.cov(X_meaned, rowvar=False)
    eigen_vals, eigen_vecs = np.linalg.eigh(cov_matrix)
    sorted_idx = np.argsort(eigen_vals)[::-1]
    eigen_vecs = eigen_vecs[:, sorted_idx]
    eigen_vecs = eigen_vecs[:, :n_components]
    return np.dot(X_meaned, eigen_vecs)

# === Main ===
X = load_data("../dataset/blood_data.txt")
X_pca = pca(X, n_components=2)

# Визуализация
plt.scatter(X_pca[:, 0], X_pca[:, 1], c='blue', alpha=0.6)
plt.title("PCA: Blood Donation Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.show()


def run():
    return "Метод главных компонент (PCA) выполнен успешно!"
