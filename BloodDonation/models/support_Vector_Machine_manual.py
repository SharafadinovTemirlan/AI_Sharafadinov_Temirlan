import numpy as np
import os
import pickle

def load_data(filepath):
    filepath = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'blood_data.txt')
    data = np.genfromtxt(filepath, delimiter=",", skip_header=1)
    X = data[:, :-1]
    y = data[:, -1]
    y = np.where(y == 0, -1, 1)  # Convert labels to -1 and 1
    return X, y

class SVM:
    def __init__(self, lr=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y[idx]))
                    self.b -= self.lr * y[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) + self.b
        return np.sign(approx)

# === Main ===
X, y = load_data("../dataset/blood_data.txt")
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

svm = SVM()
svm.fit(X_train, y_train)
predictions = svm.predict(X_test)
accuracy = np.mean(predictions == y_test)

print("SVM Accuracy:", accuracy)

def run():
    return "SVM completed successfully!"

# Save the trained SVM model
with open('svm_model.pkl', 'wb') as f:
    pickle.dump(svm, f)
