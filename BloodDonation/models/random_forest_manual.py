import os
import numpy as np
import random
import pickle
from models.decision_tree_manual import build_tree, predict  # Assuming your decision tree module exists

def load_data(filepath):
    data = np.genfromtxt(filepath, delimiter=",", skip_header=1)
    X = data[:, :-1]
    y = data[:, -1]
    return np.column_stack((X, y))

def bootstrap_sample(data):
    n_samples = len(data)
    return np.array([data[random.randint(0, n_samples - 1)] for _ in range(n_samples)])

def random_forest_train(data, n_trees=5):
    forest = []
    for _ in range(n_trees):
        sample = bootstrap_sample(data)
        tree = build_tree(sample)
        forest.append(tree)
    return forest

def random_forest_predict(forest, row):
    votes = [predict(tree, row) for tree in forest]
    return round(sum(votes) / len(votes))

def evaluate_rf(forest, X_test, y_test):
    predictions = [random_forest_predict(forest, row) for row in X_test]
    return np.mean(predictions == y_test)

# === Main ===
filepath = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'blood_data.txt')
data = np.genfromtxt(filepath, delimiter=",", skip_header=1)
split = int(0.8 * len(data))
train_data, test_data = data[:split], data[split:]
X_test, y_test = test_data[:, :-1], test_data[:, -1]

forest = random_forest_train(train_data, n_trees=5)
acc = evaluate_rf(forest, X_test, y_test)

with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(forest, f)

print("Random Forest Accuracy:", acc)

