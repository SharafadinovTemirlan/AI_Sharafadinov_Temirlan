import numpy as np
import os
import pickle

def load_data():
    filepath = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'blood_data.txt')
    data = np.genfromtxt(filepath, delimiter=",", skip_header=1)
    X = data[:, :-1]
    y = data[:, -1]
    return X, y

def gini_impurity(groups):
    total = sum(len(g) for g in groups)
    impurity = 0.0
    for group in groups:
        if len(group) == 0:
            continue
        score = 0.0
        labels = [row[-1] for row in group]
        for class_val in [0, 1]:
            p = labels.count(class_val) / len(group)
            score += p * p
        impurity += (1.0 - score) * (len(group) / total)
    return impurity

def split_dataset(index, threshold, dataset):
    left, right = [], []
    for row in dataset:
        if row[index] < threshold:
            left.append(row)
        else:
            right.append(row)
    return left, right

def get_best_split(dataset):
    best_index, best_thresh, best_score, best_groups = None, None, float('inf'), None
    for index in range(len(dataset[0]) - 1):
        values = set(row[index] for row in dataset)
        for val in values:
            groups = split_dataset(index, val, dataset)
            score = gini_impurity(groups)
            if score < best_score:
                best_index, best_thresh, best_score, best_groups = index, val, score, groups
    return {'index': best_index, 'threshold': best_thresh, 'groups': best_groups}

def predict(tree, row):
    index, threshold = tree['index'], tree['threshold']
    return tree['left'] if row[index] < threshold else tree['right']

def build_tree(dataset):
    split = get_best_split(dataset)
    left, right = split['groups']
    split['left'] = round(sum(row[-1] for row in left) / len(left))
    split['right'] = round(sum(row[-1] for row in right) / len(right))
    return split

def evaluate(X_test, y_test, tree):
    predictions = [predict(tree, row) for row in X_test]
    return np.mean(predictions == y_test)

# === Main ===
X, y = load_data()
data = np.column_stack((X, y))

split = int(0.8 * len(data))
train_data, test_data = data[:split], data[split:]
X_test, y_test = test_data[:, :-1], test_data[:, -1]

tree = build_tree(train_data)
acc = evaluate(X_test, y_test, tree)
print("Decision Tree Accuracy:", acc)

def run():
    return "Дерево решений выполнено успешно!"

with open('decision_tree_model.pkl', 'wb') as f:
    pickle.dump(tree, f)