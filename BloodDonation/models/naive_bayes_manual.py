import numpy as np
import os
import pickle

def load_data(filepath):
    filepath = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'blood_data.txt')
    data = np.genfromtxt(filepath, delimiter=",", skip_header=1)
    X = data[:, :-1]
    y = data[:, -1].astype(int)
    return X, y

def summarize_by_class(X, y):
    summaries = {}
    for c in np.unique(y):
        X_c = X[y == c]
        summaries[c] = [(np.mean(f), np.std(f)) for f in X_c.T]
    return summaries

def gaussian_prob(x, mean, stdev):
    if stdev == 0: stdev = 1e-6
    exp = np.exp(-((x - mean) ** 2) / (2 * stdev ** 2))
    return (1 / (np.sqrt(2 * np.pi) * stdev)) * exp

def predict(summaries, row):
    probs = {}
    for class_val, class_summ in summaries.items():
        probs[class_val] = 1
        for i in range(len(row)):
            mean, stdev = class_summ[i]
            probs[class_val] *= gaussian_prob(row[i], mean, stdev)
    return max(probs, key=probs.get)

def evaluate(X_test, y_test, summaries):
    predictions = [predict(summaries, row) for row in X_test]
    return np.mean(predictions == y_test)

# === Main ===
X, y = load_data("../dataset/blood_data.txt")
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

summaries = summarize_by_class(X_train, y_train)
acc = evaluate(X_test, y_test, summaries)
print("Naive Bayes Accuracy:", acc)

with open('naive_bayes_model.pkl', 'wb') as f:
    pickle.dump(summaries, f)

def run():
    return "Наивный Байес выполнен успешно!"
