import os
import numpy as np
import pickle

def load_data():
    filepath = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'blood_data.txt')
    data = np.genfromtxt(filepath, delimiter=",", skip_header=1)
    X = data[:, :-1]
    y = data[:, -1]
    return X, y

def train_linear_regression(X, y):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # добавляем столбец единичных значений
    # Используем псевдообратную матрицу для решения проблемы вырождения
    theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
    return theta

def predict(X, theta):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # добавляем столбец единичных значений
    return X_b @ theta  # предсказания

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)  # вычисление MSE

def run():
    try:
        X, y = load_data()
        split = int(0.8 * len(X))  # 80% для тренировки, 20% для теста
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        theta = train_linear_regression(X_train, y_train)
        y_pred = predict(X_test, theta)
        mse = mean_squared_error(y_test, y_pred)

        # Печать параметров модели (theta)
        print("Параметры модели (theta):", theta)

        # Путь для сохранения модели
        model_filepath = os.path.join(os.path.dirname(__file__), 'linear_regression_model.pkl')

        # Сохранение модели
        with open(model_filepath, 'wb') as f:
            pickle.dump(theta, f)

        return f"Linear Regression completed successfully! MSE = {mse:.4f}"

    except Exception as e:
        return f"Error during Linear Regression: {str(e)}"

if __name__ == '__main__':
    print(run())
