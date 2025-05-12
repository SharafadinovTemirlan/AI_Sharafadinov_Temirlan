from flask import Flask, render_template, request
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import io
import base64
import flask.cli

flask.cli.show_server_banner = lambda *args, **kwargs: None
app = Flask(__name__)

# --- Загрузка моделей ---
with open('models/linear_regression_model.pkl', 'rb') as f:
    linear_regression_model = pickle.load(f)
with open('models/logistic_regression_model.pkl', 'rb') as f:
    logistic_regression_model = pickle.load(f)
with open('models/decision_tree_model.pkl', 'rb') as f:
    decision_tree_model = pickle.load(f)
with open('models/random_forest_model.pkl', 'rb') as f:
    random_forest_model = pickle.load(f)
with open('models/naive_bayes_model.pkl', 'rb') as f:
    naive_bayes_model = pickle.load(f)
with open('models/knn_model.pkl', 'rb') as f:
    knn_model = pickle.load(f)
# --- Конец загрузки моделей ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        recency = float(request.form['recency'])
        frequency = float(request.form['frequency'])
        monetary = float(request.form['monetary'])
        time = float(request.form['time'])
        selected_model = request.form['model']

        input_data = np.array([[recency, frequency, monetary, time]])

        if selected_model == 'linear_regression':
            prediction = linear_regression_model.predict(input_data)[0]
            result = {'Модель': 'Линейная регрессия', 'Результат': f'{prediction:.2f} мл крови'}
        else:
            models = {
                'logistic_regression': (logistic_regression_model, 'Логистическая регрессия'),
                'decision_tree': (decision_tree_model, 'Дерево решений'),
                'random_forest': (random_forest_model, 'Случайный лес'),
                'naive_bayes': (naive_bayes_model, 'Наивный Байес'),
                'knn': (knn_model, 'K-ближайших соседей')
            }

            if selected_model not in models:
                return "Ошибка: неверная модель", 400

            model, name = models[selected_model]
            prediction = model.predict(input_data)[0]
            result = {'Модель': name, 'Результат': 'Сдаст кровь' if prediction == 1 else 'Не сдаст кровь'}

        img = create_plot(input_data[:, :2])
        cluster_img = create_cluster_plot(input_data[:, :2])

        return render_template('index.html', result=result, img=img, cluster_img=cluster_img)

    except Exception as e:
        return f"Ошибка: {e}", 500

def create_plot(input_data):
    x = np.linspace(-10, 10, 100)
    X_full = np.hstack([np.array([[i, 0]]), np.zeros((100, 2))])
    y = linear_regression_model.predict(X_full)

    plt.figure(figsize=(6, 4))
    plt.scatter(input_data[:, 0], input_data[:, 1], color='red')
    plt.plot(x, y, color='blue')
    plt.title('Линейная регрессия (по 2 признакам)')
    plt.xlabel('Признак 1')
    plt.ylabel('Признак 2')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_data = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    return img_data

def create_cluster_plot(input_data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(input_data)

    kmeans = KMeans(n_clusters=2, n_init=10)
    labels = kmeans.fit_predict(scaled_data)

    plt.figure(figsize=(6, 4))
    plt.scatter(input_data[:, 0], input_data[:, 1], c=labels, cmap='viridis')
    plt.title('KMeans кластеризация (2 признака)')
    plt.xlabel('Признак 1')
    plt.ylabel('Признак 2')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_data = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    return img_data

if __name__ == '__main__':
    app.run(debug=False)
