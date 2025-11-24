from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# загружаем заранее модель и вспомогательные объекты
model = joblib.load('linear_svc_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
scaler = joblib.load('scaler.pkl')
label_classes = np.load('label_classes.npy', allow_pickle=True)

def extract_features(text):
    length = len(text)
    words = text.split()
    avg_word = (length / len(words)) if words else 0
    return [length, len(words), avg_word]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')

    # Получаем текст
    subject = request.form.get('subject', '')
    body = request.form.get('body', '')
    text = (subject + ' ' + body).lower()

    # Векторизация и доп. признаки
    X_tfidf = vectorizer.transform([text])
    extra = np.array([extract_features(text)])
    X_combined = np.hstack((X_tfidf.toarray(), scaler.transform(extra)))

    # Предсказание
    pred = model.predict(X_combined)[0]
    label = label_classes[int(pred)] if isinstance(pred, (int, np.integer)) else pred

    return jsonify({'category': str(label)})

if __name__ == '__main__':
    app.run(debug=True)
