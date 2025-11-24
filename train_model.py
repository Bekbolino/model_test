import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

# Загружаем и подготавливаем данные
import pandas as pd

try:
    df = pd.read_csv(
        "cat_emails_v2.csv",
        encoding="utf-8",
        sep=",",
        quotechar='"',
        engine="python",    # или 'c' если нет проблем с многострочными полями
        dtype=str
    )
except UnicodeDecodeError:
    df = pd.read_csv("cat_emails_v2.csv", encoding="cp1251", sep=",", quotechar='"', engine="python", dtype=str)

def clean_text(text):
    text = str(text).lower()
    return text

def extract_features(text):
    length = len(text)
    word_count = len(text.split())
    avg_word_length = length / word_count if word_count > 0 else 0
    return pd.Series({
        'text_length': length,
        'word_count': word_count,
        'avg_word_length': avg_word_length
    })

# Подготовка данных
df['clean_text'] = df['subject'].fillna('') + " " + df['email'].fillna('')
df['clean_text'] = df['clean_text'].apply(clean_text)

# Добавляем дополнительные признаки
features = df['clean_text'].apply(extract_features)
df = pd.concat([df, features], axis=1)

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(
    df[['clean_text', 'text_length', 'word_count', 'avg_word_length']], 
    df['category'], 
    test_size=0.1, 
    random_state=42,
    stratify=df['category']
)

# TF-IDF
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=10000,
    min_df=2,
    max_df=0.95
)

X_train_tfidf = vectorizer.fit_transform(X_train['clean_text'])
X_test_tfidf = vectorizer.transform(X_test['clean_text'])

# Нормализация числовых признаков
scaler = StandardScaler()
X_train_extra = scaler.fit_transform(X_train[['text_length', 'word_count', 'avg_word_length']])
X_test_extra = scaler.transform(X_test[['text_length', 'word_count', 'avg_word_length']])

# Объединение признаков
X_train_combined = np.hstack((X_train_tfidf.toarray(), X_train_extra))
X_test_combined = np.hstack((X_test_tfidf.toarray(), X_test_extra))

print("=== LinearSVC с TF-IDF + нормализованные доп. признаки ===")
model = LinearSVC(random_state=42, max_iter=2000)  # Увеличили число итераций
model.fit(X_train_combined, y_train)
y_pred = model.predict(X_test_combined)
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность: {accuracy:.4f}")
print("\nПодробный отчет:")
print(classification_report(y_test, y_pred))

# Сохраняем модель, векторизатор и скейлер для дальнейшего использования
model_filename = "linear_svc_model.pkl"
vectorizer_filename = "vectorizer.pkl"
scaler_filename = "scaler.pkl"
classes_filename = "label_classes.npy"

joblib.dump(model, model_filename)
joblib.dump(vectorizer, vectorizer_filename)
joblib.dump(scaler, scaler_filename)
import numpy as _np
_np.save(classes_filename, model.classes_)

print(f"\nМодель сохранена как {model_filename}")
print(f"Векторизатор сохранён как {vectorizer_filename}")
print(f"Скейлер сохранён как {scaler_filename}")
print(f"Классы меток сохранены как {classes_filename}")