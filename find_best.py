import pandas as pd
import re
import nltk
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, make_scorer, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

# Загрузка NLTK ресурсов
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# === 1. Загрузка данных ===
df = pd.read_csv("cat_emails_v2.csv")

# === 2. Улучшенная очистка текста ===
def clean_text(text):
    # Преобразование в нижний регистр
    text = str(text).lower()
    
    # Удаление HTML тегов
    text = re.sub(r'<.*?>', '', text)
    
    # Удаление email адресов
    text = re.sub(r'\S+@\S+', 'EMAIL', text)
    
    # Удаление URL
    text = re.sub(r'http\S+|www.\S+', 'URL', text)
    
    # Удаление специальных символов, оставляя буквы и пробелы
    text = re.sub(r'[^a-zäöüß\s]', ' ', text)
    
    # Разделение на слова
    tokens = text.split()
    
    # Лемматизация и удаление стоп-слов
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    
    return " ".join(tokens)

# === 3. Создание дополнительных признаков ===
def extract_features(text):
    # Длина текста
    length = len(text)
    
    # Количество слов
    word_count = len(text.split())
    
    # Среднее количество символов в слове
    avg_word_length = length / word_count if word_count > 0 else 0
    
    return pd.Series({
        'text_length': length,
        'word_count': word_count,
        'avg_word_length': avg_word_length
    })

# Подготовка данных
df['clean_text'] = df['subject'].fillna('') + " " + df['email'].fillna('')
df['clean_text'] = df['clean_text'].apply(clean_text)

# Извлечение дополнительных признаков
features = df['clean_text'].apply(extract_features)
df = pd.concat([df, features], axis=1)

# === 4. Разделение на train/test ===
X_train, X_test, y_train, y_test = train_test_split(
    df[['clean_text', 'text_length', 'word_count', 'avg_word_length']], 
    df['category'], 
    test_size=0.1, 
    random_state=42,
    stratify=df['category']  # Стратифицированное разделение
)

# === 5. Создание пайплайна с TF-IDF и классификатором ===
# Создаем TF-IDF векторизатор с n-граммами
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),  # униграммы и биграммы
    max_features=10000,   # ограничиваем количество признаков
    min_df=2,            # минимальная частота документов
    max_df=0.95          # максимальная частота документов
)

# === 6. Подготовка данных и балансировка классов ===
# Преобразование текстовых данных
X_train_tfidf = tfidf.fit_transform(X_train['clean_text'])
X_test_tfidf = tfidf.transform(X_test['clean_text'])

# Добавление дополнительных признаков
X_train_extra = X_train[['text_length', 'word_count', 'avg_word_length']].values
X_test_extra = X_test[['text_length', 'word_count', 'avg_word_length']].values

# Объединение TF-IDF и дополнительных признаков
X_train_combined = np.hstack((X_train_tfidf.toarray(), X_train_extra))
X_test_combined = np.hstack((X_test_tfidf.toarray(), X_test_extra))

# Применение SMOTE для балансировки классов
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_combined, y_train)

# === 7. Настройка и обучение моделей ===
# Параметры для RandomForest
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Параметры для XGBoost
xgb_params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0]
}

models = {
    'MultinomialNB': MultinomialNB(),
    'LinearSVC': LinearSVC(random_state=42, max_iter=1000),
    'RandomForest': GridSearchCV(
        RandomForestClassifier(random_state=42),
        rf_params = {
    'n_estimators': [100],
    'max_depth': [10, None],
    'min_samples_split': [2],
    'min_samples_leaf': [1]
},
        cv=3,
        scoring='f1_weighted',
        n_jobs=-1
    ),

}

best_accuracy = 0
best_model_name = None
best_model = None
best_params = None

for name, model in models.items():
    print(f"\nОбучение модели: {name}")
    
    # Обучение модели
    if isinstance(model, GridSearchCV):
        model.fit(X_train_balanced, y_train_balanced)
        print(f"Лучшие параметры для {name}:")
        print(model.best_params_)
        y_pred = model.predict(X_test_combined)
    else:
        model.fit(X_train_balanced, y_train_balanced)
        y_pred = model.predict(X_test_combined)
    
    # Оценка модели
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Точность {name}: {accuracy:.4f}")
    print("\nОтчет по классификации:")
    print(classification_report(y_test, y_pred))
    
    # Проверка на лучшую модель
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = name
        best_model = model if not isinstance(model, GridSearchCV) else model.best_estimator_
        best_params = model.best_params_ if isinstance(model, GridSearchCV) else None

print(f"\The best model: {best_model_name} accuracy: {best_accuracy:.4f}")
