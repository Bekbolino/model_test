# Email Category Classifier — Demo

This workspace contains a simple pipeline and demo for classifying incoming emails into predefined categories.

Contents
- `cat_emails_v2.csv` — dataset (provided)
- `train_linear_svc_demo.py` — train LinearSVC with TF-IDF + 3 extra numeric features and save artifacts (train/test split 4500/500)
- `train_email_classifier_improved.py` — alternative multi-model script (kept for reference)
- `linear_svc_normalized.py` — experiment used to validate LinearSVC
- `linear_svc_model.pkl`, `vectorizer.pkl`, `scaler.pkl`, `label_classes.npy` — saved artifacts (created by training script)
- `metrics.json` — saved test metrics (accuracy, classification report)
- `predict_email_saved.py` — batch and single prediction utility using saved artifacts
- `predict_cli.py` — interactive CLI for entering subject + multi-line body and printing predicted category
- `predict_email.py` — simple interactive script (subject + body) using saved artifacts

Quick demo (requirements)
- Python 3.8+ (virtualenv recommended)
- Install dependencies in the venv:

```powershell
pip install -r requirements.txt
```

(If you don't have `requirements.txt`, install: `pip install pandas scikit-learn joblib numpy imbalanced-learn xgboost`)

Train (reproduce artifacts)

```powershell
python train_linear_svc_demo.py
```

This will:
- Train LinearSVC on 4,500 training and 500 test samples (test_size=0.1, stratified)
- Save `linear_svc_model.pkl`, `vectorizer.pkl`, `scaler.pkl`, `label_classes.npy`
- Save `metrics.json` with accuracy and per-class metrics

Check metrics

```powershell
python -c "import json; print(json.load(open('metrics.json'))['accuracy'])"
```

Interactive demo

```powershell
python predict_cli.py --interactive
```

Type a subject, then type email body lines. Finish body input by entering a line containing only `END`.
The script will print the predicted category.

Batch prediction

```powershell
python predict_email_saved.py --input incoming.csv --output predictions.csv
```

Input CSV must have columns `subject` and `email`.

Notes & Observations

- Best performing pipeline in experiments: LinearSVC with TF-IDF (uni+bi-grams) and 3 numeric features (text length, word count, avg word length). Accuracy on the held-out 500-sample test: ~76.4%.
- SMOTE and RandomForest were tested; LinearSVC with TF-IDF performed best in this dataset.
- TF-IDF -> sparse matrix; current implementation converts to dense arrays when concatenating numeric features (`toarray()`), which can be memory heavy for large batches. For production, wrap vectorizer + scaler + classifier in a `ColumnTransformer` or `Pipeline` to avoid densifying.

Next steps (suggested)
- Add a small web UI (Flask/FastAPI) for interactive demo.
- Replace dense concatenation with ColumnTransformer pipeline to save memory.
- Hyperparameter tuning (RandomizedSearchCV) for LinearSVC or try LightGBM/XGBoost with TF-IDF features.

If you want, I can:
- Add a one-file Flask demo that accepts subject/body and returns predicted category.
- Convert the prediction pipeline into a single `sklearn.Pipeline` and re-save artifacts.
- Prepare a short demo notebook or script that shows confusion matrix and examples of correct/incorrect classifications.

"# model_test" 
