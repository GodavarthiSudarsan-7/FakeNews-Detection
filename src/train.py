import os
import re
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score, brier_score_loss
from sklearn.calibration import calibration_curve
import nltk
from nltk.corpus import stopwords
import json

nltk.download('stopwords', quiet=True)
STOP = set(stopwords.words('english'))


class CalibratedPipeline:
    def __init__(self, tfidf, calib):
        self.tfidf = tfidf
        self.calib = calib
    def predict(self, texts):
        X = self.tfidf.transform(texts)
        return self.calib.predict(X)
    def predict_proba(self, texts):
        X = self.tfidf.transform(texts)
        return self.calib.predict_proba(X)

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP and len(t) > 1]
    return " ".join(tokens)

def load_data(path):
    df = pd.read_csv(path, encoding='utf-8')
    if 'text' not in df.columns and 'article' in df.columns:
        df['text'] = df['article']
    if 'label' not in df.columns and 'target' in df.columns:
        df['label'] = df['target']
    if df['label'].dtype == object:
        df['label'] = df['label'].apply(lambda x: 1 if str(x).strip().lower() in ['fake','1','true'] else 0)
    return df[['title','text','label']].fillna('')

def choose_split_strategy(y, min_required_per_class=2):
    vc = pd.Series(y).value_counts().to_dict()
    ok = all([cnt >= min_required_per_class for cnt in vc.values()])
    return ok, vc

def fit_and_maybe_calibrate(X_train, y_train, X_val, y_val, cv_strategy, param_grid, n_jobs=-1):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(solver='liblinear', max_iter=2000))
    ])

    cv = cv_strategy if isinstance(cv_strategy, (StratifiedKFold, KFold)) else StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

    grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring='f1', n_jobs=n_jobs, verbose=1, error_score=np.nan)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    best_params = grid.best_params_
    print("GridSearch best params:", best_params)

    tfidf = best.named_steps['tfidf']
    clf = best.named_steps['clf']

    
    calibrated_model = None
    calibr_method = None
    calibrated = False
    if hasattr(clf, "predict_proba"):
        
        val_size = len(X_val)
        if val_size >= 30:
            method = 'isotonic'
        else:
            method = 'sigmoid'  
        try:
            
            calib = CalibratedClassifierCV(clf, cv='prefit', method=method)
            X_val_t = tfidf.transform(X_val)
            calib.fit(X_val_t, y_val)
           
            probs = calib.predict_proba(X_val_t)[:,1]
            if np.all((probs <= 1e-6) | (probs >= 1 - 1e-6)):
                print("Calibration resulted in degenerate probabilities; skipping calibration.")
                calibrated_model = best
                calibrated = False
            else:
                calibrated_model = CalibratedPipeline(tfidf, calib)
                calibrated = True
                calibr_method = method
                print(f"Calibration succeeded using method={method}.")
        except Exception as e:
            print("Calibration failed (fallback to uncalibrated pipeline). Error:", e)
            calibrated_model = best
            calibrated = False
    else:
        print("Classifier has no predict_proba; skipping calibration.")
        calibrated_model = best
        calibrated = False

    return calibrated_model, best, tfidf, clf, best_params, calibrated, calibr_method

def evaluate_and_report(model, X_test, y_test):
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Classification report:\n", classification_report(y_test, preds, zero_division=0))
    if proba is not None:
       
        proba = np.nan_to_num(proba, nan=0.0)
        proba = np.clip(proba, 0.0, 1.0)
        brier = brier_score_loss(y_test, proba)
        print("Brier score (lower better):", brier)
        try:
            n_bins = min(5, max(2, len(y_test)))
            prob_true, prob_pred = calibration_curve(y_test, proba, n_bins, strategy='quantile')
            print("Calibration curve (prob_true per bin):", prob_true.tolist())
            print("Calibration curve (prob_pred per bin):", prob_pred.tolist())
        except Exception as e:
            print("Could not compute calibration curve (probably too few samples):", e)
    else:
        print("No probability estimates available for calibration metrics.")

def save_model_artifacts(model, best_params, tfidf, clf, calibrated, calibr_method, out_path='models/model.joblib'):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    joblib.dump(model, out_path)
    meta = {
        "name": "fake-news-classifier",
        "saved_at": datetime.utcnow().isoformat() + "Z" if hasattr(datetime, "utcnow") else datetime.now().isoformat(),
        "artifact": out_path,
        "best_params": best_params if isinstance(best_params, dict) else str(best_params),
        "calibrated": bool(calibrated),
        "calibration_method": calibr_method if calibrated else None
    }
   
    try:
        if tfidf is not None:
            joblib.dump(tfidf, 'models/tfidf.joblib')
            meta['tfidf'] = 'models/tfidf.joblib'
    except Exception:
        pass
    try:
        if clf is not None:
            joblib.dump(clf, 'models/clf.joblib')
            meta['clf'] = 'models/clf.joblib'
    except Exception:
        pass
    with open('models/model_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print("Saved model and metadata in models/")

def main():
    df = load_data('data/train.csv')
    df['content'] = (df['title'].fillna('') + ' ' + df['text'].fillna('')).apply(clean_text)
    X = df['content']
    y = df['label'].astype(int)

    total = len(df)
    print("Total dataset size:", total)
    ok_for_stratify, class_counts = choose_split_strategy(y, min_required_per_class=4)
    print("Class counts:", class_counts)
    if not ok_for_stratify:
        print("WARNING: Some classes have fewer than 4 samples. We will use safer splits and a reduced hyperparameter grid.")

   
    stratify_param = y if ok_for_stratify else None

   
    if total >= 10:
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=stratify_param)
        
        _, vc_temp = choose_split_strategy(y_temp, min_required_per_class=2)
        strat_temp = y_temp if all([c >= 2 for c in vc_temp.values()]) else None
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=strat_temp)
    else:
       
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=None)
        X_val, y_val = X_test, y_test

    print("Train/Val/Test sizes:", len(X_train), len(X_val), len(X_test))

   
    if ok_for_stratify and len(X_train) >= 20:
        cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    else:
        cv_strategy = KFold(n_splits=2, shuffle=True, random_state=42)

    
    if total < 200:
        param_grid = {
            'tfidf__max_features': [2000, 5000],
            'tfidf__ngram_range': [(1,1)],
            'tfidf__min_df': [1],
            'clf__C': [0.5,1.0],
            'clf__class_weight': [None, 'balanced']
        }
    else:
        param_grid = {
            'tfidf__max_features': [5000, 10000, 20000],
            'tfidf__ngram_range': [(1,1),(1,2)],
            'tfidf__min_df': [1,2],
            'clf__C': [0.5,1.0,2.0],
            'clf__class_weight': [None, 'balanced']
        }

    final_model, best_pipeline, tfidf, clf, best_params, calibrated, calibr_method = fit_and_maybe_calibrate(
        X_train, y_train, X_val, y_val, cv_strategy, param_grid, n_jobs=-1
    )

    evaluate_and_report(final_model, X_test, y_test)

    save_model_artifacts(final_model, best_params, tfidf, clf, calibrated, calibr_method, out_path='models/model.joblib')

if __name__ == "__main__":
    main()
