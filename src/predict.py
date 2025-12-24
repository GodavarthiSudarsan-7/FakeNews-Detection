
import sys
import joblib
import re
import nltk
from nltk.corpus import stopwords
import pandas as pd
import numpy as np

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


try:
    import __main__ as _main
    if not hasattr(_main, 'CalibratedPipeline'):
        setattr(_main, 'CalibratedPipeline', CalibratedPipeline)
except Exception:
    pass

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP and len(t) > 1]
    return " ".join(tokens)

class FakeNewsPredictor:
    def __init__(self, model_path='models/model.joblib'):
      
        self.raw_model = joblib.load(model_path)
        self.pipeline = None
        self.tfidf = None
        self.clf = None
        self._unpack_model()

    def _unpack_model(self):
        m = self.raw_model
       
        if hasattr(m, "named_steps"):
            self.pipeline = m
            self.tfidf = m.named_steps.get('tfidf')
            self.clf = m.named_steps.get('clf') or m.named_steps.get('classifier')
       
        elif hasattr(m, "tfidf") and hasattr(m, "calib"):
            self.pipeline = None
            self.tfidf = getattr(m, "tfidf")
           
            try:
                self.clf = getattr(m, "calib").estimator if hasattr(getattr(m, "calib"), "estimator") else getattr(m, "calib")
            except Exception:
                self.clf = None
        else:
          
            self.pipeline = None
            self.tfidf = None
            self.clf = m

    def _get_feature_names(self):
        try:
            return self.tfidf.get_feature_names_out()
        except Exception:
            try:
                return self.tfidf.get_feature_names()
            except Exception:
                return None

    def _get_coefs(self):
       
        try:
            return self.clf.coef_[0]
        except Exception:
            try:
                return getattr(self.clf, 'feature_importances_')
            except Exception:
                return None

    def explain(self, text, top_k=7):
        if self.tfidf is None or self.clf is None:
            return []
        fv = self.tfidf.transform([text])
        nz = fv.nonzero()[1]
        if len(nz) == 0:
            return []
        feature_names = self._get_feature_names()
        coefs = self._get_coefs()
        influences = []
        if coefs is not None and feature_names is not None:
            for idx in nz:
                if idx >= len(coefs):
                    weight = 0.0
                else:
                    weight = float(coefs[idx])
                word = feature_names[idx] if idx < len(feature_names) else f"w{idx}"
                tfidf_val = float(fv[0, idx])
                influence = weight * tfidf_val
                influences.append((word, float(influence), float(weight)))
            influences = sorted(influences, key=lambda x: abs(x[1]), reverse=True)
            return influences[:top_k]
        else:
            
            words = []
            for idx in nz:
                word = feature_names[idx] if feature_names is not None and idx < len(feature_names) else f"w{idx}"
                tfidf_val = float(fv[0, idx])
                words.append((word, float(tfidf_val), 0.0))
            words = sorted(words, key=lambda x: abs(x[1]), reverse=True)
            return words[:top_k]

    def predict(self, title, text):
        content = clean_text(str(title) + ' ' + str(text))
        model = self.pipeline if self.pipeline is not None else self.raw_model
        pred = model.predict([content])[0]
        proba = model.predict_proba([content])[0] if hasattr(model, "predict_proba") else [0.0,0.0]
        label = 'FAKE' if int(pred) == 1 else 'REAL'
        confidence = float(max(proba)) if isinstance(proba, (list, tuple, np.ndarray)) else float(proba)
        explanations = self.explain(content, top_k=7)
        formatted_explanations = []
        for w, infl, weight in explanations:
            formatted_explanations.append({
                "word": str(w),
                "influence": float(infl),
                "coef": float(weight)
            })
        return {'label': label, 'confidence': confidence, 'explanations': formatted_explanations}

if __name__ == "__main__":
    p = FakeNewsPredictor()
    print(p.predict("NASA confirms water on Mars", "New study shows water traces on Mars."))
