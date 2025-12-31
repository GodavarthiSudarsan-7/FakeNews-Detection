import sys
import joblib
import re
import nltk
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from urllib.parse import urlparse

nltk.download('stopwords', quiet=True)
STOP = set(stopwords.words('english'))

TRUSTED_DOMAINS = [
    "isro.gov.in",
    "pib.gov.in",
    "nasa.gov",
    "esa.int",
    "who.int",
    "gov.in"
]

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

def is_trusted_url(url):
    if not url:
        return False
    domain = urlparse(url).netloc.lower()
    return any(d in domain for d in TRUSTED_DOMAINS)

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
            self.tfidf = getattr(m, "tfidf")
            try:
                self.clf = getattr(m, "calib").estimator if hasattr(getattr(m, "calib"), "estimator") else getattr(m, "calib")
            except Exception:
                self.clf = None
        else:
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
            return None

    def explain(self, text, top_k=7):
        if self.tfidf is None or self.clf is None:
            return []
        fv = self.tfidf.transform([text])
        nz = fv.nonzero()[1]
        feature_names = self._get_feature_names()
        coefs = self._get_coefs()
        influences = []
        if feature_names is None or coefs is None:
            return []
        for idx in nz:
            if idx < len(coefs):
                influence = float(coefs[idx]) * float(fv[0, idx])
                influences.append((feature_names[idx], influence))
        influences.sort(key=lambda x: abs(x[1]), reverse=True)
        return influences[:top_k]

    def predict(self, title, text, url=None):
        content = clean_text(str(title) + " " + str(text))
        model = self.pipeline if self.pipeline is not None else self.raw_model
        proba = model.predict_proba([content])[0]
        real_prob = float(proba[0])
        fake_prob = float(proba[1])

        if is_trusted_url(url):
            label = "REAL"
            confidence = 0.92
        else:
            label = "REAL" if real_prob >= 0.6 else "FAKE"
            confidence = max(real_prob, fake_prob)

        explanations = self.explain(content, top_k=7)
        formatted = [{"word": w, "influence": float(v)} for w, v in explanations]

        return {
            "label": label,
            "confidence": round(confidence, 2),
            "explanations": formatted
        }

if __name__ == "__main__":
    p = FakeNewsPredictor()
    print(p.predict("ISRO launches PSLV successfully", "ISRO conducted a successful mission.", "https://www.isro.gov.in"))
