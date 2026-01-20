import joblib
import re
import nltk
from nltk.corpus import stopwords
import pandas as pd
from urllib.parse import urlparse

class CalibratedPipeline:
    pass

nltk.download("stopwords", quiet=True)
STOP = set(stopwords.words("english"))

TRUSTED_DOMAINS = [
    "isro.gov.in",
    "pib.gov.in",
    "nasa.gov",
    "esa.int",
    "who.int",
    "gov.in"
]

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP and len(t) > 1]
    return " ".join(tokens)

def is_trusted_url(url):
    if not url:
        return False
    domain = urlparse(url).netloc.lower()
    return any(d in domain for d in TRUSTED_DOMAINS)

class FakeNewsPredictor:
    def __init__(self, model_path="models/model.joblib"):
        self.model = joblib.load(model_path)
        self.tfidf = None
        self.clf = None
        if hasattr(self.model, "named_steps"):
            self.tfidf = self.model.named_steps.get("tfidf")
            self.clf = self.model.named_steps.get("clf") or self.model.named_steps.get("classifier")

    def _get_feature_names(self):
        if self.tfidf is None:
            return None
        try:
            return self.tfidf.get_feature_names_out()
        except Exception:
            try:
                return self.tfidf.get_feature_names()
            except Exception:
                return None

    def _get_coefs(self):
        if self.clf is None:
            return None
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
        if feature_names is None or coefs is None:
            return []
        influences = []
        for idx in nz:
            if idx < len(coefs):
                influences.append((feature_names[idx], float(coefs[idx]) * float(fv[0, idx])))
        influences.sort(key=lambda x: abs(x[1]), reverse=True)
        return influences[:top_k]

    def predict(self, title, text, url=None):
        if is_trusted_url(url):
            return {
                "label": "REAL",
                "confidence": 0.95,
                "explanations": []
            }

        content = clean_text(str(title) + " " + str(text))
        proba = self.model.predict_proba([content])[0]
        classes = self.model.classes_
        prob_map = dict(zip(classes, proba))

        real_prob = prob_map.get(0, 0.0)
        fake_prob = prob_map.get(1, 0.0)

        if real_prob >= 0.6:
            label = "REAL"
            confidence = real_prob
        else:
            label = "FAKE"
            confidence = fake_prob

        explanations = self.explain(content, top_k=7)
        formatted = [{"word": w, "influence": float(v)} for w, v in explanations]

        return {
            "label": label,
            "confidence": round(float(confidence), 2),
            "explanations": formatted
        }

if __name__ == "__main__":
    p = FakeNewsPredictor()
    print(
        p.predict(
            "ISRO launches PSLV successfully",
            "ISRO conducted a successful mission.",
            "https://www.isro.gov.in"
        )
    )
