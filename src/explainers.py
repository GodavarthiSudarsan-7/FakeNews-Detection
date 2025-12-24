
import os
import io
import base64
import joblib
import numpy as np
import matplotlib.pyplot as plt

from lime.lime_text import LimeTextExplainer
import shap


MODEL_PATH = os.path.join(os.getcwd(), "models", "model.joblib")

def _load_model():
    return joblib.load(MODEL_PATH)

def lime_explain(text, class_names=None, num_features=10):
   
    model = _load_model()

    
    def predict_proba(texts):
       
        return model.predict_proba(texts)

    explainer = LimeTextExplainer(class_names=class_names or ['REAL','FAKE'])
    exp = explainer.explain_instance(text, predict_proba, num_features=num_features)
    
    return exp.as_list()

def shap_explain_bar(texts, max_display=10, out_png_path="static/shap_explain.png"):
    
    model = _load_model()
   
    tfidf = None
    clf = None
    if hasattr(model, "named_steps"):
        tfidf = model.named_steps.get('tfidf')
        clf = model.named_steps.get('clf')
    elif hasattr(model, "tfidf") and hasattr(model, "calib"):
        tfidf = getattr(model, "tfidf")
       
        calib = getattr(model, "calib")
        clf = getattr(calib, "estimator") if hasattr(calib, "estimator") else calib
    else:
       
        tfidf = getattr(model, "tfidf", None)
        calib = getattr(model, "calib", None)
        clf = getattr(calib, "estimator", None) if calib is not None and hasattr(calib, "estimator") else calib

    if tfidf is None or clf is None:
        raise RuntimeError("SHAP: model structure not supported for shap linear explainer.")

   
    X = tfidf.transform(texts)
   
    try:
        X_small = X[: min(20, X.shape[0])].toarray()
    except Exception:
        X_small = X.toarray()

   
    try:
        explainer = shap.LinearExplainer(clf, X_small, feature_dependence="independent")
       
        shap_values = explainer.shap_values(X.toarray())
       
        if isinstance(shap_values, list) and len(shap_values) >= 2:
            vals = shap_values[1][0]
        else:
            vals = shap_values[0][0]
    except Exception as e:
        raise RuntimeError(f"SHAP explainer failed: {e}")

   
    try:
        feature_names = tfidf.get_feature_names_out()
    except Exception:
        feature_names = ['f%d' % i for i in range(X.shape[1])]
    
    idx = np.argsort(np.abs(vals))[-max_display:][::-1]
    top_names = [feature_names[i] for i in idx]
    top_vals = [float(vals[i]) for i in idx]

   
    plt.figure(figsize=(8, max(2, 0.4 * len(top_names))))
    colors = ['#d9534f' if v > 0 else '#5cb85c' for v in top_vals]
    y_pos = np.arange(len(top_names))
    plt.barh(y_pos, top_vals, color=colors)
    plt.yticks(y_pos, top_names)
    plt.xlabel("SHAP value (impact on prediction)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)
    plt.savefig(out_png_path, bbox_inches='tight')
    plt.close()
    return out_png_path
