
from flask import Flask, render_template, request, jsonify, send_file
from predict import FakeNewsPredictor, clean_text
from url_fetcher import fetch_article_text
from db import init_db, save_prediction, fetch_recent
from explainers import lime_explain, shap_explain_bar
import io, csv, math, os

app = Flask(__name__, template_folder='../templates', static_folder='../static')


THRESHOLD = 0.70


init_db()
predictor = FakeNewsPredictor()

def apply_threshold(raw_label, confidence, threshold=THRESHOLD):
   
    if confidence is None or (isinstance(confidence, float) and math.isnan(confidence)):
        return raw_label, "No confidence available"
    if confidence >= threshold:
        return raw_label, ""
    else:
        return "UNCERTAIN", f"Model suggests {raw_label} but confidence is only {confidence*100:.1f}%."

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    title = request.form.get('title', '').strip()
    text = request.form.get('text', '').strip()
    if not title and not text:
        return render_template('index.html', result={'error': 'Provide title or article text.'}, title=title, text=text)

   
    res = predictor.predict(title, text)
    raw_label = res.get('label')
    confidence = res.get('confidence', 0.0)
    display_label, note = apply_threshold(raw_label, confidence, THRESHOLD)

    snippet = (text or '')[:800]
    try:
        save_prediction(title, snippet, '', raw_label, confidence, str(res.get('explanations')))
    except Exception:
        pass

   
    lime_list = None
    try:
        cleaned = clean_text(str(title) + ' ' + str(text))
        lime_list = lime_explain(cleaned, class_names=['REAL','FAKE'], num_features=8)
    except Exception as e:
        lime_list = [("lime_error", str(e))]

   
    shap_png = None
    try:
        cleaned_for_shap = clean_text(str(title) + ' ' + str(text))
        out_path = os.path.join('static', 'shap_explain.png')
        shap_file = shap_explain_bar([cleaned_for_shap], max_display=10, out_png_path=out_path)
        
        shap_png = '/' + shap_file.replace('\\','/')
    except Exception as e:
        shap_png = None

   
    res['display_label'] = display_label
    res['note'] = note
    res['threshold'] = THRESHOLD
    res['lime'] = lime_list
    res['shap_png'] = shap_png

    return render_template('index.html', result=res, title=title, text=text)

@app.route('/predict_url', methods=['POST'])
def predict_url():
    url = request.form.get('url', '').strip()
    if not url:
        return render_template('index.html', result={'error': 'No URL provided'}, url=url)
    fetched = fetch_article_text(url)
    if not fetched:
        return render_template('index.html', result={'error': 'Could not fetch article from the provided URL.'}, url=url)
    res = predictor.predict('', fetched)
    raw_label = res.get('label')
    confidence = res.get('confidence', 0.0)
    display_label, note = apply_threshold(raw_label, confidence, THRESHOLD)

    snippet = fetched[:800]
    try:
        save_prediction('', snippet, url, raw_label, confidence, str(res.get('explanations')))
    except Exception:
        pass

    
    lime_list = None
    try:
        cleaned = clean_text(fetched)
        lime_list = lime_explain(cleaned, class_names=['REAL','FAKE'], num_features=8)
    except Exception as e:
        lime_list = [("lime_error", str(e))]

   
    shap_png = None
    try:
        cleaned_for_shap = clean_text(fetched)
        out_path = os.path.join('static', 'shap_explain.png')
        shap_file = shap_explain_bar([cleaned_for_shap], max_display=10, out_png_path=out_path)
        shap_png = '/' + shap_file.replace('\\','/')
    except Exception:
        shap_png = None

    res['display_label'] = display_label
    res['note'] = note
    res['threshold'] = THRESHOLD
    res['lime'] = lime_list
    res['shap_png'] = shap_png

    return render_template('index.html', result=res, url=url, snippet=snippet)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json() or {}
    title = data.get('title', '')
    text = data.get('text', '')
    if not title and not text:
        return jsonify({'error': 'provide title or text'}), 400
    res = predictor.predict(title, text)
    raw_label = res.get('label'); confidence = res.get('confidence', 0.0)
    display_label, note = apply_threshold(raw_label, confidence, THRESHOLD)
    try:
        save_prediction(title, (text or '')[:800], '', raw_label, confidence, str(res.get('explanations')))
    except Exception:
        pass
    res['display_label'] = display_label
    res['note'] = note
    res['threshold'] = THRESHOLD
    return jsonify(res)

@app.route('/api/predict_url', methods=['POST'])
def api_predict_url():
    data = request.get_json() or {}
    url = data.get('url', '').strip()
    if not url:
        return jsonify({'error': 'no url provided'}), 400
    fetched = fetch_article_text(url)
    if not fetched:
        return jsonify({'error': 'could not fetch article from url'}), 400
    res = predictor.predict('', fetched)
    raw_label = res.get('label'); confidence = res.get('confidence', 0.0)
    display_label, note = apply_threshold(raw_label, confidence, THRESHOLD)
    try:
        save_prediction('', fetched[:800], url, raw_label, confidence, str(res.get('explanations')))
    except Exception:
        pass
    res['display_label'] = display_label
    res['note'] = note
    res['threshold'] = THRESHOLD
    return jsonify({'url': url, 'prediction': res, 'snippet': fetched[:500]})

@app.route('/history', methods=['GET'])
def history():
    try:
        page = int(request.args.get('page', 1))
    except Exception:
        page = 1
    per = 20
    offset = (page - 1) * per
    rows = fetch_recent(limit=per, offset=offset)
    return render_template('history.html', rows=rows, page=page)

@app.route('/history/download', methods=['GET'])
def history_download():
    rows = fetch_recent(limit=100000, offset=0)
    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(['id', 'title', 'text_snippet', 'url', 'label', 'confidence', 'created_at'])
    for r in rows:
        cw.writerow(r)
    mem = io.BytesIO()
    mem.write(si.getvalue().encode('utf-8'))
    mem.seek(0)
    return send_file(mem, mimetype='text/csv', as_attachment=True, download_name='predictions.csv')

if __name__ == "__main__":
    app.run(debug=True, port=5000)
