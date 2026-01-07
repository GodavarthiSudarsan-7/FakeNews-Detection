# 📰 Fake News Detection System

An AI-powered web application that detects whether a given news article or URL is **Fake or Real** using **Natural Language Processing (NLP)** and **Machine Learning**.

This project demonstrates a complete ML pipeline — from data preprocessing and model training to prediction, explainability, and a user-friendly web interface.

---

## 🚀 Features

- 🔍 Detects **Fake vs Real News**
- 🌐 Supports **text input & news URL input**
- 🧠 Uses **TF-IDF + Machine Learning model**
- 📊 Stores prediction history
- 🧩 Modular & clean project structure
- 🖥️ Simple web UI (HTML + CSS)
- ♻️ Reproducible training pipeline

---

## 🛠️ Tech Stack

| Layer | Technology |
|-----|-----------|
| Language | Python |
| ML / NLP | Scikit-learn, TF-IDF |
| Backend | Flask |
| Frontend | HTML, CSS |
| Database | SQLite |
| Model Storage | Joblib |
| Version Control | Git & GitHub |

---

## 📂 Project Structure






---

## 🧠 Machine Learning Approach

1. **Text Preprocessing**
   - Lowercasing
   - Tokenization
   - Stopword removal

2. **Feature Extraction**
   - TF-IDF Vectorization

3. **Model Training**
   - Supervised ML classifier
   - Trained using labeled news data

4. **Prediction**
   - Classifies input as **Fake** or **Real**
   - Stores results for analysis

---

## ▶️ How to Run the Project Locally

### 1️⃣ Clone the repository
```bash

---
git clone https://github.com/your-username/FakeNews-Detection.git
cd FakeNews-Detection


python -m venv venv
venv\Scripts\activate   # Windows

pip install -r requirements.txt


python src/train.py
python src/app.py


runs in http://127.0.0.1:5000


---