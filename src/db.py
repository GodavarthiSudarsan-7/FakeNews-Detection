import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.getcwd(), "models", "predictions.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS predictions(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        text_snippet TEXT,
        url TEXT,
        label TEXT,
        confidence REAL,
        explanations TEXT,
        created_at TEXT
    )
    """)
    conn.commit()
    conn.close()

def save_prediction(title, text_snippet, url, label, confidence, explanations):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO predictions(title,text_snippet,url,label,confidence,explanations,created_at) VALUES(?,?,?,?,?,?,?)",
               (title, text_snippet, url, label, float(confidence), explanations, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def fetch_recent(limit=100, offset=0):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id,title,substr(text_snippet,1,300),url,label,confidence,created_at FROM predictions ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset))
    rows = cur.fetchall()
    conn.close()
    return rows
