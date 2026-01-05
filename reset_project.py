import sqlalchemy
import os
import joblib
import pandas as pd
import numpy as np
import re
from urllib.parse import quote_plus
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIG ---
db_pass = quote_plus("your@password") # <--- UPDATE PASSWORD HERE
engine = sqlalchemy.create_engine(f"mysql+pymysql://root:{db_pass}@localhost/resume_matcher_db")

# --- 1. CLEAN DATABASE ---
print("🧹 Cleaning Database...")
try:
    with engine.begin() as conn:
        conn.execute(sqlalchemy.text("DELETE FROM candidates"))
        # Reset ID counter to 1
        conn.execute(sqlalchemy.text("ALTER TABLE candidates AUTO_INCREMENT = 1"))
    print("✅ Database wiped clean. (0 Resumes)")
except Exception as e:
    print(f"Error cleaning DB: {e}")

# --- 2. RETRAIN & MOVE MODELS ---
print("\n🧠 Training AI Model...")
if not os.path.exists("models"):
    os.makedirs("models")

# Define the Class
class UniversalFeatureEngineer:
    def __init__(self):
        self.tfidf = TfidfVectorizer(stop_words='english')
    def fit(self, text_corpus): self.tfidf.fit(text_corpus)
    def get_clean_words(self, text): return set(re.sub(r'[^a-zA-Z\s]', '', text.lower()).split())
    def transform(self, r_text, j_text, r_exp, j_min):
        vecs = self.tfidf.transform([r_text, j_text])
        sim = cosine_similarity(vecs[0:1], vecs[1:2])[0][0]
        r_w, j_w = self.get_clean_words(r_text), self.get_clean_words(j_text)
        ovlap = len(r_w.intersection(j_w)) / len(j_w) if j_w else 0
        return [sim, ovlap, r_exp - j_min]

# Dummy Training Data
X_train = np.array([
    [0.95, 0.80, 2], [0.85, 0.60, 0], [0.70, 0.50, 5], [0.90, 0.90, -1],
    [0.10, 0.05, 5], [0.20, 0.10, -5], [0.90, 0.80, -10], [0.05, 0.00, 0]
])
y_train = np.array([1, 1, 1, 1, 0, 0, 0, 0])

engineer = UniversalFeatureEngineer()
engineer.fit(["python developer", "java engineer", "nurse", "sales manager"])

clf = RandomForestClassifier(n_estimators=100, max_depth=5)
clf.fit(X_train, y_train)

# SAVE TO CORRECT FOLDER
joblib.dump(clf, "models/rf_model.pkl")
joblib.dump(engineer, "models/feature_engineer.pkl")
print("✅ Models saved to 'models/' folder.")

print("\n🎉 SYSTEM RESET COMPLETE.")
print("Now run 'python ingest_resumes.py' to add your 2 files.")