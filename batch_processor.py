import pandas as pd
import joblib
import sqlalchemy
import numpy as np
import re
import os
from urllib.parse import quote_plus
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. CLASS DEFINITION (Must match train_model.py EXACTLY) ---
class UniversalFeatureEngineer:
    def __init__(self):
        self.tfidf = TfidfVectorizer(stop_words='english')
    
    def fit(self, text_corpus):
        self.tfidf.fit(text_corpus)

    def get_clean_words(self, text):
        clean_text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        return set(clean_text.split())

    def transform(self, resume_text, job_text, resume_exp, job_min_exp):
        # Feature 1: Cosine Similarity
        vectors = self.tfidf.transform([resume_text, job_text])
        cosine_sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        
        # Feature 2: Keyword Overlap
        r_words = self.get_clean_words(resume_text)
        j_words = self.get_clean_words(job_text)
        common_words = r_words.intersection(j_words)
        overlap_ratio = len(common_words) / len(j_words) if len(j_words) > 0 else 0.0
            
        # Feature 3: Experience Gap
        exp_gap = resume_exp - job_min_exp
        
        return [cosine_sim, overlap_ratio, exp_gap]

# --- 2. CONFIGURATION ---
db_user = "root"
db_password = "Your password"  # <--- REPLACE WITH YOUR REAL PASSWORD
db_host = "localhost"
db_name = "resume_matcher_db"

# URL Encode password to handle special chars like '@'
encoded_pass = quote_plus(db_password)
DB_CONN = f"mysql+pymysql://{db_user}:{encoded_pass}@{db_host}/{db_name}"

engine = sqlalchemy.create_engine(DB_CONN)

def run_batch_job():
    print("--- Running Batch Matcher ---")

    # 3. LOAD MODELS
    if not os.path.exists("rf_model.pkl"):
        print("Error: 'rf_model.pkl' not found. Run 'train_model.py' first.")
        return

    try:
        model = joblib.load("rf_model.pkl")
        engineer = joblib.load("feature_engineer.pkl")
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # 4. FETCH DATA (Simple version for Pandas 1.5.3)
    try:
        # No complex wrappers needed now! Just pass the engine.
        candidates = pd.read_sql("SELECT * FROM candidates", engine)
        jobs = pd.read_sql("SELECT * FROM target_jobs WHERE status='OPEN'", engine)
    except Exception as e:
        print(f"Database Error: {e}")
        return

    if candidates.empty or jobs.empty:
        print("No candidates or open jobs found. Exiting.")
        return

    print(f"Processing {len(candidates)} candidates against {len(jobs)} jobs...")

    # 5. RE-CALIBRATE ENGINEER
    # Combine text to learn new vocabulary
    all_text = pd.concat([candidates['raw_resume_text'], jobs['description']]).fillna("")
    engineer.fit(all_text)
    
    recommendations = []

    # 6. MATCHING LOOP
    for _, job in jobs.iterrows():
        for _, cand in candidates.iterrows():
            
            # Handle Nulls safely
            r_text = cand['raw_resume_text'] if cand['raw_resume_text'] else ""
            j_text = job['description'] if job['description'] else ""
            r_exp = cand['years_exp'] if cand['years_exp'] else 0
            j_min = job['min_exp'] if job['min_exp'] else 0
            
            # Create Features
            feats = engineer.transform(r_text, j_text, r_exp, j_min)
            
            # Predict
            features_reshaped = np.array(feats).reshape(1, -1)
            score = model.predict_proba(features_reshaped)[0][1]
            
            # Filter Good Matches (> 70%)
            if score > 0.70:
                recommendations.append({
                    "candidate_id": cand['id'],
                    "job_id": job['job_id'],
                    "match_probability": round(score, 2),
                    "match_reasons": f"Context: {feats[0]:.2f}, Keywords: {feats[1]:.2f}"
                })

    # 7. SAVE RESULTS
    if recommendations:
        df_rec = pd.DataFrame(recommendations)
        try:
            # Simple saving for Pandas 1.5.3
            df_rec.to_sql('recommendations', con=engine, if_exists='append', index=False)
            print(f"Success: Saved {len(df_rec)} new recommendations.")
        except Exception as e:
            print(f"Error saving to DB: {e}")
    else:
        print("No matches found above 70% threshold.")

if __name__ == "__main__":
    run_batch_job()
