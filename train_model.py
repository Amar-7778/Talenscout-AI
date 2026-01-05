import joblib
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier

# --- Feature Engineering Class ---
class UniversalFeatureEngineer:
    def __init__(self):
        self.tfidf = TfidfVectorizer(stop_words='english')
    
    def fit(self, text_corpus):
        self.tfidf.fit(text_corpus)

    def get_clean_words(self, text):
        clean = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        return set(clean.split())

    def transform(self, resume_text, job_text, resume_exp, job_min_exp):
        # Feature 1: Cosine Similarity
        vecs = self.tfidf.transform([resume_text, job_text])
        sim = cosine_similarity(vecs[0:1], vecs[1:2])[0][0]
        
        # Feature 2: Keyword Overlap
        r_words = self.get_clean_words(resume_text)
        j_words = self.get_clean_words(job_text)
        common = r_words.intersection(j_words)
        overlap = len(common) / len(j_words) if len(j_words) > 0 else 0.0
            
        # Feature 3: Experience Gap
        gap = resume_exp - job_min_exp
        
        return [sim, overlap, gap]

if __name__ == "__main__":
    print("Training Model...")
    
    # Synthetic Data: [Cosine, Overlap, ExpGap]
    X_train = np.array([
        [0.95, 0.80,  2], [0.85, 0.60,  0], [0.70, 0.50,  5], [0.90, 0.90, -1], # Matches
        [0.10, 0.05,  5], [0.20, 0.10, -5], [0.90, 0.80, -10], [0.05, 0.00,  0]  # Mismatches
    ])
    y_train = np.array([1, 1, 1, 1, 0, 0, 0, 0])

    engineer = UniversalFeatureEngineer()
    engineer.fit(["manager sales", "python developer", "nurse care"]) # Dummy init

    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)

    joblib.dump(clf, "rf_model.pkl")
    joblib.dump(engineer, "feature_engineer.pkl")
    print("Files saved: rf_model.pkl, feature_engineer.pkl")