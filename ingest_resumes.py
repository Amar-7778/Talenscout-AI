import os
import pdfplumber
import sqlalchemy
from urllib.parse import quote_plus
import re
import pandas as pd

# --- CONFIGURATION ---
UPLOAD_FOLDER = "resume_uploads"
PROCESSED_FOLDER = "processed_resumes"

# DB Connection
db_pass = quote_plus("Your password") # <--- UPDATE PASSWORD
# ensure we use pymysql
engine = sqlalchemy.create_engine(f"mysql+pymysql://root:{db_pass}@localhost/resume_matcher_db")

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted: text += extracted + "\n"
    except: pass
    return text

def parse_experience(text):
    match = re.search(r'(\d+(\.\d+)?)\+?\s+years?', text.lower())
    if match: return int(float(match.group(1)))
    return 0

def ingest_files():
    print(f"--- Scanning '{UPLOAD_FOLDER}' ---")
    if not os.path.exists(PROCESSED_FOLDER): os.makedirs(PROCESSED_FOLDER)

    files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.pdf')]
    if not files:
        print("No new PDFs found.")
        return

    candidates_data = []
    
    for filename in files:
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        print(f"Processing: {filename}...")
        
        try:
            raw_text = extract_text_from_pdf(filepath)
            if not raw_text.strip(): continue

            candidates_data.append({
                "name": filename.replace(".pdf", "").replace("_", " "),
                "email": "pending@example.com",
                "years_exp": parse_experience(raw_text),
                "raw_resume_text": raw_text
            })
            
            # Move file
            dest = os.path.join(PROCESSED_FOLDER, filename)
            if os.path.exists(dest): os.remove(dest)
            os.rename(filepath, dest)

        except Exception as e:
            print(f"Error reading {filename}: {e}")

    # --- THE VERSION FIX IS HERE ---
    if candidates_data:
        df = pd.DataFrame(candidates_data)
        try:
            # PASS THE ENGINE DIRECTLY. Do not use engine.connect()
            df.to_sql('candidates', con=engine, if_exists='append', index=False)
            print(f"Successfully uploaded {len(df)} resumes.")
        except Exception as db_err:
            print(f"Database Error: {db_err}")

if __name__ == "__main__":
    ingest_files()
