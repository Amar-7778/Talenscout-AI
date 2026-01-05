import streamlit as st
import pandas as pd
import sqlalchemy
import joblib
import re
import numpy as np
import os
from urllib.parse import quote_plus, quote
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import text
from collections import Counter

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Talentscout AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. STYLING (CSS) ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
        * { font-family: 'Inter', sans-serif; }
        
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        
        /* --- SIDEBAR --- */
        [data-testid="stSidebar"] { 
            background-color: #fff7ed; 
            border-right: 1px solid #fed7aa; 
        }
        [data-testid="stSidebar"] * { color: #431407 !important; }
        
        /* Sidebar Toggle Icon (Force White/Orange) */
        button[kind="header"] svg {
            fill: #ea580c !important;
        }

        /* --- INPUT BOX (PRESERVED) --- */
        .stTextArea textarea {
            background-color: #e5e7eb; 
            color: #000000 !important;
            border: 1px solid #d1d5db;
            font-weight: 500;
        }
        .stTextArea textarea:focus {
            border-color: #ea580c; 
            box-shadow: 0 0 0 1px #ea580c;
            background-color: #ffffff; 
        }
        /* Placeholder color override (make placeholder text black) */
        .stTextArea textarea::placeholder { color: #000000 !important; opacity: 1 !important; }
        .stTextArea textarea::-webkit-input-placeholder { color: #000000 !important; opacity: 1 !important; }
        .stTextArea textarea::-moz-placeholder { color: #000000 !important; opacity: 1 !important; }
        .stTextArea textarea:-ms-input-placeholder { color: #000000 !important; opacity: 1 !important; }
        .stTextArea textarea:-moz-placeholder { color: #000000 !important; opacity: 1 !important; }

        /* --- CARD DESIGNS --- */
        .candidate-card-pass {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #16a34a; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            margin-bottom: 15px;
        }
        
        .candidate-card-fail {
            background-color: #f9fafb;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #e5e7eb;
            margin-bottom: 10px;
            opacity: 0.7;
        }

        /* --- TAGS FOR SKILLS --- */
        .skill-tag-match {
            background-color: #dcfce7; color: #166534; 
            padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; 
            margin-right: 5px; display: inline-block; border: 1px solid #bbf7d0;
        }
        .skill-tag-miss {
            background-color: #fee2e2; color: #991b1b; 
            padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; 
            margin-right: 5px; display: inline-block; border: 1px solid #fecaca;
        }

        /* --- BUTTONS --- */
        div.stButton > button { 
            background-color: #ea580c; 
            color: white;
            border: none;
            border-radius: 6px; 
            font-weight: 500; 
        }
        div.stButton > button:hover { 
            background-color: #c2410c; 
            color: white !important;
        }
        
        /* Ghost Button */
        button[kind="secondary"] {
            background-color: transparent !important;
            border: 1px solid #fed7aa !important;
            color: #9a3412 !important;
        }
        
        /* Link Button Style (for Email) */
        .email-btn {
            text-decoration: none;
            background-color: #f3f4f6;
            color: #374151;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 0.85rem;
            border: 1px solid #d1d5db;
            display: inline-block;
        }
        .email-btn:hover {
            background-color: #e5e7eb;
            color: black;
        }
    </style>
""", unsafe_allow_html=True)

# --- 3. DATABASE ---
db_pass = quote_plus("Your Password") # <--- UPDATE PASSWORD HERE
engine = sqlalchemy.create_engine(f"mysql+pymysql://root:{db_pass}@localhost/resume_matcher_db")

# --- 4. MODELS & HELPER FUNCTIONS ---
class UniversalFeatureEngineer:
    def __init__(self):
        self.tfidf = TfidfVectorizer(stop_words='english')
    def fit(self, corpus): self.tfidf.fit(corpus)
    def get_clean_words(self, t): return set(re.sub(r'[^a-zA-Z\s]', '', t.lower()).split())
    def transform(self, r_text, j_text, r_exp, j_min):
        vecs = self.tfidf.transform([r_text, j_text])
        sim = cosine_similarity(vecs[0:1], vecs[1:2])[0][0]
        r_w, j_w = self.get_clean_words(r_text), self.get_clean_words(j_text)
        ovlap = len(r_w.intersection(j_w)) / len(j_w) if j_w else 0
        return [sim, ovlap, r_exp - j_min]

@st.cache_resource
def load_brain():
    if not os.path.exists("models/rf_model.pkl"): return None, None
    return joblib.load("models/rf_model.pkl"), joblib.load("models/feature_engineer.pkl")

model, engineer = load_brain()

# New Helper: Extract Keywords to show Match/Miss
def analyze_skills(resume_text, job_text):
    # Simple extraction of likely technical keywords (approximate)
    def extract_keywords(text):
        words = re.sub(r'[^a-zA-Z\s]', '', text.lower()).split()
        # Filter strictly for common tech terms or long words to simulate keyword extraction
        # In a real app, use a predefined list of 1000 tech skills
        return set([w for w in words if len(w) > 4]) 

    jd_skills = extract_keywords(job_text)
    res_skills = extract_keywords(resume_text)
    
    matched = list(jd_skills.intersection(res_skills))[:5] # Top 5 matches
    missing = list(jd_skills - res_skills)[:5] # Top 5 missing
    return matched, missing

# --- 5. SESSION STATE ---
if 'results' not in st.session_state: st.session_state.results = None

# --- 6. SIDEBAR ---
with st.sidebar:
    st.markdown(" Resume Ranking")
    st.caption(" Recruitment Dashboard")
    st.markdown("---")
    
    try:
        with engine.connect() as conn:
            count = pd.read_sql(text("SELECT COUNT(*) as c FROM candidates"), conn).iloc[0]['c']
    except: count = 0
    
    st.metric("Total Candidates", count)
    st.markdown("---")
    
    with st.expander(">> Advanced Options"):
        st.info("System Controls")
        if st.button("Reset Database", type="secondary"):
            try:
                with engine.begin() as conn:
                    conn.execute(text("DELETE FROM recommendations"))
                    conn.execute(text("ALTER TABLE recommendations AUTO_INCREMENT = 1"))
                    conn.execute(text("DELETE FROM candidates"))
                    conn.execute(text("ALTER TABLE candidates AUTO_INCREMENT = 1"))
                folder = "processed_resumes"
                if os.path.exists(folder):
                    for f in os.listdir(folder):
                        try: os.unlink(os.path.join(folder, f))
                        except: pass
                st.session_state.results = None
                st.rerun()
            except Exception as e: st.error(f"Error: {e}")

# --- 7. MAIN INTERFACE ---
st.markdown(" Resume Ranking System")

jd_text = st.text_area("Job Description", height=200, 
                      placeholder="Paste your job description here...",
                      key="jd_input_area",
                      label_visibility="collapsed")

col_act1, col_act2, col_act3 = st.columns([1, 1, 2])
with col_act1:
    min_exp = st.number_input("Min. Experience", 0, 20, 2)
with col_act2:
    st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
    analyze_btn = st.button("🔍 Search", type="primary", use_container_width=True)

# --- 8. ANALYSIS ---
if analyze_btn:
    if not jd_text:
        st.toast("Enter a job description!", icon="⚠️")
    elif count == 0:
        st.toast("Database is empty!", icon="⚠️")
    else:
        with st.spinner("AI is analyzing profiles..."):
            df = pd.read_sql("SELECT * FROM candidates", engine)
            all_text = pd.concat([df['raw_resume_text'], pd.Series([jd_text])]).fillna("")
            engineer.fit(all_text)

            results = []
            for _, row in df.iterrows():
                feats = engineer.transform(row['raw_resume_text'] or "", jd_text, row['years_exp'] or 0, min_exp)
                prob = model.predict_proba(np.array(feats).reshape(1, -1))[0][1]
                score = round(prob * 100, 1)
                
                status = "Recommended" if score >= 70 else "Rejected"
                
                # Get Keywords
                matched, missing = analyze_skills(row['raw_resume_text'] or "", jd_text)
                
                results.append({
                    "Name": row['name'],
                    "Score": score,
                    "Exp": row['years_exp'],
                    "Raw": row['raw_resume_text'],
                    "Filename": row.get('filename', None),
                    "Status": status,
                    "Matched": matched,
                    "Missing": missing
                })

            st.session_state.results = pd.DataFrame(results).sort_values(by="Score", ascending=False)

# --- 9. RESULTS ---
if st.session_state.results is not None:
    res = st.session_state.results
    passed = res[res['Status'] == "Recommended"]
    failed = res[res['Status'] == "Rejected"]
    
    st.markdown("---")
    
    # --- BEST FIT ---
    if not passed.empty:
        top_cand = passed.iloc[0]
        st.success(f"Best Fit: {top_cand['Name']} ({top_cand['Score']}%)")
        
        c1, c2 = st.columns([1, 3])
        with c1:
            if top_cand['Filename']:
                path = os.path.join("processed_resumes", top_cand['Filename'])
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        st.download_button("📥 Download Best Match PDF", f, file_name=f"BEST_{top_cand['Filename']}", mime="application/pdf")
    else:
        st.error("⚠️ No candidates scored above 70%.")

    # --- RECOMMENDED LIST ---
    if not passed.empty:
        st.markdown(f"✅ Recommended ({len(passed)})")
        
        csv = passed.to_csv(index=False).encode('utf-8')
        st.download_button("📄 Download Final List (CSV)", csv, "final_shortlist.csv", "text/csv")
        
        for i, cand in passed.iterrows():
            # Generate Skills HTML
            match_html = "".join([f"<span class='skill-tag-match'>✓ {m}</span>" for m in cand['Matched']])
            miss_html = "".join([f"<span class='skill-tag-miss'>⚠ {m}</span>" for m in cand['Missing']])
            
            # Email Subject/Body
            subject = quote(f"Interview Invitation - {cand['Name']}")
            body = quote(f"Hi {cand['Name']},\n\nWe reviewed your resume and would like to invite you for an interview.\n\nBest,\nHR Team")
            gmail_link = f"mailto:?subject={subject}&body={body}"

            st.markdown(f"""
            <div class="candidate-card-pass">
                <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                    <div>
                        <h3 style="margin:0; color:#000;">{cand['Name']}</h3>
                        <p style="margin:0 0 8px 0; color:#666;">{cand['Exp']} Years Experience</p>
                        <div>{match_html} {miss_html}</div>
                    </div>
                    <div style="text-align:right;">
                        <div style="background:#dcfce7; color:#166534; padding:5px 10px; border-radius:20px; font-weight:bold; display:inline-block; margin-bottom:5px;">
                            {cand['Score']}%
                        </div>
                        <br>
                        <a href="{gmail_link}" target="_blank" class="gmail-btn">g-mail Candidate</a>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            c1, c2, _ = st.columns([1, 1, 3])
            with c1:
                if cand['Filename']:
                    path = os.path.join("processed_resumes", cand['Filename'])
                    if os.path.exists(path):
                        with open(path, "rb") as f:
                            st.download_button("Download PDF", f, file_name=cand['Filename'], key=f"dl_pass_{i}")
            with c2:
                with st.popover("View Text"):
                    st.text_area("Resume", cand['Raw'], height=300)

    # --- REJECTED LIST ---
    if not failed.empty:
        st.markdown("---")
        with st.expander(f"❌ Rejected Candidates (Below 70%) - {len(failed)} found"):
            for i, cand in failed.iterrows():
                # Rejection Email
                subject = quote(f"Application Update - {cand['Name']}")
                body = quote(f"Hi {cand['Name']},\n\nThank you for applying. Unfortunately, we are not moving forward at this time.\n\nBest,\nHR Team")
                mailto_link = f"mailto:?subject={subject}&body={body}"
                
                st.markdown(f"""
                <div class="candidate-card-fail">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <span style="color:#555; font-weight:500;">{cand['Name']}</span>
                            <span style="margin-left:10px; font-size:0.8rem; color:#991b1b;">Missing: {', '.join(cand['Missing'][:3])}</span>
                        </div>
                        <div>
                            <a href="{mailto_link}" target="_blank" style="text-decoration:none; color:#6b7280; margin-right:15px; font-size:0.9rem;">📧 Send Rejection</a>
                            <span style="color:#991b1b; font-weight:bold;">{cand['Score']}%</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

elif count == 0:
    st.info("👋 Welcome! Please upload resumes to start.")
