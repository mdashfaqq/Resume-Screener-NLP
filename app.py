"""Streamlit UI for the resume screener.

Run:  streamlit run app.py
"""

import io

import pandas as pd
import streamlit as st
from pypdf import PdfReader

from ranker import ResumeRanker

st.set_page_config(page_title="Resume Screener", layout="wide")
st.title("AI Resume Screener")
st.caption("Rank resumes against a job description using TF-IDF, sentence embeddings and skill coverage.")


@st.cache_resource
def get_ranker(w_tfidf, w_sem, w_cov):
    return ResumeRanker(weights=(w_tfidf, w_sem, w_cov))


def read_file(upload) -> str:
    data = upload.read()
    if upload.name.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return " ".join(page.extract_text() or "" for page in reader.pages)
    return data.decode("utf-8", errors="ignore")


with st.sidebar:
    st.header("Scoring weights")
    w_tfidf = st.slider("Keyword match (TF-IDF)", 0.0, 1.0, 0.35, 0.05)
    w_sem = st.slider("Semantic match (embeddings)", 0.0, 1.0, 0.40, 0.05)
    w_cov = st.slider("Skill coverage", 0.0, 1.0, 0.25, 0.05)
    total = w_tfidf + w_sem + w_cov or 1.0
    weights = (w_tfidf / total, w_sem / total, w_cov / total)
    st.caption("Weights are normalized to sum to 1.")

jd = st.text_area("Job description", height=200, placeholder="Paste the job description here...")
files = st.file_uploader("Resumes (PDF or TXT)", type=["pdf", "txt"], accept_multiple_files=True)

if st.button("Rank resumes", type="primary", disabled=not (jd and files)):
    resumes = {f.name: read_file(f) for f in files}
    empty = [n for n, t in resumes.items() if not t.strip()]
    if empty:
        st.warning(f"No text extracted from: {', '.join(empty)} (scanned PDFs need OCR).")
    ranker = get_ranker(*weights)
    results = ranker.rank(jd, {n: t for n, t in resumes.items() if t.strip()})

    df = pd.DataFrame([{
        "Rank": i + 1, "Resume": r.name, "Score": r.score,
        "Keyword": r.tfidf, "Semantic": r.semantic, "Skill coverage": f"{r.coverage:.0%}",
    } for i, r in enumerate(results)])
    st.subheader("Ranking")
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.bar_chart(df.set_index("Resume")["Score"])

    st.subheader("Skill gap per candidate")
    for r in results:
        with st.expander(f"{r.name} - {r.score}"):
            st.markdown("**Matched:** " + (", ".join(r.matched) or "none"))
            st.markdown("**Missing:** " + (", ".join(r.missing) or "none"))
