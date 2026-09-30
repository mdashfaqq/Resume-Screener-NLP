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
st.caption("Rank resumes against a job description using skill match, experience evidence, semantic similarity, and qualifications.")


@st.cache_resource
def get_ranker():
    return ResumeRanker(use_embeddings=True)


def read_file(upload) -> str:
    data = upload.read()
    if upload.name.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return " ".join(page.extract_text() or "" for page in reader.pages)
    return data.decode("utf-8", errors="ignore")


jd = st.text_area("Job description", height=200, placeholder="Paste the job description here...")
files = st.file_uploader("Resumes (PDF or TXT)", type=["pdf", "txt"], accept_multiple_files=True)

if st.button("Rank resumes", type="primary", disabled=not (jd and files)):
    resumes = {f.name: read_file(f) for f in files}
    empty = [n for n, t in resumes.items() if not t.strip()]
    if empty:
        st.warning(f"No text extracted from: {', '.join(empty)} (scanned PDFs need OCR).")
    ranker = get_ranker()
    results = ranker.rank(jd, {n: t for n, t in resumes.items() if t.strip()})

    df = pd.DataFrame([{
        "Rank": i + 1, 
        "Resume": r.name, 
        "Score": r.score,
        "Skill Match": f"{r.skill_score:.1f}",
        "Experience": f"{r.experience_score:.1f}",
        "Semantic": f"{r.semantic_score:.1f}",
        "Qualification": f"{r.qualification_score:.1f}",
        "Confidence": f"{r.evidence_confidence:.0f}%",
    } for i, r in enumerate(results)])
    st.subheader("Ranking")
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.bar_chart(df.set_index("Resume")["Score"])

    st.subheader("Detailed breakdown per candidate")
    for r in results:
        with st.expander(f"{r.name} - Score: {r.score} (Confidence: {r.evidence_confidence:.0f}%)"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Skill Match ({r.skill_score:.1f}/100)**")
                st.markdown(f"Required matched: {', '.join(r.matched_required_skills) or 'none'}")
                st.markdown(f"Required missing: {', '.join(r.missing_required_skills) or 'none'}")
                if r.matched_preferred_skills:
                    st.markdown(f"Preferred matched: {', '.join(r.matched_preferred_skills)}")
                if r.missing_preferred_skills:
                    st.markdown(f"Preferred missing: {', '.join(r.missing_preferred_skills)}")
            
            with col2:
                st.markdown(f"**Experience ({r.experience_score:.1f}/100)**")
                st.markdown(f"**Semantic ({r.semantic_score:.1f}/100)**")
                st.markdown(f"**Qualification ({r.qualification_score:.1f}/100)**")
            
            if r.evidence:
                st.markdown("**Strongest Evidence:**")
                for ev in r.evidence[:5]:  # Show top 5 evidence items
                    st.text(f"• {ev.requirement}: {ev.evidence[:100]}...")
                    st.caption(f"Match type: {ev.match_type} | Confidence: {ev.confidence:.2f}")
