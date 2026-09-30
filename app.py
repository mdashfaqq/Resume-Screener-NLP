"""Streamlit UI for the resume screener.

Run:  streamlit run app.py
"""

import io

import pandas as pd
import streamlit as st
from pypdf import PdfReader

from ranker import ResumeRanker

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        background-color: #0f172a;
    }
    .stApp {
        background-color: #0f172a;
    }
    h1 {
        color: #f8fafc;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .stTextArea > div > div > textarea {
        background-color: #1e293b;
        color: #f8fafc;
        border: 1px solid #334155;
        border-radius: 0.5rem;
    }
    .stTextArea > div > div > textarea:focus {
        border-color: #6366f1;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);
    }
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(99, 102, 241, 0.3);
    }
    .stDataFrame {
        background-color: #1e293b;
        border-radius: 0.5rem;
        overflow: hidden;
    }
    .stDataFrame [data-testid="stDataFrame"] {
        color: #f8fafc;
    }
</style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="Resume Screener",
    layout="wide",
    page_icon="📄"
)

# Header
st.markdown("""
# 🎯 AI Resume Screener

**ATS-style resume analyzer with explainable, evidence-based scoring**
""")

st.caption("Distinguishes skill presence from actual experience using section-aware parsing and weighted evidence extraction.")


@st.cache_resource
def get_ranker():
    return ResumeRanker(use_embeddings=True)


def read_file(upload) -> str:
    data = upload.read()
    if upload.name.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return " ".join(page.extract_text() or "" for page in reader.pages)
    return data.decode("utf-8", errors="ignore")


# Main layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📝 Job Description")
    jd = st.text_area(
        "",
        height=200,
        placeholder="Paste the job description here...",
        label="Job Description"
    )
    
    st.markdown("### 📁 Upload Resumes")
    files = st.file_uploader(
        "Drag and drop or click to upload",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label="Resumes (PDF or TXT)"
    )

with col2:
    st.markdown("### 📊 ATS Scoring Model")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e293b 0%, #334155 100%); 
                border: 1px solid #475569; 
                border-radius: 0.75rem; 
                padding: 1.5rem;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span style="color: #94a3b8;">Required Skills</span>
            <span style="color: #6366f1; font-weight: 600;">30%</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span style="color: #94a3b8;">Experience & Responsibility</span>
            <span style="color: #8b5cf6; font-weight: 600;">25%</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span style="color: #94a3b8;">Semantic Job Fit</span>
            <span style="color: #a855f7; font-weight: 600;">20%</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span style="color: #94a3b8;">Qualifications</span>
            <span style="color: #d946ef; font-weight: 600;">10%</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span style="color: #94a3b8;">Preferred Skills</span>
            <span style="color: #ec4899; font-weight: 600;">5%</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span style="color: #94a3b8;">ATS Quality</span>
            <span style="color: #f43f5e; font-weight: 600;">10%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("💡 Section-aware evidence extraction prevents false positives (e.g., GitHub in header ≠ Git experience).")
    
    analyze = st.button("🔍 Analyze Resumes", type="primary", disabled=not (jd and files), use_container_width=True)

if analyze and jd and files:
    resumes = {f.name: read_file(f) for f in files}
    empty = [n for n, t in resumes.items() if not t.strip()]
    if empty:
        st.warning(f"⚠️ No text extracted from: {', '.join(empty)} (scanned PDFs need OCR).")
    
    ranker = get_ranker()
    results = ranker.rank(jd, {n: t for n, t in resumes.items() if t.strip()})
    
    st.markdown("---")
    st.markdown("## 📈 Ranking Results")
    
    # Summary metrics
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("Total Candidates", len(results))
    with col_b:
        avg_score = sum(r.score for r in results) / len(results) if results else 0
        st.metric("Average Score", f"{avg_score:.1f}")
    with col_c:
        high_confidence = sum(1 for r in results if r.evidence_confidence >= 70)
        st.metric("High Confidence", high_confidence)
    with col_d:
        avg_ats = sum(r.ats_quality_score for r in results) / len(results) if results else 0
        st.metric("Avg ATS Quality", f"{avg_ats:.1f}")
    
    # Ranking table
    df = pd.DataFrame([{
        "Rank": i + 1, 
        "Candidate": r.name, 
        "Score": r.score,
        "Required Skills": f"{r.required_skill_score:.1f}",
        "Experience": f"{r.experience_score:.1f}",
        "Semantic": f"{r.semantic_score:.1f}",
        "Qualifications": f"{r.qualification_score:.1f}",
        "Preferred": f"{r.preferred_skill_score:.1f}",
        "ATS Quality": f"{r.ats_quality_score:.1f}",
        "Confidence": f"{r.evidence_confidence:.0f}%",
    } for i, r in enumerate(results)])
    
    st.dataframe(
        df, 
        hide_index=True, 
        use_container_width=True,
        column_config={
            "Score": st.column_config.NumberColumn("Score", format="%.1f"),
            "Required Skills": st.column_config.NumberColumn("Required Skills", format="%.1f"),
            "Experience": st.column_config.NumberColumn("Experience", format="%.1f"),
            "Semantic": st.column_config.NumberColumn("Semantic", format="%.1f"),
            "Qualifications": st.column_config.NumberColumn("Qualifications", format="%.1f"),
            "Preferred": st.column_config.NumberColumn("Preferred", format="%.1f"),
            "ATS Quality": st.column_config.NumberColumn("ATS Quality", format="%.1f"),
        }
    )
    
    # Score chart
    st.markdown("### 📊 Score Distribution")
    chart_df = df.set_index("Candidate")[["Score"]]
    st.bar_chart(chart_df, color="#6366f1")
    
    # Detailed breakdown per candidate
    st.markdown("---")
    st.markdown("## 🔍 Detailed Analysis")
    
    for r in results:
        with st.expander(f"### {r.name} - Score: {r.score} (Confidence: {r.evidence_confidence:.0f}%)"):
            # Score breakdown
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.markdown("#### 📊 Score Breakdown")
                
                # Required Skills
                req_color = "🟢" if r.required_skill_score >= 70 else "🟡" if r.required_skill_score >= 40 else "🔴"
                st.markdown(f"{req_color} **Required Skills** ({r.required_skill_score:.1f}/100)")
                if r.matched_required_skills:
                    st.markdown(f"✅ Matched: {', '.join(r.matched_required_skills)}")
                if r.missing_required_skills:
                    st.markdown(f"❌ Missing: {', '.join(r.missing_required_skills)}")
                if r.partial_required_skills:
                    st.markdown(f"⚠️ Partial: {', '.join(r.partial_required_skills)}")
                
                # Preferred Skills
                if r.matched_preferred_skills or r.missing_preferred_skills:
                    pref_color = "🟢" if r.preferred_skill_score >= 70 else "🟡" if r.preferred_skill_score >= 40 else "🔴"
                    st.markdown(f"{pref_color} **Preferred Skills** ({r.preferred_skill_score:.1f}/100)")
                    if r.matched_preferred_skills:
                        st.markdown(f"✅ Matched: {', '.join(r.matched_preferred_skills)}")
                    if r.missing_preferred_skills:
                        st.markdown(f"❌ Missing: {', '.join(r.missing_preferred_skills)}")
            
            with col_right:
                st.markdown("#### 🎯 Other Scores")
                
                exp_color = "🟢" if r.experience_score >= 70 else "🟡" if r.experience_score >= 40 else "🔴"
                st.markdown(f"{exp_color} **Experience** ({r.experience_score:.1f}/100)")
                
                sem_color = "🟢" if r.semantic_score >= 70 else "🟡" if r.semantic_score >= 40 else "🔴"
                st.markdown(f"{sem_color} **Semantic Fit** ({r.semantic_score:.1f}/100)")
                
                qual_color = "🟢" if r.qualification_score >= 70 else "🟡" if r.qualification_score >= 40 else "🔴"
                st.markdown(f"{qual_color} **Qualifications** ({r.qualification_score:.1f}/100)")
                
                ats_color = "🟢" if r.ats_quality_score >= 70 else "🟡" if r.ats_quality_score >= 40 else "🔴"
                st.markdown(f"{ats_color} **ATS Quality** ({r.ats_quality_score:.1f}/100)")
            
            # Parsing warnings
            if r.parsing_warnings:
                st.markdown("#### ⚠️ Parsing Warnings")
                for warning in r.parsing_warnings:
                    st.warning(warning)
            
            # Strongest evidence
            if r.strongest_evidence:
                st.markdown("#### 💪 Strongest Evidence")
                for ev in r.strongest_evidence:
                    strength_emoji = "🔥" if ev.evidence_strength == "strong" else "👍" if ev.evidence_strength == "moderate" else "📝"
                    st.markdown(f"{strength_emoji} **{ev.requirement}**")
                    st.caption(f"Section: {ev.section} | Strength: {ev.evidence_strength} | Confidence: {ev.confidence:.2f}")
                    st.info(ev.evidence[:200] + "..." if len(ev.evidence) > 200 else ev.evidence)
