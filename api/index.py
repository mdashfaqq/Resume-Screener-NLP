"""FastAPI backend for the web UI (deployed on Vercel).

Run locally:  uvicorn api.index:app --reload
"""

import io
import os
import sys
from dataclasses import asdict

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pypdf import PdfReader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ranker import ResumeRanker  # noqa: E402

app = FastAPI(title="Resume Screener API")
_ranker: ResumeRanker | None = None


def get_ranker():
    """Get or create the ranker instance (new scoring model)."""
    global _ranker
    if _ranker is None:
        _ranker = ResumeRanker(use_embeddings=True)
    return _ranker


def read_file(name: str, data: bytes) -> str:
    if name.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return " ".join(page.extract_text() or "" for page in reader.pages)
    return data.decode("utf-8", errors="ignore")


def serialize_result(result):
    """Serialize RankedResume to dict, handling nested Evidence objects."""
    data = asdict(result)
    # Convert Evidence objects to dicts
    data['evidence'] = [asdict(ev) for ev in result.evidence]
    return data


@app.post("/api/rank")
async def rank(
    jd: str = Form(...),
    files: list[UploadFile] = File(...),
    w_tfidf: float = Form(0.35),
    w_sem: float = Form(0.40),
    w_cov: float = Form(0.25),
):
    """Rank resumes against job description using new scoring model.
    
    Legacy weight parameters (w_tfidf, w_sem, w_cov) are accepted for compatibility
    but are not used in the new scoring model.
    """
    if not jd.strip():
        raise HTTPException(400, "Job description is empty.")
    resumes = {f.filename: read_file(f.filename, await f.read()) for f in files}
    empty = [n for n, t in resumes.items() if not t.strip()]
    resumes = {n: t for n, t in resumes.items() if t.strip()}
    if not resumes:
        raise HTTPException(400, "No text could be extracted (scanned PDFs need OCR).")

    ranker = get_ranker()
    results = ranker.rank(jd, resumes)
    
    return {
        "results": [serialize_result(r) for r in results],
        "skipped": empty,
        "semantic_enabled": ranker.use_embeddings,
        "scoring_model": "new",  # Indicate new scoring model
    }


@app.get("/")
def home():
    return FileResponse(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "public", "index.html"))
