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
_rankers: dict = {}


def get_ranker(weights):
    if weights not in _rankers:
        _rankers[weights] = ResumeRanker(weights=weights)
    return _rankers[weights]


def read_file(name: str, data: bytes) -> str:
    if name.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return " ".join(page.extract_text() or "" for page in reader.pages)
    return data.decode("utf-8", errors="ignore")


@app.post("/api/rank")
async def rank(
    jd: str = Form(...),
    files: list[UploadFile] = File(...),
    w_tfidf: float = Form(0.35),
    w_sem: float = Form(0.40),
    w_cov: float = Form(0.25),
):
    if not jd.strip():
        raise HTTPException(400, "Job description is empty.")
    resumes = {f.filename: read_file(f.filename, await f.read()) for f in files}
    empty = [n for n, t in resumes.items() if not t.strip()]
    resumes = {n: t for n, t in resumes.items() if t.strip()}
    if not resumes:
        raise HTTPException(400, "No text could be extracted (scanned PDFs need OCR).")

    total = (w_tfidf + w_sem + w_cov) or 1.0
    weights = tuple(round(w / total, 4) for w in (w_tfidf, w_sem, w_cov))
    ranker = get_ranker(weights)
    return {
        "results": [asdict(r) for r in ranker.rank(jd, resumes)],
        "skipped": empty,
        "semantic_enabled": ranker.embedder is not None,
    }


@app.get("/")
def home():
    return FileResponse(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "public", "index.html"))
