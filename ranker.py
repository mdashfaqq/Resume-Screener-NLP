"""Resume ranking engine.

Scores each resume against a job description with three signals:
  1. TF-IDF cosine similarity  - lexical overlap, rewards exact terminology
  2. Embedding similarity      - semantic match (e.g. "ML" vs "machine learning")
  3. Skill coverage            - fraction of required skills found in the resume
The final score is a weighted blend; weights are configurable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SKILL_LEXICON = {
    "python", "java", "c++", "javascript", "typescript", "sql", "r", "go",
    "react", "node", "django", "flask", "fastapi", "spring",
    "machine learning", "deep learning", "nlp", "computer vision",
    "pytorch", "tensorflow", "keras", "scikit-learn", "pandas", "numpy",
    "docker", "kubernetes", "aws", "gcp", "azure", "git", "linux",
    "mongodb", "postgresql", "mysql", "redis", "rest api", "graphql",
    "data analysis", "statistics", "tableau", "power bi", "spark",
}

ALIASES = {
    "ml": "machine learning", "dl": "deep learning", "js": "javascript",
    "sklearn": "scikit-learn", "postgres": "postgresql", "k8s": "kubernetes",
    "nodejs": "node", "node.js": "node", "reactjs": "react", "react.js": "react",
}


def normalize(text: str) -> str:
    text = text.lower()
    for alias, canonical in ALIASES.items():
        text = re.sub(rf"(?<![\w.]){re.escape(alias)}(?![\w])", canonical, text)
    return re.sub(r"\s+", " ", text)


def extract_skills(text: str) -> set[str]:
    text = normalize(text)
    found = set()
    for skill in SKILL_LEXICON:
        if re.search(rf"(?<![\w+]){re.escape(skill)}(?![\w+])", text):
            found.add(skill)
    return found


@dataclass
class RankedResume:
    name: str
    score: float
    tfidf: float
    semantic: float
    coverage: float
    matched: list[str]
    missing: list[str]


class ResumeRanker:
    def __init__(self, weights=(0.35, 0.40, 0.25), use_embeddings: bool = True):
        self.w_tfidf, self.w_sem, self.w_cov = weights
        self.embedder = None
        if use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                # Fall back to TF-IDF + skills only; redistribute the semantic weight.
                self.w_tfidf += self.w_sem / 2
                self.w_cov += self.w_sem / 2
                self.w_sem = 0.0

    def rank(self, job_description: str, resumes: dict[str, str]) -> list[RankedResume]:
        names = list(resumes)
        docs = [normalize(job_description)] + [normalize(resumes[n]) for n in names]

        tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        matrix = tfidf.fit_transform(docs)
        tfidf_scores = cosine_similarity(matrix[0], matrix[1:]).ravel()

        if self.embedder is not None:
            vecs = self.embedder.encode(docs, normalize_embeddings=True)
            sem_scores = (vecs[1:] @ vecs[0]).clip(0, 1)
        else:
            sem_scores = np.zeros(len(names))

        required = extract_skills(job_description)
        results = []
        for i, name in enumerate(names):
            have = extract_skills(resumes[name])
            matched = sorted(required & have)
            missing = sorted(required - have)
            coverage = len(matched) / len(required) if required else 0.0
            score = (
                self.w_tfidf * tfidf_scores[i]
                + self.w_sem * sem_scores[i]
                + self.w_cov * coverage
            )
            results.append(RankedResume(
                name=name, score=round(float(score) * 100, 1),
                tfidf=round(float(tfidf_scores[i]), 3),
                semantic=round(float(sem_scores[i]), 3),
                coverage=round(coverage, 3), matched=matched, missing=missing,
            ))
        return sorted(results, key=lambda r: r.score, reverse=True)
