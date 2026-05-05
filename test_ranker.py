"""Unit tests for the ranking engine (TF-IDF + skills only, no model download)."""

from ranker import ResumeRanker, extract_skills, normalize

JD = "We need a Python developer with machine learning, PyTorch, SQL and Docker experience."

RESUMES = {
    "strong.txt": "Built ML pipelines in Python and PyTorch, deployed with Docker, wrote SQL queries.",
    "partial.txt": "Python developer. Some SQL. Mostly web work with React.",
    "weak.txt": "Graphic designer skilled in Photoshop and Illustrator.",
}


def test_aliases_are_normalized():
    assert "machine learning" in normalize("Experienced in ML")
    assert "scikit-learn" in normalize("used sklearn")


def test_skill_extraction():
    skills = extract_skills(JD)
    assert {"python", "machine learning", "pytorch", "sql", "docker"} <= skills


def test_ranking_order():
    ranker = ResumeRanker(use_embeddings=False)
    results = ranker.rank(JD, RESUMES)
    assert [r.name for r in results] == ["strong.txt", "partial.txt", "weak.txt"]
    assert results[0].coverage == 1.0
    assert results[-1].matched == []


def test_missing_skills_reported():
    ranker = ResumeRanker(use_embeddings=False)
    partial = next(r for r in ranker.rank(JD, RESUMES) if r.name == "partial.txt")
    assert "pytorch" in partial.missing and "python" in partial.matched
