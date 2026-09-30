# AI Resume Screener

Ranks a batch of resumes against a job description with explainable, job-aware scoring. Uses weighted skill matching, experience evidence, semantic similarity, and qualification matching.

## How scoring works

Each resume gets four sub-scores, blended into a final 0–100 score:

| Signal | What it captures | Default weight |
|--------|------------------|----------------|
| Skill Match | Weighted coverage of required/preferred/optional skills | 0.35 |
| Experience Match | Evidence strength of relevant experience (mention vs project vs deployment) | 0.30 |
| Semantic Match | Semantic similarity of JD responsibilities to resume sections | 0.25 |
| Qualification Match | Education, certifications, and explicit requirements | 0.10 |

### Skill Matching

Skills are extracted from job descriptions and classified as:
- **Required** (weight 1.0): Must-have skills
- **Preferred** (weight 0.5): Nice-to-have skills  
- **Optional** (weight 0.25): Bonus skills

The skill score is calculated as:
```
weighted_skill_score = (sum of matched skill weights) / (sum of all JD skill weights) × 100
```

Skills are matched against an extensible configuration file (`skills_config.json`) with alias normalization (e.g., `ML` → `machine learning`, `sklearn` → `scikit-learn`, `k8s` → `kubernetes`, `JS` → `JavaScript`).

### Experience Evidence

Experience scoring distinguishes between:
- **Weak evidence**: "Familiar with Docker"
- **Medium evidence**: "Used Docker to containerize services"
- **Strong evidence**: "Designed and deployed Docker-based production services"

The score measures relevance to the JD, not simply the number of technologies mentioned.

### Semantic Matching

Compares JD responsibilities against relevant resume sections (experience, projects, responsibilities) using sentence embeddings. For each responsibility, finds the most relevant resume evidence and calculates similarity.

### Qualification Matching

Checks explicit requirements:
- Education requirements (Bachelor's, Master's, PhD)
- Experience requirements (years of experience)
- Certifications

### Evidence & Confidence

Each match includes:
- Evidence text from the resume
- Match type (direct, semantic, partial)
- Confidence score (0-1)

Overall evidence confidence reflects how clearly the resume provides evidence for the matching decision.

## Features

- Upload PDF or TXT resumes in bulk
- Job-aware requirement extraction (required/preferred/optional skills)
- Weighted skill matching with configurable skill lexicon
- Experience evidence scoring based on strength indicators
- Semantic matching of responsibilities to resume sections
- Qualification matching for education, certifications, experience
- Explainable results with evidence tracking and confidence scores
- Per-candidate skill gap breakdown (required vs preferred)
- Strongest evidence display for each candidate
- Unit tests covering all scoring components

## Run it

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements-local.txt
uvicorn api.index:app --reload   # modern web UI at http://localhost:8000
streamlit run app.py             # or the Streamlit UI
```

## Deploy

### Railway (Recommended for Full NLP Embeddings)

1. Connect your GitHub repository (`Resume-Screener-NLP`) to [Railway](https://railway.app).
2. Railway will automatically detect Python, install dependencies from `requirements.txt`, and use the `Procfile` / `railway.json` start command:
   ```bash
   uvicorn api.index:app --host 0.0.0.0 --port $PORT
   ```
3. Your app will be live with full `sentence-transformers` embedding support!

### Vercel

```bash
npx vercel --prod
```

The web UI (`public/index.html`) is served statically and `/api/rank` runs as a Python serverless function. On Vercel, if model memory limit is reached, it automatically falls back to TF-IDF + skills.

Run the tests:

```bash
pytest -q
```

## Project structure

```
ranker.py           # scoring engine: JD parsing, skill matching, experience, semantic, qualification
skills_config.json  # extensible skill and alias configuration
api/index.py        # FastAPI backend for the web UI
public/             # modern single-page web UI
app.py              # Streamlit interface
test_ranker.py      # comprehensive pytest suite
```

## Limitations

- The skill lexicon is configuration-based. A learned skill extractor (NER) would generalize better.
- JD section detection is conservative and may not catch all formatting styles.
- Experience evidence scoring uses keyword-based strength indicators.
- Automated screening can inherit bias from job descriptions, so it should support human review, not replace it.

## Tech stack

Python, scikit-learn, sentence-transformers, Streamlit, FastAPI, pandas, pytest
