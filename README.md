# AI Resume Screener - ATS-Style Analyzer

ATS-style resume analyzer with explainable, evidence-based scoring. Distinguishes skill presence from actual experience using section-aware parsing and weighted evidence extraction.

## How ATS Scoring Works

Each resume gets six sub-scores, blended into a final 0–100 score:

| Signal | What it captures | Default weight |
|--------|------------------|----------------|
| Required Skill Match | Weighted coverage of required skills with section evidence | 0.30 |
| Experience & Responsibility | Evidence strength of relevant experience and responsibility matching | 0.25 |
| Semantic Job Fit | Semantic similarity of JD responsibilities to resume sections | 0.20 |
| Qualification Match | Education, certifications, and explicit experience requirements | 0.10 |
| Preferred Skill Match | Weighted coverage of preferred skills with section evidence | 0.05 |
| ATS Resume Quality | Resume parsing quality and ATS readiness | 0.10 |

### Key Innovation: Section-Aware Evidence Extraction

The system distinguishes between **skill presence** and **actual experience** by analyzing where skills appear in the resume:

**Section Evidence Weights:**
- Experience/Projects: 1.00 (strong evidence)
- Internship: 0.95 (strong evidence)
- Technical Skills: 0.65 (moderate evidence)
- Summary: 0.50 (weak evidence)
- Education: 0.30 (weak evidence)
- Header/Contact: 0.00 (no evidence - prevents false positives)

**Example:**
- "GitHub" in header/contact → NOT counted as Git experience
- "Git" in Technical Skills → Skill detected, but weak evidence
- "Implemented Git-based CI workflow" in Experience → Strong evidence

### Skill Matching

Skills are extracted from job descriptions and classified as:
- **Required** (weight 1.0): Must-have skills
- **Preferred** (weight 0.5): Nice-to-have skills  
- **Optional** (weight 0.25): Bonus skills

The skill score is calculated as:
```
weighted_skill_score = (sum of matched skill weights × evidence strength) / (sum of all JD skill weights) × 100
```

Skills are matched against an extensible configuration file (`skills_config.json`) with alias normalization (e.g., `ML` → `machine learning`, `sklearn` → `scikit-learn`, `k8s` → `kubernetes`, `JS` → `JavaScript`).

### Experience Evidence

Experience scoring distinguishes between:
- **Weak evidence**: "Familiar with Docker"
- **Moderate evidence**: "Used Docker to containerize services"
- **Strong evidence**: "Designed and deployed Docker-based production services"

Action verbs (developed, implemented, deployed, architected, etc.) strengthen evidence when detected with skills.

### Semantic Matching

Compares JD responsibilities against relevant resume sections (experience, projects, internship) using sentence embeddings. For each responsibility, finds the most relevant resume evidence and calculates similarity.

### Qualification Matching

Checks explicit requirements:
- Education requirements (Bachelor's, Master's, PhD)
- Experience requirements (years of experience)
- Certifications

### ATS Quality Score

Evaluates resume parsing quality:
- Section detection (experience, education, skills)
- Contact information presence
- Formatting issues (long paragraphs, broken characters)
- Duplicate sections

### Evidence & Confidence

Each match includes:
- Evidence text from the resume
- Match type (exact, alias, semantic, partial)
- Confidence score (0-1)
- Section where evidence was found
- Evidence strength (strong, moderate, weak)

Overall evidence confidence reflects how clearly the resume provides evidence for the matching decisions.

## Features

- Upload PDF or TXT resumes in bulk
- Section-aware resume parsing with normalization
- Weighted skill matching with configurable skill lexicon
- Experience evidence scoring with action verb detection
- Semantic matching of responsibilities to resume sections
- Qualification matching for education, certifications, experience
- ATS quality scoring for resume parsing
- Explainable results with evidence tracking and confidence scores
- Per-candidate skill gap breakdown (required vs preferred vs partial)
- Strongest evidence display with section context
- Parsing warnings for ATS issues
- Unit tests covering all scoring components (35 tests)

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
ranker.py           # ATS scoring engine: section detection, evidence extraction, scoring
skills_config.json  # extensible skill and alias configuration
api/index.py        # FastAPI backend for the web UI
public/             # modern single-page web UI
app.py              # Streamlit interface
test_ranker.py      # comprehensive pytest suite (35 tests)
```

## Limitations

- The skill lexicon is configuration-based. A learned skill extractor (NER) would generalize better.
- JD section detection is conservative and may not catch all formatting styles.
- Experience evidence scoring uses keyword-based strength indicators.
- Automated screening can inherit bias from job descriptions, so it should support human review, not replace it.
- Scanned PDFs require OCR (not implemented).

## Tech stack

Python, scikit-learn, sentence-transformers, Streamlit, FastAPI, pandas, pytest
