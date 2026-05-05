# AI Resume Screener

Ranks a batch of resumes against a job description and explains each score. Built to show how classic NLP (TF-IDF) and modern sentence embeddings complement each other in a real screening task.

## How scoring works

Each resume gets three sub-scores, blended into a final 0–100 score:

| Signal | What it captures | Default weight |
|--------|------------------|----------------|
| TF-IDF cosine similarity (unigrams + bigrams) | Exact terminology overlap with the JD | 0.35 |
| Sentence-embedding similarity (`all-MiniLM-L6-v2`) | Semantic match, e.g. "built neural nets" vs "deep learning" | 0.40 |
| Skill coverage | Share of the JD's required skills found in the resume | 0.25 |

Skills are matched against a curated lexicon with alias normalization (`ML` → `machine learning`, `sklearn` → `scikit-learn`, `k8s` → `kubernetes`), so different spellings count as the same skill.

If the embedding model can't be loaded, the ranker falls back to TF-IDF plus skill coverage and redistributes the weights.

## Features

- Upload PDF or TXT resumes in bulk
- Adjustable scoring weights in the sidebar
- Ranked table, score chart and a per-candidate skill-gap breakdown (matched vs missing)
- Unit tests covering normalization, skill extraction and ranking order

## Run it

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Run the tests:

```bash
pytest -q
```

## Project structure

```
ranker.py        # scoring engine: normalization, skills, TF-IDF, embeddings
app.py           # Streamlit interface
test_ranker.py   # pytest suite
```

## Limitations

- The skill lexicon is hand-curated. A learned skill extractor (NER) would generalize better.
- Automated screening can inherit bias from job descriptions, so it should support human review, not replace it.

## Tech stack

Python, scikit-learn, sentence-transformers, Streamlit, pandas, pytest
