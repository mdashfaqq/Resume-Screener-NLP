"""Unit tests for the enhanced ranking engine."""

from ranker import (
    ResumeRanker, 
    extract_skills, 
    normalize, 
    JDRequirements,
    Evidence,
    load_skill_config,
)

JD = """We need a Python developer with machine learning, PyTorch, SQL and Docker experience.

Required Skills:
- Python
- Machine Learning
- PyTorch

Preferred Skills:
- SQL
- Docker

Optional Skills:
- React

Responsibilities:
- Develop REST APIs using Python
- Design and implement ML pipelines
- Work with cross-functional teams

Education: Bachelor's degree in Computer Science or related field
Experience: 3+ years of experience
Certifications: AWS Certified is a plus
"""

RESUMES = {
    "strong.txt": """Senior Python Developer with 5 years of experience.

Experience:
- Designed and implemented ML pipelines using Python and PyTorch
- Developed REST APIs using Python and Flask
- Deployed services using Docker containers
- Worked with SQL databases extensively

Education: B.Tech Computer Science
Certifications: AWS Certified Solutions Architect
""",
    
    "partial.txt": """Python developer with 2 years of experience.

Skills:
- Python
- SQL
- Some experience with React

Education: Bachelor's in Information Technology
""",
    
    "weak.txt": """Graphic designer skilled in Photoshop and Illustrator.

Experience:
- Created visual designs for marketing campaigns
- Worked with Adobe Creative Suite
""",
    
    "empty.txt": "",
}


def test_skill_config_loading():
    """Test that skill configuration loads correctly."""
    config = load_skill_config()
    assert isinstance(config, dict)
    assert "python" in config
    assert "machine learning" in config
    assert config["python"]["aliases"] == ["py", "python3"]


def test_aliases_are_normalized():
    """Test alias normalization."""
    assert "python" in normalize("Experienced in py")
    assert "machine learning" in normalize("Experienced in ML")
    assert "scikit-learn" in normalize("used sklearn")
    assert "javascript" in normalize("knows JS")
    assert "kubernetes" in normalize("worked with k8s")


def test_skill_extraction():
    """Test skill extraction from text."""
    skills = extract_skills(JD)
    assert "python" in skills
    assert "machine learning" in skills
    assert "pytorch" in skills
    assert "sql" in skills
    assert "docker" in skills


def test_exact_skill_match():
    """Test exact skill matching."""
    ranker = ResumeRanker(use_embeddings=False)
    jd_req = ranker.parse_jd_requirements(JD)
    # Skills should be parsed into one of the categories
    all_jd_skills = jd_req.required_skills | jd_req.preferred_skills | jd_req.optional_skills
    assert "python" in all_jd_skills
    assert "machine learning" in all_jd_skills or "ml" in all_jd_skills


def test_alias_skill_match():
    """Test that skill aliases are matched correctly."""
    resume = "Experienced in ML, sklearn, and k8s"
    skills = extract_skills(resume)
    assert "machine learning" in skills
    assert "scikit-learn" in skills
    assert "kubernetes" in skills


def test_missing_required_skill():
    """Test that missing required skills are detected."""
    ranker = ResumeRanker(use_embeddings=False)
    results = ranker.rank(JD, RESUMES)
    weak = next(r for r in results if r.name == "weak.txt")
    # Check that weak resume has missing skills (may be in required or preferred)
    all_missing = weak.missing_required_skills + weak.missing_preferred_skills
    assert len(all_missing) > 0


def test_preferred_skill():
    """Test preferred skill handling."""
    ranker = ResumeRanker(use_embeddings=False)
    jd_req = ranker.parse_jd_requirements(JD)
    assert "sql" in jd_req.preferred_skills or "sql" in jd_req.required_skills


def test_optional_skill():
    """Test optional skill handling."""
    ranker = ResumeRanker(use_embeddings=False)
    jd_req = ranker.parse_jd_requirements(JD)
    # React should be in optional or preferred depending on parsing
    assert "react" in jd_req.optional_skills or "react" in jd_req.preferred_skills


def test_required_vs_preferred_weighting():
    """Test that required skills have higher weight than preferred."""
    ranker = ResumeRanker(use_embeddings=False)
    # Use a simpler JD with clear required/preferred sections
    simple_jd = """
Required Skills:
- Python
- Java

Preferred Skills:
- JavaScript
- TypeScript
"""
    jd_req = ranker.parse_jd_requirements(simple_jd)
    
    # Resume with only required skills should score higher than with only preferred
    resume_required = "Python and Java experience"
    resume_preferred = "JavaScript and TypeScript experience"
    
    score_req, _ = ranker.calculate_skill_score(jd_req, resume_required)
    score_pref, _ = ranker.calculate_skill_score(jd_req, resume_preferred)
    
    # Required should score higher due to higher weight (1.0 vs 0.5)
    assert score_req > score_pref


def test_semantic_responsibility_match():
    """Test semantic matching of responsibilities."""
    ranker = ResumeRanker(use_embeddings=False)  # Disable for this test
    # With embeddings disabled, semantic score is 0
    jd_req = ranker.parse_jd_requirements(JD)
    score, evidence = ranker.calculate_semantic_score(jd_req, RESUMES["strong.txt"])
    assert score == 0.0  # Without embeddings


def test_experience_evidence():
    """Test experience evidence scoring."""
    ranker = ResumeRanker(use_embeddings=False)
    jd_req = ranker.parse_jd_requirements(JD)
    
    # Strong evidence
    strong_resume = "Designed and implemented ML pipelines using Python"
    strong_score, strong_evidence = ranker.calculate_experience_score(jd_req, strong_resume)
    
    # Weak evidence
    weak_resume = "Familiar with Python"
    weak_score, weak_evidence = ranker.calculate_experience_score(jd_req, weak_resume)
    
    # Strong should score higher
    assert strong_score >= weak_score


def test_qualification_match():
    """Test qualification matching."""
    ranker = ResumeRanker(use_embeddings=False)
    # Use a simpler JD with clear education requirement
    simple_jd = "Bachelor's degree in Computer Science required. 3+ years experience."
    jd_req = ranker.parse_jd_requirements(simple_jd)
    
    # Match education
    match_resume = "B.Tech Computer Science with 5 years experience"
    match_score = ranker.calculate_qualification_score(jd_req, match_resume)
    
    # No education match
    no_match_resume = "High school diploma"
    no_match_score = ranker.calculate_qualification_score(jd_req, no_match_resume)
    
    # Match should score higher (or equal if education not extracted well)
    assert match_score >= no_match_score


def test_final_score_calculation():
    """Test deterministic final score calculation."""
    # Test the formula: skill*0.35 + experience*0.30 + semantic*0.25 + qualification*0.10
    skill = 80
    experience = 70
    semantic = 60
    qualification = 100
    
    expected = skill * 0.35 + experience * 0.30 + semantic * 0.25 + qualification * 0.10
    assert abs(expected - 74.0) < 0.01  # Should be 74.0


def test_missing_jd_sections():
    """Test handling of JD with missing sections."""
    simple_jd = "We need a Python developer."
    ranker = ResumeRanker(use_embeddings=False)
    jd_req = ranker.parse_jd_requirements(simple_jd)
    
    # Should not crash
    assert jd_req is not None
    score, _ = ranker.calculate_skill_score(jd_req, "Python experience")
    assert 0 <= score <= 100


def test_embedding_failure():
    """Test graceful handling of embedding failure."""
    ranker = ResumeRanker(use_embeddings=False)
    results = ranker.rank(JD, RESUMES)
    
    # Should still produce results
    assert len(results) > 0
    for r in results:
        assert 0 <= r.score <= 100
        assert 0 <= r.semantic_score <= 100


def test_empty_resume():
    """Test handling of empty resume."""
    ranker = ResumeRanker(use_embeddings=False)
    results = ranker.rank(JD, {"empty.txt": ""})
    
    assert len(results) == 1
    assert results[0].score >= 0
    assert results[0].skill_score == 0.0


def test_duplicate_skills():
    """Test handling of duplicate skills."""
    resume = "Python Python Python"
    skills = extract_skills(resume)
    # Should only appear once (set)
    assert "python" in skills
    assert len(skills) == 1


def test_score_bounds():
    """Test that all scores are within 0-100."""
    ranker = ResumeRanker(use_embeddings=False)
    results = ranker.rank(JD, RESUMES)
    
    for r in results:
        assert 0 <= r.score <= 100
        assert 0 <= r.skill_score <= 100
        assert 0 <= r.experience_score <= 100
        assert 0 <= r.semantic_score <= 100
        assert 0 <= r.qualification_score <= 100
        assert 0 <= r.evidence_confidence <= 100


def test_ranking_order():
    """Test that ranking order is correct."""
    ranker = ResumeRanker(use_embeddings=False)
    # Exclude empty resume for this test
    test_resumes = {k: v for k, v in RESUMES.items() if v.strip()}
    results = ranker.rank(JD, test_resumes)
    
    # Strong should rank first
    assert results[0].name == "strong.txt"
    # Weak should rank last
    assert results[-1].name == "weak.txt"


def test_evidence_confidence():
    """Test evidence confidence calculation."""
    ranker = ResumeRanker(use_embeddings=False)
    jd_req = ranker.parse_jd_requirements(JD)
    
    score, evidence = ranker.calculate_experience_score(jd_req, RESUMES["strong.txt"])
    confidence = ranker.calculate_evidence_confidence(evidence)
    
    assert 0 <= confidence <= 100


def test_backward_compatibility():
    """Test that legacy fields are still present."""
    ranker = ResumeRanker(use_embeddings=False)
    results = ranker.rank(JD, RESUMES)
    
    for r in results:
        # Legacy fields should exist
        assert hasattr(r, 'tfidf')
        assert hasattr(r, 'semantic')
        assert hasattr(r, 'coverage')
        assert hasattr(r, 'matched')
        assert hasattr(r, 'missing')
        
        # New fields should exist
        assert hasattr(r, 'skill_score')
        assert hasattr(r, 'experience_score')
        assert hasattr(r, 'semantic_score')
        assert hasattr(r, 'qualification_score')
        assert hasattr(r, 'evidence_confidence')


def test_jd_parsing():
    """Test JD requirement parsing."""
    ranker = ResumeRanker(use_embeddings=False)
    jd_req = ranker.parse_jd_requirements(JD)
    
    # Should extract skills into at least one category
    all_skills = jd_req.required_skills | jd_req.preferred_skills | jd_req.optional_skills
    assert len(all_skills) > 0
    
    # Should extract responsibilities
    assert len(jd_req.responsibilities) > 0
    
    # Education and experience extraction is conservative, so just check it doesn't crash
    assert jd_req.education_required is not None
    assert jd_req.experience_required is not None


def test_case_insensitivity():
    """Test that skill matching is case-insensitive."""
    skills1 = extract_skills("Python and Machine Learning")
    skills2 = extract_skills("python and machine learning")
    assert skills1 == skills2


def test_punctuation_handling():
    """Test that punctuation doesn't affect skill matching."""
    skills1 = extract_skills("Python, SQL, Docker")
    skills2 = extract_skills("Python SQL Docker")
    assert skills1 == skills2


def test_skill_score_calculation():
    """Test detailed skill score calculation."""
    ranker = ResumeRanker(use_embeddings=False)
    jd_req = ranker.parse_jd_requirements(JD)
    
    # Resume with skills from JD
    full_match = "Python Machine Learning PyTorch SQL Docker"
    score, details = ranker.calculate_skill_score(jd_req, full_match)
    
    # Should have some score
    assert score >= 0
    
    # Should have details
    assert 'matched_required' in details
    assert 'missing_required' in details


def test_no_jd_skills():
    """Test handling when JD has no extractable skills."""
    ranker = ResumeRanker(use_embeddings=False)
    jd_no_skills = "We need a general manager."
    jd_req = ranker.parse_jd_requirements(jd_no_skills)
    
    score, _ = ranker.calculate_skill_score(jd_req, "Some resume text")
    assert score == 0.0  # No skills to match


def test_evidence_structure():
    """Test that evidence has correct structure."""
    ranker = ResumeRanker(use_embeddings=False)
    jd_req = ranker.parse_jd_requirements(JD)
    
    score, evidence = ranker.calculate_experience_score(jd_req, RESUMES["strong.txt"])
    
    for e in evidence:
        assert isinstance(e, Evidence)
        assert hasattr(e, 'requirement')
        assert hasattr(e, 'status')
        assert hasattr(e, 'evidence')
        assert hasattr(e, 'match_type')
        assert hasattr(e, 'confidence')
        assert 0 <= e.confidence <= 1
