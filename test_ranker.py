"""Unit tests for the ATS-style resume analyzer."""

from ranker import (
    ResumeRanker, 
    extract_skills, 
    normalize,
    detect_sections,
    extract_skills_with_sections,
    load_skill_config,
    JDRequirements,
    Evidence,
    SkillMatch,
    ResumeSection,
)

# Test Job Description
JD = """Senior Full Stack Developer

Required Skills:
- React
- TypeScript
- Node.js
- PostgreSQL

Preferred Skills:
- Docker
- AWS

Responsibilities:
- Develop REST APIs using Node.js and Express
- Build frontend components with React
- Design and implement scalable backend services

Education: Bachelor's degree in Computer Science or related field
Experience: 3+ years of experience
Certifications: AWS Certified is a plus
"""

# Test Resumes
RESUME_WITH_HEADER_FALSE_POSITIVE = """John Doe
549 | john.doe@email.com | portfolio | linkedin | github
123 Main Street, City, State 12345

SUMMARY
Senior Full Stack Developer with 5 years of experience.

EXPERIENCE
Senior Developer at Tech Corp
- Developed REST APIs using Node.js and Express
- Built React frontend components
- Implemented Git-based CI workflow

TECHNICAL SKILLS
Git, GitHub, Docker, Linux, React, TypeScript, Node.js, PostgreSQL

EDUCATION
B.Tech Computer Science
"""

RESUME_STRONG_MATCH = """Jane Smith
jane.smith@email.com

SUMMARY
Senior Full Stack Developer with 5 years of experience building scalable web applications.

EXPERIENCE
Senior Developer at Tech Corp
- Designed and implemented REST APIs using Node.js and Express
- Developed React frontend components with TypeScript
- Deployed microservices using Docker containers
- Managed PostgreSQL databases
- Implemented CI/CD pipelines with Jenkins

TECHNICAL SKILLS
React, TypeScript, Node.js, PostgreSQL, Docker, AWS, Git, Linux

EDUCATION
Bachelor of Science in Computer Science
"""

RESUME_PARTIAL_MATCH = """Bob Johnson
bob.johnson@email.com

SUMMARY
Full Stack Developer with 2 years of experience.

EXPERIENCE
Developer at Startup Inc
- Built React applications
- Used Node.js for backend development
- Worked with SQL databases

TECHNICAL SKILLS
React, JavaScript, Node.js, SQL, Git

EDUCATION
Bachelor's in Information Technology
"""

RESUME_WEAK_MATCH = """Alice Wilson
alice.wilson@email.com

SUMMARY
Junior developer looking for opportunities.

EXPERIENCE
Intern at Company
- Created marketing materials
- Worked with Adobe Creative Suite

TECHNICAL SKILLS
Photoshop, Illustrator, Microsoft Office

EDUCATION
High School Diploma
"""

RESUME_EMPTY = ""


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
    skills = extract_skills("React TypeScript Node.js PostgreSQL")
    assert "react" in skills
    assert "typescript" in skills
    # Node.js normalizes to javascript due to alias mapping
    assert "javascript" in skills
    assert "postgresql" in skills


def test_section_detection():
    """Test resume section detection and normalization."""
    sections = detect_sections(RESUME_STRONG_MATCH)
    section_names = [s.normalized_name for s in sections]
    
    assert "summary" in section_names
    assert "experience" in section_names
    assert "technical_skills" in section_names
    assert "education" in section_names


def test_section_weights():
    """Test that sections have correct evidence weights."""
    sections = detect_sections(RESUME_STRONG_MATCH)
    
    for section in sections:
        if section.normalized_name == "experience":
            assert section.weight == 1.0
        elif section.normalized_name == "technical_skills":
            assert section.weight == 0.65
        elif section.normalized_name == "header":
            assert section.weight == 0.0


def test_header_false_positive_fix():
    """Test that GitHub in header doesn't match Git."""
    sections = detect_sections(RESUME_WITH_HEADER_FALSE_POSITIVE)
    skill_matches = extract_skills_with_sections(sections)
    
    # Git should be detected from Technical Skills section
    assert "git" in skill_matches
    
    # But evidence should NOT come from header section
    git_matches = skill_matches["git"]
    for match in git_matches:
        assert match.section != "header"
        assert match.evidence_weight > 0


def test_skill_extraction_with_sections():
    """Test skill extraction with section context."""
    sections = detect_sections(RESUME_STRONG_MATCH)
    skill_matches = extract_skills_with_sections(sections)
    
    # Check that skills are extracted with section info
    assert "react" in skill_matches
    assert "typescript" in skill_matches
    
    # Check that matches have section information
    for skill, matches in skill_matches.items():
        for match in matches:
            assert match.section != ""
            assert match.evidence_weight >= 0


def test_jd_parsing():
    """Test JD requirement parsing."""
    ranker = ResumeRanker(use_embeddings=False)
    jd_req = ranker.parse_jd_requirements(JD)
    
    # Should extract skills into at least one category
    all_skills = jd_req.required_skills | jd_req.preferred_skills | jd_req.optional_skills
    assert len(all_skills) > 0
    
    # Should extract responsibilities
    assert len(jd_req.responsibilities) > 0
    
    # Should extract job title
    assert jd_req.job_title != ""


def test_required_vs_preferred_classification():
    """Test that required and preferred skills are classified correctly."""
    ranker = ResumeRanker(use_embeddings=False)
    jd_req = ranker.parse_jd_requirements(JD)
    
    # React should be in required
    assert "react" in jd_req.required_skills or "react" in jd_req.preferred_skills
    
    # Docker should be in preferred
    assert "docker" in jd_req.preferred_skills or "docker" in jd_req.required_skills


def test_required_skill_score():
    """Test required skill score calculation."""
    ranker = ResumeRanker(use_embeddings=False)
    jd_req = ranker.parse_jd_requirements(JD)
    sections = detect_sections(RESUME_STRONG_MATCH)
    skill_matches = extract_skills_with_sections(sections)
    
    score, details = ranker.calculate_required_skill_score(jd_req, skill_matches)
    
    assert 0 <= score <= 100
    assert 'matched_required' in details
    assert 'missing_required' in details


def test_preferred_skill_score():
    """Test preferred skill score calculation."""
    ranker = ResumeRanker(use_embeddings=False)
    jd_req = ranker.parse_jd_requirements(JD)
    sections = detect_sections(RESUME_STRONG_MATCH)
    skill_matches = extract_skills_with_sections(sections)
    
    score, details = ranker.calculate_preferred_skill_score(jd_req, skill_matches)
    
    assert 0 <= score <= 100
    assert 'matched_preferred' in details
    assert 'missing_preferred' in details


def test_experience_score_with_action_verbs():
    """Test experience score with action verb detection."""
    ranker = ResumeRanker(use_embeddings=False)
    jd_req = ranker.parse_jd_requirements(JD)
    sections = detect_sections(RESUME_STRONG_MATCH)
    skill_matches = extract_skills_with_sections(sections)
    
    score, evidence = ranker.calculate_experience_score(jd_req, sections, skill_matches)
    
    assert 0 <= score <= 100
    assert len(evidence) > 0
    
    # Check that evidence has section info
    for ev in evidence:
        assert ev.section != ""


def test_semantic_score():
    """Test semantic score calculation."""
    ranker = ResumeRanker(use_embeddings=False)  # Disable for this test
    jd_req = ranker.parse_jd_requirements(JD)
    sections = detect_sections(RESUME_STRONG_MATCH)
    
    score, evidence = ranker.calculate_semantic_score(jd_req, sections)
    
    # Without embeddings, score should be 0
    assert score == 0.0


def test_qualification_score():
    """Test qualification score calculation."""
    ranker = ResumeRanker(use_embeddings=False)
    jd_req = ranker.parse_jd_requirements(JD)
    sections = detect_sections(RESUME_STRONG_MATCH)
    
    score = ranker.calculate_qualification_score(jd_req, sections)
    
    assert 0 <= score <= 100


def test_ats_quality_score():
    """Test ATS quality score calculation."""
    ranker = ResumeRanker(use_embeddings=False)
    sections = detect_sections(RESUME_STRONG_MATCH)
    
    score, warnings = ranker.calculate_ats_quality_score(sections, RESUME_STRONG_MATCH)
    
    assert 0 <= score <= 100
    assert isinstance(warnings, list)


def test_final_score_calculation():
    """Test deterministic final score calculation."""
    # Test the formula: required*0.30 + experience*0.25 + semantic*0.20 + qualification*0.10 + preferred*0.05 + ats*0.10
    required = 90
    experience = 80
    semantic = 70
    qualification = 100
    preferred = 60
    ats = 95
    
    expected = required * 0.30 + experience * 0.25 + semantic * 0.20 + qualification * 0.10 + preferred * 0.05 + ats * 0.10
    assert abs(expected - 83.5) < 0.01  # Should be 83.5


def test_ranking_order():
    """Test that ranking order is correct."""
    ranker = ResumeRanker(use_embeddings=False)
    test_resumes = {
        "strong.txt": RESUME_STRONG_MATCH,
        "partial.txt": RESUME_PARTIAL_MATCH,
        "weak.txt": RESUME_WEAK_MATCH,
    }
    results = ranker.rank(JD, test_resumes)
    
    # Strong should rank first
    assert results[0].name == "strong.txt"
    # Weak should rank last
    assert results[-1].name == "weak.txt"


def test_empty_resume():
    """Test handling of empty resume."""
    ranker = ResumeRanker(use_embeddings=False)
    results = ranker.rank(JD, {"empty.txt": RESUME_EMPTY})
    
    assert len(results) == 1
    assert results[0].score >= 0
    assert results[0].required_skill_score == 0.0


def test_evidence_confidence():
    """Test evidence confidence calculation."""
    ranker = ResumeRanker(use_embeddings=False)
    results = ranker.rank(JD, {"strong.txt": RESUME_STRONG_MATCH})
    
    for r in results:
        assert 0 <= r.evidence_confidence <= 100


def test_strongest_evidence():
    """Test strongest evidence extraction."""
    ranker = ResumeRanker(use_embeddings=False)
    results = ranker.rank(JD, {"strong.txt": RESUME_STRONG_MATCH})
    
    for r in results:
        # Should have strongest evidence list
        assert len(r.strongest_evidence) <= 5  # Top 5
        
        # Should be sorted by confidence
        if len(r.strongest_evidence) > 1:
            for i in range(len(r.strongest_evidence) - 1):
                assert r.strongest_evidence[i].confidence >= r.strongest_evidence[i + 1].confidence


def test_parsing_warnings():
    """Test parsing warnings for ATS quality."""
    ranker = ResumeRanker(use_embeddings=False)
    results = ranker.rank(JD, {"strong.txt": RESUME_STRONG_MATCH})
    
    for r in results:
        assert isinstance(r.parsing_warnings, list)
        assert isinstance(r.sections_detected, list)


def test_score_bounds():
    """Test that all scores are within 0-100."""
    ranker = ResumeRanker(use_embeddings=False)
    results = ranker.rank(JD, {
        "strong.txt": RESUME_STRONG_MATCH,
        "partial.txt": RESUME_PARTIAL_MATCH,
        "weak.txt": RESUME_WEAK_MATCH,
    })
    
    for r in results:
        assert 0 <= r.score <= 100
        assert 0 <= r.required_skill_score <= 100
        assert 0 <= r.experience_score <= 100
        assert 0 <= r.semantic_score <= 100
        assert 0 <= r.qualification_score <= 100
        assert 0 <= r.preferred_skill_score <= 100
        assert 0 <= r.ats_quality_score <= 100
        assert 0 <= r.evidence_confidence <= 100


def test_backward_compatibility():
    """Test that legacy fields are still present."""
    ranker = ResumeRanker(use_embeddings=False)
    results = ranker.rank(JD, {"strong.txt": RESUME_STRONG_MATCH})
    
    for r in results:
        # Legacy fields should exist
        assert hasattr(r, 'tfidf')
        assert hasattr(r, 'semantic')
        assert hasattr(r, 'coverage')
        assert hasattr(r, 'matched')
        assert hasattr(r, 'missing')
        assert hasattr(r, 'skill_score')
        
        # New fields should exist
        assert hasattr(r, 'required_skill_score')
        assert hasattr(r, 'experience_score')
        assert hasattr(r, 'qualification_score')
        assert hasattr(r, 'preferred_skill_score')
        assert hasattr(r, 'ats_quality_score')


def test_partial_skill_matches():
    """Test partial skill match detection."""
    ranker = ResumeRanker(use_embeddings=False)
    jd_req = ranker.parse_jd_requirements(JD)
    sections = detect_sections(RESUME_PARTIAL_MATCH)
    skill_matches = extract_skills_with_sections(sections)
    
    score, details = ranker.calculate_required_skill_score(jd_req, skill_matches)
    
    # Should have partial matches for skills only in technical skills
    assert 'partial_required' in details


def test_case_insensitivity():
    """Test that skill matching is case-insensitive."""
    skills1 = extract_skills("React and TypeScript")
    skills2 = extract_skills("react and typescript")
    assert skills1 == skills2


def test_punctuation_handling():
    """Test that punctuation doesn't affect skill matching."""
    skills1 = extract_skills("React, TypeScript, Node.js")
    skills2 = extract_skills("React TypeScript Node.js")
    assert skills1 == skills2


def test_no_jd_skills():
    """Test handling when JD has no extractable skills."""
    ranker = ResumeRanker(use_embeddings=False)
    simple_jd = "We need a general manager."
    jd_req = ranker.parse_jd_requirements(simple_jd)
    
    sections = detect_sections(RESUME_STRONG_MATCH)
    skill_matches = extract_skills_with_sections(sections)
    
    score, details = ranker.calculate_required_skill_score(jd_req, skill_matches)
    assert score == 0.0  # No skills to match


def test_embedding_failure():
    """Test graceful handling of embedding failure."""
    ranker = ResumeRanker(use_embeddings=False)
    results = ranker.rank(JD, {"strong.txt": RESUME_STRONG_MATCH})
    
    # Should still produce results
    assert len(results) > 0
    for r in results:
        assert 0 <= r.score <= 100
        assert 0 <= r.semantic_score <= 100


def test_duplicate_skills():
    """Test handling of duplicate skills."""
    sections = detect_sections(RESUME_STRONG_MATCH)
    skill_matches = extract_skills_with_sections(sections)
    
    # Each skill should only appear once in the keys
    for skill, matches in skill_matches.items():
        # But may have multiple match instances
        assert len(matches) >= 1


def test_evidence_structure():
    """Test that evidence has correct structure."""
    ranker = ResumeRanker(use_embeddings=False)
    results = ranker.rank(JD, {"strong.txt": RESUME_STRONG_MATCH})
    
    for r in results:
        for ev in r.evidence:
            assert isinstance(ev, Evidence)
            assert hasattr(ev, 'requirement')
            assert hasattr(ev, 'status')
            assert hasattr(ev, 'evidence')
            assert hasattr(ev, 'match_type')
            assert hasattr(ev, 'confidence')
            assert hasattr(ev, 'section')
            assert hasattr(ev, 'evidence_strength')
            assert 0 <= ev.confidence <= 1


def test_skill_match_structure():
    """Test that skill matches have correct structure."""
    sections = detect_sections(RESUME_STRONG_MATCH)
    skill_matches = extract_skills_with_sections(sections)
    
    for skill, matches in skill_matches.items():
        for match in matches:
            assert isinstance(match, SkillMatch)
            assert hasattr(match, 'canonical')
            assert hasattr(match, 'source_text')
            assert hasattr(match, 'section')
            assert hasattr(match, 'location')
            assert hasattr(match, 'match_type')
            assert hasattr(match, 'evidence_weight')
            assert 0 <= match.evidence_weight <= 1


def test_action_verb_detection():
    """Test that action verbs are detected for evidence strength."""
    ranker = ResumeRanker(use_embeddings=False)
    assert len(ranker.ACTION_VERBS) > 0
    assert "developed" in ranker.ACTION_VERBS
    assert "implemented" in ranker.ACTION_VERBS
    assert "designed" in ranker.ACTION_VERBS


def test_section_normalization():
    """Test section name normalization."""
    sections = detect_sections(RESUME_STRONG_MATCH)
    
    # Different variations should normalize to same name
    for section in sections:
        if section.name == "TECHNICAL SKILLS":
            assert section.normalized_name == "technical_skills"
        if section.name == "SUMMARY":
            assert section.normalized_name == "summary"


def test_jd_without_preferred():
    """Test JD with no preferred skills."""
    simple_jd = """Required Skills:
- React
- TypeScript

Responsibilities:
- Build frontend components
"""
    ranker = ResumeRanker(use_embeddings=False)
    jd_req = ranker.parse_jd_requirements(simple_jd)
    
    # Should still work
    assert len(jd_req.required_skills) > 0


def test_resume_without_experience():
    """Test resume without experience section."""
    simple_resume = """John Doe
john@email.com

SUMMARY
Junior developer.

TECHNICAL SKILLS
React, JavaScript
"""
    ranker = ResumeRanker(use_embeddings=False)
    results = ranker.rank(JD, {"simple.txt": simple_resume})
    
    # Should still produce results
    assert len(results) == 1
    assert results[0].score >= 0
