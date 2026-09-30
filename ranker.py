"""Resume ranking engine.

Scores each resume against a job description with four signals:
  1. Skill Match (35%)      - Weighted coverage of required/preferred/optional skills
  2. Experience Match (30%) - Evidence strength of relevant experience
  3. Semantic Match (25%)  - Semantic similarity of responsibilities to resume sections
  4. Qualification Match (10%) - Education, certifications, and explicit requirements
The final score is a weighted blend; all components normalized to 0-100.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load skill configuration from external file
def load_skill_config(config_path: str | None = None) -> dict[str, dict[str, Any]]:
    """Load skill configuration from JSON file."""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "skills_config.json")
    
    if not os.path.exists(config_path):
        return {}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f).get("skills", {})
    except Exception:
        return {}


SKILL_CONFIG = load_skill_config()

# Build skill lexicon and alias mapping from config
SKILL_LEXICON = set(SKILL_CONFIG.keys())
ALIASES = {}
for skill, config in SKILL_CONFIG.items():
    for alias in config.get("aliases", []):
        ALIASES[alias.lower()] = skill.lower()


def normalize(text: str) -> str:
    """Normalize text: lowercase, apply aliases, normalize whitespace."""
    text = text.lower()
    for alias, canonical in ALIASES.items():
        text = re.sub(rf"(?<![\w.]){re.escape(alias)}(?![\w])", canonical, text)
    return re.sub(r"\s+", " ", text).strip()


def extract_skills(text: str) -> set[str]:
    """Extract skills from text using the configured lexicon."""
    text = normalize(text)
    found = set()
    for skill in SKILL_LEXICON:
        # Word boundary matching
        pattern = rf"(?<![\w+]){re.escape(skill.lower())}(?![\w+])"
        if re.search(pattern, text):
            found.add(skill.lower())
    return found


@dataclass
class JDRequirements:
    """Structured job description requirements."""
    required_skills: set[str] = field(default_factory=set)
    preferred_skills: set[str] = field(default_factory=set)
    optional_skills: set[str] = field(default_factory=set)
    responsibilities: list[str] = field(default_factory=list)
    education_required: str = ""
    experience_required: str = ""
    certifications_required: list[str] = field(default_factory=list)


@dataclass
class Evidence:
    """Evidence for a match with confidence."""
    requirement: str
    status: str  # "matched" or "missing"
    evidence: str = ""
    match_type: str = "direct"  # "direct", "semantic", "partial"
    confidence: float = 0.0


@dataclass
class RankedResume:
    """Ranked resume with detailed scoring breakdown."""
    name: str
    score: float
    skill_score: float
    experience_score: float
    semantic_score: float
    qualification_score: float
    evidence_confidence: float
    
    # Legacy fields for backward compatibility
    tfidf: float = 0.0
    semantic: float = 0.0
    coverage: float = 0.0
    matched: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    
    # New detailed fields
    matched_required_skills: list[str] = field(default_factory=list)
    missing_required_skills: list[str] = field(default_factory=list)
    matched_preferred_skills: list[str] = field(default_factory=list)
    missing_preferred_skills: list[str] = field(default_factory=list)
    optional_matches: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)


class ResumeRanker:
    """Enhanced resume ranking engine with job-aware scoring."""
    
    # Default weights for new scoring model
    DEFAULT_WEIGHTS = {
        'skill': 0.35,
        'experience': 0.30,
        'semantic': 0.25,
        'qualification': 0.10,
    }
    
    def __init__(self, weights: tuple[float, float, float] | None = None, use_embeddings: bool = True):
        """Initialize ranker with optional legacy weights for compatibility."""
        self.embedder = None
        self.use_embeddings = use_embeddings
        
        # Initialize embedding model if requested
        if use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                self.use_embeddings = False
        
        # Store legacy weights for backward compatibility
        if weights is not None:
            self.w_tfidf, self.w_sem, self.w_cov = weights
        else:
            self.w_tfidf, self.w_sem, self.w_cov = 0.35, 0.40, 0.25
    
    def parse_jd_requirements(self, job_description: str) -> JDRequirements:
        """Parse job description into structured requirements."""
        jd_lower = job_description.lower()
        all_skills = extract_skills(job_description)
        
        # Conservative approach: if no clear distinction, treat all as required
        required = set(all_skills)
        preferred = set()
        optional = set()
        
        # Look for section headers to classify skills
        lines = [line.strip() for line in job_description.split('\n') if line.strip()]
        current_section = "required"
        
        # Reset to empty for section-based parsing
        required = set()
        preferred = set()
        optional = set()
        
        for line in lines:
            line_lower = line.lower()
            
            # Detect section changes - more precise matching
            if any(keyword in line_lower for keyword in ['required skills:', 'required:', 'must have:', 'essential:']):
                current_section = "required"
            elif any(keyword in line_lower for keyword in ['preferred skills:', 'preferred:', 'nice to have:', 'plus:', 'bonus:']):
                current_section = "preferred"
            elif any(keyword in line_lower for keyword in ['optional skills:', 'optional:', 'not required:']):
                current_section = "optional"
            
            # Extract skills based on current section
            line_skills = extract_skills(line)
            if current_section == "required":
                required.update(line_skills)
            elif current_section == "preferred":
                preferred.update(line_skills)
            elif current_section == "optional":
                optional.update(line_skills)
        
        # If no sections found, fall back to treating all as required
        if not required and not preferred and not optional:
            required = set(all_skills)
        
        # Extract responsibilities (sentences with action verbs)
        responsibilities = []
        for line in lines:
            if any(verb in line.lower() for verb in ['develop', 'design', 'build', 'create', 'implement', 
                                                       'manage', 'lead', 'work on', 'responsible']):
                if len(line) > 20:  # Filter out very short lines
                    responsibilities.append(line)
        
        # Extract education requirement
        education = ""
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in ['bachelor', 'master', 'phd', 'degree']):
                # Check if it's a requirement line
                if 'education' in line_lower or 'degree' in line_lower:
                    education = line
                    break
        
        # Extract experience requirement
        experience = ""
        for line in lines:
            line_lower = line.lower()
            if re.search(r'\d+\+?\s*years?', line_lower):
                if 'experience' in line_lower:
                    experience = line
                    break
        
        # Extract certifications
        certifications = []
        for line in lines:
            line_lower = line.lower()
            if 'certification' in line_lower or 'certified' in line_lower:
                # Only add if it seems like a requirement, not just a mention
                if 'plus' in line_lower or 'required' in line_lower or line.startswith('-'):
                    certifications.append(line)
        
        return JDRequirements(
            required_skills=required,
            preferred_skills=preferred,
            optional_skills=optional,
            responsibilities=responsibilities,
            education_required=education,
            experience_required=experience,
            certifications_required=certifications,
        )
    
    def calculate_skill_score(self, jd_req: JDRequirements, resume_text: str) -> tuple[float, dict]:
        """Calculate weighted skill score based on JD requirements."""
        resume_skills = extract_skills(resume_text)
        
        # Calculate weighted coverage
        total_weight = 0.0
        matched_weight = 0.0
        
        matched_required = []
        missing_required = []
        matched_preferred = []
        missing_preferred = []
        optional_matches = []
        
        # Required skills (weight 1.0)
        for skill in jd_req.required_skills:
            total_weight += 1.0
            if skill in resume_skills:
                matched_weight += 1.0
                matched_required.append(skill)
            else:
                missing_required.append(skill)
        
        # Preferred skills (weight 0.5)
        for skill in jd_req.preferred_skills:
            total_weight += 0.5
            if skill in resume_skills:
                matched_weight += 0.5
                matched_preferred.append(skill)
            else:
                missing_preferred.append(skill)
        
        # Optional skills (weight 0.25)
        for skill in jd_req.optional_skills:
            total_weight += 0.25
            if skill in resume_skills:
                matched_weight += 0.25
                optional_matches.append(skill)
        
        # Calculate score (0-100)
        if total_weight > 0:
            score = (matched_weight / total_weight) * 100
        else:
            score = 0.0
        
        details = {
            'matched_required': sorted(matched_required),
            'missing_required': sorted(missing_required),
            'matched_preferred': sorted(matched_preferred),
            'missing_preferred': sorted(missing_preferred),
            'optional_matches': sorted(optional_matches),
        }
        
        return score, details
    
    def calculate_experience_score(self, jd_req: JDRequirements, resume_text: str) -> tuple[float, list[Evidence]]:
        """Calculate experience relevance score based on evidence strength."""
        resume_lower = resume_text.lower()
        evidence_list = []
        
        # Evidence strength indicators
        weak_indicators = ['familiar with', 'knowledge of', 'basic', 'introductory']
        medium_indicators = ['used', 'worked with', 'experience with', 'developed']
        strong_indicators = ['designed', 'implemented', 'deployed', 'led', 'architected', 'built']
        
        # Check for evidence of required skills
        required_skills = jd_req.required_skills | jd_req.preferred_skills
        evidence_count = 0
        strong_evidence_count = 0
        
        for skill in required_skills:
            skill_lower = skill.lower()
            if skill_lower in resume_lower:
                evidence_count += 1
                
                # Find context around skill mention
                skill_pattern = rf'.{{0,50}}{re.escape(skill_lower)}.{{0,50}}'
                matches = re.finditer(skill_pattern, resume_lower)
                
                for match in matches:
                    context = match.group()
                    strength = 0.3  # Default weak
                    
                    if any(indicator in context for indicator in strong_indicators):
                        strength = 1.0
                        strong_evidence_count += 1
                    elif any(indicator in context for indicator in medium_indicators):
                        strength = 0.7
                    elif any(indicator in context for indicator in weak_indicators):
                        strength = 0.3
                    
                    evidence_list.append(Evidence(
                        requirement=skill,
                        status="matched",
                        evidence=context.strip(),
                        match_type="direct",
                        confidence=strength,
                    ))
                    break  # Take first match per skill
        
        # Calculate score based on evidence strength
        if len(required_skills) > 0:
            base_score = (evidence_count / len(required_skills)) * 100
            # Boost score based on strong evidence
            if evidence_count > 0:
                strength_bonus = (strong_evidence_count / evidence_count) * 20
                score = min(base_score + strength_bonus, 100)
            else:
                score = base_score
        else:
            score = 0.0
        
        return score, evidence_list
    
    def calculate_semantic_score(self, jd_req: JDRequirements, resume_text: str) -> tuple[float, list[Evidence]]:
        """Calculate semantic similarity between JD responsibilities and resume."""
        if not self.use_embeddings or self.embedder is None:
            return 0.0, []
        
        if not jd_req.responsibilities:
            return 0.0, []
        
        # Split resume into sections (naive approach)
        resume_sentences = [s.strip() for s in resume_text.split('.') if len(s.strip()) > 20]
        
        if not resume_sentences:
            return 0.0, []
        
        # Encode responsibilities and resume sentences
        resp_embeddings = self.embedder.encode(jd_req.responsibilities, normalize_embeddings=True)
        resume_embeddings = self.embedder.encode(resume_sentences, normalize_embeddings=True)
        
        # Find best matches for each responsibility
        similarities = []
        evidence_list = []
        
        for i, resp_emb in enumerate(resp_embeddings):
            # Calculate cosine similarity with all resume sentences
            sims = np.dot(resume_embeddings, resp_emb)
            best_idx = np.argmax(sims)
            best_score = float(sims[best_idx])
            
            if best_score > 0.3:  # Threshold for meaningful match
                similarities.append(best_score)
                evidence_list.append(Evidence(
                    requirement=jd_req.responsibilities[i],
                    status="matched",
                    evidence=resume_sentences[best_idx],
                    match_type="semantic",
                    confidence=best_score,
                ))
        
        # Calculate aggregate score
        if similarities:
            score = np.mean(similarities) * 100
        else:
            score = 0.0
        
        return score, evidence_list
    
    def calculate_qualification_score(self, jd_req: JDRequirements, resume_text: str) -> float:
        """Calculate qualification match score (education, certifications, experience)."""
        resume_lower = resume_text.lower()
        score = 100.0  # Start with full score, deduct for missing requirements
        
        # Check education requirement
        if jd_req.education_required:
            edu_lower = jd_req.education_required.lower()
            # Check if any education keyword matches
            if 'bachelor' in edu_lower and 'bachelor' not in resume_lower and 'b.tech' not in resume_lower:
                score -= 30
            if 'master' in edu_lower and 'master' not in resume_lower and 'm.tech' not in resume_lower:
                score -= 30
            if 'phd' in edu_lower and 'phd' not in resume_lower:
                score -= 30
        
        # Check experience requirement
        if jd_req.experience_required:
            exp_match = re.search(r'(\d+)\+?\s*years?', jd_req.experience_required.lower())
            if exp_match:
                required_years = int(exp_match.group(1))
                # Try to find years in resume
                resume_exp = re.findall(r'(\d+)\+?\s*years?', resume_lower)
                if resume_exp:
                    max_years = max(int(y) for y in resume_exp)
                    if max_years < required_years:
                        score -= 20
                else:
                    score -= 20
        
        # Check certifications
        for cert in jd_req.certifications_required:
            cert_lower = cert.lower()
            if cert_lower not in resume_lower:
                score -= 10
        
        return max(score, 0.0)
    
    def calculate_evidence_confidence(self, evidence_list: list[Evidence]) -> float:
        """Calculate overall confidence based on evidence quality."""
        if not evidence_list:
            return 0.0
        
        confidences = [e.confidence for e in evidence_list]
        return np.mean(confidences) * 100
    
    def rank(self, job_description: str, resumes: dict[str, str]) -> list[RankedResume]:
        """Rank resumes against job description using new scoring model."""
        # Parse JD requirements
        jd_req = self.parse_jd_requirements(job_description)
        
        # Calculate legacy TF-IDF for backward compatibility
        names = list(resumes)
        docs = [normalize(job_description)] + [normalize(resumes[n]) for n in names]
        
        tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        matrix = tfidf.fit_transform(docs)
        tfidf_scores = cosine_similarity(matrix[0], matrix[1:]).ravel()
        
        # Calculate legacy semantic score
        if self.use_embeddings and self.embedder is not None:
            vecs = self.embedder.encode(docs, normalize_embeddings=True)
            legacy_sem_scores = (vecs[1:] @ vecs[0]).clip(0, 1)
        else:
            legacy_sem_scores = np.zeros(len(names))
        
        results = []
        for i, name in enumerate(names):
            resume_text = resumes[name]
            
            # Calculate new scoring components
            skill_score, skill_details = self.calculate_skill_score(jd_req, resume_text)
            experience_score, experience_evidence = self.calculate_experience_score(jd_req, resume_text)
            semantic_score, semantic_evidence = self.calculate_semantic_score(jd_req, resume_text)
            qualification_score = self.calculate_qualification_score(jd_req, resume_text)
            
            # Combine evidence
            all_evidence = experience_evidence + semantic_evidence
            
            # Calculate final score
            final_score = (
                skill_score * self.DEFAULT_WEIGHTS['skill'] +
                experience_score * self.DEFAULT_WEIGHTS['experience'] +
                semantic_score * self.DEFAULT_WEIGHTS['semantic'] +
                qualification_score * self.DEFAULT_WEIGHTS['qualification']
            )
            
            # Calculate evidence confidence
            evidence_confidence = self.calculate_evidence_confidence(all_evidence)
            
            # Legacy compatibility fields
            all_skills = jd_req.required_skills | jd_req.preferred_skills
            resume_skills = extract_skills(resume_text)
            legacy_matched = sorted(all_skills & resume_skills)
            legacy_missing = sorted(all_skills - resume_skills)
            legacy_coverage = len(legacy_matched) / len(all_skills) if all_skills else 0.0
            
            results.append(RankedResume(
                name=name,
                score=round(final_score, 1),
                skill_score=round(skill_score, 1),
                experience_score=round(experience_score, 1),
                semantic_score=round(semantic_score, 1),
                qualification_score=round(qualification_score, 1),
                evidence_confidence=round(evidence_confidence, 0),
                
                # Legacy fields
                tfidf=round(float(tfidf_scores[i]), 3),
                semantic=round(float(legacy_sem_scores[i]), 3),
                coverage=round(legacy_coverage, 3),
                matched=legacy_matched,
                missing=legacy_missing,
                
                # New detailed fields
                matched_required_skills=skill_details['matched_required'],
                missing_required_skills=skill_details['missing_required'],
                matched_preferred_skills=skill_details['matched_preferred'],
                missing_preferred_skills=skill_details['missing_preferred'],
                optional_matches=skill_details['optional_matches'],
                evidence=all_evidence,
            ))
        
        return sorted(results, key=lambda r: r.score, reverse=True)
