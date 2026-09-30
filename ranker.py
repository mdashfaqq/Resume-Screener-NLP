"""ATS-style Resume Analyzer with explainable, evidence-based scoring.

Scores each resume against a job description with six signals:
  1. Required Skill Match (30%)      - Weighted coverage of required skills
  2. Experience & Responsibility (25%) - Evidence strength and responsibility matching
  3. Semantic Job Fit (20%)          - Semantic similarity of responsibilities
  4. Qualification Match (10%)       - Education, certifications, experience requirements
  5. Preferred Skill Match (5%)      - Weighted coverage of preferred skills
  6. ATS Resume Quality (10%)       - Resume parsing quality and ATS readiness
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


# Section normalization mapping
SECTION_NORMALIZATION = {
    # Summary/Profile
    'summary': 'summary',
    'profile': 'summary',
    'objective': 'summary',
    'professional summary': 'summary',
    'career summary': 'summary',
    'about me': 'summary',
    
    # Experience
    'experience': 'experience',
    'work experience': 'experience',
    'work history': 'experience',
    'employment': 'experience',
    'professional experience': 'experience',
    'job experience': 'experience',
    
    # Internship
    'internship': 'internship',
    'internships': 'internship',
    
    # Projects
    'projects': 'projects',
    'project experience': 'projects',
    'personal projects': 'projects',
    
    # Skills
    'skills': 'technical_skills',
    'technical skills': 'technical_skills',
    'technical_skills': 'technical_skills',
    'technologies': 'technical_skills',
    'tech stack': 'technical_skills',
    'technology stack': 'technical_skills',
    'core competencies': 'technical_skills',
    'competencies': 'technical_skills',
    
    # Education
    'education': 'education',
    'educational background': 'education',
    'academic': 'education',
    
    # Certifications
    'certifications': 'certifications',
    'certification': 'certifications',
    'certificates': 'certifications',
    'credentials': 'certifications',
    
    # Achievements
    'achievements': 'achievements',
    'accomplishments': 'achievements',
    'awards': 'achievements',
    
    # Header/Contact (evidence weight = 0)
    'contact': 'header',
    'contact information': 'header',
    'personal details': 'header',
    'header': 'header',
}

# Section evidence weights
SECTION_WEIGHTS = {
    'experience': 1.00,
    'projects': 1.00,
    'internship': 0.95,
    'work_history': 1.00,
    'responsibilities': 1.00,
    'technical_skills': 0.65,
    'summary': 0.50,
    'certifications': 0.70,
    'education': 0.30,
    'header': 0.00,
    'contact': 0.00,
    'achievements': 0.85,
}


def detect_sections(resume_text: str) -> list[ResumeSection]:
    """Detect and normalize resume sections."""
    lines = resume_text.split('\n')
    sections = []
    current_section = None
    current_content = []
    current_lines = []
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        
        # Check if this line is a section header
        line_lower = stripped.lower()
        is_header = False
        normalized_name = None
        
        for pattern, norm_name in SECTION_NORMALIZATION.items():
            if pattern in line_lower:
                is_header = True
                normalized_name = norm_name
                break
        
        # Also check for common header patterns (all caps, colon, etc.)
        if not is_header:
            if (stripped.isupper() and len(stripped) < 50) or \
               (stripped.endswith(':') and len(stripped) < 50) or \
               (re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$', stripped) and len(stripped) < 40):
                # Try to match against section patterns
                for pattern, norm_name in SECTION_NORMALIZATION.items():
                    if pattern in line_lower:
                        is_header = True
                        normalized_name = norm_name
                        break
        
        if is_header:
            # Save previous section
            if current_section is not None:
                sections.append(ResumeSection(
                    name=current_section,
                    normalized_name=current_normalized,
                    content='\n'.join(current_content),
                    lines=current_lines,
                    weight=SECTION_WEIGHTS.get(current_normalized, 0.5)
                ))
            
            # Start new section
            current_section = stripped
            current_normalized = normalized_name or 'other'
            current_content = []
            current_lines = []
        else:
            if current_section is not None:
                current_content.append(line)
                current_lines.append(line)
    
    # Save last section
    if current_section is not None:
        sections.append(ResumeSection(
            name=current_section,
            normalized_name=current_normalized,
            content='\n'.join(current_content),
            lines=current_lines,
            weight=SECTION_WEIGHTS.get(current_normalized, 0.5)
        ))
    
    # If no sections detected, treat entire text as one section
    if not sections:
        sections.append(ResumeSection(
            name='Full Resume',
            normalized_name='other',
            content=resume_text,
            lines=lines,
            weight=0.5
        ))
    
    return sections


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
        pattern = rf"(?<![\w]){re.escape(skill.lower())}(?![\w])"
        if re.search(pattern, text):
            found.add(skill.lower())
    return found


def extract_skills_with_sections(sections: list[ResumeSection]) -> dict[str, list[SkillMatch]]:
    """Extract skills with section context and evidence weights."""
    skill_matches = {}
    
    for section in sections:
        # Skip header/contact sections (weight=0)
        if section.weight == 0:
            continue
        
        section_lower = section.content.lower()
        normalized_content = normalize(section.content)
        
        for skill in SKILL_LEXICON:
            skill_lower = skill.lower()
            
            # Word boundary matching (strict to prevent false positives)
            pattern = rf"(?<![\w]){re.escape(skill_lower)}(?![\w])"
            matches = list(re.finditer(pattern, normalized_content))
            
            if matches:
                if skill_lower not in skill_matches:
                    skill_matches[skill_lower] = []
                
                # For each match, record the evidence
                for match in matches:
                    # Get context around the match
                    start = max(0, match.start() - 50)
                    end = min(len(normalized_content), match.end() + 50)
                    context = normalized_content[start:end].strip()
                    
                    # Determine match type
                    match_type = "exact"
                    if skill_lower in ALIASES.values():
                        # Check if this is an alias match
                        for alias, canonical in ALIASES.items():
                            if canonical == skill_lower and alias in section_lower:
                                match_type = "alias"
                                break
                    
                    skill_matches[skill_lower].append(SkillMatch(
                        canonical=skill_lower,
                        source_text=match.group(),
                        section=section.normalized_name,
                        location=context,
                        match_type=match_type,
                        evidence_weight=section.weight
                    ))
    
    return skill_matches


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
    job_title: str = ""


@dataclass
class SkillMatch:
    """Detailed skill match information."""
    canonical: str
    source_text: str
    section: str
    location: str
    match_type: str  # "exact", "alias", "semantic", "partial", "missing"
    evidence_weight: float = 0.0


@dataclass
class Evidence:
    """Evidence for a match with confidence."""
    requirement: str
    status: str  # "matched" or "missing"
    evidence: str = ""
    match_type: str = "direct"  # "direct", "semantic", "partial"
    confidence: float = 0.0
    section: str = ""
    evidence_strength: str = ""  # "strong", "moderate", "weak"


@dataclass
class ResumeSection:
    """Detected resume section with content."""
    name: str
    normalized_name: str
    content: str
    lines: list[str]
    weight: float = 0.0


@dataclass
class RankedResume:
    """Ranked resume with detailed ATS scoring breakdown."""
    name: str
    score: float
    
    # New ATS scoring components
    required_skill_score: float
    experience_score: float
    semantic_score: float
    qualification_score: float
    preferred_skill_score: float
    ats_quality_score: float
    evidence_confidence: float
    
    # Skill gap analysis
    matched_required_skills: list[str] = field(default_factory=list)
    missing_required_skills: list[str] = field(default_factory=list)
    partial_required_skills: list[str] = field(default_factory=list)
    matched_preferred_skills: list[str] = field(default_factory=list)
    missing_preferred_skills: list[str] = field(default_factory=list)
    partial_preferred_skills: list[str] = field(default_factory=list)
    
    # Evidence
    evidence: list[Evidence] = field(default_factory=list)
    strongest_evidence: list[Evidence] = field(default_factory=list)
    
    # Parsing info
    sections_detected: list[str] = field(default_factory=list)
    parsing_warnings: list[str] = field(default_factory=list)
    
    # Legacy fields for backward compatibility
    skill_score: float = 0.0  # Will be set to required_skill_score
    tfidf: float = 0.0
    semantic: float = 0.0
    coverage: float = 0.0
    matched: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


class ResumeRanker:
    """ATS-style resume analyzer with explainable, evidence-based scoring."""
    
    # Default weights for new ATS scoring model
    DEFAULT_WEIGHTS = {
        'required_skills': 0.30,
        'experience': 0.25,
        'semantic': 0.20,
        'qualification': 0.10,
        'preferred_skills': 0.05,
        'ats_quality': 0.10,
    }
    
    # Action verbs for evidence strength
    ACTION_VERBS = {
        'developed', 'built', 'implemented', 'designed', 'deployed', 'integrated',
        'configured', 'optimized', 'maintained', 'automated', 'tested', 'secured',
        'architected', 'managed', 'led', 'created', 'engineered', 'constructed',
        'programmed', 'coded', 'wrote', 'refactored', 'debugged', 'monitored',
        'scaled', 'launched', 'released', 'shipped', 'delivered', 'produced'
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
        
        # Detect required vs preferred indicators
        required_indicators = ['required', 'must have', 'mandatory', 'minimum', 'strong knowledge', 'proficient in', 'experience with']
        preferred_indicators = ['preferred', 'nice to have', 'bonus', 'plus', 'desirable', 'good to have']
        
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
        
        # Extract job title (first line or first few words)
        job_title = lines[0] if lines else ""
        
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
        
        # Extract responsibilities (sentences with action verbs, exclude generic filler)
        responsibilities = []
        for line in lines:
            line_lower = line.lower()
            if any(verb in line_lower for verb in ['develop', 'design', 'build', 'create', 'implement', 
                                                       'manage', 'lead', 'responsible', 'maintain',
                                                       'deploy', 'optimize', 'scale', 'architect']):
                # Exclude generic filler
                if not any(filler in line_lower for filler in ['team', 'culture', 'environment', 'atmosphere']):
                    if len(line) > 20:  # Filter out very short lines
                        responsibilities.append(line)
        
        # Extract education requirement
        education = ""
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in ['bachelor', 'master', 'phd', 'degree']):
                # Check if it's a requirement line
                if 'education' in line_lower or 'degree' in line_lower or 'require' in line_lower:
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
                if 'plus' in line_lower or 'required' in line_lower or 'must' in line_lower or line.startswith('-'):
                    certifications.append(line)
        
        return JDRequirements(
            required_skills=required,
            preferred_skills=preferred,
            optional_skills=optional,
            responsibilities=responsibilities,
            education_required=education,
            experience_required=experience,
            certifications_required=certifications,
            job_title=job_title,
        )
    
    def calculate_required_skill_score(self, jd_req: JDRequirements, skill_matches: dict[str, list[SkillMatch]]) -> tuple[float, dict]:
        """Calculate required skill score based on JD requirements with section evidence."""
        total_weight = 0.0
        matched_weight = 0.0
        
        matched_required = []
        missing_required = []
        partial_required = []
        
        # Required skills (weight 1.0)
        for skill in jd_req.required_skills:
            total_weight += 1.0
            skill_lower = skill.lower()
            
            if skill_lower in skill_matches:
                # Check if evidence exists in high-weight sections
                matches = skill_matches[skill_lower]
                strong_evidence = any(m.evidence_weight >= 0.7 for m in matches)
                
                if strong_evidence:
                    matched_weight += 1.0
                    matched_required.append(skill)
                else:
                    # Weak evidence (only in skills section)
                    matched_weight += 0.5
                    partial_required.append(skill)
            else:
                missing_required.append(skill)
        
        # Calculate score (0-100)
        if total_weight > 0:
            score = (matched_weight / total_weight) * 100
        else:
            score = 0.0
        
        details = {
            'matched_required': sorted(matched_required),
            'missing_required': sorted(missing_required),
            'partial_required': sorted(partial_required),
        }
        
        return score, details
    
    def calculate_preferred_skill_score(self, jd_req: JDRequirements, skill_matches: dict[str, list[SkillMatch]]) -> tuple[float, dict]:
        """Calculate preferred skill score based on JD requirements."""
        total_weight = 0.0
        matched_weight = 0.0
        
        matched_preferred = []
        missing_preferred = []
        partial_preferred = []
        
        # Preferred skills (weight 0.5)
        for skill in jd_req.preferred_skills:
            total_weight += 0.5
            skill_lower = skill.lower()
            
            if skill_lower in skill_matches:
                matches = skill_matches[skill_lower]
                strong_evidence = any(m.evidence_weight >= 0.7 for m in matches)
                
                if strong_evidence:
                    matched_weight += 0.5
                    matched_preferred.append(skill)
                else:
                    matched_weight += 0.25
                    partial_preferred.append(skill)
            else:
                missing_preferred.append(skill)
        
        # Calculate score (0-100)
        if total_weight > 0:
            score = (matched_weight / total_weight) * 100
        else:
            score = 0.0
        
        details = {
            'matched_preferred': sorted(matched_preferred),
            'missing_preferred': sorted(missing_preferred),
            'partial_preferred': sorted(partial_preferred),
        }
        
        return score, details
    
    def calculate_experience_score(self, jd_req: JDRequirements, sections: list[ResumeSection], skill_matches: dict[str, list[SkillMatch]]) -> tuple[float, list[Evidence]]:
        """Calculate experience relevance score based on evidence strength and section weights."""
        evidence_list = []
        
        # Evidence strength indicators
        weak_indicators = ['familiar with', 'knowledge of', 'basic', 'introductory', 'understanding of']
        medium_indicators = ['used', 'worked with', 'experience with', 'utilized', 'applied']
        strong_indicators = ['designed', 'implemented', 'deployed', 'led', 'architected', 'built', 'developed']
        
        # Check for evidence of required skills in high-weight sections
        required_skills = jd_req.required_skills | jd_req.preferred_skills
        total_skill_weight = 0.0
        matched_skill_weight = 0.0
        
        for skill in required_skills:
            skill_lower = skill.lower()
            skill_weight = 1.0 if skill in jd_req.required_skills else 0.5
            total_skill_weight += skill_weight
            
            if skill_lower in skill_matches:
                matches = skill_matches[skill_lower]
                
                # Find strongest evidence
                best_match = None
                best_strength = 0.0
                
                for match in matches:
                    # Skip low-weight sections (header, contact)
                    if match.evidence_weight == 0:
                        continue
                    
                    context = match.location.lower()
                    strength = 0.3  # Default weak
                    
                    # Check for action verbs
                    has_action_verb = any(verb in context for verb in self.ACTION_VERBS)
                    
                    if any(indicator in context for indicator in strong_indicators) or has_action_verb:
                        strength = 1.0
                    elif any(indicator in context for indicator in medium_indicators):
                        strength = 0.7
                    elif any(indicator in context for indicator in weak_indicators):
                        strength = 0.3
                    
                    # Apply section weight
                    weighted_strength = strength * match.evidence_weight
                    
                    if weighted_strength > best_strength:
                        best_strength = weighted_strength
                        best_match = match
                
                if best_match:
                    matched_skill_weight += skill_weight * best_strength
                    
                    # Determine evidence strength label
                    if best_strength >= 0.7:
                        strength_label = "strong"
                    elif best_strength >= 0.4:
                        strength_label = "moderate"
                    else:
                        strength_label = "weak"
                    
                    evidence_list.append(Evidence(
                        requirement=skill,
                        status="matched",
                        evidence=best_match.location.strip(),
                        match_type=best_match.match_type,
                        confidence=best_strength,
                        section=best_match.section,
                        evidence_strength=strength_label,
                    ))
        
        # Calculate score based on weighted evidence
        if total_skill_weight > 0:
            score = (matched_skill_weight / total_skill_weight) * 100
        else:
            score = 0.0
        
        return score, evidence_list
    
    def calculate_semantic_score(self, jd_req: JDRequirements, sections: list[ResumeSection]) -> tuple[float, list[Evidence]]:
        """Calculate semantic similarity between JD responsibilities and resume sections."""
        if not self.use_embeddings or self.embedder is None:
            return 0.0, []
        
        if not jd_req.responsibilities:
            return 0.0, []
        
        # Collect sentences from high-weight sections (experience, projects, internship)
        relevant_sentences = []
        for section in sections:
            if section.weight >= 0.7:  # Only consider high-weight sections
                sentences = [s.strip() for s in section.content.split('.') if len(s.strip()) > 20]
                relevant_sentences.extend(sentences)
        
        if not relevant_sentences:
            return 0.0, []
        
        # Encode responsibilities and resume sentences
        resp_embeddings = self.embedder.encode(jd_req.responsibilities, normalize_embeddings=True)
        resume_embeddings = self.embedder.encode(relevant_sentences, normalize_embeddings=True)
        
        # Find best matches for each responsibility
        similarities = []
        evidence_list = []
        
        for i, resp_emb in enumerate(resp_embeddings):
            # Calculate cosine similarity with all resume sentences
            sims = np.dot(resume_embeddings, resp_emb)
            best_idx = np.argmax(sims)
            best_score = float(sims[best_idx])
            
            if best_score > 0.4:  # Higher threshold for meaningful semantic match
                similarities.append(best_score)
                evidence_list.append(Evidence(
                    requirement=jd_req.responsibilities[i],
                    status="matched",
                    evidence=relevant_sentences[best_idx],
                    match_type="semantic",
                    confidence=best_score,
                    evidence_strength="strong" if best_score >= 0.7 else "moderate",
                ))
        
        # Calculate aggregate score (weighted average to prevent one match from dominating)
        if similarities:
            score = np.mean(similarities) * 100
        else:
            score = 0.0
        
        return score, evidence_list
    
    def calculate_qualification_score(self, jd_req: JDRequirements, sections: list[ResumeSection]) -> float:
        """Calculate qualification match score (education, certifications, experience)."""
        resume_text = '\n'.join(s.content for s in sections)
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
    
    def calculate_ats_quality_score(self, sections: list[ResumeSection], resume_text: str) -> tuple[float, list[str]]:
        """Calculate ATS resume quality score."""
        warnings = []
        score = 100.0
        
        # Check for section detection
        section_names = [s.normalized_name for s in sections]
        
        # Required sections for ATS
        required_sections = ['experience', 'education', 'technical_skills']
        for req_section in required_sections:
            if req_section not in section_names:
                score -= 15
                warnings.append(f"Missing or unrecognized {req_section} section")
        
        # Check for contact/header section
        if 'header' not in section_names and 'contact' not in section_names:
            # Check if email/phone present anywhere
            if not re.search(r'[\w\.-]+@[\w\.-]+\.\w+', resume_text):
                score -= 10
                warnings.append("No email address detected")
            if not re.search(r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}', resume_text):
                score -= 5
                warnings.append("No phone number detected")
        
        # Check for excessively long paragraphs (ATS parsing issue)
        for section in sections:
            lines = section.content.split('\n')
            for line in lines:
                if len(line) > 500:  # Very long line
                    score -= 5
                    warnings.append(f"Very long paragraph in {section.name} section")
                    break
        
        # Check for broken characters (common PDF extraction issues)
        if re.search(r'[^\x00-\x7F]', resume_text):
            # Non-ASCII characters present - minor deduction
            score -= 2
        
        # Check for duplicate sections
        section_name_counts = {}
        for section in sections:
            name = section.normalized_name
            section_name_counts[name] = section_name_counts.get(name, 0) + 1
        
        for name, count in section_name_counts.items():
            if count > 1:
                score -= 5
                warnings.append(f"Duplicate {name} section detected")
        
        return max(score, 0.0), warnings
    
    def calculate_evidence_confidence(self, evidence_list: list[Evidence]) -> float:
        """Calculate overall confidence based on evidence quality."""
        if not evidence_list:
            return 0.0
        
        confidences = [e.confidence for e in evidence_list]
        return np.mean(confidences) * 100
    
    def rank(self, job_description: str, resumes: dict[str, str]) -> list[RankedResume]:
        """Rank resumes against job description using ATS scoring model."""
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
            
            # Detect resume sections
            sections = detect_sections(resume_text)
            
            # Extract skills with section context
            skill_matches = extract_skills_with_sections(sections)
            
            # Calculate new ATS scoring components
            required_skill_score, req_details = self.calculate_required_skill_score(jd_req, skill_matches)
            preferred_skill_score, pref_details = self.calculate_preferred_skill_score(jd_req, skill_matches)
            experience_score, experience_evidence = self.calculate_experience_score(jd_req, sections, skill_matches)
            semantic_score, semantic_evidence = self.calculate_semantic_score(jd_req, sections)
            qualification_score = self.calculate_qualification_score(jd_req, sections)
            ats_quality_score, parsing_warnings = self.calculate_ats_quality_score(sections, resume_text)
            
            # Combine evidence
            all_evidence = experience_evidence + semantic_evidence
            
            # Calculate strongest evidence (top 5 by confidence)
            strongest_evidence = sorted(all_evidence, key=lambda e: e.confidence, reverse=True)[:5]
            
            # Calculate evidence confidence
            evidence_confidence = self.calculate_evidence_confidence(all_evidence)
            
            # Calculate final ATS score
            final_score = (
                required_skill_score * self.DEFAULT_WEIGHTS['required_skills'] +
                experience_score * self.DEFAULT_WEIGHTS['experience'] +
                semantic_score * self.DEFAULT_WEIGHTS['semantic'] +
                qualification_score * self.DEFAULT_WEIGHTS['qualification'] +
                preferred_skill_score * self.DEFAULT_WEIGHTS['preferred_skills'] +
                ats_quality_score * self.DEFAULT_WEIGHTS['ats_quality']
            )
            
            # Legacy compatibility fields
            all_skills = jd_req.required_skills | jd_req.preferred_skills
            resume_skills = extract_skills(resume_text)
            legacy_matched = sorted(all_skills & resume_skills)
            legacy_missing = sorted(all_skills - resume_skills)
            legacy_coverage = len(legacy_matched) / len(all_skills) if all_skills else 0.0
            
            # Combined skill score for legacy compatibility
            combined_skill_score = (required_skill_score * 0.875 + preferred_skill_score * 0.125)  # 7:1 ratio
            
            results.append(RankedResume(
                name=name,
                score=round(final_score, 1),
                
                # New ATS scoring components
                required_skill_score=round(required_skill_score, 1),
                experience_score=round(experience_score, 1),
                semantic_score=round(semantic_score, 1),
                qualification_score=round(qualification_score, 1),
                preferred_skill_score=round(preferred_skill_score, 1),
                ats_quality_score=round(ats_quality_score, 1),
                evidence_confidence=round(evidence_confidence, 0),
                
                # Skill gap analysis
                matched_required_skills=req_details['matched_required'],
                missing_required_skills=req_details['missing_required'],
                partial_required_skills=req_details['partial_required'],
                matched_preferred_skills=pref_details['matched_preferred'],
                missing_preferred_skills=pref_details['missing_preferred'],
                partial_preferred_skills=pref_details['partial_preferred'],
                
                # Evidence
                evidence=all_evidence,
                strongest_evidence=strongest_evidence,
                
                # Parsing info
                sections_detected=[s.normalized_name for s in sections],
                parsing_warnings=parsing_warnings,
                
                # Legacy fields
                skill_score=round(combined_skill_score, 1),
                tfidf=round(float(tfidf_scores[i]), 3),
                semantic=round(float(legacy_sem_scores[i]), 3),
                coverage=round(legacy_coverage, 3),
                matched=legacy_matched,
                missing=legacy_missing,
            ))
        
        return sorted(results, key=lambda r: r.score, reverse=True)
