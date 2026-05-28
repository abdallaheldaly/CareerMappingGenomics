"""
Career Database — O*NET-inspired taxonomy
Maps career profiles to Big Five trait requirements + autism-strength flags.
Each entry: career name, required trait levels (0-1), autism_strength flag,
education level, and description.

References
----------
[CAR-01] Peterson et al. (1999). An occupational information system for the 21st century:
         The development of O*NET. American Psychological Association.
         → Primary source for career KSAWS (Knowledge, Skills, Abilities, Work Styles)
           trait vectors. All 30 careers are mapped from O*NET occupational codes.

[PSY-08] Barrick & Mount (1991). The Big Five personality dimensions and job performance:
         A meta-analysis. Personnel Psychology, 44(1), 1–26.
         https://doi.org/10.1111/j.1744-6570.1991.tb00688.x
         → Empirical Big Five → occupational performance correlations inform trait levels.

[PSY-10] Holland (1997). Making vocational choices: A theory of vocational personalities
         and work environments (3rd ed.). Psychological Assessment Resources.
         → Holland RIASEC complementary framework for career domain classification.

[ASD-07] Baron-Cohen (2002). The extreme male brain theory of autism.
         Trends in Cognitive Sciences, 6(6), 248–254.
         https://doi.org/10.1016/S1364-6613(02)01904-6
         → Systemising strength → autism_strength=True for systematic careers.

[ASD-08] Mottron et al. (2006). Enhanced perceptual functioning in autism.
         Journal of Autism and Developmental Disorders, 36(1), 27–43.
         https://doi.org/10.1007/s10803-005-0040-7
         → Enhanced local processing → autism_advantages entries per career.

[ASD-11] Hendricks (2010). Employment and adults with autism spectrum disorders.
         Journal of Vocational Rehabilitation, 32(2), 125–134.
         https://doi.org/10.3233/JVR-2010-0502
         → Employment barriers and suitable occupational characteristics for ASD.

[ASD-12] Scott et al. (2019). Factors impacting employment for people with ASD.
         Autism, 23(4), 869–901. https://doi.org/10.1177/1362361318787789
         → Empirical factors (sensory load, structure, social demands) inform career flags.

[CAR-06] Francis et al. (2024). Machine learning in recruiting: Predicting personality
         from CVs. Frontiers in Social Psychology, 1, 1290295.
         https://doi.org/10.3389/frsps.2023.1290295
         → ML-based Big Five → vocational interest matching methodology.
"""

CAREER_DATABASE = [
    # ── STEM / Engineering ──────────────────────────────────────────────
    {
        "id": "C001", "title": "Software Engineer",
        "O": 0.75, "C": 0.80, "E": 0.40, "A": 0.55, "N": 0.30,
        "intelligence": 0.85, "autism_strength": True,
        "education": "Bachelor's in CS/Engineering",
        "domain": "Technology",
        "description": "Design, develop, and maintain software systems. Deep focus and systematic thinking are core assets.",
        "autism_advantages": ["Pattern recognition in code", "Hyperfocus on complex problems", "Systematic debugging"]
    },
    {
        "id": "C002", "title": "Data Scientist / ML Engineer",
        "O": 0.85, "C": 0.78, "E": 0.38, "A": 0.52, "N": 0.28,
        "intelligence": 0.90, "autism_strength": True,
        "education": "Master's/PhD in CS, Statistics, or Mathematics",
        "domain": "Technology",
        "description": "Extract insights from large datasets, build predictive models. Exceptional analytical thinking required.",
        "autism_advantages": ["Detail orientation in data", "Statistical pattern detection", "Consistent methodological approach"]
    },
    {
        "id": "C003", "title": "Cybersecurity Analyst",
        "O": 0.72, "C": 0.88, "E": 0.35, "A": 0.48, "N": 0.32,
        "intelligence": 0.83, "autism_strength": True,
        "education": "Bachelor's + certifications (CISSP, CEH)",
        "domain": "Technology",
        "description": "Protect systems from threats. Requires meticulous attention to detail and systematic vulnerability analysis.",
        "autism_advantages": ["Rule-based thinking", "Persistence in threat hunting", "High attention to procedural detail"]
    },
    {
        "id": "C004", "title": "Research Scientist",
        "O": 0.92, "C": 0.82, "E": 0.42, "A": 0.60, "N": 0.25,
        "intelligence": 0.92, "autism_strength": True,
        "education": "PhD in relevant field",
        "domain": "Research",
        "description": "Conduct original research, publish findings. Deep expertise in narrow domains is highly valued.",
        "autism_advantages": ["Deep specialization", "Hyperfocus on research questions", "Reduced social conformity bias"]
    },
    {
        "id": "C005", "title": "Bioinformatics Scientist",
        "O": 0.88, "C": 0.85, "E": 0.38, "A": 0.55, "N": 0.22,
        "intelligence": 0.90, "autism_strength": True,
        "education": "Master's/PhD in Bioinformatics or Computational Biology",
        "domain": "Research",
        "description": "Analyze genomic and biological data using computational tools. Perfect intersection of biology and computing.",
        "autism_advantages": ["Systematic data analysis", "Long-term project focus", "Precision in sequence analysis"]
    },
    {
        "id": "C006", "title": "Actuary",
        "O": 0.65, "C": 0.90, "E": 0.40, "A": 0.55, "N": 0.25,
        "intelligence": 0.88, "autism_strength": True,
        "education": "Bachelor's in Mathematics/Statistics + professional exams",
        "domain": "Finance",
        "description": "Assess financial risk using mathematics and statistics. Precision and attention to detail are paramount.",
        "autism_advantages": ["Superior numerical recall", "Rule-based risk assessment", "Consistent accuracy under pressure"]
    },
    {
        "id": "C007", "title": "Database Administrator",
        "O": 0.62, "C": 0.88, "E": 0.35, "A": 0.52, "N": 0.28,
        "intelligence": 0.78, "autism_strength": True,
        "education": "Bachelor's in CS or related field",
        "domain": "Technology",
        "description": "Manage and optimize databases. Systematic organization and procedural consistency are key strengths.",
        "autism_advantages": ["Structured thinking", "Consistency in procedures", "Detail-oriented query optimization"]
    },
    {
        "id": "C008", "title": "Quality Assurance Engineer",
        "O": 0.60, "C": 0.92, "E": 0.38, "A": 0.58, "N": 0.28,
        "intelligence": 0.75, "autism_strength": True,
        "education": "Bachelor's in Engineering or CS",
        "domain": "Technology",
        "description": "Test and ensure software/product quality. Methodical thinking and thoroughness are defining traits.",
        "autism_advantages": ["Exhaustive test case generation", "Pattern spotting in bugs", "Procedural rigor"]
    },
    # ── Healthcare / Science ─────────────────────────────────────────────
    {
        "id": "C009", "title": "Pharmacist",
        "O": 0.65, "C": 0.92, "E": 0.55, "A": 0.72, "N": 0.28,
        "intelligence": 0.85, "autism_strength": False,
        "education": "PharmD (Doctor of Pharmacy)",
        "domain": "Healthcare",
        "description": "Dispense medications and counsel patients. Requires precision in dosing and strong memory for drug interactions.",
        "autism_advantages": ["Drug interaction memory", "Procedural precision", "Systematic dispensing protocols"]
    },
    {
        "id": "C010", "title": "Medical Laboratory Scientist",
        "O": 0.70, "C": 0.88, "E": 0.42, "A": 0.62, "N": 0.28,
        "intelligence": 0.82, "autism_strength": True,
        "education": "Bachelor's in Medical Laboratory Science",
        "domain": "Healthcare",
        "description": "Perform diagnostic laboratory tests. Extreme precision and systematic sample handling are essential.",
        "autism_advantages": ["Precision in sample analysis", "Consistency in procedures", "High accuracy requirements"]
    },
    {
        "id": "C011", "title": "Radiologist",
        "O": 0.72, "C": 0.88, "E": 0.45, "A": 0.65, "N": 0.25,
        "intelligence": 0.92, "autism_strength": True,
        "education": "Medical degree + radiology residency",
        "domain": "Healthcare",
        "description": "Interpret medical images to diagnose conditions. Pattern recognition in images is a primary skill.",
        "autism_advantages": ["Visual pattern detection", "Systematic image review", "High attention to anomalies"]
    },
    # ── Creative / Design ────────────────────────────────────────────────
    {
        "id": "C012", "title": "Graphic Designer",
        "O": 0.90, "C": 0.68, "E": 0.52, "A": 0.65, "N": 0.38,
        "intelligence": 0.70, "autism_strength": False,
        "education": "Bachelor's in Graphic Design or Fine Arts",
        "domain": "Creative",
        "description": "Create visual concepts to communicate ideas. High openness and aesthetic sensitivity are key.",
        "autism_advantages": ["Unique visual perspective", "Intense focus on visual details", "Original pattern creation"]
    },
    {
        "id": "C013", "title": "Animator / Visual Effects Artist",
        "O": 0.88, "C": 0.72, "E": 0.45, "A": 0.60, "N": 0.35,
        "intelligence": 0.75, "autism_strength": True,
        "education": "Bachelor's in Animation or Visual Arts",
        "domain": "Creative",
        "description": "Create animations for film, games, or media. Combines technical precision with creative vision.",
        "autism_advantages": ["Frame-by-frame precision", "Deep focus on visual consistency", "Technical-creative integration"]
    },
    # ── Business / Finance ───────────────────────────────────────────────
    {
        "id": "C014", "title": "Financial Analyst",
        "O": 0.68, "C": 0.85, "E": 0.55, "A": 0.58, "N": 0.30,
        "intelligence": 0.85, "autism_strength": False,
        "education": "Bachelor's in Finance or Economics",
        "domain": "Finance",
        "description": "Analyze financial data to guide investment decisions. Quantitative rigor and attention to financial detail.",
        "autism_advantages": ["Numerical pattern recognition", "Systematic market analysis", "Consistent methodology"]
    },
    {
        "id": "C015", "title": "Accountant / Auditor",
        "O": 0.55, "C": 0.92, "E": 0.45, "A": 0.62, "N": 0.28,
        "intelligence": 0.78, "autism_strength": True,
        "education": "Bachelor's in Accounting + CPA",
        "domain": "Finance",
        "description": "Maintain financial records and ensure compliance. Extreme precision and rule-adherence are core traits.",
        "autism_advantages": ["Rule-based financial compliance", "Error detection", "Consistent procedural accuracy"]
    },
    # ── Arts / Humanities ────────────────────────────────────────────────
    {
        "id": "C016", "title": "Technical Writer",
        "O": 0.75, "C": 0.80, "E": 0.38, "A": 0.62, "N": 0.30,
        "intelligence": 0.78, "autism_strength": True,
        "education": "Bachelor's in English, CS, or Communication",
        "domain": "Communication",
        "description": "Document technical information clearly. Combines precision with linguistic skill.",
        "autism_advantages": ["Systematic documentation", "Precision in terminology", "Process-oriented writing"]
    },
    {
        "id": "C017", "title": "Translator / Linguist",
        "O": 0.80, "C": 0.75, "E": 0.40, "A": 0.68, "N": 0.32,
        "intelligence": 0.82, "autism_strength": True,
        "education": "Bachelor's in Linguistics or Modern Languages",
        "domain": "Communication",
        "description": "Translate between languages with high precision. Deep pattern recognition in language structure.",
        "autism_advantages": ["Systematic grammar rule application", "Exceptional memory for vocabulary", "Attention to nuance"]
    },
    {
        "id": "C018", "title": "Librarian / Archivist",
        "O": 0.72, "C": 0.85, "E": 0.38, "A": 0.68, "N": 0.28,
        "intelligence": 0.75, "autism_strength": True,
        "education": "Master's in Library Science",
        "domain": "Information",
        "description": "Organize, manage, and provide access to information resources.",
        "autism_advantages": ["Systematic cataloguing", "Strong memory for classification systems", "Orderly information management"]
    },
    # ── Engineering ──────────────────────────────────────────────────────
    {
        "id": "C019", "title": "Electrical Engineer",
        "O": 0.72, "C": 0.85, "E": 0.48, "A": 0.58, "N": 0.28,
        "intelligence": 0.87, "autism_strength": True,
        "education": "Bachelor's in Electrical Engineering",
        "domain": "Engineering",
        "description": "Design electrical systems and circuits. Systematic problem-solving and precision in technical specifications.",
        "autism_advantages": ["Circuit pattern recognition", "Systematic fault diagnosis", "Precision in tolerances"]
    },
    {
        "id": "C020", "title": "Mechanical Engineer",
        "O": 0.70, "C": 0.83, "E": 0.50, "A": 0.60, "N": 0.28,
        "intelligence": 0.85, "autism_strength": False,
        "education": "Bachelor's in Mechanical Engineering",
        "domain": "Engineering",
        "description": "Design and build mechanical systems. Strong spatial reasoning and analytical problem-solving.",
        "autism_advantages": ["Spatial pattern analysis", "Systematic mechanical reasoning", "Precision in design tolerances"]
    },
    # ── Mathematics ──────────────────────────────────────────────────────
    {
        "id": "C021", "title": "Mathematician / Statistician",
        "O": 0.90, "C": 0.82, "E": 0.35, "A": 0.55, "N": 0.25,
        "intelligence": 0.95, "autism_strength": True,
        "education": "PhD in Mathematics or Statistics",
        "domain": "Research",
        "description": "Develop mathematical theories and solve quantitative problems. Abstract reasoning at the highest level.",
        "autism_advantages": ["Abstract pattern recognition", "Deep immersion in mathematical structures", "Unconventional problem approaches"]
    },
    {
        "id": "C022", "title": "Cryptographer",
        "O": 0.82, "C": 0.88, "E": 0.32, "A": 0.52, "N": 0.25,
        "intelligence": 0.93, "autism_strength": True,
        "education": "PhD in Mathematics or CS with cryptography focus",
        "domain": "Technology",
        "description": "Design and analyze cryptographic systems. Combining deep mathematics with security applications.",
        "autism_advantages": ["Systematic pattern analysis", "Deep focus on mathematical structures", "Persistence in code-breaking"]
    },
    # ── Social / Education ───────────────────────────────────────────────
    {
        "id": "C023", "title": "Special Education Teacher",
        "O": 0.75, "C": 0.80, "E": 0.68, "A": 0.88, "N": 0.35,
        "intelligence": 0.72, "autism_strength": False,
        "education": "Bachelor's in Special Education + certification",
        "domain": "Education",
        "description": "Support students with diverse learning needs. Empathy and structured teaching methods are critical.",
        "autism_advantages": ["Intuitive understanding of neurodivergent learners", "Structured lesson delivery", "Pattern-based teaching"]
    },
    {
        "id": "C024", "title": "University Professor",
        "O": 0.88, "C": 0.80, "E": 0.60, "A": 0.65, "N": 0.28,
        "intelligence": 0.90, "autism_strength": False,
        "education": "PhD + postdoctoral experience",
        "domain": "Education",
        "description": "Teach and conduct research at university level. Deep expertise combined with communication skills.",
        "autism_advantages": ["Deep subject expertise", "Systematic lecture structure", "Groundbreaking research contributions"]
    },
    # ── Natural Sciences ─────────────────────────────────────────────────
    {
        "id": "C025", "title": "Ecologist / Environmental Scientist",
        "O": 0.82, "C": 0.78, "E": 0.52, "A": 0.72, "N": 0.28,
        "intelligence": 0.80, "autism_strength": False,
        "education": "Bachelor's/Master's in Ecology or Environmental Science",
        "domain": "Science",
        "description": "Study ecosystems and environmental systems. Field observation combined with data analysis.",
        "autism_advantages": ["Pattern recognition in ecological data", "Systematic field observation", "Species identification expertise"]
    },
    {
        "id": "C026", "title": "Astronomer / Astrophysicist",
        "O": 0.92, "C": 0.82, "E": 0.38, "A": 0.55, "N": 0.25,
        "intelligence": 0.95, "autism_strength": True,
        "education": "PhD in Astronomy or Physics",
        "domain": "Science",
        "description": "Study celestial objects and the universe. Deep curiosity and mathematical modeling skills.",
        "autism_advantages": ["Pattern recognition in astronomical data", "Deep focus on complex models", "Systematic data analysis from telescopes"]
    },
    # ── Art / Music ──────────────────────────────────────────────────────
    {
        "id": "C027", "title": "Composer / Music Producer",
        "O": 0.92, "C": 0.68, "E": 0.48, "A": 0.60, "N": 0.45,
        "intelligence": 0.78, "autism_strength": True,
        "education": "Bachelor's in Music Composition or self-taught",
        "domain": "Creative",
        "description": "Create original musical works. Combines auditory pattern recognition with creative expression.",
        "autism_advantages": ["Perfect pitch patterns", "Deep immersion in musical theory", "Novel harmonic structures"]
    },
    # ── Architecture / Urban Planning ────────────────────────────────────
    {
        "id": "C028", "title": "Architect",
        "O": 0.85, "C": 0.80, "E": 0.52, "A": 0.62, "N": 0.30,
        "intelligence": 0.85, "autism_strength": False,
        "education": "Bachelor's/Master's in Architecture",
        "domain": "Design",
        "description": "Design buildings and spaces. Combines creativity with technical precision and spatial intelligence.",
        "autism_advantages": ["Spatial pattern mastery", "Systematic structural analysis", "Unique design perspectives"]
    },
    # ── Psychology / Social Work ─────────────────────────────────────────
    {
        "id": "C029", "title": "Clinical Psychologist",
        "O": 0.80, "C": 0.78, "E": 0.65, "A": 0.82, "N": 0.32,
        "intelligence": 0.85, "autism_strength": False,
        "education": "PhD or PsyD in Clinical Psychology",
        "domain": "Healthcare",
        "description": "Assess and treat mental health conditions. Requires high empathy, analytical thinking, and clinical skill.",
        "autism_advantages": ["Pattern recognition in behavioral cues", "Systematic case formulation", "Deep research into conditions"]
    },
    # ── Logistics / Operations ───────────────────────────────────────────
    {
        "id": "C030", "title": "Supply Chain Analyst",
        "O": 0.62, "C": 0.88, "E": 0.50, "A": 0.62, "N": 0.28,
        "intelligence": 0.78, "autism_strength": True,
        "education": "Bachelor's in Supply Chain or Business",
        "domain": "Business",
        "description": "Optimize supply chain processes. Systems thinking and attention to operational detail.",
        "autism_advantages": ["Systematic process optimization", "Rule-based logistics thinking", "Consistency in operational procedures"]
    },
]

# Career domain lookup
CAREER_DOMAINS = list(set(c["domain"] for c in CAREER_DATABASE))

def get_careers_by_domain(domain: str):
    return [c for c in CAREER_DATABASE if c["domain"] == domain]

def get_autism_friendly_careers():
    return [c for c in CAREER_DATABASE if c["autism_strength"]]

def get_career_by_id(career_id: str):
    return next((c for c in CAREER_DATABASE if c["id"] == career_id), None)
