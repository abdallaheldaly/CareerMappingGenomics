"""
FastAPI REST API
Exposes the CareerMappingGenomics system as a web service.
Endpoints:
  POST /predict        — Full multi-modal prediction
  POST /predict/demo   — Demo with synthetic individual
  GET  /careers        — List all careers in database
  GET  /careers/{id}   — Get career details
  GET  /health         — Health check

References
----------
[DAT-12] FastAPI (2023). Modern, fast web framework for building APIs with Python.
         https://fastapi.tiangolo.com
         → REST API framework; automatic OpenAPI/Swagger documentation at /docs.

[ETH-09] Wachter et al. (2017). Counterfactual explanations without opening the black box.
         Harvard Journal of Law & Technology, 31(2), 841–887.
         → API returns structured explanations (trait_match, autism_advantages, disclaimer)
           to satisfy GDPR Article 22 explainability requirements.

[ETH-05] Clayton et al. (2019). The law of genetic privacy.
         Journal of Law and the Biosciences, 6(1), 1–36.
         https://doi.org/10.1093/jlb/lsz007
         → API processes embeddings only; raw SNPs/blood data never stored server-side.

[CAR-01] Peterson et al. (1999). O*NET occupational information system.
         → /careers endpoint serves O*NET-derived career database with KSAWS attributes.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.pipelines.inference import InferenceEngine, demo_inference
from src.utils.career_database import CAREER_DATABASE, get_autism_friendly_careers, get_career_by_id
from configs.config import GENOMIC_CFG, FACIAL_CFG, BIOMARKER_CFG

# ──────────────────────────────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="CareerMappingGenomics API",
    description=(
        "Multi-modal deep learning system for personality trait inference "
        "and career recommendation using genomics, facial analysis, blood biomarkers, "
        "and psychometric questionnaires. Research prototype — not for clinical use."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global inference engine (loaded once at startup)
_engine: Optional[InferenceEngine] = None


def get_engine() -> InferenceEngine:
    global _engine
    if _engine is None:
        _engine = InferenceEngine()
    return _engine


# ──────────────────────────────────────────────────────────────────────
# Request / Response Schemas
# ──────────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    snps:          List[float] = Field(..., description=f"SNP genotype values, length {GENOMIC_CFG.n_snps}")
    facial_emb:    List[float] = Field(..., description=f"Facial embedding vector, length {FACIAL_CFG.embedding_dim}")
    biomarkers:    List[float] = Field(..., description=f"Blood biomarker panel, length {BIOMARKER_CFG.n_markers}")
    questionnaire: List[float] = Field(..., description="Questionnaire embedding (Big Five + padding), length 32")
    top_k:         int         = Field(default=10, ge=1, le=30, description="Number of top careers to return")

    class Config:
        json_schema_extra = {
            "example": {
                "snps":          [0.5] * GENOMIC_CFG.n_snps,
                "facial_emb":    [0.0] * FACIAL_CFG.embedding_dim,
                "biomarkers":    [0.5] * BIOMARKER_CFG.n_markers,
                "questionnaire": [0.5] * 32,
                "top_k":         10,
            }
        }


class TraitScore(BaseModel):
    score: float
    level: str


class CareerMatch(BaseModel):
    rank:              int
    id:                str
    title:             str
    domain:            str
    compatibility_pct: float
    education:         str
    description:       str
    autism_strength:   bool
    autism_advantages: List[str]
    trait_match:       Dict[str, float]


class ASDAssessment(BaseModel):
    probability_pct: float
    tier:            str


class FacialEmotion(BaseModel):
    probabilities:    Dict[str, float]
    dominant_emotion: str


class PredictResponse(BaseModel):
    personality_profile:      Dict[str, TraitScore]
    asd_assessment:           ASDAssessment
    top_careers:              List[CareerMatch]
    modality_pgs:             Dict[str, float]
    facial_emotion:           FacialEmotion
    modality_contributions:   Dict[str, float]
    disclaimer:               str


class CareerSummary(BaseModel):
    id:             str
    title:          str
    domain:         str
    education:      str
    autism_strength: bool
    description:    str


# ──────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "system": "CareerMappingGenomics",
        "version": "1.0.0",
        "model_loaded": _engine is not None,
    }


@app.post("/predict", response_model=PredictResponse, summary="Full multi-modal prediction")
def predict(request: PredictRequest):
    """
    Run the full CareerMappingGenomics pipeline on the provided multi-modal inputs.
    Returns personality profile, ASD assessment, and ranked career recommendations.
    """
    engine = get_engine()

    # Validate input lengths
    expected = {
        "snps":          GENOMIC_CFG.n_snps,
        "facial_emb":    FACIAL_CFG.embedding_dim,
        "biomarkers":    BIOMARKER_CFG.n_markers,
        "questionnaire": 32,
    }
    for field_name, expected_len in expected.items():
        actual = len(getattr(request, field_name))
        if actual != expected_len:
            raise HTTPException(
                status_code=422,
                detail=f"Field '{field_name}' must have length {expected_len}, got {actual}"
            )

    try:
        report = engine.predict(
            snps=np.array(request.snps,          dtype=np.float32),
            facial_emb=np.array(request.facial_emb,    dtype=np.float32),
            biomarkers=np.array(request.biomarkers,    dtype=np.float32),
            questionnaire=np.array(request.questionnaire, dtype=np.float32),
            top_k=request.top_k,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return PredictResponse(**report)


@app.post("/predict/demo", response_model=PredictResponse, summary="Demo prediction with synthetic ASD individual")
def predict_demo(top_k: int = 10):
    """
    Generate a demo prediction using a synthetically generated ASD individual.
    Useful for testing and demonstration without real data.
    """
    from src.utils.synthetic_data import (
        generate_snps, generate_biomarkers,
        generate_big_five, generate_facial_embeddings
    )
    rng = np.random.default_rng(42)
    is_autism = np.array([True])
    snps       = generate_snps(1, is_autism)[0]
    biomarkers = generate_biomarkers(1, is_autism)[0]
    big_five   = generate_big_five(1, is_autism)[0]
    facial_emb = generate_facial_embeddings(1, big_five.reshape(1, -1))[0]
    quest      = np.hstack([big_five, rng.normal(0, 0.1, 27)]).astype(np.float32)

    engine = get_engine()
    report = engine.predict(snps, facial_emb, biomarkers, quest, top_k=top_k)
    return PredictResponse(**report)


@app.get("/careers", response_model=List[CareerSummary], summary="List all careers in database")
def list_careers(autism_only: bool = False, domain: Optional[str] = None):
    """Return all careers, optionally filtered by autism-strength flag or domain."""
    careers = CAREER_DATABASE
    if autism_only:
        careers = [c for c in careers if c["autism_strength"]]
    if domain:
        careers = [c for c in careers if c["domain"].lower() == domain.lower()]
    return [
        CareerSummary(
            id=c["id"], title=c["title"], domain=c["domain"],
            education=c["education"], autism_strength=c["autism_strength"],
            description=c["description"]
        )
        for c in careers
    ]


@app.get("/careers/{career_id}", summary="Get detailed career information")
def get_career(career_id: str):
    career = get_career_by_id(career_id)
    if not career:
        raise HTTPException(status_code=404, detail=f"Career {career_id} not found")
    return career


@app.get("/careers/domains/list", summary="List all career domains")
def list_domains():
    from src.utils.career_database import CAREER_DOMAINS
    return {"domains": CAREER_DOMAINS}


@app.get("/system/info", summary="System configuration info")
def system_info():
    return {
        "genomic":    {"n_snps": GENOMIC_CFG.n_snps, "n_pgs_traits": len(GENOMIC_CFG.pgs_traits)},
        "facial":     {"embedding_dim": FACIAL_CFG.embedding_dim, "fer_classes": FACIAL_CFG.fer_classes},
        "biomarker":  {"n_markers": BIOMARKER_CFG.n_markers},
        "careers":    {"n_careers": len(CAREER_DATABASE), "autism_friendly": len(get_autism_friendly_careers())},
    }


# ──────────────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
