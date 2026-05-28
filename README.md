# CareerMappingGenomics

**Multi-Modal Deep Learning for Personality-Grounded Career Recommendation with Neurodiversity Focus**

[![CI](https://github.com/abdallaheldaly/CareerMappingGenomics/actions/workflows/ci.yml/badge.svg)](https://github.com/abdallaheldaly/CareerMappingGenomics/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange.svg)](https://pytorch.org)

> **Research Prototype v1.0** — Abdallah El-Daly, 2025  
> 🌐 **Live Site:** https://abdallaheldaly.github.io/CareerMappingGenomics  
> ⚠️ Not for clinical use. IRB approval required for human subject research.

---

## Overview

CareerMappingGenomics is a research system that combines **four biological and psychological data modalities** through a cross-modal attention Transformer to predict Big Five personality traits and recommend career paths, with a particular focus on neurodivergent individuals — especially those on the autism spectrum.

```
SNP Genotypes (1,000+ SNPs)     ──→  SNP-Transformer (34,664 params)     ─┐
Facial Embeddings (128-d)        ──→  FaceGenome-CNN  (230,548 params)    ─┤
Blood Biomarkers (80 markers)    ──→  BiomarkerNet    (84,455 params)     ─┤──→ FusionNet ──→ Career Report
Questionnaire (Big Five 32-d)    ──→  QuestionnaireEncoder (~5,000 params)─┘   (268,746 params)
                                                               Total: 618,413 parameters
```

**Key results (20 epochs, 1,000 synthetic samples):**
- ASD classification: **88.5% validation accuracy** (vs. 85% majority baseline)
- Career matching: **39.5% validation accuracy** (vs. 3.3% random baseline)
- Big Five MAE: **0.118** on held-out test set

---

## Quick Start

### Option 1 — Docker (Recommended)

```bash
git clone https://github.com/abdallaheldaly/CareerMappingGenomics.git
cd CareerMappingGenomics

# Full pipeline
docker build -t careermapping:v1 .
docker run --rm -v $(pwd)/output:/app/output careermapping:v1

# Quick demo (~30 seconds)
docker run --rm -v $(pwd)/output:/app/output careermapping:v1 python main.py --quick

# REST API
docker run --rm -p 8000:8000 careermapping:v1 python src/api/app.py
# → Open http://localhost:8000/docs
```

### Option 2 — pip

```bash
git clone https://github.com/abdallaheldaly/CareerMappingGenomics.git
cd CareerMappingGenomics
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python main.py --quick
```

### Option 3 — conda

```bash
conda env create -f environment.yml
conda activate careermapping
python main.py --samples 1000 --epochs 20
```

---

## Project Structure

```
CareerMappingGenomics/
├── main.py                          # Full pipeline entry point
├── Dockerfile                       # Docker build
├── docker-compose.yml               # Multi-service compose
├── requirements.txt                 # pip dependencies
├── environment.yml                  # conda dependencies
├── REFERENCES.md                    # 94 academic references
│
├── configs/config.py                # All hyperparameters & paths
│
├── src/
│   ├── models/                      # SNP-Transformer, FaceGenome-CNN,
│   │                                #   BiomarkerNet, FusionNet
│   ├── pipelines/                   # Dataset, Trainer, InferenceEngine
│   ├── utils/                       # Career DB, Synthetic Data, Visualization
│   └── api/app.py                   # FastAPI REST API
│
├── tests/test_all.py                # 10 tests — all passing
│
├── docs/                            # GitHub Pages site
│   ├── index.html                   # Landing page + embedded paper
│   └── figures/                     # 8 research plots
│
├── CareerMappingGenomics_UI.jsx     # Interactive React demo (claude.ai)
└── CareerMappingGenomics_SystemDesign.jsx
```

---

## Data

**This prototype uses synthetically generated data** calibrated against published literature:

| Source | Use |
|--------|-----|
| UK Biobank (N=500k) | SNP allele frequency distributions |
| NHANES | Blood biomarker reference ranges (80 markers) |
| SPARK Autism Cohort | ASD SNP enrichment patterns |
| Published meta-analyses | ASD biomarker shifts (oxytocin ↓30%, IL-6 ↑40%, …) |
| O*NET Database | Career trait requirement vectors |

Real laboratory data can be substituted directly — the system architecture is data-agnostic.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/predict` | Full prediction |
| POST | `/predict/demo` | Demo (synthetic ASD individual) |
| GET | `/careers` | List all 30 careers |
| GET | `/careers/{id}` | Career details |
| GET | `/system/info` | Model configuration |

Full interactive docs at `http://localhost:8000/docs`

---

## Deploying to GitHub Pages

1. Push to GitHub: `git add . && git commit -m "v1.0" && git push`
2. **Settings → Pages → Source** → `main` branch → `docs/` folder
3. Site live at `https://abdallaheldaly.github.io/CareerMappingGenomics`

GitHub Actions (`.github/workflows/ci.yml`) auto-deploys on every push to `main`.

---

## Ethics

- Outputs are **probabilistic** (PGS explains 5–15% of variance). Never deterministic.
- Facial recognition has **demographic bias** — fairness auditing mandatory before deployment.
- Genomic data is **GDPR Article 9 special category** — IRB + consent required.
- Autism traits are framed as **cognitive strengths** matched to appropriate careers.
- This is an exploration tool — **never for hiring decisions or gatekeeping**.

---

## Citation

```bibtex
@misc{eldaly2025careermapping,
  title   = {CareerMappingGenomics: A Multi-Modal Deep Learning Framework for
             Personality-Grounded Career Recommendation with Neurodiversity Focus},
  author  = {El-Daly, Abdallah},
  year    = {2025},
  note    = {Research Prototype v1.0},
  url     = {https://github.com/abdallaheldaly/CareerMappingGenomics},
  license = {MIT}
}
```

---

## Collaboration

Seeking collaborators for IRB-approved clinical validation — especially MENA-region cohorts. Open a GitHub Issue or contact via the project page.

---

MIT License © 2025 Abdallah El-Daly
