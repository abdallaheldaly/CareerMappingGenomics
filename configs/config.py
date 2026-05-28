"""
CareerMappingGenomics — Central Configuration
All hyper-parameters, paths, and constants live here.

References
----------
[GEN-CFG] GENOMIC_CFG defaults:
  - n_snps=1,000 (simulation); production: 500,000–1,000,000 from Illumina GSA array.
    Ref: Chang et al. (2015). PLINK2. GigaScience. [GEN-08]
  - maf_threshold=0.01: standard MAF filter in GWAS QC.
    Ref: Choi et al. (2020). Nature Protocols, 15(9), 2759–2772. [GEN-03]
  - pgs_traits: Big Five + Autism + Intelligence + Education.
    Ref: Privé et al. (2020). LDpred2. Bioinformatics. [GEN-04]

[FAC-CFG] FACIAL_CFG defaults:
  - image_size=224: standard ResNet/ViT input resolution.
    Ref: He et al. (2016). Deep residual learning. CVPR. [DL-02]
  - n_landmarks=468: MediaPipe Face Mesh.
    Ref: Lugaresi et al. (2019). MediaPipe. arXiv:1906.08172. [FAC-03]
  - fer_classes: Ekman's 6 basic emotions + neutral.
    Ref: Ekman & Friesen (1978). FACS. Consulting Psychologists Press. [FAC-05]

[BIO-CFG] BIOMARKER_CFG defaults:
  - n_markers=80: CBC + hormones + metabolic + neurotransmitters + ASD-specific.
    Ref: NHANES (2023). CDC. [DAT-05]
  - marker_categories: 7 physiological categories.
    Ref: Masi et al. (2015). Molecular Psychiatry, 20(4), 440–446. [BIO-08]

[FUS-CFG] FUSION_CFG defaults:
  - n_attention_heads=4: standard for small-scale cross-modal attention.
    Ref: Xu et al. (2023). IEEE TPAMI, 45(10). [FUS-03]
  - fused_dim=256: joint embedding dimensionality.
    Ref: Baltrusaitis et al. (2018). IEEE TPAMI, 41(2), 423–443. [FUS-01]
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR        = ROOT / "data"
RAW_DIR         = DATA_DIR / "raw"
PROCESSED_DIR   = DATA_DIR / "processed"
SYNTHETIC_DIR   = DATA_DIR / "synthetic"
MODELS_DIR      = ROOT / "src" / "models"
REPORTS_DIR     = ROOT / "reports"

# ──────────────────────────────────────────────
# Big Five Traits
# ──────────────────────────────────────────────
BIG_FIVE = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
BIG_FIVE_SHORT = ["O", "C", "E", "A", "N"]

# ──────────────────────────────────────────────
# Genomic Configuration
# ──────────────────────────────────────────────
@dataclass
class GenomicConfig:
    n_snps: int = 1000            # Reduced for simulation; real: 500_000+
    n_chromosomes: int = 22
    maf_threshold: float = 0.01  # Minor allele frequency filter
    call_rate_threshold: float = 0.98
    hwe_p_threshold: float = 1e-6
    pgs_traits: List[str] = field(default_factory=lambda: BIG_FIVE + ["Autism_PGS", "Intelligence_PGS", "Education_PGS"])
    embedding_dim: int = 64

# ──────────────────────────────────────────────
# Facial Recognition Configuration
# ──────────────────────────────────────────────
@dataclass
class FacialConfig:
    image_size: int = 224
    n_landmarks: int = 468        # MediaPipe face mesh
    embedding_dim: int = 128
    cnn_backbone: str = "resnet18"
    fer_classes: List[str] = field(default_factory=lambda: [
        "neutral", "happy", "sad", "surprise", "fear", "disgust", "anger"
    ])
    dropout: float = 0.3

# ──────────────────────────────────────────────
# Blood Biomarker Configuration
# ──────────────────────────────────────────────
@dataclass
class BiomarkerConfig:
    n_markers: int = 80
    embedding_dim: int = 64
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    dropout: float = 0.25
    marker_categories: List[str] = field(default_factory=lambda: [
        "Hematology", "Hormones", "Metabolic",
        "Neurotransmitter_Precursors", "Inflammatory",
        "Nutritional", "Autism_Specific"
    ])

# ──────────────────────────────────────────────
# Fusion / Career Model Configuration
# ──────────────────────────────────────────────
@dataclass
class FusionConfig:
    genomic_dim: int = 64
    facial_dim: int = 128
    biomarker_dim: int = 64
    questionnaire_dim: int = 32
    fused_dim: int = 256
    n_attention_heads: int = 4
    dropout: float = 0.2
    n_careers: int = 50           # Top-N careers from O*NET

# ──────────────────────────────────────────────
# Training Configuration
# ──────────────────────────────────────────────
@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 10
    val_split: float = 0.2
    test_split: float = 0.1
    seed: int = 42
    device: str = "cpu"           # "cuda" if GPU available

# ──────────────────────────────────────────────
# Synthetic Data Generation
# ──────────────────────────────────────────────
@dataclass
class SyntheticConfig:
    n_samples: int = 2000
    autism_prevalence: float = 0.15   # ~1 in 7 in this research cohort
    seed: int = 42

# ──────────────────────────────────────────────
# Global config instance
# ──────────────────────────────────────────────
GENOMIC_CFG     = GenomicConfig()
FACIAL_CFG      = FacialConfig()
BIOMARKER_CFG   = BiomarkerConfig()
FUSION_CFG      = FusionConfig()
TRAIN_CFG       = TrainConfig()
SYNTHETIC_CFG   = SyntheticConfig()
