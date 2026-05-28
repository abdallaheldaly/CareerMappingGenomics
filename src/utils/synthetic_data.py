"""
Synthetic Data Generator
Produces realistic multi-modal training data:
  - SNP genotype matrices
  - Blood biomarker panels
  - Facial feature embeddings (simulated landmarks)
  - Questionnaire Big Five scores
  - Career outcome labels

References
----------
[DAT-01] Sudlow et al. (2015). UK Biobank: An open access resource.
         PLOS Medicine, 12(3), e1001779.
         https://doi.org/10.1371/journal.pmed.1001779
         → UK Biobank population statistics inform synthetic SNP allele frequency distributions.

[DAT-02] Feliciano et al. (2018). SPARK: A US cohort of 50,000 families to accelerate
         autism research. Neuron, 97(3), 488–493.
         https://doi.org/10.1016/j.neuron.2018.01.015
         → SPARK ASD genetics; informs autism SNP enrichment simulation.

[DAT-05] Centers for Disease Control and Prevention (2023). NHANES documentation.
         https://www.cdc.gov/nchs/nhanes/index.htm
         → NHANES biomarker reference ranges inform all 80 (mean, std) specifications.

[ASD-03] Hope et al. (2023). Bidirectional genetic overlap between ASD and cognitive traits.
         Translational Psychiatry, 13(1), 290.
         https://doi.org/10.1038/s41398-023-02563-7
         → 12,000 ASD–intelligence shared SNPs; basis for autism SNP enrichment in generate_snps().

[PSY-03] Jang et al. (1996). Heritability of the Big Five personality dimensions.
         Journal of Personality, 64(3), 577–591.
         https://doi.org/10.1111/j.1467-6494.1996.tb00522.x
         → 40–60% heritability; justifies genetic component of Big Five simulation.

[BIO-06] Edmiston & Corbett (2016). Biobehavioral profiles of arousal in ASD.
         https://doi.org/10.1007/s10803-016-2871-z
         → ASD Big Five profile adjustments (lower E, lower A, higher O) in generate_big_five().

[GEN-01] Visscher et al. (2017). 10 years of GWAS discovery.
         American Journal of Human Genetics, 101(1), 5–22.
         https://doi.org/10.1016/j.ajhg.2017.06.005
         → Hardy-Weinberg equilibrium model used for SNP genotype generation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from configs.config import GENOMIC_CFG, BIOMARKER_CFG, FACIAL_CFG, SYNTHETIC_CFG, BIG_FIVE, SYNTHETIC_DIR

rng = np.random.default_rng(SYNTHETIC_CFG.seed)

# ─────────────────────────────────────────────────────────────────────
# Biomarker reference ranges (mean, std) — clinically grounded
# ─────────────────────────────────────────────────────────────────────
BIOMARKER_SPECS = {
    # Hematology
    "hemoglobin_g_dl":       (14.0, 1.5),
    "wbc_count_k_ul":        (7.0, 2.0),
    "platelet_count_k_ul":   (250.0, 60.0),
    "hematocrit_pct":        (42.0, 4.0),
    "mcv_fl":                (90.0, 8.0),
    # Hormones
    "cortisol_ug_dl":        (15.0, 6.0),
    "testosterone_ng_dl":    (400.0, 150.0),
    "estradiol_pg_ml":       (50.0, 30.0),
    "dhea_s_ug_dl":          (200.0, 80.0),
    "tsh_miu_l":             (2.0, 1.0),
    "free_t3_pg_ml":         (3.0, 0.5),
    "free_t4_ng_dl":         (1.2, 0.2),
    # Metabolic
    "glucose_mg_dl":         (95.0, 15.0),
    "hba1c_pct":             (5.2, 0.5),
    "insulin_uiu_ml":        (8.0, 4.0),
    "total_cholesterol":     (185.0, 35.0),
    "hdl_mg_dl":             (55.0, 15.0),
    "ldl_mg_dl":             (110.0, 30.0),
    "triglycerides_mg_dl":   (120.0, 50.0),
    "uric_acid_mg_dl":       (5.5, 1.5),
    # Neurotransmitter precursors
    "tryptophan_umol_l":     (55.0, 15.0),
    "tyrosine_umol_l":       (65.0, 18.0),
    "phenylalanine_umol_l":  (58.0, 16.0),
    "gaba_nmol_ml":          (0.4, 0.15),
    "glutamate_nmol_ml":     (2.5, 0.8),
    "serotonin_ng_ml":       (160.0, 60.0),
    "dopamine_pg_ml":        (25.0, 10.0),
    # Inflammatory
    "hs_crp_mg_l":           (1.5, 1.2),
    "il6_pg_ml":             (2.5, 1.5),
    "tnf_alpha_pg_ml":       (3.0, 1.5),
    "homocysteine_umol_l":   (10.0, 3.0),
    # Nutritional
    "vitamin_d_ng_ml":       (30.0, 12.0),
    "b12_pg_ml":             (400.0, 150.0),
    "folate_ng_ml":          (12.0, 5.0),
    "ferritin_ng_ml":        (80.0, 50.0),
    "iron_ug_dl":            (100.0, 30.0),
    "omega3_index_pct":      (5.5, 1.5),
    # Autism-specific
    "oxytocin_pg_ml":        (20.0, 8.0),
    "avp_pg_ml":             (4.0, 1.5),
    "bdnf_ng_ml":            (25.0, 8.0),
    "gaba_glutamate_ratio":  (0.16, 0.06),
    # Additional metabolic
    "albumin_g_dl":          (4.2, 0.3),
    "creatinine_mg_dl":      (0.95, 0.2),
    "bun_mg_dl":             (15.0, 5.0),
    "sodium_meq_l":          (140.0, 3.0),
    "potassium_meq_l":       (4.2, 0.4),
    "calcium_mg_dl":         (9.5, 0.5),
    "magnesium_mg_dl":       (2.0, 0.3),
    "phosphorus_mg_dl":      (3.5, 0.5),
    "ast_u_l":               (25.0, 10.0),
    "alt_u_l":               (28.0, 12.0),
    "alkaline_phosphatase":  (75.0, 25.0),
    "bilirubin_total_mg_dl": (0.8, 0.3),
    # Hormones continued
    "prolactin_ng_ml":       (10.0, 5.0),
    "igf1_ng_ml":            (180.0, 60.0),
    "insulin_growth_factor": (150.0, 50.0),
    "leptin_ng_ml":          (12.0, 6.0),
    "ghrelin_pg_ml":         (600.0, 200.0),
    "adiponectin_ug_ml":     (10.0, 4.0),
    # Immune
    "cd4_count":             (800.0, 200.0),
    "cd8_count":             (450.0, 150.0),
    "nk_cells_pct":          (12.0, 5.0),
    "iga_mg_dl":             (200.0, 80.0),
    "igg_mg_dl":             (1200.0, 300.0),
    # Additional neurological
    "s100b_ng_ml":           (0.08, 0.03),
    "nf_light_pg_ml":        (8.0, 4.0),
    "tau_protein_pg_ml":     (200.0, 80.0),
    "amyloid_beta_42_pg_ml": (900.0, 200.0),
    "nse_ng_ml":             (8.0, 3.0),
    # Oxidative stress
    "malondialdehyde_nmol_ml": (1.5, 0.5),
    "glutathione_umol_l":    (800.0, 200.0),
    "superoxide_dismutase":  (1500.0, 400.0),
    "catalase_nmol_min_ml":  (60.0, 15.0),
    # Gut-brain axis
    "short_chain_fatty_acids": (2.5, 0.8),
    "butyrate_mmol_l":       (0.4, 0.15),
    "propionate_mmol_l":     (0.3, 0.12),
    "lipopolysaccharide_eu_ml": (0.05, 0.02),
    # Remaining
    "ceruloplasmin_mg_dl":   (30.0, 8.0),
    "zinc_ug_dl":            (90.0, 20.0),
    "copper_ug_dl":          (110.0, 25.0),
    "selenium_ng_ml":        (110.0, 25.0),
    "coq10_ug_ml":           (0.8, 0.3),
    "carnitine_umol_l":      (40.0, 12.0),
    "melatonin_pg_ml":       (25.0, 12.0),
    "cortisol_dhea_ratio":   (0.08, 0.03),
}

BIOMARKER_NAMES = list(BIOMARKER_SPECS.keys())
assert len(BIOMARKER_NAMES) >= BIOMARKER_CFG.n_markers, \
    f"Need {BIOMARKER_CFG.n_markers} markers, got {len(BIOMARKER_NAMES)}"
BIOMARKER_NAMES = BIOMARKER_NAMES[:BIOMARKER_CFG.n_markers]


def _autism_biomarker_shift(base: np.ndarray, is_autism: bool) -> np.ndarray:
    """Apply autism-associated biomarker deviations (literature-informed)."""
    if not is_autism:
        return base
    shifted = base.copy()
    names = BIOMARKER_NAMES
    mods = {
        "oxytocin_pg_ml":       -0.30,  # lower in ASD
        "serotonin_ng_ml":      -0.20,
        "gaba_glutamate_ratio": -0.25,
        "glutamate_nmol_ml":    +0.25,  # elevated in ASD
        "hs_crp_mg_l":          +0.35,
        "il6_pg_ml":            +0.40,
        "bdnf_ng_ml":           -0.20,
        "cortisol_ug_dl":       +0.20,
        "vitamin_d_ng_ml":      -0.25,
        "melatonin_pg_ml":      -0.30,
        "avp_pg_ml":            -0.20,
        "zinc_ug_dl":           -0.15,
    }
    for name, z_delta in mods.items():
        if name in names:
            idx = names.index(name)
            _, std = BIOMARKER_SPECS[name]
            shifted[idx] += z_delta * std
    return shifted


def generate_biomarkers(n: int, is_autism: np.ndarray) -> np.ndarray:
    """Generate (n, 80) blood biomarker matrix."""
    data = np.zeros((n, BIOMARKER_CFG.n_markers))
    for i, name in enumerate(BIOMARKER_NAMES):
        mean, std = BIOMARKER_SPECS[name]
        data[:, i] = rng.normal(mean, std, n)
    # Apply autism-specific shifts
    for idx in range(n):
        if is_autism[idx]:
            data[idx] = _autism_biomarker_shift(data[idx], True)
    # Clip to realistic ranges (no negatives for concentrations)
    data = np.clip(data, 0, None)
    return data


def generate_snps(n: int, is_autism: np.ndarray) -> np.ndarray:
    """
    Generate (n, n_snps) SNP matrix with genotype values {0,1,2}.
    Includes autism-associated SNP enrichment.
    """
    n_snps = GENOMIC_CFG.n_snps
    # Base population MAF for each SNP (uniform 0.05–0.50)
    maf = rng.uniform(0.05, 0.50, n_snps)

    # Draw genotypes under Hardy-Weinberg equilibrium
    p = maf
    genotypes = np.zeros((n, n_snps), dtype=np.float32)
    for j in range(n_snps):
        q = 1 - p[j]
        probs = [p[j]**2, 2*p[j]*q, q**2]  # AA, Aa, aa
        genotypes[:, j] = rng.choice([2, 1, 0], size=n, p=probs)

    # Autism PGS: shift ~50 "risk" SNPs upward for ASD individuals
    asd_snp_indices = rng.integers(0, n_snps, 50)
    for idx in range(n):
        if is_autism[idx]:
            for snp_j in asd_snp_indices:
                genotypes[idx, snp_j] = min(2, genotypes[idx, snp_j] + rng.integers(0, 2))

    return genotypes


def generate_big_five(n: int, is_autism: np.ndarray) -> np.ndarray:
    """
    Generate (n, 5) Big Five scores [0, 1].
    Autism-associated profile: higher O, lower E, lower A.
    """
    # Population base: mean 0.5, std 0.15
    scores = rng.normal(0.5, 0.15, (n, 5))

    for idx in range(n):
        if is_autism[idx]:
            # Typical ASD Big Five profile adjustments
            scores[idx, 0] += 0.08   # Openness: higher
            scores[idx, 1] += 0.05   # Conscientiousness: slightly higher
            scores[idx, 2] -= 0.12   # Extraversion: lower
            scores[idx, 3] -= 0.08   # Agreeableness: lower
            scores[idx, 4] += 0.05   # Neuroticism: slightly higher

    return np.clip(scores, 0.01, 0.99)


def generate_facial_embeddings(n: int, big_five: np.ndarray) -> np.ndarray:
    """
    Simulate facial landmark embeddings correlated with Big Five traits.
    In production these come from the CNN processing actual face images.
    Shape: (n, facial_embedding_dim)
    """
    # Base embedding (random noise representing face-specific variation)
    embeddings = rng.normal(0, 1, (n, FACIAL_CFG.embedding_dim))
    # Add weak correlation with Big Five (Pearson r ~ 0.3)
    trait_proj = rng.normal(0, 1, (5, FACIAL_CFG.embedding_dim))
    embeddings += 0.30 * (big_five @ trait_proj)
    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-8)


def _assign_career_label(big_five: np.ndarray, intelligence: float, is_autism: bool) -> int:
    """Assign a career label index based on trait profile similarity."""
    from src.utils.career_database import CAREER_DATABASE
    o, c, e, a, n = big_five

    scores = []
    for career in CAREER_DATABASE:
        # Weighted trait match
        diff = (
            abs(o - career["O"]) +
            abs(c - career["C"]) +
            abs(e - career["E"]) +
            abs(a - career["A"]) +
            abs(n - career["N"]) +
            abs(intelligence - career["intelligence"])
        )
        # Boost autism-friendly careers for ASD individuals
        boost = -0.15 if (is_autism and career["autism_strength"]) else 0
        scores.append(diff + boost)

    return int(np.argmin(scores))


def generate_dataset(n: int = None, save: bool = True) -> dict:
    """Generate complete multi-modal dataset."""
    n = n or SYNTHETIC_CFG.n_samples
    print(f"[SyntheticGen] Generating {n} samples...")

    # Autism label
    is_autism = rng.random(n) < SYNTHETIC_CFG.autism_prevalence
    # Intelligence score (correlated with education PGS)
    intelligence = np.clip(rng.normal(0.65, 0.15, n), 0, 1)

    # Generate each modality
    snps         = generate_snps(n, is_autism)
    biomarkers   = generate_biomarkers(n, is_autism)
    big_five     = generate_big_five(n, is_autism)
    facial_emb   = generate_facial_embeddings(n, big_five)

    # Career labels
    career_labels = np.array([
        _assign_career_label(big_five[i], intelligence[i], is_autism[i])
        for i in range(n)
    ])

    # Compute polygenic scores (simplified: weighted sum of SNPs)
    pgs_weights = rng.normal(0, 0.1, (GENOMIC_CFG.n_snps, len(GENOMIC_CFG.pgs_traits)))
    pgs_scores  = snps @ pgs_weights  # (n, n_pgs)

    # Questionnaire embedding (low-dim Big Five padded to 32-d)
    q_emb = np.hstack([big_five, rng.normal(0, 0.1, (n, 27))])  # pad to 32

    # Metadata DataFrame
    meta = pd.DataFrame({
        "sample_id":    [f"S{i:05d}" for i in range(n)],
        "is_autism":    is_autism.astype(int),
        "intelligence": intelligence,
        "career_label": career_labels,
        "O": big_five[:, 0], "C": big_five[:, 1],
        "E": big_five[:, 2], "A": big_five[:, 3], "N": big_five[:, 4],
    })

    dataset = {
        "snps":          snps.astype(np.float32),
        "biomarkers":    biomarkers.astype(np.float32),
        "big_five":      big_five.astype(np.float32),
        "facial_emb":    facial_emb.astype(np.float32),
        "pgs_scores":    pgs_scores.astype(np.float32),
        "questionnaire": q_emb.astype(np.float32),
        "career_labels": career_labels.astype(np.int64),
        "is_autism":     is_autism.astype(np.float32),
        "intelligence":  intelligence.astype(np.float32),
        "metadata":      meta,
    }

    if save:
        SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
        np.save(SYNTHETIC_DIR / "snps.npy",          dataset["snps"])
        np.save(SYNTHETIC_DIR / "biomarkers.npy",    dataset["biomarkers"])
        np.save(SYNTHETIC_DIR / "big_five.npy",      dataset["big_five"])
        np.save(SYNTHETIC_DIR / "facial_emb.npy",    dataset["facial_emb"])
        np.save(SYNTHETIC_DIR / "pgs_scores.npy",    dataset["pgs_scores"])
        np.save(SYNTHETIC_DIR / "questionnaire.npy", dataset["questionnaire"])
        np.save(SYNTHETIC_DIR / "career_labels.npy", dataset["career_labels"])
        np.save(SYNTHETIC_DIR / "is_autism.npy",     dataset["is_autism"])
        meta.to_csv(SYNTHETIC_DIR / "metadata.csv", index=False)
        print(f"[SyntheticGen] Saved to {SYNTHETIC_DIR}")

    # Summary
    autism_count = is_autism.sum()
    print(f"[SyntheticGen] Samples: {n} | ASD: {autism_count} ({100*autism_count/n:.1f}%)")
    print(f"[SyntheticGen] SNPs: {snps.shape} | Biomarkers: {biomarkers.shape}")
    print(f"[SyntheticGen] Big Five: {big_five.shape} | Facial: {facial_emb.shape}")
    return dataset


if __name__ == "__main__":
    generate_dataset()
