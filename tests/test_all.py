"""
Test Suite — CareerMappingGenomics
Tests all modules: models, data, inference, API.
"""

import sys
import numpy as np
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.config import GENOMIC_CFG, FACIAL_CFG, BIOMARKER_CFG, FUSION_CFG


def test_synthetic_data_generation():
    print("\n[TEST] Synthetic Data Generation...")
    from src.utils.synthetic_data import generate_dataset
    data = generate_dataset(n=50, save=False)
    assert data["snps"].shape       == (50, GENOMIC_CFG.n_snps)
    assert data["biomarkers"].shape == (50, BIOMARKER_CFG.n_markers)
    assert data["big_five"].shape   == (50, 5)
    assert data["facial_emb"].shape == (50, FACIAL_CFG.embedding_dim)
    assert data["career_labels"].shape == (50,)
    # Big Five must be in [0,1]
    assert data["big_five"].min() >= 0
    assert data["big_five"].max() <= 1
    print("  ✓ Shape checks passed")
    print(f"  ✓ ASD prevalence: {data['is_autism'].mean():.2%}")
    return data


def test_genomic_model():
    print("\n[TEST] Genomic Model (SNP-Transformer)...")
    from src.models.genomic_model import SNPTransformer, GenomicQCPipeline
    model = SNPTransformer()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    x = torch.randn(4, GENOMIC_CFG.n_snps)
    emb, pgs = model(x)
    assert emb.shape == (4, GENOMIC_CFG.embedding_dim)
    assert pgs.shape == (4, len(GENOMIC_CFG.pgs_traits))
    assert (pgs >= 0).all() and (pgs <= 1).all(), "PGS must be in [0,1]"
    print(f"  ✓ Output shapes: emb={emb.shape}, pgs={pgs.shape}")

    # QC pipeline
    snp_data = np.random.randint(0, 3, (100, GENOMIC_CFG.n_snps)).astype(np.float32)
    qc = GenomicQCPipeline()
    out = qc.fit_transform(snp_data)
    assert out.shape[0] == 100
    print(f"  ✓ QC pipeline: {snp_data.shape[1]} → {out.shape[1]} SNPs after MAF filter")


def test_facial_model():
    print("\n[TEST] Facial Model (FaceGenome-CNN)...")
    from src.models.facial_model import FaceGenomeCNN, FERInterpreter
    model = FaceGenomeCNN()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    x = torch.randn(4, FACIAL_CFG.embedding_dim)
    emb, bf, fer = model(x)
    assert emb.shape == (4, FACIAL_CFG.embedding_dim)
    assert bf.shape  == (4, 5)
    assert fer.shape == (4, len(FACIAL_CFG.fer_classes))
    assert (bf >= 0).all() and (bf <= 1).all()
    print(f"  ✓ Output shapes: emb={emb.shape}, big_five={bf.shape}, fer={fer.shape}")

    fer_result = FERInterpreter.interpret(fer[0].detach().numpy())
    assert "dominant_emotion" in fer_result
    print(f"  ✓ FER interpretation: dominant={fer_result['dominant_emotion']}")


def test_biomarker_model():
    print("\n[TEST] Biomarker Model (BiomarkerNet)...")
    from src.models.biomarker_model import BiomarkerNet, BiomarkerNormalizer
    model = BiomarkerNet()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    x = torch.randn(4, BIOMARKER_CFG.n_markers)
    emb, asd, bf, attn = model(x)
    assert emb.shape  == (4, BIOMARKER_CFG.embedding_dim)
    assert asd.shape  == (4, 1)
    assert bf.shape   == (4, 5)
    assert attn.shape[1] == 7  # n_categories
    assert (asd >= 0).all() and (asd <= 1).all()
    print(f"  ✓ Output shapes: emb={emb.shape}, asd={asd.shape}, attn={attn.shape}")

    # Normalizer
    data = np.random.randn(200, BIOMARKER_CFG.n_markers) * 50 + 100
    norm = BiomarkerNormalizer()
    out  = norm.fit_transform(data)
    assert out.shape == data.shape
    assert abs(out.mean()) < 0.2
    print(f"  ✓ Normalizer: mean={out.mean():.3f}, std={out.std():.3f}")


def test_fusion_model():
    print("\n[TEST] Fusion Model (FusionNet)...")
    from src.models.fusion_model import FusionNet
    model = FusionNet()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    g = torch.randn(4, FUSION_CFG.genomic_dim)
    f = torch.randn(4, FUSION_CFG.facial_dim)
    b = torch.randn(4, FUSION_CFG.biomarker_dim)
    q = torch.randn(4, FUSION_CFG.questionnaire_dim)

    fused, bf, careers, asd, mod_w = model(g, f, b, q)
    assert bf.shape     == (4, 5)
    assert asd.shape    == (4, 1)
    assert mod_w.shape  == (4, 4)
    assert abs(mod_w.sum(dim=-1).mean().item() - 1.0) < 0.01  # Softmax sums to 1
    print(f"  ✓ Career scores: {careers.shape}, mod_weights sum: {mod_w.sum(dim=-1).mean():.3f}")


def test_full_system():
    print("\n[TEST] Full CareerMappingSystem...")
    from src.models.genomic_model import SNPTransformer
    from src.models.facial_model import FaceGenomeCNN
    from src.models.biomarker_model import BiomarkerNet
    from src.models.fusion_model import FusionNet, CareerMappingSystem

    system = CareerMappingSystem(SNPTransformer(), FaceGenomeCNN(), BiomarkerNet(), FusionNet())
    total  = sum(p.numel() for p in system.parameters())
    print(f"  Total parameters: {total:,}")

    snps    = torch.randn(2, GENOMIC_CFG.n_snps)
    face    = torch.randn(2, FACIAL_CFG.embedding_dim)
    bio     = torch.randn(2, BIOMARKER_CFG.n_markers)
    quest   = torch.randn(2, 32)

    out = system(snps, face, bio, quest)
    assert out["personality"].shape[1]   == 5
    assert out["asd_score"].shape[1]     == 1
    assert 0 <= out["asd_score"].mean().item() <= 1
    print(f"  ✓ Forward pass: personality={out['personality'].shape}, asd={out['asd_score'].shape}")


def test_inference_engine():
    print("\n[TEST] Inference Engine...")
    from src.pipelines.inference import InferenceEngine
    from src.utils.synthetic_data import (
        generate_snps, generate_biomarkers, generate_big_five, generate_facial_embeddings
    )
    rng = np.random.default_rng(7)
    is_autism = np.array([False])
    snps       = generate_snps(1, is_autism)[0]
    biomarkers = generate_biomarkers(1, is_autism)[0]
    big_five   = generate_big_five(1, is_autism)[0]
    facial_emb = generate_facial_embeddings(1, big_five.reshape(1,-1))[0]
    quest      = np.hstack([big_five, rng.normal(0, 0.1, 27)]).astype(np.float32)

    engine = InferenceEngine()
    report = engine.predict(snps, facial_emb, biomarkers, quest, top_k=5)

    assert len(report["top_careers"]) == 5
    assert "personality_profile" in report
    assert "asd_assessment" in report
    for trait in ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]:
        assert trait in report["personality_profile"]
    print(f"  ✓ Report generated with {len(report['top_careers'])} careers")
    print(f"  ✓ Top career: {report['top_careers'][0]['title']}")


def test_dataset_pipeline():
    print("\n[TEST] Dataset & DataLoader pipeline...")
    from src.utils.synthetic_data import generate_dataset
    from src.pipelines.dataset import get_dataloaders
    from configs.config import TrainConfig

    data = generate_dataset(n=120, save=False)
    cfg  = TrainConfig(batch_size=16)
    train_loader, val_loader, test_loader = get_dataloaders(data, cfg)
    batch = next(iter(train_loader))
    assert "snps"         in batch
    assert "biomarkers"   in batch
    assert "career_label" in batch
    assert batch["snps"].shape[-1] == GENOMIC_CFG.n_snps
    print(f"  ✓ Batch keys: {list(batch.keys())}")
    print(f"  ✓ SNPs batch shape: {batch['snps'].shape}")


def test_career_database():
    print("\n[TEST] Career Database...")
    from src.utils.career_database import (
        CAREER_DATABASE, get_autism_friendly_careers, get_career_by_id
    )
    assert len(CAREER_DATABASE) > 0
    autism_careers = get_autism_friendly_careers()
    assert len(autism_careers) > 0
    career = get_career_by_id("C001")
    assert career is not None
    assert career["title"] == "Software Engineer"
    print(f"  ✓ {len(CAREER_DATABASE)} careers | {len(autism_careers)} autism-strength careers")


def test_visualization():
    print("\n[TEST] Visualization module...")
    from src.utils.synthetic_data import generate_dataset
    from src.pipelines.inference import InferenceEngine
    from src.utils.visualization import (
        plot_bigfive_distributions, plot_biomarker_comparison,
        plot_career_domain_overview, plot_modality_contributions
    )
    import numpy as np
    data = generate_dataset(n=100, save=False)
    p1 = plot_bigfive_distributions(data["big_five"], data["is_autism"].astype(int))
    p2 = plot_biomarker_comparison(data["biomarkers"], data["is_autism"].astype(int))
    p3 = plot_career_domain_overview()
    dummy_contribs = {"Genomics": 0.28, "Facial": 0.22, "Biomarkers": 0.30, "Questionnaire": 0.20}
    p4 = plot_modality_contributions(dummy_contribs)
    for p in [p1, p2, p3, p4]:
        assert Path(p).exists()
    print(f"  ✓ 4 plots generated successfully")


# ──────────────────────────────────────────────────────────────────────
# Test runner
# ──────────────────────────────────────────────────────────────────────

def run_all_tests():
    tests = [
        test_career_database,
        test_synthetic_data_generation,
        test_genomic_model,
        test_facial_model,
        test_biomarker_model,
        test_fusion_model,
        test_full_system,
        test_dataset_pipeline,
        test_inference_engine,
        test_visualization,
    ]

    passed = 0
    failed = 0
    failures = []

    print("\n" + "="*65)
    print("  CareerMappingGenomics — Full Test Suite")
    print("="*65)

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            failures.append((test.__name__, str(e)))
            print(f"  ✗ FAILED: {e}")

    print("\n" + "="*65)
    print(f"  Results: {passed} passed, {failed} failed")
    if failures:
        print("\n  Failures:")
        for name, err in failures:
            print(f"    - {name}: {err}")
    print("="*65)
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
