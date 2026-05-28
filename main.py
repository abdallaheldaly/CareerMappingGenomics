"""
CareerMappingGenomics — Main Entry Point
Runs the complete pipeline:
  1. Generate synthetic dataset
  2. Train all models
  3. Run inference on a demo individual
  4. Generate all visualizations
  5. Print final report
"""

import sys
import json
import time
import numpy as np
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from configs.config import (
    TRAIN_CFG, REPORTS_DIR, MODELS_DIR, TrainConfig, SyntheticConfig
)
from src.utils.synthetic_data import generate_dataset
from src.pipelines.trainer import run_training
from src.pipelines.inference import InferenceEngine, demo_inference
from src.utils.visualization import generate_all_plots
from src.utils.career_database import CAREER_DATABASE


def print_header():
    print("""
╔══════════════════════════════════════════════════════════════╗
║         CareerMappingGenomics — Deep Learning System         ║
║  Genomics · Facial Recognition · Blood Biomarkers · AI       ║
║  Author: Abdallah El-Daly | Research Prototype v1.0          ║
╚══════════════════════════════════════════════════════════════╝
""")


def print_system_summary():
    from src.models.genomic_model import SNPTransformer
    from src.models.facial_model import FaceGenomeCNN
    from src.models.biomarker_model import BiomarkerNet
    from src.models.fusion_model import FusionNet, CareerMappingSystem
    from configs.config import GENOMIC_CFG, FACIAL_CFG, BIOMARKER_CFG

    gm = SNPTransformer()
    fm = FaceGenomeCNN()
    bm = BiomarkerNet()
    fn = FusionNet()
    total = (
        sum(p.numel() for p in gm.parameters()) +
        sum(p.numel() for p in fm.parameters()) +
        sum(p.numel() for p in bm.parameters()) +
        sum(p.numel() for p in fn.parameters())
    )

    print("─" * 60)
    print("  SYSTEM CONFIGURATION")
    print("─" * 60)
    print(f"  Genomic model (SNP-Transformer):")
    print(f"    Input SNPs:        {GENOMIC_CFG.n_snps:,}")
    print(f"    PGS traits:        {len(GENOMIC_CFG.pgs_traits)}")
    print(f"    Parameters:        {sum(p.numel() for p in gm.parameters()):,}")
    print(f"  Facial model (FaceGenome-CNN):")
    print(f"    Embedding dim:     {FACIAL_CFG.embedding_dim}")
    print(f"    FER classes:       {len(FACIAL_CFG.fer_classes)}")
    print(f"    Parameters:        {sum(p.numel() for p in fm.parameters()):,}")
    print(f"  Biomarker model (BiomarkerNet):")
    print(f"    Blood markers:     {BIOMARKER_CFG.n_markers}")
    print(f"    Parameters:        {sum(p.numel() for p in bm.parameters()):,}")
    print(f"  FusionNet (cross-modal attention):")
    print(f"    Parameters:        {sum(p.numel() for p in fn.parameters()):,}")
    print(f"  Career database:     {len(CAREER_DATABASE)} careers")
    print(f"  ─────────────────────────────────────────────")
    print(f"  Total parameters:    {total:,}")
    print()


def print_career_report(report: dict):
    print("\n" + "═"*65)
    print("  INDIVIDUAL CAREER MAPPING REPORT")
    print("═"*65)

    print("\n  PERSONALITY PROFILE:")
    print("  " + "─"*50)
    for trait, info in report["personality_profile"].items():
        bar_len = int(info["score"] * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {trait:20s} [{bar}] {info['score']:.3f}  ({info['level']})")

    print(f"\n  ASD ASSESSMENT:")
    asd = report["asd_assessment"]
    bar_len = int(asd["probability_pct"] / 100 * 30)
    bar = "█" * bar_len + "░" * (30 - bar_len)
    print(f"  [{bar}] {asd['probability_pct']:.1f}%")
    print(f"  Classification: {asd['tier']}")

    print(f"\n  TOP CAREER MATCHES:")
    print("  " + "─"*60)
    print(f"  {'Rank':>4}  {'★':1}  {'Career Title':<32}  {'Match':>6}  {'Domain'}")
    print("  " + "─"*60)
    for c in report["top_careers"][:10]:
        star = "★" if c["autism_strength"] else " "
        print(f"  {c['rank']:>4}.  {star}  {c['title']:<32}  {c['compatibility_pct']:>5.1f}%  {c['domain']}")

    print(f"\n  MODALITY CONTRIBUTIONS:")
    for mod, weight in report["modality_contributions"].items():
        bar_len = int(weight * 50)
        bar = "█" * bar_len
        print(f"  {mod:15s} {bar} {weight:.3f}")

    print(f"\n  FACIAL EMOTION (dominant): {report['facial_emotion']['dominant_emotion']}")

    if report["top_careers"]:
        top = report["top_careers"][0]
        print(f"\n  TOP RECOMMENDATION: {top['title']}")
        print(f"  Description: {top['description']}")
        if top["autism_advantages"]:
            print(f"  Autism Strengths Applied:")
            for adv in top["autism_advantages"]:
                print(f"    • {adv}")
        print(f"  Education Path: {top['education']}")

    print(f"\n  ⚠  DISCLAIMER: {report['disclaimer'][:150]}...")
    print("═"*65)


def run_full_pipeline(
    n_samples: int = 1000,
    epochs: int = 20,
    quick_mode: bool = False
):
    """Run complete CareerMappingGenomics pipeline."""
    t_start = time.time()
    print_header()
    print_system_summary()

    if quick_mode:
        n_samples = 300
        epochs    = 5
        print("  [Quick mode: reduced samples and epochs for demonstration]\n")

    # ── 1. Generate data ──────────────────────────────────────────────
    print("─" * 60)
    print("  STEP 1: Generating Synthetic Multi-Modal Dataset")
    print("─" * 60)
    data = generate_dataset(n=n_samples, save=True)

    # ── 2. Train ──────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  STEP 2: Training Deep Learning Models")
    print("─" * 60)
    cfg = TrainConfig(epochs=epochs, batch_size=32)
    system, trainer, history, test_metrics = run_training(data, cfg)

    print(f"\n  Final Test Metrics:")
    for k, v in test_metrics.items():
        print(f"    {k}: {v}")

    # ── 3. Inference demo ─────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  STEP 3: Running Inference on Demo Individual (ASD Profile)")
    print("─" * 60)
    engine = InferenceEngine()
    # Fit preprocessors on training data
    engine.fit_preprocessors(data["snps"], data["biomarkers"])

    from src.utils.synthetic_data import (
        generate_snps, generate_biomarkers,
        generate_big_five, generate_facial_embeddings
    )
    rng = np.random.default_rng(42)
    is_autism_demo = np.array([True])
    demo_snps   = generate_snps(1, is_autism_demo)[0]
    demo_bio    = generate_biomarkers(1, is_autism_demo)[0]
    demo_bf     = generate_big_five(1, is_autism_demo)[0]
    demo_face   = generate_facial_embeddings(1, demo_bf.reshape(1,-1))[0]
    demo_quest  = np.hstack([demo_bf, rng.normal(0, 0.1, 27)]).astype(np.float32)

    report = engine.predict(demo_snps, demo_face, demo_bio, demo_quest, top_k=10)
    print_career_report(report)

    # Save report to JSON
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "sample_report.json", "w") as f:
        # Convert numpy types for serialization
        def convert(o):
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            raise TypeError(f"Not serializable: {type(o)}")
        json.dump(report, f, indent=2, default=convert)

    # ── 4. Visualizations ─────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  STEP 4: Generating Visualizations")
    print("─" * 60)
    generate_all_plots(data, history, report)

    # ── 5. Summary ────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print("\n" + "═"*65)
    print(f"  PIPELINE COMPLETE  ({elapsed:.1f}s)")
    print("═"*65)
    print(f"  Dataset:       {n_samples} synthetic individuals")
    print(f"  Training:      {epochs} epochs")
    print(f"  Best val loss: {trainer.best_val_loss:.4f}")
    print(f"  Test accuracy: {test_metrics.get('career_acc', 'N/A')}")
    print(f"  Reports:       {REPORTS_DIR}")
    print(f"  Models:        {MODELS_DIR}")
    print("═"*65)

    return system, report, history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CareerMappingGenomics Pipeline")
    parser.add_argument("--quick",   action="store_true", help="Quick mode (300 samples, 5 epochs)")
    parser.add_argument("--samples", type=int, default=1000, help="Number of synthetic samples")
    parser.add_argument("--epochs",  type=int, default=20, help="Training epochs")
    args = parser.parse_args()

    run_full_pipeline(
        n_samples=args.samples,
        epochs=args.epochs,
        quick_mode=args.quick,
    )
