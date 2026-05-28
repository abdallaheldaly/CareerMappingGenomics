"""
Inference Engine
Loads trained models and runs predictions on a single individual.
Handles preprocessing, model inference, and report generation.

References
----------
[PSY-01] Costa & McCrae (1992). Revised NEO personality inventory (NEO PI-R).
         Psychological Assessment Resources.
         → Big Five trait scores [0,1] outputted in standard NEO PI-R domain order:
           Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism.

[CAR-01] Peterson et al. (1999). O*NET occupational information system.
         American Psychological Association.
         → Career compatibility scoring via cosine similarity against O*NET-derived
           career prototype vectors.

[GEN-03] Choi et al. (2020). Tutorial: A guide to performing polygenic risk score analyses.
         Nature Protocols, 15(9), 2759–2772. https://doi.org/10.1038/s41596-020-0353-1
         → PGS normalization and interpretation guidelines applied in InferenceEngine.

[ETH-09] Wachter et al. (2017). Counterfactual explanations without opening the black box.
         Harvard Journal of Law & Technology, 31(2), 841–887.
         → GDPR explainability requirements motivate the structured career report format
           with per-trait compatibility scores and confidence disclaimers.

[GEN-07] Marquez-Luna et al. (2017). Multiethnic polygenic risk scores.
         Genetic Epidemiology, 41(8), 811–823. https://doi.org/10.1002/gepi.22083
         → Cross-ancestry PGS limitations noted in disclaimer text of build_career_report().
"""

import torch
import numpy as np
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from configs.config import MODELS_DIR, GENOMIC_CFG, FACIAL_CFG, BIOMARKER_CFG, BIG_FIVE
from src.models.genomic_model import SNPTransformer, GenomicQCPipeline
from src.models.facial_model import FaceGenomeCNN, FERInterpreter
from src.models.biomarker_model import BiomarkerNet, BiomarkerNormalizer
from src.models.fusion_model import FusionNet, CareerMappingSystem, build_career_report
from src.utils.career_database import CAREER_DATABASE


class InferenceEngine:
    """
    High-level inference API.
    Wraps the trained CareerMappingSystem with preprocessing + report generation.
    """

    def __init__(self, model_path: Path = None, device: str = "cpu"):
        self.device = device
        self.model_path = model_path or (MODELS_DIR / "best_model.pt")

        # Build models
        self.system = self._build_system()

        # Preprocessors
        self.genomic_qc       = GenomicQCPipeline()
        self.biomarker_norm   = BiomarkerNormalizer()
        self._preprocessors_fitted = False

        # Try to load trained weights
        self._load_weights()

    def _build_system(self) -> CareerMappingSystem:
        gm = SNPTransformer()
        fm = FaceGenomeCNN()
        bm = BiomarkerNet()
        fn = FusionNet()
        system = CareerMappingSystem(gm, fm, bm, fn).to(self.device)
        system.eval()
        return system

    def _load_weights(self):
        if self.model_path.exists():
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.system.load_state_dict(checkpoint["system_state"])
            print(f"[Inference] Loaded weights from {self.model_path}")
            print(f"[Inference] Checkpoint epoch: {checkpoint.get('epoch','?')}")
        else:
            print(f"[Inference] No checkpoint found — using random weights (for demo)")

    def fit_preprocessors(self, snps_data: np.ndarray, biomarker_data: np.ndarray):
        """Fit normalization pipelines on training data."""
        self.genomic_qc.fit(snps_data)
        self.biomarker_norm.fit(biomarker_data)
        self._preprocessors_fitted = True
        print("[Inference] Preprocessors fitted.")

    def _preprocess_snps(self, snps: np.ndarray) -> np.ndarray:
        if self._preprocessors_fitted and self.genomic_qc.maf_mask is not None:
            try:
                filtered = self.genomic_qc.transform(snps)
                # Pad or truncate to n_snps
                n_snps = GENOMIC_CFG.n_snps
                if filtered.shape[1] >= n_snps:
                    return filtered[:, :n_snps]
                else:
                    pad = np.zeros((filtered.shape[0], n_snps - filtered.shape[1]))
                    return np.hstack([filtered, pad])
            except Exception:
                pass
        return snps  # Pass through if not fitted

    def _preprocess_biomarkers(self, bio: np.ndarray) -> np.ndarray:
        if self._preprocessors_fitted:
            return self.biomarker_norm.transform(bio)
        # Simple z-score without fitted stats
        return (bio - bio.mean(axis=0)) / (bio.std(axis=0) + 1e-8)

    @torch.no_grad()
    def predict(
        self,
        snps:          np.ndarray,    # (1, n_snps) or (n_snps,)
        facial_emb:    np.ndarray,    # (1, 128) or (128,)
        biomarkers:    np.ndarray,    # (1, 80) or (80,)
        questionnaire: np.ndarray,    # (1, 32) or (32,)
        top_k:         int = 10,
    ) -> dict:
        """
        Run full inference pipeline for one individual.
        Returns structured career report dict.
        """
        # Ensure 2D
        def to2d(x):
            return x.reshape(1, -1) if x.ndim == 1 else x

        snps          = to2d(snps.astype(np.float32))
        facial_emb    = to2d(facial_emb.astype(np.float32))
        biomarkers    = to2d(biomarkers.astype(np.float32))
        questionnaire = to2d(questionnaire.astype(np.float32))

        # Preprocess
        snps       = self._preprocess_snps(snps)
        biomarkers = self._preprocess_biomarkers(biomarkers)

        # To tensors
        t = lambda x: torch.tensor(x, dtype=torch.float32).to(self.device)
        snps_t    = t(snps)
        face_t    = t(facial_emb)
        bio_t     = t(biomarkers)
        quest_t   = t(questionnaire)

        # Forward pass
        out = self.system(snps_t, face_t, bio_t, quest_t)

        # Extract numpy
        big_five      = out["personality"][0].cpu().numpy()
        career_scores = out["career_scores"][0].cpu().numpy()
        asd_score     = out["asd_score"][0].item()
        pgs_scores    = out["pgs_scores"][0].cpu().numpy()
        fer_logits    = out["fer_logits"][0].cpu().numpy()
        mod_weights   = out["modality_weights"][0].cpu().numpy()

        # Build report
        report = build_career_report(big_five, career_scores, asd_score, pgs_scores, top_k)

        # Add FER interpretation
        report["facial_emotion"] = FERInterpreter.interpret(fer_logits)

        # Add modality weights
        report["modality_contributions"] = {
            "Genomics":      round(float(mod_weights[0]), 3),
            "Facial":        round(float(mod_weights[1]), 3),
            "Biomarkers":    round(float(mod_weights[2]), 3),
            "Questionnaire": round(float(mod_weights[3]), 3),
        }

        return report

    def predict_from_dict(self, sample: dict, top_k: int = 10) -> dict:
        """Convenience wrapper accepting a dict with modality keys."""
        return self.predict(
            snps=np.array(sample["snps"]),
            facial_emb=np.array(sample["facial_emb"]),
            biomarkers=np.array(sample["biomarkers"]),
            questionnaire=np.array(sample["questionnaire"]),
            top_k=top_k,
        )


def demo_inference():
    """Run a demo prediction with a synthetic individual."""
    from src.utils.synthetic_data import (
        generate_snps, generate_biomarkers,
        generate_big_five, generate_facial_embeddings
    )

    rng = np.random.default_rng(99)
    n = 1
    is_autism = np.array([True])

    snps       = generate_snps(n, is_autism)
    biomarkers = generate_biomarkers(n, is_autism)
    big_five   = generate_big_five(n, is_autism)
    facial_emb = generate_facial_embeddings(n, big_five)
    quest      = np.hstack([big_five, rng.normal(0, 0.1, (n, 27))]).astype(np.float32)

    engine = InferenceEngine()
    report = engine.predict(snps[0], facial_emb[0], biomarkers[0], quest[0])

    print("\n" + "="*60)
    print("  CAREER MAPPING GENOMICS — SAMPLE REPORT")
    print("="*60)
    print(f"\nPersonality Profile:")
    for trait, info in report["personality_profile"].items():
        bar = "█" * int(info["score"] * 20)
        print(f"  {trait:20s} {info['level']:10s} {bar} ({info['score']:.3f})")

    print(f"\nASD Assessment: {report['asd_assessment']['tier']} "
          f"({report['asd_assessment']['probability_pct']:.1f}%)")

    print(f"\nTop 5 Career Matches:")
    for c in report["top_careers"][:5]:
        star = "★" if c["autism_strength"] else " "
        print(f"  {c['rank']:2d}. {star} {c['title']:35s} {c['compatibility_pct']:5.1f}%")

    print(f"\nModality Contributions:")
    for mod, w in report["modality_contributions"].items():
        print(f"  {mod:15s}: {w:.3f}")

    print(f"\nFacial Emotion: {report['facial_emotion']['dominant_emotion']}")
    print(f"\n{report['disclaimer'][:100]}...")
    return report


if __name__ == "__main__":
    demo_inference()
