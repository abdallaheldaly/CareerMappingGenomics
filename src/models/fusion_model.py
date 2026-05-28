"""
FusionNet — Multi-Modal Cross-Attention Fusion + Career Recommendation Engine

Fuses outputs from all four modality encoders:
  1. Genomic embedding  (SNP-Transformer)
  2. Facial embedding   (FaceGenome-CNN)
  3. Biomarker embedding (BiomarkerNet)
  4. Questionnaire embedding (MLP)

Produces:
  - Final unified personality vector
  - Career compatibility scores for all careers
  - Autism trait score
  - Confidence-calibrated outputs

References
----------
[FUS-01] Baltrusaitis et al. (2018). Multimodal machine learning: A survey and taxonomy.
         IEEE TPAMI, 41(2), 423–443. https://doi.org/10.1109/TPAMI.2018.2798607
         → Core survey for multi-modal fusion; cross-modal attention design pattern.

[FUS-03] Xu et al. (2023). Multimodal learning with transformers: A survey.
         IEEE TPAMI, 45(10), 12113–12132. https://doi.org/10.1109/TPAMI.2023.3275156
         → Cross-modal attention transformer for 4-modality fusion in FusionNet.

[FUS-02] Ngiam et al. (2011). Multimodal deep learning. ICML, 689–696.
         → Joint embedding learning across modalities (genomic + facial + blood + questionnaire).

[DL-08]  Kendall et al. (2018). Multi-task learning using uncertainty to weigh losses.
         CVPR, 7482–7491. https://doi.org/10.1109/CVPR.2018.00781
         → MultiTaskLoss with learnable log-variance implemented in trainer.py.

[CAR-01] Peterson et al. (1999). An occupational information system for the 21st century
         (O*NET). American Psychological Association.
         → Career trait vectors (O,C,E,A,N + intelligence) used in CareerCompatibilityHead.

[PSY-08] Barrick & Mount (1991). The Big Five personality dimensions and job performance.
         Personnel Psychology, 44(1), 1–26.
         https://doi.org/10.1111/j.1744-6570.1991.tb00688.x
         → Empirical Big Five → job performance link; basis for career compatibility scoring.

[ASD-07] Baron-Cohen (2002). The extreme male brain theory of autism.
         Trends in Cognitive Sciences, 6(6), 248–254.
         https://doi.org/10.1016/S1364-6613(02)01904-6
         → Systemising/empathising framework; autism_strength flag in career database.

[ASD-08] Mottron et al. (2006). Enhanced perceptual functioning in autism.
         Journal of Autism and Developmental Disorders, 36(1), 27–43.
         https://doi.org/10.1007/s10803-005-0040-7
         → Autism perceptual strengths used to populate autism_advantages in career entries.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import FUSION_CFG, BIG_FIVE
from src.utils.career_database import CAREER_DATABASE


# ──────────────────────────────────────────────────────────────────────
# Modality projection layers
# ──────────────────────────────────────────────────────────────────────

class ModalityProjector(nn.Module):
    """Project each modality embedding to a common dimension."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.proj(x)


class CrossModalAttention(nn.Module):
    """
    Cross-attention between all modality embeddings.
    Each modality attends to all others to capture cross-modal dependencies
    (e.g., genomic-biomarker correlations, facial-personality links).
    """
    def __init__(self, dim: int, n_modalities: int, n_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads,
            dropout=0.1, batch_first=True
        )
        self.norm  = nn.LayerNorm(dim)
        self.ff    = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_modalities, dim)"""
        attn_out, self.last_attn_weights = self.attn(x, x, x, need_weights=True, average_attn_weights=False)
        x = self.norm(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class QuestionnaireEncoder(nn.Module):
    """Encode questionnaire responses (Big Five self-report) to embeddings."""
    def __init__(self, input_dim: int = 32, out_dim: int = 32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64), nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        return self.enc(x)


class CareerCompatibilityHead(nn.Module):
    """
    Maps unified personality vector to career compatibility scores.
    Uses a learned metric space where career prototypes are trainable embeddings.
    """
    def __init__(self, personality_dim: int, n_careers: int, career_feature_dim: int = 32):
        super().__init__()
        self.n_careers = n_careers

        # Career prototype embeddings (learned)
        self.career_prototypes = nn.Embedding(n_careers, career_feature_dim)

        # Personality → career embedding space projection
        self.personality_proj = nn.Sequential(
            nn.Linear(personality_dim, career_feature_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(career_feature_dim * 2, career_feature_dim),
            nn.LayerNorm(career_feature_dim),
        )

        # Initialize career prototypes from database trait vectors
        self._init_career_prototypes()

    def _init_career_prototypes(self):
        """Initialize prototypes from O*NET trait vectors in the database."""
        n = min(self.n_careers, len(CAREER_DATABASE))
        trait_matrix = np.zeros((n, 7))  # O,C,E,A,N, intelligence, autism_strength
        for i, career in enumerate(CAREER_DATABASE[:n]):
            trait_matrix[i] = [
                career["O"], career["C"], career["E"],
                career["A"], career["N"],
                career["intelligence"],
                float(career["autism_strength"])
            ]
        # Simple random projection to career_feature_dim=32
        proj = np.random.randn(7, 32) * 0.1
        prototypes = trait_matrix @ proj
        if self.n_careers > len(CAREER_DATABASE):
            pad = np.zeros((self.n_careers - len(CAREER_DATABASE), 32))
            prototypes = np.vstack([prototypes, pad])
        with torch.no_grad():
            self.career_prototypes.weight.copy_(
                torch.tensor(prototypes[:self.n_careers], dtype=torch.float32)
            )

    def forward(self, personality_vec: torch.Tensor) -> torch.Tensor:
        """
        personality_vec: (B, personality_dim)
        Returns: compatibility scores (B, n_careers) — higher = better match
        """
        p_emb   = self.personality_proj(personality_vec)  # (B, career_feat_dim)
        c_emb   = self.career_prototypes.weight             # (n_careers, career_feat_dim)
        # Cosine similarity
        p_norm  = F.normalize(p_emb, dim=-1)
        c_norm  = F.normalize(c_emb, dim=-1)
        scores  = p_norm @ c_norm.T                         # (B, n_careers)
        return scores


class FusionNet(nn.Module):
    """
    Main multi-modal fusion network.
    Accepts pre-computed modality embeddings and fuses them.

    Input:
        genomic_emb:     (B, 64)
        facial_emb:      (B, 128)
        biomarker_emb:   (B, 64)
        questionnaire:   (B, 32)

    Output:
        personality_vec:    (B, fused_dim)
        big_five_final:     (B, 5)
        career_scores:      (B, n_careers)
        asd_score:          (B, 1)
        modality_weights:   (B, 4) — attention weights per modality
    """
    def __init__(self, cfg=None):
        super().__init__()
        cfg = cfg or FUSION_CFG
        self.cfg = cfg
        common_dim = 64  # all modalities projected to this

        # Modality projectors
        self.proj_genomic     = ModalityProjector(cfg.genomic_dim,       common_dim)
        self.proj_facial      = ModalityProjector(cfg.facial_dim,        common_dim)
        self.proj_biomarker   = ModalityProjector(cfg.biomarker_dim,     common_dim)
        self.proj_questionnaire = ModalityProjector(cfg.questionnaire_dim, common_dim)

        self.questionnaire_enc = QuestionnaireEncoder(
            cfg.questionnaire_dim, cfg.questionnaire_dim
        )

        # Cross-modal attention (2 layers)
        self.cross_attn_1 = CrossModalAttention(common_dim, 4, n_heads=4)
        self.cross_attn_2 = CrossModalAttention(common_dim, 4, n_heads=4)

        # Modality importance weights
        self.modality_gate = nn.Sequential(
            nn.Linear(4 * common_dim, 4),
            nn.Softmax(dim=-1),
        )

        # Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(4 * common_dim, cfg.fused_dim),
            nn.LayerNorm(cfg.fused_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fused_dim, cfg.fused_dim),
            nn.LayerNorm(cfg.fused_dim),
            nn.GELU(),
        )

        # Final personality vector → Big Five
        self.big_five_head = nn.Sequential(
            nn.Linear(cfg.fused_dim, 64),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(64, 5),
            nn.Sigmoid(),
        )

        # ASD composite score
        self.asd_head = nn.Sequential(
            nn.Linear(cfg.fused_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Career compatibility
        n_careers = min(cfg.n_careers, len(CAREER_DATABASE))
        self.career_head = CareerCompatibilityHead(cfg.fused_dim, n_careers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear,)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        genomic_emb:    torch.Tensor,
        facial_emb:     torch.Tensor,
        biomarker_emb:  torch.Tensor,
        questionnaire:  torch.Tensor,
    ):
        # Project all modalities to common dim
        g = self.proj_genomic(genomic_emb)
        f = self.proj_facial(facial_emb)
        b = self.proj_biomarker(biomarker_emb)
        q = self.proj_questionnaire(self.questionnaire_enc(questionnaire))

        # Stack: (B, 4, common_dim)
        x = torch.stack([g, f, b, q], dim=1)

        # Two layers of cross-modal attention
        x = self.cross_attn_1(x)
        x = self.cross_attn_2(x)  # (B, 4, common_dim)

        # Modality gating weights
        x_flat = x.view(x.shape[0], -1)  # (B, 4*common_dim)
        modality_weights = self.modality_gate(x_flat)  # (B, 4)

        # Weighted sum + flatten → fusion
        x_fused = self.fusion_mlp(x_flat)  # (B, fused_dim)

        big_five_final = self.big_five_head(x_fused)
        asd_score      = self.asd_head(x_fused)
        career_scores  = self.career_head(x_fused)

        return x_fused, big_five_final, career_scores, asd_score, modality_weights


# ──────────────────────────────────────────────────────────────────────
# Complete Inference Pipeline
# ──────────────────────────────────────────────────────────────────────

class CareerMappingSystem(nn.Module):
    """
    End-to-end system combining all four modality models + FusionNet.
    """
    def __init__(self, genomic_model, facial_model, biomarker_model, fusion_net):
        super().__init__()
        self.genomic_model   = genomic_model
        self.facial_model    = facial_model
        self.biomarker_model = biomarker_model
        self.fusion_net      = fusion_net

    def forward(self, snps, facial_emb, biomarkers, questionnaire):
        # Each modality model
        g_emb, pgs_scores           = self.genomic_model(snps)
        f_emb, bf_facial, fer_logits = self.facial_model(facial_emb)
        b_emb, asd_bio, bf_bio, _   = self.biomarker_model(biomarkers)

        # FusionNet
        fused, big_five_final, career_scores, asd_final, mod_weights = \
            self.fusion_net(g_emb, f_emb, b_emb, questionnaire)

        return {
            "personality":     big_five_final,   # (B, 5)
            "career_scores":   career_scores,     # (B, n_careers)
            "asd_score":       asd_final,         # (B, 1)
            "pgs_scores":      pgs_scores,        # (B, n_pgs)
            "fer_logits":      fer_logits,         # (B, 7)
            "modality_weights": mod_weights,       # (B, 4)
        }


def build_career_report(
    big_five: np.ndarray,
    career_scores: np.ndarray,
    asd_score: float,
    pgs_scores: np.ndarray,
    top_k: int = 10,
) -> dict:
    """
    Convert model outputs into a human-readable career report.
    """
    n_careers = min(career_scores.shape[0], len(CAREER_DATABASE))
    scores = career_scores[:n_careers]

    # Rank careers
    ranked_idx = np.argsort(scores)[::-1]
    top_careers = []
    for rank, idx in enumerate(ranked_idx[:top_k]):
        career = CAREER_DATABASE[idx]
        compatibility_pct = float(np.clip((scores[idx] + 1) / 2 * 100, 0, 100))
        top_careers.append({
            "rank":              rank + 1,
            "id":                career["id"],
            "title":             career["title"],
            "domain":            career["domain"],
            "compatibility_pct": round(compatibility_pct, 1),
            "education":         career["education"],
            "description":       career["description"],
            "autism_strength":   career["autism_strength"],
            "autism_advantages": career.get("autism_advantages", []),
            "trait_match": {
                "O": round(1 - abs(big_five[0] - career["O"]), 3),
                "C": round(1 - abs(big_five[1] - career["C"]), 3),
                "E": round(1 - abs(big_five[2] - career["E"]), 3),
                "A": round(1 - abs(big_five[3] - career["A"]), 3),
                "N": round(1 - abs(big_five[4] - career["N"]), 3),
            }
        })

    # Personality interpretation
    trait_names = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
    personality_profile = {}
    for i, trait in enumerate(trait_names):
        score = float(big_five[i])
        if score > 0.66:
            level = "High"
        elif score > 0.33:
            level = "Moderate"
        else:
            level = "Low"
        personality_profile[trait] = {"score": round(score, 3), "level": level}

    # ASD profile
    asd_pct = float(asd_score) * 100
    if asd_pct > 65:
        asd_tier = "Likely ASD Traits Present"
    elif asd_pct > 40:
        asd_tier = "Possible ASD Traits"
    else:
        asd_tier = "Neurotypical Profile"

    return {
        "personality_profile":  personality_profile,
        "asd_assessment":       {"probability_pct": round(asd_pct, 1), "tier": asd_tier},
        "top_careers":          top_careers,
        "modality_pgs": {
            name: round(float(pgs_scores[i]), 3)
            for i, name in enumerate(["Openness_PGS", "Conscientiousness_PGS",
                                       "Extraversion_PGS", "Agreeableness_PGS",
                                       "Neuroticism_PGS", "Autism_PGS",
                                       "Intelligence_PGS", "Education_PGS"])
            if i < len(pgs_scores)
        },
        "disclaimer": (
            "This report is a research tool only. Personality polygenic scores explain "
            "5–15% of trait variance. These results are probabilistic, not deterministic. "
            "Career recommendations should be considered alongside personal interests, "
            "education, experience, and professional guidance."
        )
    }


if __name__ == "__main__":
    from src.models.genomic_model import SNPTransformer
    from src.models.facial_model import FaceGenomeCNN
    from src.models.biomarker_model import BiomarkerNet
    from configs.config import GENOMIC_CFG, FACIAL_CFG, BIOMARKER_CFG, FUSION_CFG

    gm = SNPTransformer()
    fm = FaceGenomeCNN()
    bm = BiomarkerNet()
    fn = FusionNet()

    system = CareerMappingSystem(gm, fm, bm, fn)
    total  = sum(p.numel() for p in system.parameters())
    print(f"Total system parameters: {total:,}")

    snps    = torch.randn(2, GENOMIC_CFG.n_snps)
    face    = torch.randn(2, FACIAL_CFG.embedding_dim)
    bio     = torch.randn(2, BIOMARKER_CFG.n_markers)
    quest   = torch.randn(2, 32)

    out = system(snps, face, bio, quest)
    for k, v in out.items():
        print(f"  {k}: {v.shape}")
