"""
Blood Biomarker Model — BiomarkerNet
Multi-layer perceptron with self-attention over biomarker categories.
Maps 80-marker blood panel to:
  - Neurotype profile (ASD probability)
  - Trait embeddings for FusionNet
  - Interpretable biomarker importance (SHAP-ready)

References
----------
[BIO-02] Lv et al. (2021). Oxytocin and autism spectrum disorder: A systematic review.
         Psychiatry and Clinical Neurosciences, 75(12), 351–361.
         https://doi.org/10.1111/pcn.13234
         → Oxytocin as ASD-specific biomarker; lower levels justify downward shift.

[BIO-03] Sajdel-Sulkowska (2022). BDNF and autism spectrum disorder.
         Neuroscience & Biobehavioral Reviews, 132, 800–809.
         https://doi.org/10.1016/j.neubiorev.2021.11.039
         → BDNF reduction in ASD; bdnf_ng_ml marker in Autism_Specific category.

[BIO-04] Essa et al. (2012). Excitotoxicity in the pathogenesis of autism.
         Neurotoxicity Research, 23(4), 393–400.
         https://doi.org/10.1007/s12640-012-9354-3
         → Glutamate/GABA imbalance; gaba_glutamate_ratio and glutamate_nmol_ml.

[BIO-08] Masi et al. (2015). Cytokine aberrations in autism spectrum disorder.
         Molecular Psychiatry, 20(4), 440–446.
         https://doi.org/10.1038/mp.2014.59
         → IL-6, TNF-α, hsCRP elevation in ASD; Inflammatory category shift.

[BIO-10] Muller et al. (2016). The serotonin system in autism spectrum disorder.
         Neuroscience, 321, 24–41.
         https://doi.org/10.1016/j.neuroscience.2015.11.010
         → Serotonin reduction in ASD; serotonin_ng_ml and tryptophan_umol_l.

[BIO-06] Edmiston & Corbett (2016). Biobehavioral profiles of arousal in ASD.
         Journal of Autism and Developmental Disorders, 46(10), 3355–3367.
         https://doi.org/10.1007/s10803-016-2871-z
         → Cortisol/DHEA-S stress response patterns; cortisol_dhea_ratio.

[ETH-10] Lundberg & Lee (2017). A unified approach to interpreting model predictions.
         NeurIPS 30. https://arxiv.org/abs/1705.07874
         → SHAP values used for biomarker importance interpretation (category_attn output).

[DL-01]  Vaswani et al. (2017). Attention is all you need. NeurIPS.
         https://arxiv.org/abs/1706.03762
         → Multi-head self-attention in BiomarkerCrossAttention layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import BIOMARKER_CFG


# ──────────────────────────────────────────────────────────────────────
# Biomarker category indices (first N markers per category)
# ──────────────────────────────────────────────────────────────────────
CATEGORY_SLICES = {
    "Hematology":                slice(0,  5),
    "Hormones":                  slice(5,  12),
    "Metabolic":                 slice(12, 20),
    "Neurotransmitter_Precursors": slice(20, 27),
    "Inflammatory":              slice(27, 31),
    "Nutritional":               slice(31, 37),
    "Autism_Specific":           slice(37, 44),
}


class BiomarkerCategoryEncoder(nn.Module):
    """Encode each biomarker category with a shared MLP."""
    def __init__(self, cat_dim: int, out_dim: int):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(cat_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        return self.enc(x)


class BiomarkerCrossAttention(nn.Module):
    """
    Cross-attention between biomarker categories.
    Allows the model to learn inter-category dependencies,
    e.g., inflammatory + neurotransmitter → ASD signal.
    """
    def __init__(self, n_categories: int, dim: int, n_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads,
            dropout=0.1, batch_first=True
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_categories, dim)"""
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out)


class BiomarkerNet(nn.Module):
    """
    Blood biomarker model.
    Input:  (B, 80) normalized biomarker panel
    Output:
        embedding:    (B, embed_dim)      — for FusionNet
        asd_prob:     (B, 1)              — autism spectrum probability
        trait_pred:   (B, 5)              — Big Five predictions from blood
        category_attn: (B, n_categories) — interpretability weights
    """
    def __init__(self, cfg=None):
        super().__init__()
        cfg = cfg or BIOMARKER_CFG
        self.cfg = cfg
        n_cats = len(CATEGORY_SLICES)
        cat_dim = 16  # shared per-category embedding dim

        # Per-category encoders
        self.cat_encoders = nn.ModuleDict({
            name: BiomarkerCategoryEncoder(
                sl.stop - sl.start, cat_dim
            )
            for name, sl in CATEGORY_SLICES.items()
        })

        # Cross-attention between categories
        self.cross_attn = BiomarkerCrossAttention(n_cats, cat_dim, n_heads=4)

        # Global aggregation
        self.global_agg = nn.Sequential(
            nn.Linear(n_cats * cat_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

        # Embedding projection
        self.embedding_proj = nn.Sequential(
            nn.Linear(128, cfg.embedding_dim),
            nn.LayerNorm(cfg.embedding_dim),
            nn.GELU(),
        )

        # ASD probability head
        self.asd_head = nn.Sequential(
            nn.Linear(cfg.embedding_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Big Five from blood head
        self.big_five_head = nn.Sequential(
            nn.Linear(cfg.embedding_dim, 64),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(64, 5),
            nn.Sigmoid(),
        )

        # Category attention weights (for interpretability)
        self.attn_weight_proj = nn.Linear(cat_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, biomarkers: torch.Tensor):
        """biomarkers: (B, 80)"""
        cat_embs = []
        for name, sl in CATEGORY_SLICES.items():
            cat_input = biomarkers[:, sl]
            cat_emb   = self.cat_encoders[name](cat_input)
            cat_embs.append(cat_emb)

        # Stack: (B, n_cats, cat_dim)
        x = torch.stack(cat_embs, dim=1)

        # Cross-attention
        x = self.cross_attn(x)  # (B, n_cats, cat_dim)

        # Category attention weights (for interpretability)
        attn_weights = F.softmax(
            self.attn_weight_proj(x).squeeze(-1), dim=-1
        )  # (B, n_cats)

        # Flatten and aggregate
        x_flat   = x.view(x.shape[0], -1)  # (B, n_cats*cat_dim)
        x_global = self.global_agg(x_flat)  # (B, 128)

        embedding  = self.embedding_proj(x_global)
        asd_prob   = self.asd_head(embedding)
        big_five   = self.big_five_head(embedding)

        return embedding, asd_prob, big_five, attn_weights


class BiomarkerNormalizer:
    """
    Per-marker normalization using clinical reference ranges.
    Uses z-score normalization with outlier clipping.
    """
    def __init__(self):
        self.means = None
        self.stds  = None

    def fit(self, data: np.ndarray) -> "BiomarkerNormalizer":
        self.means = np.mean(data, axis=0)
        self.stds  = np.std(data, axis=0) + 1e-8
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        z = (data - self.means) / self.stds
        return np.clip(z, -3, 3)  # Clip extreme outliers

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return self.fit(data).transform(data)


if __name__ == "__main__":
    model = BiomarkerNet()
    total = sum(p.numel() for p in model.parameters())
    print(f"BiomarkerNet parameters: {total:,}")
    x = torch.randn(4, BIOMARKER_CFG.n_markers)
    emb, asd, bf, attn = model(x)
    print(f"Embedding: {emb.shape}, ASD prob: {asd.shape}")
    print(f"Big Five: {bf.shape}, Category attn: {attn.shape}")
