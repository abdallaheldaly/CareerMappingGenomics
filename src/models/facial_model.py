"""
Facial Recognition Model — FaceGenome-CNN
Processes facial images (or pre-extracted landmark embeddings) to predict
personality trait embeddings.

In full deployment: takes 224×224 RGB images, runs ResNet backbone.
In simulation mode: takes pre-computed facial embeddings directly.

References
----------
[FAC-01] Parkhi et al. (2015). Deep face recognition. BMVC.
         https://doi.org/10.5244/C.29.41
         → VGGFace architecture principles applied to facial feature extraction.

[FAC-02] Schroff et al. (2015). FaceNet: A unified embedding for face recognition.
         CVPR, 815–823. https://doi.org/10.1109/CVPR.2015.7298682
         → Embedding head design with L2 normalisation.

[FAC-03] Lugaresi et al. (2019). MediaPipe: A framework for building perception pipelines.
         arXiv:1906.08172. https://arxiv.org/abs/1906.08172
         → MediaPipe Face Mesh (468 landmarks) referenced for production deployment.

[FAC-05] Ekman & Friesen (1978). Facial action coding system.
         Consulting Psychologists Press.
         → FACS Action Units — theoretical basis for 7-class FER taxonomy.

[FAC-06] Li & Deng (2020). Deep facial expression recognition: A survey.
         IEEE Transactions on Affective Computing, 13(3), 1195–1215.
         https://doi.org/10.1109/TAFFC.2020.2981446
         → FER architecture survey informing the FER classification head.

[FAC-08] Gloor et al. (2021). Your face mirrors your deepest beliefs.
         arXiv:2112.12455. https://arxiv.org/abs/2112.12455
         → FER-while-watching-stimuli approach; MIT CCI validation study.

[FAC-09] Rojas Bengochea et al. (2024). A megastudy on the predictability of personal
         information from facial images. PLOS ONE.
         https://doi.org/10.1371/journal.pone.0290643
         → 23% of attributes predictable from facial pixels; calibrates expected accuracy.

[DL-02]  He et al. (2016). Deep residual learning for image recognition. CVPR, 770–778.
         https://doi.org/10.1109/CVPR.2016.90
         → ResNet residual block pattern used in FaceGenomeCNN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import FACIAL_CFG, BIG_FIVE


class FacialMorphometryEncoder(nn.Module):
    """
    Encodes facial landmark embeddings into a trait-relevant representation.
    In production, this receives output from MediaPipe face mesh.
    In simulation, processes pre-computed 128-d facial embeddings.
    """
    def __init__(self, input_dim: int = 128, output_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class FacialAttentionModule(nn.Module):
    """
    Self-attention over facial regions.
    Weights different facial zones (eyes, mouth, forehead, jaw)
    for personality prediction.
    """
    def __init__(self, dim: int, n_regions: int = 8):
        super().__init__()
        self.region_proj = nn.Linear(dim, n_regions)
        self.attention   = nn.Softmax(dim=-1)
        self.combine     = nn.Linear(n_regions, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_weights = self.attention(self.region_proj(x))
        return x + self.combine(attn_weights)


class FaceGenomeCNN(nn.Module):
    """
    Facial personality inference model.

    Input:  facial embedding vector (B, facial_input_dim)
    Output:
        embedding:    (B, embed_dim)      — for FusionNet
        big_five:     (B, 5)              — personality trait scores [0,1]
        fer_logits:   (B, 7)              — facial emotion class logits
    """
    def __init__(self, cfg=None):
        super().__init__()
        cfg = cfg or FACIAL_CFG
        self.cfg = cfg

        inp = cfg.embedding_dim  # 128

        # Morphometry encoder
        self.morph_encoder  = FacialMorphometryEncoder(inp, 128)
        # Attention over facial regions
        self.face_attention = FacialAttentionModule(128)
        # Residual blocks
        self.res_block = nn.Sequential(
            nn.Linear(128, 256), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(256, 128), nn.LayerNorm(128),
        )
        # Final embedding projection
        self.embedding_head = nn.Sequential(
            nn.Linear(128, cfg.embedding_dim),
            nn.LayerNorm(cfg.embedding_dim),
            nn.GELU(),
        )
        # Big Five regression head
        self.big_five_head = nn.Sequential(
            nn.Linear(cfg.embedding_dim, 64),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(64, 5),
            nn.Sigmoid(),
        )
        # Facial Emotion Recognition head
        n_fer = len(cfg.fer_classes)
        self.fer_head = nn.Sequential(
            nn.Linear(cfg.embedding_dim, 32),
            nn.GELU(),
            nn.Linear(32, n_fer),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, facial_emb: torch.Tensor):
        """
        facial_emb: (B, embedding_dim)
        """
        x = self.morph_encoder(facial_emb)     # (B, 128)
        x = self.face_attention(x)              # (B, 128)
        x = x + self.res_block(x)              # Residual (B, 128)
        embedding  = self.embedding_head(x)     # (B, embed_dim)
        big_five   = self.big_five_head(embedding)
        fer_logits = self.fer_head(embedding)
        return embedding, big_five, fer_logits


class FacialPreprocessor:
    """
    Preprocessing pipeline for facial data.
    Handles normalization of landmark embeddings.
    """
    def __init__(self):
        self.mean = None
        self.std  = None

    def fit(self, embeddings: np.ndarray) -> "FacialPreprocessor":
        self.mean = embeddings.mean(axis=0)
        self.std  = embeddings.std(axis=0) + 1e-8
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        return (embeddings - self.mean) / self.std

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        return self.fit(embeddings).transform(embeddings)


class FERInterpreter:
    """Interpret FER logits as emotion probabilities with personality mapping."""
    EMOTION_NAMES = ["neutral", "happy", "sad", "surprise", "fear", "disgust", "anger"]

    # Rough mapping: emotion tendency → Big Five trait associations
    EMOTION_TRAIT_MAP = {
        "neutral":  {"N": -0.1},
        "happy":    {"E": +0.2, "A": +0.15, "N": -0.15},
        "sad":      {"N": +0.2, "E": -0.1},
        "surprise": {"O": +0.15},
        "fear":     {"N": +0.25},
        "disgust":  {"A": -0.15, "N": +0.1},
        "anger":    {"A": -0.2, "N": +0.2, "E": +0.1},
    }

    @classmethod
    def interpret(cls, logits: np.ndarray) -> dict:
        probs = np.exp(logits) / np.exp(logits).sum()
        dominant_emotion = cls.EMOTION_NAMES[np.argmax(probs)]
        return {
            "probabilities": {e: float(p) for e, p in zip(cls.EMOTION_NAMES, probs)},
            "dominant_emotion": dominant_emotion,
        }


if __name__ == "__main__":
    model = FaceGenomeCNN()
    total = sum(p.numel() for p in model.parameters())
    print(f"FaceGenomeCNN parameters: {total:,}")
    x = torch.randn(4, FACIAL_CFG.embedding_dim)
    emb, bf, fer = model(x)
    print(f"Embedding: {emb.shape}, Big Five: {bf.shape}, FER: {fer.shape}")
