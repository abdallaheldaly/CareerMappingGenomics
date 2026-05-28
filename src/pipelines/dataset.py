"""
Multi-Modal Dataset and DataLoader
Handles loading, splitting, and batching of all four modalities.
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import SYNTHETIC_DIR, TRAIN_CFG


class MultiModalDataset(Dataset):
    """
    PyTorch Dataset for multi-modal genomics + career data.
    Loads pre-processed numpy arrays from disk or accepts in-memory dict.
    """

    def __init__(self, data: dict = None, split: str = "all"):
        """
        data: dict with keys snps, biomarkers, big_five, facial_emb,
              pgs_scores, questionnaire, career_labels, is_autism, intelligence
        split: 'all', 'train', 'val', 'test' (split done externally via Subset)
        """
        if data is not None:
            self.snps           = torch.tensor(data["snps"],          dtype=torch.float32)
            self.biomarkers     = torch.tensor(data["biomarkers"],     dtype=torch.float32)
            self.big_five       = torch.tensor(data["big_five"],       dtype=torch.float32)
            self.facial_emb     = torch.tensor(data["facial_emb"],     dtype=torch.float32)
            self.pgs_scores     = torch.tensor(data["pgs_scores"],     dtype=torch.float32)
            self.questionnaire  = torch.tensor(data["questionnaire"],  dtype=torch.float32)
            self.career_labels  = torch.tensor(data["career_labels"],  dtype=torch.long)
            self.is_autism      = torch.tensor(data["is_autism"],      dtype=torch.float32)
            self.intelligence   = torch.tensor(data["intelligence"],   dtype=torch.float32)
        else:
            self._load_from_disk()

    def _load_from_disk(self):
        d = SYNTHETIC_DIR
        self.snps          = torch.tensor(np.load(d / "snps.npy"),          dtype=torch.float32)
        self.biomarkers    = torch.tensor(np.load(d / "biomarkers.npy"),    dtype=torch.float32)
        self.big_five      = torch.tensor(np.load(d / "big_five.npy"),      dtype=torch.float32)
        self.facial_emb    = torch.tensor(np.load(d / "facial_emb.npy"),    dtype=torch.float32)
        self.pgs_scores    = torch.tensor(np.load(d / "pgs_scores.npy"),    dtype=torch.float32)
        self.questionnaire = torch.tensor(
            np.hstack([
                np.load(d / "big_five.npy"),
                np.random.randn(len(np.load(d / "big_five.npy")), 27).astype(np.float32) * 0.1
            ]), dtype=torch.float32
        )
        self.career_labels = torch.tensor(np.load(d / "career_labels.npy"), dtype=torch.long)
        self.is_autism     = torch.tensor(np.load(d / "is_autism.npy"),     dtype=torch.float32)
        self.intelligence  = torch.zeros(len(self.career_labels))  # placeholder

    def __len__(self):
        return len(self.career_labels)

    def __getitem__(self, idx):
        return {
            "snps":          self.snps[idx],
            "biomarkers":    self.biomarkers[idx],
            "big_five":      self.big_five[idx],
            "facial_emb":    self.facial_emb[idx],
            "pgs_scores":    self.pgs_scores[idx],
            "questionnaire": self.questionnaire[idx],
            "career_label":  self.career_labels[idx],
            "is_autism":     self.is_autism[idx],
            "intelligence":  self.intelligence[idx],
        }


def get_dataloaders(data: dict, cfg=None, seed: int = 42):
    """
    Split dataset into train/val/test and return DataLoaders.
    """
    cfg = cfg or TRAIN_CFG
    dataset = MultiModalDataset(data)
    n = len(dataset)

    n_test  = int(n * cfg.test_split)
    n_val   = int(n * cfg.val_split)
    n_train = n - n_val - n_test

    torch.manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    print(f"[DataLoader] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader
