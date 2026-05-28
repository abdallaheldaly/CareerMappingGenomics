"""
Training Pipeline
Multi-task training for the CareerMappingGenomics system.
Trains all modality models jointly via shared backprop.

Tasks:
  1. Career classification (cross-entropy)
  2. Personality regression (MSE on Big Five)
  3. ASD classification (binary cross-entropy)
  4. Polygenic score regression (MSE)

References
----------
[DL-07]  Caruana (1997). Multitask learning.
         Machine Learning, 28(1), 41–75.
         https://doi.org/10.1023/A:1007379606734
         → Conceptual foundation for jointly training career + personality + ASD tasks.

[DL-08]  Kendall et al. (2018). Multi-task learning using uncertainty to weigh losses.
         CVPR, 7482–7491. https://doi.org/10.1109/CVPR.2018.00781
         → MultiTaskLoss class: learnable log-variance per task (Eq. 2 in the paper).

[DL-10]  Loshchilov & Hutter (2017). Decoupled weight decay regularization (AdamW).
         ICLR 2019. https://arxiv.org/abs/1711.05101
         → AdamW optimizer used with weight_decay for regularisation.

[DL-11]  Loshchilov & Hutter (2016). SGDR: Stochastic gradient descent with warm restarts.
         ICLR 2017. https://arxiv.org/abs/1608.03983
         → CosineAnnealingLR scheduler (T_max=epochs, eta_min=1e-5).

[DL-12]  Srivastava et al. (2014). Dropout: A simple way to prevent neural networks
         from overfitting. JMLR, 15(1), 1929–1958.
         http://jmlr.org/papers/v15/srivastava14a.html
         → Dropout layers in all sub-models; gradient clipping at max_norm=1.0.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from configs.config import TRAIN_CFG, MODELS_DIR, BIG_FIVE
from src.models.genomic_model import SNPTransformer
from src.models.facial_model import FaceGenomeCNN
from src.models.biomarker_model import BiomarkerNet
from src.models.fusion_model import FusionNet, CareerMappingSystem
from src.pipelines.dataset import get_dataloaders


# ──────────────────────────────────────────────────────────────────────
# Loss Functions
# ──────────────────────────────────────────────────────────────────────

class MultiTaskLoss(nn.Module):
    """
    Combines multiple task losses with learned uncertainty weighting.
    Based on: Kendall et al. "Multi-Task Learning Using Uncertainty to Weigh Losses"
    """
    def __init__(self, n_tasks: int = 3):
        super().__init__()
        # Log-variance parameters (one per task)
        self.log_var = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses: list) -> torch.Tensor:
        total = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_var[i])
            total += precision * loss + self.log_var[i]
        return total


# ──────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────

def compute_metrics(
    career_preds: torch.Tensor,
    career_labels: torch.Tensor,
    bf_preds: torch.Tensor,
    bf_targets: torch.Tensor,
    asd_preds: torch.Tensor,
    asd_targets: torch.Tensor,
) -> dict:
    career_acc = (career_preds.argmax(dim=-1) == career_labels).float().mean().item()

    bf_mse = F.mse_loss(bf_preds, bf_targets).item() if bf_preds is not None else 0
    bf_mae = (bf_preds - bf_targets).abs().mean().item() if bf_preds is not None else 0

    asd_binary = (asd_preds.squeeze() > 0.5).float()
    asd_acc = (asd_binary == asd_targets).float().mean().item()

    return {
        "career_acc":   round(career_acc, 4),
        "bf_mse":       round(bf_mse,     4),
        "bf_mae":       round(bf_mae,     4),
        "asd_acc":      round(asd_acc,    4),
    }

import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self, system: CareerMappingSystem, cfg=None, device: str = None):
        self.cfg    = cfg or TRAIN_CFG
        self.device = device or self.cfg.device
        self.system = system.to(self.device)

        self.mt_loss = MultiTaskLoss(n_tasks=3).to(self.device)

        self.optimizer = optim.AdamW(
            list(system.parameters()) + list(self.mt_loss.parameters()),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.cfg.epochs, eta_min=1e-5
        )

        self.history = {"train": [], "val": []}
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def _run_epoch(self, loader, train: bool = True) -> dict:
        self.system.train(train)
        total_loss = 0
        all_career_preds, all_career_labels = [], []
        all_bf_preds, all_bf_targets        = [], []
        all_asd_preds, all_asd_targets      = [], []

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in loader:
                snps     = batch["snps"].to(self.device)
                bio      = batch["biomarkers"].to(self.device)
                face     = batch["facial_emb"].to(self.device)
                quest    = batch["questionnaire"].to(self.device)
                c_label  = batch["career_label"].to(self.device)
                bf_tgt   = batch["big_five"].to(self.device)
                asd_tgt  = batch["is_autism"].to(self.device)

                out = self.system(snps, face, bio, quest)

                # Task 1: Career classification
                career_loss = F.cross_entropy(out["career_scores"], c_label)
                # Task 2: Personality regression
                bf_loss = F.mse_loss(out["personality"], bf_tgt)
                # Task 3: ASD binary classification
                asd_loss = F.binary_cross_entropy(out["asd_score"].squeeze(), asd_tgt)

                loss = self.mt_loss([career_loss, bf_loss, asd_loss])

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.system.parameters(), max_norm=1.0)
                    self.optimizer.step()

                total_loss += loss.item()
                all_career_preds.append(out["career_scores"].detach().cpu())
                all_career_labels.append(c_label.cpu())
                all_bf_preds.append(out["personality"].detach().cpu())
                all_bf_targets.append(bf_tgt.cpu())
                all_asd_preds.append(out["asd_score"].detach().cpu())
                all_asd_targets.append(asd_tgt.cpu())

        metrics = compute_metrics(
            torch.cat(all_career_preds), torch.cat(all_career_labels),
            torch.cat(all_bf_preds),     torch.cat(all_bf_targets),
            torch.cat(all_asd_preds),    torch.cat(all_asd_targets),
        )
        metrics["loss"] = round(total_loss / len(loader), 4)
        return metrics

    def train(self, train_loader, val_loader):
        print(f"\n{'='*60}")
        print(f"  CareerMappingGenomics — Training")
        print(f"  Epochs: {self.cfg.epochs} | LR: {self.cfg.learning_rate} | Device: {self.device}")
        print(f"{'='*60}\n")

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        best_model_path = MODELS_DIR / "best_model.pt"

        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.time()
            train_metrics = self._run_epoch(train_loader, train=True)
            val_metrics   = self._run_epoch(val_loader,   train=False)
            self.scheduler.step()

            elapsed = time.time() - t0
            self.history["train"].append(train_metrics)
            self.history["val"].append(val_metrics)

            print(
                f"Epoch {epoch:3d}/{self.cfg.epochs} ({elapsed:.1f}s) | "
                f"Train loss: {train_metrics['loss']:.4f}  Career acc: {train_metrics['career_acc']:.3f}  "
                f"ASD acc: {train_metrics['asd_acc']:.3f} | "
                f"Val loss: {val_metrics['loss']:.4f}  Career acc: {val_metrics['career_acc']:.3f}  "
                f"ASD acc: {val_metrics['asd_acc']:.3f}"
            )

            # Early stopping + best model checkpoint
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "system_state":  self.system.state_dict(),
                    "mt_loss_state": self.mt_loss.state_dict(),
                    "optimizer":     self.optimizer.state_dict(),
                    "val_metrics":   val_metrics,
                    "history":       self.history,
                }, best_model_path)
                print(f"  ✓ Best model saved (val_loss={self.best_val_loss:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.cfg.early_stopping_patience:
                    print(f"\n  Early stopping triggered at epoch {epoch}.")
                    break

        # Save training history
        with open(MODELS_DIR / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        print(f"\n  Training complete. Best val loss: {self.best_val_loss:.4f}")
        return self.history

    def evaluate(self, test_loader) -> dict:
        metrics = self._run_epoch(test_loader, train=False)
        print(f"\n  Test Metrics:")
        for k, v in metrics.items():
            print(f"    {k}: {v}")
        return metrics


def build_system():
    gm = SNPTransformer()
    fm = FaceGenomeCNN()
    bm = BiomarkerNet()
    fn = FusionNet()
    return CareerMappingSystem(gm, fm, bm, fn)


def run_training(data: dict, cfg=None):
    torch.manual_seed(TRAIN_CFG.seed)
    np.random.seed(TRAIN_CFG.seed)

    train_loader, val_loader, test_loader = get_dataloaders(data, cfg)
    system  = build_system()
    trainer = Trainer(system, cfg)

    history = trainer.train(train_loader, val_loader)
    test_metrics = trainer.evaluate(test_loader)

    return system, trainer, history, test_metrics


if __name__ == "__main__":
    print("Loading synthetic data for training test...")
    from src.utils.synthetic_data import generate_dataset
    data = generate_dataset(n=500, save=False)
    # Quick smoke test: 3 epochs
    from configs.config import TrainConfig
    quick_cfg = TrainConfig(epochs=3, batch_size=32)
    system, trainer, history, metrics = run_training(data, quick_cfg)
    print("Training smoke test passed!")
