"""Training script for the multi-modal attention fusion model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models.fusion_model import FusionRiskModel
from modules.fusion.attention_fusion import FusionInput
from utils.logger import get_logger

logger = get_logger(__name__)


class MultiModalDataset(Dataset):
    """PyTorch Dataset that aligns multi-modal features by case ID."""

    def __init__(
        self,
        case_ids: list[str],
        questionnaire_features: Any | None,
        text_features: Any | None,
        drawing_features: Any | None,
        facial_features: Any | None,
        labels: list[int],
    ) -> None:
        self.case_ids = case_ids
        self.questionnaire_features = questionnaire_features
        self.text_features = text_features
        self.drawing_features = drawing_features
        self.facial_features = facial_features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int) -> tuple[FusionInput, int]:
        """Retrieve aligned multi-modal features for one case."""
        ...


def collate_fusion_batch(batch: list[tuple[FusionInput, int]]) -> tuple[FusionInput, torch.Tensor]:
    """Custom collate function to stack FusionInput across batch samples."""
    ...


def build_dataloaders(data_dir: Path, batch_size: int = 32) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Load aligned multi-modal features and build train/val/test DataLoaders."""
    ...


def train_epoch(
    model: FusionRiskModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Run one training epoch for the fusion model."""
    ...


def evaluate(
    model: FusionRiskModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate fusion model on val/test DataLoader."""
    ...


def train(args: argparse.Namespace) -> None:
    """Full fusion training loop with modality-dropout regularisation."""
    ...


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multi-modal attention fusion model")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/fusion"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--modality-dropout", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
