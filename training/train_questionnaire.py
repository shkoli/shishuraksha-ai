"""Training script for the questionnaire-based MLP risk classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from models.questionnaire_model import QuestionnaireRiskModel
from utils.logger import get_logger

logger = get_logger(__name__)


def load_features(data_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load questionnaire feature matrix and risk labels from disk."""
    ...


def build_dataloaders(
    data_dir: Path,
    batch_size: int = 64,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Construct train/val/test TensorDataset loaders from tabular CSV files."""
    ...


def train_epoch(
    model: QuestionnaireRiskModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Run one training epoch on the questionnaire MLP."""
    ...


def evaluate(
    model: QuestionnaireRiskModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate the questionnaire model on a DataLoader split."""
    ...


def train(args: argparse.Namespace) -> None:
    """Full training loop with cross-validation support."""
    ...


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train questionnaire MLP risk model")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/questionnaire"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/questionnaire"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[128, 64])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
