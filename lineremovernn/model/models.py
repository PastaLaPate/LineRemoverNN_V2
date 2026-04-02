from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch

from lineremovernn.model.model import LineRemoverNN
from lineremovernn.utils.consts import DEFAULT_MODELS, DEVICE
from lineremovernn.utils.logging import get_logger

logger = get_logger("ModelManager")


@dataclass
class ModelState:
    model: LineRemoverNN
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None


@dataclass
class ModelStats:
    epoch: int
    loss: float
    time: float
    channels: list[int]


@dataclass
class SavedModel:
    stats: ModelStats
    model_state: dict
    optimizer_state: dict
    scheduler_state: dict


def save_model(
    stats: ModelStats, state: ModelState, models_dir: Path = DEFAULT_MODELS
) -> Path:
    path = (
        models_dir
        / f"epoch_{stats.epoch}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pt"
    )
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": stats.epoch,
            "loss": stats.loss,
            "time": stats.time,
            "channels": stats.channels,
            "model_state": state.model.state_dict(),
            "optimizer_state": state.optimizer.state_dict(),
            "scheduler_state": state.scheduler.state_dict()
            if state.scheduler
            else None,
        },
        path,
    )
    return path


def list_models(models_dir: Path = DEFAULT_MODELS) -> list[tuple[ModelStats, Path]]:
    models = []
    for path in models_dir.iterdir():
        if not path.is_file() or path.suffix != ".pt":
            continue
        try:
            data = torch.load(path, map_location=DEVICE, weights_only=True)
            models.append(
                (
                    ModelStats(
                        epoch=data["epoch"],
                        loss=data["loss"],
                        time=data["time"],
                        channels=data["channels"],
                    ),
                    path,
                )
            )
        except Exception as e:
            logger.error("Failed to load model from %s: %s", path, e)

    models.sort(key=lambda x: x[0].epoch)
    return models


def get_latest_model() -> tuple[ModelStats, Path] | None:
    models = list_models()
    return models[-1] if models else None


def load_model(path: Path) -> SavedModel:
    data = torch.load(path, map_location=DEVICE, weights_only=True)
    return SavedModel(
        stats=ModelStats(
            epoch=data["epoch"],
            loss=data["loss"],
            time=data["time"],
            channels=data["channels"],
        ),
        model_state=data["model_state"],
        optimizer_state=data["optimizer_state"],
        scheduler_state=data.get("scheduler_state"),
    )
