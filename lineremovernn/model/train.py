from time import time_ns
from typing import cast

import torch
from torch.amp import GradScaler, autocast  # type: ignore
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from tqdm import tqdm

from lineremovernn.data.pages_dataset import PagesDataset
from lineremovernn.model import models
from lineremovernn.model.model import LineRemoverNN
from lineremovernn.model.models import ModelState, ModelStats, save_model
from lineremovernn.utils import logging
from lineremovernn.utils.consts import DEVICE, SAVED
from lineremovernn.utils.loss import CombinedDiceBCEWithLogitsLoss
from lineremovernn.utils.paired_transform import PairedTransform

logger = logging.get_logger("Trainer")

criterion = CombinedDiceBCEWithLogitsLoss()


def train_epoch(model, loader, optimizer, device, epoch) -> float:
    model.train()
    total_loss = 0.0
    scaler = GradScaler("cuda")  # Handles loss scaling for FP16
    bar = tqdm(loader, desc=f"Epoch {epoch:03d} [train]", unit="batch")

    for blank, ruled in bar:
        ruled, blank = ruled.to(device), blank.to(device)
        optimizer.zero_grad()

        # Runs the forward pass in mixed precision
        with autocast("cuda"):
            pred = model(ruled)
            loss = criterion(pred, blank)

        # Scale the loss and step the optimizer
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        bar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(loader)


@torch.no_grad()
def val_epoch(
    model: LineRemoverNN,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
) -> float:
    model.eval()
    total_loss = 0.0
    bar = tqdm(loader, desc=f"Epoch {epoch:03d} [val]  ", unit="batch", leave=False)
    for blank, ruled in bar:
        ruled, blank = ruled.to(device), blank.to(device)
        pred = model(ruled)
        loss = criterion(pred, blank)
        total_loss += loss.item()
        bar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(loader)


def run(
    dataset_dir, val_split, epochs, batch_size, lr, workers, channels, no_load
) -> None:
    device = torch.device(DEVICE)
    logger.info("Device: %s", device)

    dataset = PagesDataset(dataset_dir, transform=PairedTransform(augment=True))
    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    cast(PagesDataset, val_dataset.dataset).transform = PairedTransform(augment=False)

    logger.info("Train: %d | Val: %d", train_size, val_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=True,
    )

    model = LineRemoverNN(in_channels=1).to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("Model parameters: %.2fM", total_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if not no_load:
        l_model = models.get_latest_model()
        logger.info("Latest model: %s", l_model[1] if l_model else "None")
        if l_model is not None:
            train_state = models.load_model(l_model[1])
            logger.info(
                "Loading model from %s (epoch %d, loss %.4f)",
                l_model[1],
                train_state.stats.epoch,
                train_state.stats.loss,
            )
            model.load_state_dict(train_state.model_state)
            optimizer.load_state_dict(train_state.optimizer_state)
            scheduler.load_state_dict(train_state.scheduler_state)
        else:
            logger.warning("No model found, starting training from scratch.")
    SAVED.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        val_loss = val_epoch(model, val_loader, device, epoch)
        scheduler.step()

        logger.info(
            "Epoch %03d | train_loss=%.4f | val_loss=%.4f | lr=%.2e",
            epoch,
            train_loss,
            val_loss,
            scheduler.get_last_lr()[0],
        )

        save_model(
            ModelStats(epoch=epoch, loss=val_loss, time=time_ns(), channels=channels),
            ModelState(model=model, optimizer=optimizer, scheduler=scheduler),
        )

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.info("New best model saved (val_loss=%.4f)", best_val_loss)
