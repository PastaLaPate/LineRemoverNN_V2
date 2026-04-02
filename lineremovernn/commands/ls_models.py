import argparse
from pathlib import Path

from lineremovernn.commands.command import Command
from lineremovernn.model.models import list_models
from lineremovernn.utils import logging
from lineremovernn.utils.consts import DEFAULT_MODELS

logger = logging.get_logger("ModelLister")


class ListModels(Command):
    def __init__(self) -> None:
        super().__init__(
            "ls",
            "List all available models.",
        )

    def init_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--models-dir",
            type=Path,
            required=False,
            default=DEFAULT_MODELS,
            help="Path to the directory containing saved models.",
        )

    def execute(self, args: argparse.Namespace) -> None:
        args.models_dir.mkdir(parents=True, exist_ok=True)
        models = list_models()
        if len(models) == 0:
            logger.warning("No models found in %s", args.models_dir)
            return
        for model in models:
            stats, path = model
            logger.info(
                f"Epoch: {stats.epoch}, Loss: {stats.loss:.4f}, Time: {stats.time:.2f}s, Channels: {stats.channels} - Path: {path}"
            )
