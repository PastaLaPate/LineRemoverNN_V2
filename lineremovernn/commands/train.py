import argparse
from pathlib import Path

from lineremovernn.commands.command import Command
from lineremovernn.model.train import run
from lineremovernn.utils import logging
from lineremovernn.utils.consts import DEFAULT_MODELS, DEFAULT_PAGES

logger = logging.get_logger("ModelLister")


class TrainCommand(Command):
    def __init__(self) -> None:
        super().__init__(
            "train",
            "Train the line remover model.",
        )

    def init_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "-md",
            "--models-dir",
            type=Path,
            required=False,
            default=DEFAULT_MODELS,
            help="Path to the directory containing saved models.",
        )
        parser.add_argument(
            "--pages-dir",
            "-pd",
            type=Path,
            required=False,
            default=DEFAULT_PAGES,
            help="Path to the directory containing the pages.",
        )
        parser.add_argument(
            "-e",
            "--epochs",
            type=int,
            required=False,
            default=25,
            help="Number of epochs to train for.",
        )
        parser.add_argument(
            "-nl",
            "--no-load",
            action="store_true",
            help="Whether to start training from a new model instead of loading the latest one.",
        )
        parser.add_argument(
            "-b",
            "--batch-size",
            type=int,
            required=False,
            default=16,
            help="Batch size for training.",
        )
        parser.add_argument(
            "-lr",
            "--learning-rate",
            type=float,
            default=1e-4,
            help="Learning rate (default: 1e-4).",
        )
        parser.add_argument(
            "-w",
            "--num-workers",
            type=int,
            default=4,
            help="Number of worker processes for data loading (default: 4).",
        )
        parser.add_argument(
            "-vs",
            "--val-split",
            type=float,
            default=0.1,
            help="Fraction of dataset to use for validation (default: 0.1).",
        )
        parser.add_argument(
            "-c",
            "--channels",
            type=int,
            nargs=4,
            default=[32, 64, 128, 256],
            metavar=("C1", "C2", "C3", "C4"),
            help="Encoder channel sizes (default: 32 64 128 256). Use 64 128 256 512 for higher quality.",
        )

    def execute(self, args: argparse.Namespace) -> None:
        pages_dir = Path.absolute(args.pages_dir)
        epochs = args.epochs
        batch_size = args.batch_size
        lr = args.learning_rate
        num_workers = args.num_workers
        val_split = args.val_split
        run(
            pages_dir,
            val_split,
            epochs,
            batch_size,
            lr,
            num_workers,
            args.channels,
            args.no_load,
        )
