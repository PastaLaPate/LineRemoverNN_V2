import argparse
import random
from pathlib import Path

from lineremovernn.commands.command import Command
from lineremovernn.data.dataset import PagesDataset
from lineremovernn.utils import logging
from lineremovernn.utils.consts import DEFAULT_PAGES, ROOT

logger = logging.get_logger("DatasetPreviewer")


class PreviewDataset(Command):
    def __init__(self) -> None:
        super().__init__(
            "preview-dataset",
            "Preview the generated dataset by displaying random page pairs.",
        )

    def init_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--dataset-dir",
            type=Path,
            required=False,
            default=DEFAULT_PAGES,
            help="Path to the dataset directory containing 'blank' and 'ruled' subdirectories.",
        )
        parser.add_argument(
            "--count",
            type=int,
            default=5,
            help="Number of random page pairs to preview (default: 5).",
        )

    def execute(self, args: argparse.Namespace) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("matplotlib is required for preview: pip install matplotlib")
            return

        dataset = PagesDataset(ROOT / args.dataset_dir)
        indices = random.sample(range(len(dataset)), min(args.count, len(dataset)))

        fig, axes = plt.subplots(args.count, 2, figsize=(12, 5 * args.count))
        if args.count == 1:
            axes = [axes]

        axes[0][0].set_title("Blank", fontsize=14)
        axes[0][1].set_title("Ruled", fontsize=14)

        for row, idx in enumerate(indices):
            blank, ruled = dataset[idx]
            # decode_image returns CHW uint8 tensor, squeeze channel for grayscale
            axes[row][0].imshow(blank.squeeze().numpy(), cmap="gray")
            axes[row][1].imshow(ruled.squeeze().numpy(), cmap="gray")
            axes[row][0].axis("off")
            axes[row][1].axis("off")

        plt.tight_layout()
        plt.show()
