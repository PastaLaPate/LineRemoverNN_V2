import argparse
import random
import tempfile
from pathlib import Path

from PIL import Image

from lineremovernn.commands.command import Command
from lineremovernn.model import infer
from lineremovernn.model.model import LineRemoverNN
from lineremovernn.model.models import get_latest_model, load_model
from lineremovernn.utils import logging
from lineremovernn.utils.consts import DEFAULT_MODELS, DEFAULT_PAGES, DEVICE

logger = logging.get_logger("ModelLister")


class InferCommand(Command):
    def __init__(self) -> None:
        super().__init__(
            "test",
            "Run inference with the line remover model on some test images.",
        )

    def init_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--models-dir",
            type=Path,
            required=False,
            default=DEFAULT_MODELS,
            help="Path to the directory containing saved models.",
        )
        parser.add_argument(
            "--pages-dir",
            type=Path,
            required=False,
            default=DEFAULT_PAGES,
            help="Path to the directory containing the pages.",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            required=False,
            default=16,
            help="Batch size for training.",
        )

        parser.add_argument(
            "--n",
            type=int,
            default=5,
            help="Number of images to run inference on (default: 5).",
        )
        parser.add_argument(
            "--channels",
            type=int,
            nargs=4,
            default=[32, 64, 128, 256],
            metavar=("C1", "C2", "C3", "C4"),
            help="Encoder channel sizes (default: 32 64 128 256). Use 64 128 256 512 for higher quality.",
        )

    def execute(self, args: argparse.Namespace) -> None:
        pages_dir = Path.absolute(args.pages_dir)
        channels = args.channels
        n = args.n
        model_l = get_latest_model()
        if model_l is None:
            logger.error("No model found in %s", args.models_dir)
            return
        logger.info("Using model: %s", model_l[1])
        stats, path = model_l
        logger.info(
            "Loading model from %s (epoch %d, loss %.4f)", path, stats.epoch, stats.loss
        )
        saved = load_model(path)
        model = LineRemoverNN(channels).to(DEVICE)
        model.load_state_dict(saved.model_state)
        ruled_dir = pages_dir / "ruled"
        blank_dir = pages_dir / "blank"
        ruled_paths = sorted(
            ruled_dir.glob("*.jpg"), key=lambda p: int(p.name.split("-")[0])
        )
        samples = random.sample(ruled_paths, min(n, len(ruled_paths)))

        for ruled_path in samples:
            stem = ruled_path.stem
            blank_path = blank_dir / ruled_path.name
            output_path = Path(tempfile.gettempdir()) / f"{stem}_clean.jpg"

            infer.infer(model, ruled_path, output_path)

            # Side-by-side comparison
            if blank_path.exists():
                try:
                    import matplotlib.pyplot as plt

                    ruled_img = Image.open(ruled_path).convert("L")
                    clean_img = Image.open(output_path).convert("L")
                    blank_img = Image.open(blank_path).convert("L")

                    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                    axes[0].imshow(ruled_img, cmap="gray")
                    axes[0].set_title("Ruled (input)")
                    axes[1].imshow(clean_img, cmap="gray")
                    axes[1].set_title("Predicted (clean)")
                    axes[2].imshow(blank_img, cmap="gray")
                    axes[2].set_title("Ground truth")
                    for ax in axes:
                        ax.axis("off")
                    plt.tight_layout()
                    plt.savefig(
                        Path(tempfile.gettempdir()) / f"{stem}_compare.jpg", dpi=100
                    )
                    plt.close()
                except ImportError:
                    logger.warning(
                        "matplotlib not installed, skipping comparison plots."
                    )
