from pathlib import Path

from lineremovernn.commands.command import Command
from lineremovernn.data.datasets import DATASETS
from lineremovernn.utils import logging
from lineremovernn.utils.consts import SAVED

logger = logging.get_logger("DatasetDownloader")


class DownloadDatasetCommand(Command):
    def __init__(self):
        super().__init__(
            name="download-dataset",
            description="Download the IAM dataset for generating training data.",
        )

    def init_parser(self, parser):
        parser.add_argument(
            "-o",
            "--output-dir",
            type=Path,
            required=False,
            default="DEFAULT",
            help="Directory to save the downloaded dataset.",
        )
        parser.add_argument(
            "-fd",
            "--force-download",
            action="store_true",
            help="Force re-download even if the dataset already exists.",
        )
        parser.add_argument(
            "-fe",
            "--force-extract",
            action="store_true",
            help="Force re-extract even if the dataset has been extracted.",
        )
        parser.add_argument(
            "-d",
            "--dataset",
            type=str,
            choices=[x.lower() for x in DATASETS.keys()],
            default="iam",
            help="Which dataset to download (default: iam).",
        )

    def execute(self, args):
        install_dir = (
            Path.absolute(SAVED / "datasets")
            if args.output_dir == Path("DEFAULT")
            else Path.absolute(args.output_dir)
        )
        install_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Installing the %s dataset to %s", args.dataset, install_dir)
        dataset = DATASETS[args.dataset.upper()]
        try:
            dataset.download(install_dir, args.force_download)
        except FileExistsError:
            logger.warning(
                "Dataset already exists. Use --force-download to re-download."
            )
        try:
            dataset.extract(install_dir, args.force_extract)
        except FileExistsError:
            logger.warning(
                "Dataset already extracted. Use --force-extract to re-extract."
            )
