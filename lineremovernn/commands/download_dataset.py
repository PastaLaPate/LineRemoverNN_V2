from pathlib import Path

from lineremovernn.commands.command import Command
from lineremovernn.utils import logging
from lineremovernn.utils.consts import ROOT, SAVED

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
            "-d",
            "--force-download",
            action="store_true",
            help="Force re-download even if the dataset already exists.",
        )
        parser.add_argument(
            "-e",
            "--force-extract",
            action="store_true",
            help="Force re-extract even if the dataset has been extracted.",
        )
        parser.add_argument()

    def execute(self, args):
        install_dir = (
            Path.absolute(SAVED / args.output_dir)
            if args.output_dir.is_relative_to(ROOT)
            else Path.absolute(args.output_dir)
        )
        install_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Installing the IAM dataset to %s", install_dir)

        logger.warning("Dataset already exists. Use --force-download to re-download.")
