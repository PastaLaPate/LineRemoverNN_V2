import tarfile
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import tqdm

from lineremovernn.commands.command import Command
from lineremovernn.utils import logging
from lineremovernn.utils.consts import ROOT

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
        parser.add_argument(
            
        )

    def execute(self, args):
        install_dir = Path.absolute(ROOT / args.output_dir)
        logger.info("Installing the IAM dataset to %s", install_dir)
        
        logger.warning(
            "Dataset already exists. Use --force-download to re-download."
        )


