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
        dataset_path = install_dir / "IAM_Words"

        if not (dataset_path / "words.tgz").exists() or args.force_download:
            logger.info("Downloading IAM dataset...")
            self._download_and_unzip("https://git.io/J0fjL", install_dir)
        else:
            logger.warning(
                "Dataset already exists. Use --force-download to re-download."
            )

        logger.info("Extracting words.tgz...")
        if not (install_dir / "words").exists() or args.force_extract:
            with tarfile.open(dataset_path / "words.tgz") as f:
                f.extractall(install_dir / "words")
        else:
            logger.warning(
                "Extracted dataset already exists. Use --force-extract to re-extract."
            )

        logger.info("Moving the words.txt file to the dataset root...")
        if not (install_dir / "words.txt").exists() or args.force_extract:
            (dataset_path / "words.txt").rename(install_dir / "words.txt")
        else:
            logger.warning(
                "words.txt already exists. Use --force-extract to move it again."
            )

        logger.info("Done.")

    def _download_and_unzip(
        self, url: str, extract_to: Path, chunk_size: int = 1024 * 1024
    ) -> None:
        with urlopen(url) as response:
            total = response.length // chunk_size + 1
            data = b""
            for _ in tqdm.tqdm(range(total), desc="Downloading", unit="chunk"):
                data += response.read(chunk_size)

        with ZipFile(BytesIO(data)) as zf:
            zf.extractall(path=extract_to)
