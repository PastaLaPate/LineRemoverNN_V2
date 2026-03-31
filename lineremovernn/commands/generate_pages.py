from pathlib import Path

from lineremovernn.commands.command import Command
from lineremovernn.data.page_generator import PageGenerator
from lineremovernn.utils import logging
from lineremovernn.utils.consts import DEFAULT_IAM, DEFAULT_PAGES, ROOT

logger = logging.get_logger("DatasetDownloader")


class GeneratePagesCommand(Command):
    def __init__(self):
        super().__init__(
            name="generate-pages",
            description="Generate ruled and blank pages from the IAM dataset.",
        )

    def init_parser(self, parser):
        parser.add_argument(
            "--dataset-dir",
            type=Path,
            required=False,
            default=DEFAULT_IAM,
            help="Directory where is the downloaded dataset.",
        )
        parser.add_argument(
            "--output-dir",
            type=Path,
            required=False,
            default=DEFAULT_PAGES,
            help="Directory to save the generated pages.",
        )
        parser.add_argument(
            "--n",
            type=int,
            required=False,
            default=50,
            help="Number of pages to generate.",
        )
        parser.add_argument(
            "--disable-preload",
            action="store_true",
            default=False,
            help="Preload all images into memory before generating pages (faster but uses more RAM). You can disable this if you have a very large dataset and limited RAM.",
        )

    def execute(self, args):
        dataset_dir = Path.absolute(ROOT / args.dataset_dir)
        output_dir = Path.absolute(ROOT / args.output_dir)

        logger.info("Generating pages from dataset in %s", dataset_dir)
        generator = PageGenerator(dataset_dir)
        generator.generate(args.n, output_dir, preload=not args.disable_preload)

        logger.info("Done.")
