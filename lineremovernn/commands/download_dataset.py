import pathlib

from lineremovernn.commands.command import Command


class DownloadDatasetCommand(Command):
    def __init__(self):
        super().__init__(
            name="download-dataset",
            description="Download the IAM dataset for generating training data.",
        )

    def init_parser(self, parser):
        parser.add_argument(
            "--output-dir",
            type=pathlib.Path,
            required=False,
            default="data/iam",
            help="Directory to save the downloaded dataset.",
        )

    def execute(self, args):
        output_dir: pathlib.Path = args.output_dir
        print(output_dir.absolute())
