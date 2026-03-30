from .download_dataset import DownloadDatasetCommand
from .generate_pages import GeneratePagesCommand
from .preview_dataset import PreviewDataset

commands = [
    DownloadDatasetCommand(),
    GeneratePagesCommand(),
    PreviewDataset(),
]

__all__ = [
    "DownloadDatasetCommand",
    "GeneratePagesCommand",
    "PreviewDataset",
]
