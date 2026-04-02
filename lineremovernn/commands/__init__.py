from .download_dataset import DownloadDatasetCommand
from .generate_pages import GeneratePagesCommand
from .ls_models import ListModels
from .preview_dataset import PreviewDataset
from .test import InferCommand
from .train import TrainCommand

commands = [
    DownloadDatasetCommand(),
    GeneratePagesCommand(),
    PreviewDataset(),
    ListModels(),
    TrainCommand(),
    InferCommand(),
]

__all__ = [
    "DownloadDatasetCommand",
    "GeneratePagesCommand",
    "PreviewDataset",
    "ListModels",
    "TrainCommand",
    "InferCommand",
]
