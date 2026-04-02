from abc import ABC
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import tqdm


class Dataset(ABC):
    ID = "XXX"

    def __init__(self) -> None:
        self.INSTALL_PATH = Path(self.ID)
        self.DOWNLOAD_PATH = Path("downloads") / self.ID
        super().__init__()

    def download(self, path: Path, force: bool) -> None:
        """Download the dataset to the specified path."""
        raise NotImplementedError("Subclasses must implement this method.")

    def extract(self, path: Path, force: bool) -> None:
        """Extract the dataset from the specified path."""
        raise NotImplementedError("Subclasses must implement this method.")

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
