import tarfile

from lineremovernn.utils import logging

from .dataset import Dataset

logger = logging.get_logger("IAM")


class IAM_Dataset(Dataset):
    ID = "IAM"

    def download(self, path, force):
        dataset_path = path / self.DOWNLOAD_PATH / "IAM_Words"

        if not (dataset_path / "words.tgz").exists() or force:
            logger.info("Downloading IAM dataset...")
            self._download_and_unzip("https://git.io/J0fjL", path / self.DOWNLOAD_PATH)
        else:
            raise FileExistsError(
                f"Dataset already exists at {dataset_path}. Use force=True to re-download."
            )

    def extract(self, path, force):
        dataset_path = path / self.DOWNLOAD_PATH / "IAM_Words"
        logger.info("Extracting words.tgz...")
        if not (path / self.INSTALL_PATH / "words").exists() or force:
            with tarfile.open(dataset_path / "words.tgz") as f:
                f.extractall(path / self.INSTALL_PATH / "words")
        else:
            raise FileExistsError(
                f"Extracted dataset already exists at {path / self.INSTALL_PATH / 'words'}. Use force=True to re-extract."
            )

        logger.info("Moving the words.txt file to the dataset root...")
        if not (path / self.INSTALL_PATH / "words.txt").exists() or force:
            (dataset_path / "words.txt").rename(path / self.INSTALL_PATH / "words.txt")
        else:
            raise FileExistsError(
                f"words.txt already exists at {path / self.INSTALL_PATH / 'words.txt'}. Use force=True to re-move."
            )

        logger.info("Done.")
