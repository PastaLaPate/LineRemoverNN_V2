from pathlib import Path

from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, decode_image


class PagesDataset(Dataset):
    def __init__(self, pages_dir: Path, transform=None):
        self.pages_dir = pages_dir
        if not self.pages_dir.exists():
            raise ValueError(f"Pages directory {self.pages_dir} does not exist.")
        self.blank_dir = self.pages_dir / "blank"
        self.ruled_dir = self.pages_dir / "ruled"
        if not self.blank_dir.exists() or not self.ruled_dir.exists():
            raise ValueError(
                f"Pages directory {self.pages_dir} must contain 'blank' and 'ruled' subdirectories."
            )
        self.blank_paths = sorted(
            self.blank_dir.glob("*.jpg"), key=lambda p: int(p.name.split("-")[0])
        )
        self.ruled_paths = sorted(
            self.ruled_dir.glob("*.jpg"), key=lambda p: int(p.name.split("-")[0])
        )

        if len(self.blank_paths) != len(self.ruled_paths):
            raise ValueError(
                f"Pages directory {self.pages_dir} must contain the same number of blank and ruled pages."
            )

        self.transform = transform

    def __len__(self):
        return len(self.blank_paths)

    def __getitem__(self, index: int):
        blank_path = self.blank_paths[index]
        ruled_path = self.ruled_paths[index]
        blank = decode_image(str(blank_path), ImageReadMode.GRAY).float() / 255.0
        ruled = decode_image(str(ruled_path), ImageReadMode.GRAY).float() / 255.0
        if self.transform:
            blank, ruled = self.transform(blank, ruled)
        return blank, ruled
