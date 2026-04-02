import random
from pathlib import Path

from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms import functional as F
from torchvision.transforms.v2.functional import crop

from lineremovernn.utils.tiles import TILE_SIZE


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
        blank = (
            decode_image(str(self.blank_paths[index]), ImageReadMode.GRAY).float()
            / 255.0
        )
        ruled = (
            decode_image(str(self.ruled_paths[index]), ImageReadMode.GRAY).float()
            / 255.0
        )

        _, h, w = ruled.shape

        pad_h = max(0, TILE_SIZE - h)
        pad_w = max(0, TILE_SIZE - w)
        if pad_h > 0 or pad_w > 0:
            blank = F.pad(blank, [0, pad_w, 0, pad_h])
            ruled = F.pad(ruled, [0, pad_w, 0, pad_h])
            h, w = h + pad_h, w + pad_w

        top = random.randint(0, h - TILE_SIZE)
        left = random.randint(0, w - TILE_SIZE)
        blank = crop(blank, top, left, TILE_SIZE, TILE_SIZE)
        ruled = crop(ruled, top, left, TILE_SIZE, TILE_SIZE)

        if self.transform:
            blank, ruled = self.transform(blank, ruled)

        return blank, ruled
