import torch
import torch.nn.functional as F

TILE_SIZE = 512
TILE_OVERLAP = 64


def extract_tiles(
    img: torch.Tensor, size: int, overlap: int
) -> tuple[list[torch.Tensor], list[tuple[int, int]], tuple[int, int]]:
    """Split a CHW tensor into overlapping tiles. Returns tiles, (y,x) positions, original shape."""
    _, h, w = img.shape
    tiles, positions = [], []
    stride = size - overlap
    ys = list(range(0, max(1, h - size), stride)) + [max(0, h - size)]
    xs = list(range(0, max(1, w - size), stride)) + [max(0, w - size)]
    for y in dict.fromkeys(ys):
        for x in dict.fromkeys(xs):
            tile = img[:, y : y + size, x : x + size]
            pad_h = size - tile.shape[1]
            pad_w = size - tile.shape[2]
            if pad_h > 0 or pad_w > 0:
                tile = F.pad(tile, [0, pad_w, 0, pad_h])
            tiles.append(tile)
            positions.append((y, x))
    return tiles, positions, (h, w)


def stitch_tiles(
    tiles: list[torch.Tensor],
    positions: list[tuple[int, int]],
    original_shape: tuple[int, int],
    size: int,
) -> torch.Tensor:
    """Reconstruct image from tiles using averaging in overlapping regions."""
    h, w = original_shape
    output = torch.zeros(1, h, w)
    weight = torch.zeros(1, h, w)
    for tile, (y, x) in zip(tiles, positions):
        th = min(size, h - y)
        tw = min(size, w - x)
        output[:, y : y + th, x : x + tw] += tile[:, :th, :tw]
        weight[:, y : y + th, x : x + tw] += 1
    return output / weight.clamp(min=1)
