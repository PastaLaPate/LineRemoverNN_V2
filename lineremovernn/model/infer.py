from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.io import ImageReadMode, decode_image

from lineremovernn.model.model import LineRemoverNN
from lineremovernn.utils import logging
from lineremovernn.utils.consts import DEVICE
from lineremovernn.utils.tiles import TILE_SIZE

OVERLAP = 64

logger = logging.get_logger("Infer")


def infer(model: LineRemoverNN, image_path: Path, output_path: Path) -> None:
    img = decode_image(str(image_path), ImageReadMode.GRAY).float() / 255.0
    _, h, w = img.shape

    stride = TILE_SIZE - OVERLAP
    output = torch.zeros(1, h, w)
    weight = torch.zeros(1, h, w)

    ys = list(range(0, max(1, h - TILE_SIZE), stride)) + [max(0, h - TILE_SIZE)]
    xs = list(range(0, max(1, w - TILE_SIZE), stride)) + [max(0, w - TILE_SIZE)]

    model.eval()
    with torch.no_grad():
        for y in dict.fromkeys(ys):
            for x in dict.fromkeys(xs):
                pad_h = max(0, TILE_SIZE - (h - y))
                pad_w = max(0, TILE_SIZE - (w - x))
                tile = img[:, y : y + TILE_SIZE, x : x + TILE_SIZE]
                if pad_h > 0 or pad_w > 0:
                    tile = F.pad(tile, [0, pad_w, 0, pad_h])

                pred = model(tile.unsqueeze(0).to(DEVICE)).squeeze(0).cpu()

                th = min(TILE_SIZE, h - y)
                tw = min(TILE_SIZE, w - x)
                output[:, y : y + th, x : x + tw] += pred[:, :th, :tw]
                weight[:, y : y + th, x : x + tw] += 1

    result = (output / weight.clamp(min=1)).clamp(0, 1)
    arr = (result.squeeze().numpy() * 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(output_path)
    logger.info("Saved to %s", output_path)
