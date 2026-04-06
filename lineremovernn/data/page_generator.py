import json
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageOps
from PIL import Image
from tqdm import tqdm

from lineremovernn.utils.logging import get_logger

logger = get_logger("PageGenerator")

# --- Worker globals ---
_words: list[tuple[str, tuple[int, int, int, int], str, int]] = []
_image_cache: dict[str, bytes] = {}


def _init_worker(
    words: list[tuple[str, tuple[int, int, int, int], str, int]],
    image_cache: dict[str, bytes],
) -> None:
    global _words, _image_cache
    _words = words
    _image_cache = image_cache


def _make_transparent_bg(img: Image.Image) -> Image.Image:
    img_data = np.array(img)
    avg = img_data[..., :3].mean(axis=2)
    img_data[..., 3] = (avg <= 200) * 255
    return PIL.Image.fromarray(img_data, "RGBA")


def _load_image(path: str) -> Image.Image | None:
    data = _image_cache.get(path)
    try:
        if data is not None:
            img = PIL.Image.open(BytesIO(data)).convert("RGBA")
        else:
            img = PIL.Image.open(path).convert("RGBA")
        return _make_transparent_bg(img)
    except Exception as e:
        logger.error("Error loading image %s: %s", path, e)
        return None


def _draw_lines(
    lines_draw: PIL.ImageDraw.ImageDraw,
    lines: list[int],
    page_width: int,
    page_height: int,
    small_lines_by_line: int,
    small_line_size: float,
) -> None:
    for y in lines:
        for j in range(small_lines_by_line):
            offset = random.randint(-5, 5)
            line_y = y + small_line_size * j + offset
            line_width = (
                random.randint(1, 3) if j in (0, small_lines_by_line - 1) else 5
            )
            lines_draw.line(
                [(0, line_y), (page_width, line_y)],
                fill=random.randint(100, 255),
                width=line_width,
            )

    for j in range(page_width // 60):
        vx = j * 60
        lines_draw.line(
            [(vx, 0), (vx, page_height)],
            fill=random.randint(100, 255),
            width=random.randint(1, 3),
        )


def _make_page_worker(image_index: int, output_dir: Path) -> None:
    pages_dir = output_dir / "ruled"
    nolines_dir = output_dir / "blank"
    json_dir = output_dir / "json"

    for d in (pages_dir, nolines_dir, json_dir):
        d.mkdir(parents=True, exist_ok=True)

    page_width = random.randint(PageGenerator.MIN_PAGE_SIZE, 2000)
    page_height = random.randint(PageGenerator.MIN_PAGE_SIZE, 2000)

    page = PIL.Image.new("RGBA", (page_width, page_height), (255, 255, 255))
    lines_img = PIL.Image.new("L", (page_width, page_height), 0)
    lines_draw = PIL.ImageDraw.Draw(lines_img)

    margin_x: int = random.randint(30, 50)
    x: int = margin_x
    y: int = random.randint(30, 90)
    line_size: int = random.randint(120, 200)
    small_lines_by_line: int = random.randint(4, 7)
    small_line_size: float = line_size / small_lines_by_line
    stop_x: int = page_width - margin_x

    lines: list[int] = []
    words_meta: list[dict] = []
    max_h: int = 0

    i = random.randint(0, len(_words))
    while True:
        word = _words[i % len(_words)]
        path, _, transcript, _ = word

        word_img = _load_image(path)
        if word_img is None or word_img.width == 0 or word_img.height == 0:
            i += 1
            continue

        w = int(random.uniform(0.7, 1.3) * word_img.width)
        h = int(random.uniform(0.7, 1.3) * word_img.height)
        if w <= 0 or h <= 0:
            i += 1
            continue

        word_img = word_img.resize((w, h))
        i += 1
        max_h = max(max_h, h)

        if x + w >= stop_x:
            x = random.randint(30, 50)
            stop_x = page_width - random.randint(30, 50)
            lines.append(y)
            y += max(max_h, line_size)
            max_h = 0

        if y >= page_height - line_size:
            break

        rn = random.randint(-20, 20)
        words_meta.append({"text": transcript, "x": x, "y": y - h + rn, "w": w, "h": h})
        page.paste(word_img, (x, y - h + rn), word_img)
        x += w + random.randint(-10, 30)

    page.convert("RGB").save(nolines_dir / f"{image_index}-page.jpg", quality=95)

    with open(json_dir / f"{image_index}.json", "w") as f:
        json.dump(words_meta, f)

    _draw_lines(
        lines_draw, lines, page_width, page_height, small_lines_by_line, small_line_size
    )
    np_lines = np.array(lines_img)
    blank_np = np.array(page)
    ruled_np = np.minimum(blank_np, np_lines)
    ruled_page = PIL.Image.fromarray(ruled_np)
    ruled_page.convert("RGB").save(pages_dir / f"{image_index}-page.jpg", quality=95)


class PageGenerator:
    MIN_PAGE_SIZE = 300

    def __init__(self, dataset_dir: Path) -> None:
        self.dataset_dir = dataset_dir
        self.words: list[tuple[str, tuple[int, int, int, int], str, int]] = []
        self.load_words()

    def load_words(self) -> None:
        words_file = self.dataset_dir / "words.txt"
        with open(words_file, encoding="UTF-8") as f:
            for line in f:
                line = line.rstrip()
                if line.startswith("#") or len(line.split(" ")) != 9:
                    continue
                filename, segmentation, gray, x, y, w, h, typ, transcript = line.split(
                    " "
                )
                if segmentation == "err":
                    continue
                parts = filename.split("-")
                path = (
                    self.dataset_dir
                    / "words"
                    / parts[0]
                    / "-".join(parts[:2])
                    / f"{filename}.png"
                )
                self.words.append(
                    (str(path), (int(x), int(y), int(w), int(h)), transcript, int(gray))
                )

    def _preload_images(self) -> dict[str, bytes]:
        cache: dict[str, bytes] = {}
        for path, _, _, _ in tqdm(self.words, desc="Preloading images", unit="img"):
            if path in cache:
                continue
            try:
                with open(path, "rb") as f:
                    cache[path] = f.read()
            except Exception:
                pass
        logger.info(
            "Preloaded %d images (%.1f MB)",
            len(cache),
            sum(len(v) for v in cache.values()) / 1e6,
        )
        return cache

    def generate(
        self,
        count: int,
        output_dir: Path,
        workers: int = os.cpu_count() or 1,
        preload: bool = False,
    ) -> None:
        image_cache: dict[str, bytes] = self._preload_images() if preload else {}

        logger.info(
            "Generating %d pages — workers=%d preload=%s",
            count,
            workers,
            preload,
        )

        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(self.words, image_cache),
        ) as executor:
            futures = {
                executor.submit(_make_page_worker, i, output_dir): i
                for i in range(count)
            }
            for future in tqdm(
                as_completed(futures), total=count, desc="Generating pages", unit="page"
            ):
                i = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error("Page %d failed: %s", i, e)
