"""Чтение/запись изображений.

PIL используется только здесь: декодирование/кодирование JPEG и EXIF-ориентация.
Вся обработка (анализ, геометрия, ресемплинг) выполняется собственным кодом на numpy.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps

from .imageops import to_uint8


def load_image(path: str) -> np.ndarray:
    """Загружает изображение как float32 RGB в диапазоне 0..1."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    return arr


def save_image(path: str, img: np.ndarray, quality: int = 95) -> None:
    Image.fromarray(to_uint8(img)).save(path, quality=quality, subsampling=1)


def image_size(path: str) -> tuple[int, int]:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return im.size
