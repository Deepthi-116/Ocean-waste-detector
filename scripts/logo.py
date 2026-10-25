"""Stamps a logo on video frames. It is applied after detection, so it never affects the results."""
import os
from pathlib import Path

import cv2
import numpy as np

LOGO_PATH = Path(os.getenv("LOGO_PATH", Path(__file__).resolve().parent / "logo.png"))
LOGO_WIDTH = float(os.getenv("LOGO_WIDTH", "0.12"))        # logo width as a fraction of the video width
LOGO_CORNER = os.getenv("LOGO_CORNER", "top-right")        # top-left, top-right, bottom-left, bottom-right
MARGIN = 12
_cache = {}


def _load(frame_w):
    if frame_w not in _cache:
        img = cv2.imread(str(LOGO_PATH), cv2.IMREAD_UNCHANGED) if LOGO_PATH.exists() else None
        if img is None:
            _cache[frame_w] = None
        else:
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            w = max(16, int(frame_w * LOGO_WIDTH))
            h = max(1, int(img.shape[0] * w / img.shape[1]))
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            _cache[frame_w] = (img[:, :, :3].astype(np.float32), img[:, :, 3:4].astype(np.float32) / 255.0)
    return _cache[frame_w]


def stamp(frame):
    logo = _load(frame.shape[1])
    if logo is None:
        return frame                                        # no logo file: leave the frame unchanged
    rgb, alpha = logo
    h, w = rgb.shape[:2]
    H, W = frame.shape[:2]
    if h + MARGIN > H or w + MARGIN > W:
        return frame
    x = W - w - MARGIN if "right" in LOGO_CORNER else MARGIN
    y = H - h - MARGIN if "bottom" in LOGO_CORNER else MARGIN
    roi = frame[y:y + h, x:x + w].astype(np.float32)
    frame[y:y + h, x:x + w] = (alpha * rgb + (1 - alpha) * roi).astype(np.uint8)
    return frame