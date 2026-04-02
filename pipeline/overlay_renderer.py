import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from dataclasses import dataclass
from typing import List, Optional
import math


@dataclass
class TextOverlay:
    bbox: tuple
    original_text: str
    translated_text: str
    is_rtl: bool = False


class OverlayRenderer:

    def __init__(self, font_path: Optional[str] = None):
        self.font_path = font_path or self._find_system_font()

    def render(self, image: np.ndarray,
               overlays: List[TextOverlay]) -> np.ndarray:
        result = image.copy()
        for overlay in overlays:
            if not overlay.translated_text.strip():
                continue
            result = self._erase_text(result, overlay.bbox)
            font_size = self._estimate_font_size(
                overlay.bbox, overlay.translated_text
            )
            result = self._render_text(
                result, overlay.translated_text,
                overlay.bbox, font_size, overlay.is_rtl
            )
        return result

    def _erase_text(self, image: np.ndarray, bbox: tuple) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        pad = 3
        x1p = max(0, x1 - pad)
        y1p = max(0, y1 - pad)
        x2p = min(image.shape[1], x2 + pad)
        y2p = min(image.shape[0], y2 + pad)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[y1p:y2p, x1p:x2p] = 255
        return cv2.inpaint(image, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    def _estimate_font_size(self, bbox: tuple, text: str) -> int:
        x1, y1, x2, y2 = bbox
        box_w = x2 - x1
        box_h = y2 - y1
        if not text:
            return 12
        char_width_ratio = 0.6
        for size in range(40, 8, -1):
            chars_per_line = box_w / (size * char_width_ratio)
            lines_needed = math.ceil(len(text) / max(chars_per_line, 1))
            total_height = lines_needed * size * 1.3
            if total_height <= box_h and chars_per_line >= 1:
                return size
        return 9

    def _get_dominant_bg_color(self, image: np.ndarray,
                                bbox: tuple) -> tuple:
        x1, y1, x2, y2 = bbox
        pad = 5
        samples = []
        for sx, sy in [(x1 - pad, y1), (x2 + pad, y1),
                       (x1, y1 - pad), (x1, y2 + pad)]:
            sx = max(0, min(sx, image.shape[1] - 1))
            sy = max(0, min(sy, image.shape[0] - 1))
            samples.append(image[sy, sx])
        avg = np.mean(samples, axis=0).astype(int)
        return (int(avg[2]), int(avg[1]), int(avg[0]))

    def _get_text_color(self, bg_rgb: tuple) -> tuple:
        r, g, b = bg_rgb
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return (0, 0, 0) if luminance > 128 else (255, 255, 255)

    def _render_text(self, image: np.ndarray, text: str,
                     bbox: tuple, font_size: int,
                     rtl: bool = False) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)

        try:
            font = ImageFont.truetype(self.font_path, font_size)
        except Exception:
            font = ImageFont.load_default()

        bg_color = self._get_dominant_bg_color(image, bbox)
        text_color = self._get_text_color(bg_color)
        box_w = x2 - x1
        wrapped = self._wrap_text(text, font, box_w, draw)

        line_heights = [draw.textbbox((0, 0), line, font=font)[3]
                        for line in wrapped]
        total_h = sum(line_heights) + (len(wrapped) - 1) * 2
        y_start = y1 + max(0, (y2 - y1 - total_h) // 2)

        for i, line in enumerate(wrapped):
            lw = draw.textbbox((0, 0), line, font=font)[2]
            x_pos = x2 - lw if rtl else x1
            draw.text((x_pos, y_start), line, font=font, fill=text_color)
            y_start += line_heights[i] + 2

        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    def _wrap_text(self, text: str, font, max_width: int,
                   draw: ImageDraw.Draw) -> List[str]:
        words = text.split()
        lines, current = [], []
        for word in words:
            test = " ".join(current + [word])
            w = draw.textbbox((0, 0), test, font=font)[2]
            if w <= max_width or not current:
                current.append(word)
            else:
                lines.append(" ".join(current))
                current = [word]
        if current:
            lines.append(" ".join(current))
        return lines if lines else [text]

    def _find_system_font(self) -> str:
        import os
        candidates = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return ""