import cv2
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import List
from enum import Enum


class RegionType(Enum):
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    TITLE = "title"
    HEADER = "header"
    FOOTER = "footer"


@dataclass
class DocumentRegion:
    bbox: tuple
    region_type: RegionType
    confidence: float
    crop: any = None


class LayoutAnalyzer:
    """
    Fast contour-based layout analyzer.
    No heavy models — runs in milliseconds on CPU.
    """

    def analyze(self, pil_image: Image.Image) -> List[DocumentRegion]:
        img = np.array(pil_image.convert("RGB"))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        # Adaptive threshold to find text regions
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Dilate to merge nearby text into blocks
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
        dilated = cv2.dilate(thresh, kernel, iterations=2)

        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        regions = []
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            # Filter out noise (too small or too large)
            if bw < 20 or bh < 10:
                continue
            if bw > w * 0.98 and bh > h * 0.98:
                continue

            bbox = (x, y, x + bw, y + bh)
            crop = pil_image.crop(bbox)
            region_type = self._classify_region(bw, bh, w, h, y)
            regions.append(DocumentRegion(
                bbox=bbox,
                region_type=region_type,
                confidence=0.85,
                crop=crop
            ))

        return self._sort_reading_order(regions)

    def _classify_region(self, bw, bh, img_w, img_h, y) -> RegionType:
        aspect = bw / (bh + 1e-6)
        # Wide short regions at top = likely title/header
        if y < img_h * 0.15 and aspect > 3:
            return RegionType.TITLE
        # Very wide regions = likely full text paragraph
        if bw > img_w * 0.6:
            return RegionType.TEXT
        # Roughly square-ish large regions = possible table
        if 0.5 < aspect < 3 and bw > img_w * 0.3 and bh > img_h * 0.1:
            return RegionType.TABLE
        return RegionType.TEXT

    def _sort_reading_order(self, regions: List[DocumentRegion]) -> List[DocumentRegion]:
        band_height = 50
        return sorted(regions, key=lambda r: (r.bbox[1] // band_height, r.bbox[0]))