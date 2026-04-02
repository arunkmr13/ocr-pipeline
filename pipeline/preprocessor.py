# pipeline/preprocessor.py
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from typing import Tuple

class ImagePreprocessor:
    """
    Production preprocessor. Applies a conservative pipeline:
    denoise → deskew → CLAHE contrast → resize normalization.
    All transforms are reversible so bounding boxes can be remapped.
    """

    def __init__(self, target_dpi: int = 300):
        self.target_dpi = target_dpi

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Returns preprocessed image + transform metadata for bbox remapping."""
        meta = {}

        # 1. Upscale low-res images to OCR-friendly DPI
        image, meta["scale"] = self._normalize_resolution(image)

        # 2. Denoise (Non-local means — preserves edges unlike Gaussian)
        image = self._denoise(image)

        # 3. Deskew
        image, meta["deskew_angle"] = self._deskew(image)

        # 4. CLAHE contrast enhancement (works per-channel, avoids blown highlights)
        image = self._enhance_contrast(image)

        return image, meta

    def _normalize_resolution(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        h, w = img.shape[:2]
        # Heuristic: assume source is 72 DPI if small
        if max(h, w) < 1000:
            scale = self.target_dpi / 72.0
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            return img, scale
        return img, 1.0

    def _denoise(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(img, None, 6, 6, 7, 21)
        return cv2.fastNlMeansDenoising(img, None, 6, 7, 21)

    def _deskew(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        # Otsu threshold + minAreaRect on contours
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 100:
            return img, 0.0
        angle = cv2.minAreaRect(coords)[-1]
        # Normalize angle to [-45, 45]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        if abs(angle) < 0.3:  # Skip trivial corrections
            return img, angle

        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated, angle

    def _enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        # CLAHE on L channel of LAB — preserves color while boosting local contrast
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    def preprocess_for_handwriting(self, img: np.ndarray) -> np.ndarray:
        """Separate path for handwritten regions — adaptive thresholding + stroke enhancement."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Adaptive threshold to isolate ink strokes
        adaptive = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        # Dilate slightly to reconnect broken strokes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        enhanced = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)