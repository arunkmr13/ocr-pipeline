import pytesseract
import cv2
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import List

from langdetect import detect as detect_lang, LangDetectException

LANG_TO_TESSERACT = {
    "ja": "jpn", "ko": "kor", "zh": "chi_sim",
    "hi": "hin", "kn": "kan", "ta": "tam",
    "te": "tel", "th": "tha", "ar": "ara",
    "ru": "rus", "de": "deu", "fr": "fra",
    "es": "spa", "en": "eng",
}

@dataclass
class OCRResult:
    text: str
    bbox: tuple
    confidence: float

def detect_language(text: str) -> str:
    try:
        text = text.replace("\n", " ").strip()
        if not text or len(text) < 3:
            return "en"
        return detect_lang(text)
    except LangDetectException:
        return "en"


class PrintedOCREngine:
    def extract(self, image: np.ndarray,
                lang: str = "en") -> List[OCRResult]:
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        tess_lang = LANG_TO_TESSERACT.get(lang, "eng")
        try:
            data = pytesseract.image_to_data(
                pil_img,
                lang=tess_lang,
                output_type=pytesseract.Output.DICT,
                config='--oem 3 --psm 3'
            )
        except Exception:
            data = pytesseract.image_to_data(
                pil_img,
                lang="eng",
                output_type=pytesseract.Output.DICT,
                config='--oem 3 --psm 3'
            )
        results = []
        for i, text in enumerate(data['text']):
            text = text.strip()
            if not text:
                continue
            conf = float(data['conf'][i])
            if conf < 20:
                continue
            x, y, w, h = (data['left'][i], data['top'][i],data['width'][i], data['height'][i])
            results.append(OCRResult(
                text=text,
                bbox=(x, y, x + w, y + h),
                confidence=conf / 100
            ))
        return results

class HandwrittenOCREngine:
    def extract_region(self, pil_image: Image.Image) -> str:
        return pytesseract.image_to_string(
            pil_image, config='--oem 3 --psm 6'
        ).strip()