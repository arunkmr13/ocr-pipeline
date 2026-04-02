import cv2
import numpy as np
import pytesseract
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
import base64, time

from pipeline.preprocessor import ImagePreprocessor
from pipeline.layout_analyzer import LayoutAnalyzer, RegionType
from pipeline.ocr_engine import PrintedOCREngine
from pipeline.overlay_renderer import OverlayRenderer, TextOverlay
from models.model_registry import ModelRegistry

app = FastAPI(title="OCR Translation Pipeline")
registry = ModelRegistry()

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("templates/index.html") as f:
        return HTMLResponse(f.read())

@app.post("/extract-and-translate")
async def extract_and_translate(file: UploadFile = File(...)):
    start = time.perf_counter()

    # Load image
    raw = await file.read()
    np_arr = np.frombuffer(raw, np.uint8)
    image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    pil_image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    # Stage 1: Preprocess
    preprocessor = ImagePreprocessor()
    processed_bgr, _ = preprocessor.process(image_bgr)
    processed_pil = Image.fromarray(cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB))

    # Stage 2: Layout analysis
    layout = registry.layout_analyzer.analyze(processed_pil)

    # Stage 3: Detect language on full image first
    try:
        quick_text = pytesseract.image_to_string(
            processed_pil, config='--oem 3 --psm 3'
        )
        src_lang = registry.lang_detector.top_language(quick_text) if quick_text.strip() else "en"
    except Exception:
        src_lang = "en"

    # Stage 4: OCR each region with correct language
    all_text_blocks = []
    for region in layout:
        if region.region_type not in (RegionType.TEXT, RegionType.TITLE):
            continue
        crop_arr = np.array(region.crop.convert("RGB"))[:, :, ::-1]
        results = registry.printed_ocr.extract(crop_arr, lang=src_lang)
        text = " ".join(r.text for r in results).strip()
        if text:
            all_text_blocks.append({"text": text, "bbox": region.bbox})

    # Stage 5: Translate
    texts = [b["text"] for b in all_text_blocks if b["text"]]
    overlays = []
    if texts:
        if src_lang != "en":
            translated = registry.translator.translate(texts, src_lang)
        else:
            translated = texts
        for block, trans in zip(all_text_blocks, translated):
            block["translated"] = trans
            overlays.append(TextOverlay(
                bbox=block["bbox"],
                original_text=block["text"],
                translated_text=trans,
                is_rtl=src_lang in ("ar", "he", "fa", "ur")
            ))
    else:
        for block in all_text_blocks:
            block["translated"] = block["text"]

    # Stage 6: Render overlay
    rendered = registry.overlay_renderer.render(processed_bgr, overlays)

    # Encode output
    _, buffer = cv2.imencode(".png", rendered)
    img_b64 = base64.b64encode(buffer).decode()
    elapsed = time.perf_counter() - start

    return JSONResponse({
        "success": True,
        "processing_time_seconds": round(elapsed, 2),
        "detected_language": src_lang,
        "text_blocks": all_text_blocks,
        "rendered_image_base64": img_b64,
    })


def _classify_handwritten(crop) -> bool:
    try:
        gray = np.array(crop.convert("L"))
        if gray.size == 0 or gray.shape[0] < 10 or gray.shape[1] < 10:
            return False
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dist = cv2.distanceTransform(bw, cv2.DIST_L2, 5)
        if dist is None:
            return False
        stroke_widths = dist[dist > 0]
        if len(stroke_widths) < 50:
            return False
        cv_score = np.std(stroke_widths) / (np.mean(stroke_widths) + 1e-6)
        return float(cv_score) > 0.45
    except Exception:
        return False