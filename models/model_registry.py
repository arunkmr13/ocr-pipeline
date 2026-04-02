import threading


class ModelRegistry:
    """Thread-safe lazy model loading."""
    _lock = threading.Lock()

    def __init__(self):
        self._models = {}

    def _load(self, key: str, loader):
        if key not in self._models:
            with self._lock:
                if key not in self._models:
                    self._models[key] = loader()
        return self._models[key]

    @property
    def layout_analyzer(self):
        from pipeline.layout_analyzer import LayoutAnalyzer
        return self._load("layout", lambda: LayoutAnalyzer())

    @property
    def printed_ocr(self):
        from pipeline.ocr_engine import PrintedOCREngine
        return self._load("ocr_printed", lambda: PrintedOCREngine())

    @property
    def hw_ocr(self):
        from pipeline.ocr_engine import HandwrittenOCREngine
        return self._load("ocr_hw", lambda: HandwrittenOCREngine())

    @property
    def lang_detector(self):
        from pipeline.translator import LanguageDetector
        return self._load("lang", lambda: LanguageDetector())

    @property
    def translator(self):
        from pipeline.translator import GoogleTranslatorEngine
        return self._load("translator", lambda: GoogleTranslatorEngine())

    @property
    def overlay_renderer(self):
        from pipeline.overlay_renderer import OverlayRenderer
        return self._load("renderer", lambda: OverlayRenderer())