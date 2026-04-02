from deep_translator import GoogleTranslator
from langdetect import detect as detect_lang, LangDetectException
from typing import List


class LanguageDetector:
    def detect(self, text: str) -> str:
        try:
            text = text.replace("\n", " ").strip()
            if not text or len(text) < 3:
                return "en"
            return detect_lang(text)
        except LangDetectException:
            return "en"

    def top_language(self, text: str) -> str:
        return self.detect(text)


class GoogleTranslatorEngine:
    def translate(self, texts: List[str], source_lang: str,
                  target_lang: str = "en") -> List[str]:
        results = []
        for text in texts:
            try:
                if not text.strip() or source_lang == "en":
                    results.append(text)
                    continue
                translated = GoogleTranslator(
                    source=source_lang,
                    target=target_lang
                ).translate(text)
                results.append(translated or text)
            except Exception:
                results.append(text)
        return results