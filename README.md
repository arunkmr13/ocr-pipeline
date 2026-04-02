# OCR Translation Pipeline

A production-grade OCR pipeline that extracts text from images and translates it into English.

## Features
- Multi-language text extraction (80+ languages via Tesseract)
- Auto language detection
- Google Translate integration for fast, accurate translation
- Text overlay rendering on output image
- Clean web UI with drag and drop upload
- Side-by-side original vs translated image view
- Download translated output image

## Tech Stack
- **Backend**: Python, FastAPI
- **OCR**: Tesseract + pytesseract
- **Language Detection**: langdetect
- **Translation**: Google Translate (deep-translator)
- **Image Processing**: OpenCV, Pillow
- **Frontend**: HTML, CSS, Vanilla JS

## Setup
```bash
git clone https://github.com/arunkmr13/ocr-pipeline.git
cd ocr-pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
brew install tesseract tesseract-lang
```

## Run
```bash
PYTHONPATH=. uvicorn api.main:app --reload --port 8001
```

Then open http://127.0.0.1:8001
