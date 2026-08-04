# ARIA Local Backend

This is the local-first backend for ARIA. It accepts one or more images, runs a
replaceable local text detector and recognizer, optionally translates the
detected text with DeepL, removes the source text with OpenCV inpainting, and
renders the translated text into the detected regions.

## Requirements

- Python 3.10+
- Tesseract OCR with Japanese language data
- A working C/C++ build environment may be needed by some Python wheels on older systems

Install Tesseract on Windows from the UB Mannheim builds and make sure the
binary is on `PATH`. Install `jpn.traineddata` in Tesseract's `tessdata` folder.

For vertical Japanese, install `jpn_vert.traineddata` and set
`ARIA_OCR_LANG=jpn_vert` when processing pages that need it.

## Setup

```powershell
cd aria-local
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
uvicorn aria_local.main:app --reload --port 8000
```

The API is available at `http://127.0.0.1:8000`. Swagger documentation is at
`http://127.0.0.1:8000/docs`.

The frontend provider selectors call `/api/providers` and let each job choose a
detector and recognizer independently. The selected providers are stored on the
job, so changing the environment or starting another job does not change a job
already in progress.

Set `DEEPL_API_KEY` to use DeepL. Without it, the backend uses the original
Japanese text as a placeholder translation so the OCR, cleanup, and rendering
stages can still be tested locally.

If Tesseract is installed but not on `PATH`, set its executable explicitly:

```powershell
$env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

Useful settings:

```powershell
$env:ARIA_OCR_LANG = "jpn"
$env:ARIA_TRANSLATION_PROVIDER = "deepl"
$env:ARIA_TEXT_DETECTOR = "tesseract"
$env:ARIA_TEXT_RECOGNIZER = "tesseract"
$env:ARIA_PADDLEOCR_ENABLE_MKLDNN = "0"
$env:ARIA_PADDLEOCR_LIMIT_SIDE_LEN = "960"
$env:ARIA_DATA_DIR = ".aria-data"
$env:ARIA_FONT_PATH = "C:\Windows\Fonts\arial.ttf"
$env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

The backend stores originals, rendered images, and job state (including OCR
regions) under `.aria-data/`. This directory is intentionally local and should
not be committed.

Completed pages can be manually cleaned in the frontend and rerendered without
running detection, recognition, or translation again. Enclosed bubble contours
are persisted so cleanup and body-centered text placement can reuse the detected
shape without repeating connected-component analysis.

Automatic cleanup first looks for white connected components. If that fails, a
stricter OCR-seeded tone detector can accept enclosed cream or gray bubbles with
dark outlines; unsupported regions remain untouched for manual cleanup.

## Provider Architecture

`aria_local.pipeline.process_page` keeps the text stages replaceable:

- `TextDetector` returns page-coordinate polygons and detector confidence.
- `TextRecognizer` returns one OCR result per detected region. Recognition
  confidence is optional because providers such as `manga-ocr` do not expose it.

The default adapter remains Tesseract. A future comic-specific detector can
implement the detector protocol without changing translation or rendering.

## Optional Providers

`manga-ocr` recognizes Japanese text from a detected region. It does not detect
page-level regions by itself, so pair it with Tesseract or PaddleOCR:

```powershell
pip install -e ".[manga-ocr]"
$env:ARIA_TEXT_DETECTOR = "tesseract"
$env:ARIA_TEXT_RECOGNIZER = "manga-ocr"
```

PaddleOCR provides page-level polygons and can be used as the detector. ARIA
uses PaddleOCR's detection-only API because Manga OCR handles recognition; this
avoids running two recognizers for every page. Its
PaddlePaddle runtime is platform-specific, so install the appropriate CPU or
GPU build first, then install the ARIA extra:

```powershell
# Install PaddlePaddle for your Python and hardware from its official guide.
pip install -e ".[paddleocr]"
$env:ARIA_TEXT_DETECTOR = "paddleocr"
$env:ARIA_TEXT_RECOGNIZER = "manga-ocr"
```

The first `manga-ocr` or PaddleOCR run may download model files and can require
several hundred megabytes. If an optional package is missing, the selected job
fails with an installation message; the backend itself still starts and the
Tesseract fallback remains available.

PaddleOCR uses plain CPU execution by default for Windows compatibility. If a
machine has a working oneDNN setup, `ARIA_PADDLEOCR_ENABLE_MKLDNN=1` enables
that acceleration. Lower `ARIA_PADDLEOCR_LIMIT_SIDE_LEN` to trade small-text
recall for faster detection on large pages.

For a faster but lower-recall detector, set
`ARIA_PADDLEOCR_DET_MODEL=PP-OCRv5_mobile_det`. The default
`PP-OCRv6_medium_det` is recommended when small text matters.

Argos Translate is the offline translation alternative. Install it with the
`argos` extra, and ARIA downloads the Japanese-to-English language package on
the first Argos job unless `ARIA_ARGOS_AUTO_INSTALL=0`. Argos is free and local,
but its general-purpose model will usually be less natural than DeepL for manga
dialogue.

Helsinki OPUS-MT is a higher-quality local translation option. It uses the
`Helsinki-NLP/opus-mt-ja-en` model and is preferred over Argos when available:

```powershell
pip install -e ".[helsinki]"
```

To use a GPU for PaddleOCR or Manga OCR, first install GPU-enabled packages:

```powershell
# PyTorch with CUDA (for Manga OCR)
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# PaddlePaddle with CUDA (reinstall if CPU version is present)
python -m pip install paddlepaddle-gpu
```

Then set the device:

```powershell
$env:ARIA_DEVICE = "gpu"
```

Manga OCR detects CUDA automatically when PyTorch is GPU-enabled.

All detector, recognizer, inpainting, and rendering work runs in this local
Python process. DeepL is the only external processing option, and it is used
only when selected and `DEEPL_API_KEY` is configured.

Detector polygons and provider metadata are persisted in each job's state.
