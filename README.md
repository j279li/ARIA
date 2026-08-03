# ARIA - Local Manga Translation Tool

ARIA translates Japanese manga pages locally through a browser-based interface.
It supports selecting or pasting multiple images, local OCR, optional DeepL
translation, text cleanup, and server-side rendering of translated pages.

## Current Workflow

1. Add one or more manga pages by selecting, pasting, or dragging images.
2. Start a local processing job.
3. Choose the local text detector and recognizer in the UI.
4. ARIA runs those providers locally and stores the detected regions.
5. ARIA optionally sends only the detected text to DeepL.
6. ARIA refines accepted OCR regions to dark glyph masks inside enclosed light
   speech bubbles, then removes those glyphs with OpenCV inpainting.
7. ARIA renders the translated text and exposes the original, cleaned, and final images.

Completed pages can be edited in the frontend and rerendered without repeating
OCR or translation. The renderer automatically expands narrow OCR boxes into a
larger bubble-oriented layout area and fits English text to that area.

The local backend deliberately stores each intermediate artifact so later work
can add OCR correction, translation editing, better masks, and rerendering
without repeating every stage.

Cleanup is intentionally conservative. Detector regions rejected by OCR are not
cleaned, and a region must have a light, locally enclosed bubble-like background
before its dark pixels are masked. This avoids treating sound effects, titles,
and artwork as dialogue. `ARIA_MASK_DILATION` expands the refined glyph mask;
it no longer expands the entire detector polygon.

Bubbles that intentionally open into a panel frame are left alone when their
light interior is indistinguishable from the page background. Automatically
guessing those regions is more likely to erase side text or panel artwork; a
dedicated bubble detector or manual bubble selection is needed to handle them
reliably.

## Quick Start

### 1. Install Tesseract

Install Tesseract with Japanese language data. On Windows, the UB Mannheim
builds are a practical option. Ensure `tesseract.exe` is on `PATH` and that
`jpn.traineddata` is installed.

### 2. Start the local backend

```powershell
cd aria-local
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
uvicorn aria_local.main:app --reload --port 8000
```

The backend API and Swagger documentation are available at:

- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/docs`

### 3. Start the frontend

```powershell
cd aria-frontend
npm install
npm run dev
```

The frontend defaults to `http://127.0.0.1:8000`. To use another backend,
copy `aria-frontend/.env.example` to `.env.local` and change `VITE_API_URL`.

## Translation Providers

Without `DEEPL_API_KEY`, ARIA uses the Japanese source text as a placeholder
translation. This allows the local OCR, cleanup, and renderer stages to be
tested without an external service.

Text detection, Japanese recognition, image cleanup, and rendering stay on the
local machine. Argos Translate is also local. Only DeepL translation leaves the
machine when explicitly enabled.

The UI offers **Argos Translate** as a no-key Japanese-to-English alternative.
It is expected to be less natural than DeepL for manga dialogue, and its model
package is downloaded the first time it is selected.

**Helsinki OPUS-MT** (`Helsinki-NLP/opus-mt-ja-en`) is also available as a local
option. It produces higher-quality translations than Argos and is preferred by
the frontend when `transformers` and `sentencepiece` are installed in the backend
environment.

To use DeepL:

```powershell
$env:DEEPL_API_KEY = "your-deepl-key"
$env:ARIA_TRANSLATION_PROVIDER = "deepl"
uvicorn aria_local.main:app --reload --port 8000
```

The translation provider is intentionally isolated from OCR and image
processing so a future local translation model can be added without changing
the API or frontend.

## Useful Settings

```powershell
$env:ARIA_OCR_LANG = "jpn"
$env:ARIA_OCR_MIN_CONFIDENCE = "25"
$env:ARIA_TEXT_DETECTOR = "tesseract"
$env:ARIA_TEXT_RECOGNIZER = "tesseract"
$env:ARIA_PADDLEOCR_ENABLE_MKLDNN = "0"
$env:ARIA_PADDLEOCR_LIMIT_SIDE_LEN = "960"
$env:ARIA_MASK_DILATION = "5"
$env:ARIA_FONT_PATH = "C:\Windows\Fonts\arial.ttf"
$env:ARIA_DATA_DIR = ".aria-data"
$env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

For pages that are primarily vertical Japanese, install `jpn_vert.traineddata`
and try `ARIA_OCR_LANG=jpn_vert`.

## Architecture

The project has one backend path:

- `aria-local`: the Python backend for OCR, translation, image cleanup, and rendering

The frontend talks directly to this local backend. The CPU-heavy pipeline stays
in the same Python process as the provider adapters, avoiding an extra remote
runtime and serialization step.

## Roadmap

- Replace the default provider adapters with manga-specific detection and masks
- Use `manga-ocr` with PaddleOCR or a manga-specific detector for higher-quality Japanese OCR
- Improve speech-bubble expansion and inpainting
- Add editable OCR and translation regions
- Add page ordering and chapter-level translation context
- Add stronger local translation providers such as Ollama/Qwen
- Add Docker packaging for easier installation
