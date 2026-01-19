# ARIA - Manga Translation Tool

A modern web application for translating Japanese manga pages using AI-powered OCR and translation services.

## Features

- **Image Upload & Paste** - Upload manga pages or paste directly from clipboard
- **Automatic Text Detection** - Uses Google Cloud Vision API for OCR
- **Context-Aware Translation** - DeepL API for high-quality Japanese to English translation
- **Visual Overlay** - Numbered boxes show detected text regions on the original image
- **Retranslation** - Re-translate with corrected order for better context

## Tech Stack

### Frontend
- **React 19** with TypeScript
- **Vite** for fast development and building
- **Tailwind CSS** for styling
- **Lucide React** for icons

### Backend
- **Cloudflare Workers** for serverless backend
- **Google Cloud Vision API** for OCR
- **DeepL API** for translation
- **TypeScript** for type safety


## Setup

### Prerequisites
- Node.js 18+ and npm
- Google Cloud Vision API key
- DeepL API key (free tier available)
- Cloudflare account (for deployment)

### Backend Setup

1. Navigate to backend directory:
   ```bash
   cd aria-backend
   npm install
   ```

2. Create `.dev.vars` file with your API keys:
   ```
   GOOGLE_VISION_API_KEY=your_google_vision_key
   DEEPL_API_KEY=your_deepl_key
   ```

3. Deploy to Cloudflare Workers:
   ```bash
   npm run deploy
   ```

See [aria-backend/SETUP.md](aria-backend/SETUP.md) for detailed backend setup instructions.

### Frontend Setup

1. Navigate to frontend directory:
   ```bash
   cd aria-frontend
   npm install
   ```

2. Update the worker URL in `src/App.tsx`:
   ```typescript
   const WORKER_URL = "https://your-worker-url.workers.dev";
   ```

3. Start development server:
   ```bash
   npm run dev
   ```

4. Build for production:
   ```bash
   npm run build
   ```