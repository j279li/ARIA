## Overview

This project uses:
- **Google Cloud Vision API** for Japanese OCR (text detection)
- **DeepL API** for high-quality Japanese→English translation

## 1. Google Cloud Vision API Setup

### Step 1: Get a Google Cloud Vision API Key

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the **Cloud Vision API**:
   - Navigate to "APIs & Services" > "Library"
   - Search for "Cloud Vision API"
   - Click "Enable"
4. Create an API Key:
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "API Key"
   - Copy your API key

## 2. DeepL API Setup

DeepL provides pretty solid Japanese translation quality.

### Step 1: Get a DeepL API Key

1. Go to [DeepL API](https://www.deepl.com/pro-api)
2. Sign up for a **DeepL API Free** account
   - Free tier: 500,000 characters/month
   - No credit card required
3. Go to your account settings and copy your **Authentication Key**

### Step 2: Add API Keys to Cloudflare Worker

#### Option A: Using Wrangler Secrets

```powershell
cd aria-backend
wrangler secret put GOOGLE_VISION_API_KEY
# Paste your Google Cloud Vision API key

wrangler secret put DEEPL_API_KEY
# Paste your DeepL API key
```

#### Option B: Environment Variables (For Testing or Personal Use )

Edit `wrangler.jsonc` and uncomment the vars section:

```jsonc
"vars": {
  "GOOGLE_VISION_API_KEY": "your-google-api-key-here",
  "DEEPL_API_KEY": "your-deepl-api-key-here"
}
```

## 3. Deploy

```powershell
npx wrangler deploy
```

## How It Works

1. **OCR**: Google Cloud Vision API detects Japanese text (including vertical text)
2. **Grouping**: Nearby text is grouped into speech bubbles
3. **Sorting**: Bubbles are ordered in manga reading order (top→bottom, right→left)
4. **Translation**: DeepL API translates Japanese to English with high quality
5. **Rendering**: Frontend displays translations with numbered boxes

## Pricing

- **Google Cloud Vision**: 1,000 requests/month free, then $1.50 per 1,000 requests
- **DeepL API Free**: 500,000 characters/month free
- **Cloudflare Workers**: Free tier (100k requests/day)
- **Cloudflare R2**: Free tier (10GB storage)

**Estimated cost**: Free for personal use (within free tiers)

