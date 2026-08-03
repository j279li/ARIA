import { useEffect, useRef, useState } from 'react';
import {
  AlertCircle,
  CheckCircle2,
  Clipboard,
  Download,
  Image as ImageIcon,
  Loader2,
  MousePointer2,
  RotateCcw,
  Trash2,
  Upload,
  X,
} from 'lucide-react';

interface Point {
  x: number;
  y: number;
}

interface TextRegion {
  id: string;
  polygon: Point[];
  bbox: [number, number, number, number];
  render_bbox: [number, number, number, number] | null;
  source_text: string;
  translated_text: string;
  font_size: number | null;
  confidence: number;
  orientation: 'horizontal' | 'vertical';
  reading_order: number;
}

interface ManualInpaintRegion {
  bbox: [number, number, number, number];
}

interface PageStatus {
  id: string;
  filename: string;
  status: 'queued' | 'processing' | 'complete' | 'failed';
  original_url: string;
  cleaned_url: string | null;
  output_url: string | null;
  regions: TextRegion[];
  manual_inpaint_regions: ManualInpaintRegion[];
  warnings: string[];
  error: string | null;
}

interface JobStatus {
  id: string;
  status: 'queued' | 'processing' | 'complete' | 'failed';
  created_at: string;
  detector_provider: 'tesseract' | 'paddleocr';
  recognizer_provider: 'tesseract' | 'manga-ocr';
  translation_provider: 'deepl' | 'argos' | 'helsinki' | 'identity';
  pages: PageStatus[];
  error: string | null;
}

type DetectorProvider = 'tesseract' | 'paddleocr';
type RecognizerProvider = 'tesseract' | 'manga-ocr';
type TranslationProvider = 'deepl' | 'argos' | 'helsinki' | 'identity';

interface ProviderOption {
  id: string;
  label: string;
  available: boolean;
}

interface ProviderCatalog {
  detectors: ProviderOption[];
  recognizers: ProviderOption[];
  translators: ProviderOption[];
}

const API_URL = (import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000').replace(/\/$/, '');

const fallbackProviders: ProviderCatalog = {
  detectors: [
    { id: 'tesseract', label: 'Tesseract', available: true },
    { id: 'paddleocr', label: 'PaddleOCR', available: false },
  ],
  recognizers: [
    { id: 'tesseract', label: 'Tesseract', available: true },
    { id: 'manga-ocr', label: 'Manga OCR', available: false },
  ],
  translators: [
    { id: 'deepl', label: 'DeepL', available: false },
    { id: 'argos', label: 'Argos Translate', available: false },
    { id: 'helsinki', label: 'Helsinki OPUS-MT', available: false },
    { id: 'identity', label: 'Source text', available: true },
  ],
};

function assetUrl(path: string | null): string | null {
  if (!path) return null;
  return path.startsWith('http') ? path : `${API_URL}${path}`;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : 'Something went wrong.';
}

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function App() {
  const [files, setFiles] = useState<File[]>([]);
  const [previews, setPreviews] = useState<Record<string, string>>({});
  const [job, setJob] = useState<JobStatus | null>(null);
  const [activePage, setActivePage] = useState(0);
  const [dragActive, setDragActive] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [providers, setProviders] = useState<ProviderCatalog>(fallbackProviders);
  const [detectorProvider, setDetectorProvider] = useState<DetectorProvider>('tesseract');
  const [recognizerProvider, setRecognizerProvider] = useState<RecognizerProvider>('tesseract');
  const [translationProvider, setTranslationProvider] = useState<TranslationProvider>('identity');
  const pollingJobId = job?.id;
  const pollingStatus = job?.status;

  useEffect(() => {
    let cancelled = false;
    fetch(`${API_URL}/api/providers`)
      .then(async (response) => {
        if (!response.ok) throw new Error(`Could not read provider status (${response.status}).`);
        return (await response.json()) as ProviderCatalog;
      })
      .then((catalog) => {
        if (!cancelled) {
          const normalizedCatalog: ProviderCatalog = {
            detectors: catalog.detectors ?? fallbackProviders.detectors,
            recognizers: catalog.recognizers ?? fallbackProviders.recognizers,
            translators: catalog.translators ?? fallbackProviders.translators,
           };
           setProviders(normalizedCatalog);
           const preferredDetector = normalizedCatalog.detectors.find(
             (provider) => provider.available && provider.id === 'paddleocr',
           ) ?? normalizedCatalog.detectors.find((provider) => provider.id === 'tesseract');
           const preferredRecognizer = normalizedCatalog.recognizers.find(
             (provider) => provider.available && provider.id === 'manga-ocr',
           ) ?? normalizedCatalog.recognizers.find((provider) => provider.id === 'tesseract');
            const preferred = normalizedCatalog.translators.find(
              (provider) => provider.available && provider.id === 'helsinki',
            ) ?? normalizedCatalog.translators.find(
              (provider) => provider.available && provider.id === 'argos',
            ) ?? normalizedCatalog.translators.find(
              (provider) => provider.available && provider.id === 'deepl',
            ) ?? normalizedCatalog.translators.find((provider) => provider.id === 'identity');
           if (preferredDetector) setDetectorProvider(preferredDetector.id as DetectorProvider);
           if (preferredRecognizer) setRecognizerProvider(preferredRecognizer.id as RecognizerProvider);
           if (preferred) setTranslationProvider(preferred.id as TranslationProvider);
        }
      })
      .catch(() => {
        // Keep the Tesseract fallback if an older backend has no provider endpoint.
      });

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const nextPreviews: Record<string, string> = {};
    files.forEach((file, index) => {
      nextPreviews[`${file.name}-${file.lastModified}-${index}`] = URL.createObjectURL(file);
    });
    setPreviews(nextPreviews);

    return () => Object.values(nextPreviews).forEach((url) => URL.revokeObjectURL(url));
  }, [files]);

  useEffect(() => {
    if (!pollingJobId || pollingStatus === 'complete' || pollingStatus === 'failed') return;

    let cancelled = false;
    const poll = async () => {
      try {
        const response = await fetch(`${API_URL}/api/jobs/${pollingJobId}`);
        if (!response.ok) throw new Error(`Could not read job status (${response.status}).`);
        const nextJob = (await response.json()) as JobStatus;
        if (!cancelled) setJob(nextJob);
      } catch (pollError) {
        if (!cancelled) setError(errorMessage(pollError));
      }
    };

    void poll();
    const interval = window.setInterval(() => void poll(), 1000);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [pollingJobId, pollingStatus]);

  useEffect(() => {
    const handlePaste = (event: ClipboardEvent) => {
      const item = Array.from(event.clipboardData?.items ?? []).find((clipboardItem) =>
        clipboardItem.type.startsWith('image/'),
      );
      const blob = item?.getAsFile();
      if (!blob) return;
      event.preventDefault();
      setFiles((current) => [
        ...current,
        new File([blob], `pasted-${Date.now()}.png`, { type: blob.type || 'image/png' }),
      ]);
      setError(null);
    };

    window.addEventListener('paste', handlePaste);
    return () => window.removeEventListener('paste', handlePaste);
  }, []);

  const addFiles = (incoming: File[]) => {
    const images = incoming.filter((file) => file.type.startsWith('image/'));
    if (images.length !== incoming.length) {
      setError('Only image files can be added.');
    } else {
      setError(null);
    }
    setFiles((current) => [...current, ...images]);
  };

  const submitJob = async () => {
    if (files.length === 0 || submitting) return;
    setSubmitting(true);
    setError(null);

    try {
      const formData = new FormData();
      files.forEach((file) => formData.append('files', file, file.name));
      formData.append('detector_provider', detectorProvider);
      formData.append('recognizer_provider', recognizerProvider);
      formData.append('translation_provider', translationProvider);
      const response = await fetch(`${API_URL}/api/jobs`, { method: 'POST', body: formData });
      const payload = (await response.json().catch(() => null)) as { detail?: string } | JobStatus | null;
      if (!response.ok) {
        throw new Error(payload && 'detail' in payload ? payload.detail : `Upload failed (${response.status}).`);
      }
      setJob(payload as JobStatus);
      setActivePage(0);
    } catch (submitError) {
      setError(errorMessage(submitError));
    } finally {
      setSubmitting(false);
    }
  };

  const reset = () => {
    setFiles([]);
    setJob(null);
    setActivePage(0);
    setError(null);
  };

  const currentPage = job?.pages[activePage] ?? null;
  const isWorking = job?.status === 'queued' || job?.status === 'processing';
  const detectorReady = providers.detectors.some(
    (provider) => provider.id === detectorProvider && provider.available,
  );
  const recognizerReady = providers.recognizers.some(
    (provider) => provider.id === recognizerProvider && provider.available,
  );
  const translationReady = providers.translators.some(
    (provider) => provider.id === translationProvider && provider.available,
  );

  return (
    <div className="min-h-screen bg-zinc-950 text-white">
      <header className="border-b border-zinc-800/60 bg-zinc-950">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-5">
          <div className="flex items-center gap-3">
            <div className="rounded-xl bg-emerald-600 p-3">
              <ImageIcon className="h-6 w-6" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-emerald-300">ARIA</h1>
              <p className="text-sm text-zinc-400">Local manga translator</p>
            </div>
          </div>
          {job && (
            <button
              onClick={reset}
              className="rounded-lg bg-zinc-800 px-3 py-2 text-sm text-zinc-200 transition hover:bg-zinc-700"
            >
              New job
            </button>
          )}
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-6 py-8">
        {error && (
          <div className="mb-6 flex items-start gap-3 rounded-xl border border-red-500/30 bg-red-600 p-4 text-white">
            <AlertCircle className="mt-0.5 h-5 w-5 shrink-0" />
            <span>{error}</span>
            <button className="ml-auto" onClick={() => setError(null)} title="Dismiss error">
              <X className="h-4 w-4" />
            </button>
          </div>
        )}

        {!job && (
          <section className="mx-auto max-w-4xl">
            <div
              onClick={() => document.getElementById('file-input')?.click()}
              onDragEnter={(event) => {
                event.preventDefault();
                setDragActive(true);
              }}
              onDragOver={(event) => event.preventDefault()}
              onDragLeave={() => setDragActive(false)}
              onDrop={(event) => {
                event.preventDefault();
                setDragActive(false);
                addFiles(Array.from(event.dataTransfer.files));
              }}
              className={`cursor-pointer rounded-2xl border-2 border-dashed p-12 text-center transition ${
                dragActive
                  ? 'border-emerald-400 bg-emerald-500/10'
                  : 'border-zinc-700 hover:border-emerald-500/60 hover:bg-zinc-900/60'
              }`}
            >
              <input
                id="file-input"
                type="file"
                accept="image/*"
                multiple
                className="hidden"
                onChange={(event) => addFiles(Array.from(event.target.files ?? []))}
              />
              <Upload className="mx-auto mb-4 h-12 w-12 text-emerald-300" />
              <h2 className="text-2xl font-semibold">Drop manga pages here</h2>
              <p className="mt-2 text-sm text-zinc-400">
                Select multiple images, paste from the clipboard, or drag files into this area.
              </p>
              <div className="mt-5 inline-flex items-center gap-2 rounded-lg bg-emerald-600 px-4 py-2 font-medium">
                <ImageIcon className="h-4 w-4" />
                Choose images
              </div>
            </div>

            <div className="mt-6 grid gap-4 md:grid-cols-3">
              <label className="rounded-2xl border border-zinc-800 bg-zinc-900/60 p-5">
                <span className="text-sm font-semibold">Text detector</span>
                <span className="mt-1 block text-xs text-zinc-400">
                  Finds text regions and polygons on the page.
                </span>
                <select
                  value={detectorProvider}
                  onChange={(event) => setDetectorProvider(event.target.value as DetectorProvider)}
                  className="mt-4 w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-white outline-none focus:border-emerald-400"
                >
                  {providers.detectors.map((provider) => (
                    <option key={provider.id} value={provider.id} disabled={!provider.available}>
                      {provider.label}{provider.available ? '' : ' (not installed)'}
                    </option>
                  ))}
                </select>
              </label>

              <label className="rounded-2xl border border-zinc-800 bg-zinc-900/60 p-5">
                <span className="text-sm font-semibold">Text recognizer</span>
                <span className="mt-1 block text-xs text-zinc-400">
                  Reads Japanese text inside each detected region.
                </span>
                <select
                  value={recognizerProvider}
                  onChange={(event) => setRecognizerProvider(event.target.value as RecognizerProvider)}
                  className="mt-4 w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-white outline-none focus:border-emerald-400"
                >
                  {providers.recognizers.map((provider) => (
                    <option key={provider.id} value={provider.id} disabled={!provider.available}>
                      {provider.label}{provider.available ? '' : ' (not installed)'}
                    </option>
                  ))}
                </select>
              </label>

              <label className="rounded-2xl border border-zinc-800 bg-zinc-900/60 p-5">
                <span className="text-sm font-semibold">Translation provider</span>
                <span className="mt-1 block text-xs text-zinc-400">
                  Converts recognized Japanese into English.
                </span>
                <select
                  value={translationProvider}
                  onChange={(event) => setTranslationProvider(event.target.value as TranslationProvider)}
                  className="mt-4 w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-white outline-none focus:border-emerald-400"
                >
                  {providers.translators.map((provider) => (
                    <option key={provider.id} value={provider.id} disabled={!provider.available}>
                      {provider.label}{provider.available ? '' : ' (not installed)'}
                    </option>
                  ))}
                </select>
              </label>
            </div>

            <p className="mt-4 text-sm text-zinc-400">
              Detection, recognition, cleanup, and rendering run locally. Argos Translate is local; DeepL sends recognized text outside this computer.
            </p>

            {files.length > 0 && (
              <div className="mt-6 rounded-2xl border border-zinc-800 bg-zinc-900/60 p-5">
                <div className="mb-4 flex items-center justify-between">
                  <div>
                    <h2 className="font-semibold">Pages to process ({files.length})</h2>
                    <p className="text-sm text-zinc-400">Pages are processed in this order.</p>
                  </div>
                  <button
                    onClick={submitJob}
                    disabled={submitting || !detectorReady || !recognizerReady || !translationReady}
                    className="flex items-center gap-2 rounded-lg bg-emerald-600 px-4 py-2 font-medium transition hover:bg-emerald-500 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    {submitting && <Loader2 className="h-4 w-4 animate-spin" />}
                    {submitting ? 'Starting...' : 'Translate pages'}
                  </button>
                </div>
                {(!detectorReady || !recognizerReady || !translationReady) && (
                  <p className="mb-4 rounded-lg bg-amber-500/10 p-3 text-sm text-amber-100">
                    The selected provider is not available in the backend environment.
                  </p>
                )}
                <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                  {files.map((file, index) => {
                    const key = `${file.name}-${file.lastModified}-${index}`;
                    return (
                      <div key={key} className="flex items-center gap-3 rounded-xl bg-zinc-800/70 p-3">
                        <img src={previews[key]} alt="" className="h-16 w-12 rounded object-cover" />
                        <div className="min-w-0 flex-1">
                          <p className="truncate text-sm font-medium">{index + 1}. {file.name}</p>
                          <p className="text-xs text-zinc-400">{formatBytes(file.size)}</p>
                        </div>
                        <button
                          onClick={(event) => {
                            event.stopPropagation();
                            setFiles((current) => current.filter((_, itemIndex) => itemIndex !== index));
                          }}
                          className="rounded p-1 text-zinc-500 transition hover:bg-red-500/20 hover:text-red-300"
                          title="Remove page"
                        >
                          <Trash2 className="h-4 w-4" />
                        </button>
                      </div>
                    );
                  })}
                </div>
                <p className="mt-4 flex items-center gap-2 text-sm text-zinc-400">
                  <Clipboard className="h-4 w-4 text-emerald-400" />
                  Paste images with Ctrl+V or Cmd+V.
                </p>
              </div>
            )}
          </section>
        )}

        {job && (
          <section>
            <div className="mb-6 flex flex-wrap items-center justify-between gap-4">
              <div>
                <p className="text-sm uppercase tracking-wider text-zinc-500">Job {job.id}</p>
                <h2 className="mt-1 text-2xl font-semibold">
                  {isWorking ? 'Processing manga pages...' : job.status === 'complete' ? 'Translation complete' : 'Job finished with errors'}
                </h2>
              </div>
              <div className="flex items-center gap-2 text-sm text-zinc-300">
                {isWorking ? <Loader2 className="h-4 w-4 animate-spin text-emerald-300" /> : <CheckCircle2 className="h-4 w-4 text-emerald-400" />}
                {job.pages.filter((page) => page.status === 'complete').length}/{job.pages.length} pages complete
              </div>
            </div>

            {job.error && <p className="mb-5 rounded-lg bg-red-600 p-3 text-sm text-white">{job.error}</p>}

            <div className="grid gap-6 lg:grid-cols-[220px_minmax(0,1fr)]">
              <nav className="space-y-2">
                {job.pages.map((page, index) => (
                  <button
                    key={page.id}
                    onClick={() => setActivePage(index)}
                    className={`w-full rounded-xl border p-3 text-left transition ${
                      index === activePage
                        ? 'border-emerald-500/60 bg-emerald-600/10'
                        : 'border-zinc-800 bg-zinc-900/60 hover:border-zinc-700'
                    }`}
                  >
                    <p className="truncate text-sm font-medium">{index + 1}. {page.filename}</p>
                    <p className="mt-1 text-xs capitalize text-zinc-400">{page.status}</p>
                  </button>
                ))}
              </nav>

                {currentPage && <PagePanel page={currentPage} onJobUpdate={setJob} jobId={job.id} />}
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

function PagePanel({
  page,
  jobId,
  onJobUpdate,
}: {
  page: PageStatus;
  jobId: string;
  onJobUpdate: (job: JobStatus) => void;
}) {
  const outputUrl = assetUrl(page.output_url);
  const cleanedUrl = assetUrl(page.cleaned_url);
  const originalUrl = assetUrl(page.original_url);

  if (page.status === 'failed') {
    return (
      <div className="rounded-2xl border border-red-500/30 bg-red-600 p-6 text-white">
        <h3 className="font-semibold">This page failed</h3>
        <p className="mt-2 text-sm">{page.error || 'The backend did not provide an error.'}</p>
      </div>
    );
  }

  if (page.status !== 'complete' || !outputUrl) {
    return (
      <div className="flex min-h-[420px] items-center justify-center rounded-2xl border border-zinc-800 bg-zinc-900/60">
        <div className="text-center text-zinc-400">
          <Loader2 className="mx-auto mb-3 h-8 w-8 animate-spin text-emerald-300" />
          <p>Preparing this page...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="grid gap-4 xl:grid-cols-3">
        {[
          ['Original', originalUrl],
          ['Cleaned', cleanedUrl],
          ['Translated', outputUrl],
        ].map(([label, url]) => (
          <div key={label} className="overflow-hidden rounded-2xl border border-zinc-800 bg-zinc-900/60">
            <div className="flex items-center justify-between border-b border-zinc-800 px-4 py-3">
              <span className="text-sm font-medium">{label}</span>
              {label === 'Translated' && url && (
                <a
                  href={url}
                  download={`aria-page-${page.filename}`}
                  className="flex items-center gap-1.5 rounded-lg bg-emerald-600 px-3 py-1.5 text-xs font-medium transition hover:bg-emerald-500"
                >
                  <Download className="h-3.5 w-3.5" />
                  Save
                </a>
              )}
            </div>
            {url && <img src={url} alt={`${label} manga page`} className="max-h-[70vh] w-full object-contain" />}
          </div>
        ))}
      </div>

      <ManualCleanupEditor page={page} jobId={jobId} onJobUpdate={onJobUpdate} />

      {page.warnings.length > 0 && (
        <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-4 text-sm text-amber-100">
          {page.warnings.map((warning) => <p key={warning}>{warning}</p>)}
        </div>
      )}

      <div className="rounded-2xl border border-zinc-800 bg-zinc-900/60 p-5">
        <h3 className="mb-4 font-semibold">Regions ({page.regions.length})</h3>
        {page.regions.length === 0 ? (
          <p className="text-sm text-zinc-400">No text regions were detected.</p>
        ) : (
          <div className="grid gap-2 md:grid-cols-2">
            {page.regions.map((region) => (
              <div key={region.id} className="rounded-lg bg-zinc-800/50 p-3">
                <div className="mb-1 flex items-center justify-between text-xs text-zinc-500">
                  <span>Region {region.reading_order}</span>
                  <span>{Math.round(region.confidence)}%</span>
                </div>
                <p className="mb-2 text-sm text-zinc-300">{region.source_text}</p>
                <p className="text-sm font-medium text-emerald-300">{region.translated_text}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function ManualCleanupEditor({
  page,
  jobId,
  onJobUpdate,
}: {
  page: PageStatus;
  jobId: string;
  onJobUpdate: (job: JobStatus) => void;
}) {
  const imageRef = useRef<HTMLImageElement>(null);
  const [editing, setEditing] = useState(false);
  const [regions, setRegions] = useState<ManualInpaintRegion[]>(page.manual_inpaint_regions ?? []);
  const [imageDimensions, setImageDimensions] = useState<{ width: number; height: number } | null>(null);
  const [draft, setDraft] = useState<ManualInpaintRegion | null>(null);
  const [startPoint, setStartPoint] = useState<Point | null>(null);
  const [saving, setSaving] = useState(false);
  const [editorError, setEditorError] = useState<string | null>(null);

  useEffect(() => {
    setRegions(page.manual_inpaint_regions ?? []);
    setImageDimensions(null);
    setDraft(null);
    setStartPoint(null);
  }, [page.id, page.manual_inpaint_regions]);

  const imagePoint = (event: React.PointerEvent<HTMLDivElement>): Point | null => {
    const image = imageRef.current;
    if (!image || !image.naturalWidth || !image.naturalHeight) return null;
    const bounds = image.getBoundingClientRect();
    return {
      x: Math.max(0, Math.min(image.naturalWidth, Math.round((event.clientX - bounds.left) * image.naturalWidth / bounds.width))),
      y: Math.max(0, Math.min(image.naturalHeight, Math.round((event.clientY - bounds.top) * image.naturalHeight / bounds.height))),
    };
  };

  const screenBox = (bbox: ManualInpaintRegion['bbox']) => {
    if (!imageDimensions) return null;
    const [x, y, width, height] = bbox;
    return {
      left: `${(x / imageDimensions.width) * 100}%`,
      top: `${(y / imageDimensions.height) * 100}%`,
      width: `${(width / imageDimensions.width) * 100}%`,
      height: `${(height / imageDimensions.height) * 100}%`,
    };
  };

  const updateDraft = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!startPoint) return;
    const point = imagePoint(event);
    if (!point) return;
    setDraft({
      bbox: [
        Math.min(startPoint.x, point.x),
        Math.min(startPoint.y, point.y),
        Math.abs(point.x - startPoint.x),
        Math.abs(point.y - startPoint.y),
      ],
    });
  };

  const finishDraft = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!startPoint) return;
    updateDraft(event);
    const point = imagePoint(event);
    if (point) {
      const bbox: ManualInpaintRegion['bbox'] = [
        Math.min(startPoint.x, point.x),
        Math.min(startPoint.y, point.y),
        Math.abs(point.x - startPoint.x),
        Math.abs(point.y - startPoint.y),
      ];
      if (bbox[2] >= 3 && bbox[3] >= 3) setRegions((current) => [...current, { bbox }]);
    }
    setDraft(null);
    setStartPoint(null);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  };

  const applyRegions = async () => {
    setSaving(true);
    setEditorError(null);
    try {
      const response = await fetch(`${API_URL}/api/jobs/${jobId}/pages/${page.id}/render`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ manual_inpaint_regions: regions }),
      });
      const payload = (await response.json().catch(() => null)) as { detail?: string } | JobStatus | null;
      if (!response.ok) {
        throw new Error(payload && 'detail' in payload ? payload.detail : `Cleanup failed (${response.status}).`);
      }
      onJobUpdate(payload as JobStatus);
      setEditing(false);
    } catch (applyError) {
      setEditorError(errorMessage(applyError));
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="rounded-2xl border border-emerald-500/20 bg-zinc-900/60 p-5">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h3 className="flex items-center gap-2 font-semibold">
            <MousePointer2 className="h-4 w-4 text-emerald-300" />
            Manual cleanup
          </h3>
          <p className="mt-1 max-w-2xl text-sm text-zinc-400">
            If automatic cleanup missed text, drag rectangles over it in the original page. This editor always shows the original; compare the regenerated Cleaned and Translated previews above after applying.
          </p>
        </div>
        <button
          type="button"
          onClick={() => {
            setEditing((current) => !current);
            setEditorError(null);
          }}
          className="rounded-lg border border-emerald-500/40 px-3 py-2 text-sm font-medium text-emerald-200 transition hover:bg-emerald-500/10"
        >
          {editing ? 'Close editor' : regions.length ? `Edit selections (${regions.length})` : 'Select missed text'}
        </button>
      </div>

      {editing && (
        <div className="mt-5 border-t border-zinc-800 pt-5">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-3 text-sm">
            <span className="text-zinc-300">Drag to add a rectangle. Existing selections are outlined in red.</span>
            <div className="flex items-center gap-2">
              <button
                type="button"
                disabled={!regions.length || saving}
                onClick={() => setRegions((current) => current.slice(0, -1))}
                className="flex items-center gap-1.5 rounded-lg bg-zinc-800 px-3 py-1.5 text-zinc-300 transition hover:bg-zinc-700 disabled:cursor-not-allowed disabled:opacity-40"
              >
                <RotateCcw className="h-3.5 w-3.5" />
                Undo
              </button>
              <button
                type="button"
                disabled={!regions.length || saving}
                onClick={() => setRegions([])}
                className="rounded-lg bg-zinc-800 px-3 py-1.5 text-zinc-300 transition hover:bg-zinc-700 disabled:cursor-not-allowed disabled:opacity-40"
              >
                Clear
              </button>
            </div>
          </div>

          <div
            className="flex max-h-[70vh] touch-none select-none justify-center overflow-auto rounded-xl border border-zinc-700 bg-zinc-950"
          >
            <div
              className="relative w-fit max-w-full"
              onPointerDown={(event) => {
                if (saving) return;
                const point = imagePoint(event);
                if (!point) return;
                event.currentTarget.setPointerCapture(event.pointerId);
                setStartPoint(point);
                setDraft({ bbox: [point.x, point.y, 0, 0] });
              }}
              onPointerMove={updateDraft}
              onPointerUp={finishDraft}
              onPointerCancel={finishDraft}
            >
              <img
                ref={imageRef}
                src={assetUrl(page.original_url) ?? undefined}
                alt="Original page for manual cleanup"
                className="block max-h-[70vh] max-w-full object-contain"
                onLoad={(event) => setImageDimensions({
                  width: event.currentTarget.naturalWidth,
                  height: event.currentTarget.naturalHeight,
                })}
              />
              <div className="pointer-events-none absolute inset-0">
                {regions.map((region, index) => {
                  const box = screenBox(region.bbox);
                  return box ? (
                    <div key={`${region.bbox.join('-')}-${index}`} className="absolute border-2 border-red-400 bg-red-400/15" style={box}>
                      <span className="absolute -top-5 left-0 rounded bg-red-400 px-1 text-[10px] font-bold text-zinc-950">{index + 1}</span>
                    </div>
                  ) : null;
                })}
                {draft && <div className="absolute border-2 border-emerald-300 bg-emerald-300/15" style={screenBox(draft.bbox) ?? undefined} />}
              </div>
            </div>
          </div>

          <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
            <p className="text-sm text-zinc-500">{regions.length ? `${regions.length} rectangle${regions.length === 1 ? '' : 's'} selected` : 'No rectangles selected'}</p>
            <button
              type="button"
              onClick={() => void applyRegions()}
              disabled={saving}
              className="flex items-center gap-2 rounded-lg bg-emerald-600 px-4 py-2 text-sm font-medium transition hover:bg-emerald-500 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {saving && <Loader2 className="h-4 w-4 animate-spin" />}
              {saving ? 'Applying...' : 'Apply cleanup & regenerate'}
            </button>
          </div>
          {editorError && <p className="mt-3 rounded-lg bg-red-500/10 p-3 text-sm text-red-200">{editorError}</p>}
        </div>
      )}
    </div>
  );
}
