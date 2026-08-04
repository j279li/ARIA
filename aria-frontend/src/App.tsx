import { useCallback, useEffect, useRef, useState } from 'react';
import {
  AlertCircle,
  CheckCircle2,
  Image as ImageIcon,
  Loader2,
  Trash2,
  Upload,
  X,
} from 'lucide-react';

import { API_URL, errorMessage } from './api';
import PagePanel from './PagePanel';
import type { JobStatus } from './types';

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

interface PendingFile {
  id: string;
  file: File;
  previewUrl: string;
}

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

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function App() {
  const [files, setFiles] = useState<PendingFile[]>([]);
  const previewUrls = useRef(new Set<string>());
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
        if (cancelled) return;
        const normalizedCatalog: ProviderCatalog = {
          detectors: catalog.detectors ?? fallbackProviders.detectors,
          recognizers: catalog.recognizers ?? fallbackProviders.recognizers,
          translators: catalog.translators ?? fallbackProviders.translators,
        };
        const preferredRecognizer =
          normalizedCatalog.recognizers.find(
            (provider) => provider.available && provider.id === 'manga-ocr',
          ) ?? normalizedCatalog.recognizers.find((provider) => provider.id === 'tesseract');
        const preferredDetector =
          (preferredRecognizer?.id === 'manga-ocr'
            ? normalizedCatalog.detectors.find(
                (provider) => provider.available && provider.id === 'paddleocr',
              )
            : undefined) ??
          normalizedCatalog.detectors.find((provider) => provider.id === 'tesseract');
        const preferredTranslator =
          normalizedCatalog.translators.find(
            (provider) => provider.available && provider.id === 'deepl',
          ) ??
          normalizedCatalog.translators.find(
            (provider) => provider.available && provider.id === 'helsinki',
          ) ??
          normalizedCatalog.translators.find(
            (provider) => provider.available && provider.id === 'argos',
          ) ??
          normalizedCatalog.translators.find((provider) => provider.id === 'identity');

        setProviders(normalizedCatalog);
        if (preferredDetector) setDetectorProvider(preferredDetector.id as DetectorProvider);
        if (preferredRecognizer) {
          setRecognizerProvider(preferredRecognizer.id as RecognizerProvider);
        }
        if (preferredTranslator) {
          setTranslationProvider(preferredTranslator.id as TranslationProvider);
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
    if (!pollingJobId || pollingStatus === 'complete' || pollingStatus === 'failed') return;

    let cancelled = false;
    let timeoutId: number | undefined;
    const controller = new AbortController();
    const poll = async () => {
      let keepPolling = true;
      try {
        const response = await fetch(`${API_URL}/api/jobs/${pollingJobId}`, {
          signal: controller.signal,
        });
        if (!response.ok) throw new Error(`Could not read job status (${response.status}).`);
        const nextJob = (await response.json()) as JobStatus;
        keepPolling = nextJob.status === 'queued' || nextJob.status === 'processing';
        if (!cancelled) setJob(nextJob);
      } catch (pollError) {
        if (!cancelled && !controller.signal.aborted) setError(errorMessage(pollError));
      } finally {
        if (!cancelled && keepPolling) timeoutId = window.setTimeout(() => void poll(), 1000);
      }
    };

    void poll();
    return () => {
      cancelled = true;
      controller.abort();
      if (timeoutId !== undefined) window.clearTimeout(timeoutId);
    };
  }, [pollingJobId, pollingStatus]);

  const addFiles = useCallback((incoming: File[]) => {
    const images = incoming.filter((file) => file.type.startsWith('image/'));
    setError(images.length === incoming.length ? null : 'Only image files can be added.');
    const pending = images.map((file) => {
      const previewUrl = URL.createObjectURL(file);
      previewUrls.current.add(previewUrl);
      return { id: crypto.randomUUID(), file, previewUrl };
    });
    setFiles((current) => [...current, ...pending]);
  }, []);

  useEffect(() => {
    const handlePaste = (event: ClipboardEvent) => {
      const item = Array.from(event.clipboardData?.items ?? []).find((clipboardItem) =>
        clipboardItem.type.startsWith('image/'),
      );
      const blob = item?.getAsFile();
      if (!blob) return;
      event.preventDefault();
      addFiles([
        new File([blob], `pasted-${Date.now()}.png`, { type: blob.type || 'image/png' }),
      ]);
    };

    window.addEventListener('paste', handlePaste);
    return () => window.removeEventListener('paste', handlePaste);
  }, [addFiles]);

  useEffect(() => () => {
    previewUrls.current.forEach((url) => URL.revokeObjectURL(url));
    previewUrls.current.clear();
  }, []);

  const submitJob = async () => {
    if (files.length === 0 || submitting) return;
    setSubmitting(true);
    setError(null);

    try {
      const formData = new FormData();
      files.forEach(({ file }) => formData.append('files', file, file.name));
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
    files.forEach(({ previewUrl }) => {
      URL.revokeObjectURL(previewUrl);
      previewUrls.current.delete(previewUrl);
    });
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
  const mangaRecognizerReady = providers.recognizers.some(
    (provider) => provider.id === 'manga-ocr' && provider.available,
  );
  const providerPairReady = detectorProvider !== 'paddleocr' || recognizerProvider === 'manga-ocr';

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
                onChange={(event) => {
                  addFiles(Array.from(event.target.files ?? []));
                  event.currentTarget.value = '';
                }}
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
                  onChange={(event) => {
                    const detector = event.target.value as DetectorProvider;
                    setDetectorProvider(detector);
                    if (detector === 'paddleocr') setRecognizerProvider('manga-ocr');
                  }}
                  className="mt-4 w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-white outline-none focus:border-emerald-400"
                >
                  {providers.detectors.map((provider) => (
                    <option
                      key={provider.id}
                      value={provider.id}
                      disabled={
                        !provider.available ||
                        (provider.id === 'paddleocr' && !mangaRecognizerReady)
                      }
                    >
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
                    <option
                      key={provider.id}
                      value={provider.id}
                      disabled={
                        !provider.available ||
                        (provider.id === 'tesseract' && detectorProvider === 'paddleocr')
                      }
                    >
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
                    disabled={
                      submitting ||
                      !detectorReady ||
                      !recognizerReady ||
                      !translationReady ||
                      !providerPairReady
                    }
                    className="flex items-center gap-2 rounded-lg bg-emerald-600 px-4 py-2 font-medium transition hover:bg-emerald-500 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    {submitting && <Loader2 className="h-4 w-4 animate-spin" />}
                    {submitting ? 'Starting...' : 'Translate pages'}
                  </button>
                </div>
                {(!detectorReady || !recognizerReady || !translationReady || !providerPairReady) && (
                  <p className="mb-4 rounded-lg bg-amber-500/10 p-3 text-sm text-amber-100">
                    The selected provider is not available in the backend environment.
                  </p>
                )}
                <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                  {files.map(({ id, file, previewUrl }, index) => {
                    return (
                      <div key={id} className="flex items-center gap-3 rounded-xl bg-zinc-800/70 p-3">
                        <img
                          src={previewUrl}
                          alt=""
                          className="h-16 w-12 rounded object-cover"
                          decoding="async"
                        />
                        <div className="min-w-0 flex-1">
                          <p className="truncate text-sm font-medium">{index + 1}. {file.name}</p>
                          <p className="text-xs text-zinc-400">{formatBytes(file.size)}</p>
                        </div>
                        <button
                          onClick={(event) => {
                            event.stopPropagation();
                            URL.revokeObjectURL(previewUrl);
                            previewUrls.current.delete(previewUrl);
                            setFiles((current) => current.filter((entry) => entry.id !== id));
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

                {currentPage && (
                  <PagePanel
                    page={currentPage}
                    jobId={job.id}
                    onJobUpdate={(nextJob) => {
                      setJob((current) => (current?.id === nextJob.id ? nextJob : current));
                    }}
                  />
                )}
            </div>
          </section>
        )}
      </main>
    </div>
  );
}
