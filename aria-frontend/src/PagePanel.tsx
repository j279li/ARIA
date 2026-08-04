import { useRef, useState } from 'react';
import type { PointerEvent } from 'react';
import { Download, Loader2, MousePointer2, RotateCcw } from 'lucide-react';

import { API_URL, assetUrl, errorMessage } from './api';
import type { JobStatus, ManualInpaintRegion, PageStatus } from './types';

interface Point {
  x: number;
  y: number;
}

interface PagePanelProps {
  page: PageStatus;
  jobId: string;
  onJobUpdate: (job: JobStatus) => void;
}

function selectionBBox(start: Point, end: Point): ManualInpaintRegion['bbox'] {
  return [
    Math.min(start.x, end.x),
    Math.min(start.y, end.y),
    Math.abs(end.x - start.x),
    Math.abs(end.y - start.y),
  ];
}

export default function PagePanel({ page, jobId, onJobUpdate }: PagePanelProps) {
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

  if (page.status !== 'complete') {
    return (
      <div className="flex min-h-[420px] items-center justify-center rounded-2xl border border-zinc-800 bg-zinc-900/60">
        <div className="text-center text-zinc-400">
          <Loader2 className="mx-auto mb-3 h-8 w-8 animate-spin text-emerald-300" />
          <p>Preparing this page...</p>
        </div>
      </div>
    );
  }

  if (!outputUrl) {
    return (
      <div className="rounded-2xl border border-red-500/30 bg-red-600 p-6 text-white">
        The translated image is missing.
      </div>
    );
  }

  const artifacts = [
    ['Original', originalUrl],
    ['Cleaned', cleanedUrl],
    ['Translated', outputUrl],
  ] as const;

  return (
    <div className="space-y-6">
      <div className="grid gap-4 xl:grid-cols-3">
        {artifacts.map(([label, url]) => (
          <div key={label} className="overflow-hidden rounded-2xl border border-zinc-800 bg-zinc-900/60">
            <div className="flex items-center justify-between border-b border-zinc-800 px-4 py-3">
              <span className="text-sm font-medium">{label}</span>
              {label === 'Translated' && (
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
            {url && (
              <img
                src={url}
                alt={`${label} manga page`}
                className="max-h-[70vh] w-full object-contain"
                loading="lazy"
                decoding="async"
              />
            )}
          </div>
        ))}
      </div>

      <ManualCleanupEditor
        key={page.id}
        page={page}
        jobId={jobId}
        onJobUpdate={onJobUpdate}
      />

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

function ManualCleanupEditor({ page, jobId, onJobUpdate }: PagePanelProps) {
  const imageRef = useRef<HTMLImageElement>(null);
  const [editing, setEditing] = useState(false);
  const [regions, setRegions] = useState<ManualInpaintRegion[]>(page.manual_inpaint_regions);
  const [imageDimensions, setImageDimensions] = useState<{ width: number; height: number } | null>(null);
  const [draft, setDraft] = useState<ManualInpaintRegion | null>(null);
  const [startPoint, setStartPoint] = useState<Point | null>(null);
  const [saving, setSaving] = useState(false);
  const [editorError, setEditorError] = useState<string | null>(null);

  const imagePoint = (event: PointerEvent<HTMLDivElement>): Point | null => {
    const image = imageRef.current;
    if (!image?.naturalWidth || !image.naturalHeight) return null;
    const bounds = image.getBoundingClientRect();
    return {
      x: Math.max(
        0,
        Math.min(
          image.naturalWidth,
          Math.round(((event.clientX - bounds.left) * image.naturalWidth) / bounds.width),
        ),
      ),
      y: Math.max(
        0,
        Math.min(
          image.naturalHeight,
          Math.round(((event.clientY - bounds.top) * image.naturalHeight) / bounds.height),
        ),
      ),
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

  const updateDraft = (event: PointerEvent<HTMLDivElement>) => {
    if (!startPoint) return;
    const point = imagePoint(event);
    if (point) setDraft({ bbox: selectionBBox(startPoint, point) });
  };

  const releasePointer = (event: PointerEvent<HTMLDivElement>) => {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  };

  const finishDraft = (event: PointerEvent<HTMLDivElement>) => {
    if (!startPoint) return;
    const point = imagePoint(event);
    if (point) {
      const bbox = selectionBBox(startPoint, point);
      if (bbox[2] >= 3 && bbox[3] >= 3) setRegions((current) => [...current, { bbox }]);
    }
    setDraft(null);
    setStartPoint(null);
    releasePointer(event);
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
      const payload = (await response.json().catch(() => null)) as
        | { detail?: string }
        | JobStatus
        | null;
      if (!response.ok) {
        throw new Error(
          payload && 'detail' in payload ? payload.detail : `Cleanup failed (${response.status}).`,
        );
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
            Drag over text missed by automatic cleanup, then regenerate the page.
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
            <span className="text-zinc-300">Drag to add a rectangle.</span>
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

          <div className="flex max-h-[70vh] touch-none select-none justify-center overflow-auto rounded-xl border border-zinc-700 bg-zinc-950">
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
              onPointerCancel={(event) => {
                setDraft(null);
                setStartPoint(null);
                releasePointer(event);
              }}
            >
              <img
                ref={imageRef}
                src={assetUrl(page.original_url) ?? undefined}
                alt="Original page for manual cleanup"
                className="block max-h-[70vh] max-w-full object-contain"
                onLoad={(event) =>
                  setImageDimensions({
                    width: event.currentTarget.naturalWidth,
                    height: event.currentTarget.naturalHeight,
                  })
                }
              />
              <div className="pointer-events-none absolute inset-0">
                {regions.map((region, index) => {
                  const box = screenBox(region.bbox);
                  return box ? (
                    <div
                      key={`${region.bbox.join('-')}-${index}`}
                      className="absolute border-2 border-red-400 bg-red-400/15"
                      style={box}
                    >
                      <span className="absolute -top-5 left-0 rounded bg-red-400 px-1 text-[10px] font-bold text-zinc-950">
                        {index + 1}
                      </span>
                    </div>
                  ) : null;
                })}
                {draft && (
                  <div
                    className="absolute border-2 border-emerald-300 bg-emerald-300/15"
                    style={screenBox(draft.bbox) ?? undefined}
                  />
                )}
              </div>
            </div>
          </div>

          <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
            <p className="text-sm text-zinc-500">
              {regions.length
                ? `${regions.length} rectangle${regions.length === 1 ? '' : 's'} selected`
                : 'No rectangles selected'}
            </p>
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
          {editorError && (
            <p className="mt-3 rounded-lg bg-red-500/10 p-3 text-sm text-red-200">
              {editorError}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
