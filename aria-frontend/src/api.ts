export const API_URL = (import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000').replace(
  /\/$/,
  '',
);

export function assetUrl(path: string | null): string | null {
  if (!path) return null;
  return path.startsWith('http') ? path : `${API_URL}${path}`;
}

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : 'Something went wrong.';
}
