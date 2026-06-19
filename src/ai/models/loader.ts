export type ProgressCallback = (loaded: number, total: number, label: string) => void;

// Fetches a model file with streaming progress. Throws immediately on 404/error.
// Passes the resulting ArrayBuffer to ORT so ORT never touches a URL that might 404 silently.
export async function fetchModel(
  path: string,
  onProgress?: ProgressCallback,
): Promise<ArrayBuffer> {
  const res = await fetch(path);
  if (!res.ok) {
    throw new Error(`Model not found: ${path} — HTTP ${res.status} ${res.statusText}`);
  }

  const contentLength = res.headers.get('content-length');
  const total = contentLength ? parseInt(contentLength, 10) : 0;
  const label = path.split('/').pop() ?? path;

  if (!res.body || !onProgress) {
    onProgress?.(total, total, label);
    return res.arrayBuffer();
  }

  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  let lastReportedPct = -1;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    // Report every 5% (or always if total is unknown)
    const pct = total ? Math.floor((loaded / total) * 20) * 5 : -1;
    if (pct !== lastReportedPct) {
      lastReportedPct = pct;
      onProgress(loaded, total, label);
    }
  }
  // Always report 100%
  onProgress(loaded, total, label);

  const out = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) { out.set(chunk, offset); offset += chunk.length; }
  return out.buffer;
}
