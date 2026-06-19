export interface ImageResult {
  fileName: string;
  faceCount: number;
  detectionTimeMs: number;
  embeddingTimeMs: number;
  totalTimeMs: number;
}

export interface BenchmarkStats {
  totalImages: number;
  processed: number;
  facesFound: number;
  detectionMs: number;   // cumulative
  embeddingMs: number;   // cumulative
  totalMs: number;       // wall-clock elapsed
  errors: number;
}

export function emptyStats(): BenchmarkStats {
  return {
    totalImages: 0,
    processed: 0,
    facesFound: 0,
    detectionMs: 0,
    embeddingMs: 0,
    totalMs: 0,
    errors: 0,
  };
}

export function formatElapsed(ms: number): string {
  const s = Math.floor(ms / 1000);
  const m = Math.floor(s / 60);
  const ss = String(s % 60).padStart(2, '0');
  const mm = String(m).padStart(2, '0');
  return `${mm}:${ss}`;
}
