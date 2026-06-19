import * as ort from 'onnxruntime-web/webgpu';
import { fetchModel } from '../models/loader';
import { createSCRFDSession, type EPName } from '../models/scrfd';
import { createRecognitionSession } from '../models/recognition';
import { runSCRFD } from '../face/detect';
import { normCrop } from '../face/align';
import { runRecognitionBatch } from '../face/embedding';
import type { ImageResult } from '../benchmark/benchmark';

ort.env.wasm.wasmPaths = '/';
ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;

let detector: ort.InferenceSession | null = null;
let recognizer: ort.InferenceSession | null = null;

function report(message: string) {
  self.postMessage({ type: 'log', message });
}

async function loadSession(
  path: string,
  label: string,
  ep: EPName,
  creator: (path: string, ep: EPName) => Promise<ort.InferenceSession>,
): Promise<ort.InferenceSession> {
  const file = path.split('/').pop()!;
  report(`Fetching ${label} (${file})...`);

  const buffer = await fetchModel(path, (loaded, total) => {
    const mb = (loaded / 1024 / 1024).toFixed(1);
    const pct = total ? `${Math.round((loaded / total) * 100)}%` : `${mb} MB`;
    self.postMessage({ type: 'load_progress', label, file, loaded, total, pct });
  });

  report(`${label} downloaded (${(buffer.byteLength / 1024 / 1024).toFixed(1)} MB) — compiling ${ep.toUpperCase()} kernels...`);
  const session = await creator(buffer as unknown as string, ep);

  // Detect which EP was actually used (ORT may fall back to wasm)
  const actualEP = (session as unknown as { executionProvider?: string }).executionProvider
    ?? (session.executionProviders?.[0] ?? ep);
  report(`${label} ready · EP: ${actualEP} · inputs: [${session.inputNames.join(', ')}]`);
  self.postMessage({ type: 'ep_info', label, ep: actualEP });
  return session;
}

// Wrap creators to accept ArrayBuffer directly
async function loadSCRFD(buffer: string, ep: EPName) {
  return ort.InferenceSession.create(buffer as unknown as ArrayBuffer, {
    executionProviders: ep === 'webgpu'
      ? [{ name: 'webgpu' as const, validationMode: 'disabled' as const }, 'wasm']
      : ['wasm'],
    graphOptimizationLevel: 'all',
    enableMemPattern: true,
    extra: { session: { set_denormal_as_zero: '1' } },
  });
}

async function loadRecognizer(buffer: string, ep: EPName) {
  return ort.InferenceSession.create(buffer as unknown as ArrayBuffer, {
    executionProviders: ep === 'webgpu'
      ? [{ name: 'webgpu' as const, validationMode: 'disabled' as const }, 'wasm']
      : ['wasm'],
    graphOptimizationLevel: 'all',
    enableMemPattern: true,
    extra: { session: { set_denormal_as_zero: '1' } },
  });
}

self.onmessage = async (e: MessageEvent) => {
  const { type } = e.data;

  // ── INIT ──────────────────────────────────────────────────────────────────
  if (type === 'init') {
    const {
      detectorPath = '/models/det_10g.onnx',
      recognizerPath = '/models/w600k_r50.onnx',
      ep = 'webgpu',
    } = e.data as { detectorPath?: string; recognizerPath?: string; ep?: EPName };

    try {
      detector   = await loadSession(detectorPath,   'Detector',   ep, loadSCRFD);
      recognizer = await loadSession(recognizerPath, 'Recognizer', ep, loadRecognizer);
      self.postMessage({ type: 'ready' });
    } catch (err) {
      self.postMessage({ type: 'error', message: String(err) });
    }
    return;
  }

  // ── BATCH PROCESS ─────────────────────────────────────────────────────────
  if (type === 'process_batch') {
    const { bitmaps, fileNames } = e.data as {
      bitmaps: ImageBitmap[];
      fileNames: string[];
    };

    if (!detector || !recognizer) {
      bitmaps.forEach((b) => b.close());
      self.postMessage({ type: 'batch_error', message: 'Not initialized' });
      return;
    }

    const wallStart = performance.now();

    try {
      const detectStart = performance.now();
      const perImageFaces: { crop: OffscreenCanvas }[][] = [];

      for (let i = 0; i < bitmaps.length; i++) {
        const faces = await runSCRFD(detector, bitmaps[i]);
        perImageFaces.push(faces.map((f) => ({ crop: normCrop(bitmaps[i], f.kps) })));
      }
      const detectionTimeMs = performance.now() - detectStart;

      const allCrops = perImageFaces.flat().map((x) => x.crop);
      const embStart = performance.now();
      await runRecognitionBatch(recognizer, allCrops);
      const embeddingTimeMs = performance.now() - embStart;

      bitmaps.forEach((b) => b.close());

      const totalTimeMs = performance.now() - wallStart;
      const results: ImageResult[] = fileNames.map((fileName, i) => ({
        fileName,
        faceCount: perImageFaces[i].length,
        detectionTimeMs: detectionTimeMs / fileNames.length,
        embeddingTimeMs: embeddingTimeMs / fileNames.length,
        totalTimeMs: totalTimeMs / fileNames.length,
      }));

      self.postMessage({ type: 'batch_result', results });
    } catch (err) {
      bitmaps.forEach((b) => { try { b.close(); } catch { /* ignore */ } });
      self.postMessage({ type: 'batch_error', message: String(err) });
    }
    return;
  }
};
