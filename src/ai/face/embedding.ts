import * as ort from 'onnxruntime-web/webgpu';

const EMBED_DIM = 512;
export const EMBED_CHUNK = 16; // fixed batch size for graph capture — never changes

function l2Normalize(vec: Float32Array): Float32Array {
  let mag = 0;
  for (let i = 0; i < vec.length; i++) mag += vec[i] * vec[i];
  mag = Math.sqrt(mag);
  const out = new Float32Array(vec.length);
  for (let i = 0; i < vec.length; i++) out[i] = vec[i] / mag;
  return out;
}

// Build a [EMBED_CHUNK, 3, 112, 112] tensor.
// If fewer than EMBED_CHUNK crops are provided, remaining slots are zero-filled (black face).
// This keeps the shape constant so enableGraphCapture can replay the GPU command buffer.
function buildFixedBatchTensor(crops: OffscreenCanvas[], n: number): ort.Tensor {
  const size = 112 * 112;
  const input = new Float32Array(EMBED_CHUNK * 3 * size); // zero-initialised by default

  for (let i = 0; i < n; i++) {
    const ctx = crops[i].getContext('2d')!;
    const pixels = ctx.getImageData(0, 0, 112, 112).data;
    const base = i * 3 * size;
    for (let p = 0; p < size; p++) {
      input[base + p]            = (pixels[p * 4]     - 127.5) / 127.5;
      input[base + size + p]     = (pixels[p * 4 + 1] - 127.5) / 127.5;
      input[base + 2 * size + p] = (pixels[p * 4 + 2] - 127.5) / 127.5;
    }
  }

  return new ort.Tensor('float32', input, [EMBED_CHUNK, 3, 112, 112]);
}

export async function runRecognition(
  session: ort.InferenceSession,
  crop: OffscreenCanvas,
): Promise<Float32Array> {
  const [emb] = await runRecognitionBatch(session, [crop]);
  return emb;
}

// Processes crops in chunks of exactly EMBED_CHUNK (padded with black if short).
// Returns one normalized Float32Array(512) per input crop (padding results discarded).
export async function runRecognitionBatch(
  session: ort.InferenceSession,
  crops: OffscreenCanvas[],
): Promise<Float32Array[]> {
  if (crops.length === 0) return [];

  const results: Float32Array[] = [];

  for (let i = 0; i < crops.length; i += EMBED_CHUNK) {
    const end = Math.min(i + EMBED_CHUNK, crops.length);
    const realCount = end - i;
    const chunk = crops.slice(i, end);

    const tensor = buildFixedBatchTensor(chunk, realCount);
    const feeds: Record<string, ort.Tensor> = { [session.inputNames[0]]: tensor };
    const out = await session.run(feeds);
    const raw = out[session.outputNames[0]].data as Float32Array;

    // Only take the realCount valid embeddings — padded slots are ignored
    for (let n = 0; n < realCount; n++) {
      results.push(l2Normalize(raw.slice(n * EMBED_DIM, (n + 1) * EMBED_DIM)));
    }
  }

  return results;
}
