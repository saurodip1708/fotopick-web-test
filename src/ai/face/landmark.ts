import * as ort from 'onnxruntime-web/webgpu';
import { DetectedFace } from './detect';

const LM_SIZE = 192;

export async function createLandmarkSession(): Promise<ort.InferenceSession> {
  return ort.InferenceSession.create('/models/2d106det.onnx', {
    executionProviders: ['webgpu'],
  });
}

// Returns 106 (x, y) pairs in original image coordinates
export async function run106Landmark(
  session: ort.InferenceSession,
  bitmap: ImageBitmap,
  face: DetectedFace,
): Promise<[number, number][]> {
  const [x1, y1, x2, y2] = face.bbox;
  const bw = x2 - x1;
  const bh = y2 - y1;

  // Expand bbox by 1.5x for context
  const cx = (x1 + x2) / 2;
  const cy = (y1 + y2) / 2;
  const half = Math.max(bw, bh) * 0.75;
  const sx = Math.max(0, cx - half);
  const sy = Math.max(0, cy - half);
  const sw = Math.min(bitmap.width - sx, half * 2);
  const sh = Math.min(bitmap.height - sy, half * 2);

  const canvas = new OffscreenCanvas(LM_SIZE, LM_SIZE);
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(bitmap, sx, sy, sw, sh, 0, 0, LM_SIZE, LM_SIZE);

  const imgData = ctx.getImageData(0, 0, LM_SIZE, LM_SIZE);
  const pixels = imgData.data;
  const size = LM_SIZE * LM_SIZE;
  const input = new Float32Array(3 * size);

  for (let i = 0; i < size; i++) {
    input[i]            = (pixels[i * 4]     - 127.5) / 128.0;
    input[size + i]     = (pixels[i * 4 + 1] - 127.5) / 128.0;
    input[2 * size + i] = (pixels[i * 4 + 2] - 127.5) / 128.0;
  }

  const tensor = new ort.Tensor('float32', input, [1, 3, LM_SIZE, LM_SIZE]);
  const feeds: Record<string, ort.Tensor> = { [session.inputNames[0]]: tensor };
  const result = await session.run(feeds);
  const raw = result[session.outputNames[0]].data as Float32Array;

  // Output is [1, 212]: 106 normalized (x,y) pairs in [-1, 1]
  const points: [number, number][] = [];
  for (let i = 0; i < 106; i++) {
    const nx = raw[i * 2];
    const ny = raw[i * 2 + 1];
    // Map from [-1,1] back to original image coordinates
    const px = sx + ((nx + 1) / 2) * sw;
    const py = sy + ((ny + 1) / 2) * sh;
    points.push([px, py]);
  }
  return points;
}
