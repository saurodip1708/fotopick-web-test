import * as ort from 'onnxruntime-web/webgpu';

export interface Keypoint { x: number; y: number; }

export interface DetectedFace {
  bbox: [number, number, number, number]; // x1 y1 x2 y2 in original image coords
  score: number;
  kps: Keypoint[]; // 5 points in original image coords
}

const INPUT_SIZE = 640;
const STRIDES = [8, 16, 32];
const NUM_ANCHORS = 2;
const SCORE_THRESHOLD = 0.5;
const NMS_THRESHOLD = 0.4;

// Returns [tensor, scale, padW, padH] — scale maps original → 640 space
function preprocess(bitmap: ImageBitmap): [ort.Tensor, number, number, number] {
  const scale = Math.min(INPUT_SIZE / bitmap.width, INPUT_SIZE / bitmap.height);
  const dw = Math.round(bitmap.width * scale);
  const dh = Math.round(bitmap.height * scale);

  const canvas = new OffscreenCanvas(INPUT_SIZE, INPUT_SIZE);
  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = 'black';
  ctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  ctx.drawImage(bitmap, 0, 0, dw, dh);

  const imgData = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
  const pixels = imgData.data;
  const size = INPUT_SIZE * INPUT_SIZE;
  const input = new Float32Array(3 * size);

  for (let i = 0; i < size; i++) {
    input[i]          = (pixels[i * 4]     - 127.5) / 128.0; // R
    input[size + i]   = (pixels[i * 4 + 1] - 127.5) / 128.0; // G
    input[2 * size + i] = (pixels[i * 4 + 2] - 127.5) / 128.0; // B
  }

  const tensor = new ort.Tensor('float32', input, [1, 3, INPUT_SIZE, INPUT_SIZE]);
  return [tensor, scale, dw, dh];
}

function buildAnchors(stride: number): Float32Array {
  const n = INPUT_SIZE / stride;
  const anchors = new Float32Array(n * n * NUM_ANCHORS * 2);
  let idx = 0;
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      for (let a = 0; a < NUM_ANCHORS; a++) {
        anchors[idx++] = x * stride;
        anchors[idx++] = y * stride;
      }
    }
  }
  return anchors;
}

function iou(a: [number, number, number, number], b: [number, number, number, number]): number {
  const ix1 = Math.max(a[0], b[0]);
  const iy1 = Math.max(a[1], b[1]);
  const ix2 = Math.min(a[2], b[2]);
  const iy2 = Math.min(a[3], b[3]);
  const iw = Math.max(0, ix2 - ix1);
  const ih = Math.max(0, iy2 - iy1);
  const inter = iw * ih;
  const areaA = (a[2] - a[0]) * (a[3] - a[1]);
  const areaB = (b[2] - b[0]) * (b[3] - b[1]);
  return inter / (areaA + areaB - inter);
}

function nms(dets: DetectedFace[]): DetectedFace[] {
  dets.sort((a, b) => b.score - a.score);
  const keep: DetectedFace[] = [];
  const suppressed = new Uint8Array(dets.length);
  for (let i = 0; i < dets.length; i++) {
    if (suppressed[i]) continue;
    keep.push(dets[i]);
    for (let j = i + 1; j < dets.length; j++) {
      if (suppressed[j]) continue;
      if (iou(dets[i].bbox, dets[j].bbox) > NMS_THRESHOLD) suppressed[j] = 1;
    }
  }
  return keep;
}

export async function runSCRFD(
  session: ort.InferenceSession,
  bitmap: ImageBitmap,
): Promise<DetectedFace[]> {
  const [tensor, scale] = preprocess(bitmap);
  const feeds: Record<string, ort.Tensor> = { [session.inputNames[0]]: tensor };
  const result = await session.run(feeds);

  // Group outputs by last dim: 1=scores, 4=bbox, 10=kps
  const outputs = session.outputNames.map((n) => result[n]);
  const scoreOutputs: [ort.Tensor, number][] = [];
  const bboxOutputs: [ort.Tensor, number][] = [];
  const kpsOutputs: [ort.Tensor, number][] = [];

  for (const t of outputs) {
    const lastDim = t.dims[t.dims.length - 1];
    const count = t.dims[1]; // anchors for this stride
    if (lastDim === 1) scoreOutputs.push([t, count as number]);
    else if (lastDim === 4) bboxOutputs.push([t, count as number]);
    else if (lastDim === 10) kpsOutputs.push([t, count as number]);
  }

  // Sort each group by anchor count descending (stride 8 has most anchors)
  const sortByCount = (a: [ort.Tensor, number], b: [ort.Tensor, number]) => b[1] - a[1];
  scoreOutputs.sort(sortByCount);
  bboxOutputs.sort(sortByCount);
  kpsOutputs.sort(sortByCount);

  const candidates: DetectedFace[] = [];

  for (let si = 0; si < STRIDES.length; si++) {
    const stride = STRIDES[si];
    const scores = scoreOutputs[si][0].data as Float32Array;
    const bboxes = bboxOutputs[si][0].data as Float32Array;
    const kps = kpsOutputs[si][0].data as Float32Array;
    const anchors = buildAnchors(stride);
    const numAnchors = anchors.length / 2;

    for (let i = 0; i < numAnchors; i++) {
      const score = scores[i];
      if (score < SCORE_THRESHOLD) continue;

      const cx = anchors[i * 2];
      const cy = anchors[i * 2 + 1];

      // Decode bbox: anchor - left/top, anchor + right/bottom
      const x1 = (cx - bboxes[i * 4]     * stride) / scale;
      const y1 = (cy - bboxes[i * 4 + 1] * stride) / scale;
      const x2 = (cx + bboxes[i * 4 + 2] * stride) / scale;
      const y2 = (cy + bboxes[i * 4 + 3] * stride) / scale;

      // Decode 5 keypoints
      const kpsArr: Keypoint[] = [];
      for (let k = 0; k < 5; k++) {
        kpsArr.push({
          x: (cx + kps[i * 10 + k * 2]     * stride) / scale,
          y: (cy + kps[i * 10 + k * 2 + 1] * stride) / scale,
        });
      }

      candidates.push({ bbox: [x1, y1, x2, y2], score, kps: kpsArr });
    }
  }

  return nms(candidates);
}
