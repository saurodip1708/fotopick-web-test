import { Keypoint } from './detect';

// ArcFace 5-point template for 112x112 output
const TEMPLATE: [number, number][] = [
  [38.2946, 51.6963],
  [73.5318, 51.5014],
  [56.0252, 71.7366],
  [41.5493, 92.3655],
  [70.7299, 92.2041],
];

// Closed-form 2D similarity transform (umeyama, 4 DOF)
// Maps src points → dst points: x' = a*x - b*y + e, y' = b*x + a*y + f
function similarityTransform(
  src: [number, number][],
  dst: [number, number][],
): [number, number, number, number] {
  const n = src.length;
  let mx = 0, my = 0, tx = 0, ty = 0;
  for (let i = 0; i < n; i++) {
    mx += src[i][0]; my += src[i][1];
    tx += dst[i][0]; ty += dst[i][1];
  }
  mx /= n; my /= n; tx /= n; ty /= n;

  let numA = 0, numB = 0, denom = 0;
  for (let i = 0; i < n; i++) {
    const xi = src[i][0] - mx;
    const yi = src[i][1] - my;
    const xi_ = dst[i][0] - tx;
    const yi_ = dst[i][1] - ty;
    numA += xi * xi_ + yi * yi_;
    numB += xi * yi_ - yi * xi_;
    denom += xi * xi + yi * yi;
  }

  const a = numA / denom;
  const b = numB / denom;
  const e = tx - a * mx + b * my;
  const f = ty - b * mx - a * my;
  return [a, b, e, f];
}

// Produces a 112x112 OffscreenCanvas aligned to ArcFace template
export function normCrop(image: ImageBitmap, kps: Keypoint[]): OffscreenCanvas {
  const src = kps.slice(0, 5).map((p) => [p.x, p.y] as [number, number]);
  const [a, b, e, f] = similarityTransform(src, TEMPLATE);

  const canvas = new OffscreenCanvas(112, 112);
  const ctx = canvas.getContext('2d')!;
  // Canvas setTransform(a,b,c,d,e,f): x'=a*X+c*Y+e, y'=b*X+d*Y+f
  // Our transform: x'=a*X - b*Y + e, y'=b*X + a*Y + f → c=-b, d=a
  ctx.setTransform(a, b, -b, a, e, f);
  ctx.drawImage(image, 0, 0);
  return canvas;
}
