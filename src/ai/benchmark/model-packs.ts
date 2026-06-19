export interface ModelPack {
  id: string;
  label: string;
  description: string;
  detectorPath: string;
  recognizerPath: string;
  detectorSize: string;
  recognizerSize: string;
}

export const MODEL_PACKS: ModelPack[] = [
  {
    id: 'buffalo_l',
    label: 'buffalo_l',
    description: 'Highest accuracy — SCRFD-10G + ResNet50',
    detectorPath: '/models/det_10g.onnx',
    recognizerPath: '/models/w600k_r50.onnx',
    detectorSize: '17 MB',
    recognizerSize: '167 MB',
  },
  {
    id: 'buffalo_m',
    label: 'buffalo_m',
    description: 'Balanced — SCRFD-10G + MobileFaceNet',
    detectorPath: '/models/det_10g.onnx',
    recognizerPath: '/models/w600k_mbf.onnx',
    detectorSize: '17 MB',
    recognizerSize: '13 MB',
  },
  {
    id: 'buffalo_s',
    label: 'buffalo_s',
    description: 'Fastest — SCRFD-500M + MobileFaceNet',
    detectorPath: '/models/det_500m.onnx',
    recognizerPath: '/models/w600k_mbf.onnx',
    detectorSize: '2.5 MB',
    recognizerSize: '13 MB',
  },
];

export const DEFAULT_PACK = MODEL_PACKS[0];
