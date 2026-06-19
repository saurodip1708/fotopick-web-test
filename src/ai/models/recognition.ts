import * as ort from 'onnxruntime-web/webgpu';
import type { EPName } from './scrfd';

export async function createRecognitionSession(
  path = '/models/w600k_r50.onnx',
  ep: EPName = 'webgpu',
): Promise<ort.InferenceSession> {
  const epConfig: ort.InferenceSession.SessionOptions['executionProviders'] =
    ep === 'webgpu'
      ? [{ name: 'webgpu', validationMode: 'disabled' }, 'wasm']
      : ['wasm'];

  return ort.InferenceSession.create(path, {
    executionProviders: epConfig,
    graphOptimizationLevel: 'all',
    enableMemPattern: true,
    extra: { session: { set_denormal_as_zero: '1' } },
  });
}
