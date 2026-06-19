import * as ort from 'onnxruntime-web/webgpu';

export type EPName = 'webgpu' | 'wasm';

export async function createSCRFDSession(
  path = '/models/det_10g.onnx',
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
