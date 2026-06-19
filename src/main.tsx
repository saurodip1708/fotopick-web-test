import * as ort from 'onnxruntime-web/webgpu';
import { createRoot } from 'react-dom/client';
import BenchmarkPage from './pages/benchmark/page';
import './style.css';

ort.env.wasm.wasmPaths = '/';
ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;

createRoot(document.getElementById('root')!).render(<BenchmarkPage />);
