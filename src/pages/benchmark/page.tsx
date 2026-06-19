import { useRef, useState, useCallback, useEffect } from 'react';
import { BenchmarkStats, ImageResult, emptyStats, formatElapsed } from '../../ai/benchmark/benchmark';
import { MODEL_PACKS, ModelPack } from '../../ai/benchmark/model-packs';
import type { EPName } from '../../ai/models/scrfd';
import EmbeddingWorker from '../../ai/workers/embedding.worker.ts?worker';

type WorkerStatus = 'idle' | 'loading' | 'ready' | 'running' | 'done' | 'error';

const ACCEPTED_TYPES = new Set(['image/jpeg', 'image/png', 'image/webp', 'image/bmp']);
const CONCURRENCY_OPTIONS = [1, 2, 4, 6, 8] as const;
const BATCH_OPTIONS = [4, 8, 16] as const;
const EP_OPTIONS: { value: EPName; label: string; hint: string }[] = [
  { value: 'webgpu', label: 'WebGPU', hint: 'Fastest — Chrome/Edge on Windows/Mac' },
  { value: 'wasm',   label: 'WASM',   hint: 'CPU fallback — works everywhere, slower' },
];
type Concurrency = typeof CONCURRENCY_OPTIONS[number];
type BatchSize = typeof BATCH_OPTIONS[number];

function isImageFile(f: File) {
  return ACCEPTED_TYPES.has(f.type) || /\.(jpe?g|png|webp|bmp)$/i.test(f.name);
}

interface WorkerSlot {
  worker: Worker;
  ready: boolean;
  busy: boolean;
  processed: number;
}

export default function BenchmarkPage() {
  const [workerStatus, setWorkerStatus] = useState<WorkerStatus>('idle');
  const [concurrency, setConcurrency] = useState<Concurrency>(2);
  const [batchSize, setBatchSize] = useState<BatchSize>(8);
  const [modelPack, setModelPack] = useState<ModelPack>(MODEL_PACKS[0]);
  const [ep, setEp] = useState<EPName>('webgpu');
  const modelPackRef = useRef<ModelPack>(MODEL_PACKS[0]);
  const epRef = useRef<EPName>('webgpu');
  const [stats, setStats] = useState<BenchmarkStats>(emptyStats());
  const [log, setLog] = useState<string[]>(['[SYSTEM] Ready. Select a folder to begin.']);
  const [gpuName, setGpuName] = useState('Detecting...');
  const [webgpuOk, setWebgpuOk] = useState<boolean | null>(null);
  const [cpuThreads] = useState(navigator.hardwareConcurrency || '?');
  const [deviceMemory] = useState((navigator as unknown as { deviceMemory?: number }).deviceMemory ?? '?');
  const [workerSlotStates, setWorkerSlotStates] = useState<{ ready: boolean; busy: boolean }[]>([]);

  const poolRef = useRef<WorkerSlot[]>([]);
  const queueRef = useRef<File[]>([]);
  const statsRef = useRef<BenchmarkStats>(emptyStats());
  const startTimeRef = useRef<number>(0);
  const elapsedTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const batchSizeRef = useRef<BatchSize>(batchSize);
  batchSizeRef.current = batchSize;

  const addLog = useCallback((msg: string) => {
    setLog((prev) => [...prev.slice(-299), msg]);
  }, []);

  function syncWorkerStates() {
    setWorkerSlotStates(poolRef.current.map((s) => ({ ready: s.ready, busy: s.busy })));
  }

  // GPU info
  useEffect(() => {
    try {
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl');
      if (gl) {
        const ext = gl.getExtension('WEBGL_debug_renderer_info');
        if (ext) setGpuName(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL));
      }
    } catch { /* ignore */ }
    if ('gpu' in navigator) {
      navigator.gpu.requestAdapter().then((a) => setWebgpuOk(!!a)).catch(() => setWebgpuOk(false));
    } else {
      setWebgpuOk(false);
    }
  }, []);

  // Spawn / respawn worker pool when concurrency, EP, or model pack changes
  useEffect(() => {
    modelPackRef.current = modelPack;
    epRef.current = ep;
    poolRef.current.forEach((s) => s.worker.terminate());
    poolRef.current = [];
    setWorkerStatus('loading');
    addLog(`[SYSTEM] Spawning ${concurrency} worker(s) · ${modelPack.id} · ${ep.toUpperCase()}...`);

    const slots: WorkerSlot[] = Array.from({ length: concurrency }, () => ({
      worker: new EmbeddingWorker(),
      ready: false,
      busy: false,
      processed: 0,
    }));
    poolRef.current = slots;
    syncWorkerStates();

    slots.forEach((slot, idx) => {
      slot.worker.onmessage = (e: MessageEvent) => handleWorkerMessage(e, idx);
      slot.worker.onerror = (err) => {
        addLog(`[FATAL] Worker ${idx}: ${err.message}`);
        setWorkerStatus('error');
      };
    });

    // Init workers one at a time — WebGPU session creation is not re-entrant.
    // Worker N+1 only receives its init message after Worker N fires 'ready'.
    // initNextWorker is called from handleWorkerMessage on each 'ready' event.
    slots[0].worker.postMessage({
      type: 'init',
      detectorPath: modelPack.detectorPath,
      recognizerPath: modelPack.recognizerPath,
      ep,
    });
    addLog(`[SYSTEM] Initializing worker 0 (${modelPack.id} · ${ep.toUpperCase()})...`);

    return () => slots.forEach((s) => s.worker.terminate());
  }, [concurrency, modelPack, ep]); // eslint-disable-line react-hooks/exhaustive-deps

  function handleWorkerMessage(e: MessageEvent, workerIdx: number) {
    const { type } = e.data;
    const pool = poolRef.current;
    const slot = pool[workerIdx];

    if (type === 'log') {
      addLog(`[W${workerIdx}] ${e.data.message}`);
      return;
    }

    if (type === 'ep_info') {
      addLog(`[W${workerIdx}] ${e.data.label} running on: ${String(e.data.ep).toUpperCase()}`);
      return;
    }

    if (type === 'load_progress') {
      const { label, pct, loaded, total } = e.data;
      const mb = (loaded / 1024 / 1024).toFixed(1);
      const totalMb = total ? `/ ${(total / 1024 / 1024).toFixed(1)} MB` : '';
      addLog(`[W${workerIdx}] ${label}: ${mb} MB ${totalMb} (${pct})`);
      return;
    }

    if (type === 'ready') {
      slot.ready = true;
      syncWorkerStates();
      addLog(`[OK] Worker ${workerIdx} ready.`);

      // Chain: init next worker only after this one is fully loaded
      const nextIdx = workerIdx + 1;
      if (nextIdx < pool.length && !pool[nextIdx].ready) {
        addLog(`[SYSTEM] Initializing worker ${nextIdx} (${modelPackRef.current.id} · ${epRef.current.toUpperCase()})...`);
        pool[nextIdx].worker.postMessage({
          type: 'init',
          detectorPath: modelPackRef.current.detectorPath,
          recognizerPath: modelPackRef.current.recognizerPath,
          ep: epRef.current,
        });
      }

      if (pool.every((s) => s.ready)) {
        setWorkerStatus('ready');
        addLog(`[OK] All ${pool.length} workers ready. Batch: ${batchSizeRef.current} images.`);
      }
      return;
    }

    if (type === 'error') {
      addLog(`[ERROR] Worker ${workerIdx}: ${e.data.message}`);
      setWorkerStatus('error');
      return;
    }

    // ── Batch result ─────────────────────────────────────────────────────────
    slot.busy = false;
    slot.processed += (e.data.results?.length ?? 0);
    syncWorkerStates();

    if (type === 'batch_result') {
      const results: ImageResult[] = e.data.results;
      let newFaces = 0;
      let detMs = 0;
      let embMs = 0;
      for (const r of results) {
        newFaces += r.faceCount;
        detMs += r.detectionTimeMs;
        embMs += r.embeddingTimeMs;
        addLog(`[W${workerIdx}] ${r.fileName} — ${r.faceCount} face(s)`);
      }
      statsRef.current = {
        ...statsRef.current,
        processed: statsRef.current.processed + results.length,
        facesFound: statsRef.current.facesFound + newFaces,
        detectionMs: statsRef.current.detectionMs + detMs,
        embeddingMs: statsRef.current.embeddingMs + embMs,
      };
      setStats({ ...statsRef.current });
    } else if (type === 'batch_error') {
      addLog(`[W${workerIdx}][ERR] ${e.data.message}`);
      // We don't know exactly how many images were in the batch; best effort
      statsRef.current = { ...statsRef.current, errors: statsRef.current.errors + 1 };
      setStats({ ...statsRef.current });
    }

    // Done?
    if (queueRef.current.length === 0 && pool.every((s) => !s.busy)) {
      finishBenchmark();
      return;
    }

    dispatchBatchToWorker(workerIdx);
  }

  function dispatchBatchToWorker(workerIdx: number) {
    const queue = queueRef.current;
    if (queue.length === 0) return;

    const slot = poolRef.current[workerIdx];
    const batch = queue.splice(0, batchSizeRef.current);
    if (batch.length === 0) return;

    slot.busy = true;
    syncWorkerStates();

    // Decode ENTIRE batch in parallel on main thread.
    // This overlaps with GPU inference in other workers.
    Promise.all(
      batch.map((f) => createImageBitmap(f).catch(() => null)),
    ).then((rawBitmaps) => {
      const valid = rawBitmaps.filter(Boolean) as ImageBitmap[];
      const validNames = batch
        .map((f, i) => (rawBitmaps[i] ? f.name : null))
        .filter(Boolean) as string[];

      if (valid.length === 0) {
        slot.busy = false;
        syncWorkerStates();
        statsRef.current = { ...statsRef.current, errors: statsRef.current.errors + batch.length };
        setStats({ ...statsRef.current });
        dispatchBatchToWorker(workerIdx);
        return;
      }

      slot.worker.postMessage(
        { type: 'process_batch', bitmaps: valid, fileNames: validNames },
        valid as unknown as Transferable[],
      );
    });
  }

  function finishBenchmark() {
    if (elapsedTimerRef.current) clearInterval(elapsedTimerRef.current);
    const elapsed = performance.now() - startTimeRef.current;
    statsRef.current = { ...statsRef.current, totalMs: elapsed };
    setStats({ ...statsRef.current });
    setWorkerStatus('done');
    const s = statsRef.current;
    const tput = s.processed > 0 ? (s.processed / (elapsed / 1000)).toFixed(1) : '0';
    addLog(`[DONE] ${s.processed} images · ${s.facesFound} faces · ${tput} img/s · ${formatElapsed(elapsed)}`);
  }

  function onFolderChange(e: React.ChangeEvent<HTMLInputElement>) {
    const files = Array.from(e.target.files ?? []).filter(isImageFile);
    if (files.length === 0) return;

    const fresh: BenchmarkStats = { ...emptyStats(), totalImages: files.length };
    statsRef.current = fresh;
    setStats(fresh);
    setLog([`[INFO] Queued ${files.length} images · ${concurrency} workers · ${batchSize} images/batch`]);

    queueRef.current = files;
    startTimeRef.current = performance.now();
    setWorkerStatus('running');

    if (elapsedTimerRef.current) clearInterval(elapsedTimerRef.current);
    elapsedTimerRef.current = setInterval(() => {
      setStats((s) => ({ ...s, totalMs: performance.now() - startTimeRef.current }));
    }, 500);

    // Kickstart every worker with its first batch simultaneously
    poolRef.current.forEach((_, idx) => dispatchBatchToWorker(idx));
  }

  const isReady = workerStatus === 'ready' || workerStatus === 'done';
  const isRunning = workerStatus === 'running';
  const progress = stats.totalImages ? (stats.processed / stats.totalImages) * 100 : 0;
  const throughput = stats.processed > 0 && stats.totalMs > 500
    ? (stats.processed / (stats.totalMs / 1000)).toFixed(1)
    : '--';

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="header-content">
          <div className="badge glow-purple">FOTOPICK BENCHMARK</div>
          <h1>Face Pipeline — Browser</h1>
          <p className="subtitle">SCRFD · ArcFace Batch Recognition · WebGPU via ONNX Runtime Web</p>
        </div>
      </header>

      <main className="dashboard-grid">

        {/* System Diagnostics */}
        <section className="card diagnostic-card">
          <div className="card-header"><h2>System Diagnostics</h2><div className="header-icon">⚙️</div></div>
          <div className="card-body">
            <div className="metric-row">
              <span className="label">WebGPU:</span>
              <span className={`status-badge ${webgpuOk === null ? 'loading' : webgpuOk ? 'success' : 'error'}`}>
                {webgpuOk === null ? 'Checking...' : webgpuOk ? 'Enabled' : 'Unsupported'}
              </span>
            </div>
            <div className="metric-row">
              <span className="label">GPU:</span>
              <span className="value text-truncate" title={gpuName}>{gpuName}</span>
            </div>
            <div className="metric-row">
              <span className="label">CPU Threads:</span>
              <span className="value">{cpuThreads}</span>
            </div>
            <div className="metric-row">
              <span className="label">Device Memory:</span>
              <span className="value">{deviceMemory} GB</span>
            </div>
            <div className="metric-row" style={{ marginTop: '1rem', alignItems: 'flex-start' }}>
              <span className="label">Workers:</span>
              <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                {workerSlotStates.map((s, i) => (
                  <span key={i} className={`status-badge ${!s.ready ? 'loading' : s.busy ? 'running' : 'success'}`}
                    style={{ fontSize: '0.7rem' }}>
                    W{i} {!s.ready ? 'init' : s.busy ? 'busy' : 'idle'}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* Pipeline Config */}
        <section className="card config-card">
          <div className="card-header"><h2>Pipeline Config</h2><div className="header-icon">🧠</div></div>
          <div className="card-body">
            <div className="metric-row" style={{ marginBottom: '0.75rem' }}>
              <span className="label">Backend EP:</span>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                {EP_OPTIONS.map((o) => (
                  <button key={o.value}
                    onClick={() => !isRunning && setEp(o.value)}
                    disabled={isRunning}
                    title={o.hint}
                    className={`btn btn-small ${ep === o.value ? 'btn-primary' : 'btn-secondary'}`}
                    style={{ minWidth: '4rem' }}>
                    {o.label}
                  </button>
                ))}
              </div>
            </div>

            <div className="metric-row" style={{ marginBottom: '0.75rem' }}>
              <span className="label">Workers:</span>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                {CONCURRENCY_OPTIONS.map((n) => (
                  <button key={n} onClick={() => !isRunning && setConcurrency(n)} disabled={isRunning}
                    className={`btn btn-small ${concurrency === n ? 'btn-primary' : 'btn-secondary'}`}
                    style={{ minWidth: '2.5rem' }}>
                    {n}×
                  </button>
                ))}
              </div>
            </div>
            <div className="metric-row" style={{ marginBottom: '1rem' }}>
              <span className="label">Batch:</span>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                {BATCH_OPTIONS.map((n) => (
                  <button key={n} onClick={() => !isRunning && setBatchSize(n)} disabled={isRunning}
                    className={`btn btn-small ${batchSize === n ? 'btn-primary' : 'btn-secondary'}`}
                    style={{ minWidth: '2.5rem' }}>
                    {n}
                  </button>
                ))}
              </div>
            </div>
            <div className="metric-row" style={{ marginBottom: '1rem' }}>
              <span className="label">Model Pack:</span>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                {MODEL_PACKS.map((pack) => (
                  <button key={pack.id}
                    onClick={() => !isRunning && setModelPack(pack)}
                    disabled={isRunning}
                    className={`btn btn-small ${modelPack.id === pack.id ? 'btn-primary' : 'btn-secondary'}`}>
                    {pack.label}
                  </button>
                ))}
              </div>
            </div>
            <div style={{ fontSize: '0.76rem', marginBottom: '0.75rem' }}>
              <div className="metric-row">
                <span className="label">Detector:</span>
                <span className="value text-muted">{modelPack.detectorPath.split('/').pop()} · {modelPack.detectorSize}</span>
              </div>
              <div className="metric-row">
                <span className="label">Recognizer:</span>
                <span className="value text-muted">{modelPack.recognizerPath.split('/').pop()} · {modelPack.recognizerSize}</span>
              </div>
              <p className="text-muted" style={{ marginTop: '0.5rem', lineHeight: 1.5 }}>
                {modelPack.description}
              </p>
            </div>
            <p className="text-muted" style={{ fontSize: '0.76rem', lineHeight: 1.55 }}>
              <strong>Batch</strong> = images per dispatch. All faces across N images run through
              recognition in <strong>one GPU call</strong>. For iGPU: 2× · batch 8. dGPU: 4× · batch 16.
            </p>
          </div>
        </section>

        {/* Folder Picker */}
        <section className="card upload-card">
          <div className="card-header"><h2>Image Folder</h2><div className="header-icon">📁</div></div>
          <div className="card-body">
            <div className="upload-dropzone"
              style={{ cursor: isReady ? 'pointer' : 'not-allowed', opacity: isReady ? 1 : 0.5 }}>
              <label style={{ display: 'block', cursor: 'inherit', padding: '2rem', textAlign: 'center' }}>
                <div className="upload-icon">📂</div>
                <p className="upload-text"><strong>Select a folder</strong> of images</p>
                <p className="upload-info">JPG · PNG · WebP · BMP — fully offline</p>
                <input type="file"
                  /* @ts-ignore webkitdirectory */
                  webkitdirectory=""
                  multiple accept="image/*"
                  style={{ display: 'none' }}
                  disabled={!isReady}
                  onChange={onFolderChange}
                  onClick={(e) => { (e.target as HTMLInputElement).value = ''; }}
                />
              </label>
            </div>

            {(isRunning || workerStatus === 'done') && (
              <div className="benchmark-progress-container" style={{ marginTop: '1rem' }}>
                <div className="progress-bar-track">
                  <div className="progress-bar-fill accent-fill"
                    style={{ width: `${progress}%`, transition: 'width 0.25s' }} />
                </div>
                <div className="progress-labels">
                  <span>{workerStatus === 'done' ? '✓ Complete' : 'Processing...'}</span>
                  <span>{stats.processed} / {stats.totalImages}</span>
                </div>
              </div>
            )}
          </div>
        </section>

        {/* Live Stats */}
        <section className="card metrics-card">
          <div className="card-header"><h2>Live Stats</h2><div className="header-icon">⚡</div></div>
          <div className="card-body">
            <div className="metrics-grid">
              <div className="metric-box">
                <span className="metric-title">Selected</span>
                <span className="metric-val">{stats.totalImages.toLocaleString()}</span>
                <span className="metric-unit">images</span>
              </div>
              <div className="metric-box">
                <span className="metric-title">Processed</span>
                <span className="metric-val">{stats.processed.toLocaleString()}</span>
                <span className="metric-unit">images</span>
              </div>
              <div className="metric-box">
                <span className="metric-title">Faces Found</span>
                <span className="metric-val">{stats.facesFound.toLocaleString()}</span>
                <span className="metric-unit">faces</span>
              </div>
              <div className="metric-box">
                <span className="metric-title">Throughput</span>
                <span className="metric-val">{throughput}</span>
                <span className="metric-unit">img / sec</span>
              </div>
            </div>

            <div className="metrics-grid" style={{ marginTop: '1rem' }}>
              <div className="metric-box">
                <span className="metric-title">Detection</span>
                <span className="metric-val" style={{ fontSize: '1.4rem' }}>{formatElapsed(stats.detectionMs)}</span>
                <span className="metric-unit">cumulative</span>
              </div>
              <div className="metric-box">
                <span className="metric-title">Embedding</span>
                <span className="metric-val" style={{ fontSize: '1.4rem' }}>{formatElapsed(stats.embeddingMs)}</span>
                <span className="metric-unit">cumulative</span>
              </div>
              <div className="metric-box">
                <span className="metric-title">Elapsed</span>
                <span className="metric-val" style={{ fontSize: '1.4rem' }}>{formatElapsed(stats.totalMs)}</span>
                <span className="metric-unit">wall clock</span>
              </div>
              <div className="metric-box">
                <span className="metric-title">Avg / Image</span>
                <span className="metric-val">
                  {stats.processed > 0 && stats.totalMs > 0
                    ? (stats.totalMs / stats.processed).toFixed(0) : '--'}
                </span>
                <span className="metric-unit">ms</span>
              </div>
            </div>
          </div>
        </section>

        {/* Log */}
        <section className="card console-card" style={{ gridColumn: '1 / -1' }}>
          <div className="card-header">
            <h2>Log</h2>
            <button className="btn-clear" onClick={() => setLog([])}>Clear</button>
          </div>
          <div className="card-body">
            <div className="console-wrapper">
              <div className="console-output"
                ref={(el) => { if (el) el.scrollTop = el.scrollHeight; }}>
                {log.map((line, i) => {
                  const cls =
                    line.startsWith('[OK]') || line.startsWith('[DONE]') ? 'success' :
                    line.startsWith('[ERR') || line.startsWith('[FATAL') ? 'error' :
                    line.startsWith('[SYSTEM]') || line.startsWith('[INFO]') ? 'system' : 'info';
                  return <div key={i} className={`log-line ${cls}`}>{line}</div>;
                })}
              </div>
            </div>
          </div>
        </section>

      </main>

      <footer className="app-footer">
        <p>Fotopick · {modelPack.id} · {concurrency}× Workers · Batch {batchSize} · ONNX Runtime Web {webgpuOk ? '+ WebGPU' : '+ WASM'}</p>
      </footer>
    </div>
  );
}
