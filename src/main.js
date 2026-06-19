import * as ort from "onnxruntime-web/webgpu";
import "./style.css";

// Configure WASM paths to point to root directory where they are served
ort.env.wasm.wasmPaths = "/";
ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;

// DOM Elements
const webgpuStatusEl = document.getElementById("webgpu-status");
const gpuNameEl = document.getElementById("gpu-name");
const cpuThreadsEl = document.getElementById("cpu-threads");
const browserInfoEl = document.getElementById("browser-info");
const downloadStatusEl = document.getElementById("download-status");
const downloadPercentageEl = document.getElementById("download-percentage");
const downloadProgressFill = document.getElementById("download-progress");
const sessionProviderEl = document.getElementById("session-provider");
const consoleOutputEl = document.getElementById("console-output");

const btnLoad = document.getElementById("btn-load");
const btnRun = document.getElementById("btn-run");
const btnBenchmark = document.getElementById("btn-benchmark");
const btnClearLogs = document.getElementById("btn-clear-logs");

const metricSingleTimeEl = document.getElementById("metric-single-time");
const metricAvgTimeEl = document.getElementById("metric-avg-time");
const metricMinMaxEl = document.getElementById("metric-min-max");
const metricThroughputEl = document.getElementById("metric-throughput");
const benchProgressWrapper = document.getElementById("bench-progress-wrapper");
const benchProgressFill = document.getElementById("bench-progress");
const benchProgressText = document.getElementById("bench-progress-text");

const outShapeEl = document.getElementById("out-shape");
const outTypeEl = document.getElementById("out-type");
const outNormEl = document.getElementById("out-norm");
const tensorValuesEl = document.getElementById("tensor-values");

// New Image Upload & Face Extraction DOM Elements
const uploadZone = document.getElementById("upload-zone");
const fileInput = document.getElementById("file-input");
const canvasWrapper = document.getElementById("canvas-wrapper");
const imageCanvas = document.getElementById("image-canvas");
const btnDetectEmbed = document.getElementById("btn-detect-embed");
const btnLoadSample = document.getElementById("btn-load-sample");
const btnLoadGroup = document.getElementById("btn-load-group");
const detectionStatus = document.getElementById("detection-status");
const facesGrid = document.getElementById("faces-grid");

let session = null;
let originalImage = null;
let detectedFaces = []; // Stores cropped canvases and their calculated embeddings
let faceDetector = null; // MediaPipe detector instance

// Visual Logging System
function log(message, type = "info") {
  const line = document.createElement("div");
  line.className = `log-line ${type}`;
  const timestamp = new Date().toLocaleTimeString();
  line.textContent = `[${timestamp}] ${message}`;
  consoleOutputEl.appendChild(line);
  consoleOutputEl.parentElement.scrollTop = consoleOutputEl.parentElement.scrollHeight;

  // Mirror to browser console
  if (type === "error") {
    console.error(message);
  } else if (type === "warn") {
    console.warn(message);
  } else {
    console.log(message);
  }
}

// Clear Logs
btnClearLogs.addEventListener("click", () => {
  consoleOutputEl.innerHTML = "";
  log("Console logs cleared", "system");
});

// Detect GPU Info via WebGL (or WebGPU if available)
async function detectSystemInfo() {
  cpuThreadsEl.textContent = navigator.hardwareConcurrency || "Unknown";
  
  const ua = navigator.userAgent;
  let browser = "Unknown Browser";
  if (ua.includes("Chrome")) browser = "Chrome / Chromium";
  else if (ua.includes("Firefox")) browser = "Firefox";
  else if (ua.includes("Safari") && !ua.includes("Chrome")) browser = "Safari";
  else if (ua.includes("Edge")) browser = "Microsoft Edge";
  browserInfoEl.textContent = browser;
  browserInfoEl.title = ua;

  let gpuName = "Generic GPU";
  try {
    const canvas = document.createElement("canvas");
    const gl = canvas.getContext("webgl") || canvas.getContext("experimental-webgl");
    if (gl) {
      const debugInfo = gl.getExtension("WEBGL_debug_renderer_info");
      if (debugInfo) {
        gpuName = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
      }
    }
  } catch (e) {
    log("Failed to query WebGL debug renderer info: " + e.message, "warn");
  }
  gpuNameEl.textContent = gpuName;
  gpuNameEl.title = gpuName;

  highlightReferenceRow(gpuName);

  const hasWebGPU = "gpu" in navigator;
  if (hasWebGPU) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) {
        webgpuStatusEl.textContent = "Supported";
        webgpuStatusEl.className = "status-badge success";
        log("WebGPU is supported on this browser and hardware.", "success");
        log(`WebGPU Adapter Info: ${adapter.name || "Default adapter"}`, "info");
      } else {
        webgpuStatusEl.textContent = "Unsupported";
        webgpuStatusEl.className = "status-badge error";
        log("WebGPU supported by browser, but failed to acquire GPU Adapter.", "warn");
      }
    } catch (e) {
      webgpuStatusEl.textContent = "Error";
      webgpuStatusEl.className = "status-badge error";
      log("Error during WebGPU adapter request: " + e.message, "error");
    }
  } else {
    webgpuStatusEl.textContent = "Unsupported";
    webgpuStatusEl.className = "status-badge error";
    log("WebGPU is NOT supported in this browser. Please use Chrome, Edge, or Safari with WebGPU enabled.", "error");
  }
}

function highlightReferenceRow(gpuName) {
  const lowerGpu = gpuName.toLowerCase();
  let deviceKey = "";
  if (lowerGpu.includes("uhd")) {
    deviceKey = "Intel UHD";
  } else if (lowerGpu.includes("iris") || lowerGpu.includes("xe") || lowerGpu.includes("graphics")) {
    deviceKey = lowerGpu.includes("uhd") ? "Intel UHD" : "Intel Iris Xe";
  } else if (lowerGpu.includes("radeon") || lowerGpu.includes("amd") || lowerGpu.includes("ryzen")) {
    deviceKey = "Ryzen iGPU";
  } else if (lowerGpu.includes("apple") || lowerGpu.includes("apple gpu")) {
    if (lowerGpu.includes("m3")) deviceKey = "Apple M3";
    else if (lowerGpu.includes("m2")) deviceKey = "Apple M2";
    else if (lowerGpu.includes("m1")) deviceKey = "Apple M1";
    else deviceKey = "Apple M1";
  }

  if (deviceKey) {
    const row = document.querySelector(`tr[data-device="${deviceKey}"]`);
    if (row) {
      row.className = "active-device";
      row.querySelector("td:last-child").textContent = "Detected Match";
      log(`Matched system GPU with reference target: ${deviceKey}`, "info");
    }
  }
}

// Load Model Handler (with Progress stream)
async function loadModel() {
  btnLoad.disabled = true;
  downloadStatusEl.textContent = "Preparing...";
  log("Initiating model download from /models/buffalo_l.onnx ...", "info");

  try {
    const response = await fetch("/models/buffalo_l.onnx");
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const contentLength = response.headers.get("content-length");
    const total = contentLength ? parseInt(contentLength, 10) : 0;
    let loaded = 0;
    const reader = response.body.getReader();
    const chunks = [];

    downloadStatusEl.textContent = "Downloading...";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      chunks.push(value);
      loaded += value.length;

      if (total) {
        const percent = Math.round((loaded / total) * 100);
        downloadPercentageEl.textContent = `${percent}%`;
        downloadProgressFill.style.width = `${percent}%`;
        downloadStatusEl.textContent = `Downloading: ${(loaded / 1024 / 1024).toFixed(1)} / ${(total / 1024 / 1024).toFixed(1)} MB`;
      } else {
        downloadStatusEl.textContent = `Downloaded ${(loaded / 1024 / 1024).toFixed(1)} MB`;
      }
    }

    log("Model download complete. Compiling model with WebGPU provider...", "info");
    downloadStatusEl.textContent = "Initializing WebGPU...";

    const modelBuffer = new Uint8Array(loaded);
    let offset = 0;
    for (const chunk of chunks) {
      modelBuffer.set(chunk, offset);
      offset += chunk.length;
    }

    log("Creating Inference Session...", "info");
    session = await ort.InferenceSession.create(modelBuffer.buffer, {
      executionProviders: ["webgpu"]
    });

    log("ONNX Inference Session created successfully.", "success");

    log(`Session keys: ${Object.keys(session).join(", ")}`, "info");
    if (session.executionProviders) {
      log(`session.executionProviders: [${session.executionProviders.join(", ")}]`, "info");
    }
    
    const ep = session.executionProvider || (session.executionProviders && session.executionProviders[0]) || "unknown";
    sessionProviderEl.textContent = ep;
    if (ep === "webgpu") {
      sessionProviderEl.className = "status-badge success";
      log("WebGPU Execution Provider actively verified!", "success");
    } else {
      sessionProviderEl.className = "status-badge error";
      log(`Warning: Provider fell back to '${ep}'`, "warn");
    }

    downloadStatusEl.textContent = "Ready";
    btnRun.disabled = false;
    btnBenchmark.disabled = false;

    // Enable detect button if image is already uploaded
    if (originalImage) {
      btnDetectEmbed.disabled = false;
    }

  } catch (e) {
    downloadStatusEl.textContent = "Error loading model";
    downloadPercentageEl.textContent = "Error";
    downloadProgressFill.style.background = "var(--red)";
    log(`Error: ${e.message}`, "error");
    btnLoad.disabled = false;
  }
}

// Generate Dummy Input Tensor [1, 3, 112, 112]
function generateDummyInput() {
  const size = 112 * 112 * 3;
  const data = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    data[i] = Math.random();
  }
  return new ort.Tensor("float32", data, [1, 3, 112, 112]);
}

// Run Single Inference
async function runSingleInference() {
  if (!session) {
    log("Error: Inference session is not loaded.", "error");
    return;
  }

  log("Starting single dummy inference run...", "info");
  btnRun.disabled = true;

  try {
    const inputTensor = generateDummyInput();
    const inputName = session.inputNames[0];
    const feeds = {};
    feeds[inputName] = inputTensor;

    const startTime = performance.now();
    const result = await session.run(feeds);
    const endTime = performance.now();
    const duration = (endTime - startTime).toFixed(2);

    log(`Dummy inference completed in ${duration} ms`, "success");
    metricSingleTimeEl.textContent = duration;

    const outputName = session.outputNames[0];
    const outputTensor = result[outputName];
    
    outShapeEl.textContent = `[${outputTensor.dims.join(", ")}]`;
    outTypeEl.textContent = outputTensor.type;

    const data = outputTensor.data;
    
    let sumSq = 0;
    for (let i = 0; i < data.length; i++) {
      sumSq += data[i] * data[i];
    }
    const l2Norm = Math.sqrt(sumSq).toFixed(4);
    outNormEl.textContent = l2Norm;

    const previewCount = Math.min(12, data.length);
    let previewStr = "Dummy Output Float32Array:\n[";
    for (let i = 0; i < previewCount; i++) {
      previewStr += data[i].toFixed(5) + (i === previewCount - 1 ? "" : ", ");
    }
    previewStr += ` ... (total ${data.length} dims)]`;
    tensorValuesEl.textContent = previewStr;

  } catch (e) {
    log(`Inference failed: ${e.message}`, "error");
  } finally {
    btnRun.disabled = false;
  }
}

// Run 100x Benchmark
async function runBenchmark() {
  if (!session) {
    log("Error: Inference session is not loaded.", "error");
    return;
  }

  log("Starting 100x benchmark run...", "info");
  btnRun.disabled = true;
  btnBenchmark.disabled = true;
  benchProgressWrapper.style.display = "block";

  const totalRuns = 100;
  const latencies = [];
  const inputName = session.inputNames[0];

  try {
    log("Running warmup iteration...", "info");
    const warmupInput = generateDummyInput();
    const warmupFeeds = { [inputName]: warmupInput };
    await session.run(warmupFeeds);
    log("Warmup run completed.", "info");

    for (let i = 0; i < totalRuns; i++) {
      const inputTensor = generateDummyInput();
      const feeds = { [inputName]: inputTensor };

      const start = performance.now();
      await session.run(feeds);
      const end = performance.now();
      
      const duration = end - start;
      latencies.push(duration);

      if (i % 5 === 0 || i === totalRuns - 1) {
        const percent = Math.round(((i + 1) / totalRuns) * 100);
        benchProgressFill.style.width = `${percent}%`;
        benchProgressText.textContent = `${i + 1}/${totalRuns}`;
        await new Promise((resolve) => setTimeout(resolve, 0));
      }
    }

    const totalDuration = latencies.reduce((a, b) => a + b, 0);
    const avgLatency = totalDuration / totalRuns;
    const minLatency = Math.min(...latencies);
    const maxLatency = Math.max(...latencies);
    const throughput = (1000 / avgLatency).toFixed(2);

    log(`Benchmark completed!`, "success");
    log(`Average Latency: ${avgLatency.toFixed(2)} ms`, "success");
    log(`Latency Range: [${minLatency.toFixed(2)} ms - ${maxLatency.toFixed(2)} ms]`, "info");
    log(`Throughput: ${throughput} inferences/second`, "info");

    metricAvgTimeEl.textContent = avgLatency.toFixed(2);
    metricMinMaxEl.textContent = `${minLatency.toFixed(1)} - ${maxLatency.toFixed(1)}`;
    metricThroughputEl.textContent = throughput;

  } catch (e) {
    log(`Benchmark failed: ${e.message}`, "error");
  } finally {
    btnRun.disabled = false;
    btnBenchmark.disabled = false;
    setTimeout(() => {
      benchProgressWrapper.style.display = "none";
      benchProgressFill.style.width = "0%";
      benchProgressText.textContent = "0/100";
    }, 3000);
  }
}

// MediaPipe Face Detector lazy loader
async function initFaceDetector() {
  if (faceDetector) return faceDetector;

  log("Loading MediaPipe Tasks-Vision Face Detector from CDN...", "info");
  try {
    // Dynamic ESM import bypasses Vite compilation errors for external files
    const { FilesetResolver, FaceDetector } = await import(
      /* @vite-ignore */ "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8/vision_bundle.mjs"
    );

    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8/wasm"
    );

    faceDetector = await FaceDetector.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range_sparse/float16/1/blaze_face_full_range_sparse.tflite",
        delegate: "GPU"
      },
      minDetectionConfidence: 0.15, // Low threshold to capture small/distant faces in group images
      runningMode: "IMAGE"
    });

    log("MediaPipe Full-Range Face Detector initialized successfully.", "success");
    return faceDetector;
  } catch (e) {
    log(`Failed to load MediaPipe Face Detector: ${e.message}`, "error");
    throw e;
  }
}

// Drag & Drop / Click Upload Listeners
uploadZone.addEventListener("click", () => fileInput.click());

uploadZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadZone.classList.add("dragover");
});

uploadZone.addEventListener("dragleave", () => {
  uploadZone.classList.remove("dragover");
});

uploadZone.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadZone.classList.remove("dragover");
  const files = e.dataTransfer.files;
  if (files.length > 0) {
    handleUploadedFile(files[0]);
  }
});

fileInput.addEventListener("change", (e) => {
  if (e.target.files.length > 0) {
    handleUploadedFile(e.target.files[0]);
  }
});

// Process uploaded image file
function handleUploadedFile(file) {
  const reader = new FileReader();
  reader.onload = (event) => {
    originalImage = new Image();
    originalImage.onload = () => {
      // Draw to Canvas
      const ctx = imageCanvas.getContext("2d");
      
      // Calculate canvas bounds to fit image without distortion
      const maxW = 500;
      const scale = Math.min(1, maxW / originalImage.width);
      imageCanvas.width = originalImage.width * scale;
      imageCanvas.height = originalImage.height * scale;
      
      ctx.drawImage(originalImage, 0, 0, imageCanvas.width, imageCanvas.height);
      
      // Toggle UI visibility
      uploadZone.style.display = "none";
      canvasWrapper.style.display = "flex";
      
      // Reset faces grid
      facesGrid.innerHTML = `
        <div class="no-faces-placeholder">
          Ready to extract. Click 'Extract & Embed' to run detection and WebGPU inferences.
        </div>
      `;
      detectionStatus.textContent = "Image uploaded successfully";
      
      log(`Uploaded image: ${file.name} (${originalImage.width}x${originalImage.height})`, "info");
      
      if (session) {
        btnDetectEmbed.disabled = false;
      } else {
        log("Model not loaded yet. Load the buffalo_m model first.", "warn");
      }
    };
    originalImage.src = event.target.result;
  };
  reader.readAsDataURL(file);
}

// Convert face canvas image to InsightFace normalized RGB Float32 Tensor
function preprocessFace(faceCanvas) {
  const ctx = faceCanvas.getContext("2d");
  const imgData = ctx.getImageData(0, 0, 112, 112);
  const data = imgData.data; // RGBA

  const size = 112 * 112;
  const floatData = new Float32Array(size * 3);

  // Normalization: (value - 127.5) / 127.5
  // Store flat in RGB order as expected by InsightFace models:
  // floatData[0..size-1] = Red Channel
  // floatData[size..2*size-1] = Green Channel
  // floatData[2*size..3*size-1] = Blue Channel
  for (let i = 0; i < size; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];

    floatData[i] = (r - 127.5) / 127.5;             // R
    floatData[size + i] = (g - 127.5) / 127.5;      // G
    floatData[2 * size + i] = (b - 127.5) / 127.5;  // B
  }

  return new ort.Tensor("float32", floatData, [1, 3, 112, 112]);
}

// Extract Faces and Generate Embeddings Sequential Loop
async function runDetectAndEmbed() {
  if (!session || !originalImage) return;

  btnDetectEmbed.disabled = true;
  detectionStatus.textContent = "Detecting faces...";
  log("Starting face detection sequence...", "info");

  try {
    const detector = await initFaceDetector();
    
    // Detect faces on original native image (preserves high resolution for small/distant group faces)
    const result = detector.detect(originalImage);
    const detections = result.detections || [];
    log(`Detected ${detections.length} faces.`, "success");

    // Redraw image and draw bounding boxes on UI canvas
    const ctx = imageCanvas.getContext("2d");
    ctx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
    ctx.drawImage(originalImage, 0, 0, imageCanvas.width, imageCanvas.height);
    
    ctx.lineWidth = 3;
    ctx.strokeStyle = "#7c4dff"; // purple glowing boxes
    ctx.shadowBlur = 10;
    ctx.shadowColor = "rgba(124, 77, 255, 0.5)";

    facesGrid.innerHTML = "";
    detectedFaces = [];

    if (detections.length === 0) {
      facesGrid.innerHTML = `
        <div class="no-faces-placeholder">
          No faces detected in the image. Try uploading another one.
        </div>
      `;
      detectionStatus.textContent = "0 faces found";
      btnDetectEmbed.disabled = false;
      return;
    }

    detectionStatus.textContent = `Found ${detections.length} faces. Extracting...`;

    // Scale ratio to map coordinates from original image down to resized display canvas
    const scaleX = imageCanvas.width / originalImage.width;
    const scaleY = imageCanvas.height / originalImage.height;

    // Loop to draw boxes and create crop cards (with similarity alignment transform)
    detections.forEach((detection, idx) => {
      const box = detection.boundingBox;

      // Draw bounding box on the display canvas using scaled down coordinates
      ctx.strokeRect(box.originX * scaleX, box.originY * scaleY, box.width * scaleX, box.height * scaleY);

      // Create cropped/aligned canvas (112x112)
      const faceCanvas = document.createElement("canvas");
      faceCanvas.width = 112;
      faceCanvas.height = 112;
      const fCtx = faceCanvas.getContext("2d");

      // Check if keypoints exist for ArcFace similarity transform alignment
      if (detection.keypoints && detection.keypoints.length >= 2) {
        // Retrieve eye coordinates on the native high-resolution image
        const keypoints = detection.keypoints;
        const x1 = keypoints[0].x * originalImage.width;
        const y1 = keypoints[0].y * originalImage.height;
        const x2 = keypoints[1].x * originalImage.width;
        const y2 = keypoints[1].y * originalImage.height;

        // Midpoint of eyes in original image
        const mx = (x1 + x2) / 2;
        const my = (y1 + y2) / 2;

        // Direction vector between eyes (left eye to right eye in template, screen coordinate left-to-right)
        const dx = x2 - x1;
        const dy = y2 - y1;
        const distDet = Math.sqrt(dx * dx + dy * dy);

        // Standard ArcFace eye template targets inside the 112x112 canvas
        const targetDist = 35.2377;
        const targetMx = 47.9132;
        const targetMy = 51.5989;

        // Calculate scale factor and rotation angle
        const scale = targetDist / distDet;
        const angle = Math.atan2(dy, dx);

        fCtx.save();
        // 1. Translate to target midpoint in the new canvas
        fCtx.translate(targetMx, targetMy);
        // 2. Rotate back the detected angle to make eyes horizontal
        fCtx.rotate(-angle);
        // 3. Scale to match eye distance template
        fCtx.scale(scale, scale);
        // 4. Translate detected eye midpoint to origin
        fCtx.translate(-mx, -my);

        // Draw original image with the composite transform matrix
        fCtx.drawImage(originalImage, 0, 0);
        fCtx.restore();
        
        log(`Face #${idx + 1} aligned using eyes keypoints.`, "info");
      } else {
        // Fallback to basic bounding box crop
        const origX = box.originX * scaleX;
        const origY = box.originY * scaleY;
        const origW = box.width * scaleX;
        const origH = box.height * scaleY;

        const srcX = Math.max(0, origX);
        const srcY = Math.max(0, origY);
        const srcW = Math.min(originalImage.width - srcX, origW);
        const srcH = Math.min(originalImage.height - srcY, origH);

        fCtx.drawImage(originalImage, srcX, srcY, srcW, srcH, 0, 0, 112, 112);
        log(`Face #${idx + 1} cropped via basic bounding box (no keypoints found).`, "warn");
      }

      // Create UI Card element for face
      const card = document.createElement("div");
      card.className = "face-card";
      card.dataset.index = idx;

      const title = document.createElement("div");
      title.className = "face-name";
      title.textContent = `Face #${idx + 1}`;

      const statusBadge = document.createElement("div");
      statusBadge.className = "face-status pending";
      statusBadge.textContent = "Pending";

      const timeEl = document.createElement("div");
      timeEl.className = "face-time";
      timeEl.textContent = "-- ms";

      card.appendChild(faceCanvas);
      card.appendChild(title);
      card.appendChild(statusBadge);
      card.appendChild(timeEl);
      facesGrid.appendChild(card);

      detectedFaces.push({
        faceCanvas,
        card,
        statusBadge,
        timeEl,
        embedding: null
      });
    });

    // Disable shadow blur for future canvas drawings
    ctx.shadowBlur = 0;

    log("Running sequential WebGPU inferences on face crops...", "info");
    const inputName = session.inputNames[0];
    const outputName = session.outputNames[0];

    // Sequentially run inference on each cropped face
    for (let i = 0; i < detectedFaces.length; i++) {
      const face = detectedFaces[i];
      face.statusBadge.textContent = "Running";
      face.statusBadge.className = "face-status running";

      // Yield UI process
      await new Promise(resolve => setTimeout(resolve, 50));

      try {
        const inputTensor = preprocessFace(face.faceCanvas);
        const feeds = { [inputName]: inputTensor };

        const start = performance.now();
        const result = await session.run(feeds);
        const end = performance.now();
        const duration = (end - start).toFixed(2);

        const embeddingData = result[outputName].data; // Float32Array (512)

        // Calculate L2 Norm
        let sumSq = 0;
        for (let j = 0; j < embeddingData.length; j++) {
          sumSq += embeddingData[j] * embeddingData[j];
        }
        const l2Norm = Math.sqrt(sumSq);

        // L2 Normalize (matching python's normed_embedding)
        const normalizedEmbedding = new Float32Array(embeddingData.length);
        if (l2Norm > 0) {
          for (let j = 0; j < embeddingData.length; j++) {
            normalizedEmbedding[j] = embeddingData[j] / l2Norm;
          }
        }
        face.embedding = normalizedEmbedding;

        // Update card
        face.statusBadge.textContent = "Complete";
        face.statusBadge.className = "face-status complete";
        face.timeEl.textContent = `${duration} ms`;

        log(`Face #${i + 1} embedding generated in ${duration} ms`, "success");

        // Event listener to view this face's embedding on card click
        face.card.addEventListener("click", () => selectFace(i));
      } catch (err) {
        face.statusBadge.textContent = "Error";
        face.statusBadge.className = "face-status error";
        log(`Failed to process Face #${i + 1}: ${err.message}`, "error");
      }
    }

    detectionStatus.textContent = `Processed ${detectedFaces.length} faces. Click on any face card to inspect its embedding.`;
    
    // Auto-select the first face if successful
    if (detectedFaces.length > 0 && detectedFaces[0].embedding) {
      selectFace(0);
    }

  } catch (err) {
    log(`Face extraction sequence failed: ${err.message}`, "error");
    detectionStatus.textContent = "Error occurred";
  } finally {
    btnDetectEmbed.disabled = false;
  }
}

// Display selected face's embedding vector details
function selectFace(index) {
  detectedFaces.forEach((face, i) => {
    if (i === index) {
      face.card.classList.add("active-face");
    } else {
      face.card.classList.remove("active-face");
    }
  });

  const face = detectedFaces[index];
  if (!face || !face.embedding) return;

  const data = face.embedding;
  outShapeEl.textContent = `[1, ${data.length}]`;
  outTypeEl.textContent = "float32";

  // Calculate L2 Norm
  let sumSq = 0;
  for (let i = 0; i < data.length; i++) {
    sumSq += data[i] * data[i];
  }
  const l2Norm = Math.sqrt(sumSq).toFixed(4);
  outNormEl.textContent = l2Norm;

  // Print preview
  const previewCount = Math.min(12, data.length);
  let previewStr = `Embedding Vector (Face #${index + 1}):\n[`;
  for (let i = 0; i < previewCount; i++) {
    previewStr += data[i].toFixed(5) + (i === previewCount - 1 ? "" : ", ");
  }
  previewStr += ` ... (total ${data.length} dimensions)]`;
  tensorValuesEl.textContent = previewStr;
  
  log(`Displayed embedding vector for Face #${index + 1} (L2 Norm: ${l2Norm})`, "info");
}

// Event Listeners
btnLoad.addEventListener("click", loadModel);
btnRun.addEventListener("click", runSingleInference);
btnBenchmark.addEventListener("click", runBenchmark);
btnDetectEmbed.addEventListener("click", runDetectAndEmbed);

btnLoadSample.addEventListener("click", async () => {
  try {
    log("Fetching sample face image /test-face.png ...", "info");
    const response = await fetch("/test-face.png");
    if (!response.ok) throw new Error("Sample image not found on server");
    const blob = await response.blob();
    const file = new File([blob], "test-face.png", { type: "image/png" });
    handleUploadedFile(file);
    log("Sample face image loaded successfully.", "success");
  } catch (e) {
    log("Failed to load sample image: " + e.message, "error");
  }
});

btnLoadGroup.addEventListener("click", async () => {
  try {
    log("Fetching sample group image /test-group.png ...", "info");
    const response = await fetch("/test-group.png");
    if (!response.ok) throw new Error("Group image not found on server");
    const blob = await response.blob();
    const file = new File([blob], "test-group.png", { type: "image/png" });
    handleUploadedFile(file);
    log("Sample group image loaded successfully.", "success");
  } catch (e) {
    log("Failed to load group image: " + e.message, "error");
  }
});

// Initialize system
detectSystemInfo();
