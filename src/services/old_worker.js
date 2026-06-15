/**
 * worker.js — spawns Python inference child processes.
 *
 * person.py CLI interface (from parse_args):
 *   --library   <path>      required, path to libnn_yolo26s.so
 *   --model     <path>      required, path to yolo26s.nb
 *   --type      rtsp        source type
 *   --device    <url>       RTSP URL (or use --rtsp <url> alias)
 *   --conf      <float>     confidence threshold
 *   --nms       <float>     NMS IoU threshold
 *   --transport tcp|udp
 *   --json-stream           print JSON lines to stdout (keeps stderr clean)
 *   --jpeg-quality <int>    JPEG quality for embedded snapshots
 *   --headless              implied by --json-stream, but explicit is fine
 *   --low-light             enable CLAHE preprocessing
 *
 * JSON output line (one per frame):
 *   {"frame":N, "fps":F, "inference_ms":T, "detections":[...], "jpeg":"<b64>"}
 */

const { spawn } = require('child_process');
const path = require('path');
const { workers, cameras, models, pushLog } = require('../store');

// Absolute paths relative to project root (vision-backend/)
const PROJECT_ROOT  = path.join(__dirname, '../..');
const DETECTORS_DIR = path.join(PROJECT_ROOT, 'detectors');
const MODELS_DIR    = path.join(PROJECT_ROOT, 'models');
const LIB_DIR       = path.join(PROJECT_ROOT, 'lib');

// Model file map: modelId → { script, modelFile, library }
const MODEL_FILES = {
  mdl_person: {
    script:    'person.py',
    modelFile: 'yolo26s.nb',          // lives at models/yolo26s.nb
    library:   'libnn_yolo26s.so',
  },
  // Add more as you build them:
  // mdl_face: { script: 'face.py', modelFile: 'face/face.nb', library: 'libnn_face.so' },
  // mdl_fire: { script: 'fire_smoke.py', modelFile: 'fire/fire.nb', library: 'libnn_fire.so' },
  // mdl_ppe:  { script: 'ppe.py', modelFile: 'ppe/ppe.nb', library: 'libnn_ppe.so' },
};

function startWorker(cameraId, modelId, config = {}, enabledCapabilities = []) {
  const key = `${cameraId}::${modelId}`;

  if (workers.has(key)) {
    const w = workers.get(key);
    return { success: true, workerId: key, pid: w.pid, message: 'Already running' };
  }

  const camera = cameras.get(cameraId);
  if (!camera) throw new Error(`Camera ${cameraId} not found`);

  const modelDef = MODEL_FILES[modelId];
  if (!modelDef) throw new Error(`Model ${modelId} not supported yet. Available: ${Object.keys(MODEL_FILES).join(', ')}`);

  // Use MediaMTX local re-stream URL so the detector gets a stable local feed
  const rtspUrl = camera.local_rtsp || `rtsp://localhost:8554/${cameraId}`;

  const scriptPath  = path.join(DETECTORS_DIR, modelDef.script);
  const modelPath   = path.join(MODELS_DIR, modelDef.modelFile);
  const libraryPath = path.join(LIB_DIR, modelDef.library);

  const args = [
    '--library',      libraryPath,
    '--model',        modelPath,
    '--type',         'rtsp',
    '--device',       rtspUrl,
    '--conf',         String(config.confidence || 0.45),
    '--nms',          String(config.nms || 0.56),
    '--transport',    'tcp',
    '--jpeg-quality', String(config.jpegQuality || 75),
    '--json-stream',                    // JSON on stdout, logs on stderr
  ];

  if (config.lowLight) args.push('--low-light');

  console.log(`[worker ${key}] Spawning: python3 ${modelDef.script}`);
  console.log(`[worker ${key}]   rtsp  : ${rtspUrl}`);
  console.log(`[worker ${key}]   model : ${modelPath}`);
  console.log(`[worker ${key}]   lib   : ${libraryPath}`);

  const proc = spawn('python3', [scriptPath, ...args], {
    stdio: ['ignore', 'pipe', 'pipe'],
    cwd: DETECTORS_DIR,
    env: { ...process.env, PYTHONUNBUFFERED: '1' },
  });

  if (!proc.pid) {
    throw new Error('Failed to spawn python3 — is it installed and on PATH?');
  }

  const workerData = {
    camera_id:           cameraId,
    model_id:            modelId,
    pid:                 proc.pid,
    status:              'running',
    started_at:          new Date().toISOString(),
    fps:                 0,
    inference_ms:        0,
    enabled_capabilities: enabledCapabilities.length > 0 ? enabledCapabilities : ['person_detection'],
    config:              { ...config },
    local_rtsp:          rtspUrl,
    proc,
    lastResult:          null,
  };

  workers.set(key, workerData);

  // ── stdout: JSON detection lines ──────────────────────────────────────────
  let stdoutBuf = '';
  proc.stdout.on('data', (chunk) => {
    stdoutBuf += chunk.toString();
    const lines = stdoutBuf.split('\n');
    stdoutBuf = lines.pop(); // keep incomplete last line in buffer

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed || !trimmed.startsWith('{')) continue;
      try {
        const result = JSON.parse(trimmed);

        // Update live metrics
        if (result.fps)          workerData.fps = result.fps;
        if (result.inference_ms) workerData.inference_ms = result.inference_ms;

        // Attach camera/model context for API consumers
        result.camera_id  = cameraId;
        result.model_id   = modelId;
        result.updated_at = new Date().toISOString();

        workerData.lastResult = result;
      } catch {
        // Not JSON — print for debugging
        console.log(`[worker ${key}] stdout: ${trimmed}`);
      }
    }
  });

  // ── stderr: Python logs ───────────────────────────────────────────────────
  proc.stderr.on('data', (chunk) => {
    const msg = chunk.toString().trim();
    if (msg) console.error(`[worker ${key}] ${msg}`);
  });

  // ── exit ──────────────────────────────────────────────────────────────────
  proc.on('close', (code, signal) => {
    console.log(`[worker ${key}] exited — code=${code} signal=${signal}`);
    workers.delete(key);

    // Remove from camera assignment tracking
    const cam = cameras.get(cameraId);
    if (cam) cam.assigned_models = (cam.assigned_models || []).filter(m => m !== modelId);

    const model = models.get(modelId);
    if (model) model.assigned_cameras = model.assigned_cameras.filter(c => c !== cameraId);

    if (pushLog) pushLog(cameraId, { event: 'worker_exit', model_id: modelId, code });
  });

  proc.on('error', (err) => {
    console.error(`[worker ${key}] spawn error: ${err.message}`);
    workers.delete(key);
  });

  // Bookkeeping
  const cam = cameras.get(cameraId);
  if (cam && !cam.assigned_models?.includes(modelId)) {
    cam.assigned_models = [...(cam.assigned_models || []), modelId];
  }
  const model = models.get(modelId);
  if (model && !model.assigned_cameras?.includes(cameraId)) {
    model.assigned_cameras = [...(model.assigned_cameras || []), cameraId];
  }

  if (pushLog) pushLog(cameraId, { event: 'worker_start', model_id: modelId, pid: proc.pid });

  return {
    worker_pid: proc.pid,
    status:     'running',
    stream:     rtspUrl,
    workerId:   key,
  };
}

function stopWorker(cameraId, modelId) {
  const key = `${cameraId}::${modelId}`;
  const worker = workers.get(key);
  if (!worker) throw new Error(`No running worker for ${key}`);

  worker.proc.kill('SIGTERM');
  workers.delete(key);

  const cam = cameras.get(cameraId);
  if (cam) cam.assigned_models = (cam.assigned_models || []).filter(m => m !== modelId);
  const model = models.get(modelId);
  if (model) model.assigned_cameras = model.assigned_cameras.filter(c => c !== cameraId);

  if (pushLog) pushLog(cameraId, { event: 'worker_stop', model_id: modelId });

  return { status: 'stopped' };
}

function stopAllWorkers() {
  let count = 0;
  for (const [key, w] of workers.entries()) {
    w.proc.kill('SIGTERM');
    workers.delete(key);
    count++;
  }
  return { stopped: count };
}

function updateWorkerConfig(cameraId, modelId, patch) {
  const key = `${cameraId}::${modelId}`;
  const w = workers.get(key);
  if (!w) throw new Error(`No running worker for ${key}`);
  Object.assign(w.config, patch);
  // In production: write a config file that the Python script watches via inotify
  return { updated: true };
}

function updateWorkerZone(cameraId, modelId, zone) {
  const key = `${cameraId}::${modelId}`;
  const w = workers.get(key);
  if (!w) throw new Error(`No running worker for ${key}`);
  w.config.zone = zone;
  return { updated: true };
}

function getWorkerResult(cameraId, modelId) {
  const key = `${cameraId}::${modelId}`;
  const w = workers.get(key);
  return w?.lastResult || null;
}

module.exports = {
  startWorker,
  stopWorker,
  stopAllWorkers,
  updateWorkerConfig,
  updateWorkerZone,
  getWorkerResult,
};
