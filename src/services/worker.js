/**
 * worker.js — spawns and manages Python inference child processes.
 *
 * Each (camera_id, model_id) pair gets exactly ONE Python process.
 * The process is launched with the model's script_path and communicates
 * via stdout JSON lines (one JSON object per detection cycle).
 *
 * Architecture:
 *   Node.js (this file)
 *     └─ spawn()  → python3 detectors/person.py --rtsp ... --model ... --conf ...
 *                     └─ writes person_live.json atomically each frame
 *                     └─ also prints JSON to stdout for Node to read live
 */

const { spawn } = require('child_process');
const path = require('path');
const { workers, models, cameras, pushLog } = require('../store');

const DETECTORS_PATH = process.env.DETECTORS_PATH || path.join(__dirname, '../../detectors');
const MODELS_PATH = process.env.MODELS_PATH || path.join(__dirname, '../../models');
const ASNN_LIBRARY = process.env.ASNN_LIBRARY_PATH || './lib/libnn_yolo26s.so';

/**
 * Build CLI args for each detector script.
 * Each script follows the same interface as person.py.
 */
function buildArgs(camera, model, config, enabledCapabilities) {
  const { localRtsp } = require('./mediamtx');
  const rtspUrl = localRtsp(camera.id);

  const baseArgs = [
    path.join(DETECTORS_PATH, model.script_path.replace('detectors/', '')),
    '--rtsp', rtspUrl,
    '--model', path.join(MODELS_PATH, model.model_path.replace('models/', '')),
    '--library', model.library_path || ASNN_LIBRARY,
    '--conf', String(config.confidence || 0.45),
    '--nms', String(config.nms || 0.56),
    '--headless',
    '--json-stream',   // our scripts accept this flag to print JSON to stdout
  ];

  if (config.fps) baseArgs.push('--fps', String(config.fps));

  // Sub-capability flags — scripts accept e.g. --enable-gender-classification
  if (enabledCapabilities && enabledCapabilities.length > 0) {
    enabledCapabilities.forEach(cap => {
      baseArgs.push(`--enable-${cap.replace(/_/g, '-')}`);
    });
  }

  if (config.zone) {
    baseArgs.push('--zone', JSON.stringify(config.zone));
  }

  return baseArgs;
}

const workerKey = (camId, modelId) => `${camId}::${modelId}`;

/**
 * Start an inference worker for a camera+model pair.
 *
 * @param {string} cameraId
 * @param {string} modelId
 * @param {object} config  - { confidence, fps, nms, zone }
 * @param {string[]} enabledCapabilities - subset of model.capabilities to activate
 * @returns {{ worker_pid, status, stream }}
 */
function startWorker(cameraId, modelId, config = {}, enabledCapabilities = []) {
  const key = workerKey(cameraId, modelId);
  if (workers.has(key)) {
    throw new Error(`Worker already running for ${cameraId} + ${modelId}`);
  }

  const camera = cameras.get(cameraId);
  if (!camera) throw new Error(`Camera ${cameraId} not found`);

  const model = models.get(modelId);
  if (!model) throw new Error(`Model ${modelId} not found`);

  // Activate only requested capabilities (default: all)
  const caps = enabledCapabilities.length > 0
    ? enabledCapabilities.filter(c => model.capabilities.includes(c))
    : model.capabilities;

  const args = buildArgs(camera, model, config, caps);

  // ── In production: un-comment the real spawn ──────────────────────────────
 const proc = spawn('python3', args, {
   stdio: ['ignore', 'pipe', 'pipe'],
   env: { ...process.env },
 });
  //
 proc.stdout.on('data', data => {
   data.toString().split('\n').filter(Boolean).forEach(line => {
     try {
       const payload = JSON.parse(line);
       desc.lastResult = payload;
       desc.fps = payload.fps || desc.fps;
       desc.inference_ms = payload.inference_ms || desc.inference_ms;
     } catch {}
   });
 });
  //
 proc.stderr.on('data', d => console.error(`[worker ${key}]`, d.toString()));
  //
 proc.on('exit', code => {
   console.log(`[worker ${key}] exited with code ${code}`);
   pushLog(cameraId, { event: 'worker_exit', code, model_id: modelId });
   workers.delete(key);
 });
  // ─────────────────────────────────────────────────────────────────────────

  // ── MOCK process (for API testing without NPU hardware) ──────────────────
  const mockPid = Math.floor(10000 + Math.random() * 50000);
//  const proc = {
  //  pid: mockPid,
  //  killed: false,
  //  kill: function () { this.killed = true; workers.delete(key); },
 // };

  // Simulate live detection results updating periodically
  const { localRtsp } = require('./mediamtx');
  const mockInterval = setInterval(() => {
    if (!workers.has(key)) { clearInterval(mockInterval); return; }
    const desc = workers.get(key);
    desc.fps = parseFloat((12 + Math.random() * 6).toFixed(1));
    desc.inference_ms = parseFloat((30 + Math.random() * 20).toFixed(1));
    desc.lastResult = generateMockResult(model, caps, cameraId);
  }, 500);
  // ─────────────────────────────────────────────────────────────────────────

  const desc = {
    camera_id: cameraId,
    model_id: modelId,
    pid: proc.pid,
    fps: 0,
    inference_ms: 0,
    status: 'running',
    enabled_capabilities: caps,
    config: { ...config },
    started_at: new Date().toISOString(),
    stream: localRtsp(cameraId),
    proc,
    lastResult: null,
    _mockInterval: mockInterval,
  };

  workers.set(key, desc);

  // Track assignment on camera + model
  if (!camera.assigned_models) camera.assigned_models = [];
  if (!camera.assigned_models.includes(modelId)) camera.assigned_models.push(modelId);
  if (!model.assigned_cameras.includes(cameraId)) model.assigned_cameras.push(cameraId);

  pushLog(cameraId, { event: 'worker_start', model_id: modelId, pid: proc.pid, capabilities: caps });

  return { worker_pid: proc.pid, status: 'running', stream: localRtsp(cameraId) };
}

/**
 * Stop a specific worker.
 */
function stopWorker(cameraId, modelId) {
  const key = workerKey(cameraId, modelId);
  const desc = workers.get(key);
  if (!desc) throw new Error(`No running worker for ${cameraId} + ${modelId}`);

  if (desc._mockInterval) clearInterval(desc._mockInterval);
  desc.proc.kill('SIGTERM');
  workers.delete(key);

  // Remove assignment bookkeeping
  const camera = cameras.get(cameraId);
  if (camera) camera.assigned_models = (camera.assigned_models || []).filter(m => m !== modelId);
  const model = models.get(modelId);
  if (model) model.assigned_cameras = model.assigned_cameras.filter(c => c !== cameraId);

  pushLog(cameraId, { event: 'worker_stop', model_id: modelId });
  return { status: 'stopped' };
}

/**
 * Stop ALL workers (e.g. before OTA update).
 */
function stopAllWorkers() {
  let count = 0;
  for (const [key, desc] of workers.entries()) {
    if (desc._mockInterval) clearInterval(desc._mockInterval);
    desc.proc.kill('SIGTERM');
    workers.delete(key);
    count++;
  }
  return { stopped: count };
}

/**
 * Update config on a running worker without restarting.
 * The Python script reads a control file or accepts SIGUSR1; here we
 * just update the in-memory descriptor (real impl would signal the proc).
 */
function updateWorkerConfig(cameraId, modelId, patch) {
  const key = workerKey(cameraId, modelId);
  const desc = workers.get(key);
  if (!desc) throw new Error(`No running worker for ${cameraId} + ${modelId}`);
  Object.assign(desc.config, patch);
  // In production: write a config file or send SIGUSR1 to desc.proc.pid
  return { updated: true };
}

/**
 * Update the detection zone polygon for a running worker.
 */
function updateWorkerZone(cameraId, modelId, zone) {
  const key = workerKey(cameraId, modelId);
  const desc = workers.get(key);
  if (!desc) throw new Error(`No running worker for ${cameraId} + ${modelId}`);
  desc.config.zone = zone;
  // In production: write zone to a shared file that the Python script watches
  return { updated: true };
}

/**
 * Get the latest result from a running worker.
 */
function getWorkerResult(cameraId, modelId) {
  const key = workerKey(cameraId, modelId);
  const desc = workers.get(key);
  if (!desc) return null;
  return desc.lastResult;
}

// ── Mock result generators per model type ────────────────────────────────────

function generateMockResult(model, caps, cameraId) {
  const base = {
    updated_unix: Date.now() / 1000,
    updated_local: new Date().toLocaleString(),
    camera_id: cameraId,
    model_id: model.id,
    enabled_capabilities: caps,
    fps: parseFloat((12 + Math.random() * 6).toFixed(1)),
    inference_ms: parseFloat((30 + Math.random() * 20).toFixed(1)),
  };

  switch (model.id) {
    case 'mdl_person': return mockPersonResult(base);
    case 'mdl_face':   return mockFaceResult(base, caps);
    case 'mdl_fire':   return mockFireResult(base, caps);
    case 'mdl_ppe':    return mockPpeResult(base, caps);
    default:           return { ...base, detections: [] };
  }
}

function randBox() {
  const x1 = Math.random() * 0.6;
  const y1 = Math.random() * 0.6;
  return { x1_norm: x1, y1_norm: y1, x2_norm: x1 + 0.1 + Math.random() * 0.25, y2_norm: y1 + 0.2 + Math.random() * 0.35 };
}

function mockPersonResult(base) {
  const count = Math.random() > 0.3 ? Math.ceil(Math.random() * 3) : 0;
  const persons = Array.from({ length: count }, (_, i) => ({
    index: i,
    score: parseFloat((0.45 + Math.random() * 0.5).toFixed(3)),
    ...randBox(),
  }));
  return { ...base, any_person: count > 0, person_count: count, persons };
}

function mockFaceResult(base, caps) {
  const GENDERS = ['male', 'female'];
  const NAMES = ['Alice', 'Bob', 'Charlie', 'Unknown'];
  const count = Math.random() > 0.4 ? Math.ceil(Math.random() * 2) : 0;
  const faces = Array.from({ length: count }, (_, i) => {
    const face = { index: i, score: parseFloat((0.5 + Math.random() * 0.45).toFixed(3)), ...randBox() };
    if (caps.includes('gender_classification')) {
      face.gender = GENDERS[Math.floor(Math.random() * 2)];
      face.gender_score = parseFloat((0.7 + Math.random() * 0.28).toFixed(3));
    }
    if (caps.includes('face_recognition')) {
      face.identity = NAMES[Math.floor(Math.random() * NAMES.length)];
      face.identity_score = face.identity === 'Unknown' ? null : parseFloat((0.6 + Math.random() * 0.38).toFixed(3));
    }
    return face;
  });
  return { ...base, face_count: count, faces };
}

function mockFireResult(base, caps) {
  const events = [];
  if (caps.includes('fire_detection') && Math.random() > 0.85) {
    events.push({ type: 'fire', score: parseFloat((0.6 + Math.random() * 0.35).toFixed(3)), ...randBox() });
  }
  if (caps.includes('smoke_detection') && Math.random() > 0.9) {
    events.push({ type: 'smoke', score: parseFloat((0.55 + Math.random() * 0.4).toFixed(3)), ...randBox() });
  }
  return { ...base, alert: events.length > 0, events };
}

function mockPpeResult(base, caps) {
  const items = [];
  const violations = [];
  const capMap = {
    helmet_detection: 'helmet',
    vest_detection: 'vest',
    gloves_detection: 'gloves',
  };
  caps.forEach(cap => {
    if (capMap[cap] && Math.random() > 0.4) {
      const worn = Math.random() > 0.2;
      const item = { type: capMap[cap], worn, score: parseFloat((0.6 + Math.random() * 0.35).toFixed(3)), ...randBox() };
      items.push(item);
      if (!worn && caps.includes('no_ppe_alert')) violations.push(capMap[cap]);
    }
  });
  return { ...base, ppe_items: items, violations, alert: violations.length > 0 };
}

module.exports = { startWorker, stopWorker, stopAllWorkers, updateWorkerConfig, updateWorkerZone, getWorkerResult, workerKey };
