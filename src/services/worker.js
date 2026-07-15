/**
 * worker.js — spawns Python inference child processes.
 *
 * Built-in models  → person.py / fire_smoke.py / etc.
 *   CLI:  --library <.so>  --model <.nb>  --type rtsp  --device <url>
 *         --conf  --nms  --transport  --jpeg-quality  --json-stream
 *
 * Custom NPU model (.nb + .so)  → generic_detector.py
 *   CLI:  same as built-in PLUS  --classes '["cow","goat"]'  --imgsz 640
 *
 * Custom TFLite model (.tflite) → tflite_detector.py        ← NEW
 *   CLI:  --model <.tflite>  --classes <classes.txt>
 *         --rtsp <url>  --conf  --fps  --headless  --json-stream
 *         (NO --library — runs on CPU via tflite-runtime)
 *
 * Custom ONNX model (.onnx)     → onnx_detector.py          ← NEW
 *   CLI:  --model <.onnx>  --classes <classes.txt>
 *         --rtsp <url>  --conf  --fps  --headless  --json-stream
 *         (NO --library — runs on CPU via onnxruntime)
 *
 * JSON output line (one per frame) — same shape for all detectors:
 *   {"frame":N, "fps":F, "inference_ms":T, "detections":[...], "jpeg":"<b64>"}
 */

'use strict';

const { spawn }  = require('child_process');
const path       = require('path');
const fs         = require('fs');
const { workers, cameras, models, pushLog } = require('../store');

// Absolute paths relative to project root (vision-backend/)
const PROJECT_ROOT  = path.join(__dirname, '../..');
const DETECTORS_DIR = path.join(PROJECT_ROOT, 'detectors');
const MODELS_DIR    = path.join(PROJECT_ROOT, 'models');
const LIB_DIR       = path.join(PROJECT_ROOT, 'lib');

// ── Built-in model definitions ────────────────────────────────────────────────
const MODEL_FILES = {
  mdl_person: {
    script:    'person.py',
    modelFile: 'yolo26s.nb',
    library:   'libnn_yolo26s.so',
    defaultCaps: ['person_detection'],
  },
  mdl_fire: {
    script:    'fire_smoke.py',
    modelFile: 'fire/fire.nb',
    library:   'libnn_fire.so',
    defaultCaps: ['fire_detection', 'smoke_detection'],
  },
  // Add more built-ins here as needed
};

// ── Internal helpers ──────────────────────────────────────────────────────────

/**
 * Find the classes file inside an extracted model directory.
 * Returns the absolute path, or throws if none found.
 */
function findClassesFile(extractDir) {
  const files = fs.readdirSync(extractDir);
  const found = files.find(f =>
    ['.txt', '.names', '.labels'].includes(path.extname(f).toLowerCase())
  );
  if (!found) {
    throw new Error(
      `No class-names file found in ${extractDir}. ` +
      `Expected a .txt / .names / .labels file alongside the model.`
    );
  }
  return path.join(extractDir, found);
}

/**
 * Build the spawn args for a TFLite or ONNX custom model.
 * These detectors do NOT use an NPU library.
 */
function buildCpuArgs(scriptPath, modelPath, classesPath, rtspUrl, config, enabledCapabilities, model) {
  const args = [
    scriptPath,
    '--model',    modelPath,
    '--classes',  classesPath,
    '--rtsp',     rtspUrl,
    '--conf',     String(config.confidence  ?? model.default_conf ?? 0.45),
    '--fps',      String(config.fps         ?? 5),
    '--headless',
    '--json-out',
  ];

  // Per-class enable flags — if none specified, tflite/onnx scripts default to all
  const caps = enabledCapabilities.length > 0
    ? enabledCapabilities
    : (model.class_names || []);
  for (const cls of caps) {
    args.push(`--enable-${cls.replace(/\s+/g, '-')}`);
  }

  return args;
}

/**
 * Build the spawn args for an NPU model (.nb + .so),
 * both built-in and custom-uploaded.
 */
function buildNpuArgs(scriptPath, modelPath, libraryPath, rtspUrl, config, enabledCapabilities, model) {
  const args = [
    scriptPath,
    '--library',      libraryPath,
    '--model',        modelPath,
    '--type',         'rtsp',
    '--device',       rtspUrl,
    '--conf',         String(config.confidence ?? model?.default_conf ?? 0.45),
    '--nms',          String(config.nms        ?? model?.default_nms  ?? 0.56),
    '--transport',    'tcp',
    '--jpeg-quality', String(config.jpegQuality ?? 75),
    '--json-stream',
  ];

  if (config.lowLight) args.push('--low-light');

  // Custom NPU model gets --classes + --imgsz
  if (model?.type === 'custom') {
    args.push('--classes', JSON.stringify(model.class_names || []));
    args.push('--imgsz',   String(model.input_size || 640));

    const caps = enabledCapabilities.length > 0
      ? enabledCapabilities
      : (model.class_names || []);
    for (const cls of caps) {
      args.push(`--enable-${cls.replace(/\s+/g, '-')}`);
    }
  }

  return args;
}

// ── Public API ────────────────────────────────────────────────────────────────

function startWorker(cameraId, modelId, config = {}, enabledCapabilities = []) {
  const key = `${cameraId}::${modelId}`;

  if (workers.has(key)) {
    const w = workers.get(key);
    return { success: true, workerId: key, pid: w.pid, message: 'Already running' };
  }

  const camera = cameras.get(cameraId);
  if (!camera) throw new Error(`Camera ${cameraId} not found`);

  const rtspUrl = camera.local_rtsp || `rtsp://localhost:8554/${cameraId}`;

  let scriptPath, args, defaultCaps;
  const modelDef = MODEL_FILES[modelId];

  if (modelDef) {
    // ── Built-in NPU model ──────────────────────────────────────────────────
    scriptPath   = path.join(DETECTORS_DIR, modelDef.script);
    const modelPath   = path.join(MODELS_DIR,    modelDef.modelFile);
    const libraryPath = path.join(LIB_DIR,       modelDef.library);
    defaultCaps  = modelDef.defaultCaps;

    _assertExists('Detector script', scriptPath);
    _assertExists('Model file',      modelPath);
    _assertExists('Library file',    libraryPath);

    args = buildNpuArgs(scriptPath, modelPath, libraryPath, rtspUrl, config, enabledCapabilities, null);

  } else {
    // ── Custom uploaded model ───────────────────────────────────────────────
    const customModel = models.get(modelId);
    if (!customModel || customModel.type !== 'custom') {
      throw new Error(
        `Model ${modelId} not found. ` +
        `Available built-ins: ${Object.keys(MODEL_FILES).join(', ')}`
      );
    }
    if (!customModel.model_path || !customModel.script_path) {
      throw new Error(`Model ${modelId} is incomplete — re-upload the package`);
    }

    scriptPath  = customModel.script_path;
    defaultCaps = customModel.class_names || [];

    _assertExists('Detector script', scriptPath);
    _assertExists('Model file',      customModel.model_path);

    if (customModel.format === 'tflite' || customModel.format === 'onnx') {
      // ── CPU inference (TFLite / ONNX) — no library required ──────────────
      const classesPath = findClassesFile(customModel.extract_dir);

      args = buildCpuArgs(
        scriptPath,
        customModel.model_path,
        classesPath,
        rtspUrl,
        config,
        enabledCapabilities,
        customModel
      );

    } else {
      // ── NPU inference (.nb + .so) ─────────────────────────────────────────
      if (!customModel.library_path) {
        throw new Error(`Model ${modelId} (.nb) is missing library_path — re-upload the package`);
      }
      _assertExists('Library file', customModel.library_path);

      args = buildNpuArgs(
        scriptPath,
        customModel.model_path,
        customModel.library_path,
        rtspUrl,
        config,
        enabledCapabilities,
        customModel
      );
    }
  }

  console.log(`[worker ${key}] Spawning: python3 ${path.basename(scriptPath)}`);
  console.log(`[worker ${key}]   format: ${MODEL_FILES[modelId] ? 'nb (built-in)' : (models.get(modelId)?.format || 'unknown')}`);
  console.log(`[worker ${key}]   rtsp  : ${rtspUrl}`);

  const proc = spawn('python3', args, {
    stdio: ['ignore', 'pipe', 'pipe'],
    cwd: path.dirname(scriptPath),
    env: { ...process.env, PYTHONUNBUFFERED: '1' },
  });

  if (!proc.pid) {
    throw new Error('Failed to spawn python3 — is it installed and on PATH?');
  }

  const workerData = {
    camera_id:            cameraId,
    model_id:             modelId,
    pid:                  proc.pid,
    status:               'running',
    started_at:           new Date().toISOString(),
    fps:                  0,
    inference_ms:         0,
    enabled_capabilities: enabledCapabilities.length > 0 ? enabledCapabilities : defaultCaps,
    config:               { ...config },
    local_rtsp:           rtspUrl,
    proc,
    lastResult:           null,
  };

  workers.set(key, workerData);

  // ── stdout: JSON detection lines ──────────────────────────────────────────
  let stdoutBuf = '';
  proc.stdout.on('data', (chunk) => {
    stdoutBuf += chunk.toString();
    const lines = stdoutBuf.split('\n');
    stdoutBuf = lines.pop();

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed || !trimmed.startsWith('{')) continue;
      try {
        const result = JSON.parse(trimmed);

        if (result.fps)          workerData.fps          = result.fps;
        if (result.inference_ms) workerData.inference_ms = result.inference_ms;

        result.camera_id  = cameraId;
        result.model_id   = modelId;
        result.updated_at = new Date().toISOString();

        workerData.lastResult = result;
      } catch {
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

// ── Private ───────────────────────────────────────────────────────────────────

function _assertExists(label, filePath) {
  if (!fs.existsSync(filePath)) {
    throw new Error(`${label} not found: ${filePath}`);
  }
}

module.exports = {
  startWorker,
  stopWorker,
  stopAllWorkers,
  updateWorkerConfig,
  updateWorkerZone,
  getWorkerResult,
};