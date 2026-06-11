#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# setup-face.sh  —  Run this ON the Electron board
# Usage: cd ~/Neha/vision-backend && bash setup-face.sh
# ═══════════════════════════════════════════════════════════════

set -e
cd "$(dirname "$0")"
echo "Working in: $(pwd)"

# ── 1. Create directories ─────────────────────────────────────
echo "[1/5] Creating directories..."
mkdir -p uploads/faces
mkdir -p uploads/enrollment
mkdir -p data/crops
mkdir -p detectors
mkdir -p src/services
mkdir -p src/routes

# ── 2. Write src/index.js ─────────────────────────────────────
echo "[2/5] Writing src/index.js..."
cat > src/index.js << 'INDEXEOF'
require('dotenv').config();

const express = require('express');
const cors    = require('cors');
const morgan  = require('morgan');
const http    = require('http');
const path    = require('path');
const { WebSocketServer } = require('ws');

const app    = express();
const server = http.createServer(app);

app.use(cors());
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));
app.use(morgan('dev'));

// ── Routes ────────────────────────────────────────────────────
app.use('/api/auth',    require('./routes/auth'));
app.use('/api/cameras', require('./routes/cameras'));
app.use('/api/models',  require('./routes/models'));
app.use('/api/detect',  require('./routes/detect'));
app.use('/api/face',    require('./routes/face'));

// Serve face crop images at  GET /crops/<filename>
app.use('/crops', express.static(path.join(__dirname, '..', 'data', 'crops')));

app.get('/health', (req, res) => res.json({ ok: true, ts: new Date().toISOString() }));

app.get('/api', (req, res) => {
  res.json({
    name: 'Vision Backend API',
    version: '1.0.0',
    face_endpoints: {
      'GET  /api/face/worker/status':              'Is face_worker.py running?',
      'POST /api/face/worker/start':               'Start face_worker.py',
      'POST /api/face/worker/stop':                'Stop face_worker.py',
      'POST /api/face/stream/start':               'Start live face stream (capabilities checkbox)',
      'POST /api/face/stream/stop':                'Stop live face stream',
      'GET  /api/face/stream/result/:cameraId':    'Poll latest stream result',
      'POST /api/face/analyze':                    'One-shot image analysis',
      'POST /api/face/persons':                    'Create enrolled person',
      'GET  /api/face/persons':                    'List enrolled persons',
      'GET  /api/face/persons/:id':                'Get one person',
      'PUT  /api/face/persons/:id':                'Update person name/note',
      'DELETE /api/face/persons/:id':              'Delete person',
      'POST /api/face/persons/:id/enroll/image':   'Enroll from image upload',
      'POST /api/face/persons/:id/enroll/video':   'Enroll from video upload',
      'DELETE /api/face/persons/:id/embeddings':   'Clear all embeddings',
    },
  });
});

// ── WebSocket ─────────────────────────────────────────────────
const wss = new WebSocketServer({ server, path: '/ws' });

wss.on('connection', (ws, req) => {
  const params   = new URLSearchParams(req.url.replace('/ws?', ''));
  const cameraId = params.get('camera');
  const modelId  = params.get('model');

  if (!cameraId || !modelId) {
    ws.send(JSON.stringify({ error: 'Use ?camera=<id>&model=<id>' }));
    ws.close();
    return;
  }

  // For face model — push from faceWorkerBridge stream results
  const faceBridge = require('./services/faceWorkerBridge');
  const { getWorkerResult } = require('./services/worker');

  const interval = setInterval(() => {
    if (ws.readyState !== ws.OPEN) { clearInterval(interval); return; }
    let result = null;
    if (modelId === 'mdl_face') {
      result = faceBridge.getLatestStreamResult(cameraId);
    } else {
      result = getWorkerResult(cameraId, modelId);
    }
    if (result) ws.send(JSON.stringify(result));
  }, 500);

  ws.on('close', () => clearInterval(interval));
  ws.on('error', () => clearInterval(interval));
  ws.send(JSON.stringify({ connected: true, camera: cameraId, model: modelId }));
});

app.use((req, res) => res.status(404).json({ error: 'Route not found' }));
app.use((err, req, res, _next) => {
  console.error(err);
  res.status(500).json({ error: err.message || 'Internal server error' });
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log(`\n🚀  Vision Backend running on http://localhost:${PORT}`);
  console.log(`📋  API overview:  http://localhost:${PORT}/api`);
  console.log(`❤️   Health check:  http://localhost:${PORT}/health`);
  console.log(`🔌  WebSocket:     ws://localhost:${PORT}/ws?camera=<id>&model=<id>`);
  console.log('\nDefault credentials:  admin/admin123   viewer/viewer123');
});

module.exports = { app, server };
INDEXEOF

# ── 3. Write src/services/faceWorkerBridge.js ─────────────────
echo "[3/5] Writing faceWorkerBridge.js..."
cat > src/services/faceWorkerBridge.js << 'BRIDGEEOF'
/**
 * faceWorkerBridge.js
 * Manages one long-running face_worker.py process.
 * Commands sent via stdin JSON lines; responses read from stdout JSON lines.
 */
const { spawn }     = require('child_process');
const path          = require('path');
const EventEmitter  = require('events');

const PROJECT_ROOT  = path.join(__dirname, '../..');
const DETECTORS_DIR = path.join(PROJECT_ROOT, 'detectors');
const SCRIPT_PATH   = path.join(DETECTORS_DIR, 'face_worker.py');
const CROPS_DIR     = path.join(PROJECT_ROOT, 'data', 'crops');

class FaceWorkerBridge extends EventEmitter {
  constructor() {
    super();
    this.proc        = null;
    this.ready       = false;
    this.pendingCmds = new Map();
    this.stdoutBuf   = '';
    this.starting    = false;
    this.latestStreamResults = new Map();
  }

  async start() {
    if (this.proc && !this.proc.killed) return;
    if (this.starting) {
      await new Promise(r => this.once('ready', r));
      return;
    }
    this.starting = true;

    return new Promise((resolve, reject) => {
      console.log('[FaceWorker] Spawning face_worker.py...');

      this.proc = spawn('python3', [SCRIPT_PATH], {
        cwd: DETECTORS_DIR,
        stdio: ['pipe', 'pipe', 'pipe'],
        env: { ...process.env, PYTHONUNBUFFERED: '1' },
      });

      if (!this.proc.pid) {
        this.starting = false;
        return reject(new Error('Failed to spawn face_worker.py — is it in detectors/?'));
      }

      this.proc.stdout.on('data', (chunk) => {
        this.stdoutBuf += chunk.toString();
        const lines = this.stdoutBuf.split('\n');
        this.stdoutBuf = lines.pop();
        for (const line of lines) {
          const t = line.trim();
          if (!t || !t.startsWith('{')) continue;
          try { this._handleMessage(JSON.parse(t)); }
          catch { console.error('[FaceWorker] bad JSON:', t.slice(0, 100)); }
        }
      });

      this.proc.stderr.on('data', d => {
        const m = d.toString().trim();
        if (m) console.log('[FaceWorker]', m);
      });

      this.proc.on('close', (code) => {
        console.log(`[FaceWorker] exited (code=${code})`);
        this.ready = false; this.starting = false; this.proc = null;
        for (const [, p] of this.pendingCmds) { clearTimeout(p.timer); p.reject(new Error('face_worker.py died')); }
        this.pendingCmds.clear();
        this.emit('exit', code);
      });

      this.proc.on('error', err => { this.starting = false; reject(err); });

      this.once('ready', () => { this.starting = false; resolve(); });
      setTimeout(() => {
        if (!this.ready) { this.starting = false; reject(new Error('face_worker.py did not become ready in 120s')); }
      }, 120_000);
    });
  }

  _handleMessage(msg) {
    if (msg.event === 'ready') {
      this.ready = true;
      console.log('[FaceWorker] Ready ✓');
      this.emit('ready');
      return;
    }
    if (msg.event === 'stream_match') {
      this.latestStreamResults.set(msg.camera_id, msg);
      this.emit('stream_match', msg);
      return;
    }
    if (msg.cmd) {
      const key = msg.camera_id ? `${msg.cmd}::${msg.camera_id}` : msg.cmd;
      const pending = this.pendingCmds.get(key);
      if (pending) {
        clearTimeout(pending.timer);
        this.pendingCmds.delete(key);
        if (msg.response?.status === 'error') pending.reject(new Error(msg.response.message));
        else pending.resolve(msg.response);
      }
    }
  }

  _send(cmd, payload, timeoutMs = 30_000) {
    if (!this.proc || this.proc.killed)
      return Promise.reject(new Error('face_worker.py is not running'));
    const key = payload.camera_id ? `${cmd}::${payload.camera_id}` : cmd;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pendingCmds.delete(key);
        reject(new Error(`face_worker command "${cmd}" timed out`));
      }, timeoutMs);
      this.pendingCmds.set(key, { resolve, reject, timer });
      this.proc.stdin.write(JSON.stringify({ cmd, ...payload }) + '\n');
    });
  }

  extractEmbedding(imgPath) {
    return this._send('extract_embedding', { img_path: imgPath }, 20_000);
  }
  processVideoEnrollment(videoPath, cropsDir = CROPS_DIR) {
    return this._send('process_video_enrollment', { video_path: videoPath, crops_dir: cropsDir }, 120_000);
  }
  recognizeImage(imgPath, candidates = [], threshold = 0.60, disType = 0, cropsDir = CROPS_DIR) {
    return this._send('recognize_image', { img_path: imgPath, candidates, threshold, dis_type: disType, crops_dir: cropsDir }, 20_000);
  }
  startStream(cameraId, cameraName, rtspUrl, candidates = [], threshold = 0.60, disType = 0, cropsDir = CROPS_DIR) {
    return this._send('start_stream', { camera_id: cameraId, camera_name: cameraName, rtsp_url: rtspUrl, candidates, threshold, dis_type: disType, crops_dir: cropsDir });
  }
  stopStream(cameraId) {
    return this._send('stop_stream', { camera_id: cameraId });
  }
  updateCandidates(candidates = []) {
    return this._send('update_candidates', { candidates });
  }
  getLatestStreamResult(cameraId) {
    return this.latestStreamResults.get(cameraId) || null;
  }
  isReady() { return this.ready && !!this.proc && !this.proc.killed; }
  stop()    { if (this.proc && !this.proc.killed) this.proc.kill('SIGTERM'); }
}

module.exports = new FaceWorkerBridge();
BRIDGEEOF

# ── 4. Write src/services/personStore.js ──────────────────────
echo "[4/5] Writing personStore.js..."
cat > src/services/personStore.js << 'STOREEOF'
const { v4: uuidv4 } = require('uuid');
const persons = new Map();

function createPerson({ name, note = '' }) {
  if (!name) throw new Error('name is required');
  const id = 'p_' + uuidv4().slice(0, 8);
  const p = { person_id: id, name, note, embeddings: [], crop_filenames: [], enrolled_at: new Date().toISOString(), updated_at: new Date().toISOString() };
  persons.set(id, p);
  return p;
}
function getPerson(id)   { return persons.get(id) || null; }
function listPersons()   {
  return Array.from(persons.values()).map(p => ({
    person_id: p.person_id, name: p.name, note: p.note,
    embedding_count: p.embeddings.length, crop_filenames: p.crop_filenames,
    enrolled_at: p.enrolled_at, updated_at: p.updated_at
  }));
}
function updatePerson(id, patch) {
  const p = persons.get(id);
  if (!p) throw new Error(`Person ${id} not found`);
  if (patch.name) p.name = patch.name;
  if (patch.note !== undefined) p.note = patch.note;
  p.updated_at = new Date().toISOString();
  return p;
}
function deletePerson(id) {
  if (!persons.has(id)) throw new Error(`Person ${id} not found`);
  persons.delete(id);
}
function addEmbeddings(id, embeddings, cropFilenames = []) {
  const p = persons.get(id);
  if (!p) throw new Error(`Person ${id} not found`);
  p.embeddings.push(...embeddings);
  p.crop_filenames.push(...cropFilenames);
  p.updated_at = new Date().toISOString();
  return p;
}
function getCandidatesPayload() {
  return Array.from(persons.values())
    .filter(p => p.embeddings.length > 0)
    .map(p => ({ person_id: p.person_id, name: p.name, embeddings: p.embeddings }));
}
module.exports = { createPerson, getPerson, listPersons, updatePerson, deletePerson, addEmbeddings, getCandidatesPayload };
STOREEOF

# ── 5. Write src/routes/face.js ───────────────────────────────
echo "[5/5] Writing src/routes/face.js..."
cat > src/routes/face.js << 'FACEEOF'
const router   = require('express').Router();
const path     = require('path');
const fs       = require('fs');
const multer   = require('multer');
const { requireAuth } = require('../middleware/auth');
const bridge          = require('../services/faceWorkerBridge');
const personStore     = require('../services/personStore');
const { cameras }     = require('../store');

const PROJECT_ROOT = path.join(__dirname, '../..');
const UPLOAD_DIR   = path.join(PROJECT_ROOT, 'uploads');
const CROPS_DIR    = path.join(PROJECT_ROOT, 'data', 'crops');

[UPLOAD_DIR, CROPS_DIR,
 path.join(UPLOAD_DIR, 'faces'),
 path.join(UPLOAD_DIR, 'enrollment')
].forEach(d => fs.mkdirSync(d, { recursive: true }));

const imageUpload = multer({
  storage: multer.diskStorage({
    destination: (req, file, cb) => cb(null, path.join(UPLOAD_DIR, 'faces')),
    filename:    (req, file, cb) => cb(null, `${Date.now()}_${file.originalname}`),
  }),
  limits: { fileSize: 20 * 1024 * 1024 },
  fileFilter: (req, file, cb) => file.mimetype.startsWith('image/') ? cb(null, true) : cb(new Error('Images only')),
});

const videoUpload = multer({
  storage: multer.diskStorage({
    destination: (req, file, cb) => cb(null, path.join(UPLOAD_DIR, 'enrollment')),
    filename:    (req, file, cb) => cb(null, `${Date.now()}_${file.originalname}`),
  }),
  limits: { fileSize: 500 * 1024 * 1024 },
  fileFilter: (req, file, cb) => file.mimetype.startsWith('video/') ? cb(null, true) : cb(new Error('Videos only')),
});

const ALL_CAPS = ['face_detection', 'gender_classification', 'face_recognition'];

function parseCapabilities(body) {
  let raw = body.capabilities;
  if (typeof raw === 'string') {
    try { raw = JSON.parse(raw); } catch { raw = [raw]; }
  }
  if (!raw || !Array.isArray(raw) || raw.length === 0) return [...ALL_CAPS];
  const invalid = raw.filter(c => !ALL_CAPS.includes(c));
  if (invalid.length) throw new Error(`Unknown capabilities: ${invalid.join(', ')}. Valid: ${ALL_CAPS.join(', ')}`);
  return [...new Set(['face_detection', ...raw])];
}

function filterFace(face, caps) {
  const out = { box: face.box, detection_score: face.score ?? 0, crop_filename: face.crop_filename || null };
  if (caps.includes('gender_classification'))  out.gender     = face.gender || null;
  if (caps.includes('face_recognition'))       { out.is_known = face.is_known || false; out.match = face.match || null; out.match_score = face.score || 0; }
  return out;
}

async function ensureWorker(res) {
  try {
    if (!bridge.isReady()) await bridge.start();
    return true;
  } catch (err) {
    res.status(503).json({ error: `face_worker.py failed to start: ${err.message}` });
    return false;
  }
}

// ── Worker status ─────────────────────────────────────────────
router.get('/worker/status', requireAuth, (req, res) => {
  res.json({ running: bridge.isReady(), pid: bridge.proc?.pid || null });
});

router.post('/worker/start', requireAuth, async (req, res) => {
  try {
    await bridge.start();
    res.json({ started: true, pid: bridge.proc?.pid });
  } catch (err) { res.status(500).json({ error: err.message }); }
});

router.post('/worker/stop', requireAuth, (req, res) => {
  bridge.stop();
  res.json({ stopped: true });
});

// ── Live stream ───────────────────────────────────────────────
router.post('/stream/start', requireAuth, async (req, res) => {
  const { camera_id, threshold = 0.60, dis_type = 0 } = req.body || {};
  if (!camera_id) return res.status(400).json({ error: 'camera_id required' });

  const cam = cameras.get(camera_id);
  if (!cam) return res.status(404).json({ error: `Camera ${camera_id} not found` });

  let caps;
  try { caps = parseCapabilities(req.body); }
  catch (e) { return res.status(400).json({ error: e.message }); }

  if (!(await ensureWorker(res))) return;

  const candidates = caps.includes('face_recognition') ? personStore.getCandidatesPayload() : [];

  try {
    const result = await bridge.startStream(
      camera_id, cam.name,
      cam.local_rtsp || `rtsp://localhost:8554/${camera_id}`,
      candidates, threshold, dis_type, CROPS_DIR
    );
    res.json({ started: true, camera_id, capabilities: caps, threshold, message: result.message });
  } catch (err) { res.status(500).json({ error: err.message }); }
});

router.post('/stream/stop', requireAuth, async (req, res) => {
  const { camera_id } = req.body || {};
  if (!camera_id) return res.status(400).json({ error: 'camera_id required' });
  if (!(await ensureWorker(res))) return;
  try {
    const r = await bridge.stopStream(camera_id);
    res.json({ stopped: true, camera_id, message: r.message });
  } catch (err) { res.status(500).json({ error: err.message }); }
});

router.get('/stream/result/:cameraId', requireAuth, (req, res) => {
  const result = bridge.getLatestStreamResult(req.params.cameraId);
  if (!result) return res.status(404).json({ error: 'No result yet — is the stream running?' });
  res.json({ camera_id: result.camera_id, camera_name: result.camera_name, faces: result.faces || [], updated_at: new Date().toISOString() });
});

// ── One-shot image analysis ───────────────────────────────────
router.post('/analyze', requireAuth, imageUpload.single('image'), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'image file required (field: "image")' });
  let caps;
  try { caps = parseCapabilities(req.body); }
  catch (e) { return res.status(400).json({ error: e.message }); }
  if (!(await ensureWorker(res))) return;

  const candidates = caps.includes('face_recognition') ? personStore.getCandidatesPayload() : [];
  const threshold  = parseFloat(req.body.threshold) || 0.60;
  const disType    = parseInt(req.body.dis_type)    || 0;

  try {
    const response = await bridge.recognizeImage(req.file.path, candidates, threshold, disType, CROPS_DIR);
    const faces    = (response.faces || []).map(f => filterFace(f, caps));
    res.json({ faces, face_count: faces.length, capabilities_used: caps, image_file: req.file.filename });
  } catch (err) { res.status(500).json({ error: err.message }); }
});

// ── Person management ─────────────────────────────────────────
router.post('/persons', requireAuth, (req, res) => {
  try { res.status(201).json(personStore.createPerson(req.body)); }
  catch (e) { res.status(400).json({ error: e.message }); }
});

router.get('/persons', requireAuth, (req, res) => res.json(personStore.listPersons()));

router.get('/persons/:id', requireAuth, (req, res) => {
  const p = personStore.getPerson(req.params.id);
  if (!p) return res.status(404).json({ error: 'Person not found' });
  res.json({ ...p, embeddings: undefined, embedding_count: p.embeddings.length });
});

router.put('/persons/:id', requireAuth, (req, res) => {
  try { const p = personStore.updatePerson(req.params.id, req.body); res.json({ person_id: p.person_id, name: p.name, note: p.note, updated: true }); }
  catch (e) { res.status(404).json({ error: e.message }); }
});

router.delete('/persons/:id', requireAuth, async (req, res) => {
  try {
    personStore.deletePerson(req.params.id);
    if (bridge.isReady()) bridge.updateCandidates(personStore.getCandidatesPayload()).catch(() => {});
    res.json({ ok: true });
  } catch (e) { res.status(404).json({ error: e.message }); }
});

// ── Enroll from image ─────────────────────────────────────────
router.post('/persons/:id/enroll/image', requireAuth, imageUpload.single('image'), async (req, res) => {
  const person = personStore.getPerson(req.params.id);
  if (!person) return res.status(404).json({ error: 'Person not found' });
  if (!req.file) return res.status(400).json({ error: 'image file required (field: "image")' });
  if (!(await ensureWorker(res))) return;

  try {
    const response = await bridge.extractEmbedding(req.file.path);
    if (!response.embedding)
      return res.status(422).json({ error: 'No face detected. Use a clear frontal photo (conf>=0.90, size>=80px).' });

    personStore.addEmbeddings(person.person_id, [response.embedding], [req.file.filename]);
    if (bridge.isReady()) await bridge.updateCandidates(personStore.getCandidatesPayload());

    res.json({
      person_id:       person.person_id,
      name:            person.name,
      embedding_count: personStore.getPerson(person.person_id).embeddings.length,
      message:         'Embedding added successfully',
    });
  } catch (err) { res.status(422).json({ error: err.message }); }
});

// ── Enroll from video ─────────────────────────────────────────
router.post('/persons/:id/enroll/video', requireAuth, videoUpload.single('video'), async (req, res) => {
  const person = personStore.getPerson(req.params.id);
  if (!person) return res.status(404).json({ error: 'Person not found' });
  if (!req.file) return res.status(400).json({ error: 'video file required (field: "video")' });
  if (!(await ensureWorker(res))) return;

  try {
    const response = await bridge.processVideoEnrollment(req.file.path, CROPS_DIR);
    if (!response.faces?.length)
      return res.status(422).json({ error: 'No usable faces found in video.' });

    personStore.addEmbeddings(person.person_id, response.faces.map(f => f.embedding), response.faces.map(f => f.filename).filter(Boolean));
    if (bridge.isReady()) await bridge.updateCandidates(personStore.getCandidatesPayload());

    res.json({
      person_id:       person.person_id,
      name:            person.name,
      frames_accepted: response.faces.length,
      embedding_count: personStore.getPerson(person.person_id).embeddings.length,
      message:         `${response.faces.length} embeddings added from video`,
    });
  } catch (err) { res.status(500).json({ error: err.message }); }
});

// ── Clear embeddings ──────────────────────────────────────────
router.delete('/persons/:id/embeddings', requireAuth, async (req, res) => {
  const p = personStore.getPerson(req.params.id);
  if (!p) return res.status(404).json({ error: 'Person not found' });
  p.embeddings = []; p.crop_filenames = []; p.updated_at = new Date().toISOString();
  if (bridge.isReady()) await bridge.updateCandidates(personStore.getCandidatesPayload());
  res.json({ ok: true, person_id: req.params.id, message: 'All embeddings cleared' });
});

module.exports = router;
FACEEOF

# ── Install multer ────────────────────────────────────────────
echo ""
echo "Installing multer..."
npm install multer@1.4.5-lts.1 --save

echo ""
echo "════════════════════════════════════════════════════════"
echo "  ✅  Setup complete!"
echo "  Copy face_worker.py to:  detectors/face_worker.py"
echo "  Copy gender.onnx to:     detectors/gender.onnx"
echo "  Then restart:            node src/index.js"
echo "════════════════════════════════════════════════════════"
