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
