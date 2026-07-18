const router   = require('express').Router();
const path     = require('path');
const fs       = require('fs');
const multer   = require('multer');
const { requireAuth } = require('../middleware/auth');
const log = require('../utils/logger').child('face');
const bridge          = require('../services/faceWorkerBridge');
const personStore     = require('../services/personStore');
const clusterStore    = require('../services/clusterStore');
const lineConfigStore = require('../services/lineConfigStore');
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

// Every unrecognized ("is_known: false") face seen on a live stream is fed
// into clusterStore, which groups recurring strangers by face similarity so
// they can be reviewed and labeled later via the /clusters endpoints below.
// NOTE: the current worker build emits detection ('stream_detect') and
// recognition ('stream_recognize') as two separate, event_uuid-correlated
// events rather than one combined 'stream_match' — we only need the
// recognize side here since that's the one carrying the embedding.
bridge.on('stream_recognize', (face) => {
  if (face.is_known || !face.embedding) return;
  clusterStore.ingestUnknownFace({
    embedding:     face.embedding,
    crop_filename: face.crop_filename,
    camera_id:     face.camera_id,
    gender:        face.gender, // always null on this worker build; harmless
  });
});

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
  const {
    camera_id, threshold = 0.60, dis_type = 0,
    // Optional inline line-crossing override. If provided, this also
    // becomes the new saved config for the camera (same as calling
    // PUT /stream/line-config/:camera_id first). If omitted, whatever
    // was last saved for this camera (or the disabled default) is used.
    line_crossing_enabled, line_y, line_direction, line_x_start, line_x_end,
  } = req.body || {};
  if (!camera_id) return res.status(400).json({ error: 'camera_id required' });

  const cam = cameras.get(camera_id);
  if (!cam) return res.status(404).json({ error: `Camera ${camera_id} not found` });

  let caps;
  try { caps = parseCapabilities(req.body); }
  catch (e) { return res.status(400).json({ error: e.message }); }

  const hasInlineLineConfig = [line_crossing_enabled, line_y, line_direction, line_x_start, line_x_end]
    .some(v => v !== undefined);

  let lineConfig;
  try {
    lineConfig = hasInlineLineConfig
      ? lineConfigStore.setConfig(camera_id, {
          enabled: line_crossing_enabled, line_y, direction: line_direction,
          x_start: line_x_start, x_end: line_x_end,
        })
      : lineConfigStore.getConfig(camera_id);
  } catch (e) { return res.status(400).json({ error: e.message }); }

  if (!(await ensureWorker(res))) return;

  const candidates = caps.includes('face_recognition') ? personStore.getCandidatesPayload() : [];

  try {
    const result = await bridge.startStream(
      camera_id, cam.name,
      cam.local_rtsp || `rtsp://localhost:8554/${camera_id}`,
      candidates, threshold, dis_type, CROPS_DIR, lineConfig
    );
    res.json({ started: true, camera_id, capabilities: caps, threshold, line_config: lineConfig, message: result.message });
  } catch (err) { res.status(500).json({ error: err.message }); }
});

// ── Line-crossing config (the "dynamically drawn line") ─────────
// A frontend lets an operator draw a line over the camera preview (as
// normalized 0–1 fractions of frame width/height) and saves it here. It's
// picked up automatically the next time that camera's face stream starts
// (see /stream/start above), or you can pass line_* fields inline on that
// call instead — both paths write to the same store.
router.get('/stream/line-config/:cameraId', requireAuth, (req, res) => {
  res.json({ camera_id: req.params.cameraId, ...lineConfigStore.getConfig(req.params.cameraId) });
});

router.put('/stream/line-config/:cameraId', requireAuth, async (req, res) => {
  const { cameraId } = req.params;
  if (!cameras.get(cameraId)) return res.status(404).json({ error: `Camera ${cameraId} not found` });

  let cfg;
  try { cfg = lineConfigStore.setConfig(cameraId, req.body || {}); }
  catch (e) { return res.status(400).json({ error: e.message }); }

  // If this camera's stream is already running, restart it so the new line
  // takes effect immediately instead of waiting for the next manual start.
  let restarted = false;
  if (bridge.isReady() && bridge.isStreamActive(cameraId)) {
    try {
      const cam = cameras.get(cameraId);
      const candidates = personStore.getCandidatesPayload();
      await bridge.startStream(
        cameraId, cam.name,
        cam.local_rtsp || `rtsp://localhost:8554/${cameraId}`,
        candidates, req.body?.threshold ?? 0.60, req.body?.dis_type ?? 0, CROPS_DIR, cfg
      );
      restarted = true;
    } catch (e) { /* best-effort — config is saved either way */ }
  }

  res.json({ camera_id: cameraId, ...cfg, restarted });
});

router.delete('/stream/line-config/:cameraId', requireAuth, (req, res) => {
  lineConfigStore.deleteConfig(req.params.cameraId);
  res.json({ camera_id: req.params.cameraId, ...lineConfigStore.DEFAULTS });
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

// ── Live bounding boxes for frontend overlay rendering ──────────
// Returns both raw pixel boxes (native frame resolution) AND normalized
// (0–1) boxes, so the frontend can draw a canvas overlay on top of the
// <video> element regardless of its rendered/display size — just multiply
// box_normalized.x/y/w/h by the video element's current width/height.
router.get('/stream/boxes/:cameraId', requireAuth, (req, res) => {
  const result = bridge.getLatestStreamResult(req.params.cameraId);
  if (!result) return res.status(404).json({ error: 'No result yet — is the stream running?' });

  const fw = result.frame_width  || null;
  const fh = result.frame_height || null;

  const boxes = (result.faces || []).map(f => {
    const [x, y, w, h] = f.box || [0, 0, 0, 0];
    return {
      box: { x, y, w, h }, // raw pixel box (native camera frame resolution)
      box_normalized: (fw && fh) ? {
        x: x / fw,
        y: y / fh,
        w: w / fw,
        h: h / fh,
      } : null,
      is_known:  f.is_known || false,
      person_id: f.match?.person_id || null,
      name:      f.match?.name || null,
      score:     f.score || 0,
      gender:    f.gender || null,
    };
  });

  res.json({
    camera_id:    result.camera_id,
    camera_name:  result.camera_name,
    frame_width:  fw,
    frame_height: fh,
    boxes,
    updated_at:   new Date().toISOString(),
  });
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

// ── Face clusters (recurring, not-yet-labeled strangers) ───────
// Populated automatically from live streams (see the stream_match
// listener above). Label a cluster to turn it into a recognizable Person.
router.get('/clusters', requireAuth, (req, res) => {
  res.json({ clusters: clusterStore.listClusters() });
});

router.get('/clusters/:id', requireAuth, (req, res) => {
  const c = clusterStore.getCluster(req.params.id);
  if (!c) return res.status(404).json({ error: 'Cluster not found' });
  res.json({
    cluster_id:      c.cluster_id,
    seen_count:      c.seen_count,
    embedding_count: c.embeddings.length,
    camera_ids:      Array.from(c.camera_ids),
    crop_filenames:  c.crop_filenames,
    last_gender:     c.last_gender,
    first_seen:      c.first_seen,
    last_seen:       c.last_seen,
  });
});

// Read/update the cosine similarity threshold used to decide whether a new
// unknown face joins an existing cluster. Defaults to env var
// CLUSTER_MATCH_THRESHOLD (or 0.60); changes here apply immediately and take
// effect for the next ingested face, no restart required.
router.get('/clusters/config/threshold', requireAuth, (req, res) => {
  res.json({ threshold: clusterStore.getThreshold() });
});

router.put('/clusters/config/threshold', requireAuth, (req, res) => {
  try {
    const threshold = clusterStore.setThreshold(req.body?.threshold);
    res.json({ threshold, message: 'Cluster match threshold updated.' });
  } catch (e) { res.status(400).json({ error: e.message }); }
});

// Discard a cluster (e.g. it's noise, or crops of unrelated people that got
// merged) without creating/updating any person.
router.delete('/clusters/:id', requireAuth, (req, res) => {
  try { clusterStore.deleteCluster(req.params.id); res.json({ ok: true }); }
  catch (e) { res.status(404).json({ error: e.message }); }
});

// Label a cluster: creates a new Person from its embeddings (or, if
// person_id is supplied, merges its embeddings into an existing Person),
// pushes the updated candidate list to the running worker, and removes the
// cluster. From this point on the labeled face is recognized (is_known:
// true) on future stream/analyze detections.
router.post('/clusters/:id/label', requireAuth, async (req, res) => {
  const cluster = clusterStore.getCluster(req.params.id);
  if (!cluster) return res.status(404).json({ error: 'Cluster not found' });

  const { name, note, person_id } = req.body || {};
  if (!person_id && !name)
    return res.status(400).json({ error: 'name is required (or pass person_id to merge into an existing person)' });

  try {
    let person;
    if (person_id) {
      person = personStore.getPerson(person_id);
      if (!person) return res.status(404).json({ error: `Person ${person_id} not found` });
    } else {
      person = personStore.createPerson({ name, note });
    }

    personStore.addEmbeddings(person.person_id, cluster.embeddings, cluster.crop_filenames);
    if (bridge.isReady()) await bridge.updateCandidates(personStore.getCandidatesPayload());

    clusterStore.deleteCluster(cluster.cluster_id);

    res.json({
      person_id:       person.person_id,
      name:            person.name,
      embedding_count: personStore.getPerson(person.person_id).embeddings.length,
      cluster_id:      cluster.cluster_id,
      message:         'Cluster labeled — this face will now be recognized on future detections.',
    });
  } catch (e) { res.status(500).json({ error: e.message }); }
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
      return res.status(422).json({ error: 'No face detected. Use a clear frontal photo (conf>=0.75, size>=80px).' });

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
