/**
 * Inference / Detect routes — 6 endpoints
 *
 * POST   /api/detect/start          Spawn inference worker (with capability selection)
 * POST   /api/detect/stop           Kill specific worker
 * GET    /api/detect/status         List all running workers
 * POST   /api/detect/stop-all       Kill all workers (admin+)
 * PUT    /api/detect/config         Update conf/fps without restart
 * POST   /api/detect/zone           Update detection zone polygon
 *
 * Extra (convenience):
 * GET    /api/detect/result/:cameraId/:modelId   Latest detection result
 */

const router = require('express').Router();
const { workers, models, cameras } = require('../store');
const { requireAuth, requireRole } = require('../middleware/auth');
const {
  startWorker,
  stopWorker,
  stopAllWorkers,
  updateWorkerConfig,
  updateWorkerZone,
  getWorkerResult,
} = require('../services/worker');

// ── POST /api/detect/start ────────────────────────────────────────────────────
/**
 * Body:
 * {
 *   "camera_id": "cam_001",
 *   "model_id": "mdl_face",
 *   "confidence": 0.45,
 *   "fps": 5,
 *   "capabilities": ["face_detection", "gender_classification"]
 *   //               ^ subset of model.capabilities to enable (omit = enable all)
 * }
 *
 * The `capabilities` array lets the client act as a checkbox UI:
 *   Face model has: face_detection, gender_classification, face_recognition
 *   Checking only "face_detection" + "gender_classification" sends exactly those two.
 *   The worker spawns with --enable-face-detection --enable-gender-classification.
 */
router.post('/start', requireAuth, (req, res) => {
  const { camera_id, model_id, confidence = 0.45, fps = 5, capabilities, zone } = req.body || {};

  if (!camera_id || !model_id)
    return res.status(400).json({ error: 'camera_id and model_id are required' });

  if (!cameras.has(camera_id))
    return res.status(404).json({ error: `Camera ${camera_id} not found` });

  const model = models.get(model_id);
  if (!model)
    return res.status(404).json({ error: `Model ${model_id} not found` });

  // Validate requested capabilities against what the model supports
  let enabledCaps = model.capabilities; // default: all
  if (Array.isArray(capabilities) && capabilities.length > 0) {
    const invalid = capabilities.filter(c => !model.capabilities.includes(c));
    if (invalid.length > 0) {
      return res.status(400).json({
        error: `Unknown capabilities: ${invalid.join(', ')}`,
        available: model.capabilities,
      });
    }
    enabledCaps = capabilities;
  }

  try {
    const result = startWorker(camera_id, model_id, { confidence, fps, zone }, enabledCaps);
    res.json(result);
  } catch (err) {
    res.status(409).json({ error: err.message });
  }
});

// ── POST /api/detect/stop ─────────────────────────────────────────────────────

router.post('/stop', requireAuth, (req, res) => {
  const { camera_id, model_id } = req.body || {};
  if (!camera_id || !model_id)
    return res.status(400).json({ error: 'camera_id and model_id are required' });

  try {
    const result = stopWorker(camera_id, model_id);
    res.json(result);
  } catch (err) {
    res.status(404).json({ error: err.message });
  }
});

// ── GET /api/detect/status ────────────────────────────────────────────────────

router.get('/status', requireAuth, (req, res) => {
  const list = Array.from(workers.values()).map(w => ({
    camera_id: w.camera_id,
    model_id: w.model_id,
    pid: w.pid,
    fps: w.fps,
    inference_ms: w.inference_ms,
    status: w.status,
    enabled_capabilities: w.enabled_capabilities,
    started_at: w.started_at,
    stream: w.stream,
    config: w.config,
  }));
  res.json(list);
});

// ── POST /api/detect/stop-all ─────────────────────────────────────────────────

router.post('/stop-all', requireAuth, requireRole('admin'), (req, res) => {
  const result = stopAllWorkers();
  res.json(result);
});

// ── PUT /api/detect/config ────────────────────────────────────────────────────

router.put('/config', requireAuth, (req, res) => {
  const { camera_id, model_id, confidence, fps } = req.body || {};
  if (!camera_id || !model_id)
    return res.status(400).json({ error: 'camera_id and model_id are required' });

  try {
    const result = updateWorkerConfig(camera_id, model_id, { confidence, fps });
    res.json(result);
  } catch (err) {
    res.status(404).json({ error: err.message });
  }
});

// ── POST /api/detect/zone ─────────────────────────────────────────────────────

router.post('/zone', requireAuth, (req, res) => {
  const { camera_id, model_id, zone } = req.body || {};
  if (!camera_id || !model_id || !zone)
    return res.status(400).json({ error: 'camera_id, model_id, and zone are required' });

  if (!Array.isArray(zone) || zone.length < 3)
    return res.status(400).json({ error: 'zone must be an array of at least 3 [x,y] normalised coordinates' });

  try {
    const result = updateWorkerZone(camera_id, model_id, zone);
    res.json(result);
  } catch (err) {
    res.status(404).json({ error: err.message });
  }
});

// ── GET /api/detect/result/:cameraId/:modelId ─────────────────────────────────
// Convenience endpoint — poll the latest detection output from a running worker.

router.get('/result/:cameraId/:modelId', requireAuth, (req, res) => {
  const { cameraId, modelId } = req.params;

  if (!cameras.has(cameraId))
    return res.status(404).json({ error: `Camera ${cameraId} not found` });
  if (!models.has(modelId))
    return res.status(404).json({ error: `Model ${modelId} not found` });

  const result = getWorkerResult(cameraId, modelId);
  if (!result)
    return res.status(404).json({ error: 'No result yet — is the worker running?' });

  res.json(result);
});

// ── GET /api/detect/capabilities/:modelId ────────────────────────────────────
// Returns the available capability checkboxes for a model.

router.get('/capabilities/:modelId', requireAuth, (req, res) => {
  const model = models.get(req.params.modelId);
  if (!model) return res.status(404).json({ error: `Model ${req.params.modelId} not found` });

  res.json({
    model_id: model.id,
    model_name: model.name,
    capabilities: model.capabilities,
    description: capabilityDescriptions(model.id),
  });
});

function capabilityDescriptions(modelId) {
  const desc = {
    mdl_person: {
      person_detection: 'Detect and localise people in the frame with bounding boxes and confidence scores.',
    },
    mdl_face: {
      face_detection: 'Detect and localise faces — bounding boxes + confidence.',
      gender_classification: 'Classify detected faces as male or female.',
      face_recognition: 'Match faces against enrolled identities (requires enrollment database).',
    },
    mdl_fire: {
      fire_detection: 'Detect visible flames.',
      smoke_detection: 'Detect smoke plumes.',
    },
    mdl_ppe: {
      helmet_detection: 'Detect safety helmets (worn / not worn).',
      vest_detection: 'Detect high-visibility vests (worn / not worn).',
      gloves_detection: 'Detect protective gloves (worn / not worn).',
      no_ppe_alert: 'Raise an alert when any required PPE item is missing.',
    },
  };
  return desc[modelId] || {};
}

module.exports = router;
