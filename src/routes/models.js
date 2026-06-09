/**
 * Model routes — 7 endpoints
 *
 * GET    /api/models                   List all models
 * GET    /api/models/:id               Single model detail
 * POST   /api/models/upload            Upload custom .atomomodel (admin+)
 * POST   /api/models/:id/validate      Re-validate model package
 * POST   /api/models/:id/test          Run test inference
 * DELETE /api/models/:id               Delete custom model (admin+)
 * GET    /api/models/:id/assignments   List camera assignments
 */

const router = require('express').Router();
const { models, cameras, uuidv4 } = require('../store');
const { requireAuth, requireRole } = require('../middleware/auth');

function notFound(res, id) {
  return res.status(404).json({ error: `Model ${id} not found` });
}

// ── GET /api/models ───────────────────────────────────────────────────────────

router.get('/', requireAuth, (req, res) => {
  const list = Array.from(models.values()).map(m => ({
    id: m.id,
    name: m.name,
    type: m.type,
    is_active: m.is_active,
    tab_created: m.tab_created,
    version: m.version,
    capabilities: m.capabilities,
  }));
  res.json(list);
});

// ── GET /api/models/:id ───────────────────────────────────────────────────────

router.get('/:id', requireAuth, (req, res) => {
  const model = models.get(req.params.id);
  if (!model) return notFound(res, req.params.id);

  res.json({
    id: model.id,
    name: model.name,
    type: model.type,
    script_path: model.script_path,
    model_path: model.model_path,
    version: model.version,
    capabilities: model.capabilities,
    assigned_cameras: model.assigned_cameras,
    is_active: model.is_active,
  });
});

// ── POST /api/models/upload ───────────────────────────────────────────────────
// Must be before /:id to avoid route clash

router.post('/upload', requireAuth, requireRole('admin'), (req, res) => {
  // In production: use multer to receive multipart/form-data
  // const upload = multer({ dest: 'uploads/' });
  // The .atomomodel package is a zip containing:
  //   model.nb / model.onnx, labels.yaml, config.json, pre/post-processing config

  // Simulated upload handling:
  const { name, format = 'onnx', version = '1.0.0' } = req.body || {};
  if (!name) return res.status(400).json({ error: 'name is required' });

  const id = 'mdl_' + uuidv4().slice(0, 6);
  const model = {
    id,
    name,
    type: 'custom',
    is_active: true,
    tab_created: true,
    version,
    script_path: `detectors/${name.toLowerCase().replace(/\s+/g, '_')}.py`,
    model_path: `models/${name.toLowerCase().replace(/\s+/g, '_')}/model.${format}`,
    capabilities: ['custom_detection'],
    assigned_cameras: [],
    format,
    test_passed: true,
    created_at: new Date().toISOString(),
  };

  models.set(id, model);

  res.status(201).json({
    id,
    name: model.name,
    tab_created: true,
    test_passed: true,
    format,
  });
});

// ── POST /api/models/:id/validate ─────────────────────────────────────────────

router.post('/:id/validate', requireAuth, (req, res) => {
  const model = models.get(req.params.id);
  if (!model) return notFound(res, req.params.id);

  // In production: actually inspect the model file for shape compatibility
  res.json({
    valid: true,
    format: model.format || 'nb',
    input_shape: [1, 3, 640, 640],
    output_shape: [1, 84, 8400],
  });
});

// ── POST /api/models/:id/test ─────────────────────────────────────────────────

router.post('/:id/test', requireAuth, (req, res) => {
  const model = models.get(req.params.id);
  if (!model) return notFound(res, req.params.id);

  const { image_b64 } = req.body || {};
  if (!image_b64) return res.status(400).json({ error: 'image_b64 required' });

  // In production: run actual inference on the provided image
  const mockDetections = {
    mdl_person: [{ class: 'person', score: 0.87, box: [0.1, 0.15, 0.4, 0.85] }],
    mdl_face:   [{ class: 'face', score: 0.92, box: [0.3, 0.1, 0.6, 0.5] }],
    mdl_fire:   [{ class: 'fire', score: 0.76, box: [0.5, 0.3, 0.9, 0.7] }],
    mdl_ppe:    [
      { class: 'helmet', score: 0.91, box: [0.2, 0.05, 0.45, 0.3] },
      { class: 'vest',   score: 0.83, box: [0.15, 0.3, 0.55, 0.8] },
    ],
  };

  res.json({
    inference_ms: parseFloat((30 + Math.random() * 25).toFixed(1)),
    detections: mockDetections[req.params.id] || [{ class: 'object', score: 0.78, box: [0.1, 0.1, 0.6, 0.6] }],
  });
});

// ── DELETE /api/models/:id ────────────────────────────────────────────────────

router.delete('/:id', requireAuth, requireRole('admin'), (req, res) => {
  const model = models.get(req.params.id);
  if (!model) return notFound(res, req.params.id);

  if (model.type === 'builtin') {
    return res.status(403).json({ error: 'Built-in models cannot be deleted' });
  }

  // Stop any running workers for this model
  const { workers } = require('../store');
  const { stopWorker } = require('../services/worker');
  for (const [key] of workers.entries()) {
    if (key.endsWith('::' + req.params.id)) {
      const camId = key.split('::')[0];
      try { stopWorker(camId, req.params.id); } catch {}
    }
  }

  models.delete(req.params.id);
  res.json({ ok: true });
});

// ── GET /api/models/:id/assignments ──────────────────────────────────────────

router.get('/:id/assignments', requireAuth, (req, res) => {
  const model = models.get(req.params.id);
  if (!model) return notFound(res, req.params.id);

  const { workers } = require('../store');

  const assignments = model.assigned_cameras.map(camId => {
    const cam = cameras.get(camId);
    const key = `${camId}::${req.params.id}`;
    const worker = workers.get(key);
    return {
      camera_id: camId,
      camera_name: cam?.name || 'Unknown',
      confidence: worker?.config?.confidence || 0.45,
      fps: worker?.fps || 0,
      status: worker ? 'running' : 'idle',
      enabled_capabilities: worker?.enabled_capabilities || model.capabilities,
    };
  });

  res.json(assignments);
});

module.exports = router;
