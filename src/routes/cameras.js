/**
 * Camera routes — 10 endpoints
 *
 * POST   /api/cameras           Add camera → validate stream → register in MediaMTX
 * GET    /api/cameras           List all cameras
 * GET    /api/cameras/:id       Single camera detail
 * PUT    /api/cameras/:id       Update camera
 * DELETE /api/cameras/:id       Remove camera (admin+)
 * POST   /api/cameras/:id/validate   Test stream reachability
 * POST   /api/cameras/:id/restart    Restart MediaMTX stream path
 * GET    /api/cameras/:id/health     Live health metrics
 * GET    /api/cameras/:id/snapshot   Latest JPEG as base64
 * GET    /api/cameras/:id/logs       Stream + reconnect event log
 */

const router = require('express').Router();
const { cameras, uuidv4, pushLog, cameraLogs } = require('../store');
const { requireAuth, requireRole } = require('../middleware/auth');
const mtx = require('../services/mediamtx');

// ── Helpers ──────────────────────────────────────────────────────────────────

function notFound(res, id) {
  return res.status(404).json({ error: `Camera ${id} not found` });
}

function publicCamera(c) {
  // Don't leak raw credentials in list/get responses
  const { password, ...safe } = c;
  return safe;
}

// ── POST /api/cameras ─────────────────────────────────────────────────────────

router.post('/', requireAuth, async (req, res) => {
  const { name, type, url, username, password, location, zone, floor, department } = req.body || {};
  if (!name || !type || !url)
    return res.status(400).json({ error: 'name, type, and url are required' });

  // 1. Validate stream reachability
  let streamInfo;
  try {
    streamInfo = await mtx.validateStream(url, { username, password });
  } catch (err) {
    return res.status(422).json({ error: `Stream validation failed: ${err.message}` });
  }

  if (!streamInfo.reachable) {
    return res.status(422).json({ error: 'Stream is not reachable', details: streamInfo });
  }

  // 2. Create camera record
  const id = 'cam_' + uuidv4().slice(0, 8);
  const camera = {
    id,
    name,
    type,
    url,
    username: username || null,
    password: password || null,   // store securely (vault) in production
    location: location || null,
    zone: zone || null,
    floor: floor || null,
    department: department || null,
    status: 'idle',
    codec: streamInfo.codec,
    resolution: streamInfo.resolution,
    fps: streamInfo.fps,
    assigned_models: [],
    created_at: new Date().toISOString(),
  };

  // 3. Register in MediaMTX
  try {
    const mtxResult = await mtx.addPath(id, url, { username, password });
    camera.whep_url = mtxResult.whepUrl;
    camera.local_rtsp = mtxResult.localRtsp;
    camera.status = 'online';
  } catch (err) {
    // Don't fail the whole request — camera saved but stream may not be live
    camera.whep_url = mtx.whepUrl(id);
    camera.local_rtsp = mtx.localRtsp(id);
    camera.status = 'error';
    camera.mtx_error = err.message;
  }

  cameras.set(id, camera);
  pushLog(id, { event: 'camera_added', url });

  res.status(201).json({
    id,
    whep_url: camera.whep_url,
    status: camera.status,
    codec: camera.codec,
    resolution: camera.resolution,
  });
});

// ── GET /api/cameras ──────────────────────────────────────────────────────────

router.get('/', requireAuth, (req, res) => {
  const list = Array.from(cameras.values()).map(c => ({
    id: c.id,
    name: c.name,
    type: c.type,
    status: c.status,
    fps: c.fps,
    whep_url: c.whep_url,
    assigned_models: c.assigned_models || [],
  }));
  res.json(list);
});

// ── GET /api/cameras/:id ──────────────────────────────────────────────────────

router.get('/:id', requireAuth, (req, res) => {
  const cam = cameras.get(req.params.id);
  if (!cam) return notFound(res, req.params.id);

  res.json({
    id: cam.id,
    name: cam.name,
    type: cam.type,
    url: cam.url,
    location: cam.location,
    zone: cam.zone,
    floor: cam.floor,
    department: cam.department,
    status: cam.status,
    codec: cam.codec,
    resolution: cam.resolution,
    models: cam.assigned_models || [],
    fps: cam.fps || 0,
    latency_ms: cam.latency_ms || 0,
    reconnect_count: cam.reconnect_count || 0,
    whep_url: cam.whep_url,
    local_rtsp: cam.local_rtsp,
    last_frame: cam.last_frame || null,
    created_at: cam.created_at,
  });
});

// ── PUT /api/cameras/:id ──────────────────────────────────────────────────────

router.put('/:id', requireAuth, async (req, res) => {
  const cam = cameras.get(req.params.id);
  if (!cam) return notFound(res, req.params.id);

  const updatable = ['name', 'url', 'username', 'password', 'location', 'zone', 'floor', 'department'];
  updatable.forEach(k => { if (req.body[k] !== undefined) cam[k] = req.body[k]; });

  // If URL / credentials changed — re-register in MediaMTX
  if (req.body.url || req.body.username || req.body.password) {
    try {
      await mtx.patchPath(cam.id, cam.url, { username: cam.username, password: cam.password });
    } catch (err) {
      console.error('MediaMTX patchPath error:', err.message);
    }
    pushLog(cam.id, { event: 'stream_updated', url: cam.url });
  }

  res.json({ id: cam.id, name: cam.name, updated: true });
});

// ── DELETE /api/cameras/:id ───────────────────────────────────────────────────

router.delete('/:id', requireAuth, requireRole('admin'), async (req, res) => {
  const cam = cameras.get(req.params.id);
  if (!cam) return notFound(res, req.params.id);

  // Stop all running workers for this camera
  const { workers } = require('../store');
  const { stopWorker } = require('../services/worker');
  for (const [key] of workers.entries()) {
    if (key.startsWith(req.params.id + '::')) {
      const modelId = key.split('::')[1];
      try { stopWorker(req.params.id, modelId); } catch {}
    }
  }

  // Deregister from MediaMTX
  try { await mtx.removePath(req.params.id); } catch {}

  cameras.delete(req.params.id);
  res.json({ ok: true });
});

// ── POST /api/cameras/:id/validate ───────────────────────────────────────────

router.post('/:id/validate', requireAuth, async (req, res) => {
  const cam = cameras.get(req.params.id);
  if (!cam) return notFound(res, req.params.id);

  try {
    const result = await mtx.validateStream(cam.url, { username: cam.username, password: cam.password });
    res.json(result);
  } catch (err) {
    res.status(422).json({ reachable: false, error: err.message });
  }
});

// ── POST /api/cameras/:id/restart ────────────────────────────────────────────

router.post('/:id/restart', requireAuth, async (req, res) => {
  const cam = cameras.get(req.params.id);
  if (!cam) return notFound(res, req.params.id);

  try {
    // Remove and re-add the path to force MediaMTX to reconnect
    await mtx.removePath(cam.id);
    await new Promise(r => setTimeout(r, 300));
    await mtx.addPath(cam.id, cam.url, { username: cam.username, password: cam.password });
    cam.status = 'online';
    cam.reconnect_count = (cam.reconnect_count || 0) + 1;
    pushLog(cam.id, { event: 'stream_restarted' });
    res.json({ restarted: true });
  } catch (err) {
    res.status(500).json({ restarted: false, error: err.message });
  }
});

// ── GET /api/cameras/:id/health ───────────────────────────────────────────────

router.get('/:id/health', requireAuth, (req, res) => {
  const cam = cameras.get(req.params.id);
  if (!cam) return notFound(res, req.params.id);

  // In production: read live metrics from MediaMTX path info
  res.json({
    fps: cam.fps || parseFloat((10 + Math.random() * 8).toFixed(1)),
    bandwidth_kbps: Math.floor(700 + Math.random() * 500),
    latency_ms: Math.floor(40 + Math.random() * 80),
    drop_rate: parseFloat((Math.random() * 0.03).toFixed(4)),
    last_frame: new Date().toISOString(),
  });
});

// ── GET /api/cameras/:id/snapshot ────────────────────────────────────────────

router.get('/:id/snapshot', requireAuth, (req, res) => {
  const cam = cameras.get(req.params.id);
  if (!cam) return notFound(res, req.params.id);

  // In production: grab a frame via ffmpeg from cam.local_rtsp
  // ffmpeg -rtsp_transport tcp -i <url> -vframes 1 -f image2pipe -vcodec mjpeg pipe:1
  // Then base64-encode the output buffer.
  const placeholder =
    '/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoH' +
    'BwYIDAoMCwsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/wAALCAAB' +
    'AAEBAREAAQAB/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAAAAAAAAAAAAAA' +
    'AAD/2gAIAQEAAD8AVIP/2Q==';

  res.json({ jpeg_b64: placeholder, timestamp: new Date().toISOString() });
});

// ── GET /api/cameras/:id/logs ─────────────────────────────────────────────────

router.get('/:id/logs', requireAuth, (req, res) => {
  const cam = cameras.get(req.params.id);
  if (!cam) return notFound(res, req.params.id);

  const logs = cameraLogs.get(req.params.id) || [];
  res.json(logs);
});

module.exports = router;
