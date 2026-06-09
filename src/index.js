/**
 * Vision Backend — Express server
 *
 * Architecture overview:
 * ┌─────────────┐      RTSP      ┌──────────────┐
 * │  IP Camera  │ ─────────────► │   MediaMTX   │ ──► WHEP (browser playback)
 * └─────────────┘                └──────────────┘
 *                                       │ rtsp://localhost:8554/<cam_id>
 *                                       ▼
 *                               ┌──────────────────┐
 *                               │  Python detector  │  (person.py / face.py / ...)
 *                               │  (NPU inference)  │
 *                               └──────────────────┘
 *                                       │ JSON stdout / person_live.json
 *                                       ▼
 *                               ┌──────────────────┐
 *                               │  This Node server │ ◄── REST API clients
 *                               └──────────────────┘
 *
 * Capability-checkbox model:
 *   Each model (face, ppe, fire, person) exposes a `capabilities` array.
 *   POST /api/detect/start accepts a `capabilities` array — the subset
 *   the user checked. The worker is spawned with only those flags active.
 *
 *   Example:
 *     POST /api/detect/start
 *     { "model_id": "mdl_face",
 *       "capabilities": ["face_detection", "gender_classification"] }
 *   → spawns:  python3 face.py --enable-face-detection --enable-gender-classification
 */

require('dotenv').config();

const express = require('express');
const cors    = require('cors');
const morgan  = require('morgan');
const http    = require('http');
const { WebSocketServer } = require('ws');

const app    = express();
const server = http.createServer(app);

// ── Middleware ────────────────────────────────────────────────────────────────
app.use(cors());
app.use(express.json({ limit: '10mb' }));   // base64 images can be large
app.use(express.urlencoded({ extended: true }));
app.use(morgan('dev'));

// ── Routes ────────────────────────────────────────────────────────────────────
app.use('/api/auth',    require('./routes/auth'));
app.use('/api/cameras', require('./routes/cameras'));
app.use('/api/models',  require('./routes/models'));
app.use('/api/detect',  require('./routes/detect'));

// ── Health check ──────────────────────────────────────────────────────────────
app.get('/health', (req, res) => res.json({ ok: true, ts: new Date().toISOString() }));

// ── API overview ──────────────────────────────────────────────────────────────
app.get('/api', (req, res) => {
  res.json({
    name: 'Vision Backend API',
    version: '1.0.0',
    endpoints: {
      auth: {
        'POST /api/auth/login': 'Get JWT token',
        'GET  /api/auth/me':    'Current user info',
      },
      cameras: {
        'GET    /api/cameras':                     'List all cameras',
        'POST   /api/cameras':                     'Add camera (validates + registers in MediaMTX)',
        'GET    /api/cameras/:id':                 'Camera detail',
        'PUT    /api/cameras/:id':                 'Update camera',
        'DELETE /api/cameras/:id':                 'Remove camera [admin]',
        'POST   /api/cameras/:id/validate':        'Test stream reachability',
        'POST   /api/cameras/:id/restart':         'Restart stream',
        'GET    /api/cameras/:id/health':          'Live metrics',
        'GET    /api/cameras/:id/snapshot':        'JPEG snapshot as base64',
        'GET    /api/cameras/:id/logs':            'Event log',
      },
      models: {
        'GET    /api/models':                      'List all models',
        'GET    /api/models/:id':                  'Model detail',
        'POST   /api/models/upload':               'Upload custom model [admin]',
        'POST   /api/models/:id/validate':         'Re-validate model',
        'POST   /api/models/:id/test':             'Run test inference',
        'DELETE /api/models/:id':                  'Delete custom model [admin]',
        'GET    /api/models/:id/assignments':      'Camera assignments',
      },
      inference: {
        'POST   /api/detect/start':                'Spawn worker (with capability checkboxes)',
        'POST   /api/detect/stop':                 'Kill worker',
        'GET    /api/detect/status':               'Running workers',
        'POST   /api/detect/stop-all':             'Kill all workers [admin]',
        'PUT    /api/detect/config':               'Update conf/fps live',
        'POST   /api/detect/zone':                 'Update detection zone',
        'GET    /api/detect/result/:camId/:mdlId': 'Latest detection result',
        'GET    /api/detect/capabilities/:mdlId':  'Available capability checkboxes',
      },
    },
    models_builtin: ['mdl_person', 'mdl_face', 'mdl_fire', 'mdl_ppe'],
    notes: [
      'Authenticate: POST /api/auth/login → use returned token as "Authorization: Bearer <token>"',
      'Default users: admin/admin123 (admin role), viewer/viewer123 (viewer role)',
      'MediaMTX must be running on localhost:9997 (API) and 8554/8889 (RTSP/WHEP)',
      'Python detectors are mocked — real spawn is commented in src/services/worker.js',
    ],
  });
});

// ── WebSocket — push detection results to browser UI ─────────────────────────
// Clients subscribe to:  ws://localhost:3000/ws?camera=cam_xxx&model=mdl_yyy
// The server pushes the latest result every 500ms.

const wss = new WebSocketServer({ server, path: '/ws' });

wss.on('connection', (ws, req) => {
  const params = new URLSearchParams(req.url.replace('/ws?', ''));
  const cameraId = params.get('camera');
  const modelId  = params.get('model');

  if (!cameraId || !modelId) {
    ws.send(JSON.stringify({ error: 'Provide ?camera=<id>&model=<id> query params' }));
    ws.close();
    return;
  }

  const { getWorkerResult } = require('./services/worker');

  const interval = setInterval(() => {
    if (ws.readyState !== ws.OPEN) { clearInterval(interval); return; }
    const result = getWorkerResult(cameraId, modelId);
    if (result) ws.send(JSON.stringify(result));
  }, 500);

  ws.on('close', () => clearInterval(interval));
  ws.on('error', () => clearInterval(interval));

  ws.send(JSON.stringify({ connected: true, camera: cameraId, model: modelId }));
});

// ── 404 ───────────────────────────────────────────────────────────────────────
app.use((req, res) => res.status(404).json({ error: 'Route not found' }));

// ── Error handler ─────────────────────────────────────────────────────────────
// eslint-disable-next-line no-unused-vars
app.use((err, req, res, _next) => {
  console.error(err);
  res.status(500).json({ error: err.message || 'Internal server error' });
});

// ── Start ─────────────────────────────────────────────────────────────────────
const PORT = process.env.PORT || 3001;
server.listen(PORT, () => {
  console.log(`\n🚀  Vision Backend running on http://localhost:${PORT}`);
  console.log(`📋  API overview:  http://localhost:${PORT}/api`);
  console.log(`❤️   Health check:  http://localhost:${PORT}/health`);
  console.log(`🔌  WebSocket:     ws://localhost:${PORT}/ws?camera=<id>&model=<id>`);
  console.log('\nDefault credentials:');
  console.log('  admin  / admin123   (admin role)');
  console.log('  viewer / viewer123  (viewer role)');
  console.log('\nQuick start:');
  console.log(`  curl -X POST http://localhost:${PORT}/api/auth/login \\`);
  console.log(`    -H "Content-Type: application/json" \\`);
  console.log(`    -d '{"username":"admin","password":"admin123"}'`);
});

module.exports = { app, server };
