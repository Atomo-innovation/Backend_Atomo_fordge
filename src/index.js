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
app.use('/api/system',  require('./routes/system'));

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
      'POST /api/face/stream/start':               'Start live face stream (capabilities checkbox; accepts optional inline line_* fields)',
      'GET  /api/face/stream/line-config/:cameraId': 'Get the saved tripwire line config for a camera',
      'PUT  /api/face/stream/line-config/:cameraId': "Save/update a camera's tripwire line (restarts stream live if running)",
      'DELETE /api/face/stream/line-config/:cameraId': "Clear a camera's saved line config",
      'POST /api/face/stream/stop':                'Stop live face stream',
      'GET  /api/face/stream/result/:cameraId':    'Poll latest stream result',
      'POST /api/face/analyze':                    'One-shot image analysis',
      'GET  /api/face/clusters':                   'List recurring unlabeled-face clusters (from streams)',
      'GET  /api/face/clusters/config/threshold':   'Get the cosine cluster-join threshold',
      'PUT  /api/face/clusters/config/threshold':   'Set the cosine cluster-join threshold',
      'GET  /api/face/clusters/:id':                'Get one cluster (crops, seen count, cameras)',
      'DELETE /api/face/clusters/:id':              'Discard a cluster',
      'POST /api/face/clusters/:id/label':          'Label a cluster -> creates/updates a Person so the face becomes recognizable',
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
const { startPoller } = require('./services/systemStore');
startPoller();   // collect every 5 s, keep 60 min of history
const PORT = process.env.PORT || 3001;
server.listen(PORT, () => {
  console.log(`\n  Vision Backend running on http://localhost:${PORT}`);
  console.log(` API overview:  http://localhost:${PORT}/api`);
  console.log(` Health check:  http://localhost:${PORT}/health`);
  console.log(` WebSocket:     ws://localhost:${PORT}/ws?camera=<id>&model=<id>`);
  console.log('\nDefault credentials:  admin/admin123   viewer/viewer123');
});

module.exports = { app, server };
