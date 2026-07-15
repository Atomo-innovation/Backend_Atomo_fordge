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
    this.activeStreams = new Set(); // camera_ids with a running stream thread
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
        this.activeStreams.clear();
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
  startStream(cameraId, cameraName, rtspUrl, candidates = [], threshold = 0.60, disType = 0, cropsDir = CROPS_DIR, lineConfig = {}) {
    const {
      enabled:  line_crossing_enabled = false,
      line_y:      line_y             = 0.6,
      direction:   line_direction     = 'in',
      x_start:     line_x_start       = 0.0,
      x_end:       line_x_end         = 1.0,
    } = lineConfig;
    return this._send('start_stream', {
      camera_id: cameraId, camera_name: cameraName, rtsp_url: rtspUrl,
      candidates, threshold, dis_type: disType, crops_dir: cropsDir,
      line_crossing_enabled, line_y, line_direction, line_x_start, line_x_end,
    }).then(result => { this.activeStreams.add(cameraId); return result; });
  }
  stopStream(cameraId) {
    return this._send('stop_stream', { camera_id: cameraId })
      .then(result => { this.activeStreams.delete(cameraId); return result; });
  }
  isStreamActive(cameraId) { return this.activeStreams.has(cameraId); }
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
