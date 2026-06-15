/**
 * In-memory store — replace with SQLite/PostgreSQL in production.
 * Persists nothing on restart; used here for testability without a DB.
 */

const { v4: uuidv4 } = require('uuid');

// ── Cameras ──────────────────────────────────────────────────────────────────
const cameras = new Map();

// ── Models ───────────────────────────────────────────────────────────────────
// Built-in models are pre-seeded on start
const models = new Map([
  ['mdl_person', {
    id: 'mdl_person',
    name: 'Person Detection',
    type: 'builtin',
    is_active: true,
    tab_created: true,
    version: '1.0.0',
    script_path: 'detectors/person.py',
    model_path: 'models/person/yolo26s.nb',
    library_path: 'lib/libnn_yolo26s.so',
    assigned_cameras: [],
    capabilities: ['person_detection'],        // sub-features
  }],
  ['mdl_face', {
    id: 'mdl_face',
    name: 'Face Analysis',
    type: 'builtin',
    is_active: true,
    tab_created: true,
    version: '1.0.0',
    script_path: 'detectors/face.py',
    model_path: 'models/face/face.nb',
    library_path: 'lib/libnn_face.so',
    assigned_cameras: [],
    /**
     * Sub-capabilities users can toggle independently:
     *  face_detection  — just detect & localise faces
     *  gender_classification — add M/F label on top of detection
     *  face_recognition — match against enrolled embeddings
     */
    capabilities: ['face_detection', 'gender_classification', 'face_recognition'],
  }],
  ['mdl_fire', {
    id: 'mdl_fire',
    name: 'Fire & Smoke Detection',
    type: 'builtin',
    is_active: true,
    tab_created: true,
    version: '1.0.0',
    script_path: 'detectors/fire_smoke.py',
    model_path: 'models/fire/fire.nb',
    library_path: 'lib/libnn_fire.so',
    assigned_cameras: [],
    capabilities: ['fire_detection', 'smoke_detection'],
  }],
  ['mdl_ppe', {
    id: 'mdl_ppe',
    name: 'Safety PPE Detection',
    type: 'builtin',
    is_active: true,
    tab_created: true,
    version: '1.0.0',
    script_path: 'detectors/ppe.py',
    model_path: 'models/ppe/ppe.nb',
    library_path: 'lib/libnn_ppe.so',
    assigned_cameras: [],
    capabilities: ['helmet_detection', 'vest_detection', 'gloves_detection', 'no_ppe_alert'],
  }],
]);

// ── Workers (inference processes) ────────────────────────────────────────────
// key: `${camera_id}::${model_id}` → worker descriptor
const workers = new Map();

// ── Camera logs (ring buffer per camera) ─────────────────────────────────────
const cameraLogs = new Map();
const LOG_RING = 200;

function pushLog(cameraId, event) {
  if (!cameraLogs.has(cameraId)) cameraLogs.set(cameraId, []);
  const ring = cameraLogs.get(cameraId);
  ring.push({ ...event, timestamp: new Date().toISOString() });
  if (ring.length > LOG_RING) ring.shift();
}

// ── Users (for JWT auth demo) ─────────────────────────────────────────────────
const users = new Map([
  ['admin', { id: 'usr_admin', username: 'admin', password: 'admin123', role: 'admin' }],
  ['viewer', { id: 'usr_viewer', username: 'viewer', password: 'viewer123', role: 'viewer' }],
]);

module.exports = { cameras, models, workers, cameraLogs, pushLog, users, uuidv4 };
