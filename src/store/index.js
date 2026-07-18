const cameras = new Map();
const models = new Map();
const workers = new Map();
const cameraLogs = new Map();

const uuidv4 = () => 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
  const r = Math.random() * 16 | 0;
  const v = c === 'x' ? r : (r & 0x3 | 0x8);
  return v.toString(16);
});

function pushLog(cameraId, entry) {
  if (!cameraLogs.has(cameraId)) cameraLogs.set(cameraId, []);
  cameraLogs.get(cameraId).push({
    timestamp: new Date().toISOString(),
    ...entry
  });
}

const log = require('../utils/logger').child('store');

// Log initialization of builtin models
log.info({ builtin_count: builtinModels.length }, 'initialized builtin models');

// Initialize default models
const builtinModels = [
  { id: 'mdl_person', name: 'Person Detection', type: 'builtin', capabilities: ['person_detection'], is_active: true },
  { id: 'mdl_face', name: 'Face Detection', type: 'builtin', capabilities: ['face_detection', 'gender_classification', 'face_recognition'], is_active: true },
  { id: 'mdl_fire', name: 'Fire/Smoke Detection', type: 'builtin', capabilities: ['fire_detection', 'smoke_detection'], is_active: true },
  { id: 'mdl_ppe', name: 'PPE Detection', type: 'builtin', capabilities: ['helmet_detection', 'vest_detection', 'gloves_detection'], is_active: true },
];

builtinModels.forEach(m => models.set(m.id, m));

module.exports = { cameras, models, workers, cameraLogs, uuidv4, pushLog };
