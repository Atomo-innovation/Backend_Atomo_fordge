/**
 * customModels.js
 *
 * Handles uploading and registering custom NPU model packages.
 *
 * Expected ZIP structure (flat or in one subfolder):
 *   animal_pack.zip
 *   ├── animal.nb              ← compiled NPU model
 *   ├── libnn_animal.so        ← NPU runtime library for this model
 *   └── data.yaml              ← class names + metadata
 *
 * data.yaml format (standard YOLO-style):
 *   names:
 *     0: cow
 *     1: goat
 *     2: dog
 *   # or as a list:
 *   # names: [cow, goat, dog]
 *   nc: 3                       # optional, number of classes
 *   input_size: 640             # optional, defaults to 640
 *   conf_threshold: 0.45        # optional default confidence
 *   nms_threshold: 0.56         # optional default NMS
 *
 * On upload:
 *   1. Save zip to uploads/model_packages/
 *   2. Extract to models/custom/<model_id>/
 *   3. Locate exactly one .nb and one .so file (any names)
 *   4. Parse data.yaml for class names
 *   5. Register the model in the store with capabilities = class names
 *      (each class becomes a checkbox the user can toggle on/off)
 *   6. Move .so into lib/custom/<model_id>/ for clean separation
 */

const fs   = require('fs');
const path = require('path');
const yaml = require('js-yaml');
const AdmZip = require('adm-zip');
const { v4: uuidv4 } = require('uuid');
const { models } = require('../store');

const PROJECT_ROOT  = path.join(__dirname, '../..');
const UPLOAD_DIR    = path.join(PROJECT_ROOT, 'uploads', 'model_packages');
const MODELS_DIR    = path.join(PROJECT_ROOT, 'models', 'custom');
const LIB_DIR       = path.join(PROJECT_ROOT, 'lib', 'custom');
const DETECTORS_DIR = path.join(PROJECT_ROOT, 'detectors');

// Generic detector script used by ALL custom models (you write this once)
const GENERIC_DETECTOR = 'generic_detector.py';

[UPLOAD_DIR, MODELS_DIR, LIB_DIR].forEach(d => fs.mkdirSync(d, { recursive: true }));

/**
 * Recursively find all files matching a predicate within a directory.
 */
function findFiles(dir, predicate, found = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      findFiles(full, predicate, found);
    } else if (predicate(entry.name)) {
      found.push(full);
    }
  }
  return found;
}

/**
 * Parse data.yaml into a normalised { names: string[], ...meta } object.
 * Supports both dict form ({0: 'cow', 1: 'goat'}) and list form (['cow','goat']).
 */
function parseDataYaml(yamlPath) {
  const raw = fs.readFileSync(yamlPath, 'utf8');
  const doc = yaml.load(raw) || {};

  let names = [];
  if (Array.isArray(doc.names)) {
    names = doc.names.map(String);
  } else if (doc.names && typeof doc.names === 'object') {
    // dict form: {0: 'cow', 1: 'goat'} — sort by numeric key
    names = Object.entries(doc.names)
      .sort((a, b) => Number(a[0]) - Number(b[0]))
      .map(([, v]) => String(v));
  }

  if (names.length === 0) {
    throw new Error('data.yaml has no "names" field (expected list or dict of class names)');
  }

  return {
    names,
    nc:              doc.nc || names.length,
    input_size:      doc.input_size || doc.imgsz || 640,
    conf_threshold:  doc.conf_threshold ?? doc.conf ?? 0.45,
    nms_threshold:   doc.nms_threshold ?? doc.nms ?? 0.56,
    raw:             doc,
  };
}

/**
 * Process an uploaded zip file:
 *  - extract
 *  - validate contents (.nb, .so, data.yaml)
 *  - register model in store
 *
 * @param {string} zipPath  path to the uploaded .zip
 * @param {object} opts     { name, description }
 * @returns the registered model object
 */
function registerModelFromZip(zipPath, opts = {}) {
  const modelId   = 'mdl_custom_' + uuidv4().slice(0, 8);
  const extractDir = path.join(MODELS_DIR, modelId);
  fs.mkdirSync(extractDir, { recursive: true });

  // ── 1. Extract ────────────────────────────────────────────────────────────
  let zip;
  try {
    zip = new AdmZip(zipPath);
    zip.extractAllTo(extractDir, true);
  } catch (err) {
    cleanup(extractDir);
    throw new Error(`Failed to extract zip: ${err.message}`);
  }

  // ── 2. Locate required files (search recursively — handles nested folders) ──
  const nbFiles   = findFiles(extractDir, n => n.toLowerCase().endsWith('.nb'));
  const soFiles   = findFiles(extractDir, n => n.toLowerCase().endsWith('.so'));
  const yamlFiles = findFiles(extractDir, n => /^data\.ya?ml$/i.test(n));

  const errors = [];
  if (nbFiles.length === 0)   errors.push('No .nb model file found');
  if (nbFiles.length > 1)     errors.push(`Multiple .nb files found: ${nbFiles.map(f => path.basename(f)).join(', ')}`);
  if (soFiles.length === 0)   errors.push('No .so library file found (expected libnn_*.so)');
  if (soFiles.length > 1)     errors.push(`Multiple .so files found: ${soFiles.map(f => path.basename(f)).join(', ')}`);
  if (yamlFiles.length === 0) errors.push('No data.yaml found');
  if (yamlFiles.length > 1)   errors.push(`Multiple data.yaml files found`);

  if (errors.length > 0) {
    cleanup(extractDir);
    throw new Error(`Invalid model package: ${errors.join('; ')}`);
  }

  const nbFile   = nbFiles[0];
  const soFile   = soFiles[0];
  const yamlFile = yamlFiles[0];

  // ── 3. Parse data.yaml ───────────────────────────────────────────────────
  let meta;
  try {
    meta = parseDataYaml(yamlFile);
  } catch (err) {
    cleanup(extractDir);
    throw new Error(`Invalid data.yaml: ${err.message}`);
  }

  // ── 4. Move .so into lib/custom/<model_id>/ for clean separation ─────────
  const libDestDir = path.join(LIB_DIR, modelId);
  fs.mkdirSync(libDestDir, { recursive: true });
  const soDest = path.join(libDestDir, path.basename(soFile));
  fs.renameSync(soFile, soDest);

  // ── 5. Check generic detector script exists ───────────────────────────────
  const detectorScriptPath = path.join(DETECTORS_DIR, GENERIC_DETECTOR);
  if (!fs.existsSync(detectorScriptPath)) {
    console.warn(`[customModels] WARNING: ${GENERIC_DETECTOR} not found in detectors/. ` +
      `Worker will fail to start until you add it.`);
  }

  // ── 6. Register in store ──────────────────────────────────────────────────
  const name = opts.name || path.basename(zipPath, '.zip');

  const model = {
    id:                modelId,
    name,
    description:       opts.description || `Custom model: ${name}`,
    type:              'custom',
    is_active:         true,
    tab_created:       true,
    version:           '1.0.0',

    // Paths — all absolute so worker.js can use them directly
    script_path:       path.join(DETECTORS_DIR, GENERIC_DETECTOR),
    model_path:        nbFile,
    library_path:      soDest,
    data_yaml_path:    yamlFile,

    // Each detected class becomes a capability checkbox the user can toggle.
    // e.g. ["cow", "goat", "dog"] → user can enable detection for just "cow"
    capabilities:      meta.names,
    class_names:       meta.names,

    // Defaults from data.yaml, can be overridden per-worker at start time
    default_conf:      meta.conf_threshold,
    default_nms:       meta.nms_threshold,
    input_size:        meta.input_size,

    assigned_cameras:  [],
    format:            'nb',
    test_passed:       null,        // set true after /validate or /test
    created_at:        new Date().toISOString(),
    extract_dir:       extractDir,
  };

  models.set(modelId, model);

  return model;
}

/**
 * Remove a custom model: delete files, remove from store.
 */
function deleteCustomModel(modelId) {
  const model = models.get(modelId);
  if (!model) throw new Error(`Model ${modelId} not found`);
  if (model.type !== 'custom') throw new Error('Only custom models can be deleted');

  // Remove extracted model directory
  if (model.extract_dir && fs.existsSync(model.extract_dir)) {
    fs.rmSync(model.extract_dir, { recursive: true, force: true });
  }
  // Remove library directory
  const libDir = path.dirname(model.library_path || '');
  if (libDir.includes(LIB_DIR) && fs.existsSync(libDir)) {
    fs.rmSync(libDir, { recursive: true, force: true });
  }

  models.delete(modelId);
}

function cleanup(dir) {
  try { fs.rmSync(dir, { recursive: true, force: true }); } catch {}
}

module.exports = {
  registerModelFromZip,
  deleteCustomModel,
  parseDataYaml,
  UPLOAD_DIR,
};
