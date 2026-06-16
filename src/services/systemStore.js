/**
 * systemStore.js
 *
 * In-memory ring buffer for system metrics, consistent with the project's
 * pattern (cameras/workers/cameraLogs are all in-memory Maps).
 *
 * Sampled every SAMPLE_INTERVAL_MS, kept for HISTORY_MINUTES.
 * Imported once — Node's module cache ensures a single shared instance.
 */

const os   = require('os');
const fs   = require('fs');
const si   = require('systeminformation');
const { workers } = require('../store');   // shared worker map (src/store.js)

// ── Tunables ──────────────────────────────────────────────────────────────────
const SAMPLE_INTERVAL_MS = 5_000;          // collect every 5 s
const HISTORY_MINUTES    = 60;             // keep 1 hr of samples
const MAX_SAMPLES        = (HISTORY_MINUTES * 60_000) / SAMPLE_INTERVAL_MS; // 720

// ── Ring buffer ───────────────────────────────────────────────────────────────
/** @type {Array<SystemSample>} */
const history = [];

// ── NPU sysfs helper ─────────────────────────────────────────────────────────
// Amlogic VIM3 / A311D — galcore kernel module exposes load at this path.
// Falls back gracefully on non-VIM3 hardware (dev laptop, etc.).
const NPU_LOAD_PATHS = [
  '/sys/kernel/debug/gc/clk',                       // older galcore
  '/sys/devices/platform/soc/ffe40000.npu/load',    // some BSP builds
  '/sys/class/misc/galcore/device/load_percent',
];

function readNpuLoad() {
  for (const p of NPU_LOAD_PATHS) {
    try {
      const raw = fs.readFileSync(p, 'utf8').trim();
      // The file may contain "load: 42 %" or just "42"
      const m = raw.match(/(\d+(\.\d+)?)/);
      if (m) return Math.min(100, parseFloat(m[1]));
    } catch { /* not available */ }
  }
  return null;   // null = not readable on this device
}

// ── Device info (read once) ───────────────────────────────────────────────────
let _deviceInfo = null;

async function getDeviceInfo() {
  if (_deviceInfo) return _deviceInfo;

  const [cpu, osInfo, system, versions] = await Promise.all([
    si.cpu(),
    si.osInfo(),
    si.system(),
    si.versions(),
  ]);

  // NPU driver version from sysfs / dmesg fallback
  let npu_driver = 'unknown';
  try { npu_driver = fs.readFileSync('/sys/module/galcore/version', 'utf8').trim(); } catch {}

  // License stub — replace with real license check in production
  let license_status = 'unlicensed';
  try {
    const lic = fs.readFileSync('/etc/atomo/license.key', 'utf8').trim();
    license_status = lic.length > 0 ? 'active' : 'missing';
  } catch { license_status = 'not_found'; }

  _deviceInfo = {
    serial:         system.serial || os.hostname(),
    hostname:       os.hostname(),
    platform:       osInfo.platform,
    distro:         osInfo.distro,
    os_version:     osInfo.release,
    kernel:         osInfo.kernel,
    arch:           osInfo.arch,
    cpu_model:      cpu.manufacturer + ' ' + cpu.brand,
    cpu_cores:      os.cpus().length,
    npu_driver,
    node_version:   versions.node || process.version,
    license_status,
  };
  return _deviceInfo;
}

// ── Single sample collection ──────────────────────────────────────────────────
async function collectSample() {
  try {
    const [load, mem, temp] = await Promise.all([
      si.currentLoad(),
      si.mem(),
      si.cpuTemperature(),
    ]);

    // Aggregate live worker metrics from the shared workers Map
    let total_fps = 0, total_inf_ms = 0, worker_count = 0;
    for (const w of workers.values()) {
      total_fps    += w.fps          || 0;
      total_inf_ms += w.inference_ms || 0;
      worker_count++;
    }

    /** @type {SystemSample} */
    const sample = {
      timestamp:    new Date().toISOString(),
      ts:           Date.now(),

      // CPU
      cpu_load:     Math.round(load.currentLoad * 10) / 10,        // %
      cpu_cores:    load.cpus?.map(c => Math.round(c.load * 10) / 10) ?? [],

      // RAM
      ram_total:    mem.total,
      ram_used:     mem.used,
      ram_free:     mem.available,
      ram_pct:      Math.round((mem.used / mem.total) * 1000) / 10, // %

      // Temperature
      cpu_temp:     temp.main ?? temp.max ?? null,   // °C, null if unavailable

      // NPU
      npu_load:     readNpuLoad(),                   // %, null if unavailable

      // Workers
      worker_count,
      total_fps:    Math.round(total_fps * 10) / 10,
      avg_inf_ms:   worker_count > 0
                      ? Math.round((total_inf_ms / worker_count) * 10) / 10
                      : 0,
    };

    history.push(sample);
    if (history.length > MAX_SAMPLES) history.shift();

    return sample;
  } catch (err) {
    console.error('[systemStore] collect error:', err.message);
    return null;
  }
}

// ── Start background poller ───────────────────────────────────────────────────
let _started = false;
function startPoller() {
  if (_started) return;
  _started = true;
  // Collect immediately so first GET /stats has real data
  collectSample();
  setInterval(collectSample, SAMPLE_INTERVAL_MS);
}

// ── Public helpers ────────────────────────────────────────────────────────────

/** Latest sample, or null if none collected yet */
function latest() {
  return history.length > 0 ? history[history.length - 1] : null;
}

/**
 * Samples within the last `minutes` minutes (default 60).
 * Returns them oldest-first (natural time-series order).
 */
function since(minutes = 60) {
  const cutoff = Date.now() - minutes * 60_000;
  return history.filter(s => s.ts >= cutoff);
}

module.exports = { startPoller, latest, since, collectSample, getDeviceInfo, readNpuLoad };
