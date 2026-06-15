/**
 * mediamtx.js — MediaMTX v3 API wrapper
 *
 * Correct endpoint for adding a pull path:
 *   POST /v3/config/paths/add/{name}
 *   Body: { "source": "rtsp://...", "sourceOnDemand": false }
 *
 * NOT /v3/paths/{name}/source  ← that doesn't exist in v3
 */

const axios = require('axios');

const MTX_API = process.env.MEDIAMTX_API_URL || 'http://127.0.0.1:9997';
const RTSP_PORT = process.env.MEDIAMTX_RTSP_PORT || 8554;
const WHEP_PORT = process.env.MEDIAMTX_WHEP_PORT || 8889;

function whepUrl(pathName) {
  return `http://localhost:${WHEP_PORT}/${pathName}/whep`;
}

function localRtsp(pathName) {
  return `rtsp://localhost:${RTSP_PORT}/${pathName}`;
}

async function validateStream(url, credentials = {}) {
  // In production: run ffprobe here
  // ffprobe -v error -rtsp_transport tcp -show_entries stream=codec_name,r_frame_rate,width,height "<url>"
  return { reachable: true, codec: 'H264', resolution: '1920x1080', fps: 25 };
}

/**
 * Register a new path in MediaMTX so it pulls from sourceUrl.
 *
 * MediaMTX v3 correct endpoint:
 *   POST /v3/config/paths/add/{name}
 */
async function addPath(pathName, sourceUrl, credentials = {}) {
  // Embed credentials into the URL if provided
  let src = sourceUrl;
  if (credentials.username && credentials.password) {
    try {
      const u = new URL(sourceUrl);
      u.username = encodeURIComponent(credentials.username);
      u.password = encodeURIComponent(credentials.password);
      src = u.toString();
    } catch {
      // not a parseable URL — pass as-is, MediaMTX handles it
    }
  }

  const body = {
    source: src,
    sourceOnDemand: false,   // pull immediately, don't wait for a reader
    record: false,
  };

  try {
    await axios.post(`${MTX_API}/v3/config/paths/add/${pathName}`, body, {
      timeout: 5000,
      headers: { 'Content-Type': 'application/json' },
    });
    console.log(`[MediaMTX] Path registered: ${pathName} → ${sourceUrl}`);
  } catch (err) {
    const status = err.response?.status;
    const detail = err.response?.data?.error || err.message;

    if (status === 400 && detail?.includes('already exists')) {
      // Path already registered — patch it instead
      console.log(`[MediaMTX] Path ${pathName} already exists, patching...`);
      await patchPath(pathName, sourceUrl, credentials);
    } else {
      // Log but don't crash — MediaMTX might not be running yet during dev
      console.error(`[MediaMTX] addPath failed (${status}): ${detail}`);
    }
  }

  return {
    whepUrl: whepUrl(pathName),
    localRtsp: localRtsp(pathName),
  };
}

/**
 * Update an existing path (e.g. new source URL).
 *
 * MediaMTX v3: PATCH /v3/config/paths/patch/{name}
 */
async function patchPath(pathName, sourceUrl, credentials = {}) {
  let src = sourceUrl;
  if (credentials.username && credentials.password) {
    try {
      const u = new URL(sourceUrl);
      u.username = encodeURIComponent(credentials.username);
      u.password = encodeURIComponent(credentials.password);
      src = u.toString();
    } catch {}
  }

  try {
    await axios.patch(`${MTX_API}/v3/config/paths/patch/${pathName}`,
      { source: src },
      { timeout: 5000, headers: { 'Content-Type': 'application/json' } }
    );
    console.log(`[MediaMTX] Path patched: ${pathName}`);
  } catch (err) {
    console.error(`[MediaMTX] patchPath failed: ${err.response?.data?.error || err.message}`);
  }

  return { whepUrl: whepUrl(pathName), localRtsp: localRtsp(pathName) };
}

/**
 * Remove a path.
 *
 * MediaMTX v3: DELETE /v3/config/paths/delete/{name}
 */
async function removePath(pathName) {
  try {
    await axios.delete(`${MTX_API}/v3/config/paths/delete/${pathName}`, { timeout: 5000 });
    console.log(`[MediaMTX] Path removed: ${pathName}`);
  } catch (err) {
    if (err.response?.status !== 404) {
      console.error(`[MediaMTX] removePath failed: ${err.message}`);
    }
  }
}

/**
 * List all active paths (for health checks).
 */
async function listPaths() {
  try {
    const { data } = await axios.get(`${MTX_API}/v3/paths/list`, { timeout: 3000 });
    return data.items || [];
  } catch {
    return [];
  }
}

module.exports = {
  validateStream,
  addPath,
  patchPath,
  removePath,
  listPaths,
  whepUrl,
  localRtsp,
};
