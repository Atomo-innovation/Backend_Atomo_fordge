const axios = require('axios');

const MTX_API = 'http://localhost:9997/v3';

async function validateStream(url, credentials = {}) {
  // Mock validation for now
  return {
    reachable: true,
    codec: 'H264',
    resolution: '1920x1080',
    fps: 25
  };
}

async function addPath(pathName, sourceUrl, credentials = {}) {
  // In real: call MediaMTX API to create path
  return {
    whepUrl: `http://localhost:8889/${pathName}/whep`,
    localRtsp: `rtsp://localhost:8554/${pathName}`
  };
}

async function removePath(pathName) {
  // mock
  console.log(`[MediaMTX] Removed path: ${pathName}`);
}

async function patchPath(pathName, sourceUrl, credentials = {}) {
  console.log(`[MediaMTX] Updated path: ${pathName}`);
}

module.exports = {
  validateStream,
  addPath,
  removePath,
  patchPath,
  whepUrl: (id) => `http://localhost:8889/${id}/whep`,
  localRtsp: (id) => `rtsp://localhost:8554/${id}`
};
