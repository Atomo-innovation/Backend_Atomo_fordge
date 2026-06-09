/**
 * test-api.js — self-contained API test runner
 * Run AFTER starting the server:  node src/index.js
 * Then in another terminal:       node test-api.js
 *
 * Tests the full happy path:
 *   1.  Auth: login → get token
 *   2.  Models: list (see built-ins + capabilities)
 *   3.  Capabilities: get checkbox options for face model
 *   4.  Cameras: add → list → get detail → health → logs
 *   5.  Detect: start person worker → status → live result → update config
 *   6.  Detect: start face worker with only 2/3 capabilities checked
 *   7.  Detect: update zone polygon
 *   8.  Detect: stop workers
 *   9.  Camera: snapshot → validate → restart → delete
 *  10.  Auth: viewer can read but cannot delete (403 check)
 */

const http = require('http');

const BASE_HOST = 'localhost';
const BASE_PORT = process.env.PORT || 3000;

// ── HTTP helper ────────────────────────────────────────────────────────────────
function request(method, path, body, token) {
  return new Promise((resolve, reject) => {
    const payload = body ? JSON.stringify(body) : null;
    const opts = {
      hostname: BASE_HOST,
      port: BASE_PORT,
      path,
      method,
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
        ...(payload ? { 'Content-Length': Buffer.byteLength(payload) } : {}),
      },
    };
    const req = http.request(opts, res => {
      let data = '';
      res.on('data', c => (data += c));
      res.on('end', () => {
        try { resolve({ status: res.statusCode, body: JSON.parse(data) }); }
        catch { resolve({ status: res.statusCode, body: data }); }
      });
    });
    req.on('error', reject);
    if (payload) req.write(payload);
    req.end();
  });
}

// ── Test runner ────────────────────────────────────────────────────────────────
let passed = 0, failed = 0;

function assert(label, condition, detail = '') {
  if (condition) {
    console.log(`  ✅  ${label}`);
    passed++;
  } else {
    console.error(`  ❌  ${label}${detail ? ': ' + detail : ''}`);
    failed++;
  }
}

async function run() {
  let adminToken, viewerToken, camId;

  console.log(`\n🧪  Vision Backend API Tests  →  http://${BASE_HOST}:${BASE_PORT}\n`);

  // ── 1. Health ──────────────────────────────────────────────────────────────
  console.log('1. Health check');
  const health = await request('GET', '/health');
  assert('Health 200', health.status === 200);
  assert('Health ok=true', health.body.ok === true);

  // ── 2. Auth ────────────────────────────────────────────────────────────────
  console.log('\n2. Authentication');

  const login = await request('POST', '/api/auth/login', { username: 'admin', password: 'admin123' });
  assert('Admin login 200', login.status === 200, JSON.stringify(login.body));
  assert('Admin token returned', typeof login.body.token === 'string');
  assert('Admin role', login.body.user?.role === 'admin');
  adminToken = login.body.token;

  const viewerLogin = await request('POST', '/api/auth/login', { username: 'viewer', password: 'viewer123' });
  assert('Viewer login 200', viewerLogin.status === 200);
  viewerToken = viewerLogin.body.token;

  const badLogin = await request('POST', '/api/auth/login', { username: 'admin', password: 'wrong' });
  assert('Bad password → 401', badLogin.status === 401);

  const noToken = await request('GET', '/api/cameras');
  assert('No token → 401', noToken.status === 401);

  // ── 3. Models ──────────────────────────────────────────────────────────────
  console.log('\n3. Models');
  const modelList = await request('GET', '/api/models', null, adminToken);
  assert('List models 200', modelList.status === 200);
  assert('4 built-in models', Array.isArray(modelList.body) && modelList.body.length >= 4);

  const personModel = modelList.body.find(m => m.id === 'mdl_person');
  assert('Person model present', !!personModel);
  assert('Person has capabilities array', Array.isArray(personModel?.capabilities));

  const faceModel = await request('GET', '/api/models/mdl_face', null, adminToken);
  assert('Face model detail 200', faceModel.status === 200);
  assert('Face model has 3 capabilities', faceModel.body.capabilities?.length === 3);

  // ── 4. Capabilities (checkbox options) ────────────────────────────────────
  console.log('\n4. Capability checkboxes');
  const faceCaps = await request('GET', '/api/detect/capabilities/mdl_face', null, adminToken);
  assert('Face capabilities 200', faceCaps.status === 200);
  assert('Has face_detection cap', faceCaps.body.capabilities?.includes('face_detection'));
  assert('Has gender_classification cap', faceCaps.body.capabilities?.includes('gender_classification'));
  assert('Has face_recognition cap', faceCaps.body.capabilities?.includes('face_recognition'));
  assert('Each cap has description', Object.keys(faceCaps.body.description || {}).length > 0);

  const ppeCaps = await request('GET', '/api/detect/capabilities/mdl_ppe', null, adminToken);
  assert('PPE has 4 capabilities', ppeCaps.body.capabilities?.length === 4);

  // ── 5. Add camera ──────────────────────────────────────────────────────────
  console.log('\n5. Cameras');
  const addCam = await request('POST', '/api/cameras', {
    name: 'Front Door',
    type: 'rtsp',
    url: 'rtsp://192.168.1.10:554/stream',
    username: 'admin',
    password: 'pass123',
    location: 'Main Lobby',
    zone: 'A',
    floor: 'Ground',
    department: 'Security',
  }, adminToken);
  assert('Add camera 201', addCam.status === 201, JSON.stringify(addCam.body));
  assert('Camera has id', typeof addCam.body.id === 'string');
  assert('Camera has whep_url', typeof addCam.body.whep_url === 'string');
  assert('WHEP URL format', addCam.body.whep_url?.includes('/whep'));
  camId = addCam.body.id;

  const camList = await request('GET', '/api/cameras', null, adminToken);
  assert('List cameras 200', camList.status === 200);
  assert('Camera in list', Array.isArray(camList.body) && camList.body.some(c => c.id === camId));

  const camDetail = await request('GET', `/api/cameras/${camId}`, null, adminToken);
  assert('Camera detail 200', camDetail.status === 200);
  assert('Camera name matches', camDetail.body.name === 'Front Door');
  assert('Has local_rtsp', typeof camDetail.body.local_rtsp === 'string');

  const camHealth = await request('GET', `/api/cameras/${camId}/health`, null, adminToken);
  assert('Camera health 200', camHealth.status === 200);
  assert('Health has fps', typeof camHealth.body.fps === 'number');
  assert('Health has latency_ms', typeof camHealth.body.latency_ms === 'number');

  const camSnap = await request('GET', `/api/cameras/${camId}/snapshot`, null, adminToken);
  assert('Snapshot 200', camSnap.status === 200);
  assert('Snapshot has jpeg_b64', typeof camSnap.body.jpeg_b64 === 'string');

  const camLogs = await request('GET', `/api/cameras/${camId}/logs`, null, adminToken);
  assert('Logs 200', camLogs.status === 200);
  assert('Has camera_added event', Array.isArray(camLogs.body) && camLogs.body.some(l => l.event === 'camera_added'));

  const camValidate = await request('POST', `/api/cameras/${camId}/validate`, null, adminToken);
  assert('Validate 200', camValidate.status === 200);
  assert('Validate reachable=true', camValidate.body.reachable === true);
  assert('Validate has codec', typeof camValidate.body.codec === 'string');

  const updateCam = await request('PUT', `/api/cameras/${camId}`, { name: 'Lobby Camera' }, adminToken);
  assert('Update camera 200', updateCam.status === 200);
  assert('Name updated', updateCam.body.name === 'Lobby Camera');

  // ── 6. Inference — person detection ───────────────────────────────────────
  console.log('\n6. Person detection worker');
  const startPerson = await request('POST', '/api/detect/start', {
    camera_id: camId,
    model_id: 'mdl_person',
    confidence: 0.45,
    fps: 5,
  }, adminToken);
  assert('Start person worker 200', startPerson.status === 200, JSON.stringify(startPerson.body));
  assert('Worker has pid', typeof startPerson.body.worker_pid === 'number');
  assert('Worker status running', startPerson.body.status === 'running');

  // Duplicate start → 409
  const dupStart = await request('POST', '/api/detect/start', {
    camera_id: camId, model_id: 'mdl_person', confidence: 0.45, fps: 5,
  }, adminToken);
  assert('Duplicate start → 409', dupStart.status === 409);

  const detectStatus = await request('GET', '/api/detect/status', null, adminToken);
  assert('Detect status 200', detectStatus.status === 200);
  assert('Worker visible in status', Array.isArray(detectStatus.body) &&
    detectStatus.body.some(w => w.camera_id === camId && w.model_id === 'mdl_person'));

  // Wait for mock result
  await new Promise(r => setTimeout(r, 700));
  const personResult = await request('GET', `/api/detect/result/${camId}/mdl_person`, null, adminToken);
  assert('Person result 200', personResult.status === 200);
  assert('Result has person_count', typeof personResult.body.person_count === 'number');
  assert('Result has enabled_capabilities', Array.isArray(personResult.body.enabled_capabilities));

  // Update config live
  const updateConf = await request('PUT', '/api/detect/config', {
    camera_id: camId, model_id: 'mdl_person', confidence: 0.6, fps: 3,
  }, adminToken);
  assert('Update config 200', updateConf.status === 200);
  assert('Config updated=true', updateConf.body.updated === true);

  // Update zone
  const updateZone = await request('POST', '/api/detect/zone', {
    camera_id: camId,
    model_id: 'mdl_person',
    zone: [[0.1, 0.2], [0.6, 0.2], [0.6, 0.8], [0.1, 0.8]],
  }, adminToken);
  assert('Update zone 200', updateZone.status === 200);
  assert('Zone updated=true', updateZone.body.updated === true);

  // ── 7. Inference — face model, PARTIAL capabilities (checkbox demo) ────────
  console.log('\n7. Face model — partial capability selection (checkbox demo)');
  const startFace = await request('POST', '/api/detect/start', {
    camera_id: camId,
    model_id: 'mdl_face',
    confidence: 0.5,
    fps: 3,
    capabilities: ['face_detection', 'gender_classification'],  // ← face_recognition NOT checked
  }, adminToken);
  assert('Start face worker (2/3 caps) 200', startFace.status === 200, JSON.stringify(startFace.body));

  await new Promise(r => setTimeout(r, 700));
  const faceResult = await request('GET', `/api/detect/result/${camId}/mdl_face`, null, adminToken);
  assert('Face result 200', faceResult.status === 200);
  assert('Face result has face_count', typeof faceResult.body.face_count === 'number');
  assert('Only 2 caps active', faceResult.body.enabled_capabilities?.length === 2);
  assert('gender_classification active', faceResult.body.enabled_capabilities?.includes('gender_classification'));
  assert('face_recognition NOT active', !faceResult.body.enabled_capabilities?.includes('face_recognition'));

  // Invalid capability → 400
  const badCap = await request('POST', '/api/detect/start', {
    camera_id: camId, model_id: 'mdl_fire',
    capabilities: ['invalid_cap'],
  }, adminToken);
  assert('Invalid capability → 400', badCap.status === 400);
  assert('Error lists available caps', Array.isArray(badCap.body.available));

  // ── 8. PPE and Fire models ─────────────────────────────────────────────────
  console.log('\n8. Fire & PPE workers');
  const startFire = await request('POST', '/api/detect/start', {
    camera_id: camId, model_id: 'mdl_fire',
    capabilities: ['fire_detection'],  // smoke NOT checked
  }, adminToken);
  assert('Start fire (fire only) 200', startFire.status === 200);

  const startPpe = await request('POST', '/api/detect/start', {
    camera_id: camId, model_id: 'mdl_ppe',
    capabilities: ['helmet_detection', 'vest_detection', 'no_ppe_alert'],
  }, adminToken);
  assert('Start PPE (3/4 caps) 200', startPpe.status === 200);

  await new Promise(r => setTimeout(r, 700));
  const ppeResult = await request('GET', `/api/detect/result/${camId}/mdl_ppe`, null, adminToken);
  assert('PPE result 200', ppeResult.status === 200);
  assert('PPE has ppe_items', Array.isArray(ppeResult.body.ppe_items));
  assert('PPE has violations array', Array.isArray(ppeResult.body.violations));

  // ── 9. Model assignments ───────────────────────────────────────────────────
  console.log('\n9. Model assignments');
  const assignments = await request('GET', '/api/models/mdl_person/assignments', null, adminToken);
  assert('Assignments 200', assignments.status === 200);
  assert('Camera appears in assignments', Array.isArray(assignments.body) &&
    assignments.body.some(a => a.camera_id === camId));
  assert('Assignment has status=running', assignments.body.some(a => a.status === 'running'));

  // ── 10. Role enforcement ───────────────────────────────────────────────────
  console.log('\n10. Role enforcement');
  const viewerRead = await request('GET', '/api/cameras', null, viewerToken);
  assert('Viewer can read cameras', viewerRead.status === 200);

  const viewerDelete = await request('DELETE', `/api/cameras/${camId}`, null, viewerToken);
  assert('Viewer cannot delete camera → 403', viewerDelete.status === 403);

  const viewerStopAll = await request('POST', '/api/detect/stop-all', {}, viewerToken);
  assert('Viewer cannot stop-all → 403', viewerStopAll.status === 403);

  // ── 11. Stop workers ───────────────────────────────────────────────────────
  console.log('\n11. Stop workers');
  const stopPerson = await request('POST', '/api/detect/stop', {
    camera_id: camId, model_id: 'mdl_person',
  }, adminToken);
  assert('Stop person worker 200', stopPerson.status === 200);
  assert('Status stopped', stopPerson.body.status === 'stopped');

  const stopAll = await request('POST', '/api/detect/stop-all', {}, adminToken);
  assert('Stop-all 200', stopAll.status === 200);
  assert('Stopped ≥ 1 remaining workers', stopAll.body.stopped >= 1);

  // ── 12. Delete camera ──────────────────────────────────────────────────────
  console.log('\n12. Camera deletion');
  const camRestart = await request('POST', `/api/cameras/${camId}/restart`, null, adminToken);
  assert('Restart stream 200', camRestart.status === 200);

  const delCam = await request('DELETE', `/api/cameras/${camId}`, null, adminToken);
  assert('Delete camera 200', delCam.status === 200);
  assert('Delete ok=true', delCam.body.ok === true);

  const afterDel = await request('GET', `/api/cameras/${camId}`, null, adminToken);
  assert('Camera gone after delete → 404', afterDel.status === 404);

  // ── Summary ────────────────────────────────────────────────────────────────
  console.log(`\n${'─'.repeat(50)}`);
  console.log(`  Results:  ✅ ${passed} passed   ❌ ${failed} failed`);
  console.log(`${'─'.repeat(50)}\n`);
  process.exit(failed > 0 ? 1 : 0);
}

run().catch(err => {
  console.error('\nFatal error:', err.message);
  console.error('Make sure the server is running:  node src/index.js');
  process.exit(1);
});
