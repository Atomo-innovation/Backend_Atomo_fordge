/**
 * clusterStore.js
 *
 * Groups embeddings of *unrecognized* faces ("is_known: false") seen on live
 * camera streams into clusters, so the same recurring stranger is bucketed
 * together instead of appearing as N disconnected detections. An operator
 * can later inspect a cluster's crops and "label" it — that turns the
 * cluster into a real enrolled Person (see routes/face.js), after which the
 * face becomes recognizable (is_known: true) on future detections.
 *
 * Matching uses cosine similarity against each cluster's running centroid —
 * same distance metric (dis_type 0) the SFace matcher in face_worker.py
 * uses for person recognition, so the clustering threshold is directly
 * comparable to the recognition threshold.
 */
const { v4: uuidv4 } = require('uuid');
const log = require('../utils/logger').child('cluster');

const DEFAULT_CLUSTER_MATCH_THRESHOLD = 0.60; // cosine score to join an existing cluster
const MAX_EMBEDDINGS_PER_CLUSTER = 30;         // cap memory — keep most recent samples
const MAX_CROPS_PER_CLUSTER      = 10;
const MAX_CLUSTERS               = 500;        // safety cap on long-running streams

function parseThreshold(v, fallback) {
  const n = parseFloat(v);
  return Number.isFinite(n) && n > 0 && n < 1 ? n : fallback;
}

// Configurable via env var so it can be tuned per-deployment without a code
// change; falls back to the same default the recognition matcher uses.
let clusterMatchThreshold = parseThreshold(process.env.CLUSTER_MATCH_THRESHOLD, DEFAULT_CLUSTER_MATCH_THRESHOLD);

function getThreshold() { return clusterMatchThreshold; }
function setThreshold(v) {
  const n = parseThreshold(v, null);
  if (n === null) throw new Error('threshold must be a number between 0 and 1 (exclusive)');
  clusterMatchThreshold = n;
  log.info({ threshold: clusterMatchThreshold }, 'cluster threshold updated');
  return clusterMatchThreshold;
}

const clusters = new Map();

function cosineSim(a, b) {
  let dot = 0, na = 0, nb = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    dot += a[i] * b[i];
    na  += a[i] * a[i];
    nb  += b[i] * b[i];
  }
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

function centroid(embeddings) {
  const dim = embeddings[0].length;
  const c = new Array(dim).fill(0);
  for (const e of embeddings) for (let i = 0; i < dim; i++) c[i] += e[i];
  for (let i = 0; i < dim; i++) c[i] /= embeddings.length;
  return c;
}

function findBestCluster(embedding) {
  let best = null, bestScore = -1;
  for (const cluster of clusters.values()) {
    const score = cosineSim(embedding, cluster.centroid);
    if (score > bestScore) { bestScore = score; best = cluster; }
  }
  return { cluster: best, score: bestScore };
}

/** Add one unrecognized-face detection to the cluster pool. */
function ingestUnknownFace({ embedding, crop_filename, camera_id, gender }) {
  if (!embedding || !embedding.length) return null;

  const { cluster, score } = findBestCluster(embedding);
  const now = new Date().toISOString();

  if (cluster && score >= clusterMatchThreshold) {
    cluster.embeddings.push(embedding);
    if (cluster.embeddings.length > MAX_EMBEDDINGS_PER_CLUSTER) cluster.embeddings.shift();
    cluster.centroid = centroid(cluster.embeddings);

    if (crop_filename) {
      cluster.crop_filenames.push(crop_filename);
      if (cluster.crop_filenames.length > MAX_CROPS_PER_CLUSTER) cluster.crop_filenames.shift();
    }
    if (camera_id) cluster.camera_ids.add(camera_id);
    cluster.seen_count += 1;
    cluster.last_seen = now;
    if (gender) cluster.last_gender = gender;
    log.debug({ cluster_id: cluster.cluster_id, score }, 'added embedding to existing cluster');
    return cluster;
  }

  // No close-enough cluster — start a new one (unless we've hit the safety cap).
  if (clusters.size >= MAX_CLUSTERS) return null;

  const id = 'cl_' + uuidv4().slice(0, 8);
  const newCluster = {
    cluster_id:     id,
    embeddings:     [embedding],
    centroid:       embedding.slice(),
    crop_filenames: crop_filename ? [crop_filename] : [],
    camera_ids:     new Set(camera_id ? [camera_id] : []),
    seen_count:     1,
    first_seen:     now,
    last_seen:      now,
    last_gender:    gender || null,
  };
  clusters.set(id, newCluster);
  log.info({ cluster_id: id }, 'created new cluster');
  return newCluster;
}

/** Hook this up to faceWorkerBridge's "stream_match" event. */
function ingestStreamMatch(msg) {
  const { camera_id, faces = [] } = msg || {};
  for (const f of faces) {
    if (f.is_known || !f.embedding) continue; // only recurring strangers get clustered
    ingestUnknownFace({
      embedding:     f.embedding,
      crop_filename: f.crop_filename,
      camera_id,
      gender:        f.gender,
    });
  }
}

function listClusters() {
  return Array.from(clusters.values())
    .sort((a, b) => new Date(b.last_seen) - new Date(a.last_seen))
    .map(c => ({
      cluster_id:          c.cluster_id,
      seen_count:          c.seen_count,
      embedding_count:     c.embeddings.length,
      camera_ids:          Array.from(c.camera_ids),
      representative_crop: c.crop_filenames[c.crop_filenames.length - 1] || null,
      crop_filenames:      c.crop_filenames,
      last_gender:         c.last_gender,
      first_seen:          c.first_seen,
      last_seen:           c.last_seen,
    }));
}

function getCluster(id) { return clusters.get(id) || null; }

function deleteCluster(id) {
  if (!clusters.has(id)) throw new Error(`Cluster ${id} not found`);
  clusters.delete(id);
  log.info({ cluster_id: id }, 'deleted cluster');
}

module.exports = {
  ingestStreamMatch,
  ingestUnknownFace,
  listClusters,
  getCluster,
  deleteCluster,
  getThreshold,
  setThreshold,
};
