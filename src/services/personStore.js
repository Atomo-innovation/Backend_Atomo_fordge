const { v4: uuidv4 } = require('uuid');
const persons = new Map();

function createPerson({ name, note = '' }) {
  if (!name) throw new Error('name is required');
  const id = 'p_' + uuidv4().slice(0, 8);
  const p = { person_id: id, name, note, embeddings: [], crop_filenames: [], enrolled_at: new Date().toISOString(), updated_at: new Date().toISOString() };
  persons.set(id, p);
  return p;
}
function getPerson(id)   { return persons.get(id) || null; }
function listPersons()   {
  return Array.from(persons.values()).map(p => ({
    person_id: p.person_id, name: p.name, note: p.note,
    embedding_count: p.embeddings.length, crop_filenames: p.crop_filenames,
    enrolled_at: p.enrolled_at, updated_at: p.updated_at
  }));
}
function updatePerson(id, patch) {
  const p = persons.get(id);
  if (!p) throw new Error(`Person ${id} not found`);
  if (patch.name) p.name = patch.name;
  if (patch.note !== undefined) p.note = patch.note;
  p.updated_at = new Date().toISOString();
  return p;
}
function deletePerson(id) {
  if (!persons.has(id)) throw new Error(`Person ${id} not found`);
  persons.delete(id);
}
function addEmbeddings(id, embeddings, cropFilenames = []) {
  const p = persons.get(id);
  if (!p) throw new Error(`Person ${id} not found`);
  p.embeddings.push(...embeddings);
  p.crop_filenames.push(...cropFilenames);
  p.updated_at = new Date().toISOString();
  return p;
}
function getCandidatesPayload() {
  return Array.from(persons.values())
    .filter(p => p.embeddings.length > 0)
    .map(p => ({ person_id: p.person_id, name: p.name, embeddings: p.embeddings }));
}
module.exports = { createPerson, getPerson, listPersons, updatePerson, deletePerson, addEmbeddings, getCandidatesPayload };
