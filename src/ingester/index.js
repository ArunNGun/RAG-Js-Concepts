const fs = require('fs');
const path = require('path');
const { IndexFlatIP } = require('faiss-node');
const { embed } = require('../embedder');
require('dotenv').config();

const DOCS_PATH = './docs';
const STORE_PATH = './vector-store';
const CHUNK_SIZE = 512;
const CHUNK_OVERLAP = 50;

// --- Chunker ---
function chunkText(text, size, overlap) {
  const words = text.split(/\s+/);
  const chunks = [];

  for (let i = 0; i < words.length; i += size - overlap) {
    const chunk = words.slice(i, i + size).join(' ');
    if (chunk.trim()) chunks.push(chunk);
  }

  return chunks;
}

// --- Load all .md files from docs/ ---
function loadDocs(dir) {
  const files = fs.readdirSync(dir).filter(f => f.endsWith('.md'));
  return files.map(file => ({
    name: file,
    content: fs.readFileSync(path.join(dir, file), 'utf-8')
  }));
}

// --- Main ingest ---
async function ingest() {
  if (!fs.existsSync(DOCS_PATH)) {
    fs.mkdirSync(DOCS_PATH);
    console.log('Created docs/ folder. Add your .md files there and run again.');
    return;
  }

  const docs = loadDocs(DOCS_PATH);
  if (docs.length === 0) {
    console.log('No .md files found in docs/. Add some and run again.');
    return;
  }

  const allChunks = [];   // { text, source }
  const allVectors = [];  // float arrays

  for (const doc of docs) {
    console.log(`Processing: ${doc.name}`);
    const chunks = chunkText(doc.content, CHUNK_SIZE, CHUNK_OVERLAP);

    for (const chunk of chunks) {
      const vector = await embed(chunk);
      allChunks.push({ text: chunk, source: doc.name });
      allVectors.push(vector);
    }
  }

  // Build FAISS index (IndexFlatIP = inner product, works with normalized vectors)
  const dim = allVectors[0].length; // 384
  const index = new IndexFlatIP(dim);

  const flatVectors = allVectors.flat();
  index.add(flatVectors);

  // Save to disk
  if (!fs.existsSync(STORE_PATH)) fs.mkdirSync(STORE_PATH);
  index.write(path.join(STORE_PATH, 'index.faiss'));
  fs.writeFileSync(
    path.join(STORE_PATH, 'chunks.json'),
    JSON.stringify(allChunks, null, 2)
  );

  console.log(`\nDone. Indexed ${allChunks.length} chunks from ${docs.length} files.`);
}

ingest().catch(console.error);