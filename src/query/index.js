const fs = require('fs');
const path = require('path');
const { IndexFlatIP } = require('faiss-node');
const { embed } = require('../embedder');
const OpenAI = require('openai');
require('dotenv').config();

const STORE_PATH = './vector-store';
const TOP_K = 2;

const client = new OpenAI({
  apiKey: process.env.API_KEY,
  baseURL: process.env.BASE_URL,
});

async function query(question) {
  // 1. Load vector store from disk
  const index = IndexFlatIP.read(path.join(STORE_PATH, 'index.faiss'));
  const chunks = JSON.parse(fs.readFileSync(path.join(STORE_PATH, 'chunks.json'), 'utf-8'));

  // 2. Embed the question
  const questionVector = await embed(question);

  // 3. Search FAISS for top-K similar chunks
  const result = index.search(questionVector, TOP_K);
  const topChunks = result.labels.map(i => chunks[i]);

  console.log(`\n--- Retrieved ${TOP_K} chunks ---`);
  topChunks.forEach((c, i) => console.log(`[${i + 1}] from: ${c.source}\n${c.text.slice(0, 120)}...\n`));

  // 4. Build prompt with retrieved context
  const context = topChunks.map(c => c.text).join('\n\n---\n\n');
  const systemPrompt = `You are a helpful assistant. Answer the question using ONLY the provided context. If the answer is not in the context, say "please ask questions related to the context".`;

  // 5. Call LLM
  const response = await client.chat.completions.create({
    model: 'claude-sonnet-4-6',
    messages: [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: `Context:\n${context}\n\nQuestion: ${question}` }
    ],
    temperature: 0,
  });

  console.log('\n--- Answer ---');
  console.log(response.choices[0].message.content);
}

// Get question from command line args
const question = process.argv.slice(2).join(' ');
if (!question) {
  console.log('Usage: node src/query/index.js "your question here"');
  process.exit(1);
}

query(question).catch(console.error);