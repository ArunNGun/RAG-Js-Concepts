const express = require('express');
const path = require('path');
const fs = require('fs');
const { IndexFlatIP } = require('faiss-node');
const { embed } = require('./src/embedder');
const OpenAI = require('openai');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3001;
const STORE_PATH = './vector-store';
const TOP_K = 3;

const client = new OpenAI({
  apiKey: process.env.API_KEY,
  baseURL: process.env.BASE_URL,
});

app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// Health / wake-up endpoint
app.get('/api/health', (req, res) => {
  res.json({ status: 'online', ts: Date.now() });
});

// Query endpoint
app.post('/api/query', async (req, res) => {
  const { question } = req.body;
  if (!question || !question.trim()) {
    return res.status(400).json({ error: 'question is required' });
  }

  try {
    // Load vector store
    const index = IndexFlatIP.read(path.join(STORE_PATH, 'index.faiss'));
    const chunks = JSON.parse(
      fs.readFileSync(path.join(STORE_PATH, 'chunks.json'), 'utf-8')
    );

    // Embed question
    const questionVector = await embed(question);

    // Search FAISS
    const result = index.search(questionVector, TOP_K);
    const topChunks = result.labels.map((i) => chunks[i]).filter(Boolean);

    // Build context
    const context = topChunks.map((c) => c.text).join('\n\n---\n\n');

    const systemPrompt = `You are a helpful assistant. Answer the question using ONLY the provided context. If the answer is not in the context, say "I don't have information about that in my knowledge base — please ask questions related to the indexed documents."`;

    // Call LLM
    const response = await client.chat.completions.create({
      model: process.env.MODEL || 'claude-sonnet-4-6',
      messages: [
        { role: 'system', content: systemPrompt },
        {
          role: 'user',
          content: `Context:\n${context}\n\nQuestion: ${question}`,
        },
      ],
      temperature: 0,
    });

    const answer = response.choices[0].message.content;

    res.json({
      answer,
      sources: topChunks.map((c) => ({
        source: c.source,
        snippet: c.text.slice(0, 200) + '...',
      })),
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

app.listen(PORT, () => {
  console.log(`RAG server running on port ${PORT}`);
});
