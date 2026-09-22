# AI_Student_Assistant

A small, working AI assistant built to show how the **five architectural
layers** of an AI application fit together: **Client, Intelligence,
Inference, Knowledge, Tools**. Each layer is its own Python package with a
one-way dependency direction, and every request produces a trace showing
which layers ran and for how long.

Domain: a personal knowledge assistant over a folder of markdown notes.
Stack: Python 3.12, FastAPI, Gemini (`google-genai`), SQLite + numpy, vanilla JS.

## The five layers

```
 ┌────────────────────────────────────────────────────────────────┐
 │ 1. CLIENT   client/index.html + app.js                          │
 │    Renders chat, consumes SSE. Knows nothing about models.     │
 └───────────────▲────────────────────────────────────────────────┘
                 │  POST /chat  →  SSE: status | tool_call | tool_result | token | done
 ┌───────────────┴────────────────────────────────────────────────┐
 │ 2. INTELLIGENCE   app/intelligence/                            │
 │    agent.py  – the loop: ask model → run tools → feed back    │
 │    memory.py – per-session conversation history               │
 │    prompts.py – versioned system prompt                       │
 └───────┬───────────────────────┬────────────────────────┬───────┘
         │                       │                        │
 ┌───────▼────────┐   ┌──────────▼───────────┐   ┌────────▼────────┐
 │ 3. INFERENCE   │   │ 4. KNOWLEDGE         │   │ 5. TOOLS        │
 │ app/inference/ │◄──│ app/knowledge/       │◄──│ app/tools/      │
 │ types.py       │   │ chunking.py          │   │ registry.py     │
 │ provider.py    │   │ store.py (sqlite+np) │   │ builtin.py      │
 │ gemini.py  ←───┼───┤ ingest.py            │   │  search_notes   │
 │  (only vendor  │   │ retrieval.py         │   │  save_note      │
 │   SDK import)  │   │                      │   │  calculator     │
 │                │   │                      │   │  current_datetime│
 │                │   │                      │   │  fetch_url      │
 └────────────────┘   └──────────────────────┘   └─────────────────┘
```

| Layer | Job | Talks to | Never touches |
|---|---|---|---|
| **Client** | UI; send a message, render events | HTTP/SSE only | models, vectors, tools |
| **Intelligence** | Decide what the model sees and when tools run | inference, knowledge, tools | vendor SDKs |
| **Inference** | Call the model; translate neutral types ⇄ vendor format; embeddings | `google.genai` | the other layers |
| **Knowledge** | Chunk, embed, store, retrieve notes | inference (for embeddings) | intelligence, tools |
| **Tools** | Registry of callable functions with JSON schemas | knowledge (for `search_notes`) | intelligence |

**The dependency rule** is enforced by `tests/test_layering.py`: only
`app/inference/` may import `google.genai`, and lower layers never import
upward. `app/main.py` is the composition root — the one file that sees all
five layers and wires them together.

## One request, layer by layer

Ask *"What's on my reading list?"* and this happens:

1. **Client** posts `{session_id, message}` to `/chat` and starts reading the SSE stream.
2. **Intelligence** (`agent.run_turn`) appends the user message to session memory and calls `provider.stream_generate(system, history, registry.specs())`.
3. **Inference** (`GeminiProvider`) converts neutral `Message`s to Gemini `Content`s, streams the response, and yields `ToolCall(search_notes, {"query": "reading list"})`.
4. **Intelligence** emits a `tool_call` event to the client and calls `registry.execute(...)`.
5. **Tools** (`search_notes`) → **Knowledge** (`retrieve`) → **Inference** (`embed(query)`) → SQLite/numpy cosine top-k → chunks come back with `doc_id`s and scores.
6. **Intelligence** appends a `ToolResultPart`, loops, and the model now streams the answer as `token` events.
7. The `done` event carries sources, iterations, tokens, and a per-layer latency breakdown; the **Client** draws the timeline and the layer bar.

The server logs the same thing as one JSON line per call:

```json
{"event": "chat_call", "query": "What's on my reading list?", "iterations": 2,
 "tools_used": ["search_notes"], "sources": ["reading-list", ...],
 "per_layer_ms": {"inference": 3737.56, "knowledge": 0.75, "tools": 528.86, "intelligence": 0.15},
 "steps": [{"layer": "inference", "name": "generate", "latency_ms": 2108.7, "self_ms": 2108.7, "depth": 0, ...},
           {"layer": "tools", "name": "search_notes", ..., "depth": 1}, ...]}
```

`latency_ms` is inclusive, `self_ms` is exclusive of nested steps, so the
per-layer totals don't double count (`tools.search_notes` wraps
`inference.embed_query`).

## Run it

```bash
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # put your GEMINI_API_KEY in .env
python -m scripts.ingest    # chunk + embed data/notes/*.md into data/knowledge.sqlite
uvicorn app.main:app --reload
```

Open <http://localhost:8000>. Try the suggestions in the sidebar:

- *What's on my reading list?* → `search_notes`, answer cites `reading-list`
- *How is this assistant built?* → the notes include a description of this very architecture
- *What is 17% of 2,340?* → `calculator`, no knowledge lookup
- *Save a note titled Groceries: milk, eggs, coffee* → `save_note` writes the file and indexes it
- *What did I just save?* → session memory + `search_notes` finds the new note
- *Hi* → no tools at all

Raw SSE, if you want to see the wire format:

```bash
curl -N -X POST localhost:8000/chat -H 'Content-Type: application/json' \
  -d '{"session_id":"cli","message":"What is on my reading list?"}'
```

Other endpoints: `GET /health`, `GET /notes`, `POST /ingest`, `POST /reset/{session_id}`, `GET /docs`.

### Model choice and free-tier quotas

Default is `gemini-3.5-flash-lite` (set `GENERATION_MODEL` in `.env`). The
bigger `gemini-3.6-flash` works too but the free tier allows only **20
requests per day** on it, which an agent loop (2–3 model calls per turn)
burns quickly. Daily-quota exhaustion comes back as a 429 with a long
`retryDelay`; the provider gives up instead of sleeping when that delay
exceeds `rate_limit_max_wait_s`.

## Tests

```bash
pytest
```

Everything runs offline against `tests/fake_provider.py`, a scripted
`LLMProvider`: the layering rule, chunking, the vector store, the tool
registry, and the agent loop (tool call → result fed back → final answer,
memory persistence, max-iteration guard).

## How to swap a layer

Because each layer only depends on the contract below it, swapping one is local:

- **Different model provider** → write `app/inference/claude.py` implementing
  `LLMProvider` (two methods: `stream_generate`, `embed`) and return it from
  `get_provider()`. Nothing else changes. Provider-specific state the model
  needs echoed back (Gemini's `thought_signature`) rides along in
  `provider_state` on parts, opaque to every other layer.
- **Real vector database** → replace `app/knowledge/store.py` with a
  Qdrant/pgvector client exposing the same five methods.
- **New tool** → add a `Tool(name, description, json_schema, handler)` in
  `app/tools/builtin.py`. The model sees it on the next request.
- **Different client** → anything that can POST JSON and read SSE (a CLI,
  Slack bot, mobile app) is a client. The server doesn't care.
- **Persistent memory** → replace `SessionStore` with a Redis/Postgres-backed
  class that stores the same neutral `Message` list.

## Layout

```
app/
  main.py                composition root
  config.py              settings from .env
  observability.py       time_step(layer, name) tracing shared by all layers
  api/routes.py          HTTP edge: AgentEvents → SSE
  intelligence/          agent.py, memory.py, prompts.py
  inference/             types.py, provider.py, gemini.py
  knowledge/             chunking.py, store.py, ingest.py, retrieval.py
  tools/                 registry.py, builtin.py
client/                  index.html, app.js, style.css
data/notes/*.md          the knowledge base (5 sample notes)
data/knowledge.sqlite    built by scripts/ingest.py
scripts/ingest.py
tests/
```
