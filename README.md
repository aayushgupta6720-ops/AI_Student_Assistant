# AI_Student_Assistant

[![GitHub repo](https://img.shields.io/badge/GitHub-AI__Student__Assistant-181717?logo=github)](https://github.com/aayushgupta6720-ops/AI_Student_Assistant)
[![Live on Render](https://img.shields.io/badge/Live-Render-46E3B7?logo=render)](https://ai-student-assistant-hz02.onrender.com)

**Live demo:** https://ai-student-assistant-hz02.onrender.com
([`/health`](https://ai-student-assistant-hz02.onrender.com/health),
[`/docs`](https://ai-student-assistant-hz02.onrender.com/docs)) — Render's
free tier, so the first request after a period of inactivity takes ~30–60 s
to wake up.

A small, working AI assistant built to show how the **five architectural
layers** of an AI application fit together: **Client, Intelligence,
Inference, Knowledge, Tools**. Each layer is its own Python package with a
one-way dependency direction, and every request produces a trace showing
which layers ran and for how long.

Domain: a personal study-notes assistant. It searches a folder of shared
markdown notes plus your own notes, uploaded (`.md`, `.txt` or `.pdf`) or
saved from the chat, which are **private to your chat session**. Another
visitor's searches never see them, and they're deleted on *New session* or
after 24 hours.
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
python -m scripts.ingest    # sync data/notes/*.md into data/knowledge.sqlite (deleted notes are dropped)
uvicorn app.main:app --reload
```

Open <http://localhost:8000>. Try the suggestions in the sidebar:

- *What's on my reading list?* → `search_notes`, answer cites `reading-list`
- *How is this assistant built?* → the notes include a description of this very architecture
- *What is 17% of 2,340?* → `calculator`, no knowledge lookup
- *Save a note titled Groceries: milk, eggs, coffee* → `save_note` stores it as a private note for this chat
- *What did I just save?* → session memory + `search_notes` finds the new note
- *Hi* → no tools at all

Raw SSE, if you want to see the wire format:

```bash
curl -N -X POST localhost:8000/chat -H 'Content-Type: application/json' \
  -d '{"session_id":"cli","message":"What is on my reading list?"}'
```

`/chat` also takes an optional `timezone` (an IANA name like `Asia/Kolkata`)
for `current_datetime` to answer in; the web client sends the browser's.
Without one it uses the server's zone, which on Render is UTC.

Other endpoints: `GET /health`, `GET /notes?session_id=…` (shared notes plus
that session's private notes), `POST /notes/upload` (multipart `session_id` +
`file`; up to 2 MB and 200,000 characters of text), `DELETE /notes/{doc_id}?session_id=…`,
`POST /ingest`, `POST /reset/{session_id}` (also deletes the session's private
notes unless `?keep_uploads=true`), `GET /docs`. A session can have 10 private
notes at a time, uploads and saved notes together.

How private notes stay private: every stored chunk has an owner, either `""`
for the shared notes or the session id that uploaded or saved it. `search()`
masks out other owners' rows, and the agent sets a `current_session` context
variable each turn, so `search_notes` and `save_note` are scoped without
passing a session id through every tool. `save_note` never writes to
`data/notes/`: that folder is the same for every visitor.

What the tools refuse: `fetch_url` only fetches public http(s) addresses (not
localhost, private networks or cloud metadata endpoints), checks every
redirect, reads at most 2 MB, and gives up after 20 s. `calculator` refuses
results over about 1,200 digits, since it runs on the event loop and a
`9**9**9` would stall every chat. Answers are rendered as markdown through
DOMPurify with only markdown's own tags allowed, so HTML that the model
repeats from a web page or note can't run script or load images.

### Per-visitor rate limits

The free-tier quota is shared by everyone using the app, so each visitor
(an IP address; for IPv6, its /64) gets its own allowance: 6 chat messages a
minute and 30 a day, 10 uploads an hour and 3 re-ingests an hour. Over a
limit, the endpoint returns a 429 with a `Retry-After` header, and the chat
says how long to wait. The model isn't called for a refused request. Change
the numbers with `CHAT_LIMIT_PER_MINUTE`, `CHAT_LIMIT_PER_DAY`,
`UPLOAD_LIMIT_PER_HOUR` and `INGEST_LIMIT_PER_HOUR`; 0 turns one off. Counts
are in memory, which suits the single instance of Render's free plan, and
reset on restart. Visitors sharing an IP, like a classroom behind one
router, share one allowance.

Behind a proxy, the connecting address is the proxy's, so `CLIENT_IP_HEADER`
names the header that carries the visitor's IP. Only use a header that the
proxy *overwrites* when a client sends it. On Render that's
`CF-Connecting-IP`: Cloudflare, in front of Render, sets it and refuses
requests that bring their own (a 403, "error code: 1000"). Not
`X-Forwarded-For`: Render appends to it, and it sets `FORWARDED_ALLOW_IPS=*`
for Python services, so uvicorn takes that header's first entry, which the
visitor chose, as the connecting address. If the configured header is
missing from a request, everyone without it shares one allowance.

`render.yaml` sets `CLIENT_IP_HEADER`, but only services created from the
Blueprint pick up `render.yaml`. The live demo was created in the dashboard,
so its `CLIENT_IP_HEADER=CF-Connecting-IP` is set there; do the same for any
service you create by hand.

### Model choice and free-tier quotas

Default is `gemini-3.5-flash-lite` (set `GENERATION_MODEL` in `.env`), which
the free tier allows 500 requests/day. Free-tier quotas are per model *and*
per Google project, so give this app a key from its own project if anything
else (an eval run, another demo) uses the same key. The bigger
`gemini-3.5-flash` and `gemini-3.6-flash` work, but the free tier allows only
**20 requests per day** on each, which an agent loop (2–3 model calls per
turn) burns in a few turns. Daily-quota exhaustion comes back as a 429 with a long
`retryDelay`; the provider gives up instead of sleeping when that delay
exceeds `rate_limit_max_wait_s`, and the chat shows when the quota resets.
Gemini also has capacity spikes (503 UNAVAILABLE, "high demand"): those are
retried after 1s, 2s and 4s, then the chat says the model is overloaded and
to try again in a minute, rather than showing the raw error. A call that
sends nothing for `GEMINI_TIMEOUT_S` (60s), at the start or partway through a
streamed answer, is stopped and the chat says so, instead of hanging.

## Deploy to Render

`render.yaml` defines the service as a Render Blueprint (native Python
runtime, free plan, `/health` as the health check). In the Render dashboard
choose **New → Blueprint**, pick this repo, and paste your `GEMINI_API_KEY`
when prompted (it's marked `sync: false` so it never lives in the repo).

Render's filesystem is ephemeral, so the app re-ingests `data/notes/*.md`
on startup whenever the index is empty. If that fails (a Gemini quota 429,
a bad key), the app still boots and logs `startup_ingest_failed`; chat works
but `search_notes` finds nothing until `POST /ingest` succeeds. Private
notes (uploads and `save_note`) live in the same index, so a deploy or
restart also clears them. The free plan sleeps after inactivity, so the
first request after a while takes ~30–60 s.

To check the rate limit identifies visitors correctly after a deploy, send
11 uploads that fail validation (so no quota is spent), each with a made-up
`X-Forwarded-For` (not `CF-Connecting-IP`, which Cloudflare refuses with a 403
before it reaches the app):

```bash
for i in $(seq 11); do curl -s -o /dev/null -w '%{http_code} ' \
  -H "X-Forwarded-For: 10.9.9.$i" \
  -F session_id=check-$i -F 'file=@/dev/null;filename=x.exe' \
  https://ai-student-assistant-hz02.onrender.com/notes/upload; done
```

Ten `400`s then a `429` means the made-up headers didn't count as new
visitors. It uses up your own upload allowance for an hour. Run it against the deployed app, not localhost: uvicorn trusts
`X-Forwarded-For` on connections from `127.0.0.1` by default
(`--forwarded-allow-ips`), so requests from your own machine can pick their
address and you'll see eleven `400`s. Then check the Render logs for
`client_ip_header_missing`: if it appears, the header isn't reaching the app
and all visitors share one allowance, so `CLIENT_IP_HEADER` needs changing.

## Tests

```bash
pytest
```

Everything runs offline against `tests/fake_provider.py`, a scripted
`LLMProvider`: the layering rule, chunking, the vector store, the tool
registry, and the agent loop (tool call → result fed back → final answer,
memory persistence, max-iteration guard). `fetch_url` is tested against
local HTTP servers: private addresses, redirects, huge and slow pages.

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
