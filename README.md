# AI Student Assistant

A study-notes chatbot I built to learn how a real AI application is put together, from the chat window down to the model API.

[![GitHub repo](https://img.shields.io/badge/GitHub-AI__Student__Assistant-181717?logo=github)](https://github.com/aayushgupta6720-ops/AI_Student_Assistant)
[![Live on Render](https://img.shields.io/badge/Live-Render-46E3B7?logo=render)](https://ai-student-assistant-hz02.onrender.com)

**Live demo:** https://ai-student-assistant-hz02.onrender.com
([`/health`](https://ai-student-assistant-hz02.onrender.com/health), [`/docs`](https://ai-student-assistant-hz02.onrender.com/docs)).
It runs on Render's free tier, so the first request after a quiet period takes 30–60 seconds while the server wakes up.

## About

Most AI app tutorials fit in one file: call the model, print the answer. I wanted to understand how a production AI application is organised, so I built a small but complete one around five layers: **Client, Intelligence, Inference, Knowledge and Tools**. Each layer is its own Python package, dependencies only point one way, and a test fails if a layer imports something it shouldn't.

The app itself is a study assistant. You can ask questions about a set of shared notes, upload your own notes (Markdown, text or PDF, including scanned PDFs, which Gemini transcribes), save notes from the chat, and use tools such as a calculator and a web page reader. Every answer shows which layers did the work and how long each took, which kept the architecture visible while I was building it.

## Features

- **Streaming chat with tool calling.** The model decides when to use one of six tools (search notes, read a whole note, save a note, calculator, current date and time, read a web page or PDF), and answers stream in token by token over Server-Sent Events.
- **Retrieval over your notes (RAG).** Notes are split into chunks, embedded with Gemini and searched by cosine similarity, with codes and symbols such as `T07` or `MPSH 2A` also matched as exact keywords. Each answer lists the notes it used, and clicking one shows the exact passages.
- **Private notes.** Uploads and saved notes belong to your session only, a random id the server issues in a cookie the page's scripts can't read: other visitors' searches never see them. They're deleted when you click New session, and otherwise within 24 hours (sooner if the free server restarts). The chat history that quotes them expires after 24 hours too.
- **Visible architecture.** Under each answer is a timeline of every step and a bar showing how long each layer took.
- **Stop button.** Stops an answer mid-stream without leaving the conversation history in a broken state.
- **Built for a shared public demo.** Per-visitor rate limits, request size limits, safe URL fetching (no access to private networks) and sanitised Markdown rendering.
- **Tested.** Offline tests using a scripted fake model, run by GitHub Actions on every push, plus a small live evaluation of the model's tool choices.

## Tech stack

| Area | Technology |
|---|---|
| Backend | Python 3.12, FastAPI, Uvicorn |
| Model | Google Gemini through `google-genai`: `gemini-3.5-flash-lite` for chat, `gemini-embedding-001` for embeddings |
| Vector search | SQLite for storage, NumPy for cosine similarity |
| Frontend | HTML, CSS and vanilla JavaScript over Server-Sent Events; marked and DOMPurify for Markdown |
| Documents | pypdf |
| Testing | pytest, pytest-asyncio |
| Hosting | Render (free plan) |

## Architecture

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
 │   SDK import)  │   │ uploads.py           │   │  calculator     │
 │                │   │                      │   │  current_datetime│
 │                │   │                      │   │  fetch_url      │
 │                │   │                      │   │  read_note      │
 └────────────────┘   └──────────────────────┘   └─────────────────┘
```

| Layer | Job | Talks to | Never touches |
|---|---|---|---|
| **Client** | The UI: send a message, render the events | HTTP/SSE only | models, vectors, tools |
| **Intelligence** | Decide what the model sees and when tools run | inference, knowledge, tools | vendor SDKs |
| **Inference** | Call the model and translate to and from its format; embeddings | `google.genai` | the other layers |
| **Knowledge** | Chunk, embed, store and retrieve notes | inference (for embeddings) | intelligence, tools |
| **Tools** | A registry of functions the model can call | knowledge (for `search_notes`) | intelligence |

The dependency rule is enforced by `tests/test_layering.py`, on the import graph rather than on text: it parses every module, resolves relative imports and `importlib` calls, and checks each import against this table, so only `app/inference/` may import the Gemini SDK and no layer imports one it shouldn't. `app/main.py` is the one file that sees all five layers and wires them together.

### One request, step by step

When you ask *"What's on my reading list?"*:

1. The **client** sends the message to `POST /chat` and starts reading the event stream.
2. The **intelligence** layer (`agent.py`) adds it to the session's history and asks the model for a response, offering it the tool definitions.
3. The **inference** layer (`gemini.py`) translates the conversation into Gemini's format and streams back a tool call: `search_notes("reading list")`.
4. The agent runs the tool. `search_notes` asks the **knowledge** layer, which embeds the query and finds the closest chunks in the vector store.
5. The results go back to the model, which streams its answer. The final `done` event carries the sources, token counts and per-layer timings, which the client draws under the answer.

The server logs the same trace as one JSON line per chat (see [request flow and tracing](docs/ENGINEERING_NOTES.md#request-flow-and-tracing)).

### Extending it

Because each layer only depends on the interface below it, changing one stays local:

- **Another model provider:** add a module to `app/inference/` implementing `LLMProvider` (`stream_generate` and `embed`) and return it from `get_provider()`.
- **A real vector database:** replace `app/knowledge/store.py` with a pgvector or Qdrant client that has the same methods.
- **A new tool:** register a `Tool(name, description, json_schema, handler)` in `app/tools/builtin.py`, and the model can use it on the next request.
- **A different client:** anything that can POST JSON and read Server-Sent Events works, such as a CLI or a Slack bot.
- **Persistent chat memory:** replace `SessionStore` with a Redis- or Postgres-backed class that stores the same `Message` list.

## Getting started

### Prerequisites

- Python 3.12
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/) (the free tier is enough)

### Setup

```bash
git clone https://github.com/aayushgupta6720-ops/AI_Student_Assistant.git
cd AI_Student_Assistant
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then put your GEMINI_API_KEY in .env
uvicorn app.main:app --reload
```

On startup the app indexes the sample notes in `data/notes` (`python -m scripts.ingest` does the same from the command line). Open http://localhost:8000 and try the suggestions in the sidebar:

- *What's on my reading list?* searches the notes and cites `reading-list`
- *How is this assistant built?* answers from a note that describes this architecture
- *What is 17% of 2,340?* uses the calculator
- *Save a note titled Groceries: milk, eggs, coffee*, then *What did I just save?*
- *Hi* uses no tools at all

### Configuration

Settings are read from `.env`; `app/config.py` lists all of them. The main ones:

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | none | Required |
| `GENERATION_MODEL` | `gemini-3.5-flash-lite` | The chat model ([why this one](docs/ENGINEERING_NOTES.md#model-choice-and-free-tier-quotas)) |
| `CLIENT_IP_HEADER` | unset | The header holding the visitor's IP behind a proxy (`CF-Connecting-IP` on Render) |
| `LOG_CHAT_TEXT` | `false` | Log what visitors type, not just its length |
| `CHAT_LIMIT_PER_MINUTE`, `CHAT_LIMIT_PER_DAY` and others | 6, 30, … | Per-visitor limits ([all of them](docs/ENGINEERING_NOTES.md#rate-limits-and-request-size)) |

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/chat` | Send `{message, timezone?}` and get a Server-Sent Events stream back (`status`, `tool_call`, `tool_result`, `token`, `done`, `error`). `timezone` is an IANA name such as `Asia/Kolkata`, for the date tool (the web page sends the browser's); without it, the server's zone is used, which is UTC on Render. |
| `GET` | `/notes` | The shared notes plus this session's private notes |
| `POST` | `/notes/upload` | Upload a `.md`, `.txt` or `.pdf` note (multipart `file`, up to 2 MB) |
| `DELETE` | `/notes/{doc_id}` | Delete one of this session's private notes |
| `POST` | `/ingest` | Re-index `data/notes`; only new or changed notes are embedded |
| `POST` | `/reset` | Clear the conversation and this session's private notes (`?keep_uploads=true` keeps the notes) |
| `GET` | `/health` | Status, model, number of indexed chunks and tool names |
| `GET` | `/docs` | Interactive API documentation from FastAPI |

Every endpoint works on the caller's session, which the server issues in an HttpOnly `sid` cookie on the first request; a cookie it couldn't have issued is replaced. To see the raw event stream, keep the cookie in a jar so later calls share the session:

```bash
curl -N -c jar -b jar -X POST localhost:8000/chat -H 'Content-Type: application/json' \
  -d '{"message":"What is on my reading list?"}'
```

## Testing

```bash
pytest
```

The tests run offline in about two seconds, and GitHub Actions runs them on every push (`.github/workflows/tests.yml`). Instead of calling Gemini they use `tests/fake_provider.py`, a scripted model that returns tool calls and text on cue. That makes the agent loop, tools, retrieval, uploads, rate limits and API testable end to end without network access. `fetch_url` is tested against local HTTP servers.

Whether the model picks the right tool can only be checked against the real model, so there's also a small evaluation that asks nine typical questions and checks which tools each one used:

```bash
python -m scripts.eval_routing    # uses about 20 Gemini requests
```

A second evaluation checks the answers themselves: it uploads three study notes and asks 20 questions, from single facts and course codes to whole-document summaries and questions the notes don't cover ([more](docs/ENGINEERING_NOTES.md#evaluating-answers)). It spends about 120 requests, so it runs on a separate key (`EVAL_GEMINI_API_KEY`):

```bash
python -m scripts.eval_answers
```

## Deployment

The live demo runs on Render's free plan. `render.yaml` defines the service as a Blueprint for deploying your own copy (the live demo itself was created in the dashboard, so it doesn't read the file). In the Render dashboard, choose **New → Blueprint**, select this repository through your connected GitHub account, and enter your `GEMINI_API_KEY` when prompted. Render wipes the disk on every deploy, so the app rebuilds the index on startup. The [engineering notes](docs/ENGINEERING_NOTES.md#deploying-to-render) cover the rest, including the proxy header the rate limits depend on.

## Engineering challenges

Running a public demo on free tiers turned up problems I wouldn't have met in a tutorial. The highlights are below; [docs/ENGINEERING_NOTES.md](docs/ENGINEERING_NOTES.md) has the details and the measurements.

- **Sharing one free quota fairly.** Every visitor draws on the same 500 Gemini requests a day, so each IP address gets its own allowance. Behind Render's proxy, the usual `X-Forwarded-For` header can be forged by the visitor, so the app identifies visitors by `CF-Connecting-IP`, which Cloudflare sets itself. ([more](docs/ENGINEERING_NOTES.md#rate-limits-and-request-size))
- **A 13 KB PDF that could crash the server.** A PDF full of compressed drawing instructions pushed memory from 66 MB to 496 MB, close to the instance's 512 MB. I added limits on how far the PDF library expands and parses a file, and the same PDF is now refused in under 0.1 seconds. ([more](docs/ENGINEERING_NOTES.md#pdf-limits))
- **Letting the model fetch URLs safely.** `fetch_url` must not reach the server's private network or cloud metadata endpoints. It checks every redirect and connects only to the exact address it checked, which also defeats DNS rebinding. It opens only links the user typed, so a page that tells the model to fetch a URL carrying the user's notes gets nowhere. ([more](docs/ENGINEERING_NOTES.md#tool-safety))
- **Keeping the conversation history valid.** A calculator result like `(-8)**(1/3)` is a complex number, which can't be sent back to the model as JSON, and once it was in the history every later message failed. Tool results are now checked before they're stored, and stopped or failed turns are tidied out of the history. ([more](docs/ENGINEERING_NOTES.md#the-agent-loop))
- **Showing only relevant sources.** Search always returned the top four passages, so answers cited unrelated notes. Measuring the similarity scores showed a clear gap between relevant and unrelated passages, so results well below the best match are now dropped. ([more](docs/ENGINEERING_NOTES.md#retrieval))
- **Catching a prompt regression.** In production, the model answered "How is this assistant built?" from general knowledge instead of the notes. A small evaluation reproduced it, and adding one rule to the system prompt fixed it without breaking the other questions. ([more](docs/ENGINEERING_NOTES.md#evaluating-tool-routing))

## What I learned

- **Interfaces matter more than frameworks.** Keeping the model behind a small interface (`stream_generate` and `embed`) meant the agent, the tools and the tests never needed to know it was Gemini.
- **Constraints shape the design.** A 500-requests-a-day quota and 512 MB of memory decided more of the architecture than any feature did: rate limits, upload caps, memory limits and retry logic all came from them.
- **Measure before fixing.** Measuring changed several of my fixes. An absolute relevance cutoff looked obvious until the scores showed it would drop correct answers, and some history problems I expected to break Gemini turned out to be harmless once I tested them.
- **LLM applications fail in unusual ways.** Blocked answers with no content, tool results that can't be serialised, and half-finished turns left in the history all needed handling that an ordinary web app doesn't.
- **A fake model makes an AI app testable.** Scripting the model's responses let me test the agent loop deterministically and offline.
- **Prompts need tests too.** A prompt change can fix one question and quietly break another, and a small evaluation makes that visible.

## Future work

- Persistent storage (Postgres with pgvector, and Redis for sessions) so notes and conversations survive deploys and restarts
- User accounts instead of anonymous session ids
- Rate-limit counters shared between instances, so the app can run on more than one server
- The paid Gemini tier, whose terms don't let Google use uploaded notes to improve its products; the free tier's terms do
- A Dockerfile, so it runs the same way anywhere

## Project structure

```
app/
  main.py                composition root: builds and wires the five layers
  config.py              settings from .env
  observability.py       time_step(layer, name) tracing shared by all layers
  api/                   routes.py (HTTP and SSE), ratelimit.py, body_limit.py
  intelligence/          agent.py, memory.py, prompts.py
  inference/             types.py, provider.py, gemini.py
  knowledge/             chunking.py, store.py, ingest.py, retrieval.py, uploads.py
  tools/                 registry.py, builtin.py
client/                  index.html, app.js, style.css
data/notes/*.md          the shared sample notes
docs/                    ENGINEERING_NOTES.md: design details, limits and measurements
scripts/                 ingest.py (index data/notes), eval_routing.py (live tool-routing check),
                         eval_answers.py (live answer-quality check over data/eval)
tests/                   pytest suite, with a scripted fake model
```
