# Engineering notes

The detail behind the [README](../README.md): how each part behaves, where the limits are set and why, and the problems found while running the app as a public demo on free tiers (Gemini's 500 requests a day, and Render's 512 MB instance).

- [Request flow and tracing](#request-flow-and-tracing)
- [The agent loop](#the-agent-loop)
- [Retrieval](#retrieval)
- [Private notes](#private-notes)
- [PDF limits](#pdf-limits)
- [Tool safety](#tool-safety)
- [Rate limits and request size](#rate-limits-and-request-size)
- [Model choice and free-tier quotas](#model-choice-and-free-tier-quotas)
- [Deploying to Render](#deploying-to-render)
- [Evaluating tool routing](#evaluating-tool-routing)

## Request flow and tracing

Asking *"What's on my reading list?"* goes through the layers like this:

1. **Client** posts `{session_id, message}` to `/chat` and starts reading the SSE stream.
2. **Intelligence** (`agent.run_turn`) appends the user message to session memory and calls `provider.stream_generate(system, history, registry.specs())`.
3. **Inference** (`GeminiProvider`) converts the neutral `Message`s to Gemini `Content`s, streams the response, and yields `ToolCall(search_notes, {"query": "reading list"})`.
4. **Intelligence** emits a `tool_call` event to the client and calls `registry.execute(...)`.
5. **Tools** (`search_notes`) → **Knowledge** (`retrieve`) → **Inference** (`embed(query)`) → SQLite/NumPy cosine top-k → chunks come back with `doc_id`s and scores.
6. **Intelligence** appends a `ToolResultPart` and loops, and the model now streams the answer as `token` events.
7. The `done` event carries the sources, iterations, tokens and a per-layer latency breakdown, which the client draws as the timeline and the layer bar.

Every layer wraps its work in `time_step(layer, name)` (`app/observability.py`). The server logs each chat as one JSON line:

```json
{"event": "chat_call", "query": "<26 chars>", "iterations": 2,
 "tools_used": ["search_notes"], "sources": ["reading-list"], "finish_reason": "stop",
 "per_layer_ms": {"inference": 3737.56, "knowledge": 0.75, "tools": 528.86, "intelligence": 0.15},
 "steps": [{"layer": "inference", "name": "generate", "latency_ms": 2108.7, "self_ms": 2108.7, "depth": 0, ...},
           {"layer": "tools", "name": "search_notes", "meta": {"args": {"query": "<12 chars>"}}, ..., "depth": 1}, ...]}
```

`latency_ms` is inclusive and `self_ms` excludes nested steps, so the per-layer totals don't double-count (`tools.search_notes` wraps `inference.embed_query`).

What visitors type, and the text of tool arguments, is logged as its length. It can contain a private note ("Save a note titled…"), and the log would keep it long after the note's 24 hours. A private note's name is its title or file name, so it appears as `<private>` among the sources and as a length on upload. Search passages are never logged. Set `LOG_CHAT_TEXT=true` to log the text itself while debugging.

## The agent loop

- **Iteration cap.** A turn makes at most `MAX_AGENT_ITERATIONS` (6) model calls. The last one is offered no tools and told to answer from what it has, so a turn always ends with an answer rather than tool results nobody reads.
- **Tool results must be valid JSON.** Results stay in the chat's history and go back to the model on every later turn. A calculator result like `(-8)**(1/3)` (a complex number) crashed the SDK's JSON encoding, and `1e308*10` went out as bare `Infinity`, which Gemini rejects with a 400. Either one broke every later message in that chat. The calculator refuses such results, and the tool registry turns any result that can't be sent as JSON into an error.
- **Early stops are explained.** When the model stops for a reason other than finishing (the length limit, a safety filter, a malformed tool call), the inference layer reports it in provider-neutral terms (`finish_reason`) and the chat shows why under the answer, instead of stopping mid-sentence or showing nothing. A blocked answer can arrive with no content at all, and the adapter handles that too.
- **Stop and unfinished turns.** Stop (the button that replaces Send while an answer streams, or Esc) aborts the request. The server sees the connection close, cancels the turn and logs `chat_stopped`. The agent then leaves memory as the user saw it: the question and whatever answer text had arrived. A tool call whose result never came is dropped. So is a question with no answer yet, or a turn that ended with no answer at all (blocked or empty), since the next message would otherwise get an answer to both. The same tidy-up runs when a tab closes mid-answer or the model fails partway, and New session stops a streaming answer before clearing the chat.

## Retrieval

- Notes are split into chunks of up to 800 characters, each labelled with the note title and section headings it sits under. Splitting is line-based: fenced code blocks stay whole (so a `# comment` in code isn't taken for a heading), and a heading directly after a list starts its own section.
- A search returns up to 4 passages (`search_notes` accepts 1 to 10), minus any scoring more than 0.1 below the best match (`RETRIEVAL_SCORE_MARGIN`). For questions about the sample notes, the right note scored 0.64–0.78 and unrelated ones 0.50–0.58, and before the cutoff every answer cited the whole top 4.
- The cutoff is relative, not absolute, because a terse query's right answer can score as low as an unrelated note does for another query ("headphones" finds the travel checklist at 0.559, while an unrelated question's best match is 0.542).
- Clicking a source chip shows the passages the search returned from that note, the text the answer was based on. The `done` event carries them as `passages`.

## Private notes

- **Scoping.** Every stored chunk has an owner: `""` for the shared notes, or the session id that uploaded or saved it. `search()` masks out other owners' rows, and the agent sets a `current_session` context variable each turn, so `search_notes` and `save_note` are scoped without passing a session id through every tool. `save_note` never writes to `data/notes/`, which is the same for every visitor.
- **Lifetime.** Private notes are deleted on New session (`POST /reset/{session_id}`, unless `?keep_uploads=true`, which a page reload uses) or after 24 hours. They live in the same index as the shared notes, so a deploy or restart also clears them.
- **Caps.** A session can have 10 private notes at a time, uploads and saved notes together. An upload is at most 2 MB and 200,000 characters of text, and a saved note at most 200,000 characters. The whole app keeps at most 10,000 private chunks (`MAX_PRIVATE_CHUNKS`), because search holds them all in memory: 10,000 took the process from 86 MB to a 205 MB peak. Past the cap, uploads get a 503 until older notes expire.
- **Names.** A private note's id comes from its file name or title, in any script (`биология`, `生物`), and never matches a shared note's id: "Reading List.md" becomes `reading-list-2`, or searches and source chips would mix the two notes up.
- **Windows files.** Text files with CRLF line endings or a byte-order mark are normalised, since the chunker splits paragraphs on blank lines.

## PDF limits

PDFs, uploaded or linked, are the one input where a few KB can cost a lot. pypdf holds about 55 bytes of memory per byte of a page's drawing instructions while it extracts text, parses them at about 2 MB/s in pure Python (holding the GIL, so every other request slows down), and by default expands a compressed stream up to 75 MB. A 13 KB PDF took the process from 66 MB to a 496 MB peak, on an instance with 512 MB.

So text extraction now:
- expands no stream past 2 MB;
- skips pages with more than 512 KB of drawing instructions (drawings; a page of text is tens of KB);
- parses at most 4 MB of instructions per PDF.

That PDF is now refused in under 0.1 s. The worst variant tested, 40 pages each just under the per-page limit, is refused in 1.8 s with 111 MB peak memory. Scanned PDFs without a text layer are refused with a message saying so.

## Tool safety

**`fetch_url`** only fetches public http(s) addresses: not localhost, private networks or cloud metadata endpoints.
- It checks every redirect, reads at most 2 MB (10 MB for a PDF, which can't be read from a prefix) and gives up after 20 s.
- It resolves the host once, checks every address, and connects to the one it checked, sending the real name in `Host` and for TLS certificate checks. A DNS server that answers "public" to the check and "private" to a second lookup (DNS rebinding) gets nowhere. Before this, the request reached the private address before the response was refused.
- PDFs are recognised by their first bytes rather than their headers or address, and read with the same extraction and limits as uploads. Other binary files (images, zips) are refused rather than decoded as garbage text.

**`calculator`** evaluates arithmetic without `eval`. It refuses results over about 1,200 digits, since it runs on the event loop and a `9**9**9` would stall every chat, as well as results JSON can't carry (see [the agent loop](#the-agent-loop)).

**Rendering.** Answers are rendered as Markdown through DOMPurify with only Markdown's own tags allowed, so HTML the model repeats from a web page or note can't run scripts or load images. marked and DOMPurify load from cdnjs pinned by integrity hash, so a tampered copy doesn't run (answers then show as plain text). Links in answers open in a new tab, since leaving the page and coming back clears the chat.

## Rate limits and request size

The free-tier quota is shared by everyone using the app, so each visitor (an IP address; for IPv6, its /64) gets its own allowance:

| Limit | Default | Setting |
|---|---|---|
| Chat messages | 6 a minute, 30 a day | `CHAT_LIMIT_PER_MINUTE`, `CHAT_LIMIT_PER_DAY` |
| Uploads | 10 an hour | `UPLOAD_LIMIT_PER_HOUR` |
| Note chunks (uploads and saved notes) | 1,000 a day | `UPLOAD_CHUNK_LIMIT_PER_DAY` |
| Re-ingests | 3 an hour | `INGEST_LIMIT_PER_HOUR` |

A setting of 0 turns that limit off.
- **Chunks, not just files.** A single upload can be about 500 chunks to embed and keep in memory, and starting a new session doesn't reset the count. Notes saved from chat count against the same budget, or chat alone could fill the server-wide cap.
- **Refusals.** Over a limit, the endpoint returns a 429 with a `Retry-After` header, and the chat says how long to wait. The model isn't called for a refused request, and an upload over its chunk budget is refused before anything is embedded.
- **Invalid messages.** A chat message only counts once it's valid, so an over-long one (the page checks the 8,000-character limit before sending) doesn't use up the minute's allowance.
- **Where counts live.** They're kept in memory, which suits the single instance of Render's free plan, and reset on restart. Visitors sharing an IP, like a classroom behind one router, share one allowance.

Request bodies over 128 KB (2.1 MB for `/notes/upload`) are refused with a 413 before they're read. FastAPI otherwise reads and parses a whole body before checking `max_length`, and a 52 MB message cost about 250 MB of memory on its way to a 422.

### Identifying visitors behind a proxy

Behind a proxy, the connecting address is the proxy's, so `CLIENT_IP_HEADER` names the header that carries the visitor's IP. Only use a header that the proxy *overwrites* when a client sends it.
- **On Render, use `CF-Connecting-IP`.** Cloudflare, in front of Render, sets it and refuses requests that bring their own (a 403, "error code: 1000").
- **Not `X-Forwarded-For`.** Render appends to it and sets `FORWARDED_ALLOW_IPS=*` for Python services, so uvicorn takes that header's first entry, which the visitor chose, as the connecting address.
- **Missing header.** If the configured header is missing from a request, everyone without it shares one allowance. The first time that happens, the app logs `client_ip_header_missing`.

## Model choice and free-tier quotas

- **Default model.** `gemini-3.5-flash-lite` (`GENERATION_MODEL`), which the free tier allows 500 requests a day. `gemini-3.5-flash` and `gemini-3.6-flash` also work, but the free tier allows only 20 requests a day on each, which an agent loop (2–3 model calls per turn) uses up in a few turns.
- **Separate project.** Free-tier quotas are per model *and* per Google project, so give the app a key from its own project if anything else (an eval run, another demo) uses the same key.
- **Quota errors.** Daily-quota exhaustion comes back as a 429 with a long `retryDelay`. The provider gives up instead of sleeping when that delay exceeds `rate_limit_max_wait_s` (20 s), and the chat shows when the quota resets (midnight Pacific time).
- **Overloads.** Capacity spikes (503 UNAVAILABLE, "high demand") are retried after 1, 2 and 4 seconds, then the chat says the model is overloaded and to try again in a minute, rather than showing the raw error.
- **Timeouts.** A call that sends nothing for `GEMINI_TIMEOUT_S` (60 s), at the start or partway through a streamed answer, is stopped and the chat says so, instead of hanging.

## Deploying to Render

- **Setup.** `render.yaml` defines the service as a Render Blueprint (native Python runtime, free plan, `/health` as the health check). In the Render dashboard, choose **New → Blueprint**, pick this repository, and paste your `GEMINI_API_KEY` when prompted (it's marked `sync: false`, so it never lives in the repository).
- **Connect the repository properly.** Pick it through your connected GitHub account rather than pasting its public URL: Render only deploys on push for a repository connected through the account.
- **Settings for dashboard-created services.** Only services created from the Blueprint read `render.yaml`. The live demo was created in the dashboard, so its `CLIENT_IP_HEADER=CF-Connecting-IP` is set there; do the same for any service you create by hand.
- **Indexing at startup.** Render's filesystem is ephemeral, so the index is empty after every deploy. The app syncs it with `data/notes/*.md` on every startup:
  - elsewhere this costs nothing for unchanged notes, and re-embeds everything after an embedding-model change;
  - private notes from another model are dropped, since their vectors can't be searched with the new ones;
  - if the sync fails (a quota 429, a bad key), the app still boots and logs `startup_ingest_failed`, and `search_notes` finds nothing new until `POST /ingest` succeeds.
- **Sleeping.** The free plan sleeps after inactivity, so the first request after a while takes 30–60 seconds.

### Checking the rate limit after a deploy

To check that the rate limit identifies visitors correctly, send 11 uploads that fail validation (so no quota is spent), each with a made-up `X-Forwarded-For`. Don't use `CF-Connecting-IP`, which Cloudflare refuses with a 403 before it reaches the app.

```bash
for i in $(seq 11); do curl -s -o /dev/null -w '%{http_code} ' \
  -H "X-Forwarded-For: 10.9.9.$i" \
  -F session_id=check-$i -F 'file=@/dev/null;filename=x.exe' \
  https://ai-student-assistant-hz02.onrender.com/notes/upload; done
```

Ten `400`s then a `429` means the made-up headers didn't count as new visitors. It uses up your own upload allowance for an hour.
- **Run it against the deployed app, not localhost.** Uvicorn trusts `X-Forwarded-For` on connections from `127.0.0.1` by default (`--forwarded-allow-ips`), so on your own machine you'll see eleven `400`s.
- **Check the logs afterwards.** If the Render logs show `client_ip_header_missing`, the header isn't reaching the app, all visitors share one allowance, and `CLIENT_IP_HEADER` needs changing.

## Evaluating tool routing

The offline tests can't check which tool the real model picks, so `scripts/eval_routing.py` asks nine typical questions against Gemini, each in a fresh chat, and checks the tools each one used. The questions are the page's suggestions plus some that are easy to route wrong in either direction.

```bash
python -m scripts.eval_routing              # about 20 Gemini requests
python -m scripts.eval_routing --repeat 3 --only "assistant built"
```

The `assistant_v1` prompt answered "How is this assistant built?" from general knowledge instead of searching the notes that document it (0/1 in the eval, and seen in production). `assistant_v2` added one rule saying the notes also document the assistant itself. It searched in 2/2 runs, and every other question was still routed correctly. The script waits out a per-minute rate limit and retries that case once.
