# How this assistant is built: the five-layer architecture

This assistant is a small application organised into five layers. Each layer
has one job and only talks to the layers below it.

## 1. Client layer

The client is a single-page web app (index.html, app.js) served by FastAPI.
It knows nothing about models or vectors. It sends the user's message to
`POST /chat` and consumes a Server-Sent Events (SSE) stream of events:
status, tool_call, tool_result, token, and done. It renders a timeline of
what happened during the turn and a bar showing how long each layer took.

## 2. Intelligence layer

The intelligence layer is the agent loop in `app/intelligence/agent.py`. It
holds the session memory (the conversation so far), builds the system prompt,
and asks the inference layer for a response. If the model asks to call a
tool, the intelligence layer runs it through the tools layer, appends the
result to the conversation, and asks the model again. It repeats until the
model answers in plain text or a maximum of six iterations is reached.
The intelligence layer decides *when* to use tools and knowledge; it never
calls a vendor SDK directly.

## 3. Inference layer

The inference layer is the only place that imports the Gemini SDK
(`app/inference/gemini.py`). It translates provider-neutral Message and
ToolSpec objects into Gemini's Content/Part/Tool format, streams the
response, and translates it back into neutral events (TextDelta, ToolCall,
Usage). It also produces embeddings. Swapping Gemini for Claude means
rewriting this one file.

## 4. Knowledge layer

The knowledge layer turns notes into something searchable. Ingestion reads
markdown files, splits them into paragraph-sized chunks, embeds each chunk
via the inference layer, and stores the vectors in SQLite. Retrieval embeds
the query and does a cosine similarity search with numpy to return the
top-k chunks. The store is deliberately tiny so the mechanics are visible.

## 5. Tools layer

The tools layer is a registry of functions the model may call: search_notes
(which bridges into the knowledge layer), calculator, current_datetime,
save_note, and fetch_url. Each tool declares a name, a description, and a
JSON schema for its input. The registry converts these into ToolSpec objects
for the inference layer and executes calls by name, catching errors so the
model gets an error message instead of the app crashing.

## The dependency rule

Client -> (HTTP) -> Intelligence -> Inference, Knowledge, Tools.
Tools -> Knowledge. Knowledge -> Inference. Inference -> vendor SDK only.
A test (tests/test_layering.py) fails if any package other than inference
imports google.genai.
