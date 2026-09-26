// Client layer. Knows three things: how to POST /chat, how to parse SSE, and
// how to render the events. It has no idea what a model, tool or vector is.

// Kept in sessionStorage so a reload keeps this tab's private uploads (the
// server scopes them to this id). Falls back to a fresh id if storage is off.
// The id is all that guards those uploads, so it's 128 random bits from the
// crypto API, not Math.random (getRandomValues also works over plain http).
const sessionId = (() => {
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  const fresh = "web-" + Array.from(bytes, b => b.toString(16).padStart(2, "0")).join("");
  try {
    const saved = sessionStorage.getItem("sessionId");
    if (saved) return saved;
    sessionStorage.setItem("sessionId", fresh);
  } catch { /* storage unavailable: a new id per page load */ }
  return fresh;
})();
const $ = (s) => document.querySelector(s);
const messages = $("#messages"), input = $("#input"), sendBtn = $("#send");
const LAYERS = ["intelligence", "inference", "knowledge", "tools"];

// The server's own explanation (e.g. a rate limit's "try again in 40 seconds"), else the status.
async function errorText(res) {
  const body = await res.json().catch(() => ({}));
  return typeof body.detail === "string" ? body.detail : `HTTP ${res.status}`;
}

async function getJSON(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) throw new Error(await errorText(res));
  return res.json();
}

async function loadSidebar() {
  try {
    const [h, n] = await Promise.all([getJSON("/health"), getJSON(`/notes?session_id=${encodeURIComponent(sessionId)}`)]);
    $("#model").textContent = h.model;
    $("#chunks").textContent = h.chunks_indexed;
    $("#tools").textContent = `${h.tools.length} available`;
    $("#tools").title = h.tools.join(", ");
    $("#notes").innerHTML = n.docs.map(d => `<li><code>${esc(d.doc_id)}</code><span class="meta">`
      + (d.uploaded ? `<span class="yours">Yours</span>` : "")
      + `<span class="count">${d.chunks} chunk${d.chunks === 1 ? "" : "s"}</span>`
      + (d.uploaded ? `<button type="button" class="remove" data-doc="${esc(d.doc_id)}" aria-label="Remove ${esc(d.doc_id)}">×</button>` : "")
      + `</span></li>`).join("")
      || "<li class='muted'>No notes ingested yet</li>";
    $("#status-error").hidden = true;
  } catch (e) {
    $("#status-error-text").textContent = `Can't reach the server (${e.message}).`;
    $("#status-error").hidden = false;
  }
}

function addUser(text) {
  const el = document.createElement("div"); el.className = "msg user";
  el.innerHTML = `<div class="bubble"></div>`; el.firstChild.textContent = text;
  messages.appendChild(el); messages.scrollTop = messages.scrollHeight;
}

function addAssistant() {
  const el = document.createElement("div"); el.className = "msg assistant";
  el.innerHTML = `<div class="bubble"><div class="timeline"></div><div class="answer cursor"></div><div class="footer" hidden></div></div>`;
  messages.appendChild(el);
  return { el, timeline: el.querySelector(".timeline"), answer: el.querySelector(".answer"), footer: el.querySelector(".footer"), text: "" };
}

function row(view, layer, html) {
  const r = document.createElement("div"); r.className = "row";
  r.innerHTML = `<span class="tag layer-${layer}">${layer}</span><span>${html}</span>`;
  view.timeline.appendChild(r); messages.scrollTop = messages.scrollHeight;
}
const esc = (s) => String(s).replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

// The answer can repeat text from web pages and notes someone else wrote, so
// render only the tags markdown makes: no raw HTML or scripts, and no images or
// styles, which load a URL on their own and could carry chat contents off-site.
const MARKDOWN_ONLY = {
  ALLOWED_TAGS: ["p", "br", "hr", "h1", "h2", "h3", "h4", "h5", "h6", "em", "strong", "del", "code", "pre",
    "blockquote", "ul", "ol", "li", "a", "table", "thead", "tbody", "tr", "th", "td"],
  ALLOWED_ATTR: ["href", "title", "start", "align"],
};
function renderMarkdown(el, text) {
  if (window.marked && window.DOMPurify) el.innerHTML = DOMPurify.sanitize(marked.parse(text), MARKDOWN_ONLY);
  else el.textContent = text;  // a CDN script didn't load: plain text beats unsafe HTML
}

const handlers = {
  status: (v, d) => row(v, "intelligence", `iteration ${d.iteration}: ${esc(d.text)}`),
  tool_call: (v, d) => row(v, "tools", `call <code>${esc(d.name)}(${esc(JSON.stringify(d.args))})</code>`),
  tool_result: (v, d) => row(v, d.name === "search_notes" || d.name === "save_note" ? "knowledge" : "tools",
    `${d.is_error ? '<span class="err">error</span> ' : ""}${esc(d.name)} → <code>${esc(d.preview)}</code>`),
  token: (v, d) => { v.text += d.text; renderMarkdown(v.answer, v.text); messages.scrollTop = messages.scrollHeight; },
  done: (v, d) => {
    v.answer.classList.remove("cursor");
    if (!v.text) v.answer.innerHTML = "<em class='muted'>(no text answer)</em>";
    const total = Object.values(d.per_layer_ms).reduce((a, b) => a + b, 0) || 1;
    const bar = LAYERS.map(l => `<div class="layer-${l}" style="width:${100 * (d.per_layer_ms[l] || 0) / total}%" title="${l}: ${d.per_layer_ms[l] || 0} ms"></div>`).join("");
    const legend = LAYERS.filter(l => d.per_layer_ms[l]).map(l => `<span><span class="swatch ${l}"></span>${l} ${d.per_layer_ms[l]} ms</span>`).join("");
    v.footer.hidden = false;
    v.footer.innerHTML =
      (d.sources.length ? `<div class="chips">retrieved: ${d.sources.map(s => `<span>${esc(s)}</span>`).join("")}</div>` : "") +
      `<div class="bar">${bar}</div><div class="barlegend">${legend}</div>` +
      `<div>${d.iterations} model call(s) · ${d.total_tokens} tokens · tools: ${d.tools_used.length ? d.tools_used.map(esc).join(", ") : "none"}</div>`;
    if (d.tools_used.includes("save_note")) loadSidebar();
  },
  error: (v, d) => {
    v.answer.classList.remove("cursor");
    let message = d.message;
    if (d.resets_at) {
      const at = new Date(d.resets_at).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
      message += ` That's ${at} your time.`;
    }
    v.answer.innerHTML += `<p class="error">${esc(message)}</p>`;
  },
};

// One answer at a time. A disabled Send button isn't enough: Enter calls
// requestSubmit(), which submits anyway, and two turns at once in one session
// get their history mixed up.
let busy = false;
function setBusy(on) {
  busy = on;
  sendBtn.disabled = on;
  document.querySelectorAll(".suggestions button").forEach(b => { b.disabled = on; });
}

// SSE over a POST body: EventSource can't do that, so parse the stream by hand.
async function chat(text) {
  if (busy) return;
  setBusy(true);
  addUser(text);
  const view = addAssistant();
  try {
    const res = await fetch("/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ session_id: sessionId, message: text, timezone: Intl.DateTimeFormat().resolvedOptions().timeZone }) });
    if (!res.ok) throw new Error(await errorText(res));
    const reader = res.body.getReader(), dec = new TextDecoder();
    let buf = "", finished = false;
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const frame = buf.slice(0, idx); buf = buf.slice(idx + 2);
        let event = "message", data = "";
        for (const line of frame.split("\n")) {
          if (line.startsWith("event: ")) event = line.slice(7);
          else if (line.startsWith("data: ")) data += line.slice(6);
        }
        if (handlers[event]) handlers[event](view, JSON.parse(data));
        if (event === "done" || event === "error") finished = true;
      }
    }
    // A dropped connection or a restarting server ends the stream with no
    // done or error event; without this the cursor just kept blinking.
    if (!finished) handlers.error(view, { message: "The connection closed before the answer finished. Try sending your message again." });
  } catch (e) {
    handlers.error(view, { message: e.message });
  } finally {
    setBusy(false); input.focus();
  }
}

// While an answer streams, Enter leaves the draft in the box rather than dropping it.
$("#composer").addEventListener("submit", (e) => { e.preventDefault(); const t = input.value.trim(); if (!t || busy) return; input.value = ""; input.style.height = "auto"; chat(t); });
input.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#composer").requestSubmit(); } });
input.addEventListener("input", () => { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 160) + "px"; });
// On narrow screens the sidebar is a drawer (see style.css); on wide ones these are no-ops.
const menuBtn = $("#menu");
function setDrawer(open) {
  document.body.classList.toggle("drawer-open", open);
  menuBtn.setAttribute("aria-expanded", String(open));
}
menuBtn.addEventListener("click", () => setDrawer(!document.body.classList.contains("drawer-open")));
$("#backdrop").addEventListener("click", () => setDrawer(false));
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && document.body.classList.contains("drawer-open")) { setDrawer(false); menuBtn.focus(); }
});

document.querySelectorAll(".suggestions button").forEach(b => b.addEventListener("click", () => { setDrawer(false); chat(b.textContent.trim()); }));
$("#newsession").addEventListener("click", async () => {
  setDrawer(false);
  try {
    await getJSON(`/reset/${encodeURIComponent(sessionId)}`, { method: "POST" });
  } catch (err) {
    showNoteStatus(`Couldn't start a new session (${err.message})`, false);
    return;
  }
  messages.innerHTML = "";
  $("#ingest-status").textContent = "";
  await loadSidebar();  // the reset deleted this chat's private notes; stop listing them
});
$("#status-retry").addEventListener("click", loadSidebar);

function showNoteStatus(text, ok) {
  const status = $("#ingest-status");
  status.className = `ingest-status ${ok ? "ok" : "fail"}`;
  status.textContent = text;
}

$("#upload").addEventListener("click", () => $("#upload-input").click());
$("#upload-input").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  e.target.value = "";  // so choosing the same file again still fires "change"
  if (!file) return;
  const btn = $("#upload");
  btn.disabled = true; btn.textContent = "Uploading…";
  try {
    const form = new FormData();
    form.append("session_id", sessionId);
    form.append("file", file);
    const res = await fetch("/notes/upload", { method: "POST", body: form });
    if (!res.ok) throw new Error(await errorText(res));
    const body = await res.json();
    showNoteStatus(`Added ${body.doc_id} (${body.chunks} chunk${body.chunks === 1 ? "" : "s"}), private to this chat`, true);
  } catch (err) {
    showNoteStatus(`Upload failed: ${err.message}`, false);
  } finally {
    btn.disabled = false; btn.textContent = "Upload";
    await loadSidebar();
  }
});

$("#notes").addEventListener("click", async (e) => {
  const btn = e.target.closest(".remove");
  if (!btn) return;
  btn.disabled = true;
  try {
    const r = await getJSON(`/notes/${encodeURIComponent(btn.dataset.doc)}?session_id=${encodeURIComponent(sessionId)}`, { method: "DELETE" });
    // Not there any more, e.g. expired after 24 hours, or removed in another tab.
    showNoteStatus(r.deleted ? `Removed ${btn.dataset.doc}` : `${btn.dataset.doc} had already been removed`, true);
  } catch (err) {
    showNoteStatus(`Couldn't remove it (${err.message})`, false);
  }
  await loadSidebar();
});
$("#reingest").addEventListener("click", async (e) => {
  const btn = e.currentTarget, status = $("#ingest-status");
  btn.disabled = true; btn.textContent = "Re-ingesting…";
  status.className = "ingest-status"; status.textContent = "";
  try {
    const r = await getJSON("/ingest", { method: "POST" });
    const notes = Object.keys(r.ingested).length;
    status.textContent = `Indexed ${notes} note${notes === 1 ? "" : "s"} · ${r.chunks_indexed} chunks`;
    status.classList.add("ok");
  } catch (err) {
    status.textContent = `Re-ingest failed (${err.message})`;
    status.classList.add("fail");
  } finally {
    btn.disabled = false; btn.textContent = "Re-ingest";
    await loadSidebar();
  }
});

// A reload keeps the session id (and its uploads) but not the chat on screen,
// so clear the server-side conversation to match what the page shows.
fetch(`/reset/${encodeURIComponent(sessionId)}?keep_uploads=true`, { method: "POST" }).catch(() => {});
loadSidebar();
