// Client layer. Knows three things: how to POST /chat, how to parse SSE, and
// how to render the events. It has no idea what a model, tool or vector is.

const sessionId = "web-" + Math.random().toString(36).slice(2, 10);
const $ = (s) => document.querySelector(s);
const messages = $("#messages"), input = $("#input"), sendBtn = $("#send");
const LAYERS = ["intelligence", "inference", "knowledge", "tools"];

async function getJSON(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

async function loadSidebar() {
  try {
    const [h, n] = await Promise.all([getJSON("/health"), getJSON("/notes")]);
    $("#model").textContent = h.model;
    $("#chunks").textContent = h.chunks_indexed;
    $("#tools").textContent = `${h.tools.length} available`;
    $("#tools").title = h.tools.join(", ");
    $("#notes").innerHTML = n.docs.map(d => `<li><code>${esc(d.doc_id)}</code><span class="count">${d.chunks} chunk${d.chunks === 1 ? "" : "s"}</span></li>`).join("")
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

const handlers = {
  status: (v, d) => row(v, "intelligence", `iteration ${d.iteration}: ${esc(d.text)}`),
  tool_call: (v, d) => row(v, "tools", `call <code>${esc(d.name)}(${esc(JSON.stringify(d.args))})</code>`),
  tool_result: (v, d) => row(v, d.name === "search_notes" || d.name === "save_note" ? "knowledge" : "tools",
    `${d.is_error ? '<span class="err">error</span> ' : ""}${esc(d.name)} → <code>${esc(d.preview)}</code>`),
  token: (v, d) => { v.text += d.text; v.answer.innerHTML = marked.parse(v.text); messages.scrollTop = messages.scrollHeight; },
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

// SSE over a POST body: EventSource can't do that, so parse the stream by hand.
async function chat(text) {
  addUser(text);
  const view = addAssistant();
  sendBtn.disabled = true;
  try {
    const res = await fetch("/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ session_id: sessionId, message: text }) });
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
    const reader = res.body.getReader(), dec = new TextDecoder();
    let buf = "";
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
      }
    }
  } catch (e) {
    handlers.error(view, { message: e.message });
  } finally {
    sendBtn.disabled = false; input.focus();
  }
}

$("#composer").addEventListener("submit", (e) => { e.preventDefault(); const t = input.value.trim(); if (!t) return; input.value = ""; input.style.height = "auto"; chat(t); });
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
$("#newsession").addEventListener("click", async () => { setDrawer(false); await fetch(`/reset/${sessionId}`, { method: "POST" }); messages.innerHTML = ""; });
$("#status-retry").addEventListener("click", loadSidebar);
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
loadSidebar();
