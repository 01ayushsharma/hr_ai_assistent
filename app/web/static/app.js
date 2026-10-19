"use strict";

/* Veltrix PeopleOps web UI.
 * All text from the server (answers, names, reasons) is rendered with textContent,
 * never innerHTML, so nothing in a model answer can inject markup. */

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

function h(tag, props = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(props || {})) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") el.className = value;
    else if (key === "text") el.textContent = value;
    else if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
    else el.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

const storage = {
  get(key) { try { return localStorage.getItem(key); } catch { return null; } },
  set(key, value) { try { localStorage.setItem(key, value); } catch { /* private mode */ } },
  remove(key) { try { localStorage.removeItem(key); } catch { /* ignore */ } },
};

const state = {
  employeeId: storage.get("veltrix.employeeId"),
  me: null,
  docTitles: {},
  sessionId: null,
  view: "chat",
  balances: null,
  sending: false,
};

/* ------------------------------------------------------------------ helpers */

function newId() {
  return (crypto.randomUUID && crypto.randomUUID()) || `s-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

const WEEKDAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
function dateParts(iso) {
  const d = new Date(`${iso}T00:00:00Z`); // calendar dates: no time zones involved
  return { wd: WEEKDAYS[d.getUTCDay()], day: d.getUTCDate(), mon: MONTHS[d.getUTCMonth()], year: d.getUTCFullYear() };
}
const fmtDay = (iso) => { const p = dateParts(iso); return `${p.wd} ${p.day} ${p.mon}`; };
const fmtFull = (iso) => { const p = dateParts(iso); return `${p.wd} ${p.day} ${p.mon} ${p.year}`; };
const fmtShort = (iso) => { const p = dateParts(iso); return `${p.day} ${p.mon} ${p.year}`; };
const fmtRange = (a, b) => (a === b ? fmtFull(a) : `${fmtDay(a)} – ${fmtFull(b)}`);
const days = (n) => { const v = Math.round(Number(n) * 100) / 100; return `${v} day${v === 1 ? "" : "s"}`; };
const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
const addDays = (iso, n) => {
  const d = new Date(`${iso}T00:00:00Z`); d.setUTCDate(d.getUTCDate() + n); return d.toISOString().slice(0, 10);
};

let toastTimer;
function toast(message) {
  const el = $("#toast");
  el.textContent = message;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 4000);
}

class ApiError extends Error {
  constructor(message, status) { super(message); this.status = status; }
}

async function api(path, { method = "GET", body, employeeId = state.employeeId } = {}) {
  const headers = { "X-Employee-Id": employeeId || "" };
  if (body !== undefined) headers["Content-Type"] = "application/json";
  let res;
  try {
    res = await fetch(path, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
  } catch {
    throw new ApiError("Can't reach the server. Check your connection and try again.", 0);
  }
  let data = null;
  try { data = await res.json(); } catch { /* empty body */ }
  if (!res.ok) {
    const detail = data && typeof data.detail === "string" ? data.detail : `Request failed (${res.status})`;
    throw new ApiError(detail, res.status);
  }
  return data;
}

/* ---------------------------------------------------------- start / sign in */

async function waitUntilReady() {
  const banner = $("#startup-banner");
  for (;;) {
    try {
      const res = await fetch("/health");
      if (res.ok) { banner.hidden = true; return; }
    } catch { /* server not up yet */ }
    banner.hidden = false;
    await new Promise((r) => setTimeout(r, 2000));
  }
}

async function signIn(employeeId) {
  const me = await api("/api/me", { employeeId });
  state.employeeId = me.employee_id;
  state.me = me;
  storage.set("veltrix.employeeId", me.employee_id);
  try { state.docTitles = await api("/api/documents"); } catch { state.docTitles = {}; }
  startNewChat();
  renderShell();
  showView(location.hash.slice(1) || "chat");
}

function signOut() {
  storage.remove("veltrix.employeeId");
  state.employeeId = null;
  state.me = null;
  state.balances = null;
  $("#app").hidden = true;
  $("#signin").hidden = false;
  $("#employee-id").value = "";
  $("#employee-id").focus();
}

function renderShell() {
  const me = state.me;
  $("#signin").hidden = true;
  $("#app").hidden = false;
  $("#user-name").textContent = me.full_name;
  $("#user-meta").textContent = `${me.employee_id} · ${cap(me.role)} · ${me.region}`;
  $("#nav-team").hidden = me.role !== "manager";
}

$("#signin-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = $("#employee-id");
  const id = input.value.trim().toUpperCase();
  const error = $("#signin-error");
  error.textContent = "";
  if (!id) { error.textContent = "Enter your employee ID."; input.focus(); return; }
  const button = event.submitter || $("#signin-form button");
  button.disabled = true;
  try {
    await signIn(id);
  } catch (err) {
    error.textContent = err.status === 401 ? "We couldn't find that employee ID." : err.message;
  } finally {
    button.disabled = false;
  }
});

$("#signout").addEventListener("click", signOut);

/* -------------------------------------------------------------- navigation */

function showView(name) {
  const allowed = ["chat", "apply", "leave"].concat(state.me && state.me.role === "manager" ? ["team"] : []);
  if (!allowed.includes(name)) name = "chat";
  state.view = name;
  for (const section of $$(".view")) section.hidden = section.id !== `view-${name}`;
  for (const item of $$(".nav-item")) {
    const active = item.dataset.view === name;
    item.classList.toggle("active", active);
    if (active) {
      item.setAttribute("aria-current", "page");
      item.scrollIntoView({ block: "nearest", inline: "nearest" });
    } else {
      item.removeAttribute("aria-current");
    }
  }
  if (location.hash !== `#${name}`) history.replaceState(null, "", `#${name}`);
  if (name === "chat") $("#chat-input").focus();
  if (name === "apply") openApply();
  if (name === "leave") loadMyLeave();
  if (name === "team") openTeam();
}

for (const item of $$(".nav-item")) item.addEventListener("click", () => showView(item.dataset.view));
document.addEventListener("click", (event) => {
  const target = event.target.closest("[data-goto]");
  if (target) showView(target.dataset.goto);
});
window.addEventListener("hashchange", () => { if (state.me) showView(location.hash.slice(1)); });

/* -------------------------------------------------------------------- chat */

const ROUTE_LABELS = {
  rag: "From HR documents",
  tool: "From your leave records",
  hybrid: "Documents + leave records",
  clarify: "Needs a little more detail",
  no_answer: "Not covered by the documents",
  refuse: "Outside what I can help with",
};
const TOOL_LABELS = {
  get_leave_balance: "Checked leave balance",
  list_leave_requests: "Looked up leave requests",
  get_team_leave: "Checked team leave",
  apply_leave: "Leave request",
  cancel_leave: "Cancellation",
};
const STATUS_LABELS = { success: "done", error: "not possible", pending_confirmation: "awaiting your confirmation" };
const SUGGESTIONS = [
  "How many sick days do I get per year?",
  "What is my annual leave balance?",
  "Which public holidays are left this year?",
  "How much annual leave can I carry forward?",
  "I'd like to take annual leave next week",
];

function startNewChat() {
  state.sessionId = newId();
  const messages = $("#messages");
  messages.replaceChildren(
    h("div", { class: "empty-chat" },
      h("div", { class: "logo", "aria-hidden": "true" }, "V"),
      h("p", { text: `Hi${state.me ? " " + state.me.full_name.split(" ")[0] : ""}! Ask me about leave, public holidays or HR policies. I can also check your balance and request or cancel leave for you.` })),
  );
  renderSuggestions(true);
}

function renderSuggestions(show) {
  const box = $("#suggestions");
  box.replaceChildren(...(show ? SUGGESTIONS.map((text) =>
    h("button", { class: "suggestion", type: "button", onclick: () => sendMessage(text) }, text)) : []));
}

function scrollChat() {
  const box = $("#messages");
  box.scrollTop = box.scrollHeight;
}

function appendUser(text) {
  $(".empty-chat", $("#messages"))?.remove();
  renderSuggestions(false);
  $("#messages").append(h("div", { class: "msg msg-user" }, h("div", { class: "bubble", text })));
  scrollChat();
}

function disableOldConfirmations() {
  for (const row of $$(".confirm-row", $("#messages"))) {
    for (const button of $$("button", row)) button.disabled = true;
  }
}

function appendBot(body) {
  const bubble = h("div", { class: "bubble" });
  const label = body.pending_confirmation ? "Needs your confirmation" : (ROUTE_LABELS[body.route] || body.route);
  bubble.append(h("span", { class: `route route-${body.pending_confirmation ? "clarify" : body.route}`, text: label }));
  bubble.append(h("div", { text: body.answer }));

  if (body.citations && body.citations.length) {
    bubble.append(h("div", { class: "sources" },
      h("div", { class: "sources-title", text: "Sources" }),
      h("ul", {}, body.citations.map((c) =>
        h("li", {}, h("span", { text: `${state.docTitles[c.doc_id] || c.doc_id} — ${c.locator}` }))))));
  }
  if (body.tool_calls && body.tool_calls.length) {
    bubble.append(h("div", { class: "msg-meta" }, body.tool_calls.map((t) =>
      h("span", { class: `chip chip-${t.status}`, title: JSON.stringify(t.args) },
        `${TOOL_LABELS[t.name] || t.name}: ${STATUS_LABELS[t.status] || t.status}`))));
  }
  if (body.pending_confirmation) {
    const row = h("div", { class: "confirm-row" },
      h("button", { class: "btn btn-primary btn-sm", type: "button", onclick: () => sendMessage("Yes, confirm") }, "Confirm"),
      h("button", { class: "btn btn-ghost btn-sm", type: "button",
        onclick: () => sendMessage("No, don't go ahead with that.") }, "Not now"));
    bubble.append(row);
  }
  $("#messages").append(h("div", { class: "msg msg-bot" }, bubble));
  scrollChat();

  const wrote = (body.tool_calls || []).some((t) => ["apply_leave", "cancel_leave"].includes(t.name) && t.status === "success");
  if (wrote) state.balances = null; // refresh on next visit to the leave pages
}

function appendError(message) {
  $("#messages").append(h("div", { class: "msg msg-bot" },
    h("div", { class: "bubble" }, h("div", { class: "callout callout-error", text: message }))));
  scrollChat();
}

async function sendMessage(text) {
  text = (text || "").trim();
  if (!text || state.sending) return;
  state.sending = true;
  $("#send").disabled = true;
  disableOldConfirmations();
  appendUser(text);
  const typing = h("div", { class: "msg msg-bot" },
    h("div", { class: "bubble typing", "aria-label": "Assistant is typing" }, h("span"), h("span"), h("span")));
  $("#messages").append(typing);
  scrollChat();
  try {
    let body;
    try {
      body = await api("/chat", { method: "POST", body: { session_id: state.sessionId, message: text } });
    } catch (err) {
      if (err.status !== 403) throw err;
      state.sessionId = newId(); // session belonged to someone else: start a fresh one
      body = await api("/chat", { method: "POST", body: { session_id: state.sessionId, message: text } });
    }
    typing.remove();
    appendBot(body);
  } catch (err) {
    typing.remove();
    appendError(err.status === 503 ? "The assistant is still starting up. Please try again in a moment." : err.message);
  } finally {
    state.sending = false;
    $("#send").disabled = false;
    $("#chat-input").focus();
  }
}

const chatInput = $("#chat-input");
$("#composer").addEventListener("submit", (event) => {
  event.preventDefault();
  const text = chatInput.value;
  chatInput.value = "";
  autosize();
  sendMessage(text);
});
chatInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    $("#composer").requestSubmit();
  }
});
function autosize() {
  chatInput.style.height = "auto";
  chatInput.style.height = `${Math.min(chatInput.scrollHeight, 160)}px`;
}
chatInput.addEventListener("input", autosize);
$("#new-chat").addEventListener("click", () => { startNewChat(); chatInput.focus(); });

/* --------------------------------------------------------- apply for leave */

const applyForm = $("#apply-form");
const startInput = $("#start-date");
const endInput = $("#end-date");
let previewSeq = 0;
let previewTimer;
let lastPreview = null;
let holidaysLoaded = false;

async function loadBalances(force = false) {
  if (state.balances && !force) return state.balances;
  state.balances = await api("/api/leave/balances");
  return state.balances;
}

async function openApply() {
  const me = state.me;
  const yearEnd = `${me.leave_year}-12-31`;
  for (const input of [startInput, endInput]) { input.min = me.today; input.max = yearEnd; }
  try {
    const balances = await loadBalances();
    for (const el of $$("[data-balance]")) {
      const value = balances[el.dataset.balance];
      el.textContent = value === undefined ? "No balance" : `${days(value)} available`;
    }
  } catch (err) { toast(err.message); }
  if (!holidaysLoaded) loadHolidays();
  if (!lastPreview) renderSummaryEmpty();
}

async function loadHolidays() {
  const list = $("#holiday-list");
  $("#holiday-office").textContent = `· ${state.me.region} office`;
  try {
    const all = await api("/api/holidays");
    const upcoming = all.filter((x) => x.date >= state.me.today);
    holidaysLoaded = true;
    list.replaceChildren(...(upcoming.length ? upcoming.map((x) =>
      h("li", {}, h("span", { class: "hdate", text: fmtDay(x.date) }), h("span", { text: x.name })))
      : [h("li", { class: "muted", text: "No more public holidays this year." })]));
  } catch (err) {
    list.replaceChildren(h("li", { class: "muted", text: err.message }));
  }
}

function formValues() {
  const type = applyForm.querySelector("input[name=leave_type]:checked");
  return { leave_type: type ? type.value : "", start_date: startInput.value, end_date: endInput.value };
}

function renderSummaryEmpty(message) {
  $("#apply-summary").replaceChildren(
    h("h2", { text: "Request summary" }),
    h("p", { class: "summary-empty", text: message || "Choose a leave type and your dates to see how many working days the request uses." }));
}

function renderPreview(values, preview) {
  const box = $("#apply-summary");
  if (!preview.ok) {
    box.replaceChildren(h("h2", { text: "Request summary" }),
      h("div", { class: "callout callout-error", text: preview.reason }),
      h("button", { class: "btn btn-primary btn-block", type: "button", disabled: true }, "Submit request"));
    return;
  }
  const used = Math.max(0, Math.min(1, preview.working_days / Math.max(preview.balance_before, 0.0001)));
  const range = fmtRange(preview.start_date, preview.end_date);
  const submit = h("button", { class: "btn btn-primary btn-block", type: "button" }, "Submit request");
  submit.addEventListener("click", () => submitLeave(values, submit));
  box.replaceChildren(
    h("h2", { text: "Request summary" }),
    h("div", { class: "big-number", text: String(preview.working_days) }),
    h("div", { class: "big-label", text: `working day${preview.working_days === 1 ? "" : "s"} of ${preview.leave_type} leave` }),
    h("dl", { class: "kv" },
      h("dt", { text: "Dates" }), h("dd", { text: range }),
      h("dt", { text: "Calendar days" }), h("dd", { text: String(preview.calendar_days) }),
      h("dt", { text: "Balance now" }), h("dd", { text: days(preview.balance_before) }),
      h("dt", { text: "Balance after" }), h("dd", { text: days(preview.balance_after) })),
    preview.holidays.length ? h("p", { class: "holiday-note",
      text: `Not counted (public holiday): ${preview.holidays.map((x) => `${fmtDay(x.date)} ${x.name}`).join(", ")}` }) : null,
    h("div", { class: "meter", role: "img", "aria-label": `Uses ${Math.round(used * 100)}% of your balance` },
      h("span", { style: `width:${(used * 100).toFixed(1)}%` })),
    h("div", { class: "meter-label" }, h("span", { text: "Uses" }), h("span", { text: `${Math.round(used * 100)}% of balance` })),
    h("div", { class: "callout callout-info", text: "Your manager is asked to approve the request. It shows as pending until then." }),
    submit);
}

function schedulePreview() {
  clearTimeout(previewTimer);
  const values = formValues();
  lastPreview = null;
  if (!values.leave_type || !values.start_date || !values.end_date) { renderSummaryEmpty(); return; }
  previewTimer = setTimeout(async () => {
    const seq = ++previewSeq;
    try {
      const preview = await api("/api/leave/preview", { method: "POST", body: values });
      if (seq !== previewSeq) return; // a newer change is on its way
      lastPreview = preview;
      renderPreview(values, preview);
    } catch (err) {
      if (seq === previewSeq) renderSummaryEmpty(err.message);
    }
  }, 200);
}

startInput.addEventListener("change", () => {
  if (startInput.value && (!endInput.value || endInput.value < startInput.value)) endInput.value = startInput.value;
  endInput.min = startInput.value || state.me.today;
  schedulePreview();
});
endInput.addEventListener("change", schedulePreview);
for (const radio of $$("input[name=leave_type]")) radio.addEventListener("change", schedulePreview);

async function submitLeave(values, button) {
  button.disabled = true;
  button.replaceChildren(h("span", { class: "spinner", "aria-hidden": "true" }), "Submitting…");
  try {
    const result = await api("/api/leave/apply", { method: "POST", body: values });
    state.balances = null;
    lastPreview = null;
    applyForm.reset();
    endInput.min = state.me.today;
    await openApply();
    $("#apply-summary").replaceChildren(
      h("h2", { text: "Request submitted" }),
      h("div", { class: "callout callout-ok",
        text: `Your ${result.leave_type} leave for ${fmtDay(result.start_date)} – ${fmtFull(result.end_date)} (${days(result.working_days)}) is pending approval.` }),
      h("dl", { class: "kv" },
        h("dt", { text: "Request ID" }), h("dd", { class: "mono", text: result.request_id }),
        h("dt", { text: `${cap(result.leave_type)} balance` }), h("dd", { text: days(result.balance_after) })),
      h("button", { class: "btn btn-ghost btn-block", type: "button", "data-goto": "leave" }, "View my leave"));
  } catch (err) {
    button.disabled = false;
    button.textContent = "Submit request";
    $("#apply-summary").prepend(h("div", { class: "callout callout-error", text: err.message }));
  }
}

/* ---------------------------------------------------------------- my leave */

async function loadMyLeave() {
  const grid = $("#balance-grid");
  const box = $("#requests");
  box.replaceChildren(h("div", { class: "empty", text: "Loading…" }));
  try {
    const [balances, requests] = await Promise.all([loadBalances(true), api("/api/leave/requests")]);
    grid.replaceChildren(...Object.entries(balances).map(([type, value]) =>
      h("div", { class: "card balance-card" },
        h("div", { class: "label", text: `${cap(type)} leave` }),
        h("div", { class: "value" }, String(Math.round(value * 100) / 100), h("small", { text: " days left" })))));
    renderRequests(requests);
  } catch (err) {
    box.replaceChildren(h("div", { class: "empty", text: err.message }));
  }
}

function renderRequests(requests) {
  const box = $("#requests");
  if (!requests.length) {
    box.replaceChildren(h("div", { class: "empty" }, "No leave requests yet. ",
      h("a", { href: "#apply", text: "Apply for leave" })));
    return;
  }
  const rows = requests.map((r) => h("tr", {},
    h("td", { "data-label": "Dates", text: fmtRange(r.start_date, r.end_date) }),
    h("td", { "data-label": "Type", text: cap(r.leave_type) }),
    h("td", { "data-label": "Working days", class: "num", text: days(r.working_days) }),
    h("td", { "data-label": "Status" }, h("span", { class: `pill pill-${r.status}`, text: r.status })),
    h("td", { "data-label": "Submitted", class: "hide-sm muted", text: fmtShort(r.created_at.slice(0, 10)) }),
    h("td", { class: "actions" }, r.cancellable ? h("button", { class: "btn btn-ghost btn-sm", type: "button",
      onclick: () => confirmCancel(r) }, "Cancel") : null)));
  box.replaceChildren(h("table", { class: "responsive" },
    h("thead", {}, h("tr", {},
      h("th", { text: "Dates" }), h("th", { text: "Type" }), h("th", { text: "Working days" }),
      h("th", { text: "Status" }), h("th", { class: "hide-sm", text: "Submitted" }), h("th", {}, h("span", { class: "sr-only", text: "Actions" })))),
    h("tbody", {}, rows)));
}

function confirmCancel(request) {
  const dialog = $("#confirm-dialog");
  $("#dialog-title").textContent = "Cancel this leave request?";
  $("#dialog-body").replaceChildren(
    h("p", { text: `${cap(request.leave_type)} leave, ${fmtDay(request.start_date)} – ${fmtFull(request.end_date)} (${request.status}).` }),
    h("p", { class: "muted", text: `${days(request.working_days)} will go back to your ${request.leave_type} balance.` }));
  dialog.returnValue = "";
  dialog.onclose = async () => {
    if (dialog.returnValue !== "confirm") return;
    try {
      const result = await api("/api/leave/cancel", { method: "POST", body: { request_id: request.request_id } });
      state.balances = null;
      toast(`Request cancelled. ${days(result.working_days_returned)} returned to your balance.`);
      loadMyLeave();
    } catch (err) {
      toast(err.message);
    }
  };
  dialog.showModal();
}

/* -------------------------------------------------------------------- team */

function openTeam() {
  const start = $("#team-start");
  const end = $("#team-end");
  if (!start.value) start.value = state.me.today;
  if (!end.value) end.value = addDays(state.me.today, 30);
  loadTeam();
}

async function loadTeam() {
  const box = $("#team-results");
  const start = $("#team-start").value;
  const end = $("#team-end").value;
  box.replaceChildren(h("div", { class: "empty", text: "Loading…" }));
  try {
    const rows = await api(`/api/team/leave?start_date=${encodeURIComponent(start)}&end_date=${encodeURIComponent(end)}`);
    if (!rows.length) {
      box.replaceChildren(h("div", { class: "empty", text: "Nobody in your team has leave in this period." }));
      return;
    }
    box.replaceChildren(h("table", { class: "responsive" },
      h("thead", {}, h("tr", {},
        h("th", { text: "Employee" }), h("th", { text: "Dates" }), h("th", { text: "Type" }),
        h("th", { text: "Working days" }), h("th", { text: "Status" }))),
      h("tbody", {}, rows.map((r) => h("tr", {},
        h("td", { "data-label": "Employee" },
          h("div", {}, h("div", { text: r.full_name }), h("div", { class: "muted mono", text: r.employee_id }))),
        h("td", { "data-label": "Dates", text: fmtRange(r.start_date, r.end_date) }),
        h("td", { "data-label": "Type", text: cap(r.leave_type) }),
        h("td", { "data-label": "Working days", class: "num", text: days(r.working_days) }),
        h("td", { "data-label": "Status" }, h("span", { class: `pill pill-${r.status}`, text: r.status })))))));
  } catch (err) {
    box.replaceChildren(h("div", { class: "empty", text: err.message }));
  }
}

$("#team-form").addEventListener("submit", (event) => { event.preventDefault(); loadTeam(); });

/* -------------------------------------------------------------------- boot */

(async function boot() {
  await waitUntilReady();
  if (state.employeeId) {
    try { await signIn(state.employeeId); return; } catch { storage.remove("veltrix.employeeId"); }
  }
  $("#signin").hidden = false;
  $("#employee-id").focus();
})();
