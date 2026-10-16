"use strict";

/* Fog control panel, the page of viewer/server.py.

   The switches are built from the library's Effects schema (names, descriptions, groups, prerequisites), and whether
   each one is effective comes from the library's own Effects.missing, through /api/prerequisites, so that switches added
   to the library appear and are greyed out by themselves. Numbers and images are computed by the library through
   /api/compute. Images arrive as linear float16 and are tone-mapped here, so that exposure changes are instant. */

const $ = (selector) => document.querySelector(selector);

function h(tag, attributes = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attributes)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key === "html") node.innerHTML = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) if (child !== null && child !== undefined) node.append(child);
  return node;
}

const escapeHtml = (text) =>
  String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const clamp = (x, low, high) => Math.min(high, Math.max(low, x));

// -------------------------------------------------------------------------------------------------------------------
// Formatting

const DASH = "—";
const pct = (x, digits = 2) => (x === null || x === undefined ? DASH : `${(x * 100).toFixed(digits)}%`);
const ratio = (x) => (x === null || x === undefined ? DASH : x.toFixed(3));
function points(x, digits = 2) {
  if (x === null || x === undefined) return DASH;
  const sign = x > 0.00005 ? "+" : x < -0.00005 ? "−" : "±";
  return `${sign}${Math.abs(x).toFixed(digits)}`;
}
function duration(seconds) {
  if (seconds === null || seconds === undefined) return DASH;
  if (seconds < 0.1) return `${(seconds * 1000).toFixed(1)} ms`;
  if (seconds < 1) return `${(seconds * 1000).toFixed(0)} ms`;
  if (seconds < 60) return `${seconds.toFixed(2)} s`;
  return `${Math.floor(seconds / 60)} min ${Math.round(seconds % 60)} s`;
}
function elapsed(seconds) {
  if (seconds === null || seconds === undefined) return "";
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  return `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, "0")} min`;
}
function milliseconds(x) {
  const digits = Math.abs(x) < 10 ? 1 : 0;
  const text = Math.abs(x).toFixed(digits);
  return `${x < 0 && Number(text) !== 0 ? "−" : ""}${text} ms`;
}
const spp = (x) => (x === null || x === undefined ? DASH : x < 10 ? x.toFixed(1) : x.toFixed(0));
const humanize = (name) => name.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());

// -------------------------------------------------------------------------------------------------------------------
// State

const state = {
  scene: "day",
  frame: 255,
  effects: {},
  inputs: {},
  factor: 1,
  clear: "clean",
  device: null,
  cyclesSpp: 4,
  imageA: "cf",
  imageB: "ref",
  wipe: 0.5,
  exposure: {},
  errorRange: 0.25,
  errorBlur: false,
  scope: "eval",
  window: 9,
  flicker: false,
};

let schema = null;
let switches = {};
let prerequisites = null;
let shown = null; // {config, result} on screen
let pending = null; // response of /api/compute while the current configuration is being computed
let computeError = null;
let ablation = null;
let lastStatus = null;

/* Inputs that the scene can't give (e.g. the albedo of each pixel where it wasn't rendered), with the reason */
const unavailableInputs = (scene = state.scene) => (schema && schema.scenes[scene] && schema.scenes[scene].unavailable_inputs) || {};

/* An input that the scene can't give keeps its own state on the page, as a greyed switch does, and is sent as off */
const givenInputs = () => Object.fromEntries(Object.entries(state.inputs).map(([k, v]) => [k, v && !(k in unavailableInputs())]));

const config = () => ({
  scene: state.scene,
  frame: state.frame,
  effects: { ...state.effects },
  inputs: givenInputs(),
  factor: state.factor,
  clear: state.clear,
  device: state.device,
});

async function getJSON(url) {
  const response = await fetch(url);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw Object.assign(new Error(body.error || body.loading || response.statusText), { status: response.status, body });
  return body;
}

async function postJSON(url, value) {
  const response = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(value) });
  const body = await response.json().catch(() => ({}));
  if (!response.ok && response.status !== 503) throw Object.assign(new Error(body.error || response.statusText), { status: response.status });
  return Object.assign(body, { httpStatus: response.status });
}

// -------------------------------------------------------------------------------------------------------------------
// The address keeps the configuration, so that a view can be reloaded or shared

function saveHash() {
  const p = new URLSearchParams();
  p.set("scene", state.scene);
  p.set("frame", state.frame);
  p.set("factor", state.factor);
  p.set("clear", state.clear);
  if (state.device) p.set("device", state.device);
  const off = Object.keys(state.effects).filter((k) => !state.effects[k]);
  if (off.length) p.set("off", off.join(","));
  const inputsOff = Object.keys(state.inputs).filter((k) => !state.inputs[k] && schema.inputs[k].default);
  if (inputsOff.length) p.set("inputs_off", inputsOff.join(","));
  const inputsOn = Object.keys(state.inputs).filter((k) => state.inputs[k] && !schema.inputs[k].default);
  if (inputsOn.length) p.set("inputs_on", inputsOn.join(","));
  p.set("a", state.imageA);
  p.set("b", state.imageB);
  p.set("spp", state.cyclesSpp);
  if (state.flicker) p.set("flicker", "1");
  history.replaceState(null, "", `#${p.toString()}`);
}

function loadHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  if (p.get("scene") && schema.scenes[p.get("scene")]) state.scene = p.get("scene");
  const scene = schema.scenes[state.scene];
  const frame = Number(p.get("frame"));
  state.frame = scene.frames.includes(frame) ? frame : scene.eval_frames[0];
  if (schema.factors.includes(Number(p.get("factor")))) state.factor = Number(p.get("factor"));
  const clear = p.get("clear");
  state.clear = clear && scene.clear_spp.includes(Number(clear)) ? Number(clear) : "clean";
  if (schema.devices.some((d) => d.name === p.get("device"))) state.device = p.get("device");
  for (const name of (p.get("off") || "").split(",")) if (name in state.effects) state.effects[name] = false;
  for (const name of (p.get("inputs_off") || "").split(",")) if (name in state.inputs) state.inputs[name] = false;
  for (const name of (p.get("inputs_on") || "").split(",")) if (name in state.inputs) state.inputs[name] = true;
  if (KINDS.some((k) => k.id === p.get("a"))) state.imageA = p.get("a");
  if (KINDS.some((k) => k.id === p.get("b"))) state.imageB = p.get("b");
  if (scene.cycles_spp.includes(Number(p.get("spp")))) state.cyclesSpp = Number(p.get("spp"));
  state.flicker = p.get("flicker") === "1";
}

// -------------------------------------------------------------------------------------------------------------------
// Tooltip

const tooltip = $("#tooltip");

function attachTooltip(node, content) {
  const show = () => {
    tooltip.innerHTML = content();
    tooltip.hidden = false;
    const box = node.getBoundingClientRect();
    const tip = tooltip.getBoundingClientRect();
    // Beside the node if there is room, on its right or else its left, otherwise below or above it
    let left = box.right + 8;
    let top = box.top;
    if (left + tip.width > window.innerWidth - 8) left = box.left - tip.width - 8;
    if (left < 8) {
      left = clamp(box.left, 8, window.innerWidth - tip.width - 8);
      top = box.bottom + 6 + tip.height > window.innerHeight - 8 ? box.top - tip.height - 6 : box.bottom + 6;
    }
    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${clamp(top, 8, Math.max(8, window.innerHeight - tip.height - 8))}px`;
  };
  const hide = () => (tooltip.hidden = true);
  node.addEventListener("mouseenter", show);
  node.addEventListener("mouseleave", hide);
  node.addEventListener("focusin", show);
  node.addEventListener("focusout", hide);
}

// -------------------------------------------------------------------------------------------------------------------
// Switches

/* What to do about a condition on the inputs that a switch misses: turn on the page's input that gives it, as the
   server describes each input (its "conditions"), else it is the scene's or its lighting's */
function conditionHint(name) {
  const unavailable = unavailableInputs();
  for (const [input, about] of Object.entries(schema.inputs)) {
    if (!(about.conditions || []).includes(name)) continue;
    if (input in unavailable) return `“${about.label}” isn't available in the ${state.scene} scene`;
    if (!state.inputs[input]) return `turn on “${about.label}”`;
  }
  return `not in the ${state.scene} scene`;
}

function describeMissing(name) {
  if (switches[name]) return `<strong>${escapeHtml(humanize(name))}</strong> is off`;
  const text = schema.conditions[name] || "";
  return `<strong>${escapeHtml(name)}</strong>: ${escapeHtml(text)} <span class="reqs">(${escapeHtml(conditionHint(name))})</span>`;
}

function switchTooltip(sw) {
  const parts = [`<h3>${escapeHtml(sw.name)}</h3>`, `<p>${escapeHtml(sw.description)}</p>`];
  const requires = sw.requires.map((n) => humanize(n));
  const conditions = sw.requires_inputs;
  if (requires.length || conditions.length) {
    const list = [...requires.map((r) => `switch “${escapeHtml(r)}”`), ...conditions.map((c) => escapeHtml(c))];
    parts.push(`<p class="reqs">Requires ${list.join(", ")}.</p>`);
  }
  const missing = prerequisites && prerequisites.available ? prerequisites.missing[sw.name] || [] : [];
  if (missing.length) {
    parts.push(
      `<div class="missing"><strong>Inactive</strong>, computed as off whatever its own state (${state.effects[sw.name] ? "on" : "off"}). Missing:<ul>${missing
        .map((m) => `<li>${describeMissing(m)}</li>`)
        .join("")}</ul></div>`,
    );
  }
  return parts.join("");
}

/* A switch's description and, when it is greyed out, what it misses, as plain text for screen readers
   (aria-describedby): what the tooltip shows on hover and on focus */
function describedText(name) {
  const missing = prerequisites && prerequisites.available ? prerequisites.missing[name] || [] : [];
  const about = switches[name].description;
  if (!missing.length) return about;
  const parts = missing.map((m) => (switches[m] ? `${humanize(m)} is off` : `${m}: ${schema.conditions[m] || ""} (${conditionHint(m)})`));
  return `Inactive, computed as off. Missing: ${parts.join("; ")}. ${about}`;
}

function switchRow(sw) {
  const why = h("span", { class: "visually-hidden", id: `switch-${sw.name}-why`, text: sw.description });
  // Named by its name alone: the label also holds the hidden description, which describes it instead
  const input = h("input", { type: "checkbox", role: "switch", id: `switch-${sw.name}`, "aria-labelledby": `switch-${sw.name}-name`, "aria-describedby": why.id });
  input.checked = state.effects[sw.name];
  // A greyed-out switch stays focusable, so that what it misses can be read from the keyboard, but can't be toggled
  input.addEventListener("click", (event) => {
    if (input.getAttribute("aria-disabled") === "true") event.preventDefault();
  });
  input.addEventListener("change", () => {
    state.effects[sw.name] = input.checked;
    changed();
  });
  const row = h(
    "label",
    { class: "switch", for: input.id, "data-switch": sw.name },
    input,
    h("span", { class: "track" }),
    h("span", { class: "name", id: `switch-${sw.name}-name`, text: humanize(sw.name) }),
    h("span", { class: "state", text: "" }),
    why,
  );
  attachTooltip(row, () => switchTooltip(sw));
  return row;
}

function inputRow(name, about) {
  const why = h("span", { class: "visually-hidden", id: `input-${name}-why`, text: about.description });
  const input = h("input", { type: "checkbox", role: "switch", id: `input-${name}`, "aria-labelledby": `input-${name}-name`, "aria-describedby": why.id });
  input.checked = state.inputs[name];
  input.addEventListener("click", (event) => {
    if (input.getAttribute("aria-disabled") === "true") event.preventDefault();
  });
  input.addEventListener("change", () => {
    state.inputs[name] = input.checked;
    changed();
  });
  const row = h(
    "label",
    { class: "switch input-switch", for: input.id, "data-input": name },
    input,
    h("span", { class: "track" }),
    h("span", { class: "name", id: `input-${name}-name`, text: about.label }),
    h("span", { class: "state", text: "" }),
    why,
  );
  attachTooltip(row, () => {
    const reason = unavailableInputs()[name];
    const missing = reason ? `<div class="missing"><strong>Unavailable</strong> in the ${escapeHtml(state.scene)} scene, given as off: ${escapeHtml(reason)}.</div>` : "";
    return `<h3>${escapeHtml(about.label)}</h3><p>${escapeHtml(about.description)}</p><p class="reqs">An input of the scene, not a switch of Effects: switches that need it are greyed out without it.</p>${missing}`;
  });
  return row;
}

function segmented(id, options, current, onPick, small = false, label = null) {
  const node = h("div", { class: `segmented${small ? " small" : ""}`, role: "radiogroup", id, "aria-label": label });
  for (const option of options) {
    const button = h("button", {
      type: "button",
      role: "radio",
      "aria-checked": String(option.value === current),
      disabled: option.disabled || false,
      title: option.title || null,
      text: option.label,
    });
    button.addEventListener("click", () => {
      for (const b of node.children) b.setAttribute("aria-checked", "false");
      button.setAttribute("aria-checked", "true");
      onPick(option.value);
    });
    node.append(button);
  }
  return node;
}

function groupPanel(title, about, rows, extra = []) {
  return h(
    "section",
    { class: "panel group" },
    h("div", { class: "group-head" }, h("h2", { class: "panel-title", text: title }), h("span", { class: "group-count" })),
    about ? h("p", { class: "group-about", text: about }) : null,
    rows,
    extra,
  );
}

function buildControls() {
  switches = Object.fromEntries(schema.switches.map((s) => [s.name, s]));
  const presets = $("#presets");
  presets.replaceChildren();
  const preset = (label, title, apply) => {
    const button = h("button", { class: "ghost small", type: "button", text: label });
    attachTooltip(button, () => `<p>${escapeHtml(title)}</p>`);
    button.addEventListener("click", () => {
      apply();
      rebuildSwitchStates();
      changed();
    });
    presets.append(button);
  };
  preset("All on", "Every switch on: the complete model.", () => schema.switches.forEach((s) => (state.effects[s.name] = true)));
  preset("All off", "Every switch off: the model's terms that can't be turned off only.", () =>
    schema.switches.forEach((s) => (state.effects[s.name] = false)),
  );
  for (const [name, value] of Object.entries(schema.presets || {})) {
    preset(humanize(name), value.description || `Preset ${name} of the library.`, () => Object.assign(state.effects, value.values));
  }
  preset("Defaults", "Every switch on and every input as by default, as in eval_closed_form.py; factor 1, clean frames without fog.", () => {
    schema.switches.forEach((s) => (state.effects[s.name] = s.default));
    Object.keys(state.inputs).forEach((k) => (state.inputs[k] = schema.inputs[k].default));
    state.factor = 1;
    state.clear = "clean";
    buildControls();
  });

  // Groups of the library, in its order, then the model's inputs and the choices that trade time for error
  const groups = $("#groups");
  groups.replaceChildren();
  const byGroup = {};
  for (const sw of schema.switches) (byGroup[sw.group] ||= []).push(sw);
  const special = new Set(["model", "performance"]);
  if (!schema.has_effects) {
    const text = `The library in ${schema.library} has no Effects switches (visionsim.medium.Effects isn't there): only the inputs below apply. Start the server with --library <a checkout whose library has them> to switch its effects.`;
    groups.append(h("section", { class: "panel" }, h("p", { class: "panel-note", text })));
  }
  for (const group of Object.keys(schema.groups)) {
    if (special.has(group) || !byGroup[group]) continue;
    groups.append(groupPanel(humanize(group), schema.groups[group], byGroup[group].map(switchRow)));
  }
  const inputs = Object.entries(schema.inputs).map(([name, about]) => inputRow(name, about));
  groups.append(groupPanel("Model inputs", "What is given with each frame, or the medium's own scattering.", [...(byGroup.model || []).map(switchRow), ...inputs]));

  const scene = schema.scenes[state.scene];
  const factor = h(
    "div",
    { class: "choice" },
    h("span", { class: "field-label", text: "Resolution of the fog" }),
    segmented("factor", schema.factors.map((f) => ({ value: f, label: f === 1 ? "full" : `1/${f}`, title: f === 1 ? "every pixel" : `every ${f}th pixel along each axis, upsampled` })), state.factor, (v) => {
      state.factor = v;
      changed();
      loadAblation(); // ablations are read at the factor chosen when they have it
    }, true, "Resolution of the fog"),
    h("span", { class: "hint", text: "apply_medium_upsampled: the fog at 1/factor of the resolution, the exact transmittance of each pixel" }),
  );
  const clear = h(
    "div",
    { class: "choice" },
    h("span", { class: "field-label", text: "Frames without fog" }),
    segmented("clear", [{ value: "clean", label: "clean" }, ...scene.clear_spp.map((n) => ({ value: n, label: `${n} spp` }))], state.clear, (v) => {
      state.clear = v;
      changed();
    }, true, "Frames without fog"),
    h("span", { class: "hint", text: "given to the closed form: rendered at many samples, or by Cycles at a few, denoised" }),
  );
  const device = h(
    "div",
    { class: "choice" },
    h("span", { class: "field-label", text: "Device" }),
    segmented("device", schema.devices.map((d) => ({ value: d.name, label: d.name.toUpperCase(), title: d.label })), state.device, (v) => {
      state.device = v;
      changed();
      loadAblation();
    }, true, "Device"),
    h("span", { class: "hint", text: schema.devices.map((d) => `${d.name}: ${d.label}`).join(" · ") }),
  );
  groups.append(groupPanel("Performance", "Choices that trade error for time; inputs of the frame, not terms of the model.", (byGroup.performance || []).map(switchRow), [factor, clear, device]));
  rebuildSwitchStates();
}

function rebuildSwitchStates() {
  for (const row of document.querySelectorAll(".switch[data-switch]")) {
    const name = row.dataset.switch;
    const missing = prerequisites && prerequisites.available ? prerequisites.missing[name] || [] : [];
    const inactive = missing.length > 0;
    const input = row.querySelector("input");
    input.checked = state.effects[name];
    input.setAttribute("aria-disabled", String(inactive));
    row.classList.toggle("inactive", inactive);
    // The toggle keeps showing the switch's own state, muted when it is inactive
    row.querySelector(".state").textContent = inactive ? "inactive" : "";
    row.querySelector(".visually-hidden").textContent = describedText(name);
  }
  for (const row of document.querySelectorAll(".switch[data-input]")) {
    const name = row.dataset.input;
    const reason = unavailableInputs()[name];
    const input = row.querySelector("input");
    input.checked = state.inputs[name];
    input.setAttribute("aria-disabled", String(!!reason));
    row.classList.toggle("inactive", !!reason);
    row.querySelector(".state").textContent = reason ? "unavailable" : "";
    const about = schema.inputs[name].description;
    row.querySelector(".visually-hidden").textContent = reason ? `Unavailable in this scene, given as off: ${reason}. ${about}` : about;
  }
  for (const panel of document.querySelectorAll(".group")) {
    const rows = [...panel.querySelectorAll(".switch[data-switch]")];
    const count = panel.querySelector(".group-count");
    if (!rows.length) {
      count.textContent = "";
      continue;
    }
    const active = rows.filter((r) => !r.classList.contains("inactive") && state.effects[r.dataset.switch]).length;
    count.textContent = `${active}/${rows.length} active`;
  }
  renderAblation();
}

function applyPrerequisites(result) {
  if (!result || result.missing === undefined) return;
  prerequisites = result;
  const unknown = schema.has_effects && !result.available;
  setNotice("prerequisites", unknown ? `Switches aren't greyed out: ${escapeHtml(result.reason || "this library's Effects has no missing() helper")}.` : null);
  rebuildSwitchStates();
}

let prerequisitesSeq = 0;
async function updatePrerequisites() {
  const seq = ++prerequisitesSeq;
  try {
    const result = await postJSON("/api/prerequisites", config());
    if (seq === prerequisitesSeq && result.httpStatus === 200) applyPrerequisites(result);
  } catch (error) {
    if (seq === prerequisitesSeq) setNotice("error", `Prerequisites: ${escapeHtml(error.message)}`);
  }
}

// -------------------------------------------------------------------------------------------------------------------
// Scene and frame

function buildScenePicker() {
  const holder = $("#scene");
  holder.replaceChildren();
  const names = [...new Set(["day", "night", ...Object.keys(schema.scenes)])];
  for (const name of names) {
    const available = !!schema.scenes[name];
    const button = h("button", { type: "button", role: "radio", "aria-checked": String(name === state.scene), disabled: !available, text: humanize(name), title: available ? null : schema.scene_errors[name] || "not loaded" });
    button.addEventListener("click", () => {
      if (state.scene === name) return;
      state.scene = name;
      const scene = schema.scenes[name];
      if (!scene.frames.includes(state.frame)) state.frame = scene.eval_frames[0];
      if (state.clear !== "clean" && !scene.clear_spp.includes(state.clear)) state.clear = "clean";
      if (!scene.cycles_spp.includes(state.cyclesSpp)) state.cyclesSpp = scene.cycles_spp.includes(4) ? 4 : scene.cycles_spp[0];
      for (const b of holder.children) b.setAttribute("aria-checked", String(b === button));
      buildFramePicker();
      buildControls();
      buildImagePickers();
      loadAblation();
      changed();
    });
    holder.append(button);
  }
  buildFramePicker();
}

function buildFramePicker() {
  const select = $("#frame");
  const scene = schema.scenes[state.scene];
  select.replaceChildren(
    ...scene.frames.map((f) => h("option", { value: f, text: `${f}${scene.eval_frames.includes(f) ? " · evaluated" : ""}` })),
  );
  select.value = state.frame;
}

// -------------------------------------------------------------------------------------------------------------------
// Requests

let computeSeq = 0;
let computeTimer = null;
let computing = false; // a configuration other than the one shown is asked for

function changed() {
  saveHash();
  // Polls still running for the previous configuration are ignored from now on, so that they can't bring back its
  // prerequisites during the debounce
  computeSeq++;
  updatePrerequisites();
  clearTimeout(computeTimer);
  computeTimer = setTimeout(() => {
    computeTimer = null;
    compute();
  }, 120);
  renderStale();
}

async function compute(retry = false) {
  const seq = ++computeSeq;
  const body = { ...config(), supersede: true, flicker: state.flicker, retry };
  computeError = null;
  computing = true;
  for (;;) {
    let response;
    try {
      response = await postJSON("/api/compute", body);
    } catch (error) {
      if (seq !== computeSeq) return;
      computeError = error.message;
      pending = null;
      computing = false;
      renderOverlay();
      return;
    }
    if (seq !== computeSeq) return;
    body.retry = false;
    if (response.prerequisites) applyPrerequisites(response.prerequisites);
    if (response.state === "done") {
      pending = null;
      computing = false;
      show(response);
      renderOverlay();
      return;
    }
    if (response.state === "error") {
      pending = null;
      computing = false;
      computeError = response.error;
      renderOverlay();
      return;
    }
    pending = response;
    renderOverlay();
    await sleep(response.state === "loading" ? 800 : 350);
  }
}

function show(response) {
  const sameImages = shown && shown.result.id === response.result.id;
  shown = { config: response.config, result: response.result };
  checkAblation(response.result);
  buildImagePickers();
  renderMetrics();
  if (!sameImages) loadView();
  else renderView();
}

// -------------------------------------------------------------------------------------------------------------------
// Status and notices

const notices = {};

function setNotice(key, html) {
  notices[key] = html;
  const node = $("#notice");
  const entries = Object.entries(notices).filter(([, v]) => v);
  node.hidden = !entries.length;
  node.classList.toggle("error", entries.some(([k]) => k === "error"));
  node.innerHTML = entries.map(([, v]) => `<div>${v}</div>`).join("");
  if (notices.error) {
    const button = h("button", { class: "ghost small", type: "button", text: "Dismiss" });
    button.addEventListener("click", () => setNotice("error", null));
    node.append(button);
  }
}

async function pollStatus() {
  const busy = $("#busy");
  const text = busy.querySelector(".busy-text");
  try {
    const status = await getJSON("/api/status");
    lastStatus = status;
    busy.classList.remove("offline");
    if (schema && status.instance && status.instance !== schema.instance) {
      // Another run of the server, maybe with another library and other switches: start again from the address,
      // which keeps the configuration, rather than mix its numbers with images and switches of the previous run
      text.textContent = "The server restarted: reloading…";
      saveHash();
      location.reload();
      return;
    }
    if (status.loading) {
      busy.classList.add("working");
      text.textContent = `Loading: ${status.loading} · ${elapsed(status.uptime)}`;
    } else if (status.running) {
      const r = status.running;
      busy.classList.add("working");
      const queued = status.queue.length ? ` · ${status.queue.length} queued` : "";
      text.textContent = `${r.origin === "warmup" ? "Warm-up" : "Computing"} · ${r.scene} · ${r.stage.split(";")[0]} · ${elapsed(r.elapsed)}${queued}`;
      busy.title = `${r.scene}, frames ${r.frames.join(", ")}, ${r.device}, factor ${r.factor}: ${r.stage}`;
    } else {
      busy.classList.remove("working");
      const device = schema ? schema.devices.find((d) => d.name === (state.device || status.device)) : null;
      text.textContent = `Idle · ${device ? device.label : status.device} · ${status.cached_frames} frames in memory`;
      busy.title = "";
    }
    renderWarmup(status);
    if (pending) renderOverlay();
  } catch (error) {
    busy.classList.remove("working");
    busy.classList.add("offline");
    text.textContent = "Server unreachable";
  }
}

function renderWarmup(status) {
  const entries = Object.entries(status.warmup || {});
  if (!entries.length || entries.every(([, w]) => w.state === "done")) {
    setNotice("warmup", null);
    return;
  }
  const steps = entries
    .map(([scene, w]) => {
      if (w.state === "done") return `<span class="step-done">${scene} ✓ ${elapsed(w.seconds)}</span>`;
      if (w.state === "error") return `<span>${scene}: failed, ${escapeHtml(w.error || "")}</span>`;
      if (w.state === "running") return `<span>${scene} · computing ${elapsed(w.elapsed)}</span>`;
      return `<span>${scene} · queued</span>`;
    })
    .join("");
  const device = w0Device(entries);
  setNotice(
    "warmup",
    `<strong>Warming up</strong> the default configuration of both scenes in the background, on ${escapeHtml(device)}: <span class="steps">${steps}</span><br>The night scene's first call computes the halo of its lamp and the light that surfaces reflect, once: about 1.5–2.5 min on 4 CPU threads, 6–15 s on the GPU. Your own requests go first.`,
  );
}
const w0Device = (entries) => (entries[0] && entries[0][1].device) || "the server's device";

function renderOverlay() {
  renderStale();
  const stale = $("#stale");
  const empty = $("#empty");
  empty.classList.remove("error");
  if (computeError) {
    stale.hidden = true;
    empty.hidden = false;
    empty.classList.add("error");
    empty.replaceChildren(h("span", { text: `Error: ${computeError}` }), " ", retryButton());
    return;
  }
  if (!pending) {
    stale.hidden = true;
    empty.hidden = !!shown;
    if (!shown) empty.replaceChildren(h("span", { class: "led" }), h("span", { text: "Waiting for the server…" }));
    return;
  }
  let message;
  if (pending.state === "loading") message = `Loading: ${pending.stage}`;
  else if (pending.state === "queued") {
    const running = lastStatus && lastStatus.running;
    const behind = running ? ` behind ${running.origin === "warmup" ? "the warm-up of" : "a request for"} ${running.scene} (${elapsed(running.elapsed)})` : "";
    message = `Queued${behind}${pending.ahead > 1 ? `, ${pending.ahead} jobs ahead` : ""} · waited ${elapsed(pending.job.waited)}`;
  } else {
    const job = pending.job;
    message = `Computing · ${job.stage} · ${elapsed(job.elapsed)}`;
  }
  const target = shown ? stale : empty;
  (shown ? empty : stale).hidden = true;
  target.hidden = false;
  target.replaceChildren(h("span", { class: "led" }), h("span", { text: shown ? `${message} · showing the previous result` : message }));
  target.title = message;
}

function renderStale() {
  const panel = $("#metrics");
  const note = $("#metrics-stale");
  const waiting = !!shown && (computing || computeError !== null || computeTimer !== null);
  panel.classList.toggle("is-stale", waiting);
  note.hidden = !waiting;
  if (!waiting) return;
  note.textContent = computeError
    ? "These numbers are those of the previous configuration: the current one failed (see the image)."
    : "These numbers are those of the previous configuration, until the current one is computed.";
}

function retryButton() {
  const button = h("button", { class: "ghost small", type: "button", text: "Retry" });
  button.addEventListener("click", () => compute(true));
  return button;
}

// -------------------------------------------------------------------------------------------------------------------
// Images

const HALF = (() => {
  const table = new Float32Array(65536);
  for (let bits = 0; bits < 65536; bits++) {
    const sign = bits & 0x8000 ? -1 : 1;
    const exponent = (bits >> 10) & 0x1f;
    const fraction = bits & 0x3ff;
    if (exponent === 0) table[bits] = sign * fraction * 2 ** -24;
    else if (exponent === 31) table[bits] = fraction ? NaN : sign * Infinity;
    else table[bits] = sign * (1 + fraction / 1024) * 2 ** (exponent - 15);
  }
  return table;
})();

const SRGB = (() => {
  const table = new Uint8ClampedArray(4097);
  for (let i = 0; i <= 4096; i++) {
    const v = i / 4096;
    table[i] = Math.round(255 * (v <= 0.0031308 ? 12.92 * v : 1.055 * v ** (1 / 2.4) - 0.055));
  }
  return table;
})();

const imageCache = new Map();

async function fetchFloat(url) {
  const response = await fetch(url);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.error || response.statusText);
  }
  const buffer = await response.arrayBuffer();
  const halves = new Uint16Array(buffer);
  const data = new Float32Array(halves.length);
  for (let i = 0; i < halves.length; i++) data[i] = HALF[halves[i]];
  return { w: Number(response.headers.get("X-Width")), h: Number(response.headers.get("X-Height")), c: Number(response.headers.get("X-Channels")), data };
}

function loadFloat(url) {
  if (!imageCache.has(url)) {
    const promise = fetchFloat(url);
    promise.catch(() => setTimeout(() => imageCache.delete(url), 5000));
    imageCache.set(url, promise);
    while (imageCache.size > 96) imageCache.delete(imageCache.keys().next().value);
  }
  return imageCache.get(url);
}

/* Images on screen belong to the result shown, so that the closed form and Cycles' images always match */
const viewScene = () => (shown ? shown.config.scene : state.scene);
const viewFrame = () => (shown ? shown.config.frame : state.frame);
/* Addresses of images name the server's run (static ones by v=, the closed form's by an id salted with it), so that the
   browser never shows an image that a server started earlier, maybe with another library, made for the same view */
const staticUrl = (name, count) =>
  `/api/static?scene=${viewScene()}&frame=${viewFrame()}&name=${name}${count !== undefined ? `&spp=${count}` : ""}&v=${schema.instance}`;
const closedUrl = (name) => (shown ? `/api/image?id=${shown.result.id}&name=${name}` : null);

const KINDS = [
  { id: "cf", group: "Closed form", label: () => "Closed form · whole", url: () => closedUrl("radiance") },
  { id: "cf_fog", group: "Closed form", label: () => "Closed form · fog (in-scattering)", url: () => closedUrl("inscatter") },
  { id: "cf_surfaces", group: "Closed form", label: () => "Closed form · surfaces", url: () => closedUrl("surfaces") },
  { id: "input", group: "Closed form", label: () => `Input · frame without fog (${shown ? (shown.config.clear === "clean" ? "clean" : `${shown.config.clear} spp`) : state.clear})`, url: () => staticUrl("input", shown ? shown.config.clear : state.clear) },
  { id: "ref", group: "Cycles reference", label: () => "Reference · whole", url: () => staticUrl("reference") },
  { id: "ref_fog", group: "Cycles reference", label: () => "Reference · fog (volume passes)", url: () => staticUrl("reference_fog") },
  { id: "ref_surfaces", group: "Cycles reference", label: () => "Reference · surfaces", url: () => staticUrl("reference_surfaces") },
  { id: "cyc", group: "Cycles at a few samples", label: () => `Cycles ${state.cyclesSpp} spp + OIDN`, url: () => staticUrl("cycles", state.cyclesSpp) },
  { id: "cyc_noisy", group: "Cycles at a few samples", label: () => `Cycles ${state.cyclesSpp} spp · noisy`, url: () => staticUrl("cycles_noisy", state.cyclesSpp) },
  { id: "cyc_fog", group: "Cycles at a few samples", label: () => `Cycles ${state.cyclesSpp} spp · fog (volume passes)`, url: () => staticUrl("cycles_fog", state.cyclesSpp) },
  { id: "err", group: "Signed relative error", label: () => "Error · closed form vs reference", error: ["cf", "ref"] },
  { id: "err_fog", group: "Signed relative error", label: () => "Error · fog vs reference fog", error: ["cf_fog", "ref_fog"] },
  { id: "err_surfaces", group: "Signed relative error", label: () => "Error · surfaces vs reference", error: ["cf_surfaces", "ref_surfaces"] },
  { id: "err_cyc", group: "Signed relative error", label: () => `Error · Cycles ${state.cyclesSpp} spp vs reference`, error: ["cyc", "ref"] },
];
/* The albedo of each pixel, for a library that takes it (added once the schema is known) */
const ALBEDO_KIND = { id: "albedo", group: "Closed form", label: () => "Input · albedo of each pixel (diffuse color pass)", url: () => staticUrl("albedo") };
const kind = (id) => KINDS.find((k) => k.id === id);

function buildImagePickers() {
  for (const [id, key] of [["#image-a", "imageA"], ["#image-b", "imageB"]]) {
    const select = $(id);
    const groups = {};
    for (const k of KINDS) (groups[k.group] ||= []).push(k);
    select.replaceChildren(
      ...Object.entries(groups).map(([group, kinds]) => h("optgroup", { label: group }, kinds.map((k) => h("option", { value: k.id, text: k.label() })))),
    );
    select.value = state[key];
  }
  const spp = $("#cycles-spp");
  const scene = schema.scenes[state.scene];
  spp.replaceChildren(...scene.cycles_spp.map((n) => h("option", { value: n, text: `${n} spp` })));
  spp.value = state.cyclesSpp;
}

const view = { a: null, b: null, token: 0, w: 320, h: 180, cursor: null, dragging: false };
const errorCache = new Map();

function luminance(image) {
  const n = image.w * image.h;
  const y = new Float32Array(n);
  const d = image.data;
  for (let i = 0, j = 0; i < n; i++, j += image.c) y[i] = image.c >= 3 ? 0.2126 * d[j] + 0.7152 * d[j + 1] + 0.0722 * d[j + 2] : d[j];
  return y;
}

/* Box blur of 5x5 with zeros beyond the edges, as common.blur */
function blur5(values, w, hgt) {
  const temp = new Float32Array(values.length);
  const out = new Float32Array(values.length);
  for (let y = 0; y < hgt; y++)
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let k = -2; k <= 2; k++) if (x + k >= 0 && x + k < w) s += values[y * w + x + k];
      temp[y * w + x] = s / 5;
    }
  for (let y = 0; y < hgt; y++)
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let k = -2; k <= 2; k++) if (y + k >= 0 && y + k < hgt) s += temp[(y + k) * w + x];
      out[y * w + x] = s / 5;
    }
  return out;
}

/* Signed relative difference of luminance, (A - B) / B, with B kept above 2% of its mean where it is nearly black */
function errorMap(a, b, blurred) {
  let ya = luminance(a);
  let yb = luminance(b);
  if (blurred) {
    ya = blur5(ya, a.w, a.h);
    yb = blur5(yb, b.w, b.h);
  }
  let mean = 0;
  for (const v of yb) mean += Math.max(v, 0);
  mean /= yb.length;
  const floor = Math.max(0.02 * mean, 1e-12);
  const out = new Float32Array(ya.length);
  for (let i = 0; i < out.length; i++) out[i] = (ya[i] - yb[i]) / Math.max(yb[i], floor);
  return { w: a.w, h: a.h, c: 1, data: out, error: true };
}

async function layer(id) {
  const k = kind(id);
  if (k.error) {
    const [ka, kb] = k.error.map(kind);
    const [ua, ub] = [ka.url(), kb.url()];
    if (!ua || !ub) return { id, missing: "computing" };
    const key = `${ua}|${ub}|${state.errorBlur}`;
    if (!errorCache.has(key)) {
      try {
        const [a, b] = await Promise.all([loadFloat(ua), loadFloat(ub)]);
        errorCache.set(key, errorMap(a, b, state.errorBlur));
        while (errorCache.size > 24) errorCache.delete(errorCache.keys().next().value);
      } catch (error) {
        return { id, missing: error.message };
      }
    }
    return { id, image: errorCache.get(key) };
  }
  const url = k.url();
  if (!url) return { id, missing: "computing" };
  try {
    return { id, image: await loadFloat(url) };
  } catch (error) {
    return { id, missing: error.message };
  }
}

/* The closed form and the reference, for the numbers under the cursor */
const probe = { cf: null, cf_fog: null, ref: null, ref_fog: null };

async function loadView() {
  const token = ++view.token;
  const ids = ["cf", "cf_fog", "ref", "ref_fog"];
  const [a, b, ...probes] = await Promise.all([layer(state.imageA), layer(state.imageB), ...ids.map(layer)]);
  if (token !== view.token) return;
  view.a = a;
  view.b = b;
  ids.forEach((id, i) => (probe[id] = probes[i].image || null));
  if (state.exposure[viewScene()] === undefined && probe.ref) autoExposure(probe.ref);
  paint();
  renderCursor();
}

function renderView() {
  loadView();
}

function autoExposure(image) {
  const y = luminance(image);
  const sorted = Float32Array.from(y).sort();
  const p = sorted[Math.floor(0.97 * (sorted.length - 1))] || 1;
  const ev = clamp(Math.round(Math.log2(0.9 / Math.max(p, 1e-8)) * 4) / 4, -6, 10);
  state.exposure[viewScene()] = ev;
  syncExposure();
}

function syncExposure() {
  const ev = state.exposure[viewScene()] ?? 0;
  $("#exposure").value = ev;
  $("#exposure-value").textContent = `${ev >= 0 ? "+" : "−"}${Math.abs(ev).toFixed(2)} EV`;
}

let colormap = null;

/* Diverging blue (too dark) to red (too bright) through a neutral gray, stepped for the theme's surface */
function buildColormap() {
  const dark = document.documentElement.dataset.theme === "dark" || (document.documentElement.dataset.theme !== "light" && matchMedia("(prefers-color-scheme: dark)").matches);
  const stops = dark
    ? [[-1, "#9ec5f4"], [-0.5, "#3987e5"], [0, "#2c2c2a"], [0.5, "#e05656"], [1, "#f6b1ae"]]
    : [[-1, "#104281"], [-0.5, "#5598e7"], [0, "#f0efec"], [0.5, "#ec7a76"], [1, "#9e2222"]];
  const rgb = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
  colormap = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = (i / 255) * 2 - 1;
    let j = 0;
    while (j < stops.length - 2 && t > stops[j + 1][0]) j++;
    const [t0, c0] = stops[j];
    const [t1, c1] = stops[j + 1];
    const f = clamp((t - t0) / (t1 - t0), 0, 1);
    const [a, b] = [rgb(c0), rgb(c1)];
    for (let c = 0; c < 3; c++) colormap[i * 3 + c] = a[c] + (b[c] - a[c]) * f;
  }
  const ramp = $("#ramp");
  const css = [];
  for (let i = 0; i < 256; i += 15) css.push(`rgb(${colormap[i * 3]},${colormap[i * 3 + 1]},${colormap[i * 3 + 2]})`);
  ramp.style.background = `linear-gradient(90deg, ${css.join(",")})`;
}

const canvas = $("#canvas");
const context = canvas.getContext("2d");

function paint() {
  const a = view.a && view.a.image;
  const b = view.b && view.b.image;
  const w = (a || b || { w: view.w }).w;
  const hgt = (a || b || { h: view.h }).h;
  if (canvas.width !== w || canvas.height !== hgt) {
    canvas.width = w;
    canvas.height = hgt;
  }
  view.w = w;
  view.h = hgt;
  const pixels = context.createImageData(w, hgt);
  const out = pixels.data;
  const split = Math.round(state.wipe * w);
  const gain = 2 ** (state.exposure[viewScene()] ?? 0);
  const range = state.errorRange;
  for (const [image, x0, x1] of [[a, 0, split], [b, split, w]]) {
    for (let y = 0; y < hgt; y++) {
      for (let x = x0; x < x1; x++) {
        const o = (y * w + x) * 4;
        out[o + 3] = 255;
        if (!image || image.w !== w || image.h !== hgt) {
          const checker = ((x >> 3) + (y >> 3)) & 1 ? 46 : 38;
          out[o] = out[o + 1] = out[o + 2] = checker;
          continue;
        }
        const i = y * w + x;
        if (image.error) {
          const t = clamp(image.data[i] / range, -1, 1);
          const k = Math.round((t + 1) * 127.5) * 3;
          out[o] = colormap[k];
          out[o + 1] = colormap[k + 1];
          out[o + 2] = colormap[k + 2];
        } else {
          const j = i * image.c;
          for (let c = 0; c < 3; c++) {
            const v = image.data[j + Math.min(c, image.c - 1)] * gain;
            out[o + c] = v > 0 ? (v >= 1 ? 255 : SRGB[(v * 4096) | 0]) : 0;
          }
        }
      }
    }
  }
  context.putImageData(pixels, 0, 0);
  $("#wipe-line").style.left = `${state.wipe * 100}%`;
  const label = (layerOf, id) => {
    const k = kind(id);
    const missing = layerOf && layerOf.missing && layerOf.missing !== "computing" ? ` — ${layerOf.missing}` : layerOf && layerOf.missing ? " — computing" : "";
    return `${k.label()}${missing}`;
  };
  $("#tag-a").textContent = `A · ${label(view.a, state.imageA)}`;
  $("#tag-b").textContent = `B · ${label(view.b, state.imageB)}`;
  $("#tag-a").title = $("#tag-a").textContent;
  $("#tag-b").title = $("#tag-b").textContent;
  const errorShown = (a && a.error) || (b && b.error);
  $("#error-legend").hidden = !errorShown;
  $("#legend-note").textContent = `±${Math.round(range * 100)}% of the reference's luminance${state.errorBlur ? ", both blurred 5×5" : ""}`;
}

function setupViewer() {
  const viewer = $("#viewer");
  const position = (event) => {
    const box = canvas.getBoundingClientRect();
    return { fx: clamp((event.clientX - box.left) / box.width, 0, 1), fy: clamp((event.clientY - box.top) / box.height, 0, 1) };
  };
  viewer.addEventListener("pointerdown", (event) => {
    view.dragging = true;
    viewer.setPointerCapture(event.pointerId);
    state.wipe = position(event).fx;
    $("#wipe").value = state.wipe;
    paint();
  });
  viewer.addEventListener("pointermove", (event) => {
    const { fx, fy } = position(event);
    if (view.dragging) {
      state.wipe = fx;
      $("#wipe").value = fx;
      paint();
    }
    view.cursor = { x: Math.min(view.w - 1, Math.floor(fx * view.w)), y: Math.min(view.h - 1, Math.floor(fy * view.h)) };
    renderCursor();
  });
  const release = () => (view.dragging = false);
  viewer.addEventListener("pointerup", release);
  viewer.addEventListener("pointercancel", release);
  viewer.addEventListener("pointerleave", () => {
    view.cursor = null;
    renderCursor();
  });
  $("#wipe").addEventListener("input", (e) => {
    state.wipe = Number(e.target.value);
    paint();
  });
  $("#exposure").addEventListener("input", (e) => {
    state.exposure[viewScene()] = Number(e.target.value);
    syncExposure();
    paint();
  });
  $("#auto-exposure").addEventListener("click", () => {
    const a = view.a && view.a.image;
    autoExposure(a && !a.error ? a : probe.ref || a);
    paint();
  });
  $("#error-range").addEventListener("input", (e) => {
    state.errorRange = Number(e.target.value);
    $("#error-range-value").textContent = `±${Math.round(state.errorRange * 100)}%`;
    paint();
  });
  $("#error-blur").addEventListener("change", (e) => {
    state.errorBlur = e.target.checked;
    loadView();
  });
  for (const [id, key] of [["#image-a", "imageA"], ["#image-b", "imageB"]]) {
    $(id).addEventListener("change", (e) => {
      state[key] = e.target.value;
      saveHash();
      loadView();
    });
  }
  $("#swap").addEventListener("click", () => {
    [state.imageA, state.imageB] = [state.imageB, state.imageA];
    $("#image-a").value = state.imageA;
    $("#image-b").value = state.imageB;
    saveHash();
    loadView();
  });
  $("#cycles-spp").addEventListener("change", (e) => {
    state.cyclesSpp = Number(e.target.value);
    buildImagePickers();
    saveHash();
    loadView();
    renderMetrics();
  });
  $("#frame").addEventListener("change", (e) => {
    state.frame = Number(e.target.value);
    changed();
  });
  $("#window").addEventListener("change", (e) => {
    state.window = Number(e.target.value);
    renderCursor();
  });
}

// -------------------------------------------------------------------------------------------------------------------
// Under the cursor

function windowErrors(ours, theirs, minus, minusTheirs, x0, y0, k) {
  let error = 0;
  let total = 0;
  let sum = 0;
  const { w, c } = ours;
  const half = Math.floor(k / 2);
  for (let y = Math.max(0, y0 - half); y <= Math.min(ours.h - 1, y0 + half); y++) {
    for (let x = Math.max(0, x0 - half); x <= Math.min(w - 1, x0 + half); x++) {
      for (let ch = 0; ch < Math.min(3, c); ch++) {
        const i = (y * w + x) * c + ch;
        const o = ours.data[i] - (minus ? minus.data[i] : 0);
        const t = theirs.data[i] - (minusTheirs ? minusTheirs.data[i] : 0);
        error += Math.abs(o - t);
        total += t;
        sum += o;
      }
    }
  }
  return total > 0 ? { rel_l1: error / total, ratio: sum / total } : null;
}

function pixelText(layerOf, x, y) {
  const image = layerOf && layerOf.image;
  if (!image) return DASH;
  const i = y * image.w + x;
  if (image.error) return `${(image.data[i] * 100).toFixed(1)}%`;
  const j = i * image.c;
  const values = [0, 1, 2].map((c) => image.data[j + Math.min(c, image.c - 1)]);
  return values.map((v) => (Math.abs(v) >= 100 ? v.toFixed(1) : Math.abs(v) >= 1 ? v.toFixed(3) : v.toPrecision(3))).join(" ");
}

function renderCursor() {
  const table = $("#cursor");
  const crosshair = $("#crosshair");
  const readout = $("#readout");
  const cursor = view.cursor;
  const head = "<tr><th></th><th class=num>rel L1</th><th class=num>ratio</th></tr>";
  if (!cursor || !probe.cf || !probe.ref) {
    crosshair.hidden = true;
    readout.textContent = "Move over the image to read pixels; drag to move the wipe.";
    table.innerHTML = `${head}${["Whole", "Fog", "Surfaces"].map((n) => `<tr><td>${n}</td><td class="muted">${DASH}</td><td class="muted">${DASH}</td></tr>`).join("")}`;
    return;
  }
  const { x, y } = cursor;
  const k = state.window;
  crosshair.hidden = false;
  const half = Math.floor(k / 2);
  crosshair.style.left = `${((x - half) / view.w) * 100}%`;
  crosshair.style.top = `${((y - half) / view.h) * 100}%`;
  crosshair.style.width = `${(k / view.w) * 100}%`;
  crosshair.style.height = `${(k / view.h) * 100}%`;
  readout.textContent = `x ${x}, y ${y} · A ${pixelText(view.a, x, y)} · B ${pixelText(view.b, x, y)}`;
  const rows = [
    ["Whole", windowErrors(probe.cf, probe.ref, null, null, x, y, k)],
    ["Fog", probe.cf_fog && probe.ref_fog ? windowErrors(probe.cf_fog, probe.ref_fog, null, null, x, y, k) : null],
    ["Surfaces", probe.cf_fog && probe.ref_fog ? windowErrors(probe.cf, probe.ref, probe.cf_fog, probe.ref_fog, x, y, k) : null],
  ];
  table.innerHTML = head + rows.map(([n, e]) => `<tr><td>${n}</td><td>${e ? pct(e.rel_l1, 1) : DASH}</td><td>${e ? ratio(e.ratio) : DASH}</td></tr>`).join("");
}

// -------------------------------------------------------------------------------------------------------------------
// Numbers

function renderMetrics() {
  if (!shown) return;
  const { result, config: c } = shown;
  const scene = schema.scenes[c.scene];
  const scopeHolder = $("#scope");
  const scopes = [
    { value: "eval", label: `${scene.eval_frames.join(" + ")}` },
    { value: "frame", label: `frame ${c.frame}` },
  ];
  scopeHolder.replaceWith(
    segmented("scope", scopes, state.scope, (v) => {
      state.scope = v;
      renderMetrics();
    }, true, "Frames measured"),
  );
  const metrics = result.metrics[state.scope];
  const baseline = state.scope === "eval" ? result.baseline : null;
  const rows = [
    ["Whole frame", metrics.whole, baseline && baseline.whole],
    ["Fog", metrics.fog, baseline && baseline.fog],
    ["Surfaces", metrics.surfaces, baseline && baseline.surfaces],
  ];
  let html = `<tr><th>vs reference</th><th class=num>rel L1</th><th class=num>blurred</th><th class=num>ratio</th><th class=num title="Points of rel L1 against every switch and input on, factor 1, clean frames">Δ default</th></tr>`;
  for (const [name, e, b] of rows) {
    const delta = b ? (e.rel_l1 - b.rel_l1) * 100 : null;
    const cls = delta === null ? "muted" : delta > 0.005 ? "up" : delta < -0.005 ? "down" : "";
    html += `<tr class="${name === "Whole frame" ? "lead" : ""}"><td>${name}</td><td>${pct(e.rel_l1)}</td><td>${pct(e.rel_l1_blurred)}</td><td>${ratio(e.ratio)}</td><td class="${cls}">${points(delta)}</td></tr>`;
  }
  const curve = result.cycles && result.cycles.available ? result.cycles.curve : [];
  const cyc = curve.find((r) => r.spp === state.cyclesSpp && r.denoised);
  if (state.scope === "eval" && cyc) {
    html += `<tr class="inactive"><td>Cycles ${state.cyclesSpp} spp + OIDN</td><td>${pct(cyc.denoised.rel_l1)}</td><td>${pct(cyc.denoised.rel_l1_blurred)}</td><td>${ratio(cyc.denoised.ratio)}</td><td></td></tr>`;
  }
  $("#errors").innerHTML = html;
  const reference = escapeHtml(scene.reference);
  $("#errors-note").innerHTML =
    state.scope === "eval"
      ? `Frames ${scene.eval_frames.join(" and ")} together, as eval_closed_form.py, against ${reference}. Blurred: both 5×5 box-blurred. Ratio: total light against the reference's. Δ: points of rel L1 against the default configuration${result.baseline ? ` (${escapeHtml(result.baseline.device)})` : ", not computed yet"}.`
      : `Frame ${c.frame} alone, against ${reference}.`;

  // Regions
  let regions = `<tr><th>region</th><th class=num title="share of the pixels">px</th><th class=num>whole</th><th class=num>fog</th><th class=num>surf.</th><th class=num title="ratio of the fog's light against the reference's">fog ratio</th></tr>`;
  for (const info of scene.regions) {
    const r = metrics.regions[info.name];
    if (!r) continue;
    const cell = (e) => (e ? `<td title="as is ${pct(e.rel_l1)} · blurred ${pct(e.rel_l1_blurred)} · ratio ${ratio(e.ratio)}">${pct(e.rel_l1, 1)}</td>` : `<td class="muted">${DASH}</td>`);
    regions += `<tr title="${escapeHtml(info.description)}"><td>${escapeHtml(info.label)}</td><td class="sub">${pct(r.pixels, 0)}</td>${cell(r.whole)}${cell(r.fog)}${cell(r.surfaces)}<td>${r.fog ? ratio(r.fog.ratio) : DASH}</td></tr>`;
  }
  $("#regions").innerHTML = regions;

  renderTiming();
  renderCycles();
  renderFlicker();
}

function renderTiming() {
  const { result, config: c } = shown;
  const t = result.timing;
  const facts = [];
  const where = t.device === "cpu" ? `CPU, ${t.threads} threads` : t.device_name;
  const fastest = t.repeats > 1 ? `, fastest of ${t.repeats} calls` : "";
  facts.push(["Warm call", `${duration(t.warm)}<span class="sub">frame ${t.frame}, ${escapeHtml(where)}${fastest}</span>`]);
  if (t.first) {
    const extra = t.one_off ? `included one-off work, +${duration(t.first.seconds - t.warm)}` : "nothing to compute once";
    facts.push(["First call", `${duration(t.first.seconds)}<span class="sub">frame ${t.first.frame}: ${extra}</span>`]);
  }
  facts.push(["Fog computed at", c.factor === 1 ? "every pixel" : `1/${c.factor} × 1/${c.factor}, upsampled`]);
  const pipeline = result.cycles && result.cycles.pipeline;
  if (c.clear === "clean") facts.push(["Frames without fog", `clean<span class="sub">rendered at many samples, not timed</span>`]);
  else facts.push(["Frames without fog", `${c.clear} spp + OIDN${pipeline && pipeline.clear_seconds ? `<span class="sub">${duration(pipeline.clear_seconds)} in Cycles (benchmark)</span>` : ""}`]);
  if (pipeline && pipeline.seconds) facts.push(["Pipeline", `${duration(pipeline.seconds)}<span class="sub">Cycles without fog + the closed form</span>`]);
  $("#timing").innerHTML = facts.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
}

function renderCycles() {
  const { result, config: c } = shown;
  const holder = $("#cycles");
  const cycles = result.cycles;
  if (!cycles || !cycles.available) {
    holder.innerHTML = `<p class="panel-note">${escapeHtml((cycles && cycles.reason) || "No benchmark of Cycles.")}</p>`;
    $("#chart").replaceChildren();
    return;
  }
  const equal = cycles.rel_l1;
  const blurred = cycles.rel_l1_blurred;
  const describe = (entry) => {
    if (entry.seconds !== null && entry.seconds !== undefined) return `≈ ${spp(entry.spp)} spp · ${duration(entry.seconds)}`;
    if (!entry.bound) return DASH;
    return entry.beyond === "below"
      ? `beyond its curve: ${entry.bound.spp} spp (${duration(entry.bound.seconds)}) only reach ${pct(entry.bound.error)}`
      : `its fewest samples already do better: ${entry.bound.spp} spp (${duration(entry.bound.seconds)}) reach ${pct(entry.bound.error)}`;
  };
  const ours = cycles.pipeline.seconds ?? result.timing.warm;
  const prov = cycles.provenance;
  // Cycles' hardware is read from the logs of its renders ("NVIDIA GeForce RTX 5080, OPTIX")
  const cyclesOn = prov.cycles_hardware || "the benchmark's hardware";
  const sameHardware = !!prov.cycles_hardware && prov.cycles_hardware.startsWith(result.timing.device_name);
  let comparison = "";
  if (equal.seconds && ours) {
    const factor = equal.seconds / ours;
    const times = `${factor >= 10 ? factor.toFixed(0) : factor.toFixed(2)}×`;
    const hardware = sameHardware ? "" : ` <span class="reqs">Different hardware: the closed form ran on ${escapeHtml(result.timing.device_name)} here, Cycles on ${escapeHtml(cyclesOn)}.</span>`;
    if (cycles.pipeline.seconds) {
      // The saving of final_benchmark.py: Cycles' time at equal error over the pipeline's time
      const verdict = factor >= 1 ? `${times} faster than Cycles` : `${(1 / factor).toFixed(factor > 0.1 ? 2 : 0)}× slower than Cycles`;
      comparison = `<p class="cycles-line">The pipeline, Cycles without fog at ${c.clear} spp + the closed form, ${duration(ours)} per frame, is <strong>${verdict}</strong> at equal error.${hardware}</p>`;
    } else {
      // Clean frames without fog are rendered at many samples and not timed: no saving can be claimed from them
      const ratioText = factor >= 1
        ? `Cycles at equal error takes <strong>${times}</strong> the time of the closed form alone, ${duration(ours)} per frame`
        : `The closed form alone, ${duration(ours)} per frame, takes <strong>${(1 / factor).toFixed(factor > 0.1 ? 2 : 0)}×</strong> the time of Cycles at equal error`;
      comparison = `<p class="cycles-line">${ratioText}, which leaves out the clean frames without fog it is given (not timed). For the time of a whole pipeline, give it Cycles' frames at a few spp (Performance).${hardware}</p>`;
    }
  }
  const big = equal.seconds ? `<span class="big">${duration(equal.seconds)}</span><span class="unit">≈ ${spp(equal.spp)} spp</span>` : `<span class="unit">${describe(equal)}</span>`;
  holder.innerHTML = `
    <p class="headline">${big}</p>
    <p class="cycles-line">Cycles + OIDN for the same whole-frame error, ${pct(equal.error)}, over frames ${schema.scenes[c.scene].eval_frames.join(" + ")}</p>
    <p class="cycles-line">Same blurred error, ${pct(blurred.error)}: ${describe(blurred)}</p>
    ${comparison}
    <p class="provenance">Curve: ${escapeHtml(prov.file)}, ${escapeHtml(prov.date.replace("T", " "))} · Cycles' times from the logs of its renders in ${escapeHtml(prov.renders)}, on ${escapeHtml(cyclesOn)}${prov.closed_form_device ? ` · the benchmark ran its closed form on ${escapeHtml(prov.closed_form_device)}` : ""}; interpolated log-log, as final_benchmark.py.</p>`;
  renderChart(cycles, ours, equal);
}

function renderChart(cycles, oursSeconds, equal) {
  const figure = $("#chart");
  const curve = cycles.curve.filter((r) => r.denoised && r.seconds);
  if (!curve.length) {
    figure.replaceChildren();
    return;
  }
  const W = 320;
  const H = 168;
  const m = { l: 44, r: 12, t: 10, b: 30 };
  const xs = curve.map((r) => r.seconds * 1000).concat(oursSeconds ? [oursSeconds * 1000] : []);
  const ys = curve.map((r) => r.denoised.rel_l1 * 100).concat([equal.error * 100]);
  const [x0, x1] = [Math.min(...xs) / 1.3, Math.max(...xs) * 1.3];
  const [y0, y1] = [Math.min(...ys) / 1.25, Math.max(...ys) * 1.25];
  const X = (v) => m.l + ((Math.log(v) - Math.log(x0)) / (Math.log(x1) - Math.log(x0))) * (W - m.l - m.r);
  const Y = (v) => H - m.b - ((Math.log(v) - Math.log(y0)) / (Math.log(y1) - Math.log(y0))) * (H - m.t - m.b);
  const ticks = (low, high) => {
    const out = [];
    for (let e = Math.floor(Math.log10(low)); e <= Math.ceil(Math.log10(high)); e++)
      for (const f of [1, 2, 5]) {
        const v = f * 10 ** e;
        if (v >= low && v <= high) out.push(v);
      }
    return out;
  };
  const fmt = (v) => (v >= 1000 ? `${v / 1000}k` : v >= 1 ? `${v}` : v.toPrecision(1));
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Cycles' error against time, with the closed form's point">`;
  svg += `<g class="grid">${ticks(x0, x1).map((v) => `<line x1="${X(v)}" x2="${X(v)}" y1="${m.t}" y2="${H - m.b}"/>`).join("")}${ticks(y0, y1).map((v) => `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}"/>`).join("")}</g>`;
  svg += `<g class="axis">${ticks(x0, x1).map((v) => `<text x="${X(v)}" y="${H - m.b + 12}" text-anchor="middle">${fmt(v)}</text>`).join("")}${ticks(y0, y1).map((v) => `<text x="${m.l - 5}" y="${Y(v) + 3}" text-anchor="end">${fmt(v)}%</text>`).join("")}</g>`;
  svg += `<text class="axis-title" x="${(W + m.l) / 2}" y="${H - 4}" text-anchor="middle">ms per frame (log)</text>`;
  svg += `<text class="axis-title" transform="translate(10 ${(H - m.b + m.t) / 2}) rotate(-90)" text-anchor="middle">rel L1 (log)</text>`;
  const line = curve.map((r, i) => `${i ? "L" : "M"}${X(r.seconds * 1000).toFixed(1)},${Y(r.denoised.rel_l1 * 100).toFixed(1)}`).join("");
  svg += `<path class="series-cycles" d="${line}"/>`;
  if (equal.seconds) {
    const gx = X(equal.seconds * 1000);
    const gy = Y(equal.error * 100);
    svg += `<line class="guide" x1="${m.l}" x2="${gx}" y1="${gy}" y2="${gy}"/><line class="guide" x1="${gx}" x2="${gx}" y1="${gy}" y2="${H - m.b}"/>`;
  }
  curve.forEach((r, i) => {
    const x = X(r.seconds * 1000);
    const y = Y(r.denoised.rel_l1 * 100);
    svg += `<circle class="dot-cycles" cx="${x}" cy="${y}" r="4"/>`;
    if (i === 0 || i === curve.length - 1 || r.spp === state.cyclesSpp) svg += `<text class="label" x="${x}" y="${y - 8}" text-anchor="middle">${r.spp}</text>`;
    svg += `<circle class="hit" data-tip="Cycles ${r.spp} spp + OIDN: ${pct(r.denoised.rel_l1)} in ${duration(r.seconds)} per frame" cx="${x}" cy="${y}" r="10"/>`;
  });
  if (oursSeconds) {
    const x = X(oursSeconds * 1000);
    const y = Y(equal.error * 100);
    svg += `<rect class="dot-ours" x="${x - 5}" y="${y - 5}" width="10" height="10" transform="rotate(45 ${x} ${y})"/>`;
    svg += `<circle class="hit" data-tip="Closed form${shown.result.cycles.pipeline.seconds ? " pipeline" : ""}: ${pct(equal.error)} in ${duration(oursSeconds)} per frame (${escapeHtml(shown.result.timing.device_name)})" cx="${x}" cy="${y}" r="10"/>`;
  }
  svg += "</svg>";
  figure.innerHTML = `${svg}<div class="chart-legend"><span class="k-cycles">Cycles + OIDN (${escapeHtml(cycles.provenance.cycles_hardware || "benchmark")})</span><span class="k-ours">closed form, this configuration</span></div>`;
  for (const hit of figure.querySelectorAll(".hit")) attachTooltip(hit, () => `<p>${hit.dataset.tip}</p>`);
}

function renderFlicker() {
  const holder = $("#flicker");
  const { result, config: c } = shown;
  const scene = schema.scenes[c.scene];
  if (!scene.sequence.length) {
    holder.innerHTML = `<p class="panel-note">Measured on the day scene's consecutive frames 255–262 only.</p>`;
    return;
  }
  if (!result.flicker || !state.flicker) {
    holder.replaceChildren(
      h("p", { class: "panel-note", text: `Temporal error of log intensity over frames ${scene.sequence[0]}–${scene.sequence.at(-1)}, which drives false events in event cameras (common.flicker): ${scene.sequence.length} frames to compute.` }),
      h("button", {
        class: "ghost small",
        type: "button",
        text: state.flicker ? "Computing…" : "Measure flicker",
        disabled: state.flicker,
        onclick: () => {
          state.flicker = true;
          changed();
          renderFlicker();
        },
      }),
    );
    return;
  }
  const f = result.flicker;
  const cycles = f.cycles
    ? `<p class="cycles-line">Cycles + OIDN flickers as little at ≈ ${spp(f.spp_at_equal)} spp${f.seconds_at_equal ? `, ${duration(f.seconds_at_equal)} per frame` : ""}.</p>`
    : "";
  holder.innerHTML = `<dl class="facts"><dt>Temporal</dt><dd>${f.closed_form.temporal.toFixed(4)}</dd><dt>Static</dt><dd>${f.closed_form.static.toFixed(4)}</dd></dl>${cycles}<p class="panel-note">Frames ${f.frames[0]}–${f.frames.at(-1)} against the reference (common.flicker); measured for every configuration until stopped.</p>`;
  holder.append(
    h("button", {
      class: "ghost small",
      type: "button",
      text: "Stop measuring",
      onclick: () => {
        state.flicker = false;
        changed();
        renderFlicker();
      },
    }),
  );
}

// -------------------------------------------------------------------------------------------------------------------
// What each switch does, from the latest ablations

let ablationSeq = 0;

async function loadAblation() {
  const device = state.device || (schema && schema.device);
  const seq = ++ablationSeq;
  const scene = state.scene;
  let result;
  try {
    result = await getJSON(`/api/ablation?scene=${scene}&factor=${state.factor}&device=${device}`);
  } catch (error) {
    result = { rows: [], files: [], error: error.message };
  }
  if (seq !== ablationSeq) return; // the scene, factor or device changed since
  ablation = { ...result, scene };
  renderAblation();
}

/* Whether ablations are of the same model as this library's is told by the default configuration's errors: once they
   are known, which the result of any configuration says (its baseline), the table is asked for again */
function checkAblation(result) {
  if (ablation && ablation.checked === false && ablation.scene === state.scene && result.baseline) loadAblation();
}

let ablationDrawn = null;

function renderAblation() {
  const table = $("#ablation");
  const note = $("#ablation-note");
  if (!ablation || !schema) return;
  // Drawn again only when what it shows changes, not on every poll of a computation
  const missing = prerequisites && prerequisites.available ? prerequisites.missing : null;
  const signature = JSON.stringify([ablation, missing, state.effects, state.scene]);
  if (signature === ablationDrawn) return;
  ablationDrawn = signature;
  if (!ablation.rows.length) {
    table.innerHTML = "";
    note.textContent = ablation.error || `No ablation of the ${state.scene} scene in results/*ablation*.json.`;
    return;
  }
  const sources = [];
  /* Sources by their number; an ablation of another model than this library's (its every switch on gives other
     errors) is marked, as one whose model isn't checked yet */
  const MARKS = { other: "≠", null: "?" };
  const sourceIndex = (s) => {
    const key = `${s.file}|${s.factor}`;
    let i = sources.findIndex((x) => x.key === key);
    if (i < 0) i = sources.push({ key, ...s }) - 1;
    const mark = MARKS[s.model] || "";
    return mark ? `${i + 1}<span class="up" title="${s.model === "other" ? "another model than this library's" : "model not checked yet"}">${mark}</span>` : `${i + 1}`;
  };
  const rows = new Map(ablation.rows.map((r) => [r.switch, r]));
  const order = [];
  for (const group of Object.keys(schema.groups)) {
    const names = schema.switches.filter((s) => s.group === group).map((s) => s.name);
    if (names.length) order.push([humanize(group), names]);
  }
  const inputs = [...rows.keys()].filter((n) => !switches[n] && n in schema.inputs);
  if (inputs.length) order.push(["Model inputs", inputs]);
  let html = `<tr><th>turned off alone</th><th class=num title="time saved per frame">saved</th><th class=num title="points of rel L1 added to whole frames">Δ whole</th><th class=num>Δ fog</th><th class=num>Δ surf.</th><th class=num>fog ratio</th><th class=num>src</th></tr>`;
  for (const [group, names] of order) {
    html += `<tr class="group-row"><td colspan="7">${escapeHtml(group)}</td></tr>`;
    for (const name of names) {
      const r = rows.get(name);
      const missing = prerequisites && prerequisites.available && switches[name] ? prerequisites.missing[name] || [] : [];
      const inactive = missing.length > 0;
      const clickable = switches[name] && !inactive ? " clickable" : "";
      const title = switches[name] ? `${switches[name].description}${inactive ? " (inactive here)" : " Click to toggle."}` : "An input of the model.";
      if (!r) {
        html += `<tr class="inactive${clickable}" data-name="${name}" title="${escapeHtml(title)}"><td>${escapeHtml(humanize(name))}</td><td colspan="6">not in the ablations yet: run analysis/ablation.py</td></tr>`;
        continue;
      }
      if (r.ineffective) {
        html += `<tr class="inactive" title="${escapeHtml(title)}"><td>${escapeHtml(humanize(name))}</td><td colspan="5">ineffective in this scene (${escapeHtml(r.ineffective.join(", "))})</td><td>lib</td></tr>`;
        continue;
      }
      const cls = (x) => (x > 0.005 ? "up" : x < -0.005 ? "down" : "");
      html += `<tr class="${inactive ? "inactive" : ""}${clickable}" data-name="${name}" title="${escapeHtml(title)}"><td>${escapeHtml(humanize(name))}${switches[name] && !state.effects[name] ? ' <span class="sub">(off)</span>' : ""}</td><td>${milliseconds(r.saved_ms)}</td><td class="${cls(r.added.whole)}">${points(r.added.whole)}</td><td class="${cls(r.added.fog)}">${points(r.added.fog)}</td><td class="${cls(r.added.surfaces)}">${points(r.added.surfaces)}</td><td>${ratio(r.fog_ratio)}</td><td>${sourceIndex(r.source)}</td></tr>`;
    }
  }
  table.innerHTML = html;
  for (const row of table.querySelectorAll("tr.clickable")) {
    row.addEventListener("click", () => {
      const name = row.dataset.name;
      state.effects[name] = !state.effects[name];
      rebuildSwitchStates();
      changed();
    });
  }
  const helper = ablation.rows.some((r) => r.source && r.source.helper);
  const model = (s) => {
    if (s.model === "same") return "";
    if (s.model === "other") {
      const here = ablation.baseline ? ` vs ${pct(ablation.baseline.whole)} here` : "";
      return ` — <strong>another model</strong>: every switch on gives ${pct(s.all_on_whole)} whole${here}`;
    }
    return " — model not checked until the default configuration is computed";
  };
  note.innerHTML = `Each switch off alone against every switch on, frames 255 + 345: ms saved per frame and points of rel L1 added (red: the term removes error). Ablations of the same model as this library's come first (their every switch on gives its errors). Sources: ${sources
    .map((s, i) => `<span class="mono">[${i + 1}]</span> ${escapeHtml(s.file)}, ${escapeHtml(String(s.device))}${s.threads && s.device === "cpu" ? ` ${s.threads} threads` : ""}, factor ${s.factor}, ${escapeHtml(s.date.replace("T", " "))}${model(s)}`)
    .join("; ")}${helper ? `${sources.length ? "; " : ""}<span class="mono">lib</span>: ineffective in the scene's default configuration, as the library's Effects.missing tells` : ""}.`;
}

// -------------------------------------------------------------------------------------------------------------------
// Theme

function setupTheme() {
  const button = $("#theme");
  const modes = ["auto", "light", "dark"];
  let mode = "auto";
  try {
    mode = localStorage.getItem("fog-viewer-theme") || "auto";
  } catch {
    mode = "auto";
  }
  const apply = () => {
    if (mode === "auto") delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = mode;
    button.title = mode === "auto" ? "Theme: follows the system" : `Theme: ${mode}`;
    buildColormap();
    errorCache.clear();
    if (view.a || view.b) paint();
  };
  button.addEventListener("click", () => {
    mode = modes[(modes.indexOf(mode) + 1) % modes.length];
    try {
      localStorage.setItem("fog-viewer-theme", mode);
    } catch {
      /* the theme is only kept for this visit */
    }
    apply();
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", apply);
  apply();
}

/* Three columns from 1280 px; narrower, the numbers go under the images, and on phones after the switches, with the
   table of switches at the end: the breakpoints of style.css */
const WIDE = matchMedia("(min-width: 1280px)");
const MEDIUM = matchMedia("(min-width: 860px)");

function placePanels() {
  const layout = $(".layout");
  const stage = $(".stage");
  const metrics = $("#metrics");
  const table = $("#ablation-panel");
  if (WIDE.matches) {
    if (metrics.parentElement !== layout) layout.append(metrics);
    if (table.parentElement !== stage) stage.append(table);
  } else if (MEDIUM.matches) {
    if (metrics.parentElement !== stage) stage.append(metrics);
    stage.append(table);
  } else {
    if (metrics.parentElement !== layout) layout.append(metrics);
    if (table.parentElement !== layout) layout.append(table);
  }
}

// -------------------------------------------------------------------------------------------------------------------
// Start

async function loadSchema() {
  for (;;) {
    try {
      return await getJSON("/api/schema");
    } catch (error) {
      const empty = $("#empty");
      empty.hidden = false;
      empty.replaceChildren(h("span", { class: "led" }), h("span", { text: error.status === 503 ? `Loading the scenes: ${error.message}` : `Waiting for the server: ${error.message}` }));
      await sleep(800);
    }
  }
}

async function init() {
  setupTheme();
  setupViewer();
  placePanels();
  WIDE.addEventListener("change", placePanels);
  MEDIUM.addEventListener("change", placePanels);
  window.addEventListener("resize", placePanels);
  renderOverlay();
  pollStatus();
  setInterval(pollStatus, 700);
  schema = await loadSchema();
  $("#library").textContent = `${schema.library} · ${schema.fingerprint}`;
  $("#library").title = `visionsim imported from ${schema.library}, whose source has the fingerprint ${schema.fingerprint} (the same for identical copies); renders under ${schema.root}; run ${schema.instance} of the server`;
  state.device = schema.device;
  for (const sw of schema.switches) state.effects[sw.name] = sw.default;
  for (const [name, about] of Object.entries(schema.inputs)) state.inputs[name] = about.default;
  if (schema.inputs.albedo) KINDS.splice(KINDS.findIndex((k) => k.id === "input") + 1, 0, ALBEDO_KIND);
  const available = Object.keys(schema.scenes);
  if (!available.length) {
    setNotice("error", `No scene could be loaded: ${escapeHtml(JSON.stringify(schema.scene_errors))}`);
    return;
  }
  if (!schema.scenes[state.scene]) state.scene = available[0];
  loadHash();
  for (const [scene, error] of Object.entries(schema.scene_errors || {})) setNotice(`scene-${scene}`, `The ${escapeHtml(scene)} scene isn't available: ${escapeHtml(error)}`);
  buildScenePicker();
  buildControls();
  buildImagePickers();
  syncExposure();
  $("#error-range-value").textContent = `±${Math.round(state.errorRange * 100)}%`;
  loadAblation();
  changed();
}

init();
