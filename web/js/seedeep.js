// SeeDeep, version 2.9: object detection and counting in the viewer's own browser. Nothing leaves the device: the model, the
// WebAssembly runtime and the sample clip are files from the server that serves this page; the camera and any opened file stay in the page.
// Detector: MediaPipe Tasks Vision (Apache 2.0), EfficientDet-Lite0 (80 COCO classes). Tracking, the line, the zone and the
// counting: this file. The line can be dragged by its two ends; a zone (a polygon with draggable corners) counts who is inside and for how long.

const VEHICLES = new Set(["car", "truck", "bus", "motorcycle", "bicycle", "train"]);
const group = (c) => (c === "person" ? "person" : VEHICLES.has(c) ? "vehicle" : "other");
const COLORS = { person: "#f2a516", vehicle: "#1fa3a6", other: "#8a98a8" };
const $ = (id) => document.getElementById(id);
const video = $("video"), canvas = $("overlay"), ctx = canvas.getContext("2d"), status = $("status");
let NOTE = "", LOADING = ""; // a settings-file problem or the detector's failure to load, and the loading message: shown whenever nothing else is being said
const say = (t) => { const s = t || [NOTE, LOADING].filter(Boolean).join(" "); status.textContent = s; status.hidden = !s; };
// a page can carry <body data-config="./config.json">: its line, its zone, the classes it counts, and whether to start on the camera
let ALLOW = null, CONFIG = null;
// the line and the zone live in fractions of the frame, so they survive a change of source size
let line = { x1: 0, y1: 0.62, x2: 1, y2: 0.62 }, zone = null, lineOn = false;
const STORE_KEY = "seedeep.geometry." + location.pathname;

// ---- geometry
function sideOf(p, L, w, h) { const x1 = L.x1 * w, y1 = L.y1 * h, x2 = L.x2 * w, y2 = L.y2 * h; return (x2 - x1) * (p.y - y1) - (y2 - y1) * (p.x - x1); }
function alongLine(p, L, w, h) { const x1 = L.x1 * w, y1 = L.y1 * h, dx = L.x2 * w - x1, dy = L.y2 * h - y1; const len2 = dx * dx + dy * dy || 1; return ((p.x - x1) * dx + (p.y - y1) * dy) / len2; }
function inPoly(p, poly, w, h) { let inside = false; for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) { const xi = poly[i].x * w, yi = poly[i].y * h, xj = poly[j].x * w, yj = poly[j].y * h; if ((yi > p.y) !== (yj > p.y) && p.x < ((xj - xi) * (p.y - yi)) / (yj - yi) + xi) inside = !inside; } return inside; }
const centre = (b) => ({ x: b.x + b.w / 2, y: b.y + b.h / 2 });

// ---- a small tracker: greedy IoU matching, an id per track, a short memory for misses, the line and the zone
function iou(a, b) {
  const x1 = Math.max(a.x, b.x), y1 = Math.max(a.y, b.y), x2 = Math.min(a.x + a.w, b.x + b.w), y2 = Math.min(a.y + a.h, b.y + b.h);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1); const u = a.w * a.h + b.w * b.h - inter;
  return u > 0 ? inter / u : 0;
}
class Tracker {
  constructor() { this.tracks = []; this.nextId = 1; this.seen = {}; this.crossings = { person: { down: 0, up: 0 }, vehicle: { down: 0, up: 0 } }; this.zoneEntered = 0; this.zoneStays = []; this.events = []; }
  update(dets, w, h, L, poly, t_s) {
    const used = new Set(), pairs = [];
    this.tracks.forEach((t, ti) => dets.forEach((d, di) => { if (d.cls === t.cls) { const s = iou(t.box, d.box); if (s > 0.3) pairs.push([s, ti, di]); } }));
    pairs.sort((p, q) => q[0] - p[0]);
    const matchedT = new Set();
    for (const [, ti, di] of pairs) {
      if (matchedT.has(ti) || used.has(di)) continue;
      matchedT.add(ti); used.add(di);
      const t = this.tracks[ti], d = dets[di]; const c0 = centre(t.box), c1 = centre(d.box);
      t.box = d.box; t.score = d.score; t.hits += 1; t.misses = 0; t.lastSeen = t_s;
      const g = group(t.cls);
      if (L && t.hits >= 3 && (g === "person" || g === "vehicle")) {
        const s0 = sideOf(c0, L, w, h), s1 = sideOf(c1, L, w, h), a = alongLine(c1, L, w, h);
        if (a >= 0 && a <= 1) { if (s0 < 0 && s1 >= 0) { this.crossings[g].down += 1; this.events.push([t_s, g, t.cls, "down", t.id, pass]); } else if (s0 >= 0 && s1 < 0) { this.crossings[g].up += 1; this.events.push([t_s, g, t.cls, "up", t.id, pass]); } }
      }
      if (poly && t.hits >= 3) {
        const inside = inPoly(c1, poly, w, h);
        if (inside && !t.inZone) { t.inZone = true; t.zoneSince = t_s; this.zoneEntered += 1; this.events.push([t_s, g, t.cls, "entered", t.id, pass]); }
        else if (!inside && t.inZone) { t.inZone = false; this.zoneStays.push(t_s - t.zoneSince); this.events.push([t_s, g, t.cls, "left", t.id, pass]); }
      }
    }
    this.tracks.forEach((t, ti) => { if (!matchedT.has(ti)) t.misses += 1; });
    dets.forEach((d, di) => { if (!used.has(di)) this.tracks.push({ id: this.nextId++, cls: d.cls, box: d.box, score: d.score, hits: 1, misses: 0, counted: false, inZone: false, zoneSince: 0, lastSeen: t_s }); });
    this.tracks = this.tracks.filter((t) => { if (t.misses > 12) { if (t.inZone) this.zoneStays.push(t.lastSeen - t.zoneSince); return false; } return true; });
    for (const t of this.tracks) if (!t.counted && t.hits >= 3) { t.counted = true; this.seen[t.cls] = (this.seen[t.cls] || 0) + 1; }
    return this.tracks.filter((t) => t.misses === 0 && t.hits >= 2);
  }
  zoneNow() { return this.tracks.filter((t) => t.misses === 0 && t.inZone).length; }
  restart() { for (const t of this.tracks) if (t.inZone) { this.zoneStays.push(t.lastSeen - t.zoneSince); this.events.push([t.lastSeen, group(t.cls), t.cls, "left", t.id, pass]); } this.tracks = []; }
  newZone(t_s) { for (const t of this.tracks) if (t.inZone) { this.events.push([t.lastSeen, group(t.cls), t.cls, "left", t.id, pass]); t.inZone = false; } this.zoneEntered = 0; this.zoneStays = []; this.events.push([t_s, "", "", "zone changed", "", pass]); }
  meanStay(t_s) { const all = this.zoneStays.concat(this.tracks.filter((t) => t.inZone).map((t) => t_s - t.zoneSince)); return all.length ? all.reduce((a, b) => a + b, 0) / all.length : 0; }
}

// ---- state
let detector = null, tracker = new Tracker(), source = "sample", running = false, lastT = -1, prevT = 0, pass = 1, fileURL = null, zoneHistory = [];
// one detection per new video frame where the browser says when one is shown; otherwise whenever the video clock moves
const RVFC = "requestVideoFrameCallback" in HTMLVideoElement.prototype;
let newFrame = false, frameTime = 0;
function onFrame(now, meta) { newFrame = true; frameTime = meta.mediaTime; video.requestVideoFrameCallback(onFrame); }
if (RVFC) video.requestVideoFrameCallback(onFrame);
let fps = 0, infMs = 0, frames = 0;
const perSecond = new Map(); // "pass:second" -> {pass, sec, person, vehicle, other, zone, n}; a loop of the sample starts a new pass
let t0 = performance.now();
window.__seedeep = { ready: false, frames: 0, fps: 0, infMs: 0, delegate: null, lastDetections: 0, line, zone: null, lineOn: false, zoneNow: 0 };

async function load() {
  LOADING = "Loading the detector (about 17 MB, once)."; say("");
  const { FilesetResolver, ObjectDetector } = await import("/vendor/vision_bundle.mjs");
  const vision = await FilesetResolver.forVisionTasks("/vendor/wasm");
  const opts = (delegate) => ({ baseOptions: { modelAssetPath: "/models/efficientdet_lite0.tflite", delegate }, scoreThreshold: 0.4, maxResults: 25, runningMode: "VIDEO" });
  try { detector = await ObjectDetector.createFromOptions(vision, opts("GPU")); window.__seedeep.delegate = "GPU"; }
  catch (e) { detector = await ObjectDetector.createFromOptions(vision, opts("CPU")); window.__seedeep.delegate = "CPU"; }
  window.__seedeep.ready = true; LOADING = "";
  if (status.textContent.includes("Loading the detector")) say("");
  for (const b of ["btn-camera", "btn-file", "btn-line", "btn-zone", "btn-csv"]) $(b).disabled = false;
}

// ---- drawing
function handle(x, y, s, c) { ctx.beginPath(); ctx.arc(x, y, 7 * s, 0, Math.PI * 2); ctx.fillStyle = "#10161c"; ctx.fill(); ctx.lineWidth = 2 * s; ctx.strokeStyle = c; ctx.stroke(); }
function draw(tracks, w, h) {
  canvas.width = w; canvas.height = h; ctx.clearRect(0, 0, w, h);
  const s = Math.max(1, Math.round(w / 480));
  if (zone) {
    ctx.beginPath(); zone.forEach((p, i) => (i ? ctx.lineTo(p.x * w, p.y * h) : ctx.moveTo(p.x * w, p.y * h))); ctx.closePath();
    ctx.fillStyle = "rgba(31,163,166,0.14)"; ctx.fill(); ctx.setLineDash([8 * s, 6 * s]); ctx.lineWidth = 2 * s; ctx.strokeStyle = "rgba(31,163,166,0.95)"; ctx.stroke(); ctx.setLineDash([]);
    zone.forEach((p) => handle(p.x * w, p.y * h, s, "#1fa3a6"));
  }
  if (lineOn) {
    ctx.strokeStyle = "rgba(233,237,240,0.95)"; ctx.setLineDash([12 * s, 8 * s]); ctx.lineWidth = 2 * s; ctx.beginPath(); ctx.moveTo(line.x1 * w, line.y1 * h); ctx.lineTo(line.x2 * w, line.y2 * h); ctx.stroke(); ctx.setLineDash([]);
    handle(line.x1 * w, line.y1 * h, s, "#e9edf0"); handle(line.x2 * w, line.y2 * h, s, "#e9edf0");
  }
  ctx.font = `${12 * s}px Montserrat, system-ui, sans-serif`; ctx.textBaseline = "top"; // Montserrat if the browser has it (installed, or loaded by a page that embeds this script), else system-ui
  for (const t of tracks) {
    const g = group(t.cls), c = COLORS[g], b = t.box, k = Math.min(b.w, b.h) * 0.22;
    ctx.strokeStyle = c; ctx.lineWidth = 2 * s;
    ctx.beginPath();
    ctx.moveTo(b.x, b.y + k); ctx.lineTo(b.x, b.y); ctx.lineTo(b.x + k, b.y);
    ctx.moveTo(b.x + b.w - k, b.y); ctx.lineTo(b.x + b.w, b.y); ctx.lineTo(b.x + b.w, b.y + k);
    ctx.moveTo(b.x, b.y + b.h - k); ctx.lineTo(b.x, b.y + b.h); ctx.lineTo(b.x + k, b.y + b.h);
    ctx.moveTo(b.x + b.w - k, b.y + b.h); ctx.lineTo(b.x + b.w, b.y + b.h); ctx.lineTo(b.x + b.w, b.y + b.h - k);
    ctx.stroke();
    if (b.w < 28 * s) continue;
    const label = b.w < 80 * s ? t.cls : `${t.cls} ${Math.round(t.score * 100)}`; const tw = ctx.measureText(label).width + 8 * s;
    ctx.fillStyle = c; ctx.fillRect(b.x, Math.max(0, b.y - 16 * s), tw, 16 * s);
    ctx.fillStyle = "#10161c"; ctx.fillText(label, b.x + 4 * s, Math.max(0, b.y - 16 * s) + 2 * s);
  }
}

function paintCounts(tracks, t_s) {
  const now = { person: 0, vehicle: 0, other: 0 };
  for (const t of tracks) now[group(t.cls)] += 1;
  $("n-person").textContent = now.person; $("n-vehicle").textContent = now.vehicle; $("n-other").textContent = now.other;
  const cr = tracker.crossings;
  $("x-person").textContent = `${cr.person.down} down, ${cr.person.up} up`; $("x-vehicle").textContent = `${cr.vehicle.down} down, ${cr.vehicle.up} up`;
  const zn = zone ? tracker.zoneNow() : 0; window.__seedeep.zoneNow = zn;
  if (zone) { $("z-now").textContent = zn; $("z-entered").textContent = tracker.zoneEntered; $("z-stay").textContent = tracker.meanStay(t_s).toFixed(1) + " s"; }
  const seen = Object.entries(tracker.seen).sort((a, b) => b[1] - a[1]).slice(0, 12);
  const ul = $("seen-list"); ul.textContent = "";
  for (const [cls, n] of seen) { const li = document.createElement("li"); li.append(Object.assign(document.createElement("span"), { textContent: cls }), Object.assign(document.createElement("b"), { textContent: n })); ul.append(li); }
  $("m-fps").textContent = fps.toFixed(0); $("m-ms").textContent = infMs.toFixed(0);
  const sec = Math.floor(t_s), key = pass + ":" + sec; const row = perSecond.get(key) || { pass, sec, person: 0, vehicle: 0, other: 0, zone: 0, n: 0 };
  row.person = Math.max(row.person, now.person); row.vehicle = Math.max(row.vehicle, now.vehicle); row.other = Math.max(row.other, now.other); row.zone = Math.max(row.zone, zn); row.n += 1; perSecond.set(key, row);
}

function step() {
  if (detector && video.readyState >= 2 && !video.paused && (RVFC ? newFrame : video.currentTime !== lastT)) {
    newFrame = false;
    const t_s = RVFC ? frameTime : video.currentTime;
    if (lastT >= 0 && t_s < prevT) { tracker.restart(); pass += 1; }
    lastT = t_s; prevT = t_s;
    const a = performance.now();
    const res = detector.detectForVideo(video, a);
    const b = performance.now();
    infMs = infMs ? infMs * 0.9 + (b - a) * 0.1 : b - a;
    let dets = res.detections.map((d) => ({ cls: d.categories[0].categoryName, score: d.categories[0].score, box: { x: d.boundingBox.originX, y: d.boundingBox.originY, w: d.boundingBox.width, h: d.boundingBox.height } }));
    if (ALLOW) dets = dets.filter((d) => ALLOW.has(d.cls));
    const w = video.videoWidth, h = video.videoHeight;
    const tracks = tracker.update(dets, w, h, lineOn ? line : null, zone, t_s);
    paintCounts(tracks, t_s);  // the counts and the per-second row first, so a frame whose drawing fails is still in the record
    frames += 1; const dtp = (b - t0) / 1000; if (dtp >= 1) { fps = frames / dtp; frames = 0; t0 = b; }
    Object.assign(window.__seedeep, { frames: window.__seedeep.frames + 1, fps, infMs, lastDetections: dets.length });
    draw(tracks, w, h);
    frameOk();
  } else if (detector && video.paused && (dragging || needsRedraw)) { needsRedraw = false; draw(tracker.tracks.filter((t) => t.misses === 0 && t.hits >= 2), video.videoWidth || canvas.width, video.videoHeight || canvas.height); frameOk(); } // cleared first, so a redraw that fails is not tried again on every frame
}
const FRAME_ERR = "A video frame could not be looked at";
let frameErr = false;
function frameOk() { if (frameErr) { frameErr = false; if (status.textContent.startsWith(FRAME_ERR)) say(""); } }
function loop() {
  if (!running) return;
  try { step(); } catch (e) { if (!frameErr) console.error("SeeDeep: " + FRAME_ERR, e); frameErr = true; say(FRAME_ERR + " (" + (e && e.message ? e.message.slice(0, 120) : e) + "); the page tries again " + (video.paused && !dragging ? "after the next change." : "on the next frame.")); }
  requestAnimationFrame(loop);
}

function reset() { tracker = new Tracker(); perSecond.clear(); lastT = -1; prevT = 0; pass = 1; zoneHistory = []; newFrame = false; fps = 0; infMs = 0; frames = 0; t0 = performance.now(); ctx.clearRect(0, 0, canvas.width, canvas.height); }
function stopStream() { if (video.srcObject) { video.srcObject.getTracks().forEach((t) => t.stop()); video.srcObject = null; } }
function pressed(id) { for (const b of ["btn-sample", "btn-camera", "btn-file"]) $(b).setAttribute("aria-pressed", String(b === id)); }

async function useSample() { stopStream(); reset(); source = "sample"; pressed("btn-sample"); video.loop = true; video.muted = true; video.src = "/assets/sample_crosswalk_960.mp4"; await video.play().catch(() => {}); if (!video.error) say(""); }
async function useCamera() {
  let stream = null;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment", width: { ideal: 1280 } }, audio: false });
    stream.getVideoTracks().forEach((tr) => tr.addEventListener("ended", () => { if (source === "camera") say("The camera stopped. Press \"Use my camera\" to start it again."); }));
    stopStream(); reset(); source = "camera"; pressed("btn-camera"); video.loop = false; video.removeAttribute("src"); video.srcObject = stream; await video.play(); say("Your camera, on this device only. Nothing is sent anywhere.");
  } catch (e) {
    const n = e && e.name;
    if (stream) { stream.getTracks().forEach((t) => t.stop()); if (video.srcObject === stream) video.srcObject = null; }
    say("The camera was not opened: " + (n === "NotAllowedError" ? "permission was not given." : n === "NotFoundError" || n === "OverconstrainedError" ? "no camera was found." : n === "NotReadableError" ? "another app may be using it." : !navigator.mediaDevices ? "this page must be opened over https or from localhost." : "the browser refused (" + (n || "unknown") + ")."));
  }
}
function useFile(file) {
  if (!file) return;
  stopStream(); reset(); source = "file"; pressed("btn-file"); video.loop = false; video.muted = true; if (fileURL) URL.revokeObjectURL(fileURL); fileURL = URL.createObjectURL(file); video.src = fileURL; video.play().catch(() => {}); say(`${file.name}, opened in the page only.`);
}

// ---- the line and the zone: dragging on the overlay, remembered in this browser only
let dragging = null, needsRedraw = false;
function toFrame(ev) {
  const r = canvas.getBoundingClientRect(), w = canvas.width || 1, h = canvas.height || 1, sc = Math.min(r.width / w, r.height / h), ox = (r.width - w * sc) / 2, oy = (r.height - h * sc) / 2;
  return { x: (ev.clientX - r.left - ox) / sc, y: (ev.clientY - r.top - oy) / sc, w, h, sc };
}
function grab(p) {
  const near = (x, y) => Math.hypot(x * p.w - p.x, y * p.h - p.y) < 16 / p.sc;
  if (lineOn && near(line.x1, line.y1)) return { kind: "line", end: 1 };
  if (lineOn && near(line.x2, line.y2)) return { kind: "line", end: 2 };
  if (zone) { const i = zone.findIndex((q) => near(q.x, q.y)); if (i >= 0) return { kind: "zone", i }; }
  if (lineOn) { const a = alongLine(p, line, p.w, p.h); const d = Math.abs(sideOf(p, line, p.w, p.h)) / Math.hypot((line.x2 - line.x1) * p.w, (line.y2 - line.y1) * p.h); if (a > 0.1 && a < 0.9 && d < 14 / p.sc) return { kind: "lineAll", p0: p, l0: { ...line } }; }
  return null;
}
const clamp01 = (v) => Math.min(1, Math.max(0, v));
canvas.addEventListener("pointerdown", (ev) => { const p = toFrame(ev); const g = grab(p); if (!g) return; dragging = { ...g, before: JSON.stringify({ line, zone }) }; canvas.setPointerCapture(ev.pointerId); ev.preventDefault(); });
canvas.addEventListener("pointermove", (ev) => {
  if (!dragging) { const p = toFrame(ev); canvas.style.cursor = grab(p) ? "grab" : "default"; return; }
  const p = toFrame(ev); const fx = clamp01(p.x / p.w), fy = clamp01(p.y / p.h);
  if (dragging.kind === "line") { if (dragging.end === 1) { line.x1 = fx; line.y1 = fy; } else { line.x2 = fx; line.y2 = fy; } }
  else if (dragging.kind === "zone") { zone[dragging.i] = { x: fx, y: fy }; }
  else if (dragging.kind === "lineAll") { const dx = (p.x - dragging.p0.x) / p.w, dy = (p.y - dragging.p0.y) / p.h; line = { x1: clamp01(dragging.l0.x1 + dx), y1: clamp01(dragging.l0.y1 + dy), x2: clamp01(dragging.l0.x2 + dx), y2: clamp01(dragging.l0.y2 + dy) }; }
  window.__seedeep.line = line; window.__seedeep.zone = zone; needsRedraw = true;
});
const endDrag = (ev) => { if (!dragging) return; if (JSON.stringify({ line, zone }) !== dragging.before) tracker.events.push([Math.max(0, lastT), "", "", dragging.kind === "zone" ? "zone moved" : "line moved", "", pass]); dragging = null; try { canvas.releasePointerCapture(ev.pointerId); } catch (e) {} remember(); }; // a press that moves nothing leaves no event
canvas.addEventListener("pointerup", endDrag); canvas.addEventListener("pointercancel", endDrag);
function remember() { try { localStorage.setItem(STORE_KEY, JSON.stringify({ line, zone, lineOn })); } catch (e) {} }
const num = (v) => typeof v === "number" && Number.isFinite(v);
function cleanLine(L) { return L && typeof L === "object" && [L.x1, L.y1, L.x2, L.y2].every(num) ? { x1: clamp01(L.x1), y1: clamp01(L.y1), x2: clamp01(L.x2), y2: clamp01(L.y2) } : null; }
function cleanZone(Z) { return Array.isArray(Z) && Z.length >= 3 && Z.every((p) => p && num(p.x) && num(p.y)) ? Z.slice(0, 8).map((p) => ({ x: clamp01(p.x), y: clamp01(p.y) })) : null; }
// what this browser kept is used only when the line's values are numbers, and the zone (3 or more points, cut to 8) only with that line; values are clamped to the frame; anything else is ignored
function recall() { try { const s = JSON.parse(localStorage.getItem(STORE_KEY) || "null"); const L = s && cleanLine(s.line); if (L) { line = L; zone = cleanZone(s.zone); lineOn = s.lineOn === true; } } catch (e) {} }
function setLineOn(on) { lineOn = on; $("btn-line").setAttribute("aria-pressed", String(on)); $("x-row").hidden = !on; window.__seedeep.lineOn = on; if (on) say("Drag the ends of the line, or the line itself."); else if (status.textContent.startsWith("Drag the ends")) say(""); remember(); needsRedraw = true; }
function setZone(poly) { const t_now = Math.max(0, lastT); if (zone) zoneHistory.push({ corners: zone, entered: tracker.zoneEntered, meanStay: tracker.meanStay(t_now), until: t_now, pass }); zone = poly; tracker.newZone(t_now); $("btn-zone").setAttribute("aria-pressed", String(!!poly)); $("btn-zone").textContent = poly ? "Remove the zone" : "Add a zone"; $("z-row").hidden = !poly; window.__seedeep.zone = poly; if (poly) say("Drag the corners of the zone."); else if (status.textContent.startsWith("Drag the corners")) say(""); remember(); needsRedraw = true; }

function csv() {
  const rows = [["second", "people_in_frame_max", "vehicles_in_frame_max", "other_in_frame_max", "in_zone_max", "frames_seen", "pass"]];
  for (const r of [...perSecond.values()].sort((a, b) => a.pass - b.pass || a.sec - b.sec)) rows.push([r.sec, r.person, r.vehicle, r.other, r.zone, r.n, r.pass]);
  rows.push([]); rows.push(["seen_so_far_by_class"]); for (const [c, n] of Object.entries(tracker.seen)) rows.push([c, n]);
  rows.push([]); rows.push(["line_crossings", "down", "up"]); rows.push(["person", tracker.crossings.person.down, tracker.crossings.person.up]); rows.push(["vehicle", tracker.crossings.vehicle.down, tracker.crossings.vehicle.up]);
  rows.push(["line_from_to_fractions", line.x1.toFixed(3), line.y1.toFixed(3), line.x2.toFixed(3), line.y2.toFixed(3), lineOn ? "on" : "off"]);
  zoneHistory.forEach((z, i) => { rows.push([]); rows.push(["earlier_zone", i + 1, "until_second", z.until.toFixed(2), "pass", z.pass]); rows.push(["earlier_zone_entered", z.entered]); rows.push(["earlier_zone_mean_stay_s", z.meanStay.toFixed(1)]); rows.push(["earlier_zone_corners_fractions"].concat(z.corners.flatMap((p) => [p.x.toFixed(3), p.y.toFixed(3)]))); });
  if (zone) { rows.push([]); rows.push(["zone_entered", tracker.zoneEntered]); rows.push(["zone_mean_stay_s", tracker.meanStay(Math.max(0, lastT)).toFixed(1)]); rows.push(["zone_corners_fractions"].concat(zone.flatMap((p) => [p.x.toFixed(3), p.y.toFixed(3)]))); }
  rows.push([]); rows.push(["events", "second", "group", "class", "what", "track", "pass"]); for (const e of tracker.events) rows.push(["", e[0].toFixed(2), e[1], e[2], e[3], e[4], e[5]]);
  rows.push([]); rows.push(["source", source]); rows.push(["detector", "EfficientDet-Lite0 via MediaPipe Tasks Vision, " + (window.__seedeep.delegate || "") + " delegate, threshold 0.4"]); rows.push(["made", new Date().toISOString()]);
  const text = rows.map((r) => r.map((v) => `"${String(v).replace(/"/g, '""')}"`).join(",")).join("\n");
  const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([text], { type: "text/csv" })); a.download = "seedeep_counts.csv"; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

$("btn-sample").addEventListener("click", useSample);
$("btn-camera").addEventListener("click", useCamera);
$("btn-file").addEventListener("click", () => $("file").click());
$("file").addEventListener("change", (e) => { useFile(e.target.files && e.target.files[0]); e.target.value = ""; });
video.addEventListener("error", () => { if (source === "camera") return; say(source === "file" ? "This file could not be played in this browser; it may be in a format the browser does not decode. Try an MP4 (H.264)." : "The sample clip could not be played in this browser."); });
$("btn-line").addEventListener("click", () => setLineOn(!lineOn));
$("btn-zone").addEventListener("click", () => setZone(zone ? null : [{ x: 0.3, y: 0.35 }, { x: 0.7, y: 0.35 }, { x: 0.7, y: 0.8 }, { x: 0.3, y: 0.8 }]));
$("btn-csv").addEventListener("click", csv);
$("btn-copy").addEventListener("click", async () => {
  const t = `SeeDeep measured on this device, ${new Date().toISOString()}: ${fps.toFixed(0)} frames a second, ${infMs.toFixed(0)} ms a detection, ${window.__seedeep.delegate} delegate, ${navigator.userAgent}`;
  try { await navigator.clipboard.writeText(t); say("Copied."); setTimeout(() => { if (status.textContent === "Copied.") say(""); }, 1500); } catch (e) { say(t); }
});
video.addEventListener("ended", () => { if (source === "file") say("End of the file. The counts stand; download the CSV or open another file."); });

async function applyConfig() {
  recall();
  const src = document.body.dataset.config; if (!src) return;
  try {
    const ctl = new AbortController(), timer = setTimeout(() => ctl.abort(), 10000); let c;
    try { const r = await fetch(src, { cache: "no-store", signal: ctl.signal }); if (!r.ok) throw new Error("the server answered " + r.status); c = await r.json(); }
    catch (e) { throw e && e.name === "AbortError" ? new Error("the server did not answer in 10 s") : e; } finally { clearTimeout(timer); }
    if (!c || typeof c !== "object" || Array.isArray(c)) throw new Error("it does not hold a JSON object");
    // every field it has is checked first and all are applied only when all pass, so a file is used whole or not at all
    const L = c.line, fin = (v) => typeof v === "number" && Number.isFinite(v); let nl = null, nz = null, na = null;
    if (L !== undefined) {
      if (fin(L) && L > 0 && L < 1) nl = { x1: 0, y1: L, x2: 1, y2: L };
      else if (L && typeof L === "object" && [L.x1, L.y1, L.x2, L.y2].every(fin)) nl = { x1: clamp01(L.x1), y1: clamp01(L.y1), x2: clamp01(L.x2), y2: clamp01(L.y2) };
      else throw new Error("line is neither a fraction between 0 and 1 nor {x1, y1, x2, y2}");
    }
    if (c.zone !== undefined) {
      if (!Array.isArray(c.zone) || c.zone.length < 3) throw new Error("zone is not a list of 3 or more points");
      nz = c.zone.slice(0, 8).map((p, i) => { if (!p || typeof p !== "object" || !fin(p.x) || !fin(p.y)) throw new Error("zone point " + (i + 1) + " is not {x, y}"); return { x: clamp01(p.x), y: clamp01(p.y) }; });
    }
    if (c.classes !== undefined) { if (!Array.isArray(c.classes) || !c.classes.length || !c.classes.every((k) => typeof k === "string")) throw new Error("classes is not a list of class names"); na = new Set(c.classes); }
    if (c.start !== undefined && c.start !== "camera" && c.start !== "sample") throw new Error('start is neither "camera" nor "sample"');
    if (nl) { line = nl; lineOn = true; } if (nz) zone = nz; if (na) ALLOW = na;
    CONFIG = c; window.__seedeep.config = c;
  } catch (e) { CONFIG = null; NOTE = "The counter's settings (" + src.slice(0, 60) + ") could not be used: " + (e && e.message ? e.message.slice(0, 80) : e) + ". It counts without them."; say(""); }
}
running = true; requestAnimationFrame(loop);
applyConfig().then(() => { $("btn-line").setAttribute("aria-pressed", String(lineOn)); $("x-row").hidden = !lineOn; window.__seedeep.lineOn = lineOn; window.__seedeep.line = line; if (zone) { $("btn-zone").setAttribute("aria-pressed", "true"); $("btn-zone").textContent = "Remove the zone"; $("z-row").hidden = false; window.__seedeep.zone = zone; } })
  .then(load).then(() => (CONFIG && CONFIG.start === "camera" ? useCamera() : useSample()))
  .catch((e) => { LOADING = ""; const why = e && e.message ? e.message.slice(0, 120).replace(/[.\s]+$/, "") : e && e.target && e.target.src ? new URL(e.target.src).pathname + " did not load" : String(e);
    NOTE = "The detector could not load: " + why + ". A browser with WebAssembly is needed; if you serve this page yourself, run fetch_assets.sh first." + (NOTE ? " " + NOTE : ""); say(""); }); // kept, so "Sample clip" does not clear it
