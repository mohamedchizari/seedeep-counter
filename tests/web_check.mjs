// A check of the browser page (web/) in headless Google Chrome: the detector loads; the overlay's CSS letterboxes it like
// the video (object-fit: contain); detections in 5 s do not exceed the frames the video decoded plus 2; things enter the
// default zone and its mean stay, read twice a second, is always a number, never below zero and above zero at least once;
// no message with "could not" or "stopped" is seen in those reads; the
// clip's own loop starts a second pass and the CSV keeps the passes apart; a file that is not a video gets a message;
// and the page logs no error, refused request or blocked load. A second page is then opened with a settings file whose
// line, classes and camera start are valid but one zone point is not, a line and zone in local storage that are not
// valid, and three frames broken on purpose: it should run on the sample with none of that file applied, ignore the
// stored geometry, keep the settings message, and say, log and then clear the frame error. It takes one to a few
// minutes, depending on the machine. MediaPipe's own console lines that start "INFO: " (printed through console.error
// when it makes the CPU detector) are not counted as errors.
// Needs: the page served (python3 web/serve.py), Node 18+, `npm install playwright-core`, and Google Chrome (Playwright's
// own Chromium does not play the H.264 sample clip).
// usage: CHROME="/path/to/Google Chrome" node tests/web_check.mjs [http://127.0.0.1:8000/]
// env: CHROME (the browser to run), PLAYWRIGHT_MODULE (where playwright-core is, if not installed beside this repo),
//      ALLOW_CONSOLE (a regular expression for console errors and failed requests to ignore)
import fs from "node:fs";
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright-core");
const url = process.argv[2] || "http://127.0.0.1:8000/";
const allow = process.env.ALLOW_CONSOLE ? new RegExp(process.env.ALLOW_CONSOLE) : null;
const browser = await chromium.launch({ executablePath: process.env.CHROME || undefined, headless: true, args: ["--autoplay-policy=no-user-gesture-required"] });
const page = await browser.newPage({ viewport: { width: 1280, height: 900 }, acceptDownloads: true });
const problems = [];
page.on("pageerror", (e) => problems.push("page error: " + String(e)));
page.on("console", (m) => { if (m.type() === "error" && !/^INFO: /.test(m.text()) && !(allow && allow.test(m.text()))) problems.push("console: " + m.text()); });
page.on("requestfailed", (r) => { if (!(allow && allow.test(r.url()))) problems.push("request failed: " + r.url() + " " + (r.failure() || {}).errorText); });
let fails = 0;
const check = (ok, what) => { console.log((ok ? "PASS " : "FAIL ") + what); if (!ok) fails += 1; };
try {
  await page.goto(url, { waitUntil: "load", timeout: 60000 });
  const ready = await page.waitForFunction(() => window.__seedeep && window.__seedeep.ready && window.__seedeep.frames > 30, null, { timeout: 120000 }).then(() => true, () => false);
  check(ready, "the detector loaded and ran on 30+ frames of the sample" + (ready ? ` (${await page.evaluate(() => window.__seedeep.delegate)} delegate)` : `; status: ${await page.evaluate(() => document.getElementById("status").textContent)}`));
  if (!ready) throw new Error("the page never became ready");
  check(await page.evaluate(() => getComputedStyle(document.getElementById("overlay")).objectFit === "contain"), "the overlay is letterboxed like the video (object-fit: contain)");
  const q0 = await page.evaluate(() => ({ d: window.__seedeep.frames, f: document.getElementById("video").getVideoPlaybackQuality().totalVideoFrames }));
  await page.waitForTimeout(5000);
  const q1 = await page.evaluate(() => ({ d: window.__seedeep.frames, f: document.getElementById("video").getVideoPlaybackQuality().totalVideoFrames }));
  check(q1.d - q0.d > 0 && q1.d - q0.d <= q1.f - q0.f + 2, `detections running, no more than the frames decoded plus 2 (${q1.d - q0.d} detections, ${q1.f - q0.f} frames decoded in 5 s)`);
  await page.click("#btn-zone"); await page.click("#btn-line");
  let last = await page.evaluate(() => document.getElementById("video").currentTime), wrapped = false, after = 0, minStay = Infinity, maxStay = -Infinity, reads = 0, notNumber = 0;
  const banners = new Set();
  for (let k = 0; k < 140 && after < 8; k++) {
    await page.waitForTimeout(500);
    const s = await page.evaluate(() => ({ t: document.getElementById("video").currentTime, stay: parseFloat(document.getElementById("z-stay").textContent), said: document.getElementById("status").textContent }));
    if (s.t < last) wrapped = true; last = s.t; if (wrapped) after += 1;
    reads += 1; if (Number.isNaN(s.stay)) notNumber += 1; else { minStay = Math.min(minStay, s.stay); maxStay = Math.max(maxStay, s.stay); }
    if (/could not|stopped/.test(s.said)) banners.add(s.said.slice(0, 100));
  }
  const entered = await page.evaluate(() => parseInt(document.getElementById("z-entered").textContent, 10));
  check(wrapped, "the sample clip looped by itself");
  check(entered > 0, `things entered the zone (${entered})`);
  check(notNumber === 0 && minStay >= 0 && maxStay > 0, `the zone's mean stay, read every 0.5 s (${reads} reads), was always a number, never below zero and above zero at least once (lowest ${minStay} s, highest ${maxStay} s${notNumber ? `, ${notNumber} reads not a number` : ""})`);
  check(banners.size === 0, `no error message on the page while it counted${banners.size ? ` ("${[...banners][0]}")` : ""}`);
  const [dl] = await Promise.all([page.waitForEvent("download", { timeout: 20000 }), page.click("#btn-csv")]);
  const lines = fs.readFileSync(await dl.path(), "utf8").split("\n");
  const head = lines[0].split(",").map((x) => x.replace(/"/g, ""));
  check(head.join(",") === "second,people_in_frame_max,vehicles_in_frame_max,other_in_frame_max,in_zone_max,frames_seen,pass", "the CSV header");
  const passes = new Set(); for (const l of lines.slice(1)) { const c = l.split(",").map((x) => x.replace(/"/g, "")); if (c.length === 7 && /^\d+$/.test(c[0])) passes.add(c[6]); else break; }
  check(passes.has("1") && passes.has("2"), `the loop started pass 2 and the per-second rows keep the passes apart (passes ${[...passes].join(", ")})`);
  await page.click("#btn-zone");
  const [dl2] = await Promise.all([page.waitForEvent("download", { timeout: 20000 }), page.click("#btn-csv")]);
  check(/"earlier_zone_entered","[1-9]/.test(fs.readFileSync(await dl2.path(), "utf8")), "a removed zone's figures stay in the CSV");
  const before = problems.length;
  await page.setInputFiles("#file", { name: "not-a-video.mp4", mimeType: "video/mp4", buffer: Buffer.from("this is not a video file") });
  const msg = await page.waitForFunction(() => /This file could not be played/.test(document.getElementById("status").textContent), null, { timeout: 15000 }).then(() => true, () => false);
  check(msg, "a file that is not a video gets a message");
  problems.splice(before, problems.length - before, ...problems.slice(before).filter((p) => !/not-a-video|blob:/.test(p)));  // the bad file's own decode error is expected
} catch (e) { check(false, "the run itself: " + String(e).slice(0, 200)); }
check(problems.length === 0, `no page errors, console errors or failed requests${problems.length ? ` (${problems.length}; first: ${problems[0].slice(0, 160)})` : ""}`);

// the second page: things that go wrong on purpose (its console is read, not counted as problems)
try {
  const page2 = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const logged = []; page2.on("console", (m) => { if (m.type() === "error") logged.push(m.text()); });
  await page2.addInitScript(() => {
    try { localStorage.setItem("seedeep.geometry." + location.pathname, JSON.stringify({ line: { x1: "a", y1: 0, x2: 1, y2: 0.5 }, zone: "not a zone", lineOn: true })); } catch (e) {}
    const clear = CanvasRenderingContext2D.prototype.clearRect;
    CanvasRenderingContext2D.prototype.clearRect = function (...a) { if (window.__breakFrames > 0) { window.__breakFrames -= 1; throw new Error("a frame broken on purpose by the check"); } return clear.apply(this, a); };
  });
  const target = new URL(url).href;  // the page's own address, so a URL given without its trailing slash still matches
  await page2.route((u) => u.href === target, async (route) => { const r = await route.fetch(); const body = (await r.text()).replace("<body", '<body data-config="./seedeep-check-settings.json"'); await route.fulfill({ response: r, body }); });
  const settingsURL = new URL("./seedeep-check-settings.json", target).href;  // served by the check: valid line, classes and start, one bad zone point
  await page2.route((u) => u.href === settingsURL, (route) => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ line: 0.5, zone: [null, { x: 0, y: 0 }, { x: 1, y: 1 }], classes: ["person"], start: "camera" }) }));
  await page2.goto(url, { waitUntil: "load", timeout: 60000 });
  const ran = await page2.waitForFunction(() => window.__seedeep && window.__seedeep.ready && window.__seedeep.frames > 15, null, { timeout: 120000 }).then(() => true, () => false);
  const g = await page2.evaluate(() => ({ zone: window.__seedeep.zone, lineOn: window.__seedeep.lineOn, config: window.__seedeep.config || null, said: document.getElementById("status").textContent }));
  check(ran && g.zone === null && g.lineOn === false, `a line and zone kept in local storage that are not valid are ignored and the page runs (${ran ? "ran" : "did not run"}, zone ${JSON.stringify(g.zone)}, line ${g.lineOn ? "on" : "off"})`);
  check(/settings .* could not be used: zone point 1/.test(g.said) && g.config === null, `a settings file with one bad field is not applied at all (no config, message kept) and keeps its message after the clip starts ("${g.said.slice(0, 110)}")`);
  await page2.evaluate(() => { const st = document.getElementById("status"); window.__said = []; new MutationObserver(() => window.__said.push(st.textContent)).observe(st, { childList: true, characterData: true, subtree: true }); window.__f0 = window.__seedeep.frames; window.__breakFrames = 3; });
  await page2.waitForFunction(() => window.__breakFrames === 0 && window.__seedeep.frames - window.__f0 > 10, null, { timeout: 90000 }).catch(() => {});
  const f = await page2.evaluate(() => ({ said: window.__said, now: document.getElementById("status").textContent, more: window.__seedeep.frames - window.__f0, left: window.__breakFrames }));
  const shown = f.said.some((t) => /^A video frame could not be looked at/.test(t)), cleared = !/^A video frame could not be looked at/.test(f.now) && /settings .* could not be used/.test(f.now), inLog = logged.some((t) => /could not be looked at/.test(t));
  check(f.left === 0 && shown && cleared && inLog && f.more > 10, `three broken frames: the error is said (${shown}), logged (${inLog}), and cleared on a good frame back to the settings message (${cleared}); ${f.more} good frames after`);
} catch (e) { check(false, "the second page: " + String(e).slice(0, 200)); }
await browser.close();
console.log(fails ? `${fails} check(s) failed` : "all checks passed");
process.exit(fails ? 1 : 0);
