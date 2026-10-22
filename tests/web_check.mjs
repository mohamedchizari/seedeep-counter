// A check of the browser page (web/) in headless Google Chrome: the detector loads; the overlay is letterboxed like the
// video; the page looks at each shown video frame at most once; things enter the default zone and its sampled mean
// stay never shows below zero; the clip's own loop starts a second pass and the CSV keeps the passes apart; a file that
// is not a video gets a message; and the page logs no error, refused request or blocked load. It takes about a minute.
// Needs: the page served (python3 web/serve.py), Node 18+, `npm install playwright-core`, and Google Chrome (Playwright's
// own Chromium does not play the H.264 sample clip).
// usage: CHROME="/path/to/Google Chrome" node tests/web_check.mjs [http://127.0.0.1:8000/]
// env: CHROME (the browser to run), PLAYWRIGHT_MODULE (where playwright-core is, if not installed beside this repo),
//      ALLOW_CONSOLE (a regular expression for console errors to accept, for a host that injects its own scripts)
import fs from "node:fs";
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright-core");
const url = process.argv[2] || "http://127.0.0.1:8000/";
const allow = process.env.ALLOW_CONSOLE ? new RegExp(process.env.ALLOW_CONSOLE) : null;
const browser = await chromium.launch({ executablePath: process.env.CHROME || undefined, headless: true, args: ["--autoplay-policy=no-user-gesture-required"] });
const page = await browser.newPage({ viewport: { width: 1280, height: 900 }, acceptDownloads: true });
const problems = [];
page.on("pageerror", (e) => problems.push("page error: " + String(e)));
page.on("console", (m) => { if (m.type() === "error" && !(allow && allow.test(m.text()))) problems.push("console: " + m.text()); });
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
  check(q1.d - q0.d > 0 && q1.d - q0.d <= q1.f - q0.f + 2, `detections running, at most one per shown frame (${q1.d - q0.d} detections, ${q1.f - q0.f} frames shown in 5 s)`);
  await page.click("#btn-zone"); await page.click("#btn-line");
  let last = await page.evaluate(() => document.getElementById("video").currentTime), wrapped = false, after = 0, minStay = Infinity;
  for (let k = 0; k < 140 && after < 8; k++) {
    await page.waitForTimeout(500);
    const s = await page.evaluate(() => ({ t: document.getElementById("video").currentTime, stay: parseFloat(document.getElementById("z-stay").textContent) }));
    if (s.t < last) wrapped = true; last = s.t; if (wrapped) after += 1;
    if (!Number.isNaN(s.stay)) minStay = Math.min(minStay, s.stay);
  }
  const entered = await page.evaluate(() => parseInt(document.getElementById("z-entered").textContent, 10));
  check(wrapped, "the sample clip looped by itself");
  check(entered > 0, `things entered the zone (${entered})`);
  check(Number.isFinite(minStay) && minStay >= 0, `the zone's mean stay, read every 0.5 s, was a number and never below zero (lowest ${minStay} s)`);
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
await browser.close();
console.log(fails ? `${fails} check(s) failed` : "all checks passed");
process.exit(fails ? 1 : 0);
