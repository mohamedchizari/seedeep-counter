// A check of the browser page (web/) in headless Chrome: the detector loads, the overlay is letterboxed like the video,
// zone stays never go negative across a loop of the sample clip, the CSV keeps passes apart, and a file that is not a
// video gets a message. It takes about a minute.
// Needs: the page served (python3 web/serve.py), Node 18+, `npm install playwright-core`, and Chrome or Chromium.
// usage: CHROME=/path/to/chrome node tests/web_check.mjs [http://127.0.0.1:8000/]
import fs from "node:fs";
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright-core");
const url = process.argv[2] || "http://127.0.0.1:8000/";
const browser = await chromium.launch({ executablePath: process.env.CHROME || undefined, headless: true, args: ["--autoplay-policy=no-user-gesture-required"] });
const page = await browser.newPage({ viewport: { width: 1280, height: 900 }, acceptDownloads: true });
const errors = []; page.on("pageerror", (e) => errors.push(String(e)));
let fails = 0;
const check = (ok, what) => { console.log((ok ? "PASS " : "FAIL ") + what); if (!ok) fails += 1; };
try {
  await page.goto(url, { waitUntil: "load", timeout: 60000 });
  await page.waitForFunction(() => window.__seedeep && window.__seedeep.ready && window.__seedeep.frames > 30, null, { timeout: 120000 });
  check(true, `detector loaded (${await page.evaluate(() => window.__seedeep.delegate)} delegate) and ran on 30+ frames of the sample`);
  check(await page.evaluate(() => getComputedStyle(document.getElementById("overlay")).objectFit === "contain"), "the overlay is letterboxed like the video (object-fit: contain)");
  await page.click("#btn-zone"); await page.click("#btn-line");
  await page.evaluate(() => { document.getElementById("video").currentTime = 0; });
  let last = 0, wrapped = false, after = 0, minStay = Infinity;
  for (let k = 0; k < 140 && after < 8; k++) {
    await page.waitForTimeout(500);
    const s = await page.evaluate(() => ({ t: document.getElementById("video").currentTime, stay: parseFloat(document.getElementById("z-stay").textContent) }));
    if (s.t < last) wrapped = true; last = s.t; if (wrapped) after += 1;
    if (!Number.isNaN(s.stay)) minStay = Math.min(minStay, s.stay);
  }
  check(wrapped, "the sample clip looped");
  check(minStay >= 0, `the zone's mean stay never went below zero (lowest ${minStay} s)`);
  const [dl] = await Promise.all([page.waitForEvent("download", { timeout: 20000 }), page.click("#btn-csv")]);
  const lines = fs.readFileSync(await dl.path(), "utf8").split("\n");
  const head = lines[0].split(",").map((x) => x.replace(/"/g, ""));
  check(head.join(",") === "second,people_in_frame_max,vehicles_in_frame_max,other_in_frame_max,in_zone_max,frames_seen,pass", "the CSV header");
  const passes = new Set(); for (const l of lines.slice(1)) { const c = l.split(",").map((x) => x.replace(/"/g, "")); if (c.length === 7 && /^\d+$/.test(c[0])) passes.add(c[6]); else break; }
  check(passes.has("1") && passes.has("2"), `the per-second rows keep the passes apart (passes ${[...passes].join(", ")})`);
  await page.setInputFiles("#file", { name: "not-a-video.mp4", mimeType: "video/mp4", buffer: Buffer.from("this is not a video file") });
  await page.waitForFunction(() => /could not be played/.test(document.getElementById("status").textContent), null, { timeout: 15000 }).then(() => check(true, "a file that is not a video gets a message"), () => check(false, "a file that is not a video gets a message"));
} catch (e) { check(false, "the run itself: " + String(e).slice(0, 200)); }
check(errors.length === 0, `no script errors on the page${errors.length ? ": " + errors[0].slice(0, 160) : ""}`);
await browser.close();
console.log(fails ? `${fails} check(s) failed` : "all checks passed");
process.exit(fails ? 1 : 0);
