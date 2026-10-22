# SeeDeep counter

Finds people, vehicles and other everyday things in video and counts them: how many are in the frame each second (the tracked boxes it draws), how many cross a line, how many enter a zone and for how long. Two programs share one detector and one set of tracking rules:

- **`web/`**, in the browser: a camera or a video file, a line at any angle, a zone. Nothing you show it leaves the page. `web/js/seedeep.js` holds all of its logic; `web/index.html` and `web/style.css` are a plain page to run it in.
- **`counter/`**, in Python, offline: a video file goes in; per-second and per-minute counts (CSV), every line crossing (CSV), a summary (JSON and PDF) and the video with the boxes drawn come out.

## Run it

```sh
./fetch_assets.sh                 # downloads the model and the MediaPipe runtime (27 MB), checked by SHA-256

# in the browser
python3 web/serve.py              # then open http://127.0.0.1:8000/

# offline, on a video file
python3 -m venv .venv && . .venv/bin/activate      # Python 3.9 to 3.12, not on an Intel Mac (see counter/requirements.txt)
pip install -r counter/requirements.txt
python counter/count_video.py web/assets/sample_crosswalk_960.mp4 out/ --line 0.62 --line-text "the crosswalk"

# the tests of the Python tracker (standard library only)
python3 -m unittest discover -s tests -v
python3 tests/mutate.py           # changes the tracker rules listed in it, one at a time, and checks a test fails each time

# a check of the browser page: 19 checks, one to a few minutes depending on the machine (Node 18+, npm install playwright-core, and Google Chrome:
# Playwright's own Chromium does not play the H.264 sample; the file's header lists its settings). It first checks that the page it is
# given serves this repository's web/js/seedeep.js, so a server that already held the port is not checked by mistake.
python3 web/serve.py &
CHROME="/path/to/Google Chrome" node tests/web_check.mjs
```

`count_video.py` options: `--line` (the line's height as a fraction of the frame, 0 for none), `--every N` (look at every Nth frame), `--max-minutes`, `--title` (printed on the summary), `--line-text`, `--model`, `--ffmpeg` (the name or path of the ffmpeg that re-encodes the annotated video to H.264, `ffmpeg` by default; if none is found or it fails, the video stays MPEG-4 Part 2; `''` skips it), `--no-video`. It writes into a work folder inside the output folder (`.seedeep-counter-work`). Once the tables, `summary.json` and `summary.pdf` are written, it moves them into place and deletes an earlier run's video and `README.txt`; then it makes the video, and moves it and `README.txt` into place last. A Ctrl-C, SIGTERM or SIGHUP while files are being moved takes effect when the move is done; at any other time it ends the run, and an ffmpeg the run started with it. The folder's mark, a file named `.seedeep-counter-output`, lists the files this program put there with their sizes and modified times; a later run replaces or deletes only those, and only while their size and modified time are unchanged (one that changes during a run is left as it is, that run's file of the same name is not kept, and the run ends with exit code 4). The work folder holds a file naming this program (`made-by-count_video.txt`); a run deletes a work folder only when that file is in it. A folder holding a file under one of the output names that the mark does not list, or that has changed since, is refused (a folder marked by an earlier version, whose mark lists no files, too); so is a folder whose `.seedeep-counter-work` lacks that file, an input or a model inside the work folder, an input that is itself one of the output files, and an output name or the mark that is a folder or a link. A run locks the output folder before it looks into it and holds the lock until it ends, so a second run into it meanwhile is refused (where Python has `fcntl`, so not on Windows, and not on a disk that cannot lock, where the run says so and goes on). A run that fails or is stopped before its tables are moved leaves an earlier run's outputs as they were; one stopped during the video step leaves its tables, summary and PDF in place without the video and `README.txt`. What a stopped run leaves in the work folder, such as the partial video, stays there until the next run into that folder deletes it. The `ffmpeg` message says when the H.264 video was not made; an ffmpeg that fails or runs out of time does not cost the tables. An ffmpeg still running after ten times the video's length (at least five minutes) is stopped, and the video stays MPEG-4 Part 2. It reads to the end of the file or to `--max-minutes` (default 60), whichever comes first, and exits with an error if it could read no frame.

The browser page needs the files to come from a server, not from `file://`, and at the root of the host: the page asks for `/style.css`, `/js/`, `/vendor/`, `/models/` and `/assets/`, so a sub-path (such as a project page on GitHub Pages) breaks it. `web/serve.py` sends `.mjs` and `.wasm` with the content types the browser requires. If the page stays on its loading message or says the detector could not load, the runtime or the model is usually missing: run `fetch_assets.sh`. The message names the missing file when it is `vision_bundle.mjs`, the model or the WebAssembly loader `vision_wasm_internal.js`; when it is the `.wasm` file itself, it says that fetching the wasm failed (each tried by hand, once, with the file answering 404). That message comes back whenever the page has nothing else to say, also after "Sample clip" is pressed.

## Method

**Detector.** EfficientDet-Lite0 (float16) as published by MediaPipe, run by MediaPipe Tasks Vision 0.10.21 in the browser (GPU delegate, falling back to CPU) and by MediaPipe for Python offline. It knows the 80 classes of the COCO dataset. Settings: score threshold 0.4, at most 25 detections a frame.

**Groups.** `person`; vehicles are `car`, `truck`, `bus`, `motorcycle`, `bicycle` and `train`; everything else is "other".

**Tracker** (`counter/tracker.py`; the same rules in `web/js/seedeep.js`). A track's *hits* are the frame that started it plus every frame it was matched in since.
- a detection joins a track of the same class if their intersection over union is above 0.3; pairs are taken greedily, best overlap first and equal overlaps to the earlier track, so a detection can end up with its second-best track;
- a track counts as "seen" when it reaches 3 hits;
- a track is dropped after more than 12 frames in a row without a match (frames looked at: with `--every N` that spans 12 × N video frames, and in the browser it counts the frames it kept up with);
- boxes are drawn for tracks matched in the current frame that have at least 2 hits; "in frame" counts these.

**Line.** Crossings are counted for people and vehicles only, from a track's third hit on, when the centre of its box changes side between two matched frames.
- Browser: the line is a segment you drag, at any angle; a crossing counts only where the centre's new position is level with the segment (its projection falls between the two ends). "Down" is one side and "up" the other. Moving the line does not by itself count a crossing (both positions of a centre are judged against the line as it is now); each move leaves a "line moved" event (a press that moves nothing leaves none).
- Python: the line is horizontal and spans the frame; "down" means towards the bottom of the frame.

**Zone** (browser only): a polygon you drag. A track inside it from its third hit on is "in the zone"; the page counts entries and the mean stay, the stay of a track still inside counted up to the current moment (also while the detector has briefly lost it, until it is dropped) and that of a dropped track up to when it was last seen. Adding a new zone starts its figures again and leaves a "zone changed" line among the events. Entries and stays are per track, so a track that breaks shows as several short visits (see Limits). Every event in the CSV carries its pass. Removing a zone gives the tracks inside it a "left" event and keeps its figures in the CSV as an earlier zone; moving a corner leaves a "zone moved" event.

**Passes** (browser): when the sample clip loops (or the video's time otherwise jumps back; the page itself has no seek control), a new pass starts: the tracks start again, a track in the zone gets a "left" event where it was last seen, and the per-second rows start again under the next pass number. The totals seen, the crossings and the zone figures carry on across passes.

**Outputs.** The browser's CSV has a row for each second of video in which it looked at a frame, per pass (the most people, vehicles, other things and zone occupants in frame, the frames looked at, the pass), then the totals seen by class, the crossings, the figures of the zone and of any earlier zones, the line and zone as they were when the CSV was made, every event, the source, the detector and the time it was made. The Python counter writes `counts_per_second.csv`, `counts_per_minute.csv`, `seen_by_class.csv`, `crossings.csv`, `summary.json`, `summary.pdf`, `annotated.mp4` (unless `--no-video`) and a `README.txt` describing the files this run wrote; the time it records is in UTC.

## Limits

- **Accuracy has not been measured** on any labelled footage, by us. The annotated video is there so a person can check the counts.
- **"Seen" counts tracks, not people.** A thing the detector loses for more than 12 of the frames looked at and finds again is counted again; two things close together can be seen as one, and two that pass each other can swap tracks. On a busy scene the "seen" total can run far above the most in frame at once (132 against 9 in the run below), and in the browser a person crossing a zone can count as several entries with short stays: with the default zone on the sample clip, the mean stay the page showed stayed under a second on one fast machine; on a slow device (software graphics) it reached several seconds (5.5 and 12.1 s at most in two runs).
- The detector can miss things that are small, far away or partly hidden, and can give a thing the wrong class.
- **The browser counts only the frames it keeps up with.** It looks at each frame the video shows at most once (where the browser has `requestVideoFrameCallback`; elsewhere it looks whenever the video's clock moves, which can be the same frame twice), and a slower device skips frames, so the same video can give different counts on different devices. The Python counter looks at every frame up to `--max-minutes` (or one in every N with `--every`).
- Times in the Python outputs come from the frame rate the file states; a raw stream that states none (a bare `.h264`, say) is timed at whatever rate OpenCV reports for it, or 25 frames a second when it reports none (25 in the one case tried: a raw stream cut from the sample clip, which runs at 24 frames a second), so its seconds can be off.
- The browser line (a segment at any angle) and the Python line (horizontal, full width) are different rules, so their crossing counts on the same video can differ.
- A crossing is counted each time a centre changes side, with no margin: a thing standing on the line whose box shifts by a pixel or two can count crossings in both directions, in both programs.
- If something fails while the page looks at a frame (a lost graphics context, say), it says so on the video, logs the first error of each run of failed frames in the browser's console and tries again on the next frame; the message goes when a frame succeeds. With the video paused, a redraw that fails after the line or the zone is switched is not tried again until the next change; during a drag it is tried on every frame.
- **The browser can stay on a slow GPU.** The page takes the GPU delegate whenever MediaPipe can make one and does not move to the CPU when that is slow. In the one case tried by hand (headless Chrome made to use software WebGL, SwiftShader, on an Apple M1 Pro), it took the GPU delegate and made far fewer detections a second than with the Mac's own GPU.
- The PDF uses the standard Latin fonts, so a `--title`, `--line-text` or input file name in another script shows as boxes there; `summary.json` keeps it as written (UTF-8).
- `counter/` does not install on an Intel Mac: mediapipe 1.1.1 has wheels for macOS on Apple silicon, Linux and Windows, and none for macOS x86_64.
- It does not read faces, plates or text, and it does not know who anyone is: a track is a box that moved, nothing more.

## One run, one machine

The sample clip (`web/assets/sample_crosswalk_960.mp4`, 24.04 seconds, 960 by 540, 24 frames a second, 577 frames) with the command under "Run it", on an Apple M1 Pro (CPU, the versions in `counter/requirements.txt`):

| | |
|---|---|
| frames looked at | 577 |
| time per detection, mean | 38.8 ms |
| person tracks seen | 132 |
| vehicle tracks seen | 26 |
| line at 0.62: people down / up | 11 / 11 |
| line at 0.62: vehicles down / up | 0 / 3 |
| most people in frame at once | 9 (second 9) |

These are the counter's outputs, not checked against a person's count of the clip. The run gave 132 person tracks while it never showed more than 9 people in frame at once; how many distinct people appear in the clip was not counted, so how many of those tracks are one person's track broken and started again is not known. That is why "seen" is a count of tracks, not of people. The time per detection depends on the machine, and the counts can differ by a few with another mediapipe, OpenCV or CPU (one run on Linux x86_64 with mediapipe 0.10.32, OpenCV 5.0.0 and Python 3.13 gave 131 person and 27 vehicle tracks, people 13 / 13, vehicles 0 / 3, at most 10 people in frame).

## Privacy

The browser page loads the model, the runtime, the sample clip and any settings file from its own server (the browser may keep them in its cache). The camera and any opened file stay in the page; the counts are made into a file in the page. The page's Content-Security-Policy tells the browser to refuse scripts and requests to any other host. The line and zone you set are kept in the browser's local storage, under a key named for the page's path. "Copy these numbers" puts the speed figures, the delegate (GPU or CPU), the time and the browser's user-agent string on the clipboard, nothing else, and only when you press it. `count_video.py` makes no network request of its own.

## Settings and the test hook

A page can set the counter up from a file: `<body data-config="./config.json">`, where the JSON may hold `line` (a height fraction, or `{x1, y1, x2, y2}` in fractions of the frame), `zone` (3 or more points `{x, y}`; the first 8 are used), `classes` (the class names to count; others are ignored) and `start` (`"camera"` or `"sample"`). `web/index.html` has none. The file is used whole or not at all: if it cannot be fetched within 10 seconds, is not a JSON object, or has a field of the wrong shape (a `line` that is neither of the two, a `zone` of fewer than 3 points or with one of its first 8 points not `{x, y}` in numbers, `classes` that is not a list of names, a `start` that is neither), the page says so, shows that message whenever it has nothing else to say, and counts without it. A line kept in local storage is used only when its values are numbers, and a stored zone (3 or more points, cut to 8) only together with it; the values are clamped to the frame; anything else there is ignored. The script also keeps its state in `window.__seedeep` (among others: ready, frames, fps, infMs, delegate, line, zone) for tests such as `tests/web_check.mjs`.

## Licence

MIT for the code here (`LICENSE`). The model and the MediaPipe runtime are Google's, under the Apache License 2.0, and are downloaded by `fetch_assets.sh` rather than kept here; the sample clip is CC0. Details: `NOTICE.md`.
