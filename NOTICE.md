# Third-party parts

The code in this repository (`counter/`, `tests/`, `web/js/seedeep.js`, `web/index.html`, `web/style.css`, `web/serve.py`, `fetch_assets.sh`) is under the MIT License (`LICENSE`).

These parts are not ours and keep their own licences:

| Part | Where it comes from | Licence | In this repository? |
|---|---|---|---|
| EfficientDet-Lite0 (float16), the object detection model | MediaPipe, Google LLC: `storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/` | Apache License 2.0 | No. `fetch_assets.sh` downloads it to `models/` and `web/models/`, unmodified, and checks its SHA-256. |
| MediaPipe Tasks Vision 0.10.21 (`vision_bundle.mjs`, `wasm/`) | Google LLC, npm package `@mediapipe/tasks-vision`, served by cdn.jsdelivr.net | Apache License 2.0 | No. `fetch_assets.sh` downloads it to `web/vendor/`, unmodified, and checks each file's SHA-256. |
| MediaPipe for Python (`mediapipe`), OpenCV (contrib build), NumPy, ReportLab | installed with `pip install -r counter/requirements.txt` | Apache 2.0 (MediaPipe, OpenCV), BSD (NumPy, ReportLab) | No. |
| Playwright (`playwright-core`), used only by `tests/web_check.mjs` | installed with `npm install playwright-core` | Apache License 2.0 | No. |
| `web/assets/sample_crosswalk_960.mp4` | a 24-second cut, scaled to 960 by 540, of DiagonalCrosswalkYongeDundas.webm by Raysonho, Wikimedia Commons | CC0 1.0 (public domain dedication) | Yes. |

The text of the Apache License 2.0 is in `third_party/APACHE-2.0.txt`.
