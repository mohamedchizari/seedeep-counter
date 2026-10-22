#!/usr/bin/env bash
# Fetch the detector files this repository does not carry, and check each against its SHA-256:
#   models/efficientdet_lite0.tflite   EfficientDet-Lite0 (float16) as published by MediaPipe (Google, Apache 2.0)
#   web/models/                        the same model, for the browser page
#   web/vendor/                        MediaPipe Tasks Vision 0.10.21 (vision_bundle.mjs and wasm/), from the npm package
# usage: ./fetch_assets.sh        (run from anywhere; needs curl and shasum or sha256sum)
set -euo pipefail
cd "$(dirname "$0")"
MODEL_URL="https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite"
NPM="https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21"

sha() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
get() { # url, destination, expected sha256
  mkdir -p "$(dirname "$2")"
  if [ -f "$2" ] && [ "$(sha "$2")" = "$3" ]; then echo "ok (already here)  $2"; return; fi
  curl -fsSL "$1" -o "$2.part"
  got="$(sha "$2.part")"
  if [ "$got" != "$3" ]; then echo "CHECKSUM MISMATCH  $2: expected $3, got $got (the file was not kept)" >&2; rm -f "$2.part"; exit 1; fi
  mv "$2.part" "$2"; echo "ok                 $2"
}

get "$MODEL_URL" models/efficientdet_lite0.tflite 4b59100025bea1235a84c1038879a6cccc9f6c49f5e41144e91e74d99e780993
mkdir -p web/models && cp models/efficientdet_lite0.tflite web/models/efficientdet_lite0.tflite && echo "ok (copied)        web/models/efficientdet_lite0.tflite"
get "$NPM/vision_bundle.mjs" web/vendor/vision_bundle.mjs 40f4123dfcd75cfa58add046919cc6f52fde66f009ff582997c09ba54c6ba27c
get "$NPM/wasm/vision_wasm_internal.js" web/vendor/wasm/vision_wasm_internal.js 4a97e2520ba506c680ecd6ba6acfb146888afa0e2746d57f205352bc6ebb82eb
get "$NPM/wasm/vision_wasm_internal.wasm" web/vendor/wasm/vision_wasm_internal.wasm f00ec4731faa23b3e714d00e88d4d10e2df5c0a427d3a2b4ae6e3526fdd14ef7
get "$NPM/wasm/vision_wasm_nosimd_internal.js" web/vendor/wasm/vision_wasm_nosimd_internal.js 927def7b465c51b86e4b3060f93646aca4e27121f4b8fc0483786e407ea9cf1f
get "$NPM/wasm/vision_wasm_nosimd_internal.wasm" web/vendor/wasm/vision_wasm_nosimd_internal.wasm 3821ea9b1f7fb8c549ef2a064ef5c85750bf375c545a49fd6eea0df44a95f1f4
echo "all files present and checked"
