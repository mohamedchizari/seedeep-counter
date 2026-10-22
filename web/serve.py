#!/usr/bin/env python3
"""Serve web/ on http://127.0.0.1:8000 with the content types a module script and WebAssembly need
(some Python versions send .mjs and .wasm as application/octet-stream, which the browser refuses for a module).
usage: python web/serve.py [port]"""
import http.server, os, sys
port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
os.chdir(os.path.dirname(os.path.abspath(__file__)))
H = http.server.SimpleHTTPRequestHandler
H.extensions_map.update({'.mjs': 'text/javascript', '.js': 'text/javascript', '.wasm': 'application/wasm', '.tflite': 'application/octet-stream', '.mp4': 'video/mp4'})
print(f'serving {os.getcwd()} on http://127.0.0.1:{port}/ (Ctrl-C to stop)', flush=True)
http.server.ThreadingHTTPServer(('127.0.0.1', port), H).serve_forever()
