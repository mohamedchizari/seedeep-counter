#!/usr/bin/env python3
"""Serve web/ on http://127.0.0.1:8000 with the content types a module script and WebAssembly need
(some Python versions send .mjs and .wasm as application/octet-stream, which the browser refuses for a module).
usage: python web/serve.py [port]"""
import http.server, os, sys
try: port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
except ValueError: port = -1
if not 0 <= port <= 65535: sys.exit('usage: python web/serve.py [port], a whole number from 0 to 65535 (0: any free port): ' + sys.argv[1])
os.chdir(os.path.dirname(os.path.abspath(__file__)))
H = http.server.SimpleHTTPRequestHandler
H.extensions_map.update({'.mjs': 'text/javascript', '.js': 'text/javascript', '.wasm': 'application/wasm', '.tflite': 'application/octet-stream', '.mp4': 'video/mp4'})
try: server = http.server.ThreadingHTTPServer(('127.0.0.1', port), H)
except OSError as e: sys.exit(f'cannot serve on port {port}: {e.strerror or e}')  # a port already taken, say: nothing is served
port = server.server_address[1]  # the one the system gave, for port 0
try:
    print(f'serving {os.path.basename(os.getcwd())}/ on http://127.0.0.1:{port}/ (Ctrl-C to stop)', flush=True)  # the folder by its name only
    server.serve_forever()
except KeyboardInterrupt: print('stopped')
