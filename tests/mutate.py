#!/usr/bin/env python3
"""Checks that the tests notice a change to the tracker rules listed below: copies the repository, makes one change at a
time in counter/tracker.py, runs the tests, and prints whether they failed (caught) or passed (missed). The list is not
every possible change; a change it does not list can go unnoticed.
usage: python tests/mutate.py      (from the repository root)"""
import os, shutil, subprocess, sys, tempfile
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
SRC = open(os.path.join(ROOT, 'counter', 'tracker.py'), encoding='utf-8').read()
MUTATIONS = [
    ("if s > 0.3:", "if s > 0.1:"),                                        # the IoU needed to match
    ("if s > 0.3:", "if s > 0.5:"),
    ("if s > 0.3:", "if s >= 0.3:"),
    ("; t['misses'] = 0\n", "\n"),                                       # a match resets the misses
    ("if t['misses'] == 0 and t['hits'] >= 2]", "if t['misses'] <= 1 and t['hits'] >= 2]"),  # drawn only when matched now
    ("t['hits'] >= 3 and g in", "t['hits'] >= 2 and g in"),                # crossings from the third hit
    ("t['misses'] <= 12]", "t['misses'] <= 11]"),                          # dropped after 12 misses
    ("t['hits'] >= 3: t['counted']", "t['hits'] >= 2: t['counted']"),      # seen at 3 hits
    ("pairs.sort(key=lambda p: -p[0])", "pairs.sort(key=lambda p: p[0])"),  # best overlap first
    ("pairs.sort(key=lambda p: -p[0])", "pairs.sort(reverse=True)"),       # equal overlaps to the earlier track
    ("if cy0 < line_y <= cy1", "if cy0 <= line_y <= cy1"),                 # the crossing test
    ("t['hits'] >= 2]", "t['hits'] >= 1]"),                                # what is drawn
    ("if d['cls'] == t['cls']:", "if True:"),                              # same class only
    ("g in ('person', 'vehicle')", "g in ('person', 'vehicle', 'other')"),  # who crosses
]
caught = 0
for old, new in MUTATIONS:
    assert SRC.count(old) == 1, old
    d = tempfile.mkdtemp(prefix='mutate_')
    try:
        shutil.copytree(ROOT, os.path.join(d, 'r'), ignore=shutil.ignore_patterns('.git', '.venv', 'venv', '__pycache__', 'models', 'vendor', 'out', '*.mp4'))
        open(os.path.join(d, 'r', 'counter', 'tracker.py'), 'w', encoding='utf-8').write(SRC.replace(old, new))
        r = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests'], cwd=os.path.join(d, 'r'), capture_output=True, text=True)  # -B and no copied __pycache__: a compiled copy of the unchanged file could be run instead (seen once)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    hit = r.returncode != 0; caught += hit
    print('caught' if hit else 'MISSED', '|', new.strip() or '(the line ending removed: ' + old.strip() + ')')
print(f'{caught} of {len(MUTATIONS)} mutations caught')
sys.exit(0 if caught == len(MUTATIONS) else 1)
