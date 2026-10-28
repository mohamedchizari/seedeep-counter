#!/usr/bin/env python3
"""Checks that the tests notice a change to the rules listed below (the tracker in counter/tracker.py, the output folder's
mark in counter/output_mark.py and its use in counter/count_video.py, the frame-rate checks in counter/video_timing.py and
their use in counter/count_video.py, and the file modes in fetch_assets.sh): copies the
repository, makes one change at a time, runs the tests, and prints whether they failed (caught) or passed (missed). The list
is not every possible change; a change it does not list can go unnoticed. The changes to count_video.py are caught only by
tests that run it, which need the counter's own packages (counter/requirements.txt); without them those changes are listed
as skipped, not as caught. The test for a file that cannot be read is skipped for root (root reads it anyway), and so is its change.
usage: python tests/mutate.py      (from the repository root)"""
import importlib.util, os, shutil, subprocess, sys, tempfile
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
NEEDS = ('cv2', 'numpy', 'mediapipe', 'reportlab')  # what tests/test_count_video_mark.py needs to run
TRACKER = os.path.join('counter', 'tracker.py')
MUTATIONS = [(TRACKER, old, new) for old, new in [
    ("if s > 0.3:", "if s > 0.1:"),                                        # the IoU needed to match
    ("if s > 0.3:", "if s > 0.5:"),
    ("if s > 0.3:", "if s >= 0.3:"),
    ("; t['misses'] = 0\n", "\n"),                                       # a match resets the misses
    ("if t['misses'] == 0 and t['hits'] >= 2]", "if t['misses'] <= 1 and t['hits'] >= 2]"),  # drawn only when matched now
    ("t['hits'] >= 3 and g in", "t['hits'] >= 2 and g in"),                # crossings from the third hit
    ("t['hits'] >= 3 and g in", "t['hits'] >= 4 and g in"),
    ("t['misses'] <= 12]", "t['misses'] <= 11]"),                          # dropped after more than 12 misses
    ("t['hits'] >= 3: t['counted']", "t['hits'] >= 2: t['counted']"),      # seen at 3 hits
    ("pairs.sort(key=lambda p: -p[0])", "pairs.sort(key=lambda p: p[0])"),  # best overlap first
    ("pairs.sort(key=lambda p: -p[0])", "pairs.sort(reverse=True)"),       # equal overlaps to the earlier track
    ("if cy0 < line_y <= cy1", "if cy0 <= line_y <= cy1"),                 # the crossing test
    ("t['hits'] >= 2]", "t['hits'] >= 1]"),                                # what is drawn
    ("if d['cls'] == t['cls']:", "if True:"),                              # same class only
    ("g in ('person', 'vehicle')", "g in ('person', 'vehicle', 'other')"),  # who crosses
    ("t['score'] = d['score']; ", ""),                                      # the score follows the match
    ("'hits': 1, 'misses': 0, 'counted': False}", "'hits': 1, 'misses': 1, 'counted': False}"),  # a new track starts with no misses
    ("self.events.append((t_s, g, t['cls'], 'down'", "self.events.append((0, g, t['cls'], 'down'"),  # a crossing's time
    ("'down', t['id']))", "'down', 0))"),                                  # a crossing's track
]] + [(os.path.join('counter', 'output_mark.py'), old, new) for old, new in [
    ("    return 'same' if content == tuple(recorded) else 'other'\n", "    return 'same'\n"),  # same size is not enough: the content is compared
    ("elif len(fields) == 3 and all(", "elif len(fields) in (3, 4) and all("),         # rows of other lengths are ignored
    ("elif len(fields) == 3 and all(", "elif len(fields) in (3, 5) and all("),         # also the five-field form no release wrote
    ("_HEX = re.compile(r'[0-9a-f]{64}')", "_HEX = re.compile(r'[0-9a-fA-F]+')"),       # only a full lower-case SHA-256
    ("    if recorded[0] == 'time' and st.st_mtime_ns != recorded[2]:\n", "    if False:\n"),  # an earlier mark: the modified time counts
    ("    if recorded[0] == 'time' and st.st_mtime_ns != recorded[2]:\n        return 'other'\n",
     "    if recorded[0] == 'time':\n        return 'same' if st.st_mtime_ns == recorded[2] else 'other'\n"),  # and its files are read too (first block)
    ("if not stat.S_ISREG(st.st_mode) or st.st_size != recorded[1]:", "if not stat.S_ISREG(st.st_mode):"),  # and the size
    ("        size += len(b)\n", "        size += 1\n"),                                # the recorded size is the bytes hashed
    ("        return 'unreadable'\n", "        return 'other'\n"),                      # a file that cannot be read is told apart
]] + [(os.path.join('counter', 'video_timing.py'), old, new) for old, new in [
    ("    if not LOWEST <= reported <= HIGHEST:\n", "    if False:\n"),                   # a rate outside 0.1 to 1,000 a second is refused
    ("    if not LOWEST <= reported <= HIGHEST:\n", "    if reported < LOWEST or reported > HIGHEST:\n"),  # also NaN, which no comparison holds for
]] + [('fetch_assets.sh', old, new) for old, new in [
    ("$(( 0666 & ~0$(umask) ))", "$(( 0644 ))"),                                         # downloads follow the umask
    ('  chmod "$FILE_MODE" "$tmp"\n  mv "$tmp" "$2"', '  mv "$tmp" "$2"'),              # each fetched file is given that mode
    ('  chmod "$FILE_MODE" "$tmp"\n  mv "$tmp" web/models', '  mv "$tmp" web/models'),  # and so is the copy in web/models
]] + [(os.path.join('counter', 'count_video.py'), old, new, *(rest or (NEEDS,))) for old, new, *rest in [
    ("        return matches_mark(os.path.join(a.out, name), marks.get(name, ()))\n", "        return True\n"),  # a run takes only what its mark lists, unchanged
    ("        try: return record_of_fd(fd)\n", "        try: return ('sha256', 0, '0' * 64)\n"),  # the mark records each moved file's own size and hash
    ("    if unread:\n", "    if False:\n", NEEDS + ('not root',)),                    # a listed file that cannot be read is named (root reads it anyway)
    ("    try: fps = effective_fps(cap.get(cv2.CAP_PROP_FPS))\n", "    try: fps = cap.get(cv2.CAP_PROP_FPS) or 25\n"),  # the reported frame rate is checked before anything is written
]]
def unmet(needs):
    """What the test that catches a change needs and this run lacks: a package, or a user other than root."""
    out = []
    for m in needs:
        if m == 'not root':
            if hasattr(os, 'geteuid') and os.geteuid() == 0: out.append('a user other than root')
        elif importlib.util.find_spec(m) is None: out.append(m)
    return out

first = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests'], cwd=ROOT, capture_output=True, text=True)
if first.returncode != 0: print('the tests fail on the unchanged files; fix that first'); sys.exit(2)
caught = skipped = 0
for rel, old, new, *needs in MUTATIONS:
    SRC = open(os.path.join(ROOT, rel), encoding='utf-8').read()
    assert SRC.count(old) == 1, old
    missing = unmet(needs[0] if needs else ())
    if missing:
        skipped += 1; print('skipped (needs ' + ', '.join(missing) + ') |', new.strip()); continue
    d = tempfile.mkdtemp(prefix='mutate_')
    try:
        shutil.copytree(ROOT, os.path.join(d, 'r'), ignore=shutil.ignore_patterns('.git', '.venv', 'venv', '__pycache__', 'node_modules', 'models', 'vendor', 'out', '*.mp4'))
        open(os.path.join(d, 'r', rel), 'w', encoding='utf-8').write(SRC.replace(old, new))
        r = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests'], cwd=os.path.join(d, 'r'), capture_output=True, text=True)  # -B and no copied __pycache__, so the changed file is the one that runs
    finally:
        shutil.rmtree(d, ignore_errors=True)
    hit = r.returncode != 0; caught += hit
    print('caught' if hit else 'MISSED', '|', new.strip() or '(removed: ' + old.strip() + ')')
print(f'{caught} of {len(MUTATIONS)} mutations caught' + (f', {skipped} skipped' if skipped else ''))
sys.exit(0 if caught + skipped == len(MUTATIONS) else 1)
