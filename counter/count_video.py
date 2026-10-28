#!/usr/bin/env python3
"""SeeDeep video counter: counts people, vehicles and other things in a video file, offline. Same detector as the browser
page in web/ (MediaPipe EfficientDet-Lite0, score 0.4, 25 per frame), same tracker rules (tracker.py), one horizontal line.
usage: python count_video.py IN.mp4 OUT_DIR [--line 0.62] [--every 1] [--max-minutes 60] [--title "name"] [--line-text "the doorway"]
       [--model PATH] [--ffmpeg NAME_OR_PATH] [--no-video]
writes into OUT_DIR/.seedeep-counter-work: _annotated_raw.mp4 while it reads, then counts_per_second.csv, counts_per_minute.csv,
seen_by_class.csv, crossings.csv, summary.json, summary.pdf, which it moves into OUT_DIR at once (replacing an earlier run's);
then annotated.mp4 (made from the raw file) and README.txt, moved last. .seedeep-counter-output in OUT_DIR lists what it put there.
The model is fetched by ../fetch_assets.sh; ffmpeg (optional) re-encodes the annotated video to H.264."""
import os, signal, sys

GOT = []  # the stops this run has had, noted from the start: a library can swallow the exception a signal raised, so the run also checks this

def _early_int(n, f):
    GOT.append(n); raise KeyboardInterrupt

def ctrl_c():
    """Ctrl-C: one line instead of a traceback, then the end a parent expects from Ctrl-C (killed by SIGINT; exit 130 on Windows)."""
    print('stopped (Ctrl-C)', flush=True)
    if os.name == 'posix': signal.signal(signal.SIGINT, signal.SIG_DFL); os.kill(os.getpid(), signal.SIGINT)
    sys.exit(130)

try:
    if signal.getsignal(signal.SIGINT) is signal.default_int_handler: signal.signal(signal.SIGINT, _early_int)  # an ignored Ctrl-C stays ignored
    import argparse, contextlib, csv, hashlib, json, stat, time, subprocess, datetime, shutil
    try:
        import fcntl  # locks the output folder while a run writes into it; Windows has no fcntl, and there a second run is not stopped
    except ImportError:
        fcntl = None
    import cv2, numpy as np
    import mediapipe as mp
    from mediapipe.tasks import python as mptasks
    from mediapipe.tasks.python import vision
    try:
        import reportlab  # used for summary.pdf; checked here so a missing one stops the run before it starts
    except ImportError:
        reportlab = None
    from output_mark import check, format_row, matches_mark, parse_records, record_of_fd
    from tracker import Tracker, group, VEHICLES
    from video_timing import effective_fps
except BaseException as e:  # a Ctrl-C during the imports ends the run, also when a library swallowed or replaced its exception
    if GOT or isinstance(e, KeyboardInterrupt): ctrl_c()
    raise
if GOT: ctrl_c()

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, '..', 'models', 'efficientdet_lite0.tflite')
COL = {'person': (22, 165, 242), 'vehicle': (166, 163, 31), 'other': (168, 152, 138)}  # BGR of #f2a516, #1fa3a6, #8a98a8
FFMPEG_MIN_TIMEOUT_S = 300  # an ffmpeg still running after this, or after ten times the video's length if that is longer, is stopped
PDF_CLASSES = 16  # classes listed by name in summary.pdf; seen_by_class.csv has them all
MARK = '.seedeep-counter-output'  # written in every output folder: it lists the files this program put there, with their sizes and SHA-256 content hashes
WORK = '.seedeep-counter-work'  # this run's files, until they are moved into place
WORK_TAG = 'made-by-count_video.txt'  # written in every work folder; a run deletes a work folder only when this file is in it or the folder is empty
TAG_TEXT = 'Made by count_video.py for one run. The next run into the folder above deletes this folder.\n'
LITE0_SHA256 = '4b59100025bea1235a84c1038879a6cccc9f6c49f5e41144e91e74d99e780993'  # efficientdet_lite0.tflite, as fetch_assets.sh checks it

def listed(out):
    """The files the mark in OUT_DIR says this program put there; {} without a mark, None when it cannot be read."""
    try: rows = open(os.path.join(out, MARK), encoding='utf-8', errors='replace').read().splitlines()[1:]
    except FileNotFoundError: return {}
    except OSError: return None
    return parse_records(rows)

def inside(path, folder):
    """True if path is folder or in it, compared as files, so a different spelling or a link is caught."""
    if not os.path.isdir(folder): return False
    p = os.path.realpath(path)
    while True:
        if os.path.exists(p) and os.path.samefile(p, folder): return True
        if os.path.dirname(p) == p: return False
        p = os.path.dirname(p)

def has_entries(folder):
    """True if the folder holds anything, or cannot be listed."""
    try: return bool(os.listdir(folder))
    except OSError: return True

# Where the system can (not on Windows), a work folder is opened without following a link and what is in it is reached through
# that descriptor, so a link put in its place, or in it, is never followed; elsewhere its path is checked before each use.
DIR_FD = (hasattr(os, 'O_DIRECTORY') and hasattr(os, 'O_NOFOLLOW') and os.scandir in os.supports_fd and os.stat in os.supports_follow_symlinks
          and all(f in os.supports_dir_fd for f in (os.open, os.stat, os.unlink, os.rmdir, os.rename)))
DIR_FLAGS = os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_NOFOLLOW', 0)

def write_tag(work, fd=None):
    """Writes the work folder's tag file, never through a link: a link in its place is deleted first, and the file is opened
    without following one (where the system can); with fd, inside the folder open as fd."""
    if fd is not None:
        with contextlib.suppress(FileNotFoundError):
            if stat.S_ISLNK(os.stat(WORK_TAG, dir_fd=fd, follow_symlinks=False).st_mode): os.unlink(WORK_TAG, dir_fd=fd)
        t = os.open(WORK_TAG, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o666, dir_fd=fd)
    else:
        tag = os.path.join(work, WORK_TAG)
        if os.path.islink(tag): os.remove(tag)
        t = os.open(tag, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, 'O_NOFOLLOW', 0), 0o666)
    with os.fdopen(t, 'w', encoding='utf-8') as f: f.write(TAG_TEXT)

def _empty(fd, keep=None):
    """Deletes everything in the folder open as fd but keep, through descriptors: a link in it is deleted, never followed."""
    for e in list(os.scandir(fd)):
        if e.name == keep: continue
        if e.is_dir(follow_symlinks=False):
            sub = os.open(e.name, DIR_FLAGS, dir_fd=fd)
            try: _empty(sub)
            finally: os.close(sub)
            os.rmdir(e.name, dir_fd=fd)
        else: os.unlink(e.name, dir_fd=fd)

def clear_work(work, tries=3, expect=None):
    """Deletes a work folder this program made: everything else first, its tag last, then the empty folder. A stop part way
    leaves the folder tagged or empty, and a later run deletes either. Where the system can, the folder is opened without
    following a link, must be the folder expect names (its device and inode) when given, and is emptied through that
    descriptor, so nothing outside it is touched. If deleting fails (a file that appears meanwhile, such as a .DS_Store, or one
    that cannot be deleted), the tag is written back and it is tried again; False if the folder stays."""
    for _ in range(tries):
        if DIR_FD:
            try: fd = os.open(work, DIR_FLAGS)
            except FileNotFoundError: return True
            except OSError: return False  # a link, not a folder, or one that cannot be opened: left alone
            try:
                st = os.fstat(fd)
                if expect is not None and (st.st_dev, st.st_ino) != expect: return False
                _empty(fd, keep=WORK_TAG)
                with contextlib.suppress(FileNotFoundError): os.unlink(WORK_TAG, dir_fd=fd)
                now = os.lstat(work)
                if (now.st_dev, now.st_ino) != (st.st_dev, st.st_ino): return False  # replaced meanwhile: left alone
                os.rmdir(work); return True
            except OSError:
                with contextlib.suppress(OSError): write_tag(work, fd)
            finally: os.close(fd)
        else:
            tag = os.path.join(work, WORK_TAG)
            try:
                st = os.lstat(work)
                if not stat.S_ISDIR(st.st_mode) or (expect is not None and (st.st_dev, st.st_ino) != expect): return False
                for e in os.scandir(work):
                    if e.name == WORK_TAG: continue
                    if e.is_dir(follow_symlinks=False): shutil.rmtree(e.path)
                    else: os.remove(e.path)
                if os.path.lexists(tag): os.remove(tag)
                os.rmdir(work); return True
            except OSError:
                if not os.path.lexists(work): return True
                with contextlib.suppress(OSError): write_tag(work)
    return False

STOPS = [getattr(signal, n) for n in ('SIGINT', 'SIGTERM', 'SIGHUP') if hasattr(signal, n)]  # Windows has no SIGHUP

def stop_here(n, f):
    """SIGTERM and SIGHUP end the run as Ctrl-C does: by an exception, so the ffmpeg it started is stopped (see the video step),
    then (outside Windows) by the signal itself, at the end of the file."""
    GOT.append(n); raise SystemExit(128 + n)

def on_ctrl_c(n, f):
    GOT.append(n); raise KeyboardInterrupt

def check_stop():
    """Ends the run if a stop came whose exception did not (see GOT)."""
    if GOT: raise KeyboardInterrupt if GOT[0] == signal.SIGINT else SystemExit(128 + GOT[0])

@contextlib.contextmanager
def no_stop():
    """A Ctrl-C, SIGTERM or SIGHUP while files are moved into place takes effect when the move is done, so it cannot leave a mix."""
    got = []; old = {s: signal.signal(s, lambda n, f: got.append(n)) for s in STOPS if signal.getsignal(s) != signal.SIG_IGN}  # one ignored by the caller stays ignored
    try: yield
    finally:
        for s, h in old.items(): signal.signal(s, h if h is not None else signal.SIG_DFL)
        if got: signal.raise_signal(got[0])

def draw(frame, tracks, line_y):
    h, w = frame.shape[:2]; s = max(1, round(w / 960)); fs = max(0.4, 0.42 * w / 960); ft = 1 if w < 1500 else 2; pad = 3 * s
    if line_y:
        for x in range(0, w, 24 * s): cv2.line(frame, (x, int(line_y)), (min(w, x + 14 * s), int(line_y)), (240, 237, 233), 2 * s)
    for t in tracks:
        x, y, bw, bh = [int(v) for v in t['box']]; g = group(t['cls']); c = COL[g]; k = int(min(bw, bh) * 0.22)
        for (ax, ay, dx, dy) in [(x, y, 1, 1), (x + bw, y, -1, 1), (x, y + bh, 1, -1), (x + bw, y + bh, -1, -1)]:
            cv2.line(frame, (ax, ay), (ax + dx * k, ay), c, 2 * s); cv2.line(frame, (ax, ay), (ax, ay + dy * k), c, 2 * s)
        label = f"{t['cls']} {round(t['score'] * 100)}"; (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, ft)
        top = max(0, y - th - 2 * pad)
        cv2.rectangle(frame, (x, top), (x + tw + 2 * pad, top + th + 2 * pad), c, -1)
        cv2.putText(frame, label, (x + pad, top + th + pad - 1), cv2.FONT_HERSHEY_SIMPLEX, fs, (28, 22, 16), ft, cv2.LINE_AA)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('inp'); ap.add_argument('out'); ap.add_argument('--line', type=float, default=0.62, help='line as a fraction of the height; 0 = no line')
    ap.add_argument('--every', type=int, default=1); ap.add_argument('--max-minutes', type=float, default=60); ap.add_argument('--title', default='', help='a name printed on the summary'); ap.add_argument('--line-text', default='')
    ap.add_argument('--model', default=MODEL); ap.add_argument('--ffmpeg', default='ffmpeg', help="ffmpeg's name on the PATH or its path; '' skips re-encoding")
    ap.add_argument('--no-video', action='store_true'); a = ap.parse_args()
    for sig in STOPS[1:]:
        if signal.getsignal(sig) == signal.SIG_DFL: signal.signal(sig, stop_here)  # one the caller ignores stays ignored: nohup ignores SIGHUP
    if signal.getsignal(signal.SIGINT) in (signal.default_int_handler, _early_int): signal.signal(signal.SIGINT, on_ctrl_c)  # an ignored Ctrl-C stays ignored
    for v, what in ((a.inp, 'the input file name'), (a.out, 'the OUT_DIR name'), (a.model, 'the --model path'), (a.ffmpeg, 'the --ffmpeg name'), (a.title, '--title'), (a.line_text, '--line-text')):
        try: v.encode('utf-8')
        except UnicodeEncodeError: ap.error(what + ' is not valid UTF-8 (OpenCV or the outputs would fail on it) - change it')
    if a.every < 1: ap.error('--every must be 1 or more')
    if not 0 <= a.line < 1: ap.error('--line must be 0 (no line) or a fraction of the height below 1')
    if not a.max_minutes > 0: ap.error('--max-minutes must be a number above 0')  # written so that nan is refused too
    if not os.path.exists(a.model): print('model not found:', a.model, '(run ./fetch_assets.sh from the repository root)'); sys.exit(2)
    if reportlab is None: print('ReportLab is not installed (pip install -r counter/requirements.txt)'); sys.exit(2)
    OUTPUTS = ('counts_per_second.csv', 'counts_per_minute.csv', 'seen_by_class.csv', 'crossings.csv', 'summary.json', 'summary.pdf', 'annotated.mp4', 'README.txt', '_annotated_raw.mp4')
    if not a.out or (os.path.lexists(a.out) and not os.path.isdir(a.out)): ap.error('OUT_DIR must be a folder: ' + repr(a.out))
    if os.path.exists(a.inp) and not stat.S_ISREG(os.stat(a.inp).st_mode):  # a FIFO with no writer would hold OpenCV, and the run, past any stop: every FIFO is refused
        print(a.inp, 'is not a regular file (a folder, a FIFO, which a pipe is too, or a device) - give a video file'); sys.exit(2)
    cap = cv2.VideoCapture(a.inp)
    if not cap.isOpened(): print('cannot open', a.inp); sys.exit(2)
    try: fps = effective_fps(cap.get(cv2.CAP_PROP_FPS))
    except ValueError as e:
        cap.release(); print(a.inp, '-', str(e)); sys.exit(2)
    opts = vision.ObjectDetectorOptions(base_options=mptasks.BaseOptions(model_asset_path=a.model), running_mode=vision.RunningMode.VIDEO, score_threshold=0.4, max_results=25)
    try: det = vision.ObjectDetector.create_from_options(opts)  # before the output folder is made, so a bad model leaves no empty folder
    except Exception as e: print('the detector could not be made from', a.model, '-', str(e)[:200]); sys.exit(2)
    check_stop(); lite0 = hashlib.sha256(open(a.model, 'rb').read()).hexdigest() == LITE0_SHA256  # only this model is named EfficientDet-Lite0 in the outputs
    model = os.path.basename(a.model) + (' (EfficientDet-Lite0, float16)' if lite0 else '')
    out_abs = up = os.path.realpath(a.out); made_top = None  # the folders this run makes (OUT_DIR and any missing above it) are deleted
    if WORK in out_abs.split(os.sep): print(out_abs, 'is inside a', WORK, 'folder, which the run that made it deletes - choose another output folder'); sys.exit(2)
    while not os.path.lexists(up) and os.path.dirname(up) != up: made_top, up = up, os.path.dirname(up)  # again if it reads no frame
    try: os.makedirs(out_abs, exist_ok=True)  # the resolved path, the same one recorded above (x/../y makes y, not x)
    except OSError as e: ap.error('OUT_DIR cannot be made: ' + str(e))
    a.out = out_abs
    if not os.access(a.out, os.W_OK | os.X_OK): print(a.out, 'cannot be written or searched (its permissions) - check them, or choose another output folder'); sys.exit(2)
    if fcntl:  # taken before the folder is looked at and held until this run ends (the system lets go of it even if the run is killed), so a second run into OUT_DIR meanwhile is refused
        try: lock = os.open(a.out, os.O_RDONLY)
        except OSError as e: print(a.out, 'cannot be opened:', e.strerror or e, '- check its permissions, or choose another output folder'); sys.exit(2)
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: print('another run is writing into', a.out, '- wait for it to end, or choose another output folder'); sys.exit(2)
        except OSError: print('this disk cannot lock', a.out, '- the run goes on, but a second run into it at the same time is not stopped', flush=True)
    work, mark = os.path.join(a.out, WORK), os.path.join(a.out, MARK)  # this run's files go into work, then into OUT_DIR when they are finished
    present = [n for n in OUTPUTS if os.path.lexists(os.path.join(a.out, n))]
    odd = [n for n in present if os.path.isdir(os.path.join(a.out, n)) or os.path.islink(os.path.join(a.out, n))]
    if os.path.lexists(mark) and (os.path.islink(mark) or not os.path.isfile(mark)): odd.append(MARK)
    if odd: print(a.out, 'holds a folder or a link named like one of the outputs or the mark (' + ', '.join(odd) + ') - choose another output folder'); sys.exit(2)
    if os.path.exists(a.inp) and any(os.path.samefile(a.inp, os.path.join(a.out, n)) for n in present):  # the same file however it is spelled
        print('the input is one of the files this run writes in', a.out, '- choose another output folder'); sys.exit(2)
    marks = listed(a.out)
    if marks is None: print(a.out, 'holds', MARK, 'but it cannot be read - check its permissions, or choose another output folder'); sys.exit(2)
    def ours(name):
        return matches_mark(os.path.join(a.out, name), marks.get(name, ()))
    foreign = [n for n in present if not ours(n)]
    unread = [n for n in foreign if check(os.path.join(a.out, n), marks.get(n, ())) == 'unreadable']
    if unread:
        print(a.out, 'holds', ', '.join(unread), 'that its', MARK, 'file lists but that cannot be read now (its permissions, say, or a file kept only in the cloud while offline) - make them readable, or choose another output folder'); sys.exit(2)
    if foreign:
        print(a.out, 'holds', ', '.join(foreign), 'that this program did not put there or that changed since (its', MARK, 'file lists what it put there) - move them, or choose an empty or new output folder'); sys.exit(2)
    ff = shutil.which(a.ffmpeg) if a.ffmpeg else None
    for path, what in ((a.inp, 'the input'), (a.model, 'the model'), (ff, 'the ffmpeg')):
        if path and inside(path, work): print(what, 'is inside', work, 'which a run into', a.out, 'deletes - move it out first'); sys.exit(2)
    tag = os.path.join(work, WORK_TAG); tagged = os.path.isfile(tag) and not os.path.islink(tag)
    if os.path.lexists(work) and (os.path.islink(work) or not os.path.isdir(work) or not (tagged or not has_entries(work))):
        print(a.out, 'holds', WORK, 'that is a link, not a folder, a folder that cannot be read, or a folder with files but without', WORK_TAG, 'as a plain file (left by an earlier version, or not made by this program) - check it and delete it, or choose another output folder'); sys.exit(2)
    tr = Tracker()
    if os.path.isdir(work) and not clear_work(work):  # what a stopped run left: tagged or empty (checked above)
        print(work, 'could not be deleted (it cannot be listed, or a file in it cannot be deleted or keeps coming back) - check it, or choose another output folder'); sys.exit(2)
    try:
        os.mkdir(work, 0o700)  # its owner's only (outside Windows): another user cannot put a file or a link in it
        wfd = os.open(work, DIR_FLAGS) if DIR_FD else None  # the folder this run made, held open: its files are reached through it
        st = os.fstat(wfd) if wfd is not None else os.lstat(work); work_id = (st.st_dev, st.st_ino)
        write_tag(work, wfd)
    except OSError as e: print(a.out, 'cannot be written:', e.strerror or e, '- choose another output folder'); sys.exit(2)
    def still_ours():
        """Ends the run (exit 2) if the work folder is no longer the one this run made (a link put in its place, say)."""
        try: st = os.lstat(work); same = stat.S_ISDIR(st.st_mode) and (st.st_dev, st.st_ino) == work_id
        except OSError: same = False
        if not same:
            print(work, 'was replaced during this run (it is no longer the folder this run made) - nothing more is written there or deleted; check', a.out, flush=True); sys.exit(2)
    def replaced(name):
        """Ends the run (exit 2) when a name this run uses in the work folder is not the plain file it made (a link, say)."""
        print(work, 'holds', name, 'that is not a plain file this run made (a link put there during the run, say) - nothing more is moved, written or deleted there; check', a.out, flush=True); sys.exit(2)
    def wopen(name, mode='w', check=True, **kw):  # a file in this run's work folder, opened through the folder itself where the system can, never through a link
        if check or wfd is None: still_ours()
        if wfd is None:
            if os.path.lexists(os.path.join(work, name)) and not wplain(name): replaced(name)
            return open(os.path.join(work, name), mode, **kw)
        try: return open(name, mode, opener=lambda p, fl: os.open(p, fl | os.O_NOFOLLOW, 0o666, dir_fd=wfd), **kw)  # made as the umask allows
        except OSError:
            if wexists(name) and not wplain(name): replaced(name)
            raise
    def wexists(name):
        try:
            if wfd is None: os.lstat(os.path.join(work, name))
            else: os.stat(name, dir_fd=wfd, follow_symlinks=False)
            return True
        except FileNotFoundError: return False
    def wplain(name):  # a plain file, not a link or a folder, under that name in the work folder
        try: st = os.lstat(os.path.join(work, name)) if wfd is None else os.stat(name, dir_fd=wfd, follow_symlinks=False)
        except FileNotFoundError: return False
        return stat.S_ISREG(st.st_mode)
    def wstamp(name):  # its size and SHA-256, read through the work folder without following a link; a move into OUT_DIR, and a later copy of OUT_DIR, keep both
        flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_BINARY', 0)
        fd = os.open(os.path.join(work, name), flags) if wfd is None else os.open(name, flags, dir_fd=wfd)
        try: return record_of_fd(fd)
        finally: os.close(fd)
    def wremove(name):
        if wfd is None: still_ours(); os.remove(os.path.join(work, name))
        else: os.unlink(name, dir_fd=wfd)
    def wmove(name, dst, inside=False):  # into OUT_DIR, or (inside) to another name in the work folder
        if wfd is None: still_ours(); os.replace(os.path.join(work, name), os.path.join(work, dst) if inside else dst)
        else: os.replace(name, dst, src_dir_fd=wfd, dst_dir_fd=wfd if inside else None)
    raw, final = os.path.join(work, '_annotated_raw.mp4'), os.path.join(work, 'annotated.mp4')
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    limit = a.max_minutes * 60  # stop at the end of the file or here; the frame count in a file's header can be wrong or negative
    line_y = H * a.line if a.line else 0
    writer = None
    if not a.no_video:
        still_ours(); writer = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*'mp4v'), fps / a.every, (W, H))
        if not writer.isOpened(): print('the annotated video cannot be written here; going on without it', flush=True); writer = None
    per_sec = {}; per_min = {}; frames = 0; i = 0; t0 = time.time(); inf_ms = []; last_ms = -1
    while True:
        check_stop(); ok, frame = cap.read()
        if not ok: break
        t_s = i / fps
        if t_s > limit: break
        if i % a.every: i += 1; continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); mpi = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ms = max(int(t_s * 1000), last_ms + 1); last_ms = ms
        a0 = time.time(); res = det.detect_for_video(mpi, ms); inf_ms.append((time.time() - a0) * 1000)
        dets = [{'cls': d.categories[0].category_name, 'score': d.categories[0].score, 'box': (d.bounding_box.origin_x, d.bounding_box.origin_y, d.bounding_box.width, d.bounding_box.height)} for d in res.detections]
        tracks = tr.update(dets, line_y, t_s)
        now = {'person': 0, 'vehicle': 0, 'other': 0}
        for t in tracks: now[group(t['cls'])] += 1
        sec = int(t_s); row = per_sec.setdefault(sec, {'person': 0, 'vehicle': 0, 'other': 0, 'frames': 0})
        for k in ('person', 'vehicle', 'other'): row[k] = max(row[k], now[k])
        row['frames'] += 1
        m = sec // 60; mr = per_min.setdefault(m, {'person_max': 0, 'vehicle_max': 0, 'person_sum': 0, 'vehicle_sum': 0, 'frames': 0})
        mr['person_max'] = max(mr['person_max'], now['person']); mr['vehicle_max'] = max(mr['vehicle_max'], now['vehicle']); mr['person_sum'] += now['person']; mr['vehicle_sum'] += now['vehicle']; mr['frames'] += 1
        if writer is not None: draw(frame, tracks, line_y); writer.write(frame)
        frames += 1; i += 1
        if frames % 500 == 0: print(f'  {t_s:6.0f} s, {frames} frames, {np.mean(inf_ms[-500:]):.0f} ms a detection, {time.time() - t0:.0f} s elapsed', flush=True)
    cap.release(); dur = i / fps
    if frames == 0:
        if writer is not None: writer.release()
        still_ours()
        if not clear_work(work, expect=work_id): print(work, 'could not be deleted (it cannot be listed, a file in it cannot be deleted or keeps coming back, or it was replaced); the next run into', a.out, 'tries again', flush=True)
        elif made_top:
            d = out_abs  # the folders this run made, from OUT_DIR up, each only if nothing else is in it
            while True:
                try: os.rmdir(d)
                except OSError: break
                if d == made_top: break
                d = os.path.dirname(d)
        print('no frames could be read from', a.inp, "- an earlier run's outputs are left as they were"); sys.exit(3)
    if writer is not None:
        writer.release(); still_ours()  # OpenCV wrote the raw video by its path: a link put in its place is caught here
        if wexists('_annotated_raw.mp4') and not wplain('_annotated_raw.mp4'): replaced('_annotated_raw.mp4')
    # tables
    with wopen('counts_per_second.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['second', 'people_in_frame_max', 'vehicles_in_frame_max', 'other_in_frame_max', 'frames_seen'])
        for s in sorted(per_sec): r = per_sec[s]; w.writerow([s, r['person'], r['vehicle'], r['other'], r['frames']])
    ev_by_min = {}
    for (t_s, g, cls, d, tid) in tr.events: ev_by_min.setdefault(int(t_s) // 60, {'person_down': 0, 'person_up': 0, 'vehicle_down': 0, 'vehicle_up': 0})[f'{g}_{d}'] += 1
    with wopen('counts_per_minute.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['minute', 'people_in_frame_max', 'people_in_frame_mean', 'vehicles_in_frame_max', 'vehicles_in_frame_mean', 'people_crossed_down', 'people_crossed_up', 'vehicles_crossed_down', 'vehicles_crossed_up', 'frames_seen'])
        for m in sorted(per_min):
            r = per_min[m]; e = ev_by_min.get(m, {'person_down': 0, 'person_up': 0, 'vehicle_down': 0, 'vehicle_up': 0})
            w.writerow([m, r['person_max'], round(r['person_sum'] / max(1, r['frames']), 2), r['vehicle_max'], round(r['vehicle_sum'] / max(1, r['frames']), 2), *((e['person_down'], e['person_up'], e['vehicle_down'], e['vehicle_up']) if line_y else ('', '', '', '')), r['frames']])  # no line: the crossing columns stay empty
    with wopen('seen_by_class.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['class', 'group', 'tracks_seen']); [w.writerow([c, group(c), n]) for c, n in sorted(tr.seen.items(), key=lambda x: -x[1])]
    with wopen('crossings.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['second', 'group', 'class', 'direction', 'track_id']); [w.writerow([round(t, 2), g, c, d, tid]) for (t, g, c, d, tid) in tr.events]
    peak_p = max(per_sec.items(), key=lambda kv: kv[1]['person'], default=(0, {'person': 0})); peak_v = max(per_sec.items(), key=lambda kv: kv[1]['vehicle'], default=(0, {'vehicle': 0}))
    summary = {'input': os.path.basename(a.inp), 'title': a.title, 'made': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'), 'width': W, 'height': H, 'fps': round(fps, 3), 'duration_s': round(dur, 2), 'frames_processed': frames, 'every_nth_frame': a.every,
               'detector': f'MediaPipe Tasks object detector, model {model}, score threshold 0.4, max 25 per frame', 'tracker': 'IoU above 0.3, greedy; seen at 3 hits (the first frame and 2 matches); dropped after more than 12 misses in a row', 'line': {'fraction_of_height': a.line, 'y_px': round(line_y), 'text': a.line_text} if line_y else None,
               'ms_per_detection_mean': round(float(np.mean(inf_ms)), 1) if inf_ms else None, 'seen': tr.seen, 'people_seen': sum(n for c, n in tr.seen.items() if c == 'person'), 'vehicles_seen': sum(n for c, n in tr.seen.items() if c in VEHICLES),
               'crossings': {'person': {'down': tr.cross['person'][0], 'up': tr.cross['person'][1]}, 'vehicle': {'down': tr.cross['vehicle'][0], 'up': tr.cross['vehicle'][1]}},
               'peak_people_in_frame': {'count': peak_p[1]['person'], 'second': peak_p[0]}, 'peak_vehicles_in_frame': {'count': peak_v[1]['vehicle'], 'second': peak_v[0]}}
    with wopen('summary.json', 'w', encoding='utf-8') as f: json.dump(summary, f, indent=1, ensure_ascii=False)
    has_pdf = False
    try:
        with wopen('summary.pdf', 'wb') as f: pdf(summary, per_min, ev_by_min, f, video=writer is not None, lite0=lite0)
        has_pdf = True
    except Exception as e:  # the tables and the video still count without the PDF
        print('summary.pdf could not be written:', e, flush=True)
        if wexists('summary.pdf'): wremove('summary.pdf')
    kept = []  # output names whose file in OUT_DIR changed during this run: left as they are
    def put(new, old=()):
        """This run's files named in new go into place. For the other names (those in old, and a name in new for which this
        run made no file, such as summary.pdf when the PDF could not be written) the file an earlier run put there is deleted.
        The new mark, listing what this program will then have put in OUT_DIR, is written first, in the work folder, so a
        failure there (a full disk, say) leaves OUT_DIR as it was; then the files are moved and deleted, and the mark is moved
        in last. Only plain files this run made are moved. A file in OUT_DIR that is not ours (see above) is left alone."""
        still_ours()
        for name in new:
            if wexists(name) and not wplain(name): replaced(name)
        plan, after, names = [], {}, [*new, *old]
        for name in names:
            dst = os.path.join(a.out, name)
            if os.path.lexists(dst) and not ours(name): kept.append(name)
            elif name in new and wexists(name): plan.append((name, dst)); after[name] = wstamp(name)
            elif os.path.lexists(dst): plan.append((None, dst))
        for n in OUTPUTS:  # the names this call does not touch: listed again while unchanged since this program put them there
            if n in names: continue
            if ours(n): after[n] = marks[n]
            elif n in marks and os.path.lexists(os.path.join(a.out, n)): kept.append(n)  # changed since this run moved it
        try:
            with wopen(MARK, 'w', check=False, encoding='utf-8') as f:
                f.write('This folder holds the outputs of count_video.py. A new run into it replaces or deletes the files listed here (name, size, SHA-256 of the content), and only while their content is unchanged.\n')
                f.writelines(format_row(n, after[n]) + '\n' for n in OUTPUTS if n in after)
        except OSError as e:
            print(MARK, 'could not be written in', work, '(' + (e.strerror or str(e)) + ') - nothing more was moved into', a.out, flush=True); sys.exit(2)
        with no_stop():
            for name, dst in plan:
                if name is None: os.remove(dst)
                else: wmove(name, dst); marks[name] = after[name]
            wmove(MARK, os.path.join(a.out, MARK))
    # the tables, the summary and the PDF go into place before the video step, so a video step that fails or is stopped cannot
    # cost them; an earlier run's video and README.txt are deleted at the same time, so they cannot sit beside this run's tables
    check_stop(); put(['counts_per_second.csv', 'counts_per_minute.csv', 'seen_by_class.csv', 'crossings.csv', 'summary.json', 'summary.pdf'], ['annotated.mp4', 'README.txt', '_annotated_raw.mp4'])
    # the annotated video: re-encoded to H.264 when an ffmpeg is found and works, else kept as written (MPEG-4 Part 2)
    check_stop(); still_ours()
    if writer is not None and wexists('_annotated_raw.mp4'):
        encoded = False
        if ff:
            try:
                here = wfd is not None and os.name == 'posix'  # ffmpeg works inside the folder this run made, entered through its descriptor
                src, dst = ('_annotated_raw.mp4', 'annotated.mp4') if here else (raw, final)
                if wexists('annotated.mp4'): wremove('annotated.mp4')  # ffmpeg makes it new
                p = None
                with wopen('_ffmpeg.log', 'wb') as log:  # ffmpeg's messages, shown if it fails: nothing it or a wrapper starts holds the run's own output, so a pipe from the run ends with it
                    try:
                        with no_stop():  # a stop while ffmpeg is being started takes effect once it has started, so that it is stopped too (below)
                            p = subprocess.Popen([os.path.abspath(ff) if here else ff, '-nostdin', '-y', '-loglevel', 'error', '-i', src, '-c:v', 'libx264', '-preset', 'fast', '-crf', '23', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', dst],
                                                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=log, preexec_fn=(lambda: os.fchdir(wfd)) if here else None)  # in the run's process group, so a stop sent to the group reaches it too
                        encoded = p.wait(timeout=max(FFMPEG_MIN_TIMEOUT_S, 10 * dur)) == 0 and wexists('annotated.mp4')
                    except BaseException:  # the time limit, Ctrl-C, SIGTERM or SIGHUP: the ffmpeg ends too (not what a wrapper script started)
                        if p is not None:
                            with contextlib.suppress(OSError): p.kill()
                            p.wait()
                        raise
            except OSError as e:
                print('ffmpeg could not be run:', e, flush=True)
            except subprocess.TimeoutExpired:
                print('ffmpeg ran longer than', round(max(FFMPEG_MIN_TIMEOUT_S, 10 * dur)), 's and was stopped', flush=True)
            if not encoded:
                said = b''
                with contextlib.suppress(OSError):
                    with wopen('_ffmpeg.log', 'rb') as log: said = log.read()[-1500:]
                if said.strip(): print('ffmpeg said:', said.decode('utf-8', 'replace').strip(), flush=True)
                print('ffmpeg did not make the H.264 video; the annotated video stays MPEG-4 Part 2', flush=True)
        elif a.ffmpeg: print('ffmpeg not found (' + a.ffmpeg + '); the annotated video stays MPEG-4 Part 2', flush=True)
        still_ours()
        for name in ('_annotated_raw.mp4', 'annotated.mp4'):  # OpenCV and ffmpeg open these by name: a link put in their place is caught here, never moved
            if wexists(name) and not wplain(name): replaced(name)
        if encoded: wremove('_annotated_raw.mp4')
        else:
            if wexists('annotated.mp4'): wremove('annotated.mp4')
            wmove('_annotated_raw.mp4', 'annotated.mp4', inside=True)
    has_video = wexists('annotated.mp4')
    with wopen('README.txt', 'w', encoding='utf-8') as f: f.write('SeeDeep video counts\n\ncounts_per_second.csv: the most people, vehicles and other things in frame in each second.\ncounts_per_minute.csv: per minute, the most and the average in frame' + (', and the line crossings' if line_y else ' (no line was set, so its crossing columns are empty)') + '.\nseen_by_class.csv: how many tracks of each class reached 3 hits ("seen"; a thing the detector loses for more than 12 of the frames looked at and finds again is counted again each time; two things close together can be seen as one, and two that pass each other can swap tracks).\n' + ('crossings.csv: every crossing of the line with its time, direction and track.\n' if line_y else 'crossings.csv: empty, as no line was set.\n') + ('annotated.mp4: the video with the boxes' + (' and the line' if line_y else '') + ' drawn.\n' if has_video else '') + ('summary.json: the totals and the peaks; summary.pdf: the same, with the method and its limits.\n\n' if has_pdf else 'summary.json: the totals and the peaks.\n\n') + 'Detector: MediaPipe, model ' + model + ('; Apache 2.0, 80 COCO classes' if lite0 else '') + '; it does not read faces, plates or text. Counts can be wrong' + ('; the annotated video is there so you can judge' if has_video else '') + '.\n')
    put(['annotated.mp4', 'README.txt'])  # last, after the video step
    if kept: print('left as they were, as they changed in', a.out, "during this run (before or after this run's file of that name was moved in):", ', '.join(dict.fromkeys(kept)), '- they are not listed in', MARK, flush=True)
    still_ours()
    if not clear_work(work, expect=work_id): print(work, 'could not be deleted (it cannot be listed, or a file in it cannot be deleted or keeps coming back); the next run into', a.out, 'tries again', flush=True)
    print(json.dumps({k: summary[k] for k in ('frames_processed', 'duration_s', 'ms_per_detection_mean', 'people_seen', 'vehicles_seen', 'crossings', 'peak_people_in_frame')}))
    if kept: sys.exit(4)  # some files in OUT_DIR changed during this run (above)

def pdf(s, per_min, ev, path, video=False, lite0=True):  # path: a file name or an open binary file
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
    c = canvas.Canvas(path, pagesize=letter); Wp, Hp = letter; y = Hp - inch
    def line(txt, size=11, dy=16, bold=False, mono=False):
        nonlocal y
        font = ('Courier' if mono else 'Helvetica') + ('-Bold' if bold else ''); wide = lambda t: c.stringWidth(t, font, size) > Wp - 2 * inch; parts = [txt]
        if wide(txt):
            words = []
            for w in txt.split(' '):  # a word wider than the line is cut into pieces that fit, wherever it sits
                while wide(w):
                    k = len(w)
                    while k > 1 and wide(w[:k]): k -= 1
                    words.append(w[:k]); w = w[k:]
                words.append(w)
            parts = ['']
            for w in words:
                trial = (parts[-1] + ' ' + w) if parts[-1] else w
                if parts[-1] and wide(trial): parts.append(w)
                else: parts[-1] = trial
        for k, part in enumerate(parts):
            if y < inch: c.showPage(); y = Hp - inch
            c.setFont(font, size); c.drawString(inch, y, part); y -= dy if k == len(parts) - 1 else size + 4
    line('SeeDeep video counts', 20, 28, True)
    line(f"{s['input']}" + (f" for {s['title']}" if s['title'] else '') + f", made {s['made'][:16].replace('T', ' ')} UTC", 11, 22)
    line('What was counted', 14, 20, True)
    line("Length " + (f"{s['duration_s']:.1f} seconds" if s['duration_s'] < 60 else f"{s['duration_s'] / 60:.1f} minutes") + f", {s['width']} by {s['height']} at {s['fps']} frames a second; {s['frames_processed']} frames looked at" + (f" (one in every {s['every_nth_frame']})" if s['every_nth_frame'] > 1 else '') + '.')
    line(f"Person tracks seen: {s['people_seen']}. Vehicle tracks seen: {s['vehicles_seen']}. Peak people in frame: {s['peak_people_in_frame']['count']} at {s['peak_people_in_frame']['second'] // 60}:{s['peak_people_in_frame']['second'] % 60:02d}. Peak vehicles in frame: {s['peak_vehicles_in_frame']['count']} at {s['peak_vehicles_in_frame']['second'] // 60}:{s['peak_vehicles_in_frame']['second'] % 60:02d}.")
    if s['line']:
        line(f"Line at {round(s['line']['fraction_of_height'] * 100)}% of the height" + (f" ({s['line']['text']})" if s['line']['text'] else '') + f": people crossed down {s['crossings']['person']['down']}, up {s['crossings']['person']['up']}; vehicles down {s['crossings']['vehicle']['down']}, up {s['crossings']['vehicle']['up']}.")
    y -= 6; line('Tracks seen, by class', 14, 20, True)
    for cls, n in sorted(s['seen'].items(), key=lambda x: -x[1])[:PDF_CLASSES]: line(f"{cls}: {n}", 11, 14)
    if len(s['seen']) > PDF_CLASSES: line(f"and {len(s['seen']) - PDF_CLASSES} more classes, in seen_by_class.csv", 11, 14)
    y -= 6; line('By minute (most in frame' + ('; crossings)' if s['line'] else ')'), 14, 20, True)
    HEAD = 'minute   people   vehicles' + ('   people down/up   vehicles down/up' if s['line'] else '')
    line(HEAD, 9, 14, True, True)
    for m in sorted(per_min):
        r = per_min[m]; e = ev.get(m, {'person_down': 0, 'person_up': 0, 'vehicle_down': 0, 'vehicle_up': 0})
        if y < inch: c.showPage(); y = Hp - inch; line(HEAD, 9, 14, True, True)  # the header again at the top of a new page
        line(f"{m:>6}   {r['person_max']:>6}   {r['vehicle_max']:>8}" + (f"   {e['person_down']:>6} / {e['person_up']:<6}   {e['vehicle_down']:>6} / {e['vehicle_up']}" if s['line'] else ''), 9, 13, False, True)
    # a small bar chart of people per minute
    if per_min:
        if y < 3 * inch: c.showPage(); y = Hp - inch
        y -= 10; line('People in frame, most per minute', 12, 16, True)
        mins = sorted(per_min); m0, m1 = mins[0], mins[-1]; n = m1 - m0 + 1  # one slot a minute, by its number; a minute with no frame looked at has no bar
        mx = max(r['person_max'] for r in per_min.values()); bw = min(18, (Wp - 2 * inch) / n); base = y - 110
        c.setFillColorRGB(0.95, 0.65, 0.09)
        for m in mins: h = 100 * per_min[m]['person_max'] / (mx or 1); c.rect(inch + (m - m0) * bw, base, max(bw * 0.8, bw - 2), h, fill=1, stroke=0)
        c.setFillColorRGB(0, 0, 0); c.setFont('Helvetica', 8); c.drawString(inch, base - 12, f"minute {m0} to {m1}, one slot a minute; tallest bar = {mx}"); y = base - 30
    y -= 6; line('Method and limits', 14, 20, True)
    for t in [f"Detector: {s['detector']} (" + ('MediaPipe, Google, Apache 2.0; 80 everyday classes; ' if lite0 else '') + "it does not read faces, plates or text).", f"Tracking: {s['tracker']}. A thing the detector loses for more than 12 of the frames looked at and finds again is counted again each time; two things close together can be seen as one, and two that pass each other can swap tracks.", f"Speed on the machine that made this: {s['ms_per_detection_mean']} ms a detection.", 'Accuracy was not measured on this footage. Counts can be wrong.' + (' The annotated video is included so you can judge them yourself.' if video else ''), 'Made by count_video.py on the machine that ran it.']:
        words = t.split(); buf = ''
        for w in words:
            if c.stringWidth(buf + ' ' + w, 'Helvetica', 10) > Wp - 2 * inch: line(buf, 10, 13); buf = w
            else: buf = (buf + ' ' + w).strip()
        line(buf, 10, 15)
    c.save()

if __name__ == '__main__':
    try: main()
    except KeyboardInterrupt: ctrl_c()
    except SystemExit:  # a SIGTERM or SIGHUP: end by that signal, as Ctrl-C does, so no thread a library started can hold up the exit
        if GOT and os.name == 'posix':
            if GOT[0] == signal.SIGINT: ctrl_c()
            sys.stdout.flush(); sys.stderr.flush(); signal.signal(GOT[0], signal.SIG_DFL); os.kill(os.getpid(), GOT[0])
        raise
