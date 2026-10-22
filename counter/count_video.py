#!/usr/bin/env python3
"""SeeDeep video counter: counts people, vehicles and other things in a video file, offline. Same detector as the browser
page in web/ (MediaPipe EfficientDet-Lite0, score 0.4, 25 per frame), same tracker rules (tracker.py), one horizontal line.
usage: python count_video.py IN.mp4 OUT_DIR [--line 0.62] [--every 1] [--max-minutes 60] [--title "name"] [--line-text "the doorway"]
       [--model PATH] [--ffmpeg NAME_OR_PATH] [--no-video]
writes into OUT_DIR/.seedeep-counter-work: _annotated_raw.mp4 while it reads, then counts_per_second.csv, counts_per_minute.csv,
seen_by_class.csv, crossings.csv, summary.json, summary.pdf, then annotated.mp4 (made from the raw file), then README.txt;
when the run has finished it moves them into OUT_DIR, replacing an earlier run's, and writes .seedeep-counter-output there
The model is fetched by ../fetch_assets.sh; ffmpeg (optional) re-encodes the annotated video to H.264."""
import argparse, csv, json, os, sys, time, subprocess, datetime, shutil
import cv2, numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mptasks
from mediapipe.tasks.python import vision
try:
    import reportlab  # used for summary.pdf; checked here so a missing one stops the run before it starts
except ImportError:
    reportlab = None
from tracker import Tracker, group, VEHICLES

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, '..', 'models', 'efficientdet_lite0.tflite')
COL = {'person': (22, 165, 242), 'vehicle': (166, 163, 31), 'other': (168, 152, 138)}  # BGR of #f2a516, #1fa3a6, #8a98a8
FFMPEG_MIN_TIMEOUT_S = 300  # an ffmpeg still running after this, or after ten times the video's length if that is longer, is stopped
PDF_CLASSES = 16  # classes listed by name in summary.pdf; seen_by_class.csv has them all
MARK = '.seedeep-counter-output'  # written in every output folder; only in a folder with it (or with this program's README.txt) are earlier outputs replaced
WORK = '.seedeep-counter-work'  # this run's files, until the run has finished

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
    if a.every < 1: ap.error('--every must be 1 or more')
    if not 0 <= a.line < 1: ap.error('--line must be 0 (no line) or a fraction of the height below 1')
    if not a.max_minutes > 0: ap.error('--max-minutes must be a number above 0')  # written so that nan is refused too
    if not os.path.exists(a.model): print('model not found:', a.model, '(run ./fetch_assets.sh from the repository root)'); sys.exit(2)
    if reportlab is None: print('ReportLab is not installed (pip install -r counter/requirements.txt)'); sys.exit(2)
    OUTPUTS = ('counts_per_second.csv', 'counts_per_minute.csv', 'seen_by_class.csv', 'crossings.csv', 'summary.json', 'summary.pdf', 'annotated.mp4', 'README.txt', '_annotated_raw.mp4')
    if os.path.exists(a.out) and not os.path.isdir(a.out): ap.error('OUT_DIR must be a folder: ' + a.out)
    present = [n for n in OUTPUTS if os.path.exists(os.path.join(a.out, n))]
    if any(os.path.isdir(os.path.join(a.out, n)) for n in present):
        print(a.out, 'holds a folder named like one of the outputs (' + ', '.join(n for n in present if os.path.isdir(os.path.join(a.out, n))) + ') - choose another output folder'); sys.exit(2)
    if os.path.exists(a.inp) and any(os.path.samefile(a.inp, os.path.join(a.out, n)) for n in present):  # the same file however it is spelled
        print('the input is one of the files this run writes in', a.out, '- choose another output folder'); sys.exit(2)
    def ours():
        if os.path.exists(os.path.join(a.out, MARK)): return True
        try: return open(os.path.join(a.out, 'README.txt'), encoding='utf-8', errors='replace').readline().strip() == 'SeeDeep video counts'
        except OSError: return False
    if present and not ours():
        print(a.out, 'holds', ', '.join(present), 'but is not a folder this program wrote (no', MARK, 'in it) - choose an empty or new output folder'); sys.exit(2)
    cap = cv2.VideoCapture(a.inp)
    if not cap.isOpened(): print('cannot open', a.inp); sys.exit(2)
    opts = vision.ObjectDetectorOptions(base_options=mptasks.BaseOptions(model_asset_path=a.model), running_mode=vision.RunningMode.VIDEO, score_threshold=0.4, max_results=25)
    try: det = vision.ObjectDetector.create_from_options(opts)
    except Exception as e: print('the detector could not be made from', a.model, '-', str(e)[:200]); sys.exit(2)
    tr = Tracker()
    os.makedirs(a.out, exist_ok=True)
    work = os.path.join(a.out, WORK)  # this run's files go here and into OUT_DIR only at the end, so a run that fails or is stopped leaves an earlier run's outputs as they were
    if os.path.isdir(work): shutil.rmtree(work)  # what a stopped run left
    os.makedirs(work)
    raw, final = os.path.join(work, '_annotated_raw.mp4'), os.path.join(work, 'annotated.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS) or 25; W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    limit = a.max_minutes * 60  # stop at the end of the file or here; the frame count in a file's header can be wrong or negative
    line_y = H * a.line if a.line else 0
    writer = None
    if not a.no_video:
        writer = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*'mp4v'), fps / a.every, (W, H))
        if not writer.isOpened(): print('the annotated video cannot be written here; going on without it', flush=True); writer = None
    per_sec = {}; per_min = {}; frames = 0; i = 0; t0 = time.time(); inf_ms = []; last_ms = -1
    while True:
        ok, frame = cap.read()
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
        shutil.rmtree(work, ignore_errors=True)
        print('no frames could be read from', a.inp, "- an earlier run's outputs are left as they were"); sys.exit(3)
    if writer is not None: writer.release()
    # tables
    with open(os.path.join(work, 'counts_per_second.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['second', 'people_in_frame_max', 'vehicles_in_frame_max', 'other_in_frame_max', 'frames_seen'])
        for s in sorted(per_sec): r = per_sec[s]; w.writerow([s, r['person'], r['vehicle'], r['other'], r['frames']])
    ev_by_min = {}
    for (t_s, g, cls, d, tid) in tr.events: ev_by_min.setdefault(int(t_s) // 60, {'person_down': 0, 'person_up': 0, 'vehicle_down': 0, 'vehicle_up': 0})[f'{g}_{d}'] += 1
    with open(os.path.join(work, 'counts_per_minute.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['minute', 'people_in_frame_max', 'people_in_frame_mean', 'vehicles_in_frame_max', 'vehicles_in_frame_mean', 'people_crossed_down', 'people_crossed_up', 'vehicles_crossed_down', 'vehicles_crossed_up', 'frames_seen'])
        for m in sorted(per_min):
            r = per_min[m]; e = ev_by_min.get(m, {'person_down': 0, 'person_up': 0, 'vehicle_down': 0, 'vehicle_up': 0})
            w.writerow([m, r['person_max'], round(r['person_sum'] / max(1, r['frames']), 2), r['vehicle_max'], round(r['vehicle_sum'] / max(1, r['frames']), 2), e['person_down'], e['person_up'], e['vehicle_down'], e['vehicle_up'], r['frames']])
    with open(os.path.join(work, 'seen_by_class.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['class', 'group', 'tracks_seen']); [w.writerow([c, group(c), n]) for c, n in sorted(tr.seen.items(), key=lambda x: -x[1])]
    with open(os.path.join(work, 'crossings.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['second', 'group', 'class', 'direction', 'track_id']); [w.writerow([round(t, 2), g, c, d, tid]) for (t, g, c, d, tid) in tr.events]
    peak_p = max(per_sec.items(), key=lambda kv: kv[1]['person'], default=(0, {'person': 0})); peak_v = max(per_sec.items(), key=lambda kv: kv[1]['vehicle'], default=(0, {'vehicle': 0}))
    summary = {'input': os.path.basename(a.inp), 'title': a.title, 'made': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'), 'width': W, 'height': H, 'fps': round(fps, 3), 'duration_s': round(dur, 2), 'frames_processed': frames, 'every_nth_frame': a.every,
               'detector': 'MediaPipe Tasks EfficientDet-Lite0 (float16), score threshold 0.4, max 25 per frame', 'tracker': 'IoU above 0.3, greedy; seen at 3 hits (the first frame and 2 matches); dropped after more than 12 misses in a row', 'line': {'fraction_of_height': a.line, 'y_px': round(line_y), 'text': a.line_text} if line_y else None,
               'ms_per_detection_mean': round(float(np.mean(inf_ms)), 1) if inf_ms else None, 'seen': tr.seen, 'people_seen': sum(n for c, n in tr.seen.items() if c == 'person'), 'vehicles_seen': sum(n for c, n in tr.seen.items() if c in VEHICLES),
               'crossings': {'person': {'down': tr.cross['person'][0], 'up': tr.cross['person'][1]}, 'vehicle': {'down': tr.cross['vehicle'][0], 'up': tr.cross['vehicle'][1]}},
               'peak_people_in_frame': {'count': peak_p[1]['person'], 'second': peak_p[0]}, 'peak_vehicles_in_frame': {'count': peak_v[1]['vehicle'], 'second': peak_v[0]}}
    json.dump(summary, open(os.path.join(work, 'summary.json'), 'w'), indent=1)
    has_pdf = False
    try:
        pdf(summary, per_min, ev_by_min, os.path.join(work, 'summary.pdf'), video=writer is not None); has_pdf = True
    except Exception as e:  # the tables and the video still count without the PDF
        print('summary.pdf could not be written:', e, flush=True)
        if os.path.exists(os.path.join(work, 'summary.pdf')): os.remove(os.path.join(work, 'summary.pdf'))
    # the annotated video: re-encoded to H.264 when an ffmpeg is found and works, else kept as written (MPEG-4 Part 2)
    if writer is not None and os.path.exists(raw):
        ff = shutil.which(a.ffmpeg) if a.ffmpeg else None; encoded = False
        if ff:
            try:
                r = subprocess.run([ff, '-nostdin', '-y', '-loglevel', 'error', '-i', raw, '-c:v', 'libx264', '-preset', 'fast', '-crf', '23', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', final], check=False, timeout=max(FFMPEG_MIN_TIMEOUT_S, 10 * dur))
                encoded = r.returncode == 0 and os.path.exists(final)
            except OSError as e:
                print('ffmpeg could not be run:', e, flush=True)
            except subprocess.TimeoutExpired:
                print('ffmpeg ran longer than', round(max(FFMPEG_MIN_TIMEOUT_S, 10 * dur)), 's and was stopped', flush=True)
            if not encoded: print('ffmpeg did not make the H.264 video; the annotated video stays MPEG-4 Part 2', flush=True)
        elif a.ffmpeg: print('ffmpeg not found (' + a.ffmpeg + '); the annotated video stays MPEG-4 Part 2', flush=True)
        if encoded: os.remove(raw)
        else:
            if os.path.exists(final): os.remove(final)
            os.rename(raw, final)
    open(os.path.join(work, 'README.txt'), 'w').write('SeeDeep video counts\n\ncounts_per_second.csv: the most people, vehicles and other things in frame in each second.\ncounts_per_minute.csv: per minute, the most and the average in frame, and the line crossings.\nseen_by_class.csv: how many tracked things of each class were seen (a thing the detector loses for more than 12 of the frames looked at and finds again counts twice; two things close together can be seen as one, and two that pass each other can swap tracks).\ncrossings.csv: every crossing of the line with its time, direction and track.\n' + ('annotated.mp4: the video with the boxes and the line drawn.\n' if os.path.exists(final) else '') + ('summary.json: the totals and the peaks; summary.pdf: the same, with the method and its limits.\n\n' if has_pdf else 'summary.json: the totals and the peaks.\n\n') + 'Detector: MediaPipe EfficientDet-Lite0 (Apache 2.0), 80 COCO classes; it does not read faces, plates or text. Counts can be wrong' + ('; the annotated video is there so you can judge' if os.path.exists(final) else '') + '.\n')
    for name in OUTPUTS:  # into place: this run's files replace an earlier run's, and an earlier file this run did not make goes, so none can sit beside this run's
        src, dst = os.path.join(work, name), os.path.join(a.out, name)
        if os.path.exists(src): os.replace(src, dst)
        elif os.path.exists(dst): os.remove(dst)
    open(os.path.join(a.out, MARK), 'w').write('This folder holds the outputs of count_video.py; a new run into it replaces them.\n')
    shutil.rmtree(work, ignore_errors=True)
    print(json.dumps({k: summary[k] for k in ('frames_processed', 'duration_s', 'ms_per_detection_mean', 'people_seen', 'vehicles_seen', 'crossings', 'peak_people_in_frame')}))

def pdf(s, per_min, ev, path, video=False):
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
    line(f"Length {s['duration_s'] / 60:.1f} minutes, {s['width']} by {s['height']} at {s['fps']} frames a second; {s['frames_processed']} frames looked at" + (f" (one in every {s['every_nth_frame']})" if s['every_nth_frame'] > 1 else '') + '.')
    line(f"People seen: {s['people_seen']}. Vehicles seen: {s['vehicles_seen']}. Peak people in frame: {s['peak_people_in_frame']['count']} at {s['peak_people_in_frame']['second'] // 60}:{s['peak_people_in_frame']['second'] % 60:02d}. Peak vehicles in frame: {s['peak_vehicles_in_frame']['count']} at {s['peak_vehicles_in_frame']['second'] // 60}:{s['peak_vehicles_in_frame']['second'] % 60:02d}.")
    if s['line']:
        line(f"Line at {round(s['line']['fraction_of_height'] * 100)}% of the height" + (f" ({s['line']['text']})" if s['line']['text'] else '') + f": people crossed down {s['crossings']['person']['down']}, up {s['crossings']['person']['up']}; vehicles down {s['crossings']['vehicle']['down']}, up {s['crossings']['vehicle']['up']}.")
    y -= 6; line('Seen, by class', 14, 20, True)
    for cls, n in sorted(s['seen'].items(), key=lambda x: -x[1])[:PDF_CLASSES]: line(f"{cls}: {n}", 11, 14)
    if len(s['seen']) > PDF_CLASSES: line(f"and {len(s['seen']) - PDF_CLASSES} more classes, in seen_by_class.csv", 11, 14)
    y -= 6; line('By minute (most in frame; crossings)', 14, 20, True)
    HEAD = 'minute   people   vehicles   people down/up   vehicles down/up'
    line(HEAD, 9, 14, True, True)
    for m in sorted(per_min):
        r = per_min[m]; e = ev.get(m, {'person_down': 0, 'person_up': 0, 'vehicle_down': 0, 'vehicle_up': 0})
        if y < inch: c.showPage(); y = Hp - inch; line(HEAD, 9, 14, True, True)  # the header again at the top of a new page
        line(f"{m:>6}   {r['person_max']:>6}   {r['vehicle_max']:>8}   {e['person_down']:>6} / {e['person_up']:<6}   {e['vehicle_down']:>6} / {e['vehicle_up']}", 9, 13, False, True)
    # a small bar chart of people per minute
    if per_min:
        if y < 3 * inch: c.showPage(); y = Hp - inch
        y -= 10; line('People in frame, most per minute', 12, 16, True)
        mins = sorted(per_min); m0, m1 = mins[0], mins[-1]; n = m1 - m0 + 1  # one slot a minute, by its number; a minute with no frame looked at has no bar
        mx = max(r['person_max'] for r in per_min.values()) or 1; bw = min(18, (Wp - 2 * inch) / n); base = y - 110
        c.setFillColorRGB(0.95, 0.65, 0.09)
        for m in mins: h = 100 * per_min[m]['person_max'] / mx; c.rect(inch + (m - m0) * bw, base, max(bw * 0.8, bw - 2), h, fill=1, stroke=0)
        c.setFillColorRGB(0, 0, 0); c.setFont('Helvetica', 8); c.drawString(inch, base - 12, f"minute {m0} to {m1}, one slot a minute; tallest bar = {mx}"); y = base - 30
    y -= 6; line('Method and limits', 14, 20, True)
    for t in [f"Detector: {s['detector']} (MediaPipe, Google, Apache 2.0; 80 everyday classes; it does not read faces, plates or text).", f"Tracking: {s['tracker']}. A thing the detector loses for more than 12 of the frames looked at and finds again counts twice; two things close together can be seen as one, and two that pass each other can swap tracks.", f"Speed on the machine that made this: {s['ms_per_detection_mean']} ms a detection.", 'Accuracy was not measured on this footage. Counts can be wrong.' + (' The annotated video is included so you can judge them yourself.' if video else ''), 'Made by count_video.py on the machine that ran it; the script itself makes no network request.']:
        words = t.split(); buf = ''
        for w in words:
            if c.stringWidth(buf + ' ' + w, 'Helvetica', 10) > Wp - 2 * inch: line(buf, 10, 13); buf = w
            else: buf = (buf + ' ' + w).strip()
        line(buf, 10, 15)
    c.save()

if __name__ == '__main__': main()
