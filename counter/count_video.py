#!/usr/bin/env python3
"""SeeDeep video counter: counts people, vehicles and other things in a video file, offline. Same detector as the browser
page in web/ (MediaPipe EfficientDet-Lite0, score 0.4, 25 per frame), same tracker rules (tracker.py), one horizontal line.
usage: python count_video.py IN.mp4 OUT_DIR [--line 0.62] [--every 1] [--max-minutes 60] [--title "name"] [--line-text "the doorway"]
       [--model PATH] [--ffmpeg NAME_OR_PATH] [--no-video]
writes: counts_per_second.csv, counts_per_minute.csv, seen_by_class.csv, crossings.csv, summary.json, annotated.mp4, summary.pdf, README.txt
The model is fetched by ../fetch_assets.sh; ffmpeg (optional) re-encodes the annotated video to H.264."""
import argparse, csv, json, os, sys, time, subprocess, datetime, shutil
import cv2, numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mptasks
from mediapipe.tasks.python import vision
from tracker import Tracker, group, VEHICLES

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, '..', 'models', 'efficientdet_lite0.tflite')
COL = {'person': (22, 165, 242), 'vehicle': (166, 163, 31), 'other': (168, 152, 138)}  # BGR of #f2a516, #1fa3a6, #8a98a8

def draw(frame, tracks, line_y, scale):
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
    if a.max_minutes <= 0: ap.error('--max-minutes must be above 0')
    if not os.path.exists(a.model): print('model not found:', a.model, '(run ./fetch_assets.sh from the repository root)'); sys.exit(2)
    os.makedirs(a.out, exist_ok=True); cap = cv2.VideoCapture(a.inp)
    raw, final = os.path.join(a.out, '_annotated_raw.mp4'), os.path.join(a.out, 'annotated.mp4')
    for old in (raw, final):
        if os.path.exists(old): os.remove(old)  # never leave an earlier run's video beside this run's tables
    if not cap.isOpened(): print('cannot open', a.inp); sys.exit(2)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25; W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    limit = a.max_minutes * 60  # stop at the end of the file or here; the frame count in a file's header can be wrong or negative
    line_y = H * a.line if a.line else 0
    opts = vision.ObjectDetectorOptions(base_options=mptasks.BaseOptions(model_asset_path=a.model), running_mode=vision.RunningMode.VIDEO, score_threshold=0.4, max_results=25)
    det = vision.ObjectDetector.create_from_options(opts); tr = Tracker()
    writer = None
    if not a.no_video:
        writer = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*'mp4v'), fps / a.every, (W, H))
        if not writer.isOpened(): print('the annotated video cannot be written here; going on without it', flush=True); writer = None
    per_sec = {}; per_min = {}; frames = 0; i = 0; t0 = time.time(); inf_ms = []
    while True:
        ok, frame = cap.read()
        if not ok: break
        t_s = i / fps
        if t_s > limit: break
        if i % a.every: i += 1; continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); mpi = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        a0 = time.time(); res = det.detect_for_video(mpi, int(t_s * 1000)); inf_ms.append((time.time() - a0) * 1000)
        dets = [{'cls': d.categories[0].category_name, 'score': d.categories[0].score, 'box': (d.bounding_box.origin_x, d.bounding_box.origin_y, d.bounding_box.width, d.bounding_box.height)} for d in res.detections]
        tracks = tr.update(dets, line_y, t_s)
        now = {'person': 0, 'vehicle': 0, 'other': 0}
        for t in tracks: now[group(t['cls'])] += 1
        sec = int(t_s); row = per_sec.setdefault(sec, {'person': 0, 'vehicle': 0, 'other': 0, 'frames': 0})
        for k in ('person', 'vehicle', 'other'): row[k] = max(row[k], now[k])
        row['frames'] += 1
        m = sec // 60; mr = per_min.setdefault(m, {'person_max': 0, 'vehicle_max': 0, 'person_sum': 0, 'vehicle_sum': 0, 'frames': 0})
        mr['person_max'] = max(mr['person_max'], now['person']); mr['vehicle_max'] = max(mr['vehicle_max'], now['vehicle']); mr['person_sum'] += now['person']; mr['vehicle_sum'] += now['vehicle']; mr['frames'] += 1
        if writer is not None: draw(frame, tracks, line_y, 1); writer.write(frame)
        frames += 1; i += 1
        if frames % 500 == 0: print(f'  {t_s:6.0f} s, {frames} frames, {np.mean(inf_ms[-500:]):.0f} ms a detection, {time.time() - t0:.0f} s elapsed', flush=True)
    cap.release(); dur = i / fps
    if frames == 0:
        if writer is not None:
            writer.release()
            if os.path.exists(raw): os.remove(raw)
        print('no frames could be read from', a.inp); sys.exit(3)
    if writer is not None: writer.release()
    # tables
    with open(os.path.join(a.out, 'counts_per_second.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['second', 'people_in_frame_max', 'vehicles_in_frame_max', 'other_in_frame_max', 'frames_seen'])
        for s in sorted(per_sec): r = per_sec[s]; w.writerow([s, r['person'], r['vehicle'], r['other'], r['frames']])
    ev_by_min = {}
    for (t_s, g, cls, d, tid) in tr.events: ev_by_min.setdefault(int(t_s) // 60, {'person_down': 0, 'person_up': 0, 'vehicle_down': 0, 'vehicle_up': 0})[f'{g}_{d}'] += 1
    with open(os.path.join(a.out, 'counts_per_minute.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['minute', 'people_in_frame_max', 'people_in_frame_mean', 'vehicles_in_frame_max', 'vehicles_in_frame_mean', 'people_crossed_down', 'people_crossed_up', 'vehicles_crossed_down', 'vehicles_crossed_up', 'frames_seen'])
        for m in sorted(per_min):
            r = per_min[m]; e = ev_by_min.get(m, {'person_down': 0, 'person_up': 0, 'vehicle_down': 0, 'vehicle_up': 0})
            w.writerow([m, r['person_max'], round(r['person_sum'] / max(1, r['frames']), 2), r['vehicle_max'], round(r['vehicle_sum'] / max(1, r['frames']), 2), e['person_down'], e['person_up'], e['vehicle_down'], e['vehicle_up'], r['frames']])
    with open(os.path.join(a.out, 'seen_by_class.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['class', 'group', 'tracks_seen']); [w.writerow([c, group(c), n]) for c, n in sorted(tr.seen.items(), key=lambda x: -x[1])]
    with open(os.path.join(a.out, 'crossings.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['second', 'group', 'class', 'direction', 'track_id']); [w.writerow([round(t, 2), g, c, d, tid]) for (t, g, c, d, tid) in tr.events]
    peak_p = max(per_sec.items(), key=lambda kv: kv[1]['person'], default=(0, {'person': 0})); peak_v = max(per_sec.items(), key=lambda kv: kv[1]['vehicle'], default=(0, {'vehicle': 0}))
    summary = {'input': os.path.basename(a.inp), 'title': a.title, 'made': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'), 'width': W, 'height': H, 'fps': round(fps, 3), 'duration_s': round(dur, 2), 'frames_processed': frames, 'every_nth_frame': a.every,
               'detector': 'MediaPipe Tasks EfficientDet-Lite0 (float16), score threshold 0.4, max 25 per frame', 'tracker': 'IoU above 0.3, greedy; seen at 3 hits (the first frame and 2 matches); dropped after more than 12 misses in a row', 'line': {'fraction_of_height': a.line, 'y_px': round(line_y), 'text': a.line_text} if line_y else None,
               'ms_per_detection_mean': round(float(np.mean(inf_ms)), 1) if inf_ms else None, 'seen': tr.seen, 'people_seen': sum(n for c, n in tr.seen.items() if c == 'person'), 'vehicles_seen': sum(n for c, n in tr.seen.items() if c in VEHICLES),
               'crossings': {'person': {'down': tr.cross['person'][0], 'up': tr.cross['person'][1]}, 'vehicle': {'down': tr.cross['vehicle'][0], 'up': tr.cross['vehicle'][1]}},
               'peak_people_in_frame': {'count': peak_p[1]['person'], 'second': peak_p[0]}, 'peak_vehicles_in_frame': {'count': peak_v[1]['vehicle'], 'second': peak_v[0]}}
    json.dump(summary, open(os.path.join(a.out, 'summary.json'), 'w'), indent=1)
    pdf(summary, per_min, ev_by_min, os.path.join(a.out, 'summary.pdf'))
    # the annotated video: re-encoded to H.264 when an ffmpeg is found and works, else kept as written (MPEG-4 Part 2)
    if writer is not None and os.path.exists(raw):
        ff = shutil.which(a.ffmpeg) if a.ffmpeg else None; encoded = False
        if ff:
            r = subprocess.run([ff, '-y', '-loglevel', 'error', '-i', raw, '-c:v', 'libx264', '-preset', 'fast', '-crf', '23', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', final], check=False)
            encoded = r.returncode == 0 and os.path.exists(final)
            if not encoded: print('ffmpeg did not finish; the annotated video stays MPEG-4 Part 2', flush=True)
        elif a.ffmpeg: print('ffmpeg not found (' + a.ffmpeg + '); the annotated video stays MPEG-4 Part 2', flush=True)
        if encoded: os.remove(raw)
        else:
            if os.path.exists(final): os.remove(final)
            os.rename(raw, final)
    open(os.path.join(a.out, 'README.txt'), 'w').write('SeeDeep video counts\n\ncounts_per_second.csv: the most people, vehicles and other things in frame in each second.\ncounts_per_minute.csv: per minute, the most and the average in frame, and the line crossings.\nseen_by_class.csv: how many tracked things of each class were seen (a thing the detector loses for more than 12 frames and finds again counts twice; two that swap places can count as one).\ncrossings.csv: every crossing of the line with its time, direction and track.\n' + ('annotated.mp4: the video with the boxes and the line drawn.\n' if os.path.exists(final) else '') + 'summary.pdf and summary.json: the totals, the peaks, the method and its limits.\n\nDetector: MediaPipe EfficientDet-Lite0 (Apache 2.0), 80 COCO classes; it does not read faces, plates or text. Counts can be wrong' + ('; the annotated video is there so you can judge' if os.path.exists(final) else '') + '.\n')
    print(json.dumps({k: summary[k] for k in ('frames_processed', 'duration_s', 'ms_per_detection_mean', 'people_seen', 'vehicles_seen', 'crossings', 'peak_people_in_frame')}))

def pdf(s, per_min, ev, path):
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
    c = canvas.Canvas(path, pagesize=letter); Wp, Hp = letter; y = Hp - inch
    def line(txt, size=11, dy=16, bold=False):
        nonlocal y
        if y < inch: c.showPage(); y = Hp - inch
        c.setFont('Helvetica-Bold' if bold else 'Helvetica', size); c.drawString(inch, y, txt); y -= dy
    line('SeeDeep video counts', 20, 28, True)
    line(f"{s['input']}" + (f" for {s['title']}" if s['title'] else '') + f", made {s['made'][:16].replace('T', ' ')} UTC", 11, 22)
    line('What was counted', 14, 20, True)
    line(f"Length {s['duration_s'] / 60:.1f} minutes, {s['width']} by {s['height']} at {s['fps']} frames a second; {s['frames_processed']} frames looked at" + (f" (one in every {s['every_nth_frame']})" if s['every_nth_frame'] > 1 else '') + '.')
    line(f"People seen: {s['people_seen']}. Vehicles seen: {s['vehicles_seen']}. Peak people in frame: {s['peak_people_in_frame']['count']} at {s['peak_people_in_frame']['second'] // 60}:{s['peak_people_in_frame']['second'] % 60:02d}. Peak vehicles in frame: {s['peak_vehicles_in_frame']['count']} at {s['peak_vehicles_in_frame']['second'] // 60}:{s['peak_vehicles_in_frame']['second'] % 60:02d}.")
    if s['line']:
        line(f"Line at {round(s['line']['fraction_of_height'] * 100)}% of the height" + (f" ({s['line']['text']})" if s['line']['text'] else '') + f": people crossed down {s['crossings']['person']['down']}, up {s['crossings']['person']['up']}; vehicles down {s['crossings']['vehicle']['down']}, up {s['crossings']['vehicle']['up']}.")
    y -= 6; line('Seen, by class', 14, 20, True)
    for cls, n in sorted(s['seen'].items(), key=lambda x: -x[1])[:16]: line(f"{cls}: {n}", 11, 14)
    y -= 6; line('By minute (most in frame; crossings)', 14, 20, True)
    line('minute   people   vehicles   people down/up   vehicles down/up', 10, 14, True)
    for m in sorted(per_min):
        r = per_min[m]; e = ev.get(m, {'person_down': 0, 'person_up': 0, 'vehicle_down': 0, 'vehicle_up': 0})
        line(f"{m:>6}   {r['person_max']:>6}   {r['vehicle_max']:>8}   {e['person_down']:>6} / {e['person_up']:<6}   {e['vehicle_down']:>6} / {e['vehicle_up']}", 10, 13)
    # a small bar chart of people per minute
    if per_min:
        if y < 3 * inch: c.showPage(); y = Hp - inch
        y -= 10; line('People in frame, most per minute', 12, 16, True)
        mx = max(r['person_max'] for r in per_min.values()) or 1; n = len(per_min); bw = min(18, (Wp - 2 * inch) / max(1, n)); base = y - 110
        c.setFillColorRGB(0.95, 0.65, 0.09)
        for j, m in enumerate(sorted(per_min)): h = 100 * per_min[m]['person_max'] / mx; c.rect(inch + j * bw, base, bw - 2, h, fill=1, stroke=0)
        c.setFillColorRGB(0, 0, 0); c.setFont('Helvetica', 8); c.drawString(inch, base - 12, f"minute 0 to {n - 1}; tallest bar = {mx}"); y = base - 30
    y -= 6; line('Method and limits', 14, 20, True)
    for t in [f"Detector: {s['detector']} (MediaPipe, Google, Apache 2.0; 80 everyday classes; it does not read faces, plates or text).", f"Tracking: {s['tracker']}. A thing the detector loses for more than 12 frames and finds again counts twice; two that swap places can count as one.", f"Speed on the machine that made this: {s['ms_per_detection_mean']} ms a detection.", 'Accuracy was not measured on this footage; the annotated video is included so you can judge the counts yourself. Counts can be wrong.', 'Made by count_video.py on the machine that ran it; the script itself makes no network request.']:
        words = t.split(); buf = ''
        for w in words:
            if c.stringWidth(buf + ' ' + w, 'Helvetica', 10) > Wp - 2 * inch: line(buf, 10, 13); buf = w
            else: buf = (buf + ' ' + w).strip()
        line(buf, 10, 15)
    c.save()

if __name__ == '__main__': main()
