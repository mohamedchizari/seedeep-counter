"""The tracker shared by count_video.py and the tests: greedy IoU matching within a class, an id per track,
a short memory for misses, and crossings of one horizontal line. Pure Python, no detector needed.

Rules (the same numbers as web/js/seedeep.js):
- a detection joins an existing track of the same class if their IoU is above 0.3; pairs are taken greedily, highest IoU
  first (equal ones to the earlier track), so a detection can end up with its second-best track when its best one was
  taken by a closer pair;
- a track's "hits" count the frame that created it and every frame it was matched in since;
- a track is "seen" (counted once in `seen`) when it reaches 3 hits;
- a track is dropped after more than 12 frames in a row without a match;
- a crossing is counted for people and vehicles only, from a track's third hit on, when the centre
  of its box passes the line between two matched frames ("down" = towards the bottom of the frame).
"""

VEHICLES = {'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'train'}


def group(c):
    return 'person' if c == 'person' else 'vehicle' if c in VEHICLES else 'other'


def iou(a, b):
    """Intersection over union of two boxes given as (x, y, width, height)."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1); u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0.0


class Tracker:
    def __init__(self):
        self.tracks = []; self.next_id = 1; self.seen = {}; self.cross = {'person': [0, 0], 'vehicle': [0, 0]}; self.events = []

    def update(self, dets, line_y, t_s):
        """dets: list of {'cls', 'score', 'box': (x, y, w, h)}; line_y: the line's height in pixels (0 = no line);
        t_s: the frame's time in seconds. Returns the tracks matched in this frame that have at least 2 hits."""
        pairs = []
        for ti, t in enumerate(self.tracks):
            for di, d in enumerate(dets):
                if d['cls'] == t['cls']:
                    s = iou(t['box'], d['box'])
                    if s > 0.3: pairs.append((s, ti, di))
        pairs.sort(key=lambda p: -p[0]); mt, md = set(), set()  # stable: equal overlaps go to the earlier track, as in seedeep.js
        for s, ti, di in pairs:
            if ti in mt or di in md: continue
            mt.add(ti); md.add(di); t, d = self.tracks[ti], dets[di]
            cy0 = t['box'][1] + t['box'][3] / 2; cy1 = d['box'][1] + d['box'][3] / 2
            t['box'] = d['box']; t['score'] = d['score']; t['hits'] += 1; t['misses'] = 0
            g = group(t['cls'])
            if line_y and t['hits'] >= 3 and g in ('person', 'vehicle'):
                if cy0 < line_y <= cy1: self.cross[g][0] += 1; self.events.append((t_s, g, t['cls'], 'down', t['id']))
                elif cy1 < line_y <= cy0: self.cross[g][1] += 1; self.events.append((t_s, g, t['cls'], 'up', t['id']))
        for ti, t in enumerate(self.tracks):
            if ti not in mt: t['misses'] += 1
        for di, d in enumerate(dets):
            if di not in md: self.tracks.append({'id': self.next_id, 'cls': d['cls'], 'box': d['box'], 'score': d['score'], 'hits': 1, 'misses': 0, 'counted': False}); self.next_id += 1
        self.tracks = [t for t in self.tracks if t['misses'] <= 12]
        for t in self.tracks:
            if not t['counted'] and t['hits'] >= 3: t['counted'] = True; self.seen[t['cls']] = self.seen.get(t['cls'], 0) + 1
        return [t for t in self.tracks if t['misses'] == 0 and t['hits'] >= 2]
