"""Tests of counter/tracker.py on made-up boxes (no detector, no video). Run from the repository root:
    python -m unittest discover -s tests -v
"""
import os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'counter'))
from tracker import Tracker, iou, group


def det(cls, x, y, w=40, h=80, score=0.9):
    return {'cls': cls, 'score': score, 'box': (x, y, w, h)}


class Iou(unittest.TestCase):
    def test_same_box(self):
        self.assertAlmostEqual(iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)

    def test_apart(self):
        self.assertEqual(iou((0, 0, 10, 10), (20, 20, 10, 10)), 0.0)

    def test_half_overlap(self):
        # 10x10 boxes shifted by 5: intersection 50, union 150
        self.assertAlmostEqual(iou((0, 0, 10, 10), (5, 0, 10, 10)), 50 / 150)

    def test_empty_boxes(self):
        self.assertEqual(iou((0, 0, 0, 0), (0, 0, 0, 0)), 0.0)


class Groups(unittest.TestCase):
    def test_groups(self):
        self.assertEqual(group('person'), 'person')
        for c in ('car', 'truck', 'bus', 'motorcycle', 'bicycle', 'train'): self.assertEqual(group(c), 'vehicle')
        for c in ('dog', 'handbag', 'boat', 'airplane'): self.assertEqual(group(c), 'other')


class Tracking(unittest.TestCase):
    def walk(self, tr, cls, ys, line_y=0, x=100):
        out = None
        for i, y in enumerate(ys): out = tr.update([det(cls, x, y)], line_y, i / 25)
        return out

    def test_seen_after_three_matched_frames(self):
        tr = Tracker(); self.walk(tr, 'person', [100, 102])
        self.assertEqual(tr.seen, {})
        tr.update([det('person', 100, 104)], 0, 0.12)
        self.assertEqual(tr.seen, {'person': 1})
        self.walk(tr, 'person', [106, 108, 110])
        self.assertEqual(tr.seen, {'person': 1}, 'a track is counted once')

    def test_returned_tracks_need_two_hits(self):
        tr = Tracker()
        self.assertEqual(tr.update([det('person', 100, 100)], 0, 0), [])
        self.assertEqual(len(tr.update([det('person', 100, 101)], 0, 0.04)), 1)

    def test_overlap_of_exactly_point_three_does_not_match(self):
        # 13 wide, 7 apart: intersection 6 x 80, union 20 x 80, IoU exactly 0.3; the rule is "above 0.3"
        tr = Tracker(); tr.update([det('person', 100, 100, w=13)], 0, 0); tr.update([det('person', 107, 100, w=13)], 0, 0.04)
        self.assertEqual(len(tr.tracks), 2)

    def test_overlap_of_a_third_matches(self):
        # 40 wide, 20 apart: IoU 20 / 60
        tr = Tracker(); tr.update([det('person', 100, 100)], 0, 0); tr.update([det('person', 120, 100)], 0, 0.04)
        self.assertEqual([t['id'] for t in tr.tracks], [1])

    def test_a_match_resets_the_misses(self):
        # 8 misses, a match, 8 more misses: 16 in all but never 12 in a row, so the track lives
        tr = Tracker(); self.walk(tr, 'person', [100, 101])
        for k in range(2):
            for i in range(8): tr.update([], 0, 1 + i / 25)
            tr.update([det('person', 100, 101)], 0, 2)
        self.assertEqual([t['id'] for t in tr.tracks], [1])

    def test_a_track_missed_now_is_not_returned(self):
        tr = Tracker(); self.walk(tr, 'person', [100, 101, 102])
        self.assertEqual(tr.update([], 0, 1), [])

    def test_equal_overlaps_go_to_the_earlier_track(self):
        # two tracks 20 apart, then one detection halfway: IoU 0.6 with each; the earlier track (id 1) takes it
        tr = Tracker(); tr.update([det('person', 100, 100), det('person', 120, 100)], 0, 0)
        tr.update([det('person', 110, 100)], 0, 0.04)
        hits = {t['id']: t['hits'] for t in tr.tracks}
        self.assertEqual(hits, {1: 2, 2: 1})

    def test_low_overlap_starts_a_new_track(self):
        # 40x80 boxes 25 px apart: IoU 1200 / 5200 = 0.23, under the 0.3 needed to match
        tr = Tracker(); tr.update([det('person', 100, 100)], 0, 0); tr.update([det('person', 125, 100)], 0, 0.04)
        self.assertEqual(sorted(t['id'] for t in tr.tracks), [1, 2])

    def test_classes_never_match(self):
        tr = Tracker(); tr.update([det('person', 100, 100)], 0, 0); tr.update([det('dog', 100, 100)], 0, 0.04)
        self.assertEqual(len(tr.tracks), 2)

    def test_crossing_down_and_up(self):
        # box height 80, so the centre is y + 40; the line is at 300: centre goes 220..380 (down), then back (up)
        tr = Tracker(); ys = list(range(180, 341, 20)); self.walk(tr, 'person', ys + ys[::-1], line_y=300)
        self.assertEqual(tr.cross['person'], [1, 1])
        self.assertEqual([e[3] for e in tr.events], ['down', 'up'])

    def test_crossing_on_the_third_hit_is_counted(self):
        # centres 280, 290, 302: the line at 300 is passed between the second and the third hit
        tr = Tracker(); self.walk(tr, 'person', [240, 250, 262], line_y=300)
        self.assertEqual(tr.cross['person'], [1, 0])

    def test_crossing_too_early_is_not_counted(self):
        # the centre passes the line between the first and second frame, before the track has 3 hits
        tr = Tracker(); self.walk(tr, 'person', [250, 270, 275, 280], line_y=300)
        self.assertEqual(tr.cross['person'], [0, 0])

    def test_other_group_does_not_cross(self):
        tr = Tracker(); self.walk(tr, 'dog', list(range(180, 341, 20)), line_y=300)
        self.assertEqual(tr.cross, {'person': [0, 0], 'vehicle': [0, 0]}); self.assertEqual(tr.seen, {'dog': 1})

    def test_vehicle_crossing(self):
        tr = Tracker(); self.walk(tr, 'car', list(range(180, 341, 20)), line_y=300)
        self.assertEqual(tr.cross['vehicle'], [1, 0])

    def test_dropped_after_twelve_misses_and_counted_again(self):
        # the known limit: a thing that leaves for more than 12 frames and comes back is a new track, seen twice
        tr = Tracker(); self.walk(tr, 'person', [100, 101, 102])
        for i in range(12): tr.update([], 0, 1 + i / 25)
        self.assertEqual(len(tr.tracks), 1, 'kept through 12 misses')
        tr.update([], 0, 2); self.assertEqual(tr.tracks, [], 'dropped on the 13th')
        self.walk(tr, 'person', [100, 101, 102])
        self.assertEqual(tr.seen, {'person': 2})

    def test_a_new_track_is_kept_through_twelve_misses(self):
        tr = Tracker(); tr.update([det('person', 100, 100)], 0, 0)
        for i in range(12): tr.update([], 0, (i + 1) / 25)
        self.assertEqual(len(tr.tracks), 1, 'a track seen once is kept through 12 misses too')
        tr.update([], 0, 13 / 25); self.assertEqual(tr.tracks, [], 'and dropped on the 13th')

    def test_score_follows_the_latest_match(self):
        tr = Tracker(); tr.update([det('person', 100, 100, score=0.5)], 0, 0)
        out = tr.update([det('person', 101, 100, score=0.8)], 0, 0.04)
        self.assertEqual(out[0]['score'], 0.8)

    def test_greedy_match_takes_highest_iou(self):
        tr = Tracker(); tr.update([det('person', 100, 100)], 0, 0)
        tr.update([det('person', 115, 100), det('person', 104, 100)], 0, 0.04)
        ids = {t['box'][0]: t['id'] for t in tr.tracks}
        self.assertEqual(ids[104], 1, 'the closer box keeps the old id'); self.assertEqual(ids[115], 2)


if __name__ == '__main__':
    unittest.main()
