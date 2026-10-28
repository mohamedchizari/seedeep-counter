"""count_video.py end to end with a stand-in detector: the mark it writes and how a second run uses it.
Needs the counter's own packages (counter/requirements.txt); without them these tests are skipped."""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'counter'))
from output_mark import parse_records, stamp

NEEDS = ('cv2', 'numpy', 'mediapipe', 'reportlab')
MISSING = [m for m in NEEDS if importlib.util.find_spec(m) is None]
MARK = '.seedeep-counter-output'

# runs count_video.py's main with a detector that finds nothing, so no model file is needed; with SC_TEST_FPS set, the
# video reports that frame rate instead of its own
DRIVER = '''import os, runpy, sys, types
root = sys.argv[1]; sys.argv = [os.path.join(root, 'counter', 'count_video.py')] + sys.argv[2:]
sys.path.insert(0, os.path.join(root, 'counter'))
from mediapipe.tasks.python import vision
class Det:
    def detect_for_video(self, image, ms): return types.SimpleNamespace(detections=[])
vision.ObjectDetector.create_from_options = staticmethod(lambda options: Det())
fake = os.environ.get('SC_TEST_FPS')
if fake is not None:
    import cv2
    real = cv2.VideoCapture
    class Cap:
        def __init__(self, *args): self._cap = real(*args)
        def __getattr__(self, name): return getattr(self._cap, name)
        def get(self, prop): return float(fake) if prop == cv2.CAP_PROP_FPS else self._cap.get(prop)
    cv2.VideoCapture = Cap
runpy.run_path(sys.argv[0], run_name='__main__')
'''


@unittest.skipIf(MISSING, 'needs ' + ', '.join(MISSING))
class CountVideoMark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import cv2, numpy as np
        cls.tmp = tempfile.TemporaryDirectory()
        base = Path(cls.tmp.name)
        cls.clip = base / 'clip.avi'
        writer = cv2.VideoWriter(str(cls.clip), cv2.VideoWriter_fourcc(*'MJPG'), 10, (64, 48))
        for i in range(12):
            writer.write(np.full((48, 64, 3), i * 10, np.uint8))
        writer.release()
        cls.model = base / 'stand_in.tflite'
        cls.model.write_bytes(b'not a model: the detector is replaced')
        cls.driver = base / 'driver.py'
        cls.driver.write_text(DRIVER, encoding='utf-8')

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_counter(self, out, fps=None):
        env = dict(os.environ)
        env.pop('SC_TEST_FPS', None)
        if fps is not None:
            env['SC_TEST_FPS'] = fps
        r = subprocess.run([sys.executable, str(self.driver), str(ROOT), str(self.clip), str(out), '--no-video', '--ffmpeg', '',
                            '--model', str(self.model)], capture_output=True, text=True, timeout=300, env=env)
        return r.returncode, r.stdout + r.stderr

    def earlier_mark(self, out):
        """Rewrites the folder's mark as the version before this one wrote it: name, size, modified time in ns."""
        names = parse_records((out / MARK).read_text(encoding='utf-8').splitlines()[1:])
        rows = [f'{n}\t{(out / n).stat().st_size}\t{(out / n).stat().st_mtime_ns}\n' for n in names]
        (out / MARK).write_text('This folder holds the outputs of count_video.py. A new run into it replaces or deletes the files '
                                'listed here (name, size, modified time in ns), and only while their size and modified time are '
                                'unchanged.\n' + ''.join(rows), encoding='utf-8')
        return names

    def first_run(self, folder):
        out = Path(folder) / 'out'
        code, said = self.run_counter(out)
        self.assertEqual(code, 0, said)
        return out

    def test_the_mark_lists_each_output_by_its_size_and_content(self):
        with tempfile.TemporaryDirectory() as folder:
            out = self.first_run(folder)
            marks = parse_records((out / MARK).read_text(encoding='utf-8').splitlines()[1:])
            self.assertIn('summary.json', marks)
            self.assertIn('counts_per_second.csv', marks)
            for name, record in marks.items():
                self.assertEqual(record, stamp(out / name), name)

    def test_a_second_run_takes_its_own_outputs_also_in_a_copied_folder(self):
        with tempfile.TemporaryDirectory() as folder:
            out = self.first_run(folder)
            copy = Path(folder) / 'copy'
            shutil.copytree(out, copy, copy_function=shutil.copyfile)  # new files, new modified times
            for target in (out, copy):
                code, said = self.run_counter(target)
                self.assertEqual(code, 0, said)

    def test_an_output_changed_since_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            out = self.first_run(folder)
            path = out / 'summary.json'
            st = path.stat()
            data = bytearray(path.read_bytes())
            data[0:1] = b' ' if data[0:1] != b' ' else b'\t'
            path.write_bytes(bytes(data))
            os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))  # same size, same modified time, other content
            code, said = self.run_counter(out)
            self.assertEqual(code, 2, said)
            self.assertIn('did not put there or that changed since', said)

    @unittest.skipIf(hasattr(os, 'geteuid') and os.geteuid() == 0, 'root reads a file whatever its permissions')
    def test_an_output_that_cannot_be_read_is_named(self):
        with tempfile.TemporaryDirectory() as folder:
            out = self.first_run(folder)
            path = out / 'summary.json'
            path.chmod(0)
            try:
                code, said = self.run_counter(out)
            finally:
                path.chmod(0o644)
            self.assertEqual(code, 2, said)
            self.assertIn('cannot be read now', said)


    def test_a_folder_with_an_earlier_mark_is_taken_and_given_the_new_mark(self):
        with tempfile.TemporaryDirectory() as folder:
            out = self.first_run(folder)
            names = self.earlier_mark(out)
            self.assertTrue(names)
            self.assertIn('modified time in ns', (out / MARK).read_text(encoding='utf-8'))
            code, said = self.run_counter(out)
            self.assertEqual(code, 0, said)
            marks = parse_records((out / MARK).read_text(encoding='utf-8').splitlines()[1:])
            self.assertTrue(all(record[0] == 'sha256' for record in marks.values()), marks)

    @unittest.skipIf(hasattr(os, 'geteuid') and os.geteuid() == 0, 'root reads a file whatever its permissions')
    def test_an_unreadable_file_under_an_earlier_mark_is_named(self):
        with tempfile.TemporaryDirectory() as folder:
            out = self.first_run(folder)
            self.earlier_mark(out)
            path = out / 'summary.json'
            path.chmod(0)
            try:
                code, said = self.run_counter(out)
            finally:
                path.chmod(0o644)
            self.assertEqual(code, 2, said)
            self.assertIn('cannot be read now', said)
            self.assertIn('summary.json', said)

    def test_a_frame_rate_that_cannot_be_right_stops_the_run_before_anything_is_written(self):
        with tempfile.TemporaryDirectory() as folder:
            for fps in ('nan', '-5', '1e-9', '5000'):
                with self.subTest(fps=fps):
                    out = Path(folder) / ('out_' + fps)
                    code, said = self.run_counter(out, fps=fps)
                    self.assertEqual(code, 2, said)
                    self.assertIn('frame rate', said)
                    self.assertFalse(os.path.lexists(out), said)


if __name__ == '__main__':
    unittest.main()
