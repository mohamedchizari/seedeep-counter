"""Focused regressions for downloader, timing metadata, and output ownership."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'counter'))
import output_mark
from output_mark import check, format_row, matches_mark, parse_records, record_of_fd, stamp
from video_timing import effective_fps


ROOT = Path(__file__).resolve().parent.parent


class VideoTiming(unittest.TestCase):
    def test_missing_rate_uses_documented_fallback(self):
        self.assertEqual(effective_fps(0), 25.0)

    def test_positive_rate_is_preserved(self):
        self.assertEqual(effective_fps(29.97), 29.97)

    def test_invalid_rates_are_rejected(self):
        for rate in (-1.0, float('nan'), float('inf'), float('-inf')):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                effective_fps(rate)

    def test_rates_outside_the_documented_range_are_rejected(self):
        for rate in (1e-9, 0.05, 0.0999, 1000.0001, 1000.5, 90000.0):
            with self.subTest(rate=rate), self.assertRaises(ValueError) as refused:
                effective_fps(rate)
            self.assertIn(f'frame rate {rate!r} is outside', str(refused.exception))  # not rounded: 1000.0001 is not shown as 1000

    def test_rates_at_the_ends_of_the_range_are_kept(self):
        for rate in (0.1, 1.0, 240.0, 1000.0):
            with self.subTest(rate=rate):
                self.assertEqual(effective_fps(rate), rate)


class OutputMark(unittest.TestCase):
    def _file(self, folder, name, data):
        path = Path(folder) / name
        path.write_bytes(data)
        return path

    def test_a_file_replaced_with_same_size_and_mtime_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self._file(folder, 'output.csv', b'first')
            original = stamp(path)
            mtime = path.stat().st_mtime_ns
            replacement = self._file(folder, 'replacement.csv', b'other')
            os.utime(replacement, ns=(mtime, mtime))
            os.replace(replacement, path)
            self.assertEqual((path.stat().st_size, path.stat().st_mtime_ns), (5, mtime))
            self.assertFalse(matches_mark(path, original))

    def test_content_changed_in_place_with_same_size_and_mtime_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self._file(folder, 'output.csv', b'first')
            original = stamp(path)
            st = path.stat()
            with open(path, 'r+b') as f:
                f.write(b'other')
            os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
            self.assertEqual((path.stat().st_ino, path.stat().st_size, path.stat().st_mtime_ns), (st.st_ino, st.st_size, st.st_mtime_ns))
            self.assertFalse(matches_mark(path, original))

    def test_a_copied_or_restored_output_folder_is_still_recognised(self):
        # cp -a, a backup restore or a sync client makes new files (new inode, maybe another device) with the same content
        with tempfile.TemporaryDirectory() as folder:
            src = self._file(folder, 'summary.json', b'{"people_seen": 3}')
            record = stamp(src)
            kept_time = Path(folder) / 'copy_a' / 'summary.json'
            kept_time.parent.mkdir()
            shutil.copy2(src, kept_time)
            new_time = Path(folder) / 'copy_b' / 'summary.json'
            new_time.parent.mkdir()
            shutil.copyfile(src, new_time)
            os.utime(new_time, ns=(0, 2_000_000_000))  # a disk that keeps times to 2 s, or a copy that does not keep them
            self.assertNotEqual(kept_time.stat().st_ino, src.stat().st_ino)
            self.assertTrue(matches_mark(kept_time, record))
            self.assertTrue(matches_mark(new_time, record))

    def test_a_foreign_file_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            ours = self._file(folder, 'counts_per_second.csv', b'second,people\n0,1\n')
            theirs = self._file(folder, 'crossings.csv', b'second,people\n0,2\n')
            self.assertFalse(matches_mark(theirs, stamp(ours)))
            self.assertFalse(matches_mark(theirs, ()))  # a name the mark does not list

    def test_mark_taken_before_a_move_matches_after_it(self):
        # count_video.py writes the mark in its work folder before it moves the files into the output folder
        with tempfile.TemporaryDirectory() as folder:
            work = Path(folder) / '.seedeep-counter-work'
            work.mkdir()
            src = self._file(work, 'summary.json', b'{}')
            fd = os.open(src, os.O_RDONLY)
            try:
                before = record_of_fd(fd)
            finally:
                os.close(fd)
            self.assertEqual(before, stamp(src))
            dst = Path(folder) / 'summary.json'
            os.replace(src, dst)
            self.assertTrue(matches_mark(dst, before))

    def test_earlier_marks_are_read_and_matched_on_size_and_modified_time(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self._file(folder, 'output.csv', b'content')
            st = path.stat()
            marks = parse_records([f'two.csv\t{st.st_size}\t{st.st_mtime_ns}'])
            self.assertEqual(marks, {'two.csv': ('time', st.st_size, st.st_mtime_ns)})
            copy = Path(folder) / 'copy' / 'output.csv'
            copy.parent.mkdir()
            shutil.copy2(path, copy)  # another inode: device and inode in an old mark are not used
            for record in marks.values():
                self.assertTrue(matches_mark(path, record))
                self.assertTrue(matches_mark(copy, record))
            os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
            self.assertFalse(matches_mark(path, marks['two.csv']))
            with open(copy, 'ab') as f:
                f.write(b'+')
            os.utime(copy, ns=(st.st_atime_ns, st.st_mtime_ns))
            self.assertFalse(matches_mark(copy, marks['two.csv']))  # same time, another size

    def test_malformed_rows_are_ignored(self):
        good = '0' * 64
        rows = [
            'new.csv\t10\tsha256:' + good,
            'three_numbers.csv\t10\t20\t30',
            'five_fields.csv\t10\t20\t1\t2',  # no released version wrote this form
            'six.csv\t10\t20\t1\t2\t30',
            'short_hash.csv\t10\tsha256:' + '0' * 63,
            'upper_hash.csv\t10\tsha256:' + 'A' * 64,
            'no_size.csv\tx\tsha256:' + good,
            'broken.csv\t10\tinvalid',
        ]
        self.assertEqual(parse_records(rows), {'new.csv': ('sha256', 10, good)})

    def test_an_unreadable_listed_file_is_told_apart(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self._file(folder, 'summary.json', b'{}')
            record = stamp(path)
            self.assertEqual(check(path, record), 'same')
            with mock.patch.object(output_mark.os, 'open', side_effect=PermissionError(13, 'Permission denied')):
                self.assertEqual(check(path, record), 'unreadable')
                self.assertFalse(matches_mark(path, record))
            self.assertEqual(check(path, ()), 'other')  # not listed: never read
            path.write_bytes(b'{"x": 1}')
            self.assertEqual(check(path, record), 'other')  # another size: not read either

    def test_an_unreadable_file_under_an_earlier_mark_is_told_apart(self):
        # a mark written by an earlier version (size and modified time): the file is read too (its first block) before it counts
        with tempfile.TemporaryDirectory() as folder:
            path = self._file(folder, 'summary.json', b'{}')
            st = path.stat()
            record = ('time', st.st_size, st.st_mtime_ns)
            self.assertEqual(check(path, record), 'same')
            with mock.patch.object(output_mark.os, 'open', side_effect=PermissionError(13, 'Permission denied')) as opened:
                self.assertEqual(check(path, record), 'unreadable')
                self.assertFalse(matches_mark(path, record))
                opened.reset_mock()
                self.assertEqual(check(path, ('time', st.st_size, st.st_mtime_ns + 1)), 'other')  # another time: not read
                opened.assert_not_called()

    def test_rows_written_are_read_back(self):
        records = {'a.csv': ('sha256', 7, 'f' * 64), 'b.csv': ('time', 7, 123456789)}
        self.assertEqual(parse_records([format_row(n, r) for n, r in records.items()]), records)

    def test_symlink_is_not_owned_by_a_mark(self):
        with tempfile.TemporaryDirectory() as folder:
            target = self._file(folder, 'target', b'content')
            link = Path(folder) / 'output'
            link.symlink_to(target)
            self.assertFalse(matches_mark(link, stamp(target)))
            self.assertFalse(matches_mark(link, ('time', target.stat().st_size, target.stat().st_mtime_ns)))
            with self.assertRaises(OSError):
                stamp(link)


class AssetDownloader(unittest.TestCase):
    @unittest.skipUnless(shutil.which('bash'), 'requires bash')
    def test_download_does_not_follow_existing_part_symlink(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'repo'
            root.mkdir()
            shutil.copy2(ROOT / 'fetch_assets.sh', root / 'fetch_assets.sh')
            models = root / 'models'
            models.mkdir()
            victim = Path(folder) / 'victim'
            victim.write_text('keep this file intact', encoding='utf-8')
            part = models / 'efficientdet_lite0.tflite.part'
            part.symlink_to(victim)

            mock_bin = Path(folder) / 'bin'
            mock_bin.mkdir()
            curl = mock_bin / 'curl'
            curl.write_text(
                '#!/usr/bin/env python3\n'
                'import pathlib, sys\n'
                'args = sys.argv[1:]\n'
                'pathlib.Path(args[args.index("-o") + 1]).write_bytes(b"not the expected asset")\n',
                encoding='utf-8',
            )
            curl.chmod(0o755)
            env = os.environ.copy()
            env['PATH'] = str(mock_bin) + os.pathsep + env['PATH']
            result = subprocess.run(
                ['bash', str(root / 'fetch_assets.sh')],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(victim.read_text(encoding='utf-8'), 'keep this file intact')
            self.assertTrue(part.is_symlink())
            self.assertEqual(list(models.glob('*.part.*')), [])


    @unittest.skipUnless(shutil.which('bash'), 'requires bash')
    def test_downloaded_files_follow_the_umask(self):
        line = [l for l in (ROOT / 'fetch_assets.sh').read_text(encoding='utf-8').splitlines() if l.startswith('FILE_MODE=')]
        self.assertEqual(len(line), 1)
        for umask, mode in (('022', '644'), ('027', '640'), ('077', '600'), ('002', '664')):
            out = subprocess.run(['bash', '-c', f'umask {umask}\n{line[0]}\nprintf %s "$FILE_MODE"'], capture_output=True, text=True, timeout=10)
            self.assertEqual(out.stdout, mode, umask)
        self.assertNotIn('chmod 644', (ROOT / 'fetch_assets.sh').read_text(encoding='utf-8'))

    @unittest.skipUnless(shutil.which('bash'), 'requires bash')
    def test_a_run_leaves_every_file_with_the_mode_of_the_umask(self):
        # the whole script, with a made-up curl and the expected checksums swapped for those of the made-up files:
        # each fetched file and the copy in web/models must have the mode a new file gets under the umask (mktemp gives 600)
        import hashlib, re
        script = (ROOT / 'fetch_assets.sh').read_text(encoding='utf-8')
        urls = {'MODEL_URL': re.search(r'^MODEL_URL="([^"]+)"', script, re.M).group(1), 'NPM': re.search(r'^NPM="([^"]+)"', script, re.M).group(1)}
        gets = re.findall(r'^get "\$(MODEL_URL|NPM)([^"]*)" (\S+) ([0-9a-f]{64})$', script, re.M)
        self.assertEqual(len(gets), 6)
        for var, rest, dest, digest in gets:
            body = ('made-up content of ' + urls[var] + rest).encode('utf-8')
            script = script.replace(digest, hashlib.sha256(body).hexdigest())
        for umask, mode in (('022', 0o644), ('027', 0o640)):
            with self.subTest(umask=umask), tempfile.TemporaryDirectory() as folder:
                root = Path(folder) / 'repo'
                root.mkdir()
                (root / 'fetch_assets.sh').write_text(script, encoding='utf-8')
                mock_bin = Path(folder) / 'bin'
                mock_bin.mkdir()
                curl = mock_bin / 'curl'
                curl.write_text(
                    '#!/usr/bin/env python3\n'
                    'import pathlib, sys\n'
                    'args = sys.argv[1:]\n'
                    'pathlib.Path(args[args.index("-o") + 1]).write_bytes(("made-up content of " + args[-3]).encode("utf-8"))\n',
                    encoding='utf-8',
                )
                curl.chmod(0o755)
                env = os.environ.copy()
                env['PATH'] = str(mock_bin) + os.pathsep + env['PATH']
                result = subprocess.run(['bash', '-c', f'umask {umask} && exec bash fetch_assets.sh'], cwd=root, env=env,
                                        capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                made = [root / dest for _, _, dest, _ in gets] + [root / 'web' / 'models' / 'efficientdet_lite0.tflite']
                for path in made:
                    self.assertEqual(path.stat().st_mode & 0o777, mode, str(path.relative_to(root)))
                self.assertEqual(sorted(p.name for p in root.rglob('*.part.*')), [])

if __name__ == '__main__':
    unittest.main()
