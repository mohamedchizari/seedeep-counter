"""File identity for the mark count_video.py keeps in its output folder (.seedeep-counter-output).

A file is recorded by its size and the SHA-256 of its content. The content, not where the file sits on a disk, says whether it
is still the file this program wrote: a move into the output folder, or a later copy, backup restore or sync of that folder,
keeps it, and a file put in its place is caught even with the same size and modified time. Marks written by earlier versions
(name, size, modified time) are still read and are matched on those two values. Before a listed file counts as the same it
is read (through to the end for this version's records, its first block for an earlier version's), so one that cannot be
read is told apart."""
import hashlib, os, re, stat

_HEX = re.compile(r'[0-9a-f]{64}')
_READ = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_BINARY', 0)


def record_of_fd(fd):
    """('sha256', size, hex digest) of the open file: the size is the number of bytes read for the hash."""
    h, size = hashlib.sha256(), 0
    while True:
        b = os.read(fd, 1 << 20)
        if not b:
            return ('sha256', size, h.hexdigest())
        h.update(b)
        size += len(b)


def stamp(path):
    """The record of a plain file; a link is not followed (OSError)."""
    fd = os.open(path, _READ)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError('not a plain file: ' + str(path))
        return record_of_fd(fd)
    finally:
        os.close(fd)


def first_block(path):
    """Read the first block of a plain file, a link not followed: enough to tell that it can be read (OSError if not)."""
    fd = os.open(path, _READ)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError('not a plain file: ' + str(path))
        os.read(fd, 1 << 16)
    finally:
        os.close(fd)


def format_row(name, record):
    """One line of the mark: name, size and sha256:<hex>; a record from an earlier version keeps its name, size and modified time."""
    if record[0] == 'sha256':
        return f'{name}\t{record[1]}\tsha256:{record[2]}'
    return f'{name}\t{record[1]}\t{record[2]}'


def parse_records(rows):
    """Read mark rows: name, size and sha256:<64 hex> (this version), or name, size and modified time (earlier versions).
    Any other row is not read, so the file it names counts as one this program did not put there."""
    marks = {}
    for row in rows:
        fields = row.split('\t')
        if len(fields) == 3 and fields[2].startswith('sha256:') and fields[1].isascii() and fields[1].isdigit() and _HEX.fullmatch(fields[2][7:]):
            marks[fields[0]] = ('sha256', int(fields[1]), fields[2][7:])
        elif len(fields) == 3 and all(v.isascii() and v.isdigit() for v in fields[1:]):
            marks[fields[0]] = ('time', int(fields[1]), int(fields[2]))
    return marks


def check(path, recorded):
    """'same' when path is a plain file (not a link) that the record describes: same size and content for this version's
    records, same size and modified time for an earlier version's. 'unreadable' when it would otherwise be the same but
    cannot be read (its permissions, say, or a file kept only in the cloud while offline), whichever version wrote the
    record: a file under an earlier version's record is read only as far as its first block. 'other' for anything else."""
    if not recorded or recorded[0] not in ('sha256', 'time'):
        return 'other'
    try:
        st = os.lstat(path)
    except OSError:
        return 'other'
    if not stat.S_ISREG(st.st_mode) or st.st_size != recorded[1]:
        return 'other'
    if recorded[0] == 'time' and st.st_mtime_ns != recorded[2]:
        return 'other'
    try:
        content = stamp(path) if recorded[0] == 'sha256' else first_block(path)
    except OSError:
        return 'unreadable'
    if recorded[0] == 'time':
        return 'same'  # an earlier version's record: size and modified time, and the file's first block can be read
    return 'same' if content == tuple(recorded) else 'other'


def matches_mark(path, recorded):
    """True when check() finds the file the record describes."""
    return check(path, recorded) == 'same'
