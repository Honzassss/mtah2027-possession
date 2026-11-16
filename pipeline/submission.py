"""The submission file: written, read back from disk and checked before it counts.

    digest = write(path, frame, labels)          # frame,label CSV; returns its SHA-256
    reason, codes = check(path)                  # reason None when the file passes, else one line
    n = spell_count(codes)                       # our own H/A spells inside the two halves

The format rules are SUBMISSION.md's, read the way upstream/evaluation/validate_submission.py applies
them (header frame,label, 199,319 rows, frame 0..199318 in order, each label one of H A D X, no blank
cell), only stricter about spelling, since the file checked is always our own. On top of them, ours:
no X on a graded frame (one inside a published half), X on every other frame, and D on every frame of a
published dead block. Reasons name a rule and a count, never a frame number or a label.
"""
import hashlib

import numpy as np

from .epts import COMPETITION_DEAD_BLOCKS, COMPETITION_FRAMES, COMPETITION_HALVES

LABELS = ("H", "A", "D", "X")
H, A, D, X = range(4)
HEADER = "frame,label"


def graded_mask(n_frames=COMPETITION_FRAMES, halves=COMPETITION_HALVES):
    mask = np.zeros(n_frames, dtype=bool)
    for _, first, last in halves:
        mask[first:last + 1] = True
    return mask


def dead_mask(n_frames=COMPETITION_FRAMES, blocks=COMPETITION_DEAD_BLOCKS):
    mask = np.zeros(n_frames, dtype=bool)
    for first, last in blocks:
        mask[first:last + 1] = True
    return mask


def write(path, frame, labels):
    """Write one frame,label row per entry, in the order given. Returns the file's SHA-256."""
    frame, labels = np.asarray(frame), np.asarray(labels)
    if frame.ndim != 1 or frame.shape != labels.shape:
        raise ValueError("frame and labels must be (F,) each")
    rows = "".join("{},{}\n".format(f, lab) for f, lab in zip(frame.tolist(), labels.tolist()))
    data = (HEADER + "\n" + rows).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(data)
    return hashlib.sha256(data).hexdigest()


def check(path, n_frames=COMPETITION_FRAMES, halves=COMPETITION_HALVES, dead_blocks=COMPETITION_DEAD_BLOCKS):
    """(reason, codes). reason is None when the file on disk passes every rule; codes is the (F,) int8
    label column (H 0, A 1, D 2, X 3) whenever the rows could be read, else None."""
    try:
        with open(path, "rb") as fh:
            text = fh.read().decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return "the file cannot be read as UTF-8 text", None
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if not lines or lines[0].rstrip("\r") != HEADER:
        return "the header is not exactly {}".format(HEADER), None
    rows = lines[1:]
    if len(rows) != n_frames:
        return "row count {} != {}".format(len(rows), n_frames), None

    lookup = {name: code for code, name in enumerate(LABELS)}
    codes = np.full(n_frames, -1, dtype=np.int8)
    bad_shape = bad_frame = blank = bad_label = 0
    for i, row in enumerate(rows):
        parts = row.rstrip("\r").split(",")
        if len(parts) != 2:
            bad_shape += 1
            continue
        number, label = parts
        if not number.isdigit() or int(number) != i:
            bad_frame += 1
        if label == "":
            blank += 1
        elif label not in lookup:
            bad_label += 1
        else:
            codes[i] = lookup[label]
    if bad_shape:
        return "{} rows do not have exactly two fields".format(bad_shape), None
    if bad_frame:
        return "frame column is not 0..{} in order ({} rows off)".format(n_frames - 1, bad_frame), None
    if blank:
        return "empty label in {} rows".format(blank), None
    if bad_label:
        return "{} labels are not one of {}".format(bad_label, " ".join(LABELS)), None

    graded = graded_mask(n_frames, halves)
    x_graded = int(np.sum(graded & (codes == X)))
    if x_graded:
        return "X on {} graded frames".format(x_graded), codes
    labelled_outside = int(np.sum(~graded & (codes != X)))
    if labelled_outside:
        return "{} frames outside the halves are not X".format(labelled_outside), codes
    dead = dead_mask(n_frames, dead_blocks)
    live_in_dead = int(np.sum(dead & (codes != D)))
    if live_in_dead:
        return "{} frames of a published dead block are not D".format(live_in_dead), codes
    return None, codes


def spell_count(codes, halves=COMPETITION_HALVES):
    """Our own H/A spells: maximal runs of one side's label inside each half. With no X inside a half
    (check() makes sure) these are the spells the scorer matches against the truth."""
    n = 0
    for _, first, last in halves:
        w = np.asarray(codes[first:last + 1])
        if not len(w):
            continue
        starts = np.concatenate(([True], w[1:] != w[:-1]))
        n += int(np.sum(starts & ((w == H) | (w == A))))
    return n
