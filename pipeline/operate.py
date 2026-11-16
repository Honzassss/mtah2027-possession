"""The harness of the scripts the operator runs on the competition match (scripts/match_*.py).

The pattern of scripts/predict_match.py, shared:

    explicit paths   every input and the report come in as arguments; nothing reads the environment
    silence          while the job runs, everything Python or a library writes to fd 1 and 2 goes to
                     /dev/null, warnings included
    one-line errors  a failure prints `FAIL <stage>: <reason>`: the message of this repository's own
                     errors, which name a rule, a count or a path and never a value from the data, and
                     the error type for anything else
    the report       every figure goes to a text file for the operator's eyes only, written whole or
                     not at all, and outside the repository whenever the bundle is outside it

So stdout carries PASS/FAIL lines only, which are what may be shared from a run on the match.
"""
import contextlib
import hashlib
import os
import sys
import traceback
import warnings

import numpy as np

from .schema import FormatError, SchemaError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REPORT_NOTICE = (
    "FOR THE OPERATOR'S EYES ONLY. Every figure in this file summarises the competition match. Do not",
    "paste this file, or any line of it, into Claude or any other cloud AI, and do not commit it. What",
    "may go back to Claude is the PASS/FAIL output of the script, or a general code request without",
    "figures; the last section lists the requests that fit.",
)
MISSING = "-"


class OperatorError(ValueError):
    """A precondition of an operator script. The message names a rule only."""


SAFE_ERRORS = (FormatError, SchemaError, OperatorError)


def inside(path, root=ROOT):
    path, root = os.path.realpath(path), os.path.realpath(root)
    return path == root or path.startswith(root + os.sep)


@contextlib.contextmanager
def silenced(enabled=True):
    """Send everything written to stdout and stderr, Python's or a library's, to /dev/null."""
    if not enabled:
        yield
        return
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    null = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null, 1)
        os.dup2(null, 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with np.errstate(all="ignore"):
                yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        for fd in (null,) + saved:
            os.close(fd)


def prepare_report(report, bundle, inputs=()):
    """Refuse a report that would land inside the repository for a bundle outside it, or overwrite an input."""
    if not inside(bundle) and inside(report):
        raise OperatorError("the bundle is outside the repository, so the report must be too")
    target = os.path.realpath(report)
    if any(os.path.realpath(path) == target for path in inputs):
        raise OperatorError("the report would overwrite an input")
    if os.path.isdir(report):
        raise OperatorError("the report path is a folder")
    os.makedirs(os.path.dirname(os.path.abspath(report)), exist_ok=True)


def write_report(path, text):
    """Write the whole file or nothing: a failed run leaves any earlier report as it was."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(tmp, path)


def fingerprint(*paths):
    """First 12 hex digits of the SHA-256 over source files, to tie a report to the code that wrote it."""
    digest = hashlib.sha256()
    for path in paths:
        with open(path, "rb") as handle:
            digest.update(handle.read())
    return digest.hexdigest()[:12]


def run(job, verbose=False):
    """Run job(on_stage), which returns the stdout lines, with output silenced. Returns the exit code.

    A failure prints one line naming the stage; --verbose shows the traceback and library output,
    for the operator's own eyes only.
    """
    stage = ["start"]

    def on_stage(name):
        stage[0] = name

    try:
        with silenced(not verbose):
            lines = job(on_stage)
    except Exception as error:
        if verbose:
            traceback.print_exc()
        detail = str(error) if isinstance(error, SAFE_ERRORS) else type(error).__name__
        print("FAIL {}: {}".format(stage[0], detail))
        return 1
    for line in lines:
        print(line)
    return 1 if any(line.startswith("FAIL") for line in lines) else 0


# ---------------------------------------------------------------- report text

def fmt(value, spec="{:.2f}"):
    """A number, or MISSING for None and NaN."""
    if value is None:
        return MISSING
    try:
        if not np.isfinite(value):
            return MISSING
    except TypeError:
        return str(value)
    return spec.format(value)


def pct(value, digits=1):
    return fmt(None if value is None else 100.0 * value, "{:." + str(digits) + "f}%")


class Report:
    """Plain text, fixed-width tables. Nothing in it may be a frame index or a row of the data."""

    def __init__(self, title, bundle, code):
        self.lines = [title, "=" * len(title), ""]
        self.lines += list(REPORT_NOTICE)
        self.lines += ["", "bundle  {}".format(os.path.abspath(bundle)), "code    {}".format(code)]

    def section(self, title):
        self.lines += ["", title, "-" * len(title)]

    def text(self, *lines):
        self.lines += list(lines)

    def table(self, header, rows):
        cells = [[str(c) for c in header]] + [[str(c) for c in row] for row in rows]
        widths = [max(len(row[i]) for row in cells) for i in range(len(header))]
        for row in cells:
            self.lines.append("  " + "  ".join(
                cell.ljust(widths[i]) if i == 0 else cell.rjust(widths[i]) for i, cell in enumerate(row)).rstrip())

    def render(self):
        return "\n".join(self.lines).rstrip() + "\n"
