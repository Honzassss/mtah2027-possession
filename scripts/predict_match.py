"""Write the submission file from the competition's tactical bundle with a saved model (v3 by default).

    .venv/bin/python -I scripts/predict_match.py <tactical_dir> <out.csv>
    .venv/bin/python -I scripts/predict_match.py twin/J03WMX/data/tactical results/twin/J03WMX_v3.csv

<tactical_dir> is the folder holding BAR-ATH_tactical_atd.xml and BAR-ATH_tactical_atd.txt (the
homography file is not read: no feature uses it). The heads and the decoder come from
models/<tag>_decoder.json and the head files it pins by hash: the published model, tracked in git.
analysis/model_v0.py final, analysis/model_v2.py bundle and analysis/model_v3.py bundle make fresh sets
in results/models; --models points elsewhere. --tag v2 runs v3's heads with the decoder as counted,
--tag v0 the same recipe on the labels of convention T = 0.

The file is written to <out.csv>.tmp, read back from disk and checked (pipeline/submission.py: the
SUBMISSION.md rules, no X on a graded frame, D on the published dead block); only a file that passes
becomes <out.csv>, with <out.csv>.sha256 beside it (`shasum -a 256 -c <out.csv>.sha256`). A failed run
leaves any earlier <out.csv> as it was.

It prints three lines and nothing else, so they are what may be shared from a run on the match:

    PASS                    or   FAIL <stage>: <reason>
    rows 199319
    spells <n>              our own H/A spells inside the two halves

Library output and warnings are silenced while it works, and a failure names the stage and the error
type only, plus the message for this repository's own errors, which never quote the data. --verbose
shows everything, with the traceback: for the operator's own eyes, never for sharing from a run on the
match.

When the bundle lies outside this repository, <out.csv> must lie outside it too, so nothing made from
the match lands in the working tree.
"""
import argparse
import contextlib
import os
import sys
import traceback
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline import submission  # noqa: E402
from pipeline.epts import COMPETITION_FRAMES, load_competition  # noqa: E402
from pipeline.predict import ModelError, frame_labels, load_model  # noqa: E402
from pipeline.schema import FormatError, SchemaError  # noqa: E402


class OperatorError(ValueError):
    """A precondition of this script. The message names a rule only."""


SAFE_ERRORS = (FormatError, SchemaError, ModelError, OperatorError)


def inside(path, root):
    path, root = os.path.realpath(path), os.path.realpath(root)
    return path == root or path.startswith(root + os.sep)


@contextlib.contextmanager
def silenced(enabled):
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


def run(tactical_dir, out, models, tag, on_stage):
    """(reason, rows, spells): reason None when the file passed and is in place."""
    on_stage("output")
    if not inside(tactical_dir, ROOT) and inside(out, ROOT):
        raise OperatorError("the bundle is outside the repository, so the output must be too")
    folder = os.path.dirname(os.path.abspath(out))
    os.makedirs(folder, exist_ok=True)
    on_stage("model")
    model = load_model(models, tag)
    on_stage("load")
    match = load_competition(tactical_dir, with_homography=False)
    if not np.array_equal(match.frame, np.arange(COMPETITION_FRAMES)):
        raise OperatorError("frame numbers are not 0..{} in order".format(COMPETITION_FRAMES - 1))
    labels = frame_labels(match, model, on_stage)
    on_stage("write")
    tmp = out + ".tmp"
    digest = submission.write(tmp, match.frame, labels)
    on_stage("check")
    reason, codes = submission.check(tmp)
    if reason is not None:
        os.remove(tmp)
        return reason, None, None
    os.replace(tmp, out)
    with open(out + ".sha256", "w") as fh:
        fh.write("{}  {}\n".format(digest, os.path.basename(out)))
    return None, len(codes), submission.spell_count(codes)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tactical_dir", help="folder holding BAR-ATH_tactical_atd.xml and _atd.txt")
    parser.add_argument("out", help="submission CSV to write")
    parser.add_argument("--models", default=os.path.join(ROOT, "models"),
                        help="folder with <tag>_decoder.json and its heads (default: the published models/)")
    parser.add_argument("--tag", default="v3",
                        help="model in --models (default v3: v2's heads, trained on the labels of convention "
                             "T = 1, with a restarter that may change during a stoppage)")
    parser.add_argument("--verbose", action="store_true",
                        help="library output and tracebacks; never share it from a run on the match")
    args = parser.parse_args()

    stage = ["start"]

    def on_stage(name):
        stage[0] = name

    try:
        with silenced(not args.verbose):
            reason, rows, spells = run(args.tactical_dir, args.out, args.models, args.tag, on_stage)
    except Exception as error:
        if args.verbose:
            traceback.print_exc()
        detail = str(error) if isinstance(error, SAFE_ERRORS) else type(error).__name__
        print("FAIL {}: {}".format(stage[0], detail))
        return 1
    if reason is not None:
        print("FAIL check: {}".format(reason))
        return 1
    print("PASS")
    print("rows {}".format(rows))
    print("spells {}".format(spells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
