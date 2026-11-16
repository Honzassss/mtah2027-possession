"""Model v2: v0's recipe on the labels of boundary convention T = 1 s (test 1), as a bundle. Its
heads are the published ones: v3 (analysis/model_v3.py) decodes them with a restarter that may change.

    python -I analysis/model_v2.py bundle    # T1 heads -> results/models/v2_*, and v2_decoder.json beside them
    python -I analysis/model_v2.py check     # load_model + step_states on the 54 held-out copies = test 1's T1 rows

The heads are the ones `analysis/test1_convention.py final` trained on the T = 1 labels of the 15
training copies (rounds from its folds: 508 / 291 / 144), alpha is the one its sweep chose on the T = 1
folds (1.4), and the transition table is counted on the T = 1 labels of the same training copies.
Decoding is v0's: the counted table at every step, no change-head hazards (pipeline/predict.py).
`v1_*` is the failed change-heads candidate, hence v2.

T = 1 is the convention chosen for the submission: models/ holds a copy of results/models/v2_*, which
tests/check_predict.py checks. This script writes results/models/v2_*
and nothing else; v0's files and results/cache/v0 are only read. IDSSE only.
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis.model_v0 import MODELS, cache_path, eval_keys, key_name, train_keys  # noqa: E402
from analysis.test1_convention import OUT as TEST1, conventions_of, cross, table_for, tag  # noqa: E402
from analysis.test2_dead_shift import OWN_CACHE, light  # noqa: E402
from pipeline.model import HEADS  # noqa: E402
from pipeline.predict import BUNDLE, HEAD_FILE, load_model, sha256_file, step_states, write_bundle  # noqa: E402

T = 1.0
TAG = "v2"
SOURCE = os.path.join(OWN_CACHE, "models")
EXPECT = {"alpha": 1.4, "rounds": {"dead": 508, "team": 291, "restart": 144}}


def test1():
    with open(TEST1) as fh:
        return json.load(fh)


def feature_names():
    with np.load(cache_path(train_keys()[0]), allow_pickle=False) as d:
        return [str(v) for v in d["names"]]


def training_table():
    train = [light(k) for k in train_keys()]
    for c in train:
        c["conv"] = conventions_of(c)
    return table_for(train, T)


def run_bundle(args):
    data = test1()
    alpha, rounds = float(data["heldout"]["alpha"][tag(T)]), data["heldout"]["rounds"][tag(T)]
    if alpha != EXPECT["alpha"] or rounds != EXPECT["rounds"]:
        raise SystemExit("test 1 chose alpha {} and rounds {}, not the scored T1 model".format(alpha, rounds))
    for name in HEADS:
        src = os.path.join(SOURCE, "{}_{}.txt".format(tag(T), name))
        dst = os.path.join(MODELS, HEAD_FILE.format(TAG, name))
        if os.path.isfile(dst):
            if sha256_file(dst) != sha256_file(src):
                raise SystemExit("{} exists and is another file than {}".format(dst, src))
            continue
        shutil.copyfile(src, dst + ".tmp")
        os.replace(dst + ".tmp", dst)
    path = write_bundle(MODELS, TAG, alpha, rounds, training_table(), feature_names())
    print("wrote", path, flush=True)
    for name in HEADS:
        print(" ", HEAD_FILE.format(TAG, name), sha256_file(os.path.join(MODELS, HEAD_FILE.format(TAG, name))))
    print(" ", BUNDLE.format(TAG), sha256_file(path), flush=True)


def run_check(args):
    """The bundle through the deployed path gives test 1's held-out T1 numbers back exactly, against the
    labels of every convention."""
    model = load_model(MODELS, TAG)
    table = training_table()
    if (model.table.matrix.tolist() != table.matrix.tolist() or model.table.log_prior.tolist() != table.log_prior.tolist()
            or model.table.counts.tolist() != table.counts.tolist()):
        raise SystemExit("the bundle's table is not the one counted on the T1 training labels")
    if model.alpha != EXPECT["alpha"]:
        raise SystemExit("the bundle's alpha is not 1.4")
    want = test1()["heldout"]["rows"]
    bad = []
    for key in eval_keys():
        with np.load(cache_path(key), allow_pickle=False) as d:
            X, Xs = d["X"], d["Xs"]
        c = light(key)
        c["conv"] = conventions_of(c)
        got = cross(c, step_states(model, X, Xs, c["steps"], c["dead"]), c["conv"])
        same = got == want[key_name(key)][tag(T)]
        bad += [] if same else [key_name(key)]
        print("{:<32} {}  {}".format(key_name(key), "same" if same else "DIFFERENT",
                                     " ".join("{} {:.4f}".format(t, v["spell_f1"]) for t, v in got.items())), flush=True)
    if bad:
        raise SystemExit("v2 differs from test 1's T1 model on " + ", ".join(bad))
    print("v2 gives test 1's T1 held-out rows back exactly on all {} copies (every convention)".format(len(eval_keys())))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=("bundle", "check"))
    args = parser.parse_args()
    {"bundle": run_bundle, "check": run_check}[args.stage](args)


if __name__ == "__main__":
    main()
