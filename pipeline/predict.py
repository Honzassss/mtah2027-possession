"""A trained model from its saved files onto any Match: the path the submission file comes from.

    model = load_model("models", "v3")                # heads, decoder table and alpha; file hashes checked
    labels = frame_labels(match, model)                # (F,) "<U1": H, A or D inside the halves, X outside

The order is the one analysis/model_v0.py scored on IDSSE: clean, 5 Hz steps, features in both team
views, the three heads averaged over both views, the integer Viterbi with published dead blocks kept
dead, then the step's label on each of its frames. analysis/model_v0.py writes the bundle file beside
the heads (write_bundle) when it trains them, and analysis/model_v2.py for v2; the bundle pins each
head file by its SHA-256, so a head from another run cannot slip in.

Nothing here prints, and errors name files and counts only, never a value from the data.
"""
import hashlib
import json
import os
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np

from .clean import clean
from .decode import Transitions, decode, labels_of
from .features import dead_steps, features, step_index
from .model import HEADS, predict

BUNDLE = "{}_decoder.json"
HEAD_FILE = "{}_{}.txt"


class ModelError(ValueError):
    """The saved model is incomplete or does not match its bundle. The message names files only."""


@dataclass(frozen=True)
class Model:
    tag: str
    boosters: dict          # head name -> lgb.Booster
    table: Transitions
    alpha: float
    feature_names: tuple


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_bundle(model_dir, tag, alpha, rounds, table, names, heads_of=None):
    """Record alpha, the decoder table, the feature names and each head file's hash beside the heads.
    heads_of: the tag whose head files the bundle uses, when they are another bundle's (v3 uses v2's)."""
    heads = {}
    for name in HEADS:
        path = os.path.join(model_dir, HEAD_FILE.format(heads_of or tag, name))
        heads[name] = {"file": os.path.basename(path), "sha256": sha256_file(path), "rounds": int(rounds[name])}
    out = {"tag": tag, "alpha": float(alpha), "heads": heads,
           "transitions": table.matrix.tolist(), "log_prior": table.log_prior.tolist(),
           "counts": table.counts.tolist(), "feature_names": list(names)}
    path = os.path.join(model_dir, BUNDLE.format(tag))
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=1)
    os.replace(tmp, path)
    return path


def load_model(model_dir, tag="v3"):
    path = os.path.join(model_dir, BUNDLE.format(tag))
    if not os.path.isfile(path):
        raise ModelError("no model bundle: {}".format(path))
    with open(path) as fh:
        bundle = json.load(fh)
    if set(bundle["heads"]) != set(HEADS):
        raise ModelError("the bundle does not name the heads {}".format(sorted(HEADS)))
    names = tuple(bundle["feature_names"])
    boosters = {}
    for name in HEADS:
        entry = bundle["heads"][name]
        head_path = os.path.join(model_dir, entry["file"])
        if not os.path.isfile(head_path):
            raise ModelError("head file missing: {}".format(entry["file"]))
        if sha256_file(head_path) != entry["sha256"]:
            raise ModelError("head file does not match the bundle: {}".format(entry["file"]))
        booster = lgb.Booster(model_file=head_path)
        if booster.num_trees() != entry["rounds"]:
            raise ModelError("head {} has {} trees, the bundle says {}".format(name, booster.num_trees(), entry["rounds"]))
        if tuple(booster.feature_name()) != names:
            raise ModelError("head {} was trained on other features than the bundle names".format(name))
        boosters[name] = booster
    table = Transitions(matrix=np.array(bundle["transitions"], dtype=np.int64),
                        log_prior=np.array(bundle["log_prior"], dtype=np.float64),
                        counts=np.array(bundle["counts"], dtype=np.int64))
    if table.matrix.shape != (4, 4) or table.log_prior.shape != (4,):
        raise ModelError("the bundle's decoder table is not 4 states")
    return Model(tag=tag, boosters=boosters, table=table, alpha=float(bundle["alpha"]), feature_names=names)


def head_probabilities(model, X, Xs):
    return {name: predict(model.boosters[name], HEADS[name], X, Xs) for name in HEADS}


def _quiet(name):
    pass


def step_states(model, X, Xs, steps, dead, on_stage=_quiet):
    """(K,) decoder states (H, A, DH, DA) from the features of both views and the published dead steps."""
    on_stage("heads")
    p = head_probabilities(model, X, Xs)
    on_stage("decode")
    return decode(p["dead"], p["team"], p["restart"], steps, model.table, model.alpha, dead)


def frame_labels(match, model, on_stage=_quiet):
    """(F,) "<U1" labels for every row of the match: H, A or D on each frame of a half, X elsewhere.
    on_stage(name) is called as each stage starts, so a caller can say where a failure happened."""
    on_stage("clean")
    cleaned = clean(match)
    on_stage("features")
    steps = step_index(cleaned)
    names, X = features(cleaned, steps)
    _, Xs = features(cleaned, steps, swap=True)
    if tuple(names) != model.feature_names:
        raise ModelError("the features differ from the ones the heads were trained on")
    states = step_states(model, X, Xs, steps, dead_steps(cleaned, steps), on_stage)
    return steps.frame_labels(labels_of(states), cleaned.n_frames, "X")
