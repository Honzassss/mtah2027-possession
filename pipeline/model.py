"""LightGBM heads on the 5 Hz features. CPU, deterministic, the same settings everywhere.

    targets = step_targets(step_codes, steps)           # training only: needs labels
    booster = fit(HEADS["team"], X, Xs, targets)        # both team views, flipped labels for the swap
    p_home = predict(booster, HEADS["team"], X, Xs)     # averaged over both views

Heads
    dead     dead ball against live play, on every step
    team     home in possession, on live steps
    restart  home has the ball when play resumes, on dead steps that have a restart in the half

Change heads (team change and stoppage edge), per-step hazards for the decoder's transitions
    switch   play stays live and the side in possession changes at this step, given live play before it
    stop     play goes dead at this step, given live play before it
    resume   play resumes at this step, given a dead ball before it

Steps within CHANGE_S of a label change weigh CHANGE_WEIGHT; the change heads are not
weighted, so their probabilities stay hazards. Each head is trained on the features as they are and on
the swapped features with the side flipped, and predicts the average of both views, so nothing it learns
can favour the home side.
"""
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np

from .decode import DH, states_from_labels

CHANGE_S = 1.0
CHANGE_WEIGHT = 3.0
PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 400,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "lambda_l2": 10.0,
    "max_bin": 255,
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": 8,
    "seed": 20261107,
    "verbose": -1,
}
MAX_ROUNDS = 3000
EARLY_STOP = 100


@dataclass(frozen=True)
class Head:
    name: str
    flips: bool     # True when the swap exchanges the classes


HEADS = {"dead": Head("dead", False), "team": Head("team", True), "restart": Head("restart", True)}
CHANGE_HEADS = {"switch": Head("switch", False), "stop": Head("stop", False), "resume": Head("resume", False)}


def label_codes(labels):
    """H, A, D (and X) to 0, 1, 2 (and 3)."""
    labels = np.asarray(labels)
    out = np.full(len(labels), 3, dtype=np.int8)
    for code, name in enumerate(("H", "A", "D")):
        out[labels == name] = code
    return out


def step_targets(codes, steps):
    """Per step: y and row mask for each head, and the change weights. codes are the step labels."""
    codes = np.asarray(codes, dtype=np.int8)
    half = steps.half
    states = states_from_labels(codes, half)
    change = np.zeros(len(codes), dtype=bool)
    has_next = np.zeros(len(codes), dtype=bool)
    reach = max(1, int(round(CHANGE_S * steps.fps / steps.size)))
    for a, b in steps.bounds():
        c = codes[a:b]
        edge = np.flatnonzero(c[1:] != c[:-1]) + 1          # first step of each new run
        near = np.zeros(b - a, dtype=bool)
        for lo in range(-reach, reach):
            idx = edge + lo
            near[idx[(idx >= 0) & (idx < b - a)]] = True
        change[a:b] = near
        live = c != 2
        t = np.arange(b - a)
        nxt = np.minimum.accumulate(np.where(live, t, b - a)[::-1])[::-1]
        has_next[a:b] = ~live & (nxt < b - a)
    weight = np.where(change, CHANGE_WEIGHT, 1.0).astype(np.float32)
    ones = np.ones(len(codes), dtype=np.float32)
    live = codes != 2
    prev = np.full(len(codes), -1, dtype=np.int8)        # the step before, inside the half
    for a, b in steps.bounds():
        prev[a + 1:b] = codes[a:b - 1]
    labelled = (codes < 3) & (prev >= 0) & (prev < 3)
    was_live = labelled & (prev != 2)
    was_dead = labelled & (prev == 2)
    return {
        "dead": (codes == 2, codes < 3, weight),
        "team": (codes == 0, live & (codes < 2), weight),
        "restart": (states == DH, has_next, ones),
        "switch": (was_live & live & (codes != prev), was_live, ones),
        "stop": (was_live & ~live, was_live, ones),
        "resume": (was_dead & live, was_dead, ones),
        "states": states,
    }


def _views(head, X, Xs, y, rows, weight):
    y = y[rows].astype(np.float32)
    ys = 1.0 - y if head.flips else y
    return (np.vstack((X[rows], Xs[rows])), np.concatenate((y, ys)),
            np.concatenate((weight[rows], weight[rows])))


def fit(head, parts, names, rounds=None, valid=None):
    """Train one head. parts: [(X, Xs, targets), ...]. valid: the same for early stopping.
    Returns the booster; with `valid`, booster.best_iteration is the early-stopping round."""
    def stack(items):
        xs, ys, ws = [], [], []
        for X, Xs, t in items:
            y, rows, w = t[head.name]
            a, b, c = _views(head, X, Xs, y, rows, w)
            xs.append(a)
            ys.append(b)
            ws.append(c)
        return np.vstack(xs), np.concatenate(ys), np.concatenate(ws)

    Xt, yt, wt = stack(parts)
    train = lgb.Dataset(Xt, label=yt, weight=wt, feature_name=list(names), free_raw_data=True)
    if valid is None:
        return lgb.train(PARAMS, train, num_boost_round=int(rounds or MAX_ROUNDS))
    Xv, yv, wv = stack(valid)
    check = lgb.Dataset(Xv, label=yv, weight=wv, reference=train, free_raw_data=True)
    return lgb.train(PARAMS, train, num_boost_round=MAX_ROUNDS, valid_sets=[check],
                     callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])


def predict(booster, head, X, Xs, rounds=None):
    """Probability of the head's positive class, averaged over both team views."""
    kw = {"num_iteration": rounds} if rounds else {}
    p, ps = booster.predict(X, **kw), booster.predict(Xs, **kw)
    return 0.5 * (p + (1.0 - ps if head.flips else ps))
