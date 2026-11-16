"""The 4-state decoder at 5 Hz, in whole milli-nats so every machine decodes the same path.

    table = transitions(state_sequences)                 # from labelled open data, once
    scores = emissions(p_dead, p_home, p_restart, table, alpha)
    path = viterbi(scores, table.matrix, dead_only)      # one half at a time

States
    H, A    live play, home or away in possession
    DH, DA  dead ball, and the side that has the ball next is home or away. A dead spell can only end
            in play for the side it names, so the restart model decides who comes out of a stoppage.
            Labels never step between DH and DA; restarter_may_change() allows it at the cost of staying
            dead, so the decoder can change its mind about the restarter without a false live step.

Emissions are hybrid-HMM scaled likelihoods: alpha * (log posterior - log prior). The posteriors come
from three binary models (dead, home given live, home restarts given dead). Neighbouring 5 Hz steps
share most of their windows, so their evidence is far from independent; alpha < 1 discounts it.

Transitions are counted on 5 Hz label sequences and made symmetric between home and away, so the
decoder carries no home-side prior either. With the change heads (pipeline/model.py) they vary per step
instead: varying_matrix() turns the hazards of a switch, a stop and a resume at each step into that
step's transition matrix, and only which side restarts after a stop still comes from the counts.
Everything after the logs is integer arithmetic, so the path does not depend on floating-point summation
order, and ties go to the lower state number.
"""
from dataclasses import dataclass, replace

import numpy as np

H, A, DH, DA = 0, 1, 2, 3
STATES = ("H", "A", "DH", "DA")
SCALE = 1000                    # milli-nats
NEVER = -(10 ** 12)
FLOOR = -(10 ** 15)             # scores are clamped here, far below anything reachable
PROB_EPS = 1e-4
HAZARD_EPS = 1e-4
HAZARD_MAX = 0.45               # per change, so staying keeps at least a tenth of the mass
ALLOWED =np.array([[1, 1, 1, 1],      # from H
                    [1, 1, 1, 1],      # from A
                    [1, 0, 1, 0],      # from DH: stay, or play resumes for home
                    [0, 1, 0, 1]],     # from DA
                   dtype=bool)
SWAP = np.array([A, H, DA, DH])


@dataclass(frozen=True)
class Transitions:
    matrix: np.ndarray      # (4, 4) int64 milli-nats, from row to column, NEVER where not allowed
    log_prior: np.ndarray   # (4,) float64, log of each state's share of steps
    counts: np.ndarray      # (4, 4) int64, symmetric counts they came from


def states_from_labels(codes, half):
    """5 Hz states from H/A/D step codes (0, 1, 2). A dead step takes the side of the next live step in
    its half; a dead spell that ends the half takes the side of the live step before it."""
    codes = np.asarray(codes)
    out = np.full(len(codes), -1, dtype=np.int8)
    for h in np.unique(half):
        idx = np.flatnonzero(half == h)
        c = codes[idx]
        live = c != 2
        t = np.arange(len(c))
        nxt = np.minimum.accumulate(np.where(live, t, len(c))[::-1])[::-1]
        prev = np.maximum.accumulate(np.where(live, t, -1))
        side = np.where(nxt < len(c), c[np.minimum(nxt, len(c) - 1)], np.where(prev >= 0, c[np.maximum(prev, 0)], 0))
        out[idx] = np.where(live, c, DH + side)
    return out


def transitions(sequences, pseudo=1.0):
    """Transitions from a list of (states, half) pairs, symmetric between the sides."""
    counts = np.zeros((4, 4), dtype=np.int64)
    occupancy = np.zeros(4, dtype=np.int64)
    for states, half in sequences:
        for h in np.unique(half):
            s = states[half == h]
            np.add.at(counts, (s[:-1], s[1:]), 1)
            occupancy += np.bincount(s, minlength=4)
    counts = counts + counts[np.ix_(SWAP, SWAP)]
    occupancy = occupancy + occupancy[SWAP]
    if (counts[~ALLOWED] > 0).any():
        raise ValueError("label sequences hold a transition the decoder forbids")
    rate = np.where(ALLOWED, counts + pseudo, 0.0)
    rate = rate / rate.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore"):
        matrix = np.where(ALLOWED, np.round(np.log(rate) * SCALE), NEVER).astype(np.int64)
    log_prior = np.log(occupancy / occupancy.sum())
    return Transitions(matrix=matrix, log_prior=log_prior, counts=counts)


def restarter_may_change(table):
    """The table with DH -> DA and DA -> DH at the cost of staying dead. A dead spell takes the side of
    the play after it, so counted labels never hold that step; without it the only way to change the
    restarter during a stoppage is through a live step, which the metric counts as a spell of its own."""
    matrix = table.matrix.copy()
    matrix[DH, DA] = matrix[DH, DH]
    matrix[DA, DH] = matrix[DA, DA]
    return replace(table, matrix=matrix)


def emissions(p_dead, p_home, p_restart, table, alpha):
    """(K, 4) int64 milli-nats from the three posteriors, each (K,)."""
    pd = np.clip(np.asarray(p_dead, dtype=np.float64), PROB_EPS, 1 - PROB_EPS)
    ph = np.clip(np.asarray(p_home, dtype=np.float64), PROB_EPS, 1 - PROB_EPS)
    pr = np.clip(np.asarray(p_restart, dtype=np.float64), PROB_EPS, 1 - PROB_EPS)
    logs = np.column_stack((np.log1p(-pd) + np.log(ph), np.log1p(-pd) + np.log1p(-ph),
                            np.log(pd) + np.log(pr), np.log(pd) + np.log1p(-pr)))
    return np.round(alpha * (logs - table.log_prior) * SCALE).astype(np.int64)


def varying_matrix(h_switch, h_stop, h_resume, table):
    """(K, 4, 4) int64 milli-nats: row t holds the transitions into step t. The hazards are (K,)
    probabilities of a change at that step given the play before it, from the change heads."""
    hs = np.clip(np.asarray(h_switch, dtype=np.float64), HAZARD_EPS, HAZARD_MAX)
    ht = np.clip(np.asarray(h_stop, dtype=np.float64), HAZARD_EPS, HAZARD_MAX)
    hr = np.clip(np.asarray(h_resume, dtype=np.float64), HAZARD_EPS, HAZARD_MAX)
    counts = table.counts.astype(np.float64)
    same = (counts[H, DH] + 1.0) / (counts[H, DH] + counts[H, DA] + 2.0)   # the side that had it restarts
    k = len(hs)
    logs = np.full((k, 4, 4), np.nan)
    stay_live = np.log1p(-(hs + ht))
    for mine, theirs, dead_mine, dead_theirs in ((H, A, DH, DA), (A, H, DA, DH)):
        logs[:, mine, mine] = stay_live
        logs[:, mine, theirs] = np.log(hs)
        logs[:, mine, dead_mine] = np.log(ht) + np.log(same)
        logs[:, mine, dead_theirs] = np.log(ht) + np.log1p(-same)
        logs[:, dead_mine, dead_mine] = np.log1p(-hr)
        logs[:, dead_mine, mine] = np.log(hr)
    out = np.where(ALLOWED[None], np.round(np.nan_to_num(logs) * SCALE), NEVER).astype(np.int64)
    return out


def viterbi(scores, matrix, dead_only=None):
    """Most likely state path of one half. scores (K, 4) int64; matrix (4, 4) int64, or (K, 4, 4) with
    row t the transitions into step t; dead_only (K,) bool marks published dead steps, where the live
    states are not allowed."""
    scores = np.asarray(scores, dtype=np.int64).copy()
    if dead_only is not None:
        scores[np.asarray(dead_only, dtype=bool), :2] = NEVER
    k = len(scores)
    matrix = np.asarray(matrix, dtype=np.int64)
    varying = matrix.ndim == 3
    if varying and len(matrix) != k:
        raise ValueError("one transition matrix per step")
    back = np.zeros((k, 4), dtype=np.int8)
    best = scores[0].copy()
    cols = np.arange(4)
    for t in range(1, k):
        cand = best[:, None] + (matrix[t] if varying else matrix)
        pick = np.argmax(cand, axis=0)
        back[t] = pick
        best = np.maximum(cand[pick, cols] + scores[t], FLOOR)
    path = np.empty(k, dtype=np.int8)
    path[-1] = int(np.argmax(best))
    for t in range(k - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(p_dead, p_home, p_restart, steps, table, alpha, dead_only=None, hazards=None):
    """(K,) states for every step, each half decoded on its own. hazards: (switch, stop, resume), each
    (K,), for per-step transitions; without them the counted transitions hold at every step."""
    scores = emissions(p_dead, p_home, p_restart, table, alpha)
    matrix = table.matrix if hazards is None else varying_matrix(*hazards, table)
    out = np.empty(len(scores), dtype=np.int8)
    for a, b in steps.bounds():
        out[a:b] = viterbi(scores[a:b], matrix if hazards is None else matrix[a:b],
                           None if dead_only is None else dead_only[a:b])
    return out


def labels_of(states):
    """H, A or D for each state."""
    return np.array(["H", "A", "D", "D"])[states]
