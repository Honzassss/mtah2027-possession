"""Self-checks for the 5 Hz features, the model heads' targets and the decoder.

    python -I tests/check_features.py [--idsse external/processed/idsse] [--match J03WOH] [--twin twin/J03WMX]
        [--json results.json]

Open data and the twin only: the IDSSE copy goes through degrade(), which refuses unlabelled data, and
the twin is the synthetic stand-in. Exit code 1 if a check fails. Messages name fields and counts only.
"""
import argparse
import dataclasses
import hashlib
import itertools
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline.clean import clean  # noqa: E402
from pipeline.decode import (ALLOWED, DA, DH, NEVER, A, H, emissions, states_from_labels,  # noqa: E402
                             transitions, varying_matrix, viterbi)
from pipeline.degrade import SIMULATORS, degrade  # noqa: E402
from pipeline.epts import load_competition  # noqa: E402
from pipeline.features import dead_steps, features, step_index  # noqa: E402
from pipeline.idsse import load_idsse  # noqa: E402
from pipeline.model import label_codes, step_targets  # noqa: E402

results = []

# How each feature maps under features(..., swap=True): the partner column and the sign.
INVARIANT_PREFIX = ("ball_seen", "ball_y_raw", "ball_speed", "speed_all", "since_seen", "until_seen", "gap_len",
                    "ball_out_", "last_out_", "next_out_", "last_speed", "next_speed", "seen_", "cx_diff")
NEGATED_PREFIX = ("ball_xa", "ball_ya", "ball_vxa", "last_xa", "next_xa", "last_ya", "next_ya", "last_vxa",
                  "next_vxa", "guess_xa", "cx_all", "ball_dd", "ball_rd", "last_dd", "next_dd", "last_rd",
                  "next_rd", "guess_dd", "dd_", "rd_", "gdd_", "spread_diff", "since_touch_diff",
                  "until_touch_diff")
SIGNED_POSITION = ("cx_h", "cx_a")


def check(name, ok, detail=""):
    results.append({"name": name, "ok": bool(ok), "detail": str(detail)})
    print("{:4} {:<70} {}".format("PASS" if ok else "FAIL", name, detail), flush=True)


def swap_rule(name, names):
    """(partner, sign) for a feature, or None if it fits no rule."""
    if name.startswith(INVARIANT_PREFIX):
        return name, 1.0
    if name.startswith(NEGATED_PREFIX):
        return name, -1.0
    for mine, theirs in (("_h", "_a"), ("_a", "_h")):
        if name.endswith(mine):
            partner = name[:-2] + theirs
            return (partner, -1.0 if name in SIGNED_POSITION else 1.0) if partner in names else None
        if mine + "_" in name:
            partner = name.replace(mine + "_", theirs + "_")
            return (partner, 1.0) if partner in names else None
    return None


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]


def brute_force(scores, matrix):
    best, best_path = None, None
    k = len(scores)
    for path in itertools.product(range(4), repeat=k):
        total = int(scores[0, path[0]]) + sum(int(matrix[path[t - 1], path[t]]) + int(scores[t, path[t]])
                                              for t in range(1, k))
        if best is None or total > best:
            best, best_path = total, path
    return best, np.array(best_path, dtype=np.int8)


def path_score(path, scores, matrix):
    return int(scores[0, path[0]]) + sum(int(matrix[path[t - 1], path[t]]) + int(scores[t, path[t]])
                                         for t in range(1, len(path)))


def path_score_varying(path, scores, matrix):
    return int(scores[0, path[0]]) + sum(int(matrix[t, path[t - 1], path[t]]) + int(scores[t, path[t]])
                                         for t in range(1, len(path)))


def brute_force_varying(scores, matrix):
    best, best_path = None, None
    for path in itertools.product(range(4), repeat=len(scores)):
        total = path_score_varying(path, scores, matrix)
        if best is None or total > best:
            best, best_path = total, path
    return best, np.array(best_path, dtype=np.int8)


def check_decoder():
    print("--- decoder", flush=True)
    rng = np.random.default_rng(7)
    matrix = np.where(ALLOWED, -rng.integers(0, 4000, size=(4, 4)), NEVER).astype(np.int64)
    worst, mismatches = 0, 0
    for trial in range(60):
        k = int(rng.integers(1, 7))
        scores = rng.integers(-5000, 0, size=(k, 4)).astype(np.int64)
        dead = rng.random(k) < 0.2
        masked = scores.copy()
        masked[dead, :2] = NEVER
        best, _ = brute_force(masked, matrix)
        path = viterbi(scores, matrix, dead)
        got = path_score(path, masked, matrix)
        mismatches += got != best
        worst = max(worst, abs(got - best))
    check("viterbi finds the best path on 60 random problems (brute force)", mismatches == 0, worst)

    states = viterbi(rng.integers(-5000, 0, size=(500, 4)).astype(np.int64), matrix)
    legal = ALLOWED[states[:-1], states[1:]].all()
    check("viterbi never takes a forbidden transition", legal)
    dead = np.zeros(500, dtype=bool)
    dead[100:200] = True
    states = viterbi(rng.integers(-5000, 0, size=(500, 4)).astype(np.int64), matrix, dead)
    check("published dead steps decode as dead", np.isin(states[100:200], (DH, DA)).all())

    codes = np.array([0, 0, 2, 2, 1, 1, 2, 2, 0, 2, 2], dtype=np.int8)
    half = np.ones(len(codes), dtype=np.int8)
    got = states_from_labels(codes, half).tolist()
    want = [H, H, DA, DA, A, A, DH, DH, H, DH, DH]
    check("dead steps take the side of the next live step, or the last one at the end", got == want, got)
    table = transitions([(np.array(want, dtype=np.int8), half)])
    check("transitions are symmetric between the sides",
          np.array_equal(table.matrix, table.matrix[np.ix_([1, 0, 3, 2], [1, 0, 3, 2])]))
    try:
        transitions([(np.array([H, DH, A], dtype=np.int8), np.ones(3, dtype=np.int8))])
        refused = False
    except ValueError:
        refused = True
    check("transitions refuse a dead spell that ends in play for the other side", refused)
    p = np.array([0.2, 0.9])
    e1 = emissions(p, p, p, table, 0.5)
    e2 = emissions(p.copy(), p.copy(), p.copy(), table, 0.5)
    check("emissions are whole milli-nats and repeat exactly", e1.dtype == np.int64 and np.array_equal(e1, e2))

    mismatches = 0
    for trial in range(40):
        k = int(rng.integers(2, 7))
        scores = rng.integers(-5000, 0, size=(k, 4)).astype(np.int64)
        varying = varying_matrix(rng.random(k) * 0.5, rng.random(k) * 0.5, rng.random(k), table)
        best, _ = brute_force_varying(scores, varying)
        mismatches += path_score_varying(viterbi(scores, varying), scores, varying) != best
    check("viterbi with per-step transitions finds the best path (brute force)", mismatches == 0, mismatches)
    hazards = rng.random((3, 50)) * np.array([[0.5], [0.5], [1.0]])
    varying = varying_matrix(*hazards, table)
    mass = np.where(ALLOWED[None], np.exp(varying / 1000.0), 0.0).sum(axis=2)
    check("per-step transitions are distributions over the allowed states", np.abs(mass - 1).max() < 0.01,
          round(float(np.abs(mass - 1).max()), 4))
    check("per-step transitions forbid what the counted ones forbid", (varying[:, ~ALLOWED] == NEVER).all())
    steady = varying_matrix(np.full(3, 0.02), np.full(3, 0.005), np.full(3, 0.008), table)
    check("a likelier switch makes switching cheaper and staying dearer",
          (varying_matrix(np.full(3, 0.2), np.full(3, 0.005), np.full(3, 0.008), table)[:, H, A] > steady[:, H, A]).all()
          and (varying_matrix(np.full(3, 0.2), np.full(3, 0.005), np.full(3, 0.008), table)[:, H, H] < steady[:, H, H]).all())


def check_idsse(path):
    print("--- features on a degraded IDSSE copy", flush=True)
    match = load_idsse(path)
    copy = degrade(match, 1, SIMULATORS["camera"])
    cleaned = clean(copy)
    steps = step_index(cleaned)

    sizes = steps.stop - steps.start
    covered = np.zeros(cleaned.n_frames, dtype=np.int32)
    np.add.at(covered, np.concatenate([np.arange(a, b) for a, b in zip(steps.start, steps.stop)]), 1)
    check("steps tile every frame of both halves exactly once",
          np.array_equal(covered, (cleaned.period > 0).astype(np.int32)))
    check("steps are 5 frames at 25 fps, the last of a half may be shorter",
          steps.size == 5 and (sizes <= 5).all() and (sizes == 5).mean() > 0.999)
    check("step centres lie inside their step and their half",
          ((steps.centre >= steps.start) & (steps.centre < steps.stop)).all()
          and np.array_equal(cleaned.period[steps.centre], steps.half))

    started = time.time()
    names, table = features(cleaned, steps)
    seconds = time.time() - started
    _, again = features(cleaned, steps)
    check("features repeat byte for byte", digest(table) == digest(again), "{:.1f}s per view".format(seconds))
    check("features are float32 (K, n), no infinities", table.dtype == np.float32 and table.shape == (steps.n_steps, len(names))
          and not np.isinf(table).any(), table.shape)
    check("feature names are unique", len(set(names)) == len(names), len(names))

    stripped = dataclasses.replace(cleaned, extras={}, view_polygon=None, labels=None, ball_z=None)
    _, bare = features(stripped, steps)
    check("features never read extras, view_polygon, labels or ball_z", digest(bare) == digest(table))
    poked = dataclasses.replace(cleaned, extras={"truth_ball_xy": np.zeros_like(cleaned.ball_xy)},
                                view_polygon=np.zeros_like(cleaned.view_polygon),
                                labels=np.full(cleaned.n_frames, "H"), ball_z=np.zeros(cleaned.n_frames, np.float32))
    _, poked_table = features(poked, steps)
    check("features ignore every value written into those fields", digest(poked_table) == digest(table))

    _, swapped = features(cleaned, steps, swap=True)
    unclassified, broken = [], []
    for j, name in enumerate(names):
        rule = swap_rule(name, names)
        if rule is None:
            unclassified.append(name)
            continue
        partner, sign = rule
        a, b = swapped[:, j], sign * table[:, names.index(partner)]
        if not (np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(a[~np.isnan(a)], b[~np.isnan(b)], atol=1e-3)):
            broken.append(name)
    check("every feature has a swap rule", not unclassified, unclassified[:5])
    check("swap=True exchanges home and away and turns the pitch round", not broken, broken[:5])
    _, twice = features(dataclasses.replace(cleaned, players_team=np.where(cleaned.players_team >= 0, 1 - cleaned.players_team, -1).astype(np.int8),
                                            keeper_slot=cleaned.keeper_slot[:, ::-1].copy(),
                                            attack_sign=(-cleaned.attack_sign).astype(np.int8)), steps, swap=True)
    check("swapping twice gives the features back", digest(twice) == digest(table))

    nan_share = np.isnan(table).mean(axis=0)
    check("no feature is missing on more than half the steps", nan_share.max() <= 0.5, round(float(nan_share.max()), 3))
    seen = names.index("ball_seen")
    check("ball features are missing exactly where no ball was seen in the step",
          np.array_equal(np.isnan(table[:, names.index("ball_xa")]), table[:, seen] == 0))

    codes = label_codes(cleaned.labels)
    targets = step_targets(codes[steps.centre], steps)
    y, rows, weight = targets["team"]
    check("team rows are the live steps", np.array_equal(rows, np.isin(codes[steps.centre], (0, 1))))
    check("about a fifth to a half of the steps sit within 1 s of a change",
          0.15 < float((weight > 1).mean()) < 0.5, round(float((weight > 1).mean()), 3))
    y, rows, _ = targets["restart"]
    check("restart rows are dead steps with a restart later in the half",
          rows.sum() > 0 and (codes[steps.centre][rows] == 2).all())
    step_codes = codes[steps.centre]
    prev = np.concatenate(([3], step_codes[:-1]))
    prev[[a for a, _ in steps.bounds()]] = 3
    counts = {name: int(targets[name][0][targets[name][1]].sum()) for name in ("switch", "stop", "resume")}
    want = {"switch": int(((prev < 2) & (step_codes < 2) & (step_codes != prev)).sum()),
            "stop": int(((prev < 2) & (step_codes == 2)).sum()),
            "resume": int(((prev == 2) & (step_codes < 2)).sum())}
    check("change targets count every switch, stop and resume between steps of a half", counts == want, counts)
    check("change rows follow live play (switch, stop) or a dead ball (resume)",
          np.array_equal(targets["switch"][1], prev < 2) and np.array_equal(targets["resume"][1], prev == 2))
    expanded = steps.frame_labels(codes[steps.centre], cleaned.n_frames, np.int8(3))
    inside = cleaned.period > 0
    check("step labels spread back over 25 Hz agree with the labels on 95% of frames",
          float((expanded[inside] == codes[inside]).mean()) > 0.95, round(float((expanded[inside] == codes[inside]).mean()), 4))
    return {"seconds_per_view": seconds, "features": len(names), "steps": int(steps.n_steps)}


def check_twin(twin):
    print("--- features on the twin (competition format)", flush=True)
    match = load_competition(os.path.join(twin, "data", "tactical"))
    cleaned = clean(match)
    steps = step_index(cleaned)
    names, table = features(cleaned, steps)
    _, swapped = features(cleaned, steps, swap=True)
    check("twin: steps cover both published halves", int((steps.stop - steps.start).sum()) == int((match.period > 0).sum()))
    check("twin: features have the same columns as on IDSSE", table.shape[1] == len(names) and swapped.shape == table.shape)
    check("twin: no infinities", not np.isinf(table).any())
    dead = dead_steps(cleaned, steps)
    check("twin: published dead block becomes dead steps", dead.sum() > 0 and dead.sum() <= int(match.forced_dead.sum() / 5) + 2,
          int(dead.sum()))
    return {"steps": int(steps.n_steps)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--idsse", default=os.path.join(ROOT, "external", "processed", "idsse"))
    parser.add_argument("--match", default="J03WOH")
    parser.add_argument("--twin", default=os.path.join(ROOT, "twin", "J03WMX"))
    parser.add_argument("--json")
    args = parser.parse_args()
    started = time.time()
    measurements = {}
    check_decoder()
    measurements["idsse"] = check_idsse(os.path.join(args.idsse, args.match + ".npz"))
    if os.path.isdir(args.twin):
        measurements["twin"] = check_twin(args.twin)
    failed = [r for r in results if not r["ok"]]
    print("{} checks, {} failed, {:.0f}s".format(len(results), len(failed), time.time() - started), flush=True)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump({"checks": results, "measurements": measurements}, handle, indent=1, default=float)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
