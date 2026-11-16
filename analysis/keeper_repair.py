"""Keeper repair: the most that a better relabel of keepers filed under the other team's slot can gain,
measured through the gate (analysis/gate.py) under both conventions.

    python -I analysis/keeper_repair.py build [--workers 5]   # 15 training + 36 gate copies, keepers on their true team
    python -I analysis/keeper_repair.py gate                  # heads on them under T0 and T1, gated against v0 and v2

The oracle is clean() as it is, except that every fragment whose majority person is a keeper gets that
keeper's true team from the degrader's own record, before the team cap and the keeper slot are worked out.
No rule can relabel keepers better, so if the oracle does not pass the gate (+0.02 in each of the four
cells), under T0 against v0 and under T1 against v2, no keeper rule will. The heads are retrained on the
oracle copies with the rounds and alpha of v0 (T0) and v2 (T1), not tuned again: a bound, not a candidate.

What clean() misses on the degraded copies (seeds 1-3, all seven matches; it puts 0.81 of the misfiled
keeper time back under camera, 0.89 under plain): keepers whose team camps in the other half stand past the
40 m depth cap; at set pieces opponents stand deeper than the keeper; in long stoppages teammates walk
behind the goal line, so the median lead over the own defenders drops under 2 m; and fragments under 5 s are
never tested. J03WPY's 0 at seed 1 is one tracklet: that copy drew a keeper reuse of 0.004.

Result (7 Nov 2026): the oracle puts 0.999-1.000 of the misfiled keeper time on the right team (clean()
0.70-0.80 on the same copies) and changes the features on 1-27 % of the steps, and the gate does not move.
T0 against v0: 0.5551 vs 0.5555; camera J03WMX +0.003, camera J03WN1 -0.005, plain J03WMX +0.000, plain
J03WN1 -0.000. T1 against v2: 0.6239 vs 0.6252; +0.007, +0.003, -0.013, -0.003. The keeper rule stays as
it is, and the spread of those cells is what retraining on slightly different copies does on its own.

`build` first rebuilds one training copy without the oracle and checks that its features equal
results/cache/v0 to the bit, so the keepers are the only difference. IDSSE only, through degrade(), which
refuses unlabelled data. Writes results/cache/keeper and results/gate_keeper_oracle.json only.
"""
import argparse
import contextlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import pipeline.clean as clean_module  # noqa: E402
from analysis.gate import GATE, cells_names, frames_of, gate_frames, gate_keys  # noqa: E402
from analysis.model_v0 import IDSSE, MODELS, cache_path, key_name, load, table_from, train_keys  # noqa: E402
from analysis.model_v2 import EXPECT as V2  # noqa: E402
from analysis.test1_convention import conventions_of, step_codes_of, table_for, tag  # noqa: E402
from analysis.test2_dead_shift import light  # noqa: E402
from pipeline.degrade import SIMULATORS, degrade  # noqa: E402
from pipeline.features import dead_steps, features, step_index  # noqa: E402
from pipeline.idsse import load_idsse  # noqa: E402
from pipeline.model import HEADS, fit, step_targets  # noqa: E402
from pipeline.predict import BUNDLE, load_model, step_states, write_bundle  # noqa: E402

OWN = os.path.join(ROOT, "results", "cache", "keeper")
MODEL_DIR = os.path.join(OWN, "models")
OUT = os.path.join(ROOT, "results", "gate_keeper_oracle.json")
TAGS = {0.0: "k0", 1.0: "k1"}
REFERENCE = {0.0: "v0", 1.0: "v2"}


def keeper_truth(copy):
    """relabel_keepers as clean() runs it, then every keeper fragment on the keeper's true team."""
    person = copy.extras["slot_person"]
    true_team = copy.extras["person_team"]
    is_keeper = copy.extras["person_is_keeper"]
    original = clean_module.relabel_keepers

    def relabel(players_xy, table, slot_team, period, forced_dead, attack_sign, pitch, fps, cfg):
        team = original(players_xy, table, slot_team, period, forced_dead, attack_sign, pitch, fps, cfg)
        for i, (slot, start, end) in enumerate(zip(table["slot"], table["start"], table["end"])):
            who = person[start:end + 1, slot]
            who = who[who >= 0]
            if len(who):
                major = int(np.bincount(who).argmax())
                if is_keeper[major]:
                    team[i] = true_team[major]
        return team
    return relabel


@contextlib.contextmanager
def relabel_with(relabel):
    """clean() looks relabel_keepers up in its module on every call."""
    saved = clean_module.relabel_keepers
    clean_module.relabel_keepers = relabel
    try:
        yield
    finally:
        clean_module.relabel_keepers = saved


def copy_features(key, oracle=True):
    """The steps of analysis/model_v0.build_one, with or without the oracle."""
    match, sim, live, seed = key
    cfg = SIMULATORS[sim] if live is None else replace(SIMULATORS[sim], live_share=live)
    copy = degrade(load_idsse(os.path.join(IDSSE, match + ".npz")), seed, cfg)
    with relabel_with(keeper_truth(copy) if oracle else clean_module.relabel_keepers):
        cleaned = clean_module.clean(copy)
    steps = step_index(cleaned)
    names, X = features(cleaned, steps)
    _, Xs = features(cleaned, steps, swap=True)
    return names, X, Xs, steps, dead_steps(cleaned, steps)


def own_path(key):
    return os.path.join(OWN, key_name(key) + ".npz")


def differ(a, b):
    """True on rows where any feature differs, NaN equal to NaN."""
    return np.any(~((a == b) | (np.isnan(a) & np.isnan(b))), axis=1)


def build_one(key):
    path = own_path(key)
    if os.path.isfile(path):
        return key_name(key), None
    names, X, Xs, steps, dead = copy_features(key)
    with np.load(cache_path(key), allow_pickle=False) as v0:
        same = (all(np.array_equal(getattr(steps, f), v0[f]) for f in ("start", "stop", "centre", "half"))
                and np.array_equal(dead, v0["dead"]) and [str(v) for v in v0["names"]] == names)
        if not same:
            raise RuntimeError(key_name(key) + ": steps, dead steps or feature names differ from results/cache/v0")
        changed = float(np.mean(differ(X, v0["X"]) | differ(Xs, v0["Xs"])))
    tmp = path + ".tmp.npz"
    np.savez(tmp, X=X, Xs=Xs)
    os.replace(tmp, path)
    return key_name(key), changed


def run_build(args):
    os.makedirs(OWN, exist_ok=True)
    probe = train_keys()[0]
    names, X, Xs, _, _ = copy_features(probe, oracle=False)
    with np.load(cache_path(probe), allow_pickle=False) as v0:
        if not (np.array_equal(X, v0["X"], equal_nan=True) and np.array_equal(Xs, v0["Xs"], equal_nan=True)):
            raise SystemExit("without the oracle, {} does not rebuild results/cache/v0's features".format(key_name(probe)))
    print("without the oracle,", key_name(probe), "rebuilds v0's features to the bit", flush=True)
    keys = train_keys() + gate_keys()
    started = time.time()
    changed = {}
    with ProcessPoolExecutor(args.workers) as pool:
        for name, share in pool.map(build_one, keys):
            changed[name] = share
            print("built", name, "cached" if share is None else "steps changed {:.4f}".format(share), flush=True)
    print("{} copies, {:.0f}s".format(len(keys), time.time() - started), flush=True)
    with open(os.path.join(OWN, "build.json"), "w") as fh:
        json.dump({"steps_changed": changed}, fh, indent=1)


def table_matches(table, model_dir, tag_name):
    with open(os.path.join(model_dir, BUNDLE.format(tag_name))) as fh:
        bundle = json.load(fh)
    return (table.matrix.tolist() == bundle["transitions"] and table.log_prior.tolist() == bundle["log_prior"]
            and table.counts.tolist() == bundle["counts"])


def run_gate(args):
    with open(os.path.join(ROOT, "results", "model_v0_cv.json")) as fh:
        v0cv = json.load(fh)
    settings = {0.0: (float(v0cv["alpha"]), v0cv["rounds"]), 1.0: (float(V2["alpha"]), V2["rounds"])}
    train = []
    for key in train_keys():
        c = load(key)
        with np.load(own_path(key), allow_pickle=False) as d:
            c["X"], c["Xs"] = d["X"], d["Xs"]
        c["conv"] = conventions_of(c)
        train.append(c)
    names = train[0]["names"]
    os.makedirs(MODEL_DIR, exist_ok=True)
    for T, (alpha, rounds) in settings.items():
        if T == 0.0:
            table, targets = table_from(train), [c["targets"] for c in train]
        else:
            table = table_for(train, T)
            targets = [step_targets(step_codes_of(c, c["conv"][T]), c["steps"]) for c in train]
        if not table_matches(table, MODELS, REFERENCE[T]):
            raise SystemExit("the {} decoder table differs from {}'s".format(tag(T), REFERENCE[T]))
        for name, head in HEADS.items():
            path = os.path.join(MODEL_DIR, "{}_{}.txt".format(TAGS[T], name))
            if not os.path.isfile(path):
                started = time.time()
                booster = fit(head, [(c["X"], c["Xs"], t) for c, t in zip(train, targets)], names, rounds=rounds[name])
                booster.save_model(path + ".tmp")
                os.replace(path + ".tmp", path)
                print("trained", TAGS[T], name, rounds[name], "{:.0f}s".format(time.time() - started), flush=True)
        write_bundle(MODEL_DIR, TAGS[T], alpha, rounds, table, names)
    del train, targets

    models = {T: load_model(MODEL_DIR, TAGS[T]) for T in settings}
    frames = {T: {} for T in settings}
    for key in gate_keys():
        c = light(key)
        with np.load(own_path(key), allow_pickle=False) as d:
            X, Xs = d["X"], d["Xs"]
        for T, model in models.items():
            frames[T][key] = frames_of(c["steps"], step_states(model, X, Xs, c["steps"], c["dead"]), len(c["codes"]))

    out = {"definition": __doc__.split("\n\n")[2], "gate": GATE, "results": {}}
    for T in settings:
        result = gate_frames(frames[T], T, REFERENCE[T])
        out["results"][tag(T)] = result
        cmp = result["compare"]
        print("truth {}: oracle {:.4f} vs {} {:.4f} ({:+.4f})".format(
            tag(T), result["candidate"]["gate"], REFERENCE[T], result["reference_cells"]["gate"], cmp["gate_gain"]))
        for cell in cells_names():
            print("  {:<16} {:.4f} vs {:.4f}  {:+.4f}".format(
                cell, result["candidate"][cell], result["reference_cells"][cell], cmp["gains"][cell]))
        print("  spells {:.0f} vs {:.0f}, truth {:.0f}".format(result["candidate"]["pred_spells"],
                                                             result["reference_cells"]["pred_spells"],
                                                             result["candidate"]["gt_spells"]))
        print("  PASS" if cmp["passes"] else "  FAIL", "(+{:g} needed in every cell)".format(GATE), flush=True)
    with open(OUT + ".tmp", "w") as fh:
        json.dump(out, fh, indent=1)
    os.replace(OUT + ".tmp", OUT)
    print("wrote", OUT, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=("build", "gate"))
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    {"build": run_build, "gate": run_gate}[args.stage](args)


if __name__ == "__main__":
    main()
