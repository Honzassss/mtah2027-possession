"""Test 2: shift the dead-ball emission of the v0 decoder by +-1 nat, stress it, and rescore.

    python -I analysis/test2_dead_shift.py predict [--workers 5]   # v0 heads on the held-out copies, cached
    python -I analysis/test2_dead_shift.py run [--workers 6]       # the grid on OOF and held-out copies

The shift b (nats per 5 Hz step, in the decoder's own units) is added to the emission of both dead states,
DH and DA, after alpha. Because Viterbi only compares states, b is the same as moving the dead head's
log-odds by b / alpha (2.8): b = +-1 is a 0.36 nat miscalibration of p_dead, b = +-2.8 is a full nat.
Scopes: every step, only steps without a ball seen (inside blackouts), only steps with a ball seen.
Stress beside the shift: the dead head's log-odds times 0.5 ... 2 (under- and over-confident), and the
whole evidence times 0.5 and 2 (alpha 1.4 and 5.6).

Choice on training folds only: the out-of-fold predictions that chose alpha (results/cache/v0/oof_*),
each fold decoded with the transitions of its own four training matches, as in model_v0 sweep. The
held-out copies (J03WMX, J03WN1; camera, plain and camera-parallax; live 0.1/0.3/0.5; seeds 1-3) are
decoded with the saved v0 heads and table and only reported; a shift chosen on the folds becomes a
change only through the gate (+0.02 on both held-out matches under camera and plain). Rules fixed
before the results: a shift is a candidate only if it gains at least CANDIDATE on the folds; the D
calibration counts as robust if no |b| <= 1 loses more than ROBUST, as critical above CRITICAL.

Reads results/cache/v0 and results/models/v0_* and never writes there; its own cache is
results/cache/tests12. IDSSE only, like model_v0.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis.model_v0 import (CACHE, EVAL_LIVE, EVAL_SIMS, EXTRA_SIMS, HELD_OUT, MODELS, TRAIN,  # noqa: E402
                               cache_path, eval_keys, key_name, score, train_keys)
from pipeline.decode import DA, DH, SCALE, emissions, states_from_labels, transitions, viterbi  # noqa: E402
from pipeline.features import Steps  # noqa: E402
from pipeline.model import HEADS, predict  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
OWN_CACHE = os.path.join(RESULTS, "cache", "tests12")
SHIFTS = (-4.0, -3.0, -2.0, -1.5, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
SCOPES = ("all", "missing", "seen")
TEMPERATURES = (0.5, 0.75, 1.5, 2.0)
EVIDENCE = (0.5, 2.0)
CANDIDATE = 0.005
ROBUST = 0.01
CRITICAL = 0.03
GATE = 0.02
FIELDS = ("spell_f1", "pred_spells", "gt_spells", "macro_f1", "dead_share_pred", "dead_share_true")


def light(key):
    """What decoding and scoring need from a cached copy: no feature matrices kept."""
    with np.load(cache_path(key), allow_pickle=False) as d:
        names = [str(v) for v in d["names"]]
        out = {"steps": Steps(start=d["start"], stop=d["stop"], centre=d["centre"], half=d["half"],
                              size=int(d["size"]), fps=float(d["fps"])),
               "codes": d["codes"], "step_codes": d["step_codes"], "dead": d["dead"],
               "seen": d["X"][:, names.index("ball_seen")] > 0, "info": json.loads(str(d["info"]))}
    return out


def heldout_path(key, tag="v0"):
    return os.path.join(OWN_CACHE, "heldout_{}_{}.npz".format(tag, key_name(key)))


def oof_path(key):
    return os.path.join(CACHE, "oof_" + key_name(key) + ".npz")


def probs(path):
    with np.load(path) as d:
        return {name: d[name] for name in HEADS}


def table_of(copies):
    return transitions([(states_from_labels(c["step_codes"], c["steps"].half), c["steps"].half) for c in copies])


def decode_with(p, steps, table, alpha, dead_only, shift=None, temperature=1.0):
    """model_v0's decode with b nats added to DH and DA (shift: (K,) or None) and the dead head's log-odds
    times `temperature`. With neither it is decode() itself."""
    pd = p["dead"]
    if temperature != 1.0:
        z = np.log(np.clip(pd, 1e-12, 1.0)) - np.log(np.clip(1.0 - pd, 1e-12, 1.0))
        pd = 1.0 / (1.0 + np.exp(-temperature * z))
    scores = emissions(pd, p["team"], p["restart"], table, alpha)
    if shift is not None:
        add = np.round(np.asarray(shift, dtype=np.float64) * SCALE).astype(np.int64)
        scores[:, DH] += add
        scores[:, DA] += add
    out = np.empty(len(scores), dtype=np.int8)
    for a, b in steps.bounds():
        out[a:b] = viterbi(scores[a:b], table.matrix, None if dead_only is None else dead_only[a:b])
    return out


def variants():
    out = [("base", None, 0.0)]
    out += [("shift", scope, b) for scope in SCOPES for b in SHIFTS]
    out += [("temperature", None, t) for t in TEMPERATURES]
    out += [("evidence", None, s) for s in EVIDENCE]
    return out


def variant_name(v):
    kind, scope, value = v
    if kind == "base":
        return "base"
    if kind == "shift":
        return "shift {} {:+g}".format(scope, value)
    return "{} {:g}".format(kind, value)


def eval_copy(job):
    """Every variant on one copy. job: (key, probability file, table, alpha)."""
    key, path, table, alpha = job
    c = light(key)
    p = probs(path)
    k = len(c["step_codes"])
    rows = {}
    for v in variants():
        kind, scope, value = v
        shift, temp, a = None, 1.0, alpha
        if kind == "shift":
            mask = {"all": np.ones(k, dtype=bool), "missing": ~c["seen"], "seen": c["seen"]}[scope]
            shift = np.where(mask, value, 0.0)
        elif kind == "temperature":
            temp = value
        elif kind == "evidence":
            a = alpha * value
        states = decode_with(p, c["steps"], table, a, c["dead"], shift, temp)
        s = score(c, states)
        rows[variant_name(v)] = {f: s[f] for f in FIELDS}
    return key, rows, {"seen_share": float(c["seen"].mean()), "info": c["info"]}


def run_predict(args):
    """v0's saved heads on the 54 held-out copies (`run` checks that b = 0 gives model_v0_eval.json back)."""
    import lightgbm as lgb
    os.makedirs(OWN_CACHE, exist_ok=True)
    boosters = {name: lgb.Booster(model_file=os.path.join(MODELS, "v0_{}.txt".format(name))) for name in HEADS}
    started = time.time()
    for key in eval_keys():
        path = heldout_path(key)
        if not os.path.isfile(path):
            with np.load(cache_path(key), allow_pickle=False) as d:
                X, Xs = d["X"], d["Xs"]
            p = {name: predict(boosters[name], head, X, Xs) for name, head in HEADS.items()}
            np.savez(path + ".tmp.npz", **p)
            os.replace(path + ".tmp.npz", path)
        print("predicted", key_name(key), "{:.0f}s".format(time.time() - started), flush=True)
    print("held-out predictions in", OWN_CACHE, flush=True)


def summarise(rows, keys, groups):
    """Mean of each field per variant over each group of keys. groups: {name: predicate(key)}."""
    out = {}
    for vname in rows[keys[0]]:
        out[vname] = {}
        for gname, pred in groups.items():
            sel = [rows[k][vname] for k in keys if pred(k)]
            out[vname][gname] = {f: float(np.mean([r[f] for r in sel])) for f in FIELDS}
    return out


def run_run(args):
    with open(os.path.join(RESULTS, "model_v0_eval.json")) as fh:
        v0 = json.load(fh)
    with open(os.path.join(RESULTS, "model_v0_cv.json")) as fh:
        cv = json.load(fh)
    alpha = float(v0["alpha"])
    if alpha != float(cv["alpha"]):
        raise SystemExit("alpha differs between the cv and the eval file")
    tkeys, hkeys = train_keys(), eval_keys()
    train = {k: light(k) for k in tkeys}
    full = table_of([train[k] for k in tkeys])
    if full.matrix.tolist() != v0["transitions"]:
        raise SystemExit("the training copies give another table than v0's")
    folds = {held: table_of([train[k] for k in tkeys if k[0] != held]) for held in TRAIN}
    for k in hkeys:
        if not os.path.isfile(heldout_path(k)):
            raise SystemExit("run `predict` first: no " + heldout_path(k))
    jobs = [(k, oof_path(k), folds[k[0]], alpha) for k in tkeys] + [(k, heldout_path(k), full, alpha) for k in hkeys]
    rows, meta = {}, {}
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for key, r, m in pool.map(eval_copy, jobs):
            rows[key], meta[key] = r, m
            print("{:<32} base {:.4f}  shift all -1 {:.4f}  +1 {:.4f}  {:.0f}s".format(
                key_name(key), r["base"]["spell_f1"], r["shift all -1"]["spell_f1"], r["shift all +1"]["spell_f1"],
                time.time() - started), flush=True)

    # b = 0 reproduces v0 exactly: the alpha sweep's OOF mean, and every held-out score
    want = {(r["match"], r["sim"], r["live"], r["seed"]): r["decoded"]["spell_f1"] for r in v0["rows"]}
    bad = [key_name(k) for k in hkeys if rows[k]["base"]["spell_f1"] != want[k]]
    if bad:
        raise SystemExit("b = 0 does not reproduce v0 on " + ", ".join(bad))
    oof_base = float(np.mean([rows[k]["base"]["spell_f1"] for k in tkeys]))
    if abs(oof_base - cv["alpha_sweep"]["{:g}".format(alpha)]["spell_f1"]) > 1e-12:
        raise SystemExit("b = 0 does not reproduce the OOF sweep at alpha {:g}".format(alpha))

    oof_groups = {"all": lambda k: True}
    oof_groups.update({m: (lambda k, m=m: k[0] == m) for m in TRAIN})
    ho_groups = {"gate": lambda k: k[1] in EVAL_SIMS}
    for sim in EVAL_SIMS + EXTRA_SIMS:
        ho_groups[sim] = lambda k, sim=sim: k[1] == sim
        for m in HELD_OUT:
            ho_groups["{} {}".format(sim, m)] = lambda k, sim=sim, m=m: (k[1], k[0]) == (sim, m)
        for live in EVAL_LIVE:
            ho_groups["{} live {:g}".format(sim, live)] = lambda k, sim=sim, live=live: (k[1], k[2]) == (sim, live)
    for live in EVAL_LIVE:
        ho_groups["gate live {:g}".format(live)] = lambda k, live=live: k[1] in EVAL_SIMS and k[2] == live
    oof = summarise(rows, tkeys, oof_groups)
    ho = summarise(rows, hkeys, ho_groups)

    def curve(table, group, scope):
        base = table["base"][group]["spell_f1"]
        out = {"0": 0.0}
        for b in SHIFTS:
            out["{:+g}".format(b)] = table["shift {} {:+g}".format(scope, b)][group]["spell_f1"] - base
        return dict(sorted(out.items(), key=lambda kv: float(kv[0])))

    def worst_within(table, group, scope, limit):
        return min(table["shift {} {:+g}".format(scope, b)][group]["spell_f1"] - table["base"][group]["spell_f1"]
                   for b in SHIFTS if abs(b) <= limit)

    # the choice, on the training folds only
    choice = {}
    for scope in SCOPES:
        gains = {b: oof["shift {} {:+g}".format(scope, b)]["all"]["spell_f1"] - oof["base"]["all"]["spell_f1"]
                 for b in SHIFTS}
        best = max(gains, key=lambda b: gains[b])
        choice[scope] = {"best_shift": best, "oof_gain": gains[best], "at_edge": best in (SHIFTS[0], SHIFTS[-1]),
                         "candidate": gains[best] >= CANDIDATE}
    gate_checks = {}
    for scope, ch in choice.items():
        if not ch["candidate"]:
            continue
        name = "shift {} {:+g}".format(scope, ch["best_shift"])
        cells = {}
        for sim in EVAL_SIMS:
            for m in HELD_OUT:
                g = "{} {}".format(sim, m)
                cells[g] = ho[name][g]["spell_f1"] - ho["base"][g]["spell_f1"]
        gate_checks[name] = {"cells": cells, "passes": all(v >= GATE for v in cells.values())}

    sensitivity = {
        "oof worst |b|<=1, all": worst_within(oof, "all", "all", 1.0),
        "oof worst |b|<=2.8, all": worst_within(oof, "all", "all", 2.8),
        "held-out gate worst |b|<=1, all": worst_within(ho, "gate", "all", 1.0),
        "held-out gate worst |b|<=2.8, all": worst_within(ho, "gate", "all", 2.8),
        "oof worst |b|<=1, missing": worst_within(oof, "all", "missing", 1.0),
        "held-out gate worst |b|<=1, missing": worst_within(ho, "gate", "missing", 1.0),
    }
    worst_cell = {}
    for g in ho_groups:
        worst_cell[g] = worst_within(ho, g, "all", 1.0)
    # reporting only: what a perfectly calibrated shift would gain per held-out live share
    oracle = {}
    for g in ho_groups:
        for scope in ("all", "missing"):
            gains = {b: ho["shift {} {:+g}".format(scope, b)][g]["spell_f1"] - ho["base"][g]["spell_f1"] for b in SHIFTS}
            best = max(gains, key=lambda b: gains[b])
            oracle["{} | {}".format(g, scope)] = {"best_shift": best, "gain": gains[best]}
    robust = sensitivity["oof worst |b|<=1, all"] >= -ROBUST and sensitivity["held-out gate worst |b|<=1, all"] >= -ROBUST
    critical = min(sensitivity["oof worst |b|<=1, all"], sensitivity["held-out gate worst |b|<=1, all"]) < -CRITICAL
    out = {
        "definition": {"shift": "nats added to the DH and DA emissions per 5 Hz step, after alpha",
                       "head_logodds_per_nat": 1.0 / alpha, "alpha": alpha, "shifts": list(SHIFTS),
                       "scopes": {"all": "every step", "missing": "steps without a ball seen",
                                  "seen": "steps with a ball seen"},
                       "temperature": "dead head log-odds times t", "evidence": "alpha times s",
                       "rules": {"candidate_oof_gain": CANDIDATE, "robust_drop": ROBUST, "critical_drop": CRITICAL,
                                 "gate": GATE}},
        "oof_base": oof["base"]["all"], "heldout_gate_base": ho["base"]["gate"],
        "oof_curve": {scope: curve(oof, "all", scope) for scope in SCOPES},
        "heldout_gate_curve": {scope: curve(ho, "gate", scope) for scope in SCOPES},
        "heldout_curve_by_cell": {g: curve(ho, g, "all") for g in ho_groups},
        "oof_stress": {name: oof[name]["all"]["spell_f1"] - oof["base"]["all"]["spell_f1"]
                       for name in oof if name.startswith(("temperature", "evidence"))},
        "heldout_stress": {name: ho[name]["gate"]["spell_f1"] - ho["base"]["gate"]["spell_f1"]
                           for name in ho if name.startswith(("temperature", "evidence"))},
        "sensitivity": sensitivity, "worst_cell_within_1": worst_cell,
        "choice": choice, "gate_checks": gate_checks, "oracle_heldout_reporting_only": oracle,
        "verdict": {"robust": robust, "critical": critical,
                    "change": any(g["passes"] for g in gate_checks.values())},
        "oof": oof, "heldout": ho,
        "rows": {key_name(k): {"rows": rows[k], "meta": meta[k]} for k in tkeys + hkeys},
    }
    with open(os.path.join(RESULTS, "test2_dead_shift.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("OOF base {:.4f}  held-out gate base {:.4f}".format(oof["base"]["all"]["spell_f1"], ho["base"]["gate"]["spell_f1"]))
    for scope in SCOPES:
        print("OOF curve", scope, {b: round(v, 4) for b, v in out["oof_curve"][scope].items()})
        print("held-out gate curve", scope, {b: round(v, 4) for b, v in out["heldout_gate_curve"][scope].items()})
    print("stress OOF", {k: round(v, 4) for k, v in out["oof_stress"].items()})
    print("stress held-out", {k: round(v, 4) for k, v in out["heldout_stress"].items()})
    print("sensitivity", {k: round(v, 4) for k, v in sensitivity.items()})
    print("choice", choice)
    print("gate checks", gate_checks)
    print("verdict", out["verdict"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=("predict", "run"))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    {"predict": run_predict, "run": run_run}[args.stage](args)


if __name__ == "__main__":
    main()
