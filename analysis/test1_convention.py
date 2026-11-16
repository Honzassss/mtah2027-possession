"""Test 1: the boundary convention for short possessions, T = 0, 0.4, 1 and 2 s.

    python -I analysis/test1_convention.py labels                # spells per convention
    python -I analysis/test1_convention.py cv [--T 0.4 1 2 0]    # v0-style heads per T, one training match out
                                                                 # (T 0 re-runs v0's folds as a bit-exact check)
    python -I analysis/test1_convention.py sweep                 # alpha per T on its OOF, then the OOF cross-score
    python -I analysis/test1_convention.py final                 # heads per T on all five, held-out cross-score
    python -I analysis/test1_convention.py decoder               # v0's heads with the decoder of each T
    python -I analysis/test1_convention.py report                # the decision record, from the stages above

Run in that order, after `analysis/test2_dead_shift.py predict` (the held-out v0 predictions it caches).

Convention T: in live play, a run of one team shorter than T seconds does not change possession; it takes
the label of the team that had the ball before it, so touches, deflections and duels without control
stay with the previous team ("in phases where the ball is contested or loose, possession stays with the
last team that had it", CHALLENGE.md). The first run after a stoppage or at the start of a half is the
restarter's and stays whatever its length ("whenever play restarts after a stoppage, there is a clear
possessor"). Run lengths are those of the DFL labels; dead frames never change. T = 0 is the DFL label
as it is, the one v0 was trained on.

A v0-style model per T is v0's recipe on the relabelled training copies: the same cached copies and
features, the three heads with rounds from leave-one-match-out early stopping, the transitions counted on
the T labels, and alpha chosen on that T's out-of-fold predictions. Each model is then scored against the
labels of every convention (the cross-score). The decision is taken on the training folds only: if no
model loses more than LOSS against the model of the convention the truth follows, T does not matter and
is set in advance from the rules text. J03WMX and J03WN1 are only reported. Nothing here looks at the
delay table in EVALUATION.md or at anyone's hand labels.

Reads results/cache/v0 and results/models/v0_* and never writes there; its own cache is
results/cache/tests12 (OOF and held-out predictions, the heads per T in models/). IDSSE only.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis.model_v0 import (ALPHAS, CACHE, EVAL_LIVE, EVAL_SIMS, EXTRA_SIMS, HELD_OUT, MODELS, TRAIN,  # noqa: E402
                               eval_keys, key_name, load, predict_all, score, train_keys)
from analysis.test2_dead_shift import OWN_CACHE, heldout_path, light  # noqa: E402
from pipeline.decode import decode, states_from_labels, transitions  # noqa: E402
from pipeline.model import HEADS, fit, predict, step_targets  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
CONVENTIONS = (0.0, 0.4, 1.0, 2.0)
LOSS = 0.01
GATE = 0.02
OUT = os.path.join(RESULTS, "test1_convention.json")


def tag(T):
    return "T{:g}".format(T)


def frame_half(steps, n):
    return steps.frame_labels(steps.half, n, np.int8(0))


def relabel(codes, half, fps, T):
    """25 Hz codes under convention T (seconds). half: (F,) the half of each frame."""
    codes = np.asarray(codes)
    out = codes.copy()
    if T <= 0:
        return out
    need = int(round(T * fps))
    edge = np.flatnonzero((codes[1:] != codes[:-1]) | (half[1:] != half[:-1])) + 1
    bounds = np.concatenate(([0], edge, [len(codes)]))
    current, last_half = -1, None
    for a, b in zip(bounds[:-1].tolist(), bounds[1:].tolist()):
        v, h = int(codes[a]), int(half[a])
        if h != last_half:
            current, last_half = -1, h
        if v >= 2:                      # dead (or unlabelled): the next run is a restart
            current = -1
            continue
        if current < 0 or (v != current and b - a >= need):
            current = v
        out[a:b] = current
    return out


def conventions_of(c):
    """{T: 25 Hz codes} for one copy (a light() or load() dict)."""
    half = frame_half(c["steps"], len(c["codes"]))
    return {T: relabel(c["codes"], half, c["steps"].fps, T) for T in CONVENTIONS}


def step_codes_of(c, codes):
    return codes[c["steps"].centre]


def table_for(copies, T):
    seqs = []
    for c in copies:
        sc = step_codes_of(c, c["conv"][T])
        seqs.append((states_from_labels(sc, c["steps"].half), c["steps"].half))
    return transitions(seqs)


def oof_file(key, T, own_t0=False):
    """v0's saved OOF stands for T = 0 unless own_t0 asks for this script's re-run of those folds."""
    if T == 0.0 and not own_t0:
        return os.path.join(CACHE, "oof_" + key_name(key) + ".npz")
    return os.path.join(OWN_CACHE, "oof_{}_{}.npz".format(tag(T), key_name(key)))


def spells_in(codes):
    live = codes < 2
    e = np.diff(np.concatenate(([0], live.astype(np.int8), [0])))
    starts, stops = np.flatnonzero(e == 1), np.flatnonzero(e == -1)
    n = 0
    for a, b in zip(starts, stops):
        n += 1 + int(np.count_nonzero(codes[a + 1:b] != codes[a:b - 1]))
    return n


def run_labels(args):
    out = {}
    for key in [k for k in train_keys() if k[3] == 1] + [k for k in eval_keys() if k[1] == "camera" and k[2] == 0.1 and k[3] == 1]:
        c = light(key)
        conv = conventions_of(c)
        base = conv[0.0]
        row = {}
        for T, codes in conv.items():
            row[tag(T)] = {"spells": spells_in(codes), "frames_changed": int(np.count_nonzero(codes != base)),
                           "live_frames_changed_share": float(np.count_nonzero(codes != base) / max(1, np.count_nonzero(base < 2))),
                           "dead_frames_equal": bool(np.array_equal(codes == 2, base == 2))}
        out[key[0]] = row
        print(key[0], {t: (r["spells"], round(r["live_frames_changed_share"], 4)) for t, r in row.items()}, flush=True)
    save({"labels": out})


def read_out():
    if not os.path.isfile(OUT):
        return {}
    with open(OUT) as fh:
        return json.load(fh)


def save(update):
    data = read_out()
    data.update(update)
    with open(OUT + ".tmp", "w") as fh:
        json.dump(data, fh, indent=1)
    os.replace(OUT + ".tmp", OUT)


def run_cv(args):
    os.makedirs(OWN_CACHE, exist_ok=True)
    keys = train_keys()
    copies = {k: load(k) for k in keys}
    names = copies[keys[0]]["names"]
    for k in keys:
        copies[k]["conv"] = conventions_of(copies[k])
    cv_all = read_out().get("cv", {})
    for T in args.T:
        targets = {k: step_targets(step_codes_of(copies[k], copies[k]["conv"][T]), copies[k]["steps"]) for k in keys}
        folds = []
        for held in TRAIN:
            started = time.time()
            parts = [(copies[k]["X"], copies[k]["Xs"], targets[k]) for k in keys if k[0] != held]
            valid = [(copies[k]["X"], copies[k]["Xs"], targets[k]) for k in keys if k[0] == held]
            boosters, best = {}, {}
            for name, head in HEADS.items():
                boosters[name] = fit(head, parts, names, valid=valid)
                best[name] = int(boosters[name].best_iteration)
            for k in keys:
                if k[0] == held:
                    p = predict_all(boosters, copies[k], best)
                    path = os.path.join(OWN_CACHE, "oof_{}_{}.npz".format(tag(T), key_name(k)))
                    np.savez(path + ".tmp.npz", **p)
                    os.replace(path + ".tmp.npz", path)
            folds.append({"held": held, "best_iteration": best, "seconds": time.time() - started})
            print(tag(T), "fold", held, best, "{:.0f}s".format(time.time() - started), flush=True)
        cv_all[tag(T)] = {"folds": folds,
                          "rounds": {name: int(np.median([f["best_iteration"][name] for f in folds])) for name in HEADS}}
        save({"cv": cv_all})
        print(tag(T), "rounds", cv_all[tag(T)]["rounds"], flush=True)


def cross(copy, states, conv):
    """Scores of one decoded path against the labels of every convention."""
    out = {}
    for T, codes in conv.items():
        s = score({"steps": copy["steps"], "codes": codes}, states)
        out[tag(T)] = {f: s[f] for f in ("spell_f1", "pred_spells", "gt_spells", "macro_f1")}
    return out


def matrix_of(rows, keys):
    """{model T: {truth T: mean spellF1}} over keys. rows[key][model][truth]."""
    return {tag(m): {tag(t): float(np.mean([rows[k][tag(m)][tag(t)]["spell_f1"] for k in keys])) for t in CONVENTIONS}
            for m in CONVENTIONS}


def losses_of(mat):
    """loss[model][truth] = matched model's score on that truth minus this model's."""
    return {m: {t: mat[t][t] - mat[m][t] for t in mat[m]} for m in mat}


def run_sweep(args):
    with open(os.path.join(RESULTS, "model_v0_cv.json")) as fh:
        v0cv = json.load(fh)
    keys = train_keys()
    copies = {k: light(k) for k in keys}
    for k in keys:
        copies[k]["conv"] = conventions_of(copies[k])
    oof = {}
    for T in CONVENTIONS:
        for k in keys:
            with np.load(oof_file(k, T, args.own_t0)) as d:
                oof[(T, k)] = {name: d[name] for name in HEADS}
    check = {}
    own0 = [os.path.join(OWN_CACHE, "oof_T0_{}.npz".format(key_name(k))) for k in keys]
    if all(os.path.isfile(p) for p in own0):     # the re-run of T = 0 must be v0's OOF, bit for bit
        same = True
        for k, p in zip(keys, own0):
            with np.load(p) as a, np.load(os.path.join(CACHE, "oof_" + key_name(k) + ".npz")) as b:
                same &= all(np.array_equal(a[n], b[n]) for n in HEADS)
        check["T0 re-run equals v0 OOF"] = bool(same)
        print("T0 re-run equals v0's OOF:", same, flush=True)
    sweep, alpha = {}, {}
    for T in CONVENTIONS:
        tables = {held: table_for([copies[k] for k in keys if k[0] != held], T) for held in TRAIN}
        sweep[tag(T)] = {}
        for a in ALPHAS:
            f1 = []
            for k in keys:
                p = oof[(T, k)]
                st = decode(p["dead"], p["team"], p["restart"], copies[k]["steps"], tables[k[0]], a, copies[k]["dead"])
                f1.append(score({"steps": copies[k]["steps"], "codes": copies[k]["conv"][T]}, st)["spell_f1"])
            sweep[tag(T)]["{:g}".format(a)] = float(np.mean(f1))
        best = max(sweep[tag(T)], key=lambda s: sweep[tag(T)][s])
        alpha[tag(T)] = float(best)
        print(tag(T), "alpha", best, {s: round(v, 4) for s, v in sweep[tag(T)].items()}, flush=True)
    if alpha["T0"] != float(v0cv["alpha"]) or abs(sweep["T0"]["{:g}".format(v0cv["alpha"])]
                                                   - v0cv["alpha_sweep"]["{:g}".format(v0cv["alpha"])]["spell_f1"]) > 1e-12:
        raise SystemExit("T = 0 does not reproduce v0's alpha sweep")
    rows = {}
    for k in keys:
        rows[k] = {}
        for T in CONVENTIONS:
            tables = table_for([copies[kk] for kk in keys if kk[0] != k[0]], T)
            p = oof[(T, k)]
            st = decode(p["dead"], p["team"], p["restart"], copies[k]["steps"], tables, alpha[tag(T)], copies[k]["dead"])
            rows[k][tag(T)] = cross(copies[k], st, copies[k]["conv"])
    mat = matrix_of(rows, keys)
    by_match = {m: matrix_of(rows, [k for k in keys if k[0] == m]) for m in TRAIN}
    loss = losses_of(mat)
    worst = {m: max(loss[m].values()) for m in loss}
    minimax = min(worst, key=lambda m: worst[m])
    spells = {tag(t): {"gt": float(np.mean([rows[k]["T0"][tag(t)]["gt_spells"] for k in keys]))} for t in CONVENTIONS}
    for m in CONVENTIONS:
        spells[tag(m)]["pred"] = float(np.mean([rows[k][tag(m)][tag(m)]["pred_spells"] for k in keys]))
    verdict = {"max_loss": max(worst.values()), "max_loss_pair": max(((m, t) for m in loss for t in loss[m]),
                                                                       key=lambda mt: loss[mt[0]][mt[1]]),
               "T_matters": max(worst.values()) >= LOSS, "worst_loss_by_model": worst,
               "minimax_model": minimax, "minimax_worst_loss": worst[minimax]}
    save({"sweep": {"alpha_sweep": sweep, "alpha": alpha, "checks": check},
          "oof": {"matrix": mat, "loss": loss, "by_match": by_match, "spells": spells, "verdict": verdict,
                  "rows": {key_name(k): rows[k] for k in keys}}})
    print_matrix("OOF spellF1, model (row) x truth (column)", mat)
    print_matrix("OOF loss against the matched model", loss)
    print("spells", spells)
    print("verdict", verdict, flush=True)


def run_decoder(args):
    """Which part of a T model carries the convention: v0's own OOF heads with the decoder of convention T
    (fold transitions counted on the T labels, alpha chosen on the T truth), scored against every truth and
    compared with the full T models of `sweep`. Training folds only."""
    data = read_out()
    full = data["oof"]["matrix"]
    keys = train_keys()
    copies = {k: light(k) for k in keys}
    for k in keys:
        copies[k]["conv"] = conventions_of(copies[k])
    oof = {}
    for k in keys:
        with np.load(oof_file(k, 0.0)) as d:
            oof[k] = {name: d[name] for name in HEADS}
    sweep, alpha, rows = {}, {}, {k: {} for k in keys}
    for T in CONVENTIONS:
        tables = {held: table_for([copies[k] for k in keys if k[0] != held], T) for held in TRAIN}
        sweep[tag(T)] = {}
        paths = {}
        for a in ALPHAS:
            f1 = []
            for k in keys:
                p = oof[k]
                paths[(a, k)] = decode(p["dead"], p["team"], p["restart"], copies[k]["steps"], tables[k[0]], a,
                                       copies[k]["dead"])
                f1.append(score({"steps": copies[k]["steps"], "codes": copies[k]["conv"][T]}, paths[(a, k)])["spell_f1"])
            sweep[tag(T)]["{:g}".format(a)] = float(np.mean(f1))
        best = max(sweep[tag(T)], key=lambda s: sweep[tag(T)][s])
        alpha[tag(T)] = float(best)
        for k in keys:
            rows[k][tag(T)] = cross(copies[k], paths[(float(best), k)], copies[k]["conv"])
        print("v0 heads, decoder", tag(T), "alpha", best, flush=True)
    mat = matrix_of(rows, keys)
    loss = {m: {t: full[t][t] - mat[m][t] for t in mat[m]} for m in mat}   # against the matched full model
    worst = {m: max(v.values()) for m, v in loss.items()}
    # regret against the best of all seven candidates (four full models, three v0-head decoders) per truth
    cands = {"full " + m: full[m] for m in full}
    cands.update({"v0 heads + decoder " + m: mat[m] for m in mat if m != "T0"})
    best = {tag(t): max(c[tag(t)] for c in cands.values()) for t in CONVENTIONS}
    regret = {name: max(best[t] - c[t] for t in best) for name, c in cands.items()}
    out = {"alpha": alpha, "alpha_sweep": sweep, "matrix": mat, "loss_vs_full_matched": loss,
           "worst_loss_by_decoder": worst, "best_by_truth": best, "regret_vs_best_candidate": regret,
           "minimax_candidate": min(regret, key=lambda n: regret[n]),
           "rows": {key_name(k): rows[k] for k in keys}}
    print_matrix("OOF spellF1, v0 heads with the decoder of T (row) x truth (column)", mat)
    print_matrix("loss against the matched full model", loss)
    print("worst loss by decoder", {m: round(v, 4) for m, v in worst.items()})
    print("regret against the best candidate per truth", {n: round(v, 4) for n, v in regret.items()}, flush=True)

    # held-out, reporting only: the saved v0 heads with each T's table from all five training matches
    hkeys = eval_keys()
    train = {k: light(k) for k in keys}
    for k in keys:
        train[k]["conv"] = conventions_of(train[k])
    tables = {tag(T): table_for([train[k] for k in keys], T) for T in CONVENTIONS}
    hrows = {}
    for key in hkeys:
        c = light(key)
        c["conv"] = conventions_of(c)
        with np.load(heldout_path(key, "v0")) as d:
            p = {name: d[name] for name in HEADS}
        hrows[key] = {}
        for T in CONVENTIONS:
            st = decode(p["dead"], p["team"], p["restart"], c["steps"], tables[tag(T)], alpha[tag(T)], c["dead"])
            hrows[key][tag(T)] = cross(c, st, c["conv"])
    groups = heldout_groups(hkeys)
    out["heldout"] = {"matrix": {g: matrix_of(hrows, ks) for g, ks in groups.items()},
                      "rows": {key_name(k): hrows[k] for k in hkeys}}
    out["heldout"]["gate_view"] = gate_view_of(out["heldout"]["matrix"])
    save({"decoder_only": out})
    print_matrix("held-out gate, v0 heads with the decoder of T x truth", out["heldout"]["matrix"]["gate"])
    print("gate view", {k: v["passes"] for k, v in out["heldout"]["gate_view"].items()}, flush=True)


def run_report(args):
    """The decision record: chosen on the folds, held-out beside it. Reads what sweep, decoder and final
    wrote; candidates are the four full models and v0's heads with the decoders of T > 0."""
    data = read_out()

    def candidates(full, dec):
        out = {"full " + m: full[m] for m in full}
        out.update({"v0 heads + decoder " + m: dec[m] for m in dec if m != "T0"})
        return out

    def regret(cands):
        best = {t: max(c[t] for c in cands.values()) for t in next(iter(cands.values()))}
        return {n: {t: best[t] - c[t] for t in best} for n, c in cands.items()}

    oof = regret(candidates(data["oof"]["matrix"], data["decoder_only"]["matrix"]))
    held = regret(candidates(data["heldout"]["matrix"]["gate"], data["decoder_only"]["heldout"]["matrix"]["gate"]))
    worst_oof = {n: max(v.values()) for n, v in oof.items()}
    worst_held = {n: max(v.values()) for n, v in held.items()}
    chosen = min(worst_oof, key=lambda n: worst_oof[n])
    v = data["oof"]["verdict"]
    decision = {
        "T_matters": v["T_matters"], "max_cross_loss_oof": v["max_loss"], "max_cross_loss_pair": v["max_loss_pair"],
        "minimax_model_oof": v["minimax_model"], "minimax_worst_loss_oof": v["minimax_worst_loss"],
        "minimax_worst_loss_heldout": data["heldout"]["worst_loss_by_model_gate"][v["minimax_model"]],
        "regret_vs_best_candidate_oof": worst_oof, "regret_vs_best_candidate_heldout": worst_held,
        "regret_rows_oof": oof, "regret_rows_heldout": held, "chosen_candidate_oof": chosen,
        "gate_on_idsse_labels": data["heldout"]["gate_view"]["model {} vs v0, truth T0".format(v["minimax_model"])],
        "rule": "T is set from the rules text only if no cross loss reaches {:g} on the folds".format(LOSS),
    }
    save({"decision": decision})
    for name in worst_oof:
        print("{:<26} worst regret OOF {:.4f}  held-out {:.4f}".format(name, worst_oof[name], worst_held[name]))
    print("chosen on the folds:", chosen, "| T matters:", v["T_matters"], flush=True)


def heldout_groups(hkeys):
    groups = {"gate": [k for k in hkeys if k[1] in EVAL_SIMS]}
    for sim in EVAL_SIMS + EXTRA_SIMS:
        groups[sim] = [k for k in hkeys if k[1] == sim]
        for m in HELD_OUT:
            groups["{} {}".format(sim, m)] = [k for k in hkeys if (k[1], k[0]) == (sim, m)]
        for live in EVAL_LIVE:
            groups["{} live {:g}".format(sim, live)] = [k for k in hkeys if (k[1], k[2]) == (sim, live)]
    return groups


def gate_view_of(mats):
    """Per truth T': model T against v0 (T0) in each gate cell, as if the truth followed T'."""
    out = {}
    for t in CONVENTIONS:
        for m in CONVENTIONS[1:]:
            cells = {"{} {}".format(sim, mt): mats["{} {}".format(sim, mt)][tag(m)][tag(t)]
                     - mats["{} {}".format(sim, mt)]["T0"][tag(t)] for sim in EVAL_SIMS for mt in HELD_OUT}
            out["model {} vs v0, truth {}".format(tag(m), tag(t))] = {
                "cells": cells, "passes": all(v >= GATE for v in cells.values())}
    return out


def print_matrix(title, mat):
    print(title)
    cols = list(next(iter(mat.values())))
    print("{:>8}".format("") + "".join("{:>9}".format(c) for c in cols))
    for m, row in mat.items():
        print("{:>8}".format(m) + "".join("{:>9.4f}".format(row[c]) for c in cols), flush=True)


def run_final(args):
    import lightgbm as lgb
    data = read_out()
    with open(os.path.join(RESULTS, "model_v0_cv.json")) as fh:
        v0cv = json.load(fh)
    with open(os.path.join(RESULTS, "model_v0_eval.json")) as fh:
        v0 = json.load(fh)
    alpha = data["sweep"]["alpha"]
    rounds = {tag(T): (v0cv["rounds"] if T == 0.0 else data["cv"][tag(T)]["rounds"]) for T in CONVENTIONS}
    model_dir = os.path.join(OWN_CACHE, "models")
    os.makedirs(model_dir, exist_ok=True)
    tkeys, hkeys = train_keys(), eval_keys()

    train = {k: load(k) for k in tkeys}
    names = train[tkeys[0]]["names"]
    for k in tkeys:
        train[k]["conv"] = conventions_of(train[k])
    tables = {tag(T): table_for([train[k] for k in tkeys], T) for T in CONVENTIONS}
    if tables["T0"].matrix.tolist() != v0["transitions"]:
        raise SystemExit("T = 0 gives another table than v0's")
    boosters = {"T0": {name: lgb.Booster(model_file=os.path.join(MODELS, "v0_{}.txt".format(name))) for name in HEADS}}
    for T in CONVENTIONS[1:]:
        boosters[tag(T)] = {}
        for name, head in HEADS.items():
            path = os.path.join(model_dir, "{}_{}.txt".format(tag(T), name))
            if not os.path.isfile(path):
                started = time.time()
                targets = [step_targets(step_codes_of(train[k], train[k]["conv"][T]), train[k]["steps"]) for k in tkeys]
                parts = [(train[k]["X"], train[k]["Xs"], t) for k, t in zip(tkeys, targets)]
                b = fit(head, parts, names, rounds=rounds[tag(T)][name])
                b.save_model(path + ".tmp")
                os.replace(path + ".tmp", path)
                del parts, targets
                print("trained", tag(T), name, rounds[tag(T)][name], "{:.0f}s".format(time.time() - started), flush=True)
            boosters[tag(T)][name] = lgb.Booster(model_file=path)
    del train

    want = {(r["match"], r["sim"], r["live"], r["seed"]): r["decoded"]["spell_f1"] for r in v0["rows"]}
    rows = {}
    for key in hkeys:
        c = light(key)
        c["conv"] = conventions_of(c)
        X = Xs = None
        rows[key] = {}
        for T in CONVENTIONS:
            path = heldout_path(key, "v0" if T == 0.0 else tag(T))
            if not os.path.isfile(path):
                if X is None:
                    with np.load(os.path.join(CACHE, key_name(key) + ".npz"), allow_pickle=False) as d:
                        X, Xs = d["X"], d["Xs"]
                p = {name: predict(boosters[tag(T)][name], head, X, Xs) for name, head in HEADS.items()}
                np.savez(path + ".tmp.npz", **p)
                os.replace(path + ".tmp.npz", path)
            with np.load(path) as d:
                p = {name: d[name] for name in HEADS}
            st = decode(p["dead"], p["team"], p["restart"], c["steps"], tables[tag(T)], alpha[tag(T)], c["dead"])
            rows[key][tag(T)] = cross(c, st, c["conv"])
        if rows[key]["T0"]["T0"]["spell_f1"] != want[key]:
            raise SystemExit("T = 0 does not reproduce v0 on " + key_name(key))
        print("{:<32}".format(key_name(key)) + "  ".join(
            "{}: ".format(tag(m)) + " ".join("{:.3f}".format(rows[key][tag(m)][tag(t)]["spell_f1"]) for t in CONVENTIONS)
            for m in CONVENTIONS), flush=True)

    groups = heldout_groups(hkeys)
    mats = {g: matrix_of(rows, ks) for g, ks in groups.items()}
    loss = losses_of(mats["gate"])
    gate_view = gate_view_of(mats)
    spells = {tag(t): {"gt": float(np.mean([rows[k]["T0"][tag(t)]["gt_spells"] for k in groups["gate"]]))}
              for t in CONVENTIONS}
    for m in CONVENTIONS:
        spells[tag(m)]["pred"] = float(np.mean([rows[k][tag(m)][tag(m)]["pred_spells"] for k in groups["gate"]]))
    save({"heldout": {"alpha": alpha, "rounds": rounds, "matrix": mats, "loss_gate": loss,
                      "worst_loss_by_model_gate": {m: max(v.values()) for m, v in loss.items()},
                      "gate_view": gate_view, "spells": spells,
                      "rows": {key_name(k): rows[k] for k in hkeys}}})
    print_matrix("held-out gate (camera + plain) spellF1, model x truth", mats["gate"])
    print_matrix("held-out gate loss against the matched model", loss)
    for g in ("camera J03WMX", "camera J03WN1", "plain J03WMX", "plain J03WN1", "camera-parallax"):
        print_matrix(g, mats[g])
    print("spells", spells)
    print("gate view", {k: v["passes"] for k, v in gate_view.items()}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=("labels", "cv", "sweep", "decoder", "final", "report"))
    parser.add_argument("--T", type=float, nargs="+", default=[0.4, 1.0, 2.0])
    parser.add_argument("--own-t0", action="store_true", help="sweep: T = 0 from its own re-run, not v0's OOF")
    args = parser.parse_args()
    {"labels": run_labels, "cv": run_cv, "sweep": run_sweep, "decoder": run_decoder,
     "final": run_final, "report": run_report}[args.stage](args)


if __name__ == "__main__":
    main()
