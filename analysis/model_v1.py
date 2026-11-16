"""Model v1 candidate: v0 plus the change heads (switch, stop, resume) as per-step transitions.

    python -I analysis/model_v1.py cv       # change heads, leave one training match out; needs model_v0 cv
    python -I analysis/model_v1.py sweep    # alpha on the out-of-fold predictions of the training matches
    python -I analysis/model_v1.py final    # train on all five, score the held-out copies against v0

Everything v0 has stays as it is: the cached copies, the dead, team and restart heads (results/models/v0_*)
and their out-of-fold predictions. Only the transitions change, so the comparison with v0 is one change.
The gate: the change passes if it improves both held-out matches by at least 0.02 under both
simulators, averaged over seeds 1-3 (and the three live shares). IDSSE only, like model_v0.
"""
import argparse
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis.model_v0 import (CACHE, EVAL_LIVE, EVAL_SIMS, EXTRA_SIMS, HELD_OUT, MODELS, TRAIN,  # noqa: E402
                               argmax_states, eval_keys, head_quality, key_name, load, score, table_from,
                               team_by_ball, train_keys)
from pipeline.decode import decode  # noqa: E402
from pipeline.model import CHANGE_HEADS, HEADS, fit, predict  # noqa: E402

TAG = "v1"
ALPHAS = (0.2, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0)
GATE = 0.02
RESULTS = os.path.join(ROOT, "results")


def oof_path(key, tag):
    return os.path.join(CACHE, "oof{}_{}.npz".format("" if tag == "v0" else "_" + tag, key_name(key)))


def run_cv(args):
    keys = train_keys()
    copies = {k: load(k) for k in keys}
    names = copies[keys[0]]["names"]
    folds = []
    for held in TRAIN:
        started = time.time()
        parts = [(copies[k]["X"], copies[k]["Xs"], copies[k]["targets"]) for k in keys if k[0] != held]
        valid = [(copies[k]["X"], copies[k]["Xs"], copies[k]["targets"]) for k in keys if k[0] == held]
        best, boosters = {}, {}
        for name, head in CHANGE_HEADS.items():
            boosters[name] = fit(head, parts, names, valid=valid)
            best[name] = int(boosters[name].best_iteration)
        for k in keys:
            if k[0] == held:
                c = copies[k]
                np.savez(oof_path(k, TAG), **{name: predict(boosters[name], head, c["X"], c["Xs"], best[name])
                                              for name, head in CHANGE_HEADS.items()})
        folds.append({"held": held, "best_iteration": best, "seconds": time.time() - started})
        print("fold", held, best, "{:.0f}s".format(time.time() - started), flush=True)
    oof = {k: dict(np.load(oof_path(k, TAG))) for k in keys}
    quality = {name: head_quality([copies[k] for k in keys], [oof[k] for k in keys], name) for name in CHANGE_HEADS}
    for name in CHANGE_HEADS:
        rates = [float(copies[k]["targets"][name][0][copies[k]["targets"][name][1]].mean()) for k in keys]
        quality[name]["positive_rate"] = float(np.mean(rates))
        quality[name]["mean_prediction"] = float(np.mean([oof[k][name][copies[k]["targets"][name][1]].mean() for k in keys]))
    out = {"folds": folds, "head_quality": quality,
           "rounds": {name: int(np.median([f["best_iteration"][name] for f in folds])) for name in CHANGE_HEADS}}
    with open(os.path.join(RESULTS, "model_{}_cv.json".format(TAG)), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1), flush=True)
    run_sweep(args, copies)


def run_sweep(args, copies=None):
    keys = train_keys()
    copies = copies or {k: load(k) for k in keys}
    tables = {held: table_from([copies[k] for k in keys if k[0] != held]) for held in TRAIN}
    base = {k: dict(np.load(oof_path(k, "v0"))) for k in keys}
    change = {k: dict(np.load(oof_path(k, TAG))) for k in keys}
    sweep = {}
    for alpha in ALPHAS:
        rows = []
        for k in keys:
            p, h = base[k], change[k]
            states = decode(p["dead"], p["team"], p["restart"], copies[k]["steps"], tables[k[0]], alpha,
                            copies[k]["dead"], hazards=(h["switch"], h["stop"], h["resume"]))
            rows.append(score(copies[k], states))
        sweep["{:g}".format(alpha)] = {
            "spell_f1": float(np.mean([r["spell_f1"] for r in rows])),
            "spell_f1_by_match": {m: float(np.mean([r["spell_f1"] for kk, r in zip(keys, rows) if kk[0] == m])) for m in TRAIN},
            "macro_f1": float(np.mean([r["macro_f1"] for r in rows])),
            "pred_spells": float(np.mean([r["pred_spells"] for r in rows])),
            "gt_spells": float(np.mean([r["gt_spells"] for r in rows]))}
        print("alpha", alpha, {k: v for k, v in sweep["{:g}".format(alpha)].items() if k != "spell_f1_by_match"},
              flush=True)
    chosen = max(sweep, key=lambda a: sweep[a]["spell_f1"])
    path = os.path.join(RESULTS, "model_{}_cv.json".format(TAG))
    with open(path) as fh:
        out = json.load(fh)
    with open(os.path.join(RESULTS, "model_v0_cv.json")) as fh:
        v0 = json.load(fh)
    out.update({"alpha_sweep": sweep, "alpha": float(chosen),
                "alpha_at_edge": chosen in ("{:g}".format(ALPHAS[0]), "{:g}".format(ALPHAS[-1])),
                "v0_oof_spell_f1": v0["alpha_sweep"]["{:g}".format(v0["alpha"])]["spell_f1"]})
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print("alpha", chosen, "at the edge" if out["alpha_at_edge"] else "inside the grid",
          "OOF spellF1 {:.4f} (v0 {:.4f})".format(sweep[chosen]["spell_f1"], out["v0_oof_spell_f1"]), flush=True)


def run_final(args):
    with open(os.path.join(RESULTS, "model_{}_cv.json".format(TAG))) as fh:
        cv = json.load(fh)
    with open(os.path.join(RESULTS, "model_v0_eval.json")) as fh:
        v0 = json.load(fh)
    alpha, rounds = float(cv["alpha"]), cv["rounds"]
    base = {name: lgb.Booster(model_file=os.path.join(MODELS, "v0_{}.txt".format(name))) for name in HEADS}
    train = [load(k) for k in train_keys()]
    names = train[0]["names"]
    table = table_from(train)
    if table.matrix.tolist() != v0["transitions"]:
        raise SystemExit("the training transitions differ from v0's")
    parts = [(c["X"], c["Xs"], c["targets"]) for c in train]
    boosters = {}
    for name, head in CHANGE_HEADS.items():
        started = time.time()
        boosters[name] = fit(head, parts, names, rounds=rounds[name])
        boosters[name].save_model(os.path.join(MODELS, "{}_{}.txt".format(TAG, name)))
        print("trained", name, rounds[name], "{:.0f}s".format(time.time() - started), flush=True)
    del parts, train

    v0_rows = {(r["match"], r["sim"], r["live"], r["seed"]): r for r in v0["rows"]}
    rows = []
    for n, key in enumerate(eval_keys()):
        copy = load(key)
        p = {name: predict(base[name], HEADS[name], copy["X"], copy["Xs"]) for name in HEADS}
        if n == 0:   # the saved v0 heads with the counted transitions must give v0's result exactly
            again = score(copy, decode(p["dead"], p["team"], p["restart"], copy["steps"], table, float(v0["alpha"]),
                                       copy["dead"]))["spell_f1"]
            if again != v0_rows[key]["decoded"]["spell_f1"]:
                raise SystemExit("the saved v0 heads do not reproduce v0")
        h = {name: predict(boosters[name], head, copy["X"], copy["Xs"]) for name, head in CHANGE_HEADS.items()}
        states = decode(p["dead"], p["team"], p["restart"], copy["steps"], table, alpha, copy["dead"],
                        hazards=(h["switch"], h["stop"], h["resume"]))
        decoded = score(copy, states)
        match, sim, live, seed = key
        before = v0_rows[key]["decoded"]["spell_f1"]
        rows.append({"match": match, "sim": sim, "live": live, "seed": seed, "decoded": decoded, "v0": before,
                     "change_quality": {name: head_quality([copy], [h], name) for name in CHANGE_HEADS},
                     "team_by_ball": team_by_ball(copy, p["team"])})
        print("{:<28} spellF1 {:.3f} (v0 {:.3f}, {:+.3f})  spells {}/{}  D {:.3f}/{:.3f}".format(
            key_name(key), decoded["spell_f1"], before, decoded["spell_f1"] - before, decoded["pred_spells"],
            decoded["gt_spells"], decoded["dead_share_pred"], decoded["dead_share_true"]), flush=True)

    def mean(sel, field):
        return float(np.mean([r["decoded"]["spell_f1"] if field == "v1" else r["v0"] for r in sel]))

    summary, gate = {}, {}
    for sim in EVAL_SIMS + EXTRA_SIMS:
        for match in HELD_OUT:
            sel = [r for r in rows if (r["sim"], r["match"]) == (sim, match)]
            gain = mean(sel, "v1") - mean(sel, "v0")
            summary["{} {}".format(sim, match)] = {"v1": mean(sel, "v1"), "v0": mean(sel, "v0"), "gain": gain}
            if sim in EVAL_SIMS:
                gate["{} {}".format(sim, match)] = gain >= GATE
        for live in EVAL_LIVE:
            sel = [r for r in rows if (r["sim"], r["live"]) == (sim, live)]
            summary["{} live {:g}".format(sim, live)] = {"v1": mean(sel, "v1"), "v0": mean(sel, "v0"),
                                                        "gain": mean(sel, "v1") - mean(sel, "v0")}
        sel = [r for r in rows if r["sim"] == sim]
        summary[sim] = {"v1": mean(sel, "v1"), "v0": mean(sel, "v0"), "gain": mean(sel, "v1") - mean(sel, "v0"),
                        "pred_spells": float(np.mean([r["decoded"]["pred_spells"] for r in sel])),
                        "gt_spells": float(np.mean([r["decoded"]["gt_spells"] for r in sel]))}
    sel = [r for r in rows if r["sim"] in EVAL_SIMS]
    summary["gate (camera + plain)"] = {"v1": mean(sel, "v1"), "v0": mean(sel, "v0"),
                                        "gain": mean(sel, "v1") - mean(sel, "v0")}
    out = {"alpha": alpha, "rounds": rounds, "summary": summary, "gate": gate, "passes": all(gate.values()),
           "rows": rows}
    with open(os.path.join(RESULTS, "model_{}_eval.json".format(TAG)), "w") as fh:
        json.dump(out, fh, indent=1)
    for k, v in summary.items():
        print("{:<30} v1 {:.3f}  v0 {:.3f}  {:+.3f}".format(k, v["v1"], v["v0"], v["gain"]))
    print("gate", gate, "PASS" if out["passes"] else "FAIL", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=("cv", "sweep", "final"))
    args = parser.parse_args()
    {"cv": run_cv, "sweep": run_sweep, "final": run_final}[args.stage](args)


if __name__ == "__main__":
    main()
