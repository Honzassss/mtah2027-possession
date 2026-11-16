"""Model v0 on open data: 5 Hz features, three LightGBM heads and the 4-state decoder,
trained on degraded copies of the five Duesseldorf IDSSE matches, scored on the two held-out ones.

    python -I analysis/model_v0.py build [--workers 5]   # degraded copies and their features, cached
    python -I analysis/model_v0.py cv                    # leave one training match out: rounds and alpha
    python -I analysis/model_v0.py sweep                 # alpha again, from the saved out-of-fold predictions
    python -I analysis/model_v0.py final                 # train on all five, score J03WMX and J03WN1
    python -I analysis/model_v0.py bundle                # the decoder file beside heads trained earlier

Training copies: clean(degrade(load_idsse(p), seed, SIMULATORS["camera"])), seeds 1-3, live share drawn
per copy. Held-out copies: J03WMX and J03WN1 under "camera" and "plain" at live share 0.1, 0.3 and 0.5,
seeds 1-3, scored with analysis/spellmetric.py; "camera-parallax" is reported beside them, outside the
gate. The decoder's one free number, alpha, is chosen on the out-of-fold predictions of the training
matches only, and the boosting rounds are the median early-stopping round of those folds; the held-out
matches choose nothing. `final` saves the heads and, beside them, the decoder bundle that
scripts/predict_match.py runs on (pipeline/predict.py); `bundle` writes that file for heads trained
before `final` did, after checking that the training copies give the table those heads were scored with.

IDSSE only, through degrade(), which refuses unlabelled data, so nothing here can run on the
competition's match. The cache and the results stay in results/ (gitignored).
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis.spellmetric import macro_f1, spell_f1  # noqa: E402
from pipeline.clean import clean  # noqa: E402
from pipeline.decode import decode, labels_of, transitions  # noqa: E402
from pipeline.degrade import SIMULATORS, degrade  # noqa: E402
from pipeline.features import Steps, dead_steps, features, step_index  # noqa: E402
from pipeline.idsse import load_idsse  # noqa: E402
from pipeline.model import HEADS, fit, label_codes, predict, step_targets  # noqa: E402
from pipeline.predict import write_bundle  # noqa: E402

TRAIN = ("J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
HELD_OUT = ("J03WMX", "J03WN1")
SEEDS = (1, 2, 3)
EVAL_SIMS = ("camera", "plain")
EXTRA_SIMS = ("camera-parallax",)   # reported beside the gate, not part of it: "camera" draws parallax per
                                    # (seed, match), and seeds 1-3 give it to only 3 of the 18 held-out copies
EVAL_LIVE = (0.1, 0.3, 0.5)
ALPHAS = (0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0)   # the first run peaked at its edge, 1.0
IDSSE = os.path.join(ROOT, "external", "processed", "idsse")
CACHE = os.path.join(ROOT, "results", "cache", "v0")
MODELS = os.path.join(ROOT, "results", "models")
TAG = "v0"


def train_keys():
    return [(m, "camera", None, s) for m in TRAIN for s in SEEDS]


def eval_keys(sims=EVAL_SIMS + EXTRA_SIMS):
    return [(m, sim, live, s) for m in HELD_OUT for sim in sims for live in EVAL_LIVE for s in SEEDS]


def key_name(key):
    match, sim, live, seed = key
    return "{}_{}_{}_{}".format(match, sim, "drawn" if live is None else "{:g}".format(live), seed)


def cache_path(key):
    return os.path.join(CACHE, key_name(key) + ".npz")


def build_one(key):
    path = cache_path(key)
    if os.path.isfile(path):
        return key_name(key), 0.0, None
    started = time.time()
    match, sim, live, seed = key
    cfg = SIMULATORS[sim] if live is None else replace(SIMULATORS[sim], live_share=live)
    copy = degrade(load_idsse(os.path.join(IDSSE, match + ".npz")), seed, cfg)
    stats = copy.extras["degrade"]
    cleaned = clean(copy)
    steps = step_index(cleaned)
    names, X = features(cleaned, steps)
    _, Xs = features(cleaned, steps, swap=True)
    codes = label_codes(cleaned.labels)
    info = {"missing_share": stats["missing_share"], "live_share_of_missing": stats["live_share_of_missing"],
            "parallax": stats["parameters"]["parallax"], "live_share_param": stats["parameters"]["live_share"]}
    tmp = path + ".tmp.npz"
    np.savez(tmp, X=X, Xs=Xs, names=np.array(names), codes=codes, step_codes=codes[steps.centre],
             start=steps.start, stop=steps.stop, centre=steps.centre, half=steps.half,
             size=np.int64(steps.size), fps=np.float64(steps.fps), dead=dead_steps(cleaned, steps),
             info=json.dumps(info))
    os.replace(tmp, path)
    return key_name(key), time.time() - started, info


def load(key):
    with np.load(cache_path(key), allow_pickle=False) as data:
        out = {k: data[k] for k in data.files}
    out["steps"] = Steps(start=out["start"], stop=out["stop"], centre=out["centre"], half=out["half"],
                         size=int(out["size"]), fps=float(out["fps"]))
    out["names"] = [str(v) for v in out["names"]]
    out["info"] = json.loads(str(out["info"]))
    out["targets"] = step_targets(out["step_codes"], out["steps"])
    return out


def table_from(copies):
    return transitions([(c["targets"]["states"], c["half"]) for c in copies])


def score(copy, states):
    """spellF1 and friends of a 5 Hz state path against the copy's 25 Hz labels."""
    step_codes = np.where(states >= 2, 2, states).astype(np.int8)
    pred = copy["steps"].frame_labels(step_codes, len(copy["codes"]), np.int8(2))
    gt = copy["codes"]
    f1, n_gt, n_pred, per_iou = spell_f1(gt, pred)
    graded = gt != 3
    return {"spell_f1": f1, "gt_spells": n_gt, "pred_spells": n_pred, "f1_by_iou": per_iou,
            "macro_f1": macro_f1(gt, pred), "accuracy": float((gt[graded] == pred[graded]).mean()),
            "dead_share_pred": float((pred[graded] == 2).mean()), "dead_share_true": float((gt[graded] == 2).mean())}


def argmax_states(p_dead, p_home):
    probs = np.column_stack(((1 - p_dead) * p_home, (1 - p_dead) * (1 - p_home), p_dead))
    return np.argmax(probs, axis=1).astype(np.int8)


def head_quality(copies, probs, name):
    """Log loss and accuracy of one head on its own rows, pooled over copies."""
    ys, ps = [], []
    for c, p in zip(copies, probs):
        y, rows, _ = c["targets"][name]
        ys.append(y[rows])
        ps.append(p[name][rows])
    y, p = np.concatenate(ys).astype(float), np.clip(np.concatenate(ps), 1e-6, 1 - 1e-6)
    return {"rows": int(len(y)), "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            "accuracy": float(np.mean((p > 0.5) == (y > 0.5)))}


def team_by_ball(copy, p_home):
    """Accuracy of the team head on live steps, with and without a ball seen in the step."""
    y, rows, _ = copy["targets"]["team"]
    seen = copy["X"][:, copy["names"].index("ball_seen")] > 0
    out = {}
    for tag, mask in (("ball_seen", rows & seen), ("ball_missing", rows & ~seen)):
        out[tag] = {"steps": int(mask.sum()),
                    "accuracy": float(np.mean((p_home[mask] > 0.5) == y[mask])) if mask.any() else None}
    return out


def predict_all(boosters, copy, rounds=None):
    return {name: predict(boosters[name], HEADS[name], copy["X"], copy["Xs"],
                          None if rounds is None else rounds[name]) for name in HEADS}


def run_build(args):
    os.makedirs(CACHE, exist_ok=True)
    keys = train_keys() + eval_keys()
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for name, seconds, info in pool.map(build_one, keys):
            print("built" if info else "cached", name, "{:.1f}s".format(seconds),
                  "" if not info else "missing {:.3f} live-of-missing {:.3f} parallax {:g}".format(
                      info["missing_share"], info["live_share_of_missing"], info["parallax"]), flush=True)
    print("build done in {:.0f}s".format(time.time() - started), flush=True)


def run_cv(args):
    keys = train_keys()
    copies = {k: load(k) for k in keys}
    names = copies[keys[0]]["names"]
    folds, oof = [], {}
    for held in TRAIN:
        started = time.time()
        fit_copies = [copies[k] for k in keys if k[0] != held]
        val_copies = [copies[k] for k in keys if k[0] == held]
        parts = [(c["X"], c["Xs"], c["targets"]) for c in fit_copies]
        valid = [(c["X"], c["Xs"], c["targets"]) for c in val_copies]
        boosters, best = {}, {}
        for name, head in HEADS.items():
            boosters[name] = fit(head, parts, names, valid=valid)
            best[name] = int(boosters[name].best_iteration)
        for k in keys:
            if k[0] == held:
                oof[k] = predict_all(boosters, copies[k], best)
        folds.append({"held": held, "best_iteration": best, "seconds": time.time() - started})
        print("fold", held, best, "{:.0f}s".format(time.time() - started), flush=True)
        for k in keys:
            if k[0] == held:
                np.savez(os.path.join(CACHE, "oof_" + key_name(k) + ".npz"), **oof[k])

    quality = {name: head_quality([copies[k] for k in keys], [oof[k] for k in keys], name) for name in HEADS}
    out = {"folds": folds, "head_quality": quality,
           "rounds": {name: int(np.median([f["best_iteration"][name] for f in folds])) for name in HEADS}}
    with open(os.path.join(ROOT, "results", "model_{}_cv.json".format(TAG)), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1), flush=True)
    run_sweep(args, copies)


def run_sweep(args, copies=None):
    """Choose alpha on the saved out-of-fold predictions. Each fold decodes with the transitions of its
    own four training matches."""
    keys = train_keys()
    copies = copies or {k: load(k) for k in keys}
    oof = {}
    for k in keys:
        with np.load(os.path.join(CACHE, "oof_" + key_name(k) + ".npz")) as data:
            oof[k] = {name: data[name] for name in HEADS}
    tables = {held: table_from([copies[k] for k in keys if k[0] != held]) for held in TRAIN}
    sweep = {}
    for alpha in ALPHAS:
        rows = []
        for k in keys:
            p = oof[k]
            states = decode(p["dead"], p["team"], p["restart"], copies[k]["steps"], tables[k[0]], alpha,
                            copies[k]["dead"])
            rows.append(score(copies[k], states))
        sweep["{:g}".format(alpha)] = {"spell_f1": float(np.mean([r["spell_f1"] for r in rows])),
                                       "spell_f1_by_match": {m: float(np.mean([r["spell_f1"] for kk, r in zip(keys, rows) if kk[0] == m]))
                                                             for m in TRAIN},
                                       "macro_f1": float(np.mean([r["macro_f1"] for r in rows])),
                                       "pred_spells": float(np.mean([r["pred_spells"] for r in rows])),
                                       "gt_spells": float(np.mean([r["gt_spells"] for r in rows]))}
        print("alpha", alpha, {k: v for k, v in sweep["{:g}".format(alpha)].items() if k != "spell_f1_by_match"},
              flush=True)
    plain = [score(copies[k], argmax_states(oof[k]["dead"], oof[k]["team"])) for k in keys]
    chosen = max(sweep, key=lambda a: sweep[a]["spell_f1"])
    path = os.path.join(ROOT, "results", "model_{}_cv.json".format(TAG))
    with open(path) as fh:
        out = json.load(fh)
    out.update({"alpha_sweep": sweep, "alpha": float(chosen), "alpha_at_edge": chosen in ("{:g}".format(ALPHAS[0]), "{:g}".format(ALPHAS[-1])),
                "argmax": {"spell_f1": float(np.mean([r["spell_f1"] for r in plain])),
                           "macro_f1": float(np.mean([r["macro_f1"] for r in plain])),
                           "pred_spells": float(np.mean([r["pred_spells"] for r in plain]))}})
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print("alpha", chosen, "at the edge of the grid" if out["alpha_at_edge"] else "inside the grid",
          "argmax", out["argmax"], flush=True)


def run_final(args):
    with open(os.path.join(ROOT, "results", "model_{}_cv.json".format(TAG))) as fh:
        cv = json.load(fh)
    alpha, rounds = float(cv["alpha"]), cv["rounds"]
    train = [load(k) for k in train_keys()]
    names = train[0]["names"]
    table = table_from(train)
    parts = [(c["X"], c["Xs"], c["targets"]) for c in train]
    os.makedirs(MODELS, exist_ok=True)
    boosters = {}
    for name, head in HEADS.items():
        started = time.time()
        boosters[name] = fit(head, parts, names, rounds=rounds[name])
        boosters[name].save_model(os.path.join(MODELS, "{}_{}.txt".format(TAG, name)))
        print("trained", name, rounds[name], "{:.0f}s".format(time.time() - started), flush=True)
    print("wrote", write_bundle(MODELS, TAG, alpha, rounds, table, names), flush=True)
    del parts, train

    rows = []
    for key in eval_keys():
        copy = load(key)
        p = predict_all(boosters, copy)
        states = decode(p["dead"], p["team"], p["restart"], copy["steps"], table, alpha, copy["dead"])
        decoded = score(copy, states)
        plain = score(copy, argmax_states(p["dead"], p["team"]))
        match, sim, live, seed = key
        rows.append({"match": match, "sim": sim, "live": live, "seed": seed, "info": copy["info"],
                     "decoded": decoded, "argmax": {k: plain[k] for k in ("spell_f1", "pred_spells", "macro_f1")},
                     "head_quality": {name: head_quality([copy], [p], name) for name in HEADS},
                     "team_by_ball": team_by_ball(copy, p["team"])})
        print("{:<28} spellF1 {:.3f} (argmax {:.3f})  spells {}/{}  macroF1 {:.3f}  D {:.3f}/{:.3f}".format(
            key_name(key), decoded["spell_f1"], plain["spell_f1"], decoded["pred_spells"], decoded["gt_spells"],
            decoded["macro_f1"], decoded["dead_share_pred"], decoded["dead_share_true"]), flush=True)

    summary = {}
    for sim in EVAL_SIMS + EXTRA_SIMS:
        for live in EVAL_LIVE:
            for match in HELD_OUT:
                sel = [r["decoded"]["spell_f1"] for r in rows if (r["sim"], r["live"], r["match"]) == (sim, live, match)]
                summary["{} live {:g} {}".format(sim, live, match)] = float(np.mean(sel))
            sel = [r["decoded"]["spell_f1"] for r in rows if (r["sim"], r["live"]) == (sim, live)]
            summary["{} live {:g}".format(sim, live)] = float(np.mean(sel))
        summary[sim] = float(np.mean([r["decoded"]["spell_f1"] for r in rows if r["sim"] == sim]))
        summary[sim + " argmax"] = float(np.mean([r["argmax"]["spell_f1"] for r in rows if r["sim"] == sim]))
        for tag in ("ball_seen", "ball_missing"):
            acc = [r["team_by_ball"][tag]["accuracy"] for r in rows if r["sim"] == sim]
            summary["{} team accuracy, {}".format(sim, tag)] = float(np.mean(acc))
    gate = [r for r in rows if r["sim"] in EVAL_SIMS]
    summary["gate (camera + plain)"] = float(np.mean([r["decoded"]["spell_f1"] for r in gate]))
    summary["gate argmax"] = float(np.mean([r["argmax"]["spell_f1"] for r in gate]))
    out = {"alpha": alpha, "rounds": rounds, "transitions": table.matrix.tolist(),
           "log_prior": table.log_prior.tolist(), "summary": summary, "rows": rows}
    with open(os.path.join(ROOT, "results", "model_{}_eval.json".format(TAG)), "w") as fh:
        json.dump(out, fh, indent=1)
    for k, v in summary.items():
        print("{:<32} {:.3f}".format(k, v))


def run_bundle(args):
    """The decoder bundle for heads saved before `final` wrote it. Refuses when the training copies give
    another table, alpha or rounds than the ones the saved heads were scored with."""
    with open(os.path.join(ROOT, "results", "model_{}_cv.json".format(TAG))) as fh:
        cv = json.load(fh)
    with open(os.path.join(ROOT, "results", "model_{}_eval.json".format(TAG))) as fh:
        scored = json.load(fh)
    train = [load(k) for k in train_keys()]
    table = table_from(train)
    if table.matrix.tolist() != scored["transitions"] or table.log_prior.tolist() != scored["log_prior"]:
        raise SystemExit("the training copies give another decoder table than the scored one")
    if float(cv["alpha"]) != scored["alpha"] or cv["rounds"] != scored["rounds"]:
        raise SystemExit("alpha or rounds differ between the cv and the scored run")
    print("wrote", write_bundle(MODELS, TAG, cv["alpha"], cv["rounds"], table, train[0]["names"]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=("build", "cv", "sweep", "final", "bundle"))
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    {"build": run_build, "cv": run_cv, "sweep": run_sweep, "final": run_final,
     "bundle": run_bundle}[args.stage](args)


if __name__ == "__main__":
    main()
