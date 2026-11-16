"""Add-ons against v3: what is left after v3, out of fold on the 15 training copies, and the best
add-on once through the gate, all under convention T = 1.

    python -I analysis/addons_v3.py decoder   # decoder settings, oracles and the diagnosis, on the saved OOF heads
    python -I analysis/addons_v3.py team      # the team head retrained out of fold, and a stacked team head
    python -I analysis/addons_v3.py gate      # the stacked head through the gate against v3

Result (12 Nov 2026): no add-on clears the gate, so v3 stays and nothing here goes into the operator path.

What is left after v3. Out of fold, v3 finds (IoU 0.5) 17 % of the true possessions of 1-2 s that follow a
turnover and 40 % of those of 2-4 s, and such runs are a third of all spells (1,734 of 5,295). Fed true
labels in place of one head, v3's decoder scores 0.9424 with a perfect team head on live steps and 0.7169
with a perfect dead-ball head, against 0.6672: the loss is in the team head, not in the decoder. In the
short runs v3 misses the ball is seen on two thirds of the steps, yet the team head favours the true team
on only 43 % of them (1-2 s) and 55 % (2-4 s).

Add-ons, OOF spellF1 against v3's 0.6672:
- decoder only, nothing retrained: switches between the sides cheaper by 0.25 to 3 nats, -0.0161 to
  +0.0009; the team head's logit scaled by 0.7 to 3, -0.0091 to +0.0008; the T = 1 rule applied to the
  output, +0.0014, and +0.0038 with switches 1.5 nats cheaper. The labels agree: after a turnover a switch
  comes at 0.026-0.028 per step over the next 1-4 s (never under 1 s, 0.012 after 8 s) against the
  decoder's flat 0.016, about half a nat, the size that moves nothing above.
- the team head retrained: steps within 1 s of a change weighted 8 instead of 3, +0.0043; finer trees
  (127 leaves, 100 rows a leaf), -0.0031; a stacked team head, a second LightGBM on the first team head's
  logit at 15 offsets over +-3 s, the dead head's at 5 and 14 touch and distance features, +0.0063 (team
  log loss on live steps 0.2243 -> 0.2213); the same stacked on the weight-8 head, +0.0062.
- the stacked head, the best of them, through the gate against v3 (113 rounds, alpha 1.4): 0.6561 vs
  0.6459, +0.0102; camera J03WMX +0.0111, camera J03WN1 +0.0126, plain J03WMX +0.0088, plain J03WN1
  +0.0082; 31 of 36 copies better. FAIL: half of the +0.02 asked of every cell, and within the 0.013 that
  retraining alone moves a cell.

Settings come from the training folds only: test 1's leave-one-match-out folds (analysis/test1_convention.py),
early stopping on the held match as there, alphas the OOF best of 1.0, 1.4, 2.0 and 2.8, and the stacked
head's rounds the median of its folds. `team` first re-runs v2's team head and checks that it gives the
saved OOF posteriors back exactly, so the variants differ from v3 by their settings only; `gate` first
checks that its path gives v3's stored gate rows back exactly.

Writes results/addons_v3.json, results/cache/addons and results/gate_addons_stack_T1_vs_v3.json only.
IDSSE only.
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
import pipeline.model as pm  # noqa: E402
from analysis.gate import frames_of, gate_frames, gate_keys, cells_names, read_references  # noqa: E402
from analysis.model_v0 import MODELS, TRAIN, cache_path, key_name, load, train_keys  # noqa: E402
from analysis.spellmetric import fill, overlaps, runs_of, segments, spell_f1, spells  # noqa: E402
from analysis.test1_convention import conventions_of, frame_half, oof_file, relabel, step_codes_of, table_for  # noqa: E402
from analysis.test2_dead_shift import light, probs  # noqa: E402
from pipeline.decode import A, H, decode, restarter_may_change, states_from_labels  # noqa: E402
from pipeline.predict import head_probabilities, load_model  # noqa: E402

T = 1.0
REFERENCE = "v3"
RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(RESULTS, "addons_v3.json")
CACHE = os.path.join(RESULTS, "cache", "addons")
GATE_OUT = os.path.join(RESULTS, "gate_addons_stack_T1_vs_v3.json")
WORKERS = 8

SWITCH_NATS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0)     # H <-> A made this much cheaper
TEAM_SCALES = (0.7, 1.25, 1.5, 2.0, 2.5, 3.0)                 # the team head's logit times this
RULE_SWITCH_NATS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)             # the T = 1 rule on the output, with cheaper switches
ALPHAS = (1.0, 1.4, 2.0, 2.8)
LENGTHS_S = (0, 1, 2, 4, 8, 16)                               # bins of true run length, the last one open
AGES_S = (0.2, 1, 2, 4, 8)                                    # bins of the age of a live run, the last one open

HEAD_VARIANTS = {                                             # the team head retrained; "v2" is v2's own settings
    "v2": {},
    "weight8": {"change_weight": 8.0},                        # steps within 1 s of a change weigh 8, not 3
    "leaves127": {"params": {"num_leaves": 127, "min_data_in_leaf": 100}},
}
OFFSETS = (-15, -10, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 10, 15)    # steps of 0.2 s, for the team posterior
DEAD_OFFSETS = (-5, -2, 0, 2, 5)
RAW = ("since_touch_diff", "until_touch_diff", "since_touch_h", "since_touch_a", "until_touch_h", "until_touch_a",
       "ball_dd", "dd_past1", "dd_next1", "ball_d1_h", "ball_d1_a", "ball_seen", "since_seen", "until_seen")
STACK_NAMES = (["team{:+d}".format(o) for o in OFFSETS] + ["dead{:+d}".format(o) for o in DEAD_OFFSETS] + list(RAW))
STACKS = {"stack": "v2", "stack_on_weight8": "weight8"}       # stacked head -> the first head it reads


# ---------------------------------------------------------------- shared

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


def fold_tables(copies):
    """v3's decoder table for each training fold: counted on the other four matches, restarter may change."""
    keys = list(copies)
    return {m: restarter_may_change(table_for([copies[k] for k in keys if k[0] != m], T)) for m in TRAIN}


def frames(c, states):
    return frames_of(c["steps"], states, len(c["codes"]))


def logit(p, eps=1e-4):
    q = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return np.log(q) - np.log1p(-q)


def by_match(values, keys):
    return {m: float(np.mean([v for v, k in zip(values, keys) if k[0] == m])) for m in TRAIN}


# ---------------------------------------------------------------- decoder

def cheaper_switch(table, nats):
    matrix = table.matrix.copy()
    matrix[H, A] += int(round(nats * 1000))
    matrix[A, H] += int(round(nats * 1000))
    return replace(table, matrix=matrix)


def scaled(p, scale):
    return p if scale == 1.0 else 1.0 / (1.0 + np.exp(-scale * logit(p, 1e-6)))


def decoder_variants():
    out = [("v3", 0.0, 1.0, False)]
    out += [("switch -{:g} nats".format(n), n, 1.0, False) for n in SWITCH_NATS]
    out += [("team logit x{:g}".format(s), 0.0, s, False) for s in TEAM_SCALES]
    out += [("T1 rule, switch -{:g} nats".format(n), n, 1.0, True) for n in RULE_SWITCH_NATS]
    return out


def run_types(codes, runs):
    """For each H/A spell: 'switch' when it follows a live spell of the other side, else 'restart'."""
    segs = segments(codes, runs)
    out = []
    for i, (a, b, c) in enumerate(segs):
        if c in (H, A):
            prev = segs[i - 1] if i and segs[i - 1][1] == a else None
            out.append("switch" if prev is not None and prev[2] in (H, A) else "restart")
    return out


def matched(gt, pred, thr=0.5):
    """True and predicted spells, their types, and which of them the metric's greedy matching pairs at thr."""
    runs = runs_of(gt)
    f = fill(pred, runs)
    gs, ps = spells(gt, runs), spells(f, runs)
    mg, mp = set(), set()
    for iou, i, j in overlaps(gs, ps):
        if iou < thr:
            break
        if i not in mg and j not in mp:
            mg.add(i)
            mp.add(j)
    return gs, run_types(gt, runs), mg, ps, run_types(f, runs), mp


def decoder_job(job):
    key, table, alpha, variants = job
    c = light(key)
    c["conv"] = conventions_of(c)
    gt = c["conv"][T]
    p = probs(oof_file(key, T))
    scores = {}
    for name, nats, scale, rule in variants:
        st = decode(p["dead"], scaled(p["team"], scale), p["restart"], c["steps"], cheaper_switch(table, nats),
                    alpha, c["dead"])
        pred = frames(c, st)
        if rule:
            pred = relabel(pred, frame_half(c["steps"], len(gt)), c["steps"].fps, T)
        f1, n_gt, n_pred, _ = spell_f1(gt, pred)
        scores[name] = (f1, n_gt, n_pred)
        if name == "v3":
            v3_pred = pred

    # v3's true and predicted runs, and what the team head says inside the true ones
    gs, gtypes, mg, ps, ptypes, mp = matched(gt, v3_pred)
    fps, centre = c["steps"].fps, c["steps"].centre
    true_runs = []
    for i, (a, b, team) in enumerate(gs):
        ss = np.arange(np.searchsorted(centre, a), np.searchsorted(centre, b))
        p_true = p["team"][ss] if team == H else 1.0 - p["team"][ss]
        true_runs.append({"s": (b - a) / fps, "type": gtypes[i], "found": i in mg,
                          "favour": float(np.mean(p_true > 0.5)) if len(ss) else None,
                          "seen": float(np.mean(c["seen"][ss])) if len(ss) else None})
    pred_runs = [{"s": (b - a) / fps, "type": ptypes[j], "matched": j in mp} for j, (a, b, _) in enumerate(ps)]

    # oracles: v3's decoder fed true labels in place of one head
    sc = gt[centre]
    true_dead = np.where(sc >= 2, 0.98, 0.02)
    true_home = np.where(sc == H, 0.98, 0.02)
    true_restart = np.where(states_from_labels(sc, c["steps"].half) == 2, 0.98, 0.02)
    oracles = {}
    for name, (d, t, r) in {"dead": (true_dead, p["team"], p["restart"]),
                            "dead and restart": (true_dead, p["team"], np.where(sc >= 2, true_restart, p["restart"])),
                            "team on live steps": (p["dead"], np.where(sc < 2, true_home, p["team"]), p["restart"])}.items():
        oracles[name] = spell_f1(gt, frames(c, decode(d, t, r, c["steps"], table, alpha, c["dead"])))[0]
    return {"key": key_name(key), "match": key[0], "scores": scores, "true_runs": true_runs,
            "pred_runs": pred_runs, "oracles": oracles}


def length_bins(runs, flag):
    out = {}
    for typ in ("switch", "restart"):
        for i, lo in enumerate(LENGTHS_S):
            hi = LENGTHS_S[i + 1] if i + 1 < len(LENGTHS_S) else None
            sel = [r[flag] for r in runs if r["type"] == typ and r["s"] >= lo and (hi is None or r["s"] < hi)]
            out["{} {}-{} s".format(typ, lo, hi if hi else "")] = {"n": len(sel), "share": float(np.mean(sel)) if sel else None}
    return out


def hazard_by_age(copies):
    """Per-step rate of a switch (live play to the other side), by the age of the live run and by what
    started it, on the T = 1 labels of one copy per training match (the seeds share their labels)."""
    counts = {}
    for c in copies:
        sc = c["conv"][T][c["steps"].centre]
        for a, b in c["steps"].bounds():
            s = sc[a:b]
            age, after = 0, None
            for t in range(1, len(s)):
                if s[t - 1] < 2 and after is not None:
                    age += 1
                    n, k = counts.get((after, age), (0, 0))
                    counts[(after, age)] = (n + 1, k + int(s[t] < 2 and s[t] != s[t - 1]))
                if s[t] != s[t - 1]:
                    after = None if s[t] >= 2 else ("switch" if s[t - 1] < 2 else "restart")
                    age = 0
    out = {}
    for after in ("switch", "restart"):
        for i, lo in enumerate(AGES_S):
            hi = AGES_S[i + 1] if i + 1 < len(AGES_S) else None
            sel = [v for (aft, age), v in counts.items()
                   if aft == after and age / 5.0 >= lo - 1e-9 and (hi is None or age / 5.0 < hi - 1e-9)]
            n, k = sum(v[0] for v in sel), sum(v[1] for v in sel)
            out["after a {}, {:g}-{} s".format(after, lo, "{:g}".format(hi) if hi else "")] = {"steps": n, "rate": k / max(n, 1)}
    return out


def run_decoder(args):
    keys = train_keys()
    copies = {k: light(k) for k in keys}
    for k in keys:
        copies[k]["conv"] = conventions_of(copies[k])
    tables = fold_tables(copies)
    alpha = load_model(MODELS, REFERENCE).alpha
    variants = decoder_variants()
    started = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        res = list(ex.map(decoder_job, [(k, tables[k[0]], alpha, variants) for k in keys]))
    base = float(np.mean([r["scores"]["v3"][0] for r in res]))
    rows = {}
    for name, *_ in variants:
        f1 = [r["scores"][name][0] for r in res]
        rows[name] = {"oof": float(np.mean(f1)), "gain": float(np.mean(f1)) - base, "by_match": by_match(f1, keys),
                      "pred_spells": float(np.mean([r["scores"][name][2] for r in res])),
                      "gt_spells": float(np.mean([r["scores"][name][1] for r in res]))}
    true_runs = [x for r in res for x in r["true_runs"]]
    pred_runs = [x for r in res for x in r["pred_runs"]]
    inside = {}
    for lo, hi in ((1, 2), (2, 4), (4, 8)):
        for found in (True, False):
            sel = [x for x in true_runs if x["type"] == "switch" and lo <= x["s"] < hi and x["found"] == found
                   and x["favour"] is not None]
            inside["switch {}-{} s, {}".format(lo, hi, "found" if found else "missed")] = {
                "n": len(sel), "favour": float(np.mean([x["favour"] for x in sel])),
                "favour_over_half": float(np.mean([x["favour"] > 0.5 for x in sel])),
                "seen": float(np.mean([x["seen"] for x in sel]))}
    flat = float(np.exp(tables[TRAIN[0]].matrix[H, A] / 1000.0))
    out = {"T": T, "alpha": alpha, "variants": rows,
           "oracles": {n: float(np.mean([r["oracles"][n] for r in res])) for n in res[0]["oracles"]},
           "true_runs_found": length_bins(true_runs, "found"), "pred_runs_matched": length_bins(pred_runs, "matched"),
           "team_head_inside_true_switch_runs": inside,
           "switch_rate_by_age": hazard_by_age([copies[k] for k in keys if k[3] == 1]),
           "decoder_switch_rate": {m: float(np.exp(tables[m].matrix[H, A] / 1000.0)) for m in TRAIN},
           "seconds": time.time() - started}
    save({"decoder": out})

    print("OOF spellF1 on the 15 training copies, T = 1, against v3 {:.4f} (decoded in {:.0f} s)".format(base, out["seconds"]))
    for name, r in rows.items():
        print("  {:<28} {:.4f} {:+.4f}  {}  spells {:.0f}/{:.0f}".format(
            name, r["oof"], r["gain"], " ".join("{:.3f}".format(v) for v in r["by_match"].values()),
            r["pred_spells"], r["gt_spells"]))
    print("oracles, v3's decoder fed true labels for:", ", ".join("{} {:.4f}".format(n, v) for n, v in out["oracles"].items()))
    print("v3: true runs found at IoU 0.5 | predicted runs matched")
    for (name, f), (_, m) in zip(out["true_runs_found"].items(), out["pred_runs_matched"].items()):
        print("  {:<18} {:>5} found {}   {:>5} matched {}".format(
            name, f["n"], "-" if f["share"] is None else "{:.2f}".format(f["share"]), m["n"],
            "-" if m["share"] is None else "{:.2f}".format(m["share"])))
    print("team head inside true runs after a switch (share of steps favouring the true team; ball seen):")
    for name, v in inside.items():
        print("  {:<26} n {:4d}  favour {:.2f}  favour on most steps {:.2f}  seen {:.2f}".format(
            name, v["n"], v["favour"], v["favour_over_half"], v["seen"]))
    print("switch rate per step by the age of the live run (labels), against the decoder's flat {:.4f}:".format(flat))
    for name, v in out["switch_rate_by_age"].items():
        print("  {:<26} {:.4f}  ({} steps)".format(name, v["rate"], v["steps"]))


# ---------------------------------------------------------------- team

@contextlib.contextmanager
def lightgbm_params(overrides):
    saved = pm.PARAMS
    pm.PARAMS = dict(saved, **overrides)
    try:
        yield
    finally:
        pm.PARAMS = saved


def team_targets(c, change_weight=None):
    """The heads' targets under T = 1; change_weight replaces the weight of steps near a change (3)."""
    targets = pm.step_targets(step_codes_of(c, c["conv"][T]), c["steps"])
    if change_weight is not None:
        y, rows, w = targets["team"]
        targets = dict(targets, team=(y, rows, np.where(w > 1.0, change_weight, 1.0).astype(np.float32)))
    return targets


def shifted(values, steps, offset):
    """values[t + offset] inside each half, held at the half's first and last step."""
    out = np.empty_like(values)
    for a, b in steps.bounds():
        out[a:b] = values[np.clip(np.arange(a, b) + offset, a, b - 1)]
    return out


def stack_views(steps, names, X, Xs, p):
    """The stacked head's features in both team views: the first team head's logit at OFFSETS (negated
    in the swapped view, as the head flips), the dead head's logit at DEAD_OFFSETS, and RAW from X or Xs."""
    team, dead = logit(p["team"]), logit(p["dead"])
    cols = [names.index(n) for n in RAW]
    views = []
    for sign, Z in ((1.0, X), (-1.0, Xs)):
        parts = ([shifted(sign * team, steps, o) for o in OFFSETS] + [shifted(dead, steps, o) for o in DEAD_OFFSETS]
                 + [Z[:, i].astype(np.float64) for i in cols])
        views.append(np.column_stack(parts).astype(np.float32))
    return views


def oof_head(views, targets, names, out_dir):
    """The team head out of fold on test 1's folds; posteriors to out_dir, early-stopping round per fold."""
    os.makedirs(out_dir, exist_ok=True)
    keys = list(views)
    rounds = {}
    for held in TRAIN:
        started = time.time()
        parts = [(views[k][0], views[k][1], targets[k]) for k in keys if k[0] != held]
        valid = [(views[k][0], views[k][1], targets[k]) for k in keys if k[0] == held]
        booster = pm.fit(pm.HEADS["team"], parts, names, valid=valid)
        rounds[held] = int(booster.best_iteration)
        for k in keys:
            if k[0] == held:
                p = pm.predict(booster, pm.HEADS["team"], views[k][0], views[k][1], rounds[held])
                np.save(os.path.join(out_dir, key_name(k) + ".npy"), p)
        print("  {} fold {}: {} rounds, {:.0f} s".format(os.path.basename(out_dir), held, rounds[held],
                                                        time.time() - started), flush=True)
    return rounds


def team_posteriors(name, keys):
    return {k: np.load(os.path.join(CACHE, name, key_name(k) + ".npy")) for k in keys}


def oof_scores(copies, tables, saved, team):
    """OOF spellF1 at each alpha with the saved dead and restart heads and the given team posteriors."""
    keys = list(copies)
    out = {}
    for alpha in ALPHAS:
        f1, n_pred = [], []
        for k in keys:
            c, p = copies[k], saved[k]
            st = decode(p["dead"], team[k], p["restart"], c["steps"], tables[k[0]], alpha, c["dead"])
            s = spell_f1(c["conv"][T], frames(c, st))
            f1.append(s[0])
            n_pred.append(s[2])
        out["{:g}".format(alpha)] = {"oof": float(np.mean(f1)), "by_match": by_match(f1, keys),
                                     "pred_spells": float(np.mean(n_pred))}
    return out


def log_loss(copies, targets, team):
    out = []
    for k in copies:
        y, rows, _ = targets[k]["team"]
        q = np.clip(team[k][rows], 1e-6, 1 - 1e-6)
        out.append(float(-np.mean(np.where(y[rows], np.log(q), np.log1p(-q)))))
    return float(np.mean(out))


def run_team(args):
    keys = train_keys()
    copies = {k: load(k) for k in keys}
    names = copies[keys[0]]["names"]
    for k in keys:
        copies[k]["conv"] = conventions_of(copies[k])
    saved = {k: probs(oof_file(k, T)) for k in keys}
    targets = {k: team_targets(copies[k]) for k in keys}
    heads = {}
    for name, v in HEAD_VARIANTS.items():
        tg = targets if "change_weight" not in v else {k: team_targets(copies[k], v["change_weight"]) for k in keys}
        with lightgbm_params(v.get("params", {})):
            rounds = oof_head({k: (copies[k]["X"], copies[k]["Xs"]) for k in keys}, tg, names, os.path.join(CACHE, name))
        heads[name] = {"settings": v, "rounds": rounds}
        if name == "v2":
            got = team_posteriors(name, keys)
            if any(not np.array_equal(got[k], saved[k]["team"]) for k in keys):
                raise SystemExit("v2's team head re-run out of fold differs from the saved OOF posteriors")
            print("  v2's team head re-run gives the saved OOF posteriors back exactly", flush=True)
    for name, first in STACKS.items():
        first_team = {k: saved[k]["team"] for k in keys} if first == "v2" else team_posteriors(first, keys)
        views = {k: stack_views(copies[k]["steps"], names, copies[k]["X"], copies[k]["Xs"],
                                dict(saved[k], team=first_team[k])) for k in keys}
        rounds = oof_head(views, targets, STACK_NAMES, os.path.join(CACHE, name))
        heads[name] = {"settings": {"first_head": first, "offsets": OFFSETS, "dead_offsets": DEAD_OFFSETS, "raw": RAW},
                       "rounds": rounds}
        del views
    for k in keys:
        del copies[k]["X"], copies[k]["Xs"]
    tables = fold_tables(copies)
    base = None
    print("OOF spellF1 on the 15 training copies, T = 1; v3 is v2's team head at alpha 1.4")
    for name in heads:
        team = team_posteriors(name, keys)
        heads[name]["scores"] = oof_scores(copies, tables, saved, team)
        heads[name]["team_log_loss"] = log_loss(copies, targets, team)
        if base is None:
            base = heads[name]["scores"]["1.4"]["oof"]
        best = max(heads[name]["scores"], key=lambda a: heads[name]["scores"][a]["oof"])
        heads[name]["best_alpha"] = float(best)
        for a, s in heads[name]["scores"].items():
            print("  {:<17} alpha {:<4} {:.4f} {:+.4f}  {}  spells {:.0f}{}".format(
                name, a, s["oof"], s["oof"] - base, " ".join("{:.3f}".format(v) for v in s["by_match"].values()),
                s["pred_spells"], "  <- best" if a == best else ""))
        print("  {:<17} team log loss on live steps {:.4f}".format(name, heads[name]["team_log_loss"]), flush=True)
    save({"team": {"T": T, "v3_oof": base, "heads": heads}})


# ---------------------------------------------------------------- gate

def run_gate(args):
    team = read_out().get("team")
    if not team:
        raise SystemExit("run `team` first")
    entry = team["heads"]["stack"]
    rounds = int(np.median(list(entry["rounds"].values())))
    alpha = entry["best_alpha"]
    keys = train_keys()
    copies = {k: load(k) for k in keys}
    names = copies[keys[0]]["names"]
    parts = []
    for k in keys:
        c = copies.pop(k)
        c["conv"] = conventions_of(c)
        S, Ss = stack_views(c["steps"], names, c["X"], c["Xs"], probs(oof_file(k, T)))
        parts.append((S, Ss, team_targets(c)))
    booster = pm.fit(pm.HEADS["team"], parts, STACK_NAMES, rounds=rounds)
    del parts
    v3 = load_model(MODELS, REFERENCE)
    reference, candidate = {}, {}
    for key in gate_keys():
        with np.load(cache_path(key), allow_pickle=False) as d:
            X, Xs = d["X"], d["Xs"]
            if [str(v) for v in d["names"]] != names:
                raise SystemExit("the gate copies have other features than the training copies")
        c = light(key)
        p = head_probabilities(v3, X, Xs)
        reference[key] = frames(c, decode(p["dead"], p["team"], p["restart"], c["steps"], v3.table, v3.alpha, c["dead"]))
        S, Ss = stack_views(c["steps"], names, X, Xs, p)
        stacked = pm.predict(booster, pm.HEADS["team"], S, Ss)
        candidate[key] = frames(c, decode(p["dead"], stacked, p["restart"], c["steps"], v3.table, alpha, c["dead"]))
    refs = read_references()
    check = gate_frames(reference, T, REFERENCE, refs)
    stored = refs["rows"][REFERENCE]
    bad = [name for name, row in check["rows"].items() if row["T1"] != stored[name]["T1"]]
    if bad:
        raise SystemExit("this path does not give v3's stored gate rows back on " + ", ".join(bad))
    result = gate_frames(candidate, T, REFERENCE, refs)
    result["candidate"]["settings"] = {"stacked team head": entry["settings"], "rounds": rounds, "alpha": alpha,
                                      "first heads": "v3's (v2's)", "decoder table": "v3's"}
    with open(GATE_OUT + ".tmp", "w") as fh:
        json.dump(result, fh, indent=1)
    os.replace(GATE_OUT + ".tmp", GATE_OUT)
    cmp = result["compare"]
    print("v3 through this path equals its stored gate rows on all 36 copies")
    print("truth T1: stacked team head ({} rounds, alpha {:g}) gate {:.4f} vs v3 {:.4f} ({:+.4f})".format(
        rounds, alpha, result["candidate"]["gate"], result["reference_cells"]["gate"], cmp["gate_gain"]))
    for c in cells_names():
        print("  {:<16} {:.4f} vs {:.4f}  {:+.4f}".format(c, result["candidate"][c], result["reference_cells"][c],
                                                        cmp["gains"][c]))
    better = sum(result["rows"][n]["T1"]["spell_f1"] > stored[n]["T1"]["spell_f1"] for n in result["rows"])
    print("better on {} of 36 copies".format(better))
    print("PASS" if cmp["passes"] else "FAIL", "(+0.02 needed in every cell) ->", GATE_OUT, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=("decoder", "team", "gate"))
    args = parser.parse_args()
    {"decoder": run_decoder, "team": run_team, "gate": run_gate}[args.stage](args)


if __name__ == "__main__":
    main()
