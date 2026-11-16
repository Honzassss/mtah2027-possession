"""Model v3, the published one: v2's heads and alpha, with a decoder that may change the side to restart
during a stoppage (pipeline/decode.py restarter_may_change), on the labels of convention T = 1.

    python -I analysis/model_v3.py oof       # the 15 training copies out of fold: v2's decoder against v3's
    python -I analysis/model_v3.py bundle    # results/models/v3_decoder.json, using v2's head files
    python -I analysis/gate.py check --tag v3 --T 1 --reference v2

Why: a dead spell keeps the restarter it starts with, so when the restart head changes its mind during
a stoppage, v2's decoder can only switch sides through a live step. Each such step is a spell of 0.2 s
that the metric counts and never matches. Out of fold on the training copies, v2 decodes 152 one-step
live spells between two dead ones and the truth has none. Allowing DH <-> DA removes them. Its cost is flat
from 0 to 3 nats on the training folds, so it is the cost of staying dead, and alpha stays v2's 1.4:
nothing new is fitted, and nothing at all on the held-out matches. The gate missed +0.02 in one cell
by 0.0009 (camera J03WN1 +0.0191, the others +0.0206 to +0.0220, all 36 copies better); v3 ships
anyway, Honza's decision of 11 Nov 2026.

Writes results/models/v3_decoder.json and results/model_v3.json only. IDSSE only.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis.model_v0 import MODELS, TRAIN, key_name, score, train_keys  # noqa: E402
from analysis.model_v2 import TAG as V2  # noqa: E402
from analysis.test1_convention import conventions_of, oof_file, table_for, tag  # noqa: E402
from analysis.test2_dead_shift import light, probs  # noqa: E402
from pipeline.decode import decode, restarter_may_change  # noqa: E402
from pipeline.predict import BUNDLE, load_model, sha256_file, write_bundle  # noqa: E402

T = 1.0
TAG = "v3"
OUT = os.path.join(ROOT, "results", "model_v3.json")


def one_step_restarts(steps, states):
    """Live runs of a single step with dead steps on both sides, inside the halves."""
    live = np.asarray(states) < 2
    n = 0
    for a, b in steps.bounds():
        x = live[a:b]
        n += int(np.count_nonzero(x[1:-1] & ~x[:-2] & ~x[2:]))
    return n


def run_oof(args):
    keys = train_keys()
    copies = {k: light(k) for k in keys}
    for k in keys:
        copies[k]["conv"] = conventions_of(copies[k])
    alpha = load_model(MODELS, V2).alpha
    rows, totals = {}, {"truth": 0, "v2": 0, "v3": 0}
    for k in keys:
        c = copies[k]
        table = table_for([copies[kk] for kk in keys if kk[0] != k[0]], T)
        p = probs(oof_file(k, T))
        truth = {"steps": c["steps"], "codes": c["conv"][T]}
        rows[key_name(k)] = {}
        for name, tb in (("v2", table), ("v3", restarter_may_change(table))):
            st = decode(p["dead"], p["team"], p["restart"], c["steps"], tb, alpha, c["dead"])
            s = score(truth, st)
            rows[key_name(k)][name] = {f: s[f] for f in ("spell_f1", "pred_spells", "gt_spells")}
            totals[name] += one_step_restarts(c["steps"], st)
        codes = c["conv"][T][c["steps"].centre]
        totals["truth"] += one_step_restarts(c["steps"], np.where(codes >= 2, 2, codes))
    mean = {name: float(np.mean([r[name]["spell_f1"] for r in rows.values()])) for name in ("v2", "v3")}
    out = {"T": tag(T), "alpha": alpha, "oof_spell_f1": mean, "one_step_live_between_dead": totals,
           "rows": rows, "by_match": {m: {name: float(np.mean([rows[key_name(k)][name]["spell_f1"]
                                                               for k in keys if k[0] == m]))
                                          for name in ("v2", "v3")} for m in TRAIN}}
    save({"oof": out})
    print("OOF spellF1 under {} at alpha {:g}: v2 {:.4f}, v3 {:.4f} ({:+.4f})".format(
        tag(T), alpha, mean["v2"], mean["v3"], mean["v3"] - mean["v2"]))
    print("one-step live spells between dead ones:", totals)
    for m, v in out["by_match"].items():
        print("  {}  v2 {:.4f}  v3 {:.4f}  {:+.4f}".format(m, v["v2"], v["v3"], v["v3"] - v["v2"]), flush=True)


def run_bundle(args):
    v2 = load_model(MODELS, V2)
    with open(os.path.join(MODELS, BUNDLE.format(V2))) as fh:
        rounds = {name: entry["rounds"] for name, entry in json.load(fh)["heads"].items()}
    path = write_bundle(MODELS, TAG, v2.alpha, rounds, restarter_may_change(v2.table), v2.feature_names,
                        heads_of=V2)
    v3 = load_model(MODELS, TAG)
    if v3.alpha != v2.alpha or v3.table.counts.tolist() != v2.table.counts.tolist():
        raise SystemExit("v3 is not v2's alpha and counts")
    print("wrote", path, sha256_file(path), flush=True)


def save(update):
    data = {}
    if os.path.isfile(OUT):
        with open(OUT) as fh:
            data = json.load(fh)
    data.update(update)
    with open(OUT + ".tmp", "w") as fh:
        json.dump(data, fh, indent=1)
    os.replace(OUT + ".tmp", OUT)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=("oof", "bundle"))
    args = parser.parse_args()
    {"oof": run_oof, "bundle": run_bundle}[args.stage](args)


if __name__ == "__main__":
    main()
