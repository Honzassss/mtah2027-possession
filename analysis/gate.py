"""The gate on the labels of a chosen boundary convention T (test 1).

    python -I analysis/gate.py references                                    # v0, v2, v3 -> results/gate_references.json
    python -I analysis/gate.py check --tag v4 --T 1 --reference v3 [--models results/models]

A change passes when it improves both held-out IDSSE matches (J03WMX, J03WN1) by at least GATE under
both simulators (camera, plain): four cells, each the mean spellF1 over live share 0.1, 0.3 and 0.5 and
seeds 1-3 (9 copies). The truth is relabelled to convention T with test 1's rule (a live run shorter
than T s stays with the previous team; the first run after a stoppage is kept), analysis/test1_convention
relabel(); T = 0 is the DFL label itself, the one v0 (0.556) and v1 were gated on. Which T applies and
which model is the reference follow from Honza's decision on the convention, so `check` asks for both.

A candidate is a bundle that pipeline.predict.load_model reads, run through step_states on the cached
copies in results/cache/v0 (v0's features). A change that needs other features or cleaning rebuilds the
36 copies its own way and calls gate_frames() with its 25 Hz labels per copy: the truth always comes from
the cached copies' labels, which degrade() never changes. `references` stores v0, v2 and v3 under every
tested T with the hashes of their files, and checks itself against model_v0_eval.json (v0, T = 0),
test 1's held-out rows (v2) and the gate check v3 shipped on (results/gate_v3_T1_vs_v2.json); `check` refuses a stored reference whose files have changed since.

Reads results/cache/v0 and results/models; writes results/gate_*.json only. IDSSE only.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis.model_v0 import EVAL_LIVE, EVAL_SIMS, HELD_OUT, MODELS, SEEDS, cache_path, key_name  # noqa: E402
from analysis.spellmetric import macro_f1, spell_f1  # noqa: E402
from analysis.test1_convention import CONVENTIONS, OUT as TEST1, conventions_of, tag  # noqa: E402
from analysis.test2_dead_shift import light  # noqa: E402
from pipeline.model import HEADS  # noqa: E402
from pipeline.predict import BUNDLE, load_model, sha256_file, step_states  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
REFERENCES = os.path.join(RESULTS, "gate_references.json")
GATE = 0.02
REFERENCE_TAGS = ("v0", "v2", "v3")


def gate_keys():
    return [(m, sim, live, s) for m in HELD_OUT for sim in EVAL_SIMS for live in EVAL_LIVE for s in SEEDS]


def cell_of(key):
    return "{} {}".format(key[1], key[0])


def cells_names():
    return ["{} {}".format(sim, m) for sim in EVAL_SIMS for m in HELD_OUT]


def frames_of(steps, states, n_frames):
    """25 Hz codes (0 H, 1 A, 2 D) from a 5 Hz state path, the way analysis/model_v0.score spreads them."""
    step_codes = np.where(np.asarray(states) >= 2, 2, states).astype(np.int8)
    return steps.frame_labels(step_codes, n_frames, np.int8(2))


def score_frames(gt, pred):
    f1, n_gt, n_pred, _ = spell_f1(gt, pred)
    return {"spell_f1": f1, "gt_spells": n_gt, "pred_spells": n_pred, "macro_f1": macro_f1(gt, pred)}


def bundle_frames(model_dir, tag_name, keys=None):
    """{key: 25 Hz codes} of a saved bundle on the cached gate copies."""
    model = load_model(model_dir, tag_name)
    out = {}
    for key in keys or gate_keys():
        with np.load(cache_path(key), allow_pickle=False) as d:
            X, Xs = d["X"], d["Xs"]
        c = light(key)
        out[key] = frames_of(c["steps"], step_states(model, X, Xs, c["steps"], c["dead"]), len(c["codes"]))
    return out


def evaluate(frames_by_key, Ts=CONVENTIONS):
    """{key: {T tag: scores}} against the labels of each convention."""
    rows = {}
    for key, pred in frames_by_key.items():
        c = light(key)
        if len(pred) != len(c["codes"]):
            raise ValueError("{}: {} labels for {} frames".format(key_name(key), len(pred), len(c["codes"])))
        conv = conventions_of(c)
        rows[key] = {tag(T): score_frames(conv[T], np.asarray(pred, dtype=np.int8)) for T in Ts}
    return rows


def cells(rows, T):
    """Mean spellF1 per gate cell and over all 36 copies, under convention T."""
    t = tag(T)
    keys = list(rows)
    if sorted(keys) != sorted(gate_keys()):
        raise ValueError("the gate needs exactly the 36 gate copies")
    out = {name: float(np.mean([rows[k][t]["spell_f1"] for k in keys if cell_of(k) == name])) for name in cells_names()}
    out["gate"] = float(np.mean([rows[k][t]["spell_f1"] for k in keys]))
    out["pred_spells"] = float(np.mean([rows[k][t]["pred_spells"] for k in keys]))
    out["gt_spells"] = float(np.mean([rows[k][t]["gt_spells"] for k in keys]))
    return out


def compare(candidate, reference):
    gains = {name: candidate[name] - reference[name] for name in cells_names()}
    return {"gains": gains, "gate_gain": candidate["gate"] - reference["gate"],
            "passes": all(g >= GATE for g in gains.values())}


def bundle_files(model_dir, tag_name):
    """The bundle file and the head files it names, each with its SHA-256."""
    with open(os.path.join(model_dir, BUNDLE.format(tag_name))) as fh:
        heads = json.load(fh)["heads"]
    files = [BUNDLE.format(tag_name)] + [heads[name]["file"] for name in HEADS]
    return {f: sha256_file(os.path.join(model_dir, f)) for f in files}


def read_references():
    if not os.path.isfile(REFERENCES):
        raise SystemExit("no {}: run `references` first".format(REFERENCES))
    with open(REFERENCES) as fh:
        return json.load(fh)


def gate_frames(frames_by_key, T, reference, refs=None):
    """The gate for any change: its 25 Hz labels on the 36 gate copies against the stored reference model
    under convention T. Returns the candidate's cells, the reference's and the comparison."""
    refs = refs or read_references()
    if reference not in refs["references"] or tag(T) not in refs["references"][reference]:
        raise SystemExit("no stored reference {} under {}".format(reference, tag(T)))
    files = refs["bundles"][reference]
    if bundle_files(os.path.join(ROOT, files["dir"]), reference) != files["sha256"]:
        raise SystemExit("the files of reference {} changed since `references` ran".format(reference))
    rows = evaluate(frames_by_key, (T,))
    got = cells(rows, T)
    ref = refs["references"][reference][tag(T)]
    return {"T": T, "reference": reference, "candidate": got, "reference_cells": ref, "compare": compare(got, ref),
            "rows": {key_name(k): v for k, v in rows.items()}}


def run_references(args):
    keys = gate_keys()
    out = {"definition": {"cells": cells_names(), "copies_per_cell": len(EVAL_LIVE) * len(SEEDS), "gate": GATE,
                          "conventions": [tag(T) for T in CONVENTIONS], "requested": ["T0", "T1"],
                          "note": "T0.4 and T2 are extra; T0 is the DFL label the v0 gate used"},
           "bundles": {}, "references": {}, "rows": {}}
    rows = {}
    for name in REFERENCE_TAGS:
        rows[name] = evaluate(bundle_frames(args.models, name, keys))
        out["bundles"][name] = {"dir": os.path.relpath(os.path.abspath(args.models), ROOT),
                                "sha256": bundle_files(args.models, name)}
        out["references"][name] = {tag(T): cells(rows[name], T) for T in CONVENTIONS}
        out["rows"][name] = {key_name(k): v for k, v in rows[name].items()}
        print(name, {t: round(v["gate"], 4) for t, v in out["references"][name].items()}, flush=True)

    # the stored numbers are the ones already scored: v0 under T0 is model_v0_eval.json, v2 is test 1's T1 model
    with open(os.path.join(RESULTS, "model_v0_eval.json")) as fh:
        v0 = {(r["match"], r["sim"], r["live"], r["seed"]): r["decoded"]["spell_f1"] for r in json.load(fh)["rows"]}
    bad = [key_name(k) for k in keys if rows["v0"][k]["T0"]["spell_f1"] != v0[k]]
    with open(TEST1) as fh:
        t1 = json.load(fh)["heldout"]["rows"]
    bad += [key_name(k) + " (v2)" for k in keys if rows["v2"][k] != t1[key_name(k)]["T1"]]
    with open(os.path.join(RESULTS, "gate_v3_T1_vs_v2.json")) as fh:
        g3 = json.load(fh)["rows"]
    bad += [key_name(k) + " (v3)" for k in keys if rows["v3"][k]["T1"] != g3[key_name(k)]["T1"]]
    if bad:
        raise SystemExit("references differ from the scored runs on " + ", ".join(bad))
    out["checks"] = {"v0 T0 equals model_v0_eval.json on all 36 copies": True,
                     "v2 equals test 1's held-out T1 model under every convention": True,
                     "v3 under T1 equals the gate check it shipped on": True}
    out["v2_vs_v0"] = {tag(T): compare(out["references"]["v2"][tag(T)], out["references"]["v0"][tag(T)])
                       for T in CONVENTIONS}
    out["v3_vs_v2"] = {tag(T): compare(out["references"]["v3"][tag(T)], out["references"]["v2"][tag(T)])
                       for T in CONVENTIONS}
    with open(REFERENCES + ".tmp", "w") as fh:
        json.dump(out, fh, indent=1)
    os.replace(REFERENCES + ".tmp", REFERENCES)
    for t in ("T0", "T1"):
        for name in REFERENCE_TAGS:
            r = out["references"][name][t]
            print("{} truth {}: gate {:.4f}  ".format(name, t, r["gate"])
                  + "  ".join("{} {:.4f}".format(c, r[c]) for c in cells_names())
                  + "  spells {:.0f}/{:.0f}".format(r["pred_spells"], r["gt_spells"]))
    for t, v in out["v2_vs_v0"].items():
        print("v2 vs v0, truth {}: {}  {}".format(t, "PASS" if v["passes"] else "fail",
                                                 {c: round(g, 4) for c, g in v["gains"].items()}))
    print("wrote", REFERENCES, flush=True)


def run_check(args):
    refs = read_references()
    result = gate_frames(bundle_frames(args.models, args.tag), args.T, args.reference, refs)
    result["candidate_files"] = bundle_files(args.models, args.tag)
    path = os.path.join(RESULTS, "gate_{}_{}_vs_{}.json".format(args.tag, tag(args.T), args.reference))
    with open(path + ".tmp", "w") as fh:
        json.dump(result, fh, indent=1)
    os.replace(path + ".tmp", path)
    cmp = result["compare"]
    print("truth {}: {} gate {:.4f} vs {} {:.4f} ({:+.4f})".format(
        tag(args.T), args.tag, result["candidate"]["gate"], args.reference, result["reference_cells"]["gate"],
        cmp["gate_gain"]))
    for c in cells_names():
        print("  {:<16} {:.4f} vs {:.4f}  {:+.4f}".format(c, result["candidate"][c], result["reference_cells"][c],
                                                        cmp["gains"][c]))
    print("PASS" if cmp["passes"] else "FAIL", "(+{:g} needed in every cell)".format(GATE), "->", path, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=("references", "check"))
    parser.add_argument("--models", default=MODELS, help="folder with <tag>_decoder.json and its heads")
    parser.add_argument("--tag", help="check: the candidate bundle")
    parser.add_argument("--T", type=float, help="check: the convention of the truth, in seconds (0 = DFL labels)")
    parser.add_argument("--reference", help="check: the stored reference model, " + " or ".join(REFERENCE_TAGS))
    args = parser.parse_args()
    if args.stage == "check" and (args.tag is None or args.T is None or args.reference is None):
        parser.error("check needs --tag, --T and --reference")
    if args.stage == "check" and args.T not in CONVENTIONS:
        parser.error("--T must be one of {}".format(", ".join("{:g}".format(T) for T in CONVENTIONS)))
    {"references": run_references, "check": run_check}[args.stage](args)


if __name__ == "__main__":
    main()
