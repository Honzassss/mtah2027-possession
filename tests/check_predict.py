"""Self-checks for the path the submission file comes from: pipeline/predict.py, pipeline/submission.py
and scripts/predict_match.py.

    python -I tests/check_predict.py [--twin twin/J03WMX] [--copies all|gate|one] [--json results.json]

1. The saved v0 model loads, its bundle pins the head files, and the decoder table is the scored one.
2. Loaded from its files, v0 gives back every spellF1 in results/model_v0_eval.json exactly, from the
   cached features of the held-out copies (analysis/model_v0.py).
3. The same for v2, whose heads v3 uses (analysis/model_v2.py): its bundle is test 1's model of
   convention T = 1, and it gives back that model's held-out rows in results/test1_convention.json
   exactly, against the labels of every convention.
4. v3, the published default (analysis/model_v3.py): v2's heads and alpha with the restarter free to
   change during a stoppage, and on the 36 gate copies it gives back the gate check it shipped on
   (results/gate_v3_T1_vs_v2.json) exactly. models/ holds the scored v0, v2 and v3 files.
5. From scratch, a held-out IDSSE copy run through frame_labels() with v3 gives the same frame labels as
   the cached path, so the features recompute to the bit and the steps spread onto the right frames.
6. The file rules accept a good file, reject each broken one, and no reason names a frame.
7. The script on the twin with its default model, twice: three lines on stdout and nothing on stderr, a
   file that passes the rules, the same bytes both times, a matching .sha256, a sane spellF1 against the
   twin's truth (the sides line up) and frame agreement that peaks at no shift (the frames line up).
   Neither is a model number: the twin carries artificial hints. It also refuses an output inside the
   repository for a bundle outside it, and fails in one line.

Open data and the twin only. Exit code 1 if a check fails.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import replace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis.model_v0 import CACHE, IDSSE, MODELS, cache_path, eval_keys, key_name, load, score  # noqa: E402
from analysis.gate import bundle_frames, evaluate, gate_keys  # noqa: E402
from analysis.model_v2 import T as V2_T, TAG as V2, test1, training_table  # noqa: E402
from analysis.model_v3 import TAG as V3  # noqa: E402
from analysis.spellmetric import read_labels, spell_f1, spells  # noqa: E402
from analysis.test1_convention import OUT as TEST1, conventions_of, cross, tag as convention_tag  # noqa: E402
from analysis.test2_dead_shift import OWN_CACHE, light  # noqa: E402
from pipeline import submission  # noqa: E402
from pipeline.decode import labels_of, restarter_may_change  # noqa: E402
from pipeline.degrade import SIMULATORS, degrade  # noqa: E402
from pipeline.epts import COMPETITION_FRAMES, COMPETITION_HALVES  # noqa: E402
from pipeline.idsse import load_idsse  # noqa: E402
from pipeline.predict import BUNDLE, ModelError, frame_labels, load_model, sha256_file, step_states  # noqa: E402

SCRIPT = os.path.join(ROOT, "scripts", "predict_match.py")
GATE_V3 = os.path.join(ROOT, "results", "gate_v3_T1_vs_v2.json")
FROM_SCRATCH = ("J03WMX", "camera", 0.3, 1)
TWIN_FLOOR = 0.30       # far below v0 and v2 on degraded held-out copies (0.46 and up); swapped sides give 0.01
MAX_SHIFT = 10          # frames scanned each way for the agreement peak
PEAK_FRAMES = 2         # the twin peaks at 0; 5 frames off costs 0.003 of agreement, far beyond noise
results = []


def check(name, ok, detail=""):
    results.append({"name": name, "ok": bool(ok), "detail": str(detail)})
    print("{:4} {:<74} {}".format("PASS" if ok else "FAIL", name, detail), flush=True)


def check_published(tag):
    published = os.path.join(ROOT, "models")
    names = [n for n in sorted(os.listdir(MODELS)) if n.startswith(tag + "_")]
    same = bool(names) and all(os.path.isfile(os.path.join(published, n))
                               and sha256_file(os.path.join(published, n)) == sha256_file(os.path.join(MODELS, n))
                               for n in names)
    check("bundle: the published models/ holds the scored {} files".format(tag), same, " ".join(names))


def selected(which):
    keys = eval_keys()
    if which == "gate":
        return [k for k in keys if k[1] in ("camera", "plain")]
    if which == "one":
        return [FROM_SCRATCH]
    return keys


def check_bundle():
    print("--- the saved model", flush=True)
    model = load_model(MODELS, "v0")
    with open(os.path.join(ROOT, "results", "model_v0_eval.json")) as fh:
        scored = json.load(fh)
    check("bundle: three heads load", sorted(model.boosters) == ["dead", "restart", "team"])
    check("bundle: alpha is the scored one", model.alpha == scored["alpha"], model.alpha)
    check("bundle: transitions are the scored ones", model.table.matrix.tolist() == scored["transitions"])
    check("bundle: log prior is the scored one", model.table.log_prior.tolist() == scored["log_prior"])
    trees = {name: b.num_trees() for name, b in model.boosters.items()}
    check("bundle: trees per head are the cv rounds", trees == scored["rounds"], trees)
    check_published("v0")

    with tempfile.TemporaryDirectory() as tmp:
        for name in os.listdir(MODELS):
            if name.startswith("v0_"):
                shutil.copy(os.path.join(MODELS, name), tmp)
        with open(os.path.join(tmp, "v0_team.txt"), "a") as fh:
            fh.write("\n")
        try:
            load_model(tmp, "v0")
            refused = False
        except ModelError:
            refused = True
        check("bundle: a changed head file is refused", refused)
        os.remove(os.path.join(tmp, BUNDLE.format("v0")))
        try:
            load_model(tmp, "v0")
            refused = False
        except ModelError:
            refused = True
        check("bundle: a missing bundle is refused", refused)
    return model, scored


def check_scored(model, scored, which):
    print("--- v0 from its files against results/model_v0_eval.json", flush=True)
    rows = {(r["match"], r["sim"], r["live"], r["seed"]): r for r in scored["rows"]}
    keys = selected(which)
    exact, worst, started = 0, 0.0, time.time()
    for key in keys:
        copy = load(key)
        states = step_states(model, copy["X"], copy["Xs"], copy["steps"], copy["dead"])
        got, want = score(copy, states), rows[key]["decoded"]
        same = all(got[k] == want[k] for k in ("spell_f1", "pred_spells", "macro_f1", "accuracy"))
        exact += same
        worst = max(worst, abs(got["spell_f1"] - want["spell_f1"]))
        if not same:
            check("scored: " + key_name(key), False, "spellF1 {:.6f} vs {:.6f}".format(got["spell_f1"], want["spell_f1"]))
    check("scored: copies that give back the scored numbers exactly", exact == len(keys),
          "{} of {}, largest spellF1 gap {:.2g}, {:.0f}s".format(exact, len(keys), worst, time.time() - started))


def check_v2(which):
    """v2 as analysis/model_v2.py check sees it: the bundle is test 1's T = 1 model, and its held-out rows
    come back exactly, against the labels of every convention."""
    print("--- v2 from its files against test 1's T = 1 model in results/test1_convention.json", flush=True)
    model = load_model(MODELS, V2)
    held, name = test1()["heldout"], convention_tag(V2_T)
    table = training_table()
    check("v2 bundle: three heads load", sorted(model.boosters) == ["dead", "restart", "team"])
    check("v2 bundle: alpha is test 1's choice for T = 1", model.alpha == float(held["alpha"][name]), model.alpha)
    check("v2 bundle: transitions are counted on the T = 1 training labels",
          model.table.matrix.tolist() == table.matrix.tolist()
          and model.table.log_prior.tolist() == table.log_prior.tolist()
          and model.table.counts.tolist() == table.counts.tolist())
    trees = {head: b.num_trees() for head, b in model.boosters.items()}
    check("v2 bundle: trees per head are test 1's rounds for T = 1", trees == held["rounds"][name], trees)
    check_published(V2)
    keys = selected(which)
    exact, started = 0, time.time()
    for key in keys:
        with np.load(cache_path(key), allow_pickle=False) as d:
            X, Xs = d["X"], d["Xs"]
        copy = light(key)
        copy["conv"] = conventions_of(copy)
        got = cross(copy, step_states(model, X, Xs, copy["steps"], copy["dead"]), copy["conv"])
        same = got == held["rows"][key_name(key)][name]
        exact += same
        if not same:
            check("v2 scored: " + key_name(key), False, "spellF1 " + " ".join(
                "{} {:.6f}".format(t, v["spell_f1"]) for t, v in got.items()))
    check("v2 scored: copies that give back test 1's T = 1 rows exactly", exact == len(keys),
          "{} of {}, {:.0f}s".format(exact, len(keys), time.time() - started))
    return model


def check_v3(which, v2):
    """v3 is v2's heads and alpha with restarter_may_change(v2's table), and its 25 Hz labels on the gate
    copies score exactly as in the gate check it shipped on."""
    print("--- v3 from its files against results/gate_v3_T1_vs_v2.json", flush=True)
    model = load_model(MODELS, V3)
    bundles = {}
    for name in (V2, V3):
        with open(os.path.join(MODELS, BUNDLE.format(name))) as fh:
            bundles[name] = json.load(fh)
    want = restarter_may_change(v2.table)
    check("v3 bundle: its heads are v2's files", bundles[V3]["heads"] == bundles[V2]["heads"])
    check("v3 bundle: alpha is v2's", model.alpha == v2.alpha, model.alpha)
    check("v3 bundle: the table is v2's with the restarter free to change",
          model.table.matrix.tolist() == want.matrix.tolist()
          and model.table.log_prior.tolist() == want.log_prior.tolist()
          and model.table.counts.tolist() == want.counts.tolist())
    check_published(V3)
    gate = set(gate_keys())
    keys = [k for k in selected(which) if k in gate] if which == "one" else gate_keys()
    with open(GATE_V3) as fh:
        rows = json.load(fh)["rows"]
    started = time.time()
    got = evaluate(bundle_frames(MODELS, V3, keys), (V2_T,))
    name = convention_tag(V2_T)
    exact = sum(got[k][name] == rows[key_name(k)][name] for k in keys)
    check("v3 scored: gate copies that give back the gate check's rows exactly", exact == len(keys),
          "{} of {}, {:.0f}s".format(exact, len(keys), time.time() - started))
    return model


def check_from_scratch(model):
    print("--- a held-out copy from scratch through frame_labels()", flush=True)
    match, sim, live, seed = FROM_SCRATCH
    started = time.time()
    copy = degrade(load_idsse(os.path.join(IDSSE, match + ".npz")), seed, replace(SIMULATORS[sim], live_share=live))
    stages = []
    labels = frame_labels(copy, model, stages.append)
    cached = load(FROM_SCRATCH)
    states = step_states(model, cached["X"], cached["Xs"], cached["steps"], cached["dead"])
    want = cached["steps"].frame_labels(labels_of(states), len(cached["codes"]), "X")
    check("scratch: stages run in order", stages == ["clean", "features", "heads", "decode"], " ".join(stages))
    check("scratch: one label per row", labels.shape == (copy.n_frames,) and labels.dtype == np.dtype("<U1"))
    check("scratch: frame labels equal the cached path's", np.array_equal(labels, want),
          "{} rows differ, {:.0f}s".format(int(np.sum(labels != want)) if labels.shape == want.shape else -1,
                                          time.time() - started))
    inside = copy.period > 0
    check("scratch: H, A or D inside the halves, X outside",
          np.isin(labels[inside], ("H", "A", "D")).all() and (labels[~inside] == "X").all())


def check_rules(tmp):
    print("--- the file rules", flush=True)
    n = COMPETITION_FRAMES
    graded = submission.graded_mask()
    dead = submission.dead_mask()
    frames = np.arange(n)
    good = np.where(graded, "D", "X")
    probe = 123457                     # a graded live frame; no reason may name it
    path = os.path.join(tmp, "probe.csv")

    def verdict(labels=good, header=None, drop=False, frame_shift=False, raw=None):
        if raw is not None:
            with open(path, "w") as fh:
                fh.write(raw)
        else:
            digest = submission.write(path, frames, labels)
            check_digest = sha256_file(path) == digest
            if not check_digest:
                return "digest mismatch", None
            with open(path) as fh:
                lines = fh.read().split("\n")
            if header is not None:
                lines[0] = header
            if drop:
                del lines[1 + probe]
            if frame_shift:
                lines[1 + probe] = "{},{}".format(probe + 1, labels[probe])
            with open(path, "w") as fh:
                fh.write("\n".join(lines))
        return submission.check(path)

    reason, codes = verdict()
    check("rules: a good file passes", reason is None and codes is not None and len(codes) == n, reason)
    crlf = "frame,label\r\n" + "".join("{},{}\r\n".format(i, lab) for i, lab in enumerate(good.tolist()))
    check("rules: CRLF line ends pass", verdict(raw=crlf)[0] is None)
    broken = {
        "header": verdict(header="Frame,Label"),
        "one row short": verdict(drop=True),
        "frame number off": verdict(frame_shift=True),
        "blank label": verdict(labels=np.where(frames == probe, "", good)),
        "lower-case label": verdict(labels=np.where(frames == probe, "h", good)),
        "X on a graded frame": verdict(labels=np.where(frames == probe, "X", good)),
        "H outside the halves": verdict(labels=np.where(frames == 5, "H", good)),
        "H in the dead block": verdict(labels=np.where(dead & (np.cumsum(dead) == 7), "H", good)),
    }
    for name, (reason, _) in broken.items():
        check("rules: refused, " + name, reason is not None, reason)
    named = [name for name, (reason, _) in broken.items() if reason and (str(probe) in reason or str(probe + 1) in reason)]
    check("rules: no reason names a frame", not named, " ".join(named))

    codes = np.full(n, submission.X, dtype=np.int8)
    codes[graded] = submission.D
    first = COMPETITION_HALVES[0][1]
    codes[first:first + 6] = [0, 0, 1, 1, 2, 0]      # H H A A D H: three spells
    second = COMPETITION_HALVES[1][1] + 5000
    codes[second:second + 3] = [1, 1, 1]
    runs = [(a, b + 1) for _, a, b in COMPETITION_HALVES]
    check("rules: spell count", submission.spell_count(codes) == 4 == len(spells(codes, runs)),
          submission.spell_count(codes))


def run_script(tactical, out):
    proc = subprocess.run([sys.executable, "-I", SCRIPT, tactical, out], capture_output=True, text=True)
    return proc.returncode, proc.stdout.splitlines(), proc.stderr


def check_twin(twin):
    print("--- the script on the twin", flush=True)
    tactical = os.path.join(twin, "data", "tactical")
    outdir = os.path.join(ROOT, "results", "twin")
    os.makedirs(outdir, exist_ok=True)
    paths = [os.path.join(outdir, "check_predict_{}.csv".format(i)) for i in (1, 2)]
    for p in paths:
        for q in (p, p + ".sha256", p + ".tmp"):
            if os.path.exists(q):
                os.remove(q)
    runs = []
    for p in paths:
        started = time.time()
        code, out, err = run_script(tactical, p)
        runs.append((code, out, err, time.time() - started))
    code, out, err, seconds = runs[0]
    check("twin: exit code 0", code == 0, code)
    check("twin: nothing on stderr", err == "", "{} chars".format(len(err)))
    shape = len(out) == 3 and out[0] == "PASS" and out[1] == "rows {}".format(COMPETITION_FRAMES) \
        and out[2].startswith("spells ") and out[2][7:].isdigit()
    check("twin: exactly PASS, rows and spells on stdout", shape, " | ".join(out) + ", {:.0f}s".format(seconds))
    if not shape or not os.path.isfile(paths[0]):
        return {}
    reason, codes = submission.check(paths[0])
    check("twin: the file passes the rules again", reason is None, reason)
    check("twin: no temporary file left", not os.path.exists(paths[0] + ".tmp"))
    with open(paths[0] + ".sha256") as fh:
        recorded = fh.read().split()
    check("twin: .sha256 matches the file", recorded == [sha256_file(paths[0]), os.path.basename(paths[0])])
    same = runs[1][0] == 0 and sha256_file(paths[0]) == sha256_file(paths[1])
    check("twin: a second run writes the same bytes", same, "{:.0f}s".format(runs[1][3]))
    halves = [(a, b + 1) for _, a, b in COMPETITION_HALVES]
    n_spells = int(out[2][7:])
    check("twin: printed spells = the scorer's spells", n_spells == submission.spell_count(codes) == len(spells(codes, halves)),
          n_spells)

    truth = read_labels(os.path.join(twin, "gt", "possession_gt_full.csv"))
    f1, n_gt, n_pred, _ = spell_f1(truth, codes)
    graded = truth != 3
    agree = float(np.mean(truth[graded] == codes[graded]))
    check("twin: truth is X exactly outside the halves", np.array_equal(graded, submission.graded_mask()))
    check("twin: spellF1 against the twin's truth above {:.2f}".format(TWIN_FLOOR), f1 > TWIN_FLOOR,
          "{:.3f}, spells {}/{}, frames agreeing {:.3f}".format(f1, n_pred, n_gt, agree))
    rows = np.flatnonzero(graded)
    rows = rows[(rows >= MAX_SHIFT) & (rows < len(truth) - MAX_SHIFT)]
    shifts = np.arange(-MAX_SHIFT, MAX_SHIFT + 1)
    curve = [float(np.mean(truth[rows] == codes[rows + s])) for s in shifts]
    peak = int(shifts[int(np.argmax(curve))])
    check("twin: frame agreement peaks within {} frames of no shift".format(PEAK_FRAMES), abs(peak) <= PEAK_FRAMES,
          "peak at {:+d}, {:.4f} there, {:.4f} at -5, {:.4f} at +5".format(
              peak, max(curve), curve[MAX_SHIFT - 5], curve[MAX_SHIFT + 5]))

    code, out, _ = run_script("/nonexistent-bundle-outside-the-repo", os.path.join(outdir, "refused.csv"))
    check("twin: output inside the repo refused for a bundle outside it",
          code == 1 and out == ["FAIL output: the bundle is outside the repository, so the output must be too"]
          and not os.path.exists(os.path.join(outdir, "refused.csv")), " | ".join(out))
    code, out, _ = run_script(os.path.join(twin, "no-such-folder"), os.path.join(outdir, "missing.csv"))
    check("twin: a missing bundle fails in one line",
          code == 1 and len(out) == 1 and out[0].startswith("FAIL load: ")
          and not os.path.exists(os.path.join(outdir, "missing.csv")), " | ".join(out))
    for p in paths:
        for q in (p, p + ".sha256"):
            os.remove(q)
    return {"spell_f1": f1, "pred_spells": n_pred, "gt_spells": n_gt, "frames_agreeing": agree, "seconds": seconds}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--twin", default=os.path.join(ROOT, "twin", "J03WMX"))
    parser.add_argument("--copies", choices=("all", "gate", "one"), default="all")
    parser.add_argument("--json")
    args = parser.parse_args()
    if not os.path.isdir(CACHE):
        raise SystemExit("needs the v0 cache: run analysis/model_v0.py build first")
    if not os.path.isfile(TEST1) or not os.path.isdir(OWN_CACHE):
        raise SystemExit("needs test 1's results: run analysis/test2_dead_shift.py predict, then the stages of "
                         "analysis/test1_convention.py")
    if not os.path.isfile(GATE_V3):
        raise SystemExit("needs the gate check v3 shipped on: analysis/gate.py check --tag v3 --T 1 --reference v2")
    started = time.time()
    measurements = {}
    model, scored = check_bundle()
    check_scored(model, scored, args.copies)
    v2 = check_v2(args.copies)
    published = check_v3(args.copies, v2)
    check_from_scratch(published)
    with tempfile.TemporaryDirectory() as tmp:
        check_rules(tmp)
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
