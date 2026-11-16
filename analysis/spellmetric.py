"""Clean-room reimplementation of the board's two headline metrics, written from the spec in
EVALUATION.md / possession_metrics.py. Reads CSVs as plain data; imports nothing from the repo.
Checked against the scoreboard printed in the challenge repository's README."""
import numpy as np

H, A, D, X = 0, 1, 2, 3
IOUS = (0.3, 0.4, 0.5, 0.6, 0.7)
CODE = {"H": H, "A": A, "D": D, "X": X}


def read_labels(path):
    out = []
    with open(path) as fh:
        next(fh)
        for line in fh:
            out.append(CODE[line.rstrip("\r\n").split(",")[1].strip().upper()])
    return np.array(out, dtype=np.int8)


def runs_of(gt):
    g = (gt != X).astype(np.int8)
    e = np.diff(np.concatenate(([0], g, [0])))
    return list(zip(np.flatnonzero(e == 1).tolist(), np.flatnonzero(e == -1).tolist()))


def fill(pred, runs):
    f = pred.copy()
    for a, b in runs:
        w = f[a:b]
        known = np.flatnonzero(w != X)
        if not len(known):
            continue
        c = np.maximum.accumulate(np.where(w != X, np.arange(len(w)), -1))
        c[c < 0] = known[0]
        f[a:b] = w[c]
    return f


def segments(codes, runs):
    out = []
    for a, b in runs:
        w = codes[a:b]
        if not len(w):
            continue
        br = np.flatnonzero(w[1:] != w[:-1]) + 1
        bd = np.concatenate(([0], br, [len(w)]))
        for i in range(len(bd) - 1):
            out.append((a + int(bd[i]), a + int(bd[i + 1]), int(w[bd[i]])))
    return out


def spells(codes, runs):
    return [s for s in segments(codes, runs) if s[2] in (H, A)]


def overlaps(gs, ps):
    """Same-team overlapping pairs sorted for greedy matching. gs/ps are start-sorted."""
    out = []
    j0 = 0
    for i, (ga, gb, gc) in enumerate(gs):
        while j0 < len(ps) and ps[j0][1] <= ga:
            j0 += 1
        j = j0
        while j < len(ps) and ps[j][0] < gb:
            pa, pb, pc = ps[j]
            if pc == gc:
                out.append(((min(gb, pb) - max(ga, pa)) / float(max(gb, pb) - min(ga, pa)), i, j))
            j += 1
    out.sort(key=lambda t: (-t[0], t[1], t[2]))
    return out


def spell_f1(gt, pred):
    runs = runs_of(gt)
    f = fill(pred, runs)
    gs, ps = spells(gt, runs), spells(f, runs)
    ov = overlaps(gs, ps)
    f1s = []
    for thr in IOUS:
        mg, mp = set(), set()
        for iou, i, j in ov:
            if iou < thr:
                break
            if i in mg or j in mp:
                continue
            mg.add(i)
            mp.add(j)
        tp = len(mg)
        f1s.append(2.0 * tp / (len(gs) + len(ps)) if tp else 0.0)
    return float(np.mean(f1s)), len(gs), len(ps), f1s


def macro_f1(gt, pred):
    mask = gt != X
    g, p = gt[mask], pred[mask]
    out = []
    for k in (H, A, D):
        tp = np.sum((g == k) & (p == k))
        fp = np.sum((g != k) & (p == k))
        fn = np.sum((g == k) & (p != k))
        out.append(2.0 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(out))
