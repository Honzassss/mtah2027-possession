"""Break-even probability for emitting one suspected short flip inside a long possession.
F1 = 2TP/D with D = 2TP+FP+FN, so dF = (2(1-F)dTP - F(dFP+dFN))/D. Counts are summed over the five
IoU thresholds. Two cases when the flip is real: boundaries exact, and boundaries off by about 0.2 s
so the short spell itself only matches at the three loosest thresholds."""
import sys
sys.path.insert(0, sys.argv[1])
import spellmetric as sm

L, d = 750, 25


def score(gs, ps, skip_short_above=None):
    tp = fp = fn = 0
    for thr in sm.IOUS:
        mg, mp = set(), set()
        for iou, i, j in sm.overlaps(gs, ps):
            if iou < thr:
                break
            if i in mg or j in mp:
                continue
            if skip_short_above is not None and thr > skip_short_above and gs[i][1] - gs[i][0] <= d:
                continue
            mg.add(i); mp.add(j)
        tp += len(mg); fp += len(ps) - len(mg); fn += len(gs) - len(mg)
    return tp, fp, fn


def delta(a, b, F):
    return 2 * (1 - F) * (b[0] - a[0]) - F * ((b[1] - a[1]) + (b[2] - a[2]))


print("flip position | F1 background | break-even P(real), exact boundaries | sloppy boundaries")
for frac in (0.05, 0.2, 0.33, 0.5):
    t = int(L * frac)
    whole = [(0, L, sm.H)]
    split = [(0, t, sm.H), (t, t + d, sm.A), (t + d, L, sm.H)]
    keep_real, keep_false = score(split, whole), score(whole, whole)
    emit_false = score(whole, split)
    for F in (0.4, 0.6, 0.8):
        loss = -delta(keep_false, emit_false, F)
        exact = delta(keep_real, score(split, split), F)
        sloppy = delta(keep_real, score(split, split, skip_short_above=0.5), F)
        print(f"  {frac:4.2f}  F={F}:  exact {loss/(exact+loss):.2f}   sloppy {loss/(sloppy+loss):.2f}")
