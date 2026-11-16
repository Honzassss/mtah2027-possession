"""The parallax check on the competition match: is the ball tracked without its height?

    .venv/bin/python -I scripts/match_parallax.py <tactical_dir> <report.txt>
    .venv/bin/python -I scripts/match_parallax.py twin/J03WMX/data/tactical results/twin/parallax.txt

<tactical_dir> holds BAR-ATH_tactical_atd.xml, _atd.txt and _homography.csv. A tracker that maps the ball's
pixel to the pitch as if it lay on the grass puts a ball in the air further along the camera ray (test 3,
analysis/test3_virtual_camera.py). Three label-free signatures, on the raw ball inside the halves
with the published dead block left out:

    fast steps   steps over 40 m/s between consecutive frames run along the camera ray: the median |cos|
                 between step and ray is near 1 with parallax and 0.71 for random directions. The ray runs
                 from the camera's foot, solved for every frame from its homography as pipeline/camera.py
                 solves the DATA.md row, to the ball
    far line     the ball more than 2 m beyond the touchline away from the camera, against the near one
    gate         the share of ball observations the 40 m/s cleaning gate removes

Each is set against the same code on degraded IDSSE copies with parallax (`camera-parallax`) and without
(`camera-no-parallax`), and against test 3's upper bounds (IDSSE live frames, the ball always seen).

It prints one line: PASS once the report is written, or FAIL <stage>: <reason>. Every figure goes to
<report.txt>; they summarise the match, so the report is for the operator's eyes only, and what goes back
to Claude is the printed line or a general code request without figures (the report ends with the ones
that fit). When the bundle lies outside this repository, the report must lie outside it too.
--idsse, --copies and --verbose as in scripts/match_test7.py.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline import operate, probes  # noqa: E402
from pipeline.camera import solve_camera  # noqa: E402
from pipeline.clean import clean  # noqa: E402
from pipeline.epts import load_competition  # noqa: E402
from pipeline.operate import MISSING, fmt, pct  # noqa: E402

TITLE = "Parallax check: is the ball tracked without its height?"
SIMULATORS = ("camera-parallax", "camera-no-parallax")
# analysis/test3_virtual_camera.py on the seven IDSSE matches (run of 7 Nov 2026): live frames, the
# ball always seen, the camera of the DATA.md row. Upper bounds: nothing is lost in the air there.
TEST3 = {"median_cos": ("0.96", "-"), "beyond_far": ("4.3%", "0.01%"), "gate_removed": ("2.5%", "0.05%")}


def references(idsse_dir, copies):
    if not os.path.isdir(idsse_dir):
        return {}
    out = {}
    for simulator in SIMULATORS:
        out[simulator] = [probes.parallax_measurements(copy, cleaned, matrices)
                          for _, copy, cleaned, matrices in probes.simulated_copies(idsse_dir, simulator, limit=copies)]
    return out


def mean_of(results, get):
    values = np.array([get(r) for r in results], dtype=np.float64)
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


def render(args, m, sims):
    report = operate.Report(TITLE, args.tactical_dir, operate.fingerprint(__file__, probes.__file__))

    def sim(name, get):
        return mean_of(sims[name], get) if sims.get(name) else float("nan")

    report.section("1. The camera, solved for every frame from its homography")
    data_md = solve_camera()
    report.text("Frames with a solution: {} of the play. Horizontal principal point {} px (DATA.md row {} px).".format(
        pct(m["camera_usable"]), fmt(m["principal_x"], "{:.1f}"), fmt(data_md.principal[0], "{:.1f}")))
    report.table(("camera", "x (m)", "y (m)", "height (m)", "behind the near touchline (m)"), [
        ("median over frames",) + tuple(fmt(v, "{:.1f}") for v in m["camera_median"]) +
        (fmt(-34.0 - m["camera_median"][1], "{:.1f}"),),
        ("spread (interquartile)",) + tuple(fmt(v, "{:.1f}") for v in m["camera_iqr"]) + ("",),
        ("DATA.md row",) + tuple(fmt(v, "{:.1f}") for v in data_md.centre) + (fmt(-34.0 - data_md.centre[1], "{:.1f}"),)])

    report.section("2. Fast ball steps (over {:g} m/s between consecutive frames)".format(probes.FAST_STEP_MPS))
    report.text("Steps with the ball seen on both frames: {}; fast: {} of them ({} steps).".format(
        "{:.1f} thousand".format(m["steps"] / 1000.0), pct(m["fast_share"], 2), m["fast_steps"] if m["fast_steps"] < 10000
        else "{:.1f} thousand".format(m["fast_steps"] / 1000.0)))
    rows = [("match, ray from the {}".format(name), fmt(r["median_cos"]), pct(r["aligned_share"]))
            for name, r in m["rays"].items()]
    for name in SIMULATORS:
        rows.append(("sim {}, per-frame camera".format(name), fmt(sim(name, lambda r: r["rays"]["per-frame camera"]["median_cos"])),
                     pct(sim(name, lambda r: r["rays"]["per-frame camera"]["aligned_share"]))))
    rows.append(("test 3, with parallax (upper bound)", TEST3["median_cos"][0], ""))
    rows.append(("random directions", fmt(probes.RANDOM_MEDIAN_COS), "28.7%"))
    report.table(("", "median |cos|", "|cos| over {:g}".format(probes.RAY_ALIGNED_COS)), rows)
    report.text("Fast steps in the sims: {} / {} of the steps.".format(
        *[pct(sim(name, lambda r: r["fast_share"]), 2) for name in SIMULATORS]))

    report.section("3. The ball beyond the touchlines (by over {:g} m)".format(probes.BEYOND_LINE_M))
    def ratio(far, near):
        return fmt(far / near if np.isfinite(far) and np.isfinite(near) and near > 0 else float("nan"))

    rows = [("match", pct(m["beyond_far"], 2), pct(m["beyond_near"], 2), ratio(m["beyond_far"], m["beyond_near"]))]
    for name in SIMULATORS:
        far, near = sim(name, lambda r: r["beyond_far"]), sim(name, lambda r: r["beyond_near"])
        rows.append(("sim " + name, pct(far, 2), pct(near, 2), ratio(far, near)))
    rows.append(("test 3, with / without parallax", "{} / {}".format(*TEST3["beyond_far"]), "", ""))
    report.table(("share of the seen ball", "beyond the far line", "beyond the near line", "far / near"), rows)
    report.text("The match and the sims count every seen frame of play, dead balls out of play included, so the",
                "far/near ratio is the robust part: parallax pushes the ball away from the camera, a dead ball",
                "beyond a line sits on either side. Test 3 counts live frames only.")

    report.section("4. The 40 m/s cleaning gate")
    report.table(("share of ball observations removed", ""), [
        ("match", pct(m["gate_removed"], 2))] + [
        ("sim " + name, pct(sim(name, lambda r: r["gate_removed"]), 2)) for name in SIMULATORS] + [
        ("test 3, with / without parallax", "{} / {}".format(*TEST3["gate_removed"]))])
    if not sims:
        report.text("", "No simulated references: no IDSSE copies found in {} (the sim rows show {}).".format(
            args.idsse, MISSING))

    report.section("What to tell Claude")
    report.text(
        "Set the match rows of parts 2 (median |cos| and the share of fast steps), 3 (far/near) and 4 against",
        "the two sim rows; a median |cos| from a few dozen fast steps weighs little. Then send one sentence",
        "without figures:",
        "- All three close to camera-parallax: 'The parallax check says the ball has no height correction;",
        "  train with camera-parallax.'",
        "- All three close to camera-no-parallax: 'The parallax check says the ball is height-corrected; train",
        "  with camera-no-parallax.'",
        "- Mixed, or the match outside both: 'The parallax check is mixed; keep the 50/50 camera simulator.'",
        "- Part 1 with no solution on many frames, or a camera far from the DATA.md row: add 'the per-frame",
        "  camera solve looks wrong'.")
    return report.render()


def run(args, on_stage):
    on_stage("output")
    operate.prepare_report(args.report, args.tactical_dir)
    on_stage("load")
    match = load_competition(args.tactical_dir, with_homography=True)
    matrices = probes.metres_matrices(match)
    on_stage("clean")
    cleaned = clean(match)
    on_stage("signatures")
    measured = probes.parallax_measurements(match, cleaned, matrices)
    on_stage("reference")
    sims = references(args.idsse, args.copies)
    on_stage("report")
    operate.write_report(args.report, render(args, measured, sims))
    return ["PASS"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tactical_dir", help="folder holding BAR-ATH_tactical_atd.xml, _atd.txt and _homography.csv")
    parser.add_argument("report", help="text file for the figures, outside the repository for the match")
    parser.add_argument("--idsse", default=os.path.join(ROOT, "external", "processed", "idsse"))
    parser.add_argument("--copies", type=int, default=None, help="how many IDSSE matches to copy (default all)")
    parser.add_argument("--verbose", action="store_true",
                        help="library output and tracebacks; never share it from a run on the match")
    args = parser.parse_args()
    return operate.run(lambda on_stage: run(args, on_stage), args.verbose)


if __name__ == "__main__":
    sys.exit(main())
