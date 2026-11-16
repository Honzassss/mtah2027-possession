"""Test 9 on the competition match: does the camera's aim follow the visible ball?

    .venv/bin/python -I scripts/match_test9.py <tactical_dir> <report.txt>
    .venv/bin/python -I scripts/match_test9.py twin/J03WMX/data/tactical results/twin/test9.txt

<tactical_dir> holds BAR-ATH_tactical_atd.xml, _atd.txt and _homography.csv. The aim is the image
centre's column, from the per-frame homography; the ball is the cleaned ball (pipeline/clean.py), inside
the halves, the published dead block left out. The report has:

    1. a sanity check: the seen ball inside the image, with the homography shifted by a few frames
    2. the aim against the ball: the error along the pitch at the ball's depth, the distance from the aim
       point, the ball's offset from the image centre in pixels, and the same errors for the centroid of
       the tracked players, the cheap ball proxy the aim has to beat
    3. the error by lag, to find how late the camera follows
    4. at the edges of ball gaps (1 s and over) and across them: which of last seen position, camera aim
       and player centroid best predicts where the ball reappears

Every part is repeated on degraded IDSSE copies (simulator `camera`), whose virtual camera follows the
true ball: that is what the training data assumes about the real camera.

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
from pipeline.clean import clean  # noqa: E402
from pipeline.epts import load_competition  # noqa: E402
from pipeline.operate import MISSING, fmt, pct  # noqa: E402

TITLE = "Test 9: camera aim against the visible ball"
SIMULATOR = "camera"
SUMMARY_ROWS = (
    ("x error at the ball's depth, median (m)", "x_median", "{:.1f}"),
    ("x error at the ball's depth, 90th percentile (m)", "x_p90", "{:.1f}"),
    ("distance from the aim point, median (m)", "dist_median", "{:.1f}"),
    ("ball off the image centre, median (px)", "offset_px_median", "{:.0f}"),
    ("ball off the image centre, signed median (px)", "offset_px_bias", "{:+.0f}"),
    ("players' centroid x error, median (m)", "centroid_x_median", "{:.1f}"),
    ("players' centroid distance, median (m)", "centroid_dist_median", "{:.1f}"),
    ("aim x error / centroid x error", "ratio", "{:.2f}"),
)


def mean_of(results, get):
    values = np.array([get(r) for r in results], dtype=np.float64)
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


def references(idsse_dir, copies):
    if not os.path.isdir(idsse_dir):
        return []
    out = []
    for _, copy, cleaned, matrices in probes.simulated_copies(idsse_dir, SIMULATOR, limit=copies):
        result = probes.aim_measurements(cleaned, matrices)
        result["shift"] = probes.ball_in_image_by_shift(copy, matrices)
        out.append(result)
    return out


def render(args, shift, aim, sims):
    report = operate.Report(TITLE, args.tactical_dir, operate.fingerprint(__file__, probes.__file__))
    sim = (lambda get: mean_of(sims, get)) if sims else (lambda get: float("nan"))

    report.section("1. Sanity: the homography against the tracking")
    report.text("Share of the seen ball inside the image when the homography is taken N frames later:")
    report.table(("shift, frames",) + tuple("{:+d}".format(s) for s in shift), [
        ("match",) + tuple(pct(v, 2) for v in shift.values()),
        ("sim",) + tuple(pct(sim(lambda r, s=s: r["shift"][s]), 2) for s in shift)])
    ok, reason = probes.homography_aligned(shift)
    report.text("Aligned (the seen ball inside its own frame's image on {} or more): {}.".format(
        pct(probes.ALIGNED_FLOOR, 0), "yes" if ok else "NO, " + reason),
        "A camera that lags the ball shows it slightly more often in later images, so a rise to the right is normal.")

    report.section("2. Aim against the visible ball")
    report.text("Ball seen: {} min; seen and moving (open play): {} min.".format(
        fmt(aim["seen"]["seconds"] / 60, "{:.1f}"), fmt(aim["moving"]["seconds"] / 60, "{:.1f}")))
    report.table(("measure", "ball seen", "open play", "sim, seen", "sim, open play"), [
        (label, fmt(aim["seen"][key], spec), fmt(aim["moving"][key], spec),
         fmt(sim(lambda r, k=key: r["seen"][k]), spec), fmt(sim(lambda r, k=key: r["moving"][k]), spec))
        for label, key, spec in SUMMARY_ROWS])

    report.section("3. How late the camera follows")
    report.text("Median x error with the camera taken L seconds after the ball (positive: the camera lags):")
    report.table(("lag (s)", "match (m)", "sim (m)"), [
        ("{:+.1f}".format(lag), fmt(value, "{:.1f}"), fmt(sim(lambda r, i=i: r["lags"][i][1]), "{:.1f}"))
        for i, (lag, value) in enumerate(aim["lags"])])
    report.text("", "Best lag: match {} s at {} m; sim {} s at {} m.".format(
        fmt(aim["best_lag_s"], "{:+.1f}"), fmt(aim["best_lag_x_median"], "{:.1f}"),
        fmt(sim(lambda r: r["best_lag_s"]), "{:+.1f}"), fmt(sim(lambda r: r["best_lag_x_median"]), "{:.1f}")))

    report.section("4. Gap edges and where the ball reappears (gaps of {:g} s and over)".format(probes.AIM_GAP_MIN_S))
    report.text("x error at the gap edges (last seen, first seen again), median: match {} m over {} points; sim {} m.".format(
        fmt(aim["edge_x_median"], "{:.1f}"), aim["edge_points"], fmt(sim(lambda r: r["edge_x_median"]), "{:.1f}")))
    report.text("", "Median error in x of predicting where the ball reappears, from: its last seen x, the camera aim at",
                "the end of the gap (at the last seen depth), the players' centroid at the end of the gap (m):")
    rows = []
    for i, row in enumerate(aim["reappear"]):
        name = "all" if i == 0 else ("{:g}-{:g} s".format(row["low"], row["high"]) if np.isfinite(row["high"])
                                     else "{:g} s and over".format(row["low"]))
        rows.append((name, row["gaps"], fmt(row["last_seen"], "{:.1f}"), fmt(row["aim"], "{:.1f}"),
                     fmt(row["centroid"], "{:.1f}"),
                     " / ".join(fmt(sim(lambda r, i=i, k=k: r["reappear"][i][k]), "{:.1f}")
                                for k in ("last_seen", "aim", "centroid"))))
    report.table(("gap length", "gaps", "last seen", "camera aim", "centroid", "sim: last seen / aim / centroid"), rows)
    report.text("", "Gaps whose ball moved over 10 m in x: the camera panned the same way in {} of {} (sim {}); the pan".format(
        pct(aim["pan_follows_share"]), aim["pan_follows_gaps"], pct(sim(lambda r: r["pan_follows_share"]))),
        "against the ball's move: r {} (sim {}).".format(fmt(aim["pan_corr"]), fmt(sim(lambda r: r["pan_corr"]))))
    if not sims:
        report.text("", "No simulated references: no IDSSE copies found in {} (the sim columns show {}).".format(
            args.idsse, MISSING))

    report.section("What to tell Claude")
    report.text(
        "Pick the sentences that fit and send them without figures:",
        "- Part 2 open play and part 4 edges with an x error under about 10 m, and the aim/centroid ratio under",
        "  about 0.7: 'Test 9 passed: features may read the camera aim.'",
        "- Otherwise: 'Test 9 failed: no feature reads the view polygon.'",
        "- The camera aim predicting the reappearing ball better than both the last seen x and the centroid",
        "  (part 4): add 'the aim predicts where the ball comes back'.",
        "- The match clearly worse than the sim columns: add 'the simulated camera follows the ball too",
        "  closely'; clearly better: '... too loosely'. A best lag far from the sim's: 'the real camera",
        "  lags more (or less) than the simulated one'.",
        "- Part 1 not aligned: 'the homography does not line up with the tracking' (test 8 checks the same).")
    return report.render()


def run(args, on_stage):
    on_stage("output")
    operate.prepare_report(args.report, args.tactical_dir)
    on_stage("load")
    match = load_competition(args.tactical_dir, with_homography=True)
    matrices = probes.metres_matrices(match)
    on_stage("clean")
    cleaned = clean(match)
    on_stage("shift")
    shift = probes.ball_in_image_by_shift(match, matrices)
    on_stage("aim")
    aim = probes.aim_measurements(cleaned, matrices)
    on_stage("reference")
    sims = references(args.idsse, args.copies)
    on_stage("report")
    operate.write_report(args.report, render(args, shift, aim, sims))
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
