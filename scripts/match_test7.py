"""Test 7 on the competition match: what the ball gaps hide, without labels.

    .venv/bin/python -I scripts/match_test7.py <tactical_dir> <report.txt>
    .venv/bin/python -I scripts/match_test7.py twin/J03WMX/data/tactical results/twin/test7.txt

<tactical_dir> holds BAR-ATH_tactical_atd.xml, _atd.txt and _homography.csv. The question is how much of
the time without a ball is live play (estimates disagree widely). The report has four parts:

    1. the gaps in the ball track by length, inside the halves, the published dead block left out
    2. the players' mean speed inside gaps of each length, as a ratio to their speed where the ball is
       seen and moving, unmixed into a live share: ratio = p * live + (1 - p) * dead, with the live and
       dead ratios of each length from degraded IDSSE copies (whose truth is known, so the estimate is
       also checked there, one copy left out at a time), and again with the match's own published dead
       block as the dead reference
    3. visibility against movement per 5 minutes of play
    4. what the tactical camera does during gaps: zoom, how fast its aim moves, reprojection error, and
       how often the ball was last seen, or is next seen, at the left or right edge of the image

It prints one line: PASS once the report is written, or FAIL <stage>: <reason>. Every figure goes to
<report.txt>; they summarise the match, so the report is for the operator's eyes only, and what goes back
to Claude is the printed line or a general code request without figures (the report ends with the ones
that fit). When the bundle lies outside this repository, the report must lie outside it too.

--idsse and --copies choose the degraded IDSSE copies behind the references (default: every match in
external/processed/idsse, one copy each, simulator `camera`, seed 1). --verbose shows library output and
tracebacks, for the operator's own eyes only.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline import operate, probes  # noqa: E402
from pipeline.clean import clean  # noqa: E402
from pipeline.epts import load_competition  # noqa: E402
from pipeline.operate import fmt, pct  # noqa: E402

TITLE = "Test 7: what the ball gaps hide"
SIMULATOR = "camera"


def references(idsse_dir, copies):
    """Gap tables of degraded IDSSE copies, with their truth; empty when there are none."""
    if not os.path.isdir(idsse_dir):
        return []
    return [probes.gap_table(copy, cleaned, labels=copy.labels)
            for _, copy, cleaned, _ in probes.simulated_copies(idsse_dir, SIMULATOR, limit=copies)]


def render(args, table, windows, camera, tables):
    report = operate.Report(TITLE, args.tactical_dir, operate.fingerprint(__file__, probes.__file__))

    report.section("1. Ball gaps inside the halves (the published dead block left out)")
    report.text("{} min of play, ball seen on {} of it, {} gaps, {} min without the ball".format(
        fmt(table["play_s"] / 60, "{:.1f}"), pct(table["seen_share"]), table["gaps"],
        fmt(table["missing_s"] / 60, "{:.1f}")))
    report.table(("gap length", "gaps", "minutes", "of the missing time"), [
        (row["name"], row["gaps"], fmt(row["seconds"] / 60, "{:.1f}"), pct(row["share_of_missing"]))
        for row in table["buckets"]])

    refs = probes.gap_references(tables) if tables else [(float("nan"), float("nan"))] * len(table["buckets"])
    estimate = probes.estimate_live_share(table, refs)
    from_block = probes.estimate_live_share(table, refs, dead_ratio=table["block_ratio"])
    report.section("2. Player speed in the gaps")
    report.text("Mean speed of the tracked players (m/s). ratio = speed / speed where the ball is seen and moving",
                "(at least {} m/s), the reference.".format(probes.LIVE_BALL_SPEED))
    report.table(("where", "minutes", "speed", "ratio"), [
        ("ball seen and moving", fmt(table["confident_live_s"] / 60, "{:.1f}"), fmt(table["confident_live_speed"]), "1.00"),
        ("ball seen", "", fmt(table["seen_speed"]), fmt(table["seen_ratio"])),
        ("published dead block", fmt(table["block_s"] / 60, "{:.1f}"), fmt(table["block_speed"]), fmt(table["block_ratio"]))])
    report.text("", "Live share of each gap length, unmixed from its ratio (sim = degraded IDSSE copies: their live and",
                "dead ratios for that length; block = the same with the match's dead block as the dead ratio):")
    report.table(("gap length", "speed", "ratio", "sim live", "sim dead", "live share, sim", "live share, block"), [
        (row["name"], fmt(row["speed"]), fmt(row["ratio"]), fmt(live), fmt(dead), pct(p), pct(q))
        for row, (live, dead), p, q in zip(table["buckets"], refs, estimate["buckets"], from_block["buckets"])])
    report.text("", "Live share of all gap time: {} with the simulated references, {} with the dead block.".format(
        pct(estimate["overall"]), pct(from_block["overall"])))
    if tables:
        check = probes.validate_estimator(tables)
        report.text(
            "Check on {} simulated copies ({} simulator), each estimated from the others' references against".format(
                check["copies"], SIMULATOR),
            "its truth: mean error {}, worst {}, bias {}, over copies whose true live share ran from {} to {}.".format(
                pct(check.get("mean_abs_error")), pct(check.get("max_abs_error")), pct(check.get("bias")),
                pct(check.get("truth_low")), pct(check.get("truth_high"))),
            "In the copies, live play over all frames runs at ratio {} and dead play at {}.".format(
                fmt(probes.pooled(tables, "live_ratio")), fmt(probes.pooled(tables, "dead_ratio"))))
    else:
        report.text("No simulated references: no IDSSE copies found in {}.".format(args.idsse))

    report.section("3. Visibility against movement, per 5 minutes of play")
    report.text("Windows start at each half's kickoff. The first window of the first half is the public window.")
    report.table(("half", "minutes", "ball seen", "speed", "speed, ball seen", "speed, ball missing", "in gaps of 10 s+"), [
        (w["half"], "{:.0f}-{:.0f}".format(w["from_min"], w["to_min"]), pct(w["seen"]), fmt(w["speed"]),
         fmt(w["speed_seen"]), fmt(w["speed_missing"]), pct(w["long_gap_share"])) for w in windows["windows"]])
    report.text("", "Across windows: visibility against mean speed r {} (rank {}); visibility against the speed".format(
                    fmt(windows["corr_seen_speed"]), fmt(windows["rank_corr_seen_speed"])),
                "while the ball is missing r {}; time in long gaps against mean speed r {}.".format(
                    fmt(windows["corr_seen_missing_speed"]), fmt(windows["corr_long_gap_speed"])))

    report.section("4. The tactical camera during gaps")
    report.text("Medians over the frames. zoom = ground width of the middle image row (m); aim speed = speed of the",
                "image centre's ground point (m/s); at edge = share of gaps whose ball was last seen (before) or is",
                "next seen (after) within {} of the image width from its left or right edge.".format(pct(probes.EDGE_SHARE, 0)))
    rows = [("ball seen", fmt(camera["seen"]["seconds"] / 60, "{:.1f}"), fmt(camera["seen"]["zoom"], "{:.1f}"),
             fmt(camera["seen"]["aim_speed"]), fmt(camera["seen"]["error"]), pct(camera["seen"]["edge_seen"]), ""),
            ("published dead block", fmt(camera["block"]["seconds"] / 60, "{:.1f}"), fmt(camera["block"]["zoom"], "{:.1f}"),
             fmt(camera["block"]["aim_speed"]), fmt(camera["block"]["error"]), "", "")]
    rows += [("gaps " + row["name"], fmt(row["seconds"] / 60, "{:.1f}"), fmt(row["zoom"], "{:.1f}"), fmt(row["aim_speed"]),
              fmt(row["error"]), pct(row["edge_before"]), pct(row["edge_after"])) for row in camera["buckets"]]
    report.table(("where", "minutes", "zoom", "aim speed", "reproj. error", "at edge before", "after"), rows)
    report.text("For 'ball seen' the edge column is the share of frames with the ball at the edge.")

    report.section("What to tell Claude")
    report.text(
        "Pick the sentences that fit and send them without figures:",
        "- Live share of all gap time (part 2) under about 15 %: 'Test 7: the ball gaps are mostly dead play;",
        "  raise the D prior for long gaps.'",
        "- Over about 40 %: 'Test 7: the ball gaps hide a lot of live play; team changes without the ball",
        "  come first.'",
        "- In between: 'Test 7: the gaps are mixed, as the degrader assumes; no change.'",
        "- Long gaps unmixing to about zero while short ones do not: add 'gaps of 20 s and over are dead'.",
        "- The two estimates (sim, block) further apart than the check's worst error: add 'the dead-block",
        "  reference disagrees with the simulated one'.",
        "- Visibility rising and falling with the players' speed across windows (r above about 0.5): add",
        "  'visibility follows movement'; near zero: 'visibility does not follow movement'.",
        "- The camera zooming in or standing still in long gaps, against the ball-seen row: add 'in long gaps",
        "  the camera zooms in' or '... stands still'.")
    return report.render()


def run(args, on_stage):
    on_stage("output")
    operate.prepare_report(args.report, args.tactical_dir)
    on_stage("load")
    match = load_competition(args.tactical_dir, with_homography=True)
    on_stage("clean")
    cleaned = clean(match)
    on_stage("gaps")
    table = probes.gap_table(match, cleaned)
    on_stage("windows")
    windows = probes.visibility_windows(match, probes.player_speed(cleaned))
    on_stage("camera")
    camera = probes.camera_by_gap(match, probes.camera_state(match, probes.metres_matrices(match)))
    on_stage("reference")
    tables = references(args.idsse, args.copies)
    on_stage("report")
    operate.write_report(args.report, render(args, table, windows, camera, tables))
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
