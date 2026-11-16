"""Test 8 on the competition match: integrity, PASS or FAIL per check.

    .venv/bin/python -I scripts/match_test8.py <tactical_dir> <baseline_raw.csv> <report.txt>
    .venv/bin/python -I scripts/match_test8.py twin/J03WMX/data/tactical <twin baseline.csv> results/twin/test8.txt

<tactical_dir> holds BAR-ATH_tactical_atd.xml, _atd.txt and _homography.csv; <baseline_raw.csv> is the
published baseline, upstream/evaluation/results/baseline_raw.csv. The checks:

    metadata halves              the half boundaries in the metadata, where it has them, are the published ones
    baseline file                the baseline reads as a frame,label file with one row per frame
    baseline first half          our parser, put through the baseline's rule (upstream/starter/baseline.py,
    baseline second half         step 1: the team of the tracked slot nearest to the ball, D without a
    baseline outside the halves  ball), gives baseline_raw.csv's label on every frame; frames where the
                                 nearest home and away players are within 1 mm are excused, because our
                                 parser stores float32
    kickoff first half           on the published kickoff frame the teams stand in their own halves and,
    kickoff second half          where the ball is seen, it is on the centre spot and leaves it soon after
    attack direction             the metadata's attack direction, where it has one, is the tracking's
    homography                   the seen ball falls inside the image of its own frame (95 % or more)

It prints one line per check, `PASS <check>`, `FAIL <check>: <reason>` or `SKIP <check>: <reason>`, with
no figures, and exits 1 if any check fails; those lines may be shared. The figures behind them go to
<report.txt>, which is for the operator's eyes only. A failure before the checks prints one line,
FAIL <stage>: <reason>. When the bundle lies outside this repository, the report must lie outside it too.
--verbose shows library output and tracebacks, for the operator's own eyes only.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline import operate, probes  # noqa: E402
from pipeline.epts import (COMPETITION_FRAMES, COMPETITION_HALVES, COMPETITION_STEM,  # noqa: E402
                           COMPETITION_SUFFIXES, load_competition, read_homography, read_metadata)
from pipeline.operate import SAFE_ERRORS, OperatorError, fmt, pct  # noqa: E402

TITLE = "Test 8: integrity"
HALF_NAMES = {1: "first half", 2: "second half"}


def bundle_paths(tactical_dir):
    return {name: os.path.join(tactical_dir, COMPETITION_STEM + suffix) for name, suffix in COMPETITION_SUFFIXES.items()}


def count(n):
    """A count that cannot be mistaken for a frame number: no integer of five digits or more."""
    return "{}".format(n) if n < 10000 else "{:.1f} thousand".format(n / 1000.0)


def frames_apart(d, fps=25.0):
    return "{:+d} frames".format(d) if abs(d) < 10000 else "{:+.1f} min".format(d / fps / 60.0)


def metadata_halves(meta):
    """(status, reason, lines) for the metadata's own half boundaries."""
    parameters = meta["global_parameters"]
    names = ("first_half_start", "first_half_end", "second_half_start", "second_half_end")
    if not all(parameters.get(name) for name in names):
        return "SKIP", "the metadata has no half boundaries", ["the metadata has no half boundaries"]
    try:
        values = [int(parameters[name]) - 1 for name in names]
    except ValueError:
        return "FAIL", "the metadata's half boundaries cannot be read", []
    published = [COMPETITION_HALVES[0][1], COMPETITION_HALVES[0][2], COMPETITION_HALVES[1][1], COMPETITION_HALVES[1][2]]
    differences = [v - p for v, p in zip(values, published)]
    lines = ["{:<18} minus the published boundary: {}".format(name, frames_apart(d))
             for name, d in zip(names, differences)]
    if any(differences):
        return "FAIL", "the metadata's half boundaries differ from the published ones", lines
    return "PASS", None, lines


def run(args, on_stage):
    on_stage("output")
    operate.prepare_report(args.report, args.tactical_dir, inputs=(args.baseline,))
    on_stage("load")
    paths = bundle_paths(args.tactical_dir)
    meta = read_metadata(paths["meta"])
    match = load_competition(args.tactical_dir, with_homography=False)
    if not np.array_equal(match.frame, np.arange(COMPETITION_FRAMES)):
        raise OperatorError("frame numbers are not 0..{} in order".format(COMPETITION_FRAMES - 1))

    report = operate.Report(TITLE, args.tactical_dir, operate.fingerprint(__file__, probes.__file__))
    results = []

    def attempt(name, check):
        on_stage(name)
        try:
            status, reason = check()
        except Exception as error:
            status = "FAIL"
            reason = str(error) if isinstance(error, SAFE_ERRORS) else type(error).__name__
            report.text("{}: stopped by {}".format(name, type(error).__name__))
        results.append((status, name, reason))

    def halves_check():
        status, reason, lines = metadata_halves(meta)
        report.section("Metadata halves")
        report.text(*lines)
        return status, reason

    attempt("metadata halves", halves_check)

    theirs = []

    def file_check():
        report.section("Baseline file")
        labels = probes.read_label_csv(args.baseline, COMPETITION_FRAMES)
        counts = {label: int((labels == label).sum()) for label in "HADX"}
        report.text("read {}: one row per frame, labels H {} / A {} / D {} / X {}".format(
            os.path.basename(args.baseline), *[pct(counts[k] / len(labels)) for k in "HADX"]))
        theirs.append(labels)
        if counts["X"]:
            return "FAIL", "the baseline has X labels, which baseline_raw.csv never writes"
        return "PASS", None

    attempt("baseline file", file_check)

    comparison = {}

    def compare():
        codes, tie_gap = probes.nearest_player_codes(match)
        ours = probes.baseline_labels(codes)
        regions = {"first half": match.period == 1, "second half": match.period == 2, "outside the halves": match.period == 0}
        comparison.update(probes.compare_labels(ours, theirs[0], tie_gap, regions))
        layouts = []
        for first, last, _ in meta["specs"]:
            mask = (match.frame >= first - 1) & (match.frame <= last - 1)
            layouts.append(probes.compare_labels(ours, theirs[0], tie_gap, {"layout": mask})["layout"])
        report.section("Baseline, frame by frame")
        report.text("our parser through the baseline's rule against {}".format(os.path.basename(args.baseline)))
        report.table(("region", "minutes", "differ", "near ties", "unexplained", "of them ball seen by one only"), [
            (name, fmt(row["frames"] / match.fps / 60, "{:.1f}"), count(row["differ"]), count(row["near_ties"]),
             count(row["unexplained"]), count(row["ball_presence"])) for name, row in comparison.items()])
        report.text("", "layouts (DataFormatSpecification) in the metadata: {}".format(len(layouts)))
        if len(layouts) > 1:
            report.table(("layout", "minutes", "differ", "unexplained"), [
                ("{} of {}".format(i + 1, len(layouts)), fmt(row["frames"] / match.fps / 60, "{:.1f}"),
                 pct(row["differ"] / max(1, row["frames"]), 3), count(row["unexplained"])) for i, row in enumerate(layouts)])

    def region_check(name):
        def check():
            if not theirs:
                return "SKIP", "no baseline to compare with"
            if not comparison:
                compare()
            row = comparison[name]
            if row["unexplained"]:
                return "FAIL", "our labels differ from the baseline's beyond float32 near ties"
            return "PASS", None
        return check

    for name in ("first half", "second half", "outside the halves"):
        attempt("baseline " + name, region_check(name))

    signs = probes.attack_signs(match)
    report.section("Kickoffs on the published frames")
    report.text("attack direction from the tracking: home attacks towards {} x in the first half".format(
        {1: "+", -1: "-"}.get(int(signs[1]), "no")))

    def kickoff(half, frame):
        def check():
            ok, reason, d = probes.kickoff_check(match, half, frame, signs)
            report.text("{}: players in their own half {} ({} tracked per frame); ball near the frame: {}; in the".format(
                HALF_NAMES[half], pct(d.get("formation_share")), fmt(d.get("tracked_per_frame"), "{:.1f}"),
                "{} m from the spot".format(fmt(d["ball_spot_m"], "{:.2f}")) if d["ball_seen"] else "not seen"),
                "    two seconds after it: {}".format(
                    ("leaves the spot" if d["ball_leaves"] else "stays on the spot") if d["ball_seen_after"] else "not seen"))
            return ("PASS", None) if ok else ("FAIL", reason)
        return check

    for half, frame in probes.KICKOFFS:
        attempt("kickoff " + HALF_NAMES[half], kickoff(half, frame))

    def direction():
        report.section("Attack direction")
        sign, reason = probes.metadata_attack_sign(match.extras.get("attack_direction_first_half", ()))
        report.text("metadata: {}".format(" / ".join(str(v) for v in match.extras.get("attack_direction_first_half", ()))))
        if reason:
            return "FAIL", reason
        if sign is None:
            return "SKIP", "the metadata names no attack direction"
        if signs[1] == 0:
            return "FAIL", "the tracking gives no attack direction"
        if sign != signs[1]:
            return "FAIL", "the metadata's attack direction disagrees with the tracking"
        return "PASS", None

    attempt("attack direction", direction)

    def homography():
        report.section("Homography against the tracking")
        matrices, error, valid = read_homography(paths["homography"], match.n_frames)
        match.extras.update(homography=matrices, homography_error=error, homography_valid=valid)
        shares = probes.ball_in_image_by_shift(match, probes.metres_matrices(match))
        report.text("valid rows {}; share of the seen ball inside the image with the homography taken N frames later:".format(
            pct(float(np.mean(valid)))))
        report.table(("shift, frames",) + tuple("{:+d}".format(s) for s in shares), [
            ("inside the image",) + tuple(pct(v, 2) for v in shares.values())])
        ok, reason = probes.homography_aligned(shares)
        return ("PASS", None) if ok else ("FAIL", reason)

    attempt("homography", homography)

    report.section("What to tell Claude")
    report.text(
        "The PASS/FAIL/SKIP lines the script printed can be pasted as they are; they carry no figures.",
        "If a baseline check fails, add in words where it goes wrong, e.g. 'only in the second half' or",
        "'from the second layout on', never a count or a frame. A failing kickoff or homography check is",
        "best followed by a look at the video before anything else changes.")
    on_stage("report")
    operate.write_report(args.report, report.render())
    return ["{} {}{}".format(status, name, ": " + reason if reason else "") for status, name, reason in results]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tactical_dir", help="folder holding BAR-ATH_tactical_atd.xml, _atd.txt and _homography.csv")
    parser.add_argument("baseline", help="the published baseline_raw.csv")
    parser.add_argument("report", help="text file for the figures, outside the repository for the match")
    parser.add_argument("--verbose", action="store_true",
                        help="library output and tracebacks; never share it from a run on the match")
    args = parser.parse_args()
    return operate.run(lambda on_stage: run(args, on_stage), args.verbose)


if __name__ == "__main__":
    sys.exit(main())
