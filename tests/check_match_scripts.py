"""Self-checks for the scripts the operator runs on the competition match: scripts/match_test7.py,
match_test8.py, match_test9.py, match_parallax.py, and pipeline/probes.py and pipeline/operate.py
behind them.

    python -I tests/check_match_scripts.py [--twin twin/J03WMX] [--idsse external/processed/idsse] [--json out.json]

1. Geometry: the per-frame camera solve gives pipeline/camera.solve_camera's answer on the DATA.md row
   and the true centre for any pan and zoom; the homography rebuilt from a view polygon projects like
   the camera; the aim point is the polygon's diagonal crossing.
2. Parallax signatures on IDSSE with the DATA.md camera, live frames: with parallax the fast steps run
   along the ray and the ball lands beyond the far line, as test 3 found; without, neither.
3. Camera aim on a synthetic camera that centres the ball (x error 0, best lag 0) and one that
   follows a second late (best lag +1 s), and on the twin, whose camera follows the ball late.
4. Test 7's live-share estimate on degraded IDSSE copies, one left out at a time, and on the twin
   against its truth.
5. Test 8's parts: the baseline's rule from an independent text parser (upstream/starter/baseline.py
   read, never run) equals our parser's on the twin; the label reader refuses broken files; the
   kickoff rule passes the twin's and the IDSSE kickoffs that start on the kick and fails frames moved
   by a second or more; the attack direction convention holds on the twin and on Metrica Sample Game 3.
6. The four scripts end to end on the twin: one PASS line (test 8: one PASS/SKIP line per check, no
   digits), nothing on stderr, a report with the notice on top and no five-digit integer in it (no
   frame index can hide there), the same bytes on a second run; a corrupted or broken baseline fails the
   right checks; a report inside the repository for a bundle outside it is refused, and a missing
   bundle fails in one line.

Open data and the twin only. Exit code 1 if a check fails.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import replace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis.spellmetric import read_labels  # noqa: E402
from pipeline import probes  # noqa: E402
from pipeline.camera import DATA_MD_PITCH, DATA_MD_ROW, PanCamera, apparent_ground, project, solve_camera  # noqa: E402
from pipeline.clean import clean  # noqa: E402
from pipeline.epts import COMPETITION_FRAMES, COMPETITION_PITCH, load_competition, load_epts  # noqa: E402
from pipeline.idsse import load_idsse  # noqa: E402
from pipeline.schema import FormatError  # noqa: E402

SCRIPTS = {name: os.path.join(ROOT, "scripts", "match_{}.py".format(name)) for name in ("test7", "test8", "test9", "parallax")}
TEST8_CHECKS = ("metadata halves", "baseline file", "baseline first half", "baseline second half",
                "baseline outside the halves", "kickoff first half", "kickoff second half", "attack direction", "homography")
NOTICE = "FOR THE OPERATOR'S EYES ONLY."
BIG_INTEGER = re.compile(r"\d{5,}|\d{1,3}(?:,\d{3})+")
results = []


def check(name, ok, detail=""):
    results.append({"name": name, "ok": bool(ok), "detail": str(detail)})
    print("{:4} {:<78} {}".format("PASS" if ok else "FAIL", name, detail), flush=True)


# ---------------------------------------------------------------- 1. geometry

def check_geometry():
    print("--- geometry", flush=True)
    camera = solve_camera()
    row = np.asarray(DATA_MD_ROW).reshape(1, 3, 3) @ probes.metres_to_normalised(DATA_MD_PITCH)
    centres, principal_x = probes.camera_centres(row)
    check("camera solve on the DATA.md row = solve_camera()",
          np.abs(centres[0] - camera.centre).max() < 1e-9 and abs(principal_x - camera.principal[0]) < 1e-6,
          "{:.1e} m".format(np.abs(centres[0] - camera.centre).max()))
    rng = np.random.default_rng(7)
    pans = np.concatenate((np.linspace(-1.0, 1.0, 201), rng.uniform(-1.0, 1.0, 300)))   # headings -59..+56 deg
    worst_centre = worst_px = worst_dlt_centre = worst_aim = 0.0
    for zoom in (0.7, 1.0, 1.3):
        zoomed = replace(camera, focal=camera.focal * zoom)
        matrices = zoomed.matrices(pans)
        worst_centre = max(worst_centre, np.abs(probes.camera_centres(matrices)[0] - camera.centre).max())
        polygons = zoomed.view_polygons(pans)
        rebuilt = probes.matrices_from_polygon(polygons)
        points = rng.uniform((-52, -34), (52, 34), (len(pans), 2))
        worst_px = max(worst_px, np.nanmax(np.abs(project(matrices, points) - project(rebuilt, points))))
        worst_dlt_centre = max(worst_dlt_centre, np.nanmax(np.abs(probes.camera_centres(rebuilt)[0] - camera.centre)))
        aim = probes.ground_of_pixel(probes.safe_inverse(matrices), 960.0, 540.0)
        p = polygons.astype(np.float64)
        d1, d2 = p[:, 2] - p[:, 0], p[:, 3] - p[:, 1]
        cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
        t = ((p[:, 1] - p[:, 0])[:, 0] * d2[:, 1] - (p[:, 1] - p[:, 0])[:, 1] * d2[:, 0]) / cross
        worst_aim = max(worst_aim, np.abs(p[:, 0] + t[:, None] * d1 - aim).max())
    check("camera solve: true centre for every pan (45 degrees included) and zoom", worst_centre < 1e-6,
          "{:.1e} m".format(worst_centre))
    check("homography from the view polygon (float32 corners) projects like the camera", worst_px < 0.05,
          "{:.1e} px".format(worst_px))
    check("homography from the view polygon gives the camera centre", worst_dlt_centre < 1e-3,
          "{:.1e} m".format(worst_dlt_centre))
    check("aim point = crossing of the view polygon's diagonals", worst_aim < 1e-3, "{:.1e} m".format(worst_aim))
    matrices = camera.matrices(pans)
    inverse = probes.safe_inverse(matrices)
    v = rng.uniform(600, 1080, len(pans))
    on_column = np.stack([probes.ground_of_pixel(inverse[i:i + 1], 960.0, v[i])[0] for i in range(len(pans))])
    aim, foot = probes.ground_of_pixel(inverse, 960.0, 540.0), probes.ground_of_pixel(inverse, 960.0, 1080.0)
    error = np.abs(probes.x_on_line(aim, foot, on_column[:, 1]) - on_column[:, 0]).max()
    check("a point on the image's centre column has no x error", error < 1e-6, "{:.1e} m".format(error))


# ---------------------------------------------------------------- 2. parallax on IDSSE

def check_parallax(idsse):
    print("--- parallax signatures on IDSSE, DATA.md camera, live frames", flush=True)
    camera = solve_camera()
    for name in ("J03WMX", "J03WN1"):
        match = load_idsse(os.path.join(idsse, name + ".npz"))
        live = (match.labels != "D") & (match.period > 0)
        true_xy = match.ball_xy.astype(np.float64)
        matrices = camera.matrices(camera.pan_to_face(true_xy[:, 0], true_xy[:, 1]))
        out = {}
        for variant, xy in (("parallax", apparent_ground(true_xy, match.ball_z, camera.centre)), ("none", true_xy)):
            ball = np.where(live[:, None], xy, np.nan).astype(np.float32)
            copy = replace(match, ball_xy=ball, ball_z=None)
            out[variant] = probes.parallax_measurements(copy, clean(copy), matrices)
        p, n = out["parallax"], out["none"]
        check("{} camera solved on every frame, at the true centre".format(name),
              p["camera_usable"] == 1.0 and np.allclose(p["camera_median"], camera.centre, atol=1e-6),
              "usable {:.3f}".format(p["camera_usable"]))
        cos = p["rays"]["per-frame camera"]["median_cos"]
        check("{} parallax: fast steps run along the ray (median |cos| > 0.9)".format(name), cos > 0.9,
              "{:.3f} over {} steps; none: {} steps".format(cos, p["fast_steps"], n["fast_steps"]))
        check("{} parallax: beyond the far line on over 2 % of live frames, without under 0.1 %".format(name),
              p["beyond_far"] > 0.02 and n["beyond_far"] < 0.001,
              "{:.2%} / {:.3%}, near line {:.3%}".format(p["beyond_far"], n["beyond_far"], p["beyond_near"]))
        check("{} parallax: the gate removes over 1 %, without under 0.2 %".format(name),
              p["gate_removed"] > 0.01 and n["gate_removed"] < 0.002,
              "{:.2%} / {:.3%}".format(p["gate_removed"], n["gate_removed"]))


# ---------------------------------------------------------------- 3. camera aim

def check_aim(idsse, twin):
    print("--- camera aim", flush=True)
    camera = solve_camera()
    centred = PanCamera(focal=camera.focal, principal=(960.0, 540.0), rotation=camera.rotation,
                        centre=camera.centre, image_size=camera.image_size)
    match = load_idsse(os.path.join(idsse, "J03WMX.npz"))
    cleaned = clean(match)
    xy = match.ball_xy.astype(np.float64)
    pan = centred.pan_to_face(xy[:, 0], xy[:, 1])
    exact = probes.aim_measurements(cleaned, centred.matrices(pan))
    check("a camera centred on the ball: x error about 0 (its slight roll), best lag 0",
          exact["seen"]["x_median"] < 0.1 and exact["best_lag_s"] == 0.0,
          "{:.3f} m, lag {:+.1f} s".format(exact["seen"]["x_median"], exact["best_lag_s"]))
    late = np.concatenate((np.repeat(pan[:1], 25), pan[:-25]))
    lagged = probes.aim_measurements(cleaned, centred.matrices(late))
    check("a camera a second late: best lag +1 s", lagged["best_lag_s"] == 1.0,
          "lag {:+.1f} s, {:.2f} m there".format(lagged["best_lag_s"], lagged["best_lag_x_median"]))
    t = load_competition(os.path.join(twin, "data", "tactical"), with_homography=True)
    aim = probes.aim_measurements(clean(t), probes.metres_matrices(t))
    check("twin (its camera smooths the ball 2 s and reacts 1 s late): best lag +1.5..+3.5 s",
          1.5 <= aim["best_lag_s"] <= 3.5, "{:+.1f} s".format(aim["best_lag_s"]))
    check("twin: the aim beats the players' centroid and pans with the ball",
          aim["seen"]["ratio"] < 1.0 and aim["pan_follows_share"] > 0.8,
          "ratio {:.2f}, pans the same way {:.0%}".format(aim["seen"]["ratio"], aim["pan_follows_share"]))
    shares = probes.ball_in_image_by_shift(t, probes.metres_matrices(t))
    check("twin: the seen ball is inside its own frame's image", probes.homography_aligned(shares)[0],
          "{:.2%}".format(shares[0]))
    return t


# ---------------------------------------------------------------- 4. test 7's estimate

def check_estimate(idsse, twin_match, twin):
    print("--- test 7: live share in gaps from player speed", flush=True)
    tables = [probes.gap_table(copy, cleaned, labels=copy.labels)
              for _, copy, cleaned, _ in probes.simulated_copies(idsse, "camera")]
    validation = probes.validate_estimator(tables)
    check("copies, one left out: mean error under 10 points, bias under 5",
          validation["mean_abs_error"] < 0.10 and abs(validation["bias"]) < 0.05,
          "mean {:.3f}, worst {:.3f}, bias {:+.3f}, truth {:.2f}-{:.2f}, {} copies".format(
              validation["mean_abs_error"], validation["max_abs_error"], validation["bias"],
              validation["truth_low"], validation["truth_high"], validation["copies"]))
    truth = np.array(["H", "A", "D", "X"])[read_labels(os.path.join(twin, "gt", "possession_gt_full.csv"))]
    cleaned = clean(twin_match)
    table = probes.gap_table(twin_match, cleaned, labels=truth)
    estimate = probes.estimate_live_share(table, probes.gap_references(tables))
    real = probes.true_live_share(table)
    check("twin: the estimate within 15 points of its truth", abs(estimate["overall"] - real) < 0.15,
          "{:.3f} against {:.3f}".format(estimate["overall"], real))
    check("twin: gaps of 60 s and over unmix to almost all dead, like its truth",
          estimate["buckets"][-1] < 0.2 and table["buckets"][-1]["live_share"] < 0.2,
          "{:.2f} against {:.2f}".format(estimate["buckets"][-1], table["buckets"][-1]["live_share"]))
    windows = probes.visibility_windows(twin_match, probes.player_speed(cleaned))
    covered = sum(w["to_min"] - w["from_min"] for w in windows["windows"]) * 60 * twin_match.fps
    play = int(probes.play_mask(twin_match).sum())
    check("twin: the 5-minute windows cover the play exactly once", abs(covered - play) < 1e-3,
          "{} windows".format(len(windows["windows"])))


# ---------------------------------------------------------------- 5. test 8's parts

def baseline_codes_from_text(meta_path, raw_path, n_frames, pitch):
    """The rule of upstream/starter/baseline.py, step 1, written again from its text with plain string
    handling and float64: slots in the order of the raw line, teams in the order of the metadata's
    Player elements, the first team home, the first slot wins a tie. Nothing from pipeline/epts.py."""
    root = ET.parse(meta_path).getroot()
    home_id = [team.get("id") for team in root.iter("Team")][0]
    team_of = [0 if player.get("teamId") == home_id else 1 for player in root.iter("Player")]
    length, width = pitch
    out = np.full(n_frames, -1, dtype=np.int8)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3 or parts[2][0] == "N":
                continue
            bx, by = (float(v) for v in parts[2].split(",")[:2])
            bx, by = bx * length, by * width
            best, best_team = None, -1
            for slot, group in enumerate(parts[1].split(";")):
                if group[0] == "N":
                    continue
                x, y = (float(v) for v in group.split(",")[:2])
                distance = (x * length - bx) ** 2 + (y * width - by) ** 2
                if best is None or distance < best:
                    best, best_team = distance, team_of[slot]
            out[int(parts[0]) - 1] = best_team
    return out


def write_labels(path, labels):
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))


def check_integrity(idsse, twin, twin_match, metrica, tmp):
    print("--- test 8's parts", flush=True)
    tactical = os.path.join(twin, "data", "tactical")
    started = time.time()
    text_codes = baseline_codes_from_text(os.path.join(tactical, "BAR-ATH_tactical_atd.xml"),
                                          os.path.join(tactical, "BAR-ATH_tactical_atd.txt"),
                                          COMPETITION_FRAMES, COMPETITION_PITCH)
    codes, tie_gap = probes.nearest_player_codes(twin_match)
    differ = codes != text_codes
    ties = differ & (tie_gap < probes.NEAR_TIE_M)
    check("twin: the baseline's rule, our parser = an independent text parser",
          int((differ & ~ties).sum()) == 0 and int(ties.sum()) <= 20,
          "{} frames differ, {} of them near ties, {:.0f}s".format(int(differ.sum()), int(ties.sum()), time.time() - started))
    baseline = probes.baseline_labels(text_codes)
    path = os.path.join(tmp, "twin_baseline_raw.csv")
    write_labels(path, baseline)
    check("label reader: a written file reads back the same",
          np.array_equal(probes.read_label_csv(path, COMPETITION_FRAMES), baseline))
    broken = {"header": lambda lines: ["frame,lab"] + lines[1:], "short": lambda lines: lines[:-1],
              "order": lambda lines: lines[:1] + [lines[2], lines[1]] + lines[3:],
              "label": lambda lines: lines[:5] + ["4,Q"] + lines[6:]}
    with open(path) as handle:
        lines = handle.read().splitlines()
    refused = []
    for name, mutate in broken.items():
        bad = os.path.join(tmp, "broken_{}.csv".format(name))
        with open(bad, "w") as handle:
            handle.write("\n".join(mutate(list(lines))) + "\n")
        try:
            probes.read_label_csv(bad, COMPETITION_FRAMES)
        except FormatError as error:
            refused.append(not re.search(r"\d{3,}", str(error)))
    check("label reader: refuses a wrong header, a missing row, rows out of order, a bad label",
          len(refused) == len(broken) and all(refused), "{} of {}".format(len(refused), len(broken)))

    signs = probes.attack_signs(twin_match)
    outcomes = [probes.kickoff_check(twin_match, half, frame, signs)[0] for half, frame in probes.KICKOFFS]
    check("twin: both kickoffs on the published frames", all(outcomes), outcomes)
    moved = [probes.kickoff_check(twin_match, half, frame + int(s * twin_match.fps), signs)[0]
             for half, frame in probes.KICKOFFS for s in (1, 3, 10)]
    check("twin: kickoff frames moved by 1, 3 or 10 s fail", not any(moved), moved)
    on_spot, passed, late_failed, late_total = 0, 0, 0, 0
    for path_ in sorted(glob.glob(os.path.join(idsse, "*.npz"))):
        match = load_idsse(path_)
        signs = probes.attack_signs(match)
        for half in (1, 2):
            first = int(match.frame[match.period == half][0])
            ok, _, details = probes.kickoff_check(match, half, first, signs)
            if details.get("ball_seen") and details["ball_spot_m"] <= probes.SPOT_M:
                on_spot += 1
                passed += ok
            late_total += 1
            late_failed += not probes.kickoff_check(match, half, first + int(match.fps), signs)[0]
    check("IDSSE: every half that starts on the kick passes; a second later every one fails",
          on_spot >= 3 and passed == on_spot and late_failed == late_total,
          "{} of {} start on the kick, {} pass; {} of {} fail a second later".format(
              on_spot, late_total, passed, late_failed, late_total))

    sign, reason = probes.metadata_attack_sign(twin_match.extras["attack_direction_first_half"])
    check("twin: metadata attack direction = the tracking's", sign == probes.attack_signs(twin_match)[1] and reason is None,
          twin_match.extras["attack_direction_first_half"])
    if os.path.isdir(metrica):
        game = load_epts(os.path.join(metrica, "Sample_Game_3_metadata.xml"),
                         os.path.join(metrica, "Sample_Game_3_tracking.txt"), match_id="metrica3")
        sign, reason = probes.metadata_attack_sign(game.extras["attack_direction_first_half"])
        check("Metrica game 3: left_to_right means +x (metadata = tracking)",
              sign == probes.attack_signs(game)[1] and reason is None, game.extras["attack_direction_first_half"])
    check("attack direction: none named, or both teams the same, is caught",
          probes.metadata_attack_sign((None, None)) == (None, None)
          and probes.metadata_attack_sign(("left_to_right", "left_to_right"))[0] is None
          and probes.metadata_attack_sign(("left_to_right", "left_to_right"))[1] is not None)
    return path, baseline


# ---------------------------------------------------------------- 6. the scripts end to end

def run_script(name, *arguments):
    started = time.time()
    proc = subprocess.run([sys.executable, "-I", SCRIPTS[name]] + list(arguments), capture_output=True, text=True)
    return proc.returncode, proc.stdout.splitlines(), proc.stderr, time.time() - started


def report_ok(path):
    """(ok, detail): the notice on top, a closing section, no integer of five digits or more."""
    if not os.path.isfile(path):
        return False, "no report"
    with open(path) as handle:
        text = handle.read()
    body = [line for line in text.splitlines() if not line.startswith(("bundle ", "code "))]
    big = [m.group(0) for line in body for m in BIG_INTEGER.finditer(line)]
    ok = NOTICE in text.splitlines()[3] and "What to tell Claude" in text and not big and not os.path.exists(path + ".tmp")
    return ok, "{} lines{}".format(len(text.splitlines()), ", big integers: " + " ".join(big[:5]) if big else "")


def check_scripts(twin, baseline_path, baseline, tmp):
    print("--- the scripts on the twin", flush=True)
    tactical = os.path.join(twin, "data", "tactical")
    outdir = os.path.join(ROOT, "results", "twin")
    os.makedirs(outdir, exist_ok=True)
    reports = {name: os.path.join(outdir, "check_{}.txt".format(name)) for name in SCRIPTS}
    for path in reports.values():
        for p in (path, path + ".tmp"):
            if os.path.exists(p):
                os.remove(p)
    seconds = {}
    for name in ("test7", "test9", "parallax"):
        code, out, err, seconds[name] = run_script(name, tactical, reports[name])
        check("{}: exit 0, exactly PASS on stdout, nothing on stderr".format(name),
              code == 0 and out == ["PASS"] and err == "", "{} | {:.0f}s".format(" | ".join(out), seconds[name]))
        check("{}: report has the notice, a closing section, no five-digit integer".format(name), *report_ok(reports[name]))

    code, out, err, seconds["test8"] = run_script("test8", tactical, baseline_path, reports["test8"])
    names = [re.sub(r"^(PASS|SKIP) ", "", line).split(":")[0] for line in out]
    shape = code == 0 and err == "" and names == list(TEST8_CHECKS) and all(line.startswith(("PASS ", "SKIP ")) for line in out)
    check("test8: exit 0, one PASS or SKIP line per check, nothing on stderr", shape,
          "{} lines, {:.0f}s".format(len(out), seconds["test8"]))
    check("test8: no digit on stdout", not any(re.search(r"\d", line) for line in out))
    check("test8: the baseline checks pass", all("PASS baseline" in line for line in out if "baseline" in line), out[1:5])
    check("test8: report has the notice, a closing section, no five-digit integer", *report_ok(reports["test8"]))

    second = os.path.join(outdir, "check_test7_again.txt")
    first = os.path.join(outdir, "check_test7_once.txt")
    for path in (first, second):
        run_script("test7", tactical, path, "--copies", "1")
    with open(first, "rb") as a, open(second, "rb") as b:
        check("test7: a second run writes the same bytes", a.read() == b.read())

    flipped = baseline.copy()
    rows = np.flatnonzero((twin_seen_frames(twin) & (flipped == "H")) & (np.arange(len(flipped)) > 120000))
    flipped[rows[0]] = "A"
    corrupt = os.path.join(tmp, "baseline_flipped.csv")
    write_labels(corrupt, flipped)
    code, out, _, _ = run_script("test8", tactical, corrupt, reports["test8"])
    check("test8: one label flipped in the second half fails that check only",
          code == 1 and "FAIL baseline second half: our labels differ from the baseline's beyond float32 near ties" in out
          and "PASS baseline first half" in out and "PASS baseline outside the halves" in out, [l for l in out if "baseline" in l])
    bad = os.path.join(tmp, "broken_header.csv")
    code, out, _, _ = run_script("test8", tactical, bad, reports["test8"])
    check("test8: a broken baseline file fails it and skips the comparisons",
          code == 1 and any(l.startswith("FAIL baseline file: ") for l in out)
          and sum(l.startswith("SKIP baseline ") for l in out) == 3, [l for l in out if "baseline" in l])

    outside = "/nonexistent-bundle-outside-the-repo"
    refused = os.path.join(outdir, "refused.txt")
    for name in SCRIPTS:
        arguments = (outside, baseline_path, refused) if name == "test8" else (outside, refused)
        code, out, _, _ = run_script(name, *arguments)
        check("{}: a report inside the repo for a bundle outside it is refused".format(name),
              code == 1 and out == ["FAIL output: the bundle is outside the repository, so the report must be too"]
              and not os.path.exists(refused), " | ".join(out))
    missing = os.path.join(twin, "no-such-folder")
    for name in SCRIPTS:
        arguments = (missing, baseline_path, refused) if name == "test8" else (missing, refused)
        code, out, _, _ = run_script(name, *arguments)
        check("{}: a missing bundle fails in one line".format(name),
              code == 1 and len(out) == 1 and out[0].startswith("FAIL load: ") and not os.path.exists(refused), " | ".join(out))
    for path in list(reports.values()) + [first, second]:
        if os.path.exists(path):
            os.remove(path)
    return seconds


def twin_seen_frames(twin):
    match = load_competition(os.path.join(twin, "data", "tactical"), with_homography=False)
    return np.isfinite(match.ball_xy[:, 0])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--twin", default=os.path.join(ROOT, "twin", "J03WMX"))
    parser.add_argument("--idsse", default=os.path.join(ROOT, "external", "processed", "idsse"))
    parser.add_argument("--metrica", default=os.path.join(ROOT, "external", "metrica", "data", "Sample_Game_3"))
    parser.add_argument("--json")
    args = parser.parse_args()
    started = time.time()
    check_geometry()
    check_parallax(args.idsse)
    twin_match = check_aim(args.idsse, args.twin)
    check_estimate(args.idsse, twin_match, args.twin)
    with tempfile.TemporaryDirectory() as tmp:
        baseline_path, baseline = check_integrity(args.idsse, args.twin, twin_match, args.metrica, tmp)
        seconds = check_scripts(args.twin, baseline_path, baseline, tmp)
    failed = [r for r in results if not r["ok"]]
    print("{} checks, {} failed, {:.0f}s".format(len(results), len(failed), time.time() - started), flush=True)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump({"checks": results, "seconds": seconds}, handle, indent=1)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
