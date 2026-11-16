"""Self-checks for the camera and the degrader, on IDSSE only.

    python -I tests/check_degrade.py [--idsse external/processed/idsse] [--matches J03WMX,J03WN1]
        [--json results.json]

The degrader refuses unlabelled data and load_idsse reads only IDSSE archives, so nothing here can run
on the competition's match. Floors marked "first run" come from the first full run on this data: they
guard against regressions and say nothing about the competition's match. Exit code 1 if a check fails.
"""
import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline.camera import DATA_MD_PITCH, DATA_MD_ROW, apparent_ground, project, solve_camera  # noqa: E402
from pipeline.clean import clean  # noqa: E402
from pipeline.degrade import SIMULATORS, DegradeConfig, degrade  # noqa: E402
from pipeline.epts import view_polygon_from_homography  # noqa: E402
from pipeline.idsse import load_idsse  # noqa: E402
from pipeline.schema import SchemaError  # noqa: E402

LIVE_SHARES = (0.1, 0.3, 0.5)
MISSING_TOLERANCE = 0.02
LIVE_SHARE_TOLERANCE = 0.05
KEEPER_RECALL_FLOOR = {"camera": 0.70, "plain": 0.85}  # pooled over copies, first run: 0.79 and 0.91
results = []


def check(name, ok, detail=""):
    results.append({"name": name, "ok": bool(ok), "detail": str(detail)})
    print("{:4} {:<66} {}".format("PASS" if ok else "FAIL", name, detail), flush=True)


def note(name, detail=""):
    """A measurement worth seeing that is not a pass or fail."""
    print("{:4} {:<66} {}".format("INFO", name, detail), flush=True)


def runs(mask):
    edges = np.diff(np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def digest(*arrays):
    h = hashlib.sha256()
    for array in arrays:
        h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------- camera

def check_camera():
    camera = solve_camera()
    length, width = DATA_MD_PITCH
    printed = np.asarray(DATA_MD_ROW).reshape(3, 3)
    grid = np.array([(x, y) for x in np.linspace(-length / 2, length / 2, 9) for y in np.linspace(-width / 2, width / 2, 5)])
    normalised = np.column_stack((grid[:, 0] / length + 0.5, 0.5 - grid[:, 1] / width))
    expected = project(np.repeat(printed[None], len(grid), axis=0), normalised)
    solved = project(np.repeat(camera.matrices(0.0), len(grid), axis=0), grid)
    error = float(np.max(np.hypot(*(solved - expected).T)))
    check("camera reproduces the printed homography", error < 0.05, "max {:.4f} px over the pitch".format(error))

    height, behind = camera.centre[2], -width / 2 - camera.centre[1]
    hfov = 2 * np.degrees(np.arctan(camera.image_size[0] / 2 / camera.focal))
    check("camera stands where an independent solve put it",
          29.5 <= height <= 31.5 and 45.0 <= behind <= 50.0 and abs(camera.centre[0]) <= 3.0 and 47.0 <= hfov <= 52.0,
          "height {:.2f} m, {:.2f} m behind the near line, x {:.2f}, {:.1f} deg".format(height, behind, camera.centre[0], hfov))

    polygon = camera.view_polygons(0.0)[0]
    adapter = view_polygon_from_homography(printed[None], np.array([True]), DATA_MD_PITCH)[0]
    gap = float(np.nanmax(np.abs(polygon - adapter)))
    check("view polygon matches the competition adapter's", gap < 0.05, "max {:.4f} m".format(gap))

    rng = np.random.default_rng(3)
    count = 2000
    xyz = np.column_stack((rng.uniform(-52, 52, count), rng.uniform(-34, 34, count), rng.uniform(0, 20, count)))
    pan = rng.uniform(-0.6, 0.6, count)
    pixels = camera.pixels_3d(xyz, pan)
    back = project(np.linalg.inv(camera.matrices(pan)), pixels)
    analytic = apparent_ground(xyz[:, :2], xyz[:, 2], camera.centre)
    error = float(np.max(np.hypot(*(back - analytic).T)))
    check("ground mapping of a high point equals the pixel route", error < 1e-6, "max {:.2e} m".format(error))

    foot = camera.centre[:2]
    shift = analytic - xyz[:, :2]
    ray = xyz[:, :2] - foot
    cross = np.abs(shift[:, 0] * ray[:, 1] - shift[:, 1] * ray[:, 0]) / np.hypot(*ray.T)
    along = (shift * ray).sum(axis=1) / np.hypot(*ray.T)
    check("height moves a point straight away from the camera", float(cross.max()) < 1e-6 and (along >= -1e-9).all(),
          "max sideways {:.1e} m".format(float(cross.max())))
    flat = apparent_ground(xyz[:, :2], np.zeros(count), camera.centre)
    check("a point on the ground stays put", float(np.max(np.abs(flat - xyz[:, :2]))) < 1e-9)

    aims = [(30, 0), (-40, 20), (50, -30), (0, 34), (-52, -34)]
    miss = max(abs(project(camera.matrices(camera.pan_to_face(x, y)), np.array([[x, y]]))[0, 0] - camera.principal[0])
               for x, y in aims)
    check("pan puts the aim point on the principal column", miss < 2.0, "max {:.2f} px".format(miss))


# ---------------------------------------------------------------- contract

def check_contract(match):
    before = digest(match.ball_xy, match.players_xy, match.ball_z, match.labels.astype("S1"), match.slot_team)
    one = degrade(match, 7)
    again = degrade(match, 7)
    other = degrade(match, 8)
    after = digest(match.ball_xy, match.players_xy, match.ball_z, match.labels.astype("S1"), match.slot_team)
    check("degrade leaves its input alone", before == after)
    same = digest(one.ball_xy, one.players_xy, one.slot_team) == digest(again.ball_xy, again.players_xy, again.slot_team)
    check("the same seed gives the same bytes", same)
    check("another seed gives another copy",
          digest(one.ball_xy, one.players_xy) != digest(other.ball_xy, other.players_xy))
    check("labels, frames and halves come through unchanged",
          np.array_equal(one.labels, match.labels) and np.array_equal(one.frame, match.frame)
          and np.array_equal(one.period, match.period))
    check("the copy has no ball height, 46 slots and track names",
          one.ball_z is None and one.n_slots == 46 and one.slot_id[0] == "Track 1" and one.view_polygon is not None)
    source_keys = [key for key in one.extras if "speed" in key or key in ("ball_status", "ball_possession")]
    check("no source field survives in extras", not source_keys, ", ".join(source_keys))

    shares = [degrade(match, 7, dataclasses.replace(DegradeConfig(), live_share=v)) for v in (0.1, 0.5)]
    check("live share changes the ball only, not the players",
          digest(shares[0].players_xy, shares[0].slot_team) == digest(shares[1].players_xy, shares[1].slot_team)
          and digest(shares[0].ball_xy) != digest(shares[1].ball_xy))
    try:
        degrade(dataclasses.replace(match, labels=None), 1)
        refused = False
    except SchemaError:
        refused = True
    check("degrade refuses a match without labels", refused)
    cleaned = clean(one)
    check("the cleaning step runs on a degraded copy", cleaned.n_frames == match.n_frames)


# ---------------------------------------------------------------- one copy

def still_dead_share(ball_xy, dead, inside, fps):
    """P(dead | seen and moved under 0.5 m over the next second)."""
    window = int(fps)
    seen = np.isfinite(ball_xy[:, 0])
    gaps = np.concatenate(([0], np.cumsum(~seen)))
    whole = (gaps[window + 1:] - gaps[:-window - 1]) == 0
    step = np.hypot(*(ball_xy[window:] - ball_xy[:-window]).T)
    still = np.zeros(len(dead), dtype=bool)
    with np.errstate(invalid="ignore"):
        still[:-window] = whole & (step < 0.5)
    still &= inside
    return float(dead[still].mean()) if still.any() else float("nan"), int(still.sum())


def measure_copy(match, raw_still, sim, live_share, seed=1):
    cfg = dataclasses.replace(SIMULATORS[sim], live_share=live_share)
    copy = degrade(match, seed, cfg)
    stats = copy.extras["degrade"]
    tag = "{} {} live {:.1f}".format(match.match_id, sim, live_share)
    fps = match.fps
    inside = match.period > 0
    dead = (match.labels == "D") & inside
    live = ~dead & inside
    seen = np.isfinite(copy.ball_xy[:, 0])
    missing = inside & ~seen
    false_ball = copy.extras["false_ball"] > 0
    out = {"match": match.match_id, "sim": sim, "live_share": live_share, "stats": stats}

    out["missing_share"] = float(missing[inside].mean())
    out["live_share_reached"] = float(missing[live].sum() / max(1, missing.sum()))
    check(tag + ": missing share", abs(out["missing_share"] - 0.41) <= MISSING_TOLERANCE,
          "{:.3f}".format(out["missing_share"]))
    check(tag + ": live share of missing frames", abs(out["live_share_reached"] - live_share) <= LIVE_SHARE_TOLERANCE,
          "{:.3f}".format(out["live_share_reached"]))

    lengths = []
    for half in (1, 2):
        rows = np.flatnonzero(match.period == half)
        a, b = runs(missing[rows])
        lengths.extend((b - a).tolist())
    lengths = np.array(lengths)
    in_long = float(lengths[lengths >= fps].sum() / max(1, lengths.sum()))
    out.update(gaps=int(len(lengths)), gaps_over_1s=int((lengths >= fps).sum()), in_gaps_over_1s=in_long,
               gap_median_over_1s=float(np.median(lengths[lengths >= fps]) / fps))
    check(tag + ": missing frames come in blocks", in_long >= 0.9,
          "{:.3f} in gaps over 1 s, {} gaps over 1 s, median {:.1f} s".format(
              in_long, out["gaps_over_1s"], out["gap_median_over_1s"]))

    truth = copy.extras["truth_ball_xy"]
    check(tag + ": live ball truth is the source ball", np.array_equal(truth[live], match.ball_xy[live]))
    step = np.full(len(truth), np.nan)
    step[1:] = np.hypot(*np.diff(truth, axis=0).T)
    restart = np.flatnonzero(live[1:] & dead[:-1]) + 1
    stop = np.flatnonzero(dead[1:] & live[:-1]) + 1
    out["restart_step_m"] = float(np.median(step[restart]))
    out["stop_step_m"] = float(np.median(step[stop]))
    check(tag + ": no ball jump at restarts", out["restart_step_m"] < 0.5,
          "median {:.2f} m, into a stoppage {:.2f} m".format(out["restart_step_m"], out["stop_step_m"]))

    seen_step = np.full(len(truth), np.nan)
    seen_step[1:] = np.hypot(*np.diff(copy.ball_xy.astype(np.float64), axis=0).T)
    parked = dead & seen & ~false_ball & (seen_step < 0.01)
    seen_dead = dead & seen & ~false_ball
    out["parked_share"] = float(parked.sum() / max(1, seen_dead.sum()))
    check(tag + ": the ball does not stand still in stoppages", out["parked_share"] < 0.05,
          "{:.3f} of seen dead frames under 1 cm per frame".format(out["parked_share"]))
    out["still_dead"], out["still_frames"] = still_dead_share(copy.ball_xy, dead, inside, fps)
    out["missing_dead"] = float(dead[missing].mean())
    # A ball held near a standing player is still, in a real stoppage too, so this is a probe only.
    note(tag + ": how often a still or missing ball is dead",
         "P(D | still 1 s) {:.3f} against {:.3f} on the source; P(D | missing) {:.3f}".format(
             out["still_dead"], raw_still, out["missing_dead"]))

    cleaned = clean(copy)
    spike_frames = copy.extras["false_ball"] == 1
    if spike_frames.any():
        removed = float(cleaned.ball_removed[spike_frames].mean())
        check(tag + ": the gate removes false spikes", removed >= 0.8, "{:.3f} of {} frames".format(
            removed, int(spike_frames.sum())))
    out["gate_removed_share"] = float(cleaned.ball_removed[inside].sum() / max(1, seen[inside].sum()))

    tracked = np.isfinite(copy.players_xy[..., 0]).sum(axis=1)[inside].mean()
    out["tracked_mean"] = float(tracked)
    check(tag + ": tracked objects per frame", 19.8 <= tracked <= 21.2,
          "{:.2f}, target {:.2f}, view lost {:.3f}, loss scale {:.2f}".format(
              tracked, stats["parameters"]["tracked_mean"], stats["view_lost_share"], stats["loss_scale"]))
    check(tag + ": every track found a slot", stats["dropped_no_slot"] == 0, str(stats["dropped_no_slot"]))
    lengths = (cleaned.fragments["end"] - cleaned.fragments["start"] + 1) / fps
    out["fragments"] = int(len(lengths))
    out["fragment_median_s"] = float(np.median(lengths))
    check(tag + ": fragments look like tracklets", 3.0 <= out["fragment_median_s"] <= 60.0,
          "{} fragments, median {:.1f} s".format(out["fragments"], out["fragment_median_s"]))

    person = copy.extras["slot_person"]
    person_team = copy.extras["person_team"]
    is_person = person >= 0
    true_team = np.where(is_person, person_team[np.clip(person, 0, None)], -1)
    slot_team = np.broadcast_to(copy.slot_team, person.shape)
    wrong = is_person & (slot_team != true_team)
    out["wrong_team_share"] = float(wrong.sum() / max(1, is_person.sum()))
    check(tag + ": tracks on the wrong team stay a minority", out["wrong_team_share"] < 0.1,
          "{:.4f} of person frames".format(out["wrong_team_share"]))
    kept = cleaned.players_team >= 0
    right_after = kept & is_person & (cleaned.players_team == true_team)
    out["team_right_after_clean"] = float(right_after.sum() / max(1, (kept & is_person).sum()))
    keeper = copy.extras["person_is_keeper"][np.clip(person, 0, None)] & is_person
    moved = keeper & wrong & kept
    out["keeper_recall"] = float((cleaned.players_team[moved] == true_team[moved]).mean()) if moved.any() else float("nan")
    out["keeper_wrong_frames"] = int(moved.sum())
    if moved.any():
        note(tag + ": cleaning returns keeper time to the right team",
             "{:.3f} of {} frames, team right on {:.4f} of person frames".format(
                 out["keeper_recall"], int(moved.sum()), out["team_right_after_clean"]))

    if copy.view_polygon is not None:
        seen_people = np.zeros(match.players_xy.shape[:2], dtype=bool)
        rows, slots = np.nonzero(is_person)
        seen_people[rows, person[rows, slots]] = True
        unseen = np.isfinite(match.players_xy[..., 0]) & ~seen_people & inside[:, None]
        near_side = float((match.players_xy[..., 1][unseen] < 0).mean()) if unseen.any() else float("nan")
        out["unseen_near_side"] = near_side
    return out


def check_parallax(match):
    """Seen high balls sit further from the camera than they are, unless the copy has no parallax."""
    camera = solve_camera()
    shares = {}
    for sim in ("camera-parallax", "camera-no-parallax"):
        copy = degrade(match, 3, dataclasses.replace(SIMULATORS[sim], air_loss=0.0))
        truth, z = copy.extras["truth_ball_xy"].astype(np.float64), copy.extras["truth_ball_z"]
        pick = np.isfinite(copy.ball_xy[:, 0]) & (z > 2.0) & (copy.extras["false_ball"] == 0)
        ray = truth[pick] - camera.centre[:2]
        along = ((copy.ball_xy[pick] - truth[pick]) * ray).sum(axis=1) / np.hypot(*ray.T)
        shares[sim] = float(np.median(along))
    check(match.match_id + ": high balls land along the camera ray only with parallax",
          shares["camera-parallax"] > 3.0 and abs(shares["camera-no-parallax"]) < 0.5,
          "median offset away from the camera {:.2f} m with, {:.2f} m without".format(
              shares["camera-parallax"], shares["camera-no-parallax"]))
    return shares


def check_blocks(match):
    """Visibility drifts between 5-minute blocks with block_sigma on, much less with it off."""
    spread = {}
    for sigma in (0.0, 1.0):
        cfg = dataclasses.replace(SIMULATORS["plain"], block_sigma=sigma, live_share=0.3)
        copy = degrade(match, 5, cfg)
        seen = np.isfinite(copy.ball_xy[:, 0])
        values = []
        for half in (1, 2):
            rows = np.flatnonzero(match.period == half)
            for start in range(0, len(rows) - 7500 + 1, 7500):
                values.append(seen[rows[start:start + 7500]].mean())
        spread[sigma] = float(np.std(values))
    check(match.match_id + ": visibility drifts from block to block", spread[1.0] > spread[0.0] + 0.03,
          "sd of 5-minute visibility {:.3f} with drift, {:.3f} without".format(spread[1.0], spread[0.0]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--idsse", default=os.path.join(ROOT, "external/processed/idsse"))
    parser.add_argument("--matches", help="comma-separated match ids, all by default")
    parser.add_argument("--json")
    args = parser.parse_args()

    started = time.time()
    check_camera()
    names = sorted(name[:-4] for name in os.listdir(args.idsse) if name.endswith(".npz"))
    if args.matches:
        names = [name for name in names if name in args.matches.split(",")]
    measurements = []
    for index, name in enumerate(names):
        match = load_idsse(os.path.join(args.idsse, name + ".npz"))
        if index == 0:
            check_contract(match)
            check_blocks(match)
        inside = match.period > 0
        raw_still, _ = still_dead_share(match.ball_xy, (match.labels == "D") & inside, inside, match.fps)
        for sim in ("camera", "plain"):
            for live_share in LIVE_SHARES:
                measurements.append(measure_copy(match, raw_still, sim, live_share))
        measurements.append({"match": name, "parallax": check_parallax(match)})

    for sim, floor in KEEPER_RECALL_FLOOR.items():
        rows = [m for m in measurements if m.get("sim") == sim and m["keeper_wrong_frames"] > 0]
        frames = sum(m["keeper_wrong_frames"] for m in rows)
        if frames:
            pooled = sum(m["keeper_recall"] * m["keeper_wrong_frames"] for m in rows) / frames
            check("{}: cleaning returns keeper time to the right team, pooled".format(sim), pooled >= floor,
                  "{:.3f} of {} frames over {} copies, worst copy {:.3f}".format(
                      pooled, frames, len(rows), min(m["keeper_recall"] for m in rows)))

    failed = [r for r in results if not r["ok"]]
    print("\n{} checks, {} failed, {:.0f} s".format(len(results), len(failed), time.time() - started), flush=True)
    for item in failed:
        print("FAIL " + item["name"] + "  " + item["detail"], flush=True)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump({"checks": results, "measurements": measurements}, handle, indent=1, default=float)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
