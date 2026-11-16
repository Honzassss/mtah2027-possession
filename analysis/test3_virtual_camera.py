"""Test 3: what a camera solved from the DATA.md homography row does to high balls on open data.

    python -I analysis/test3_virtual_camera.py [--idsse external/processed/idsse] [--json out.json]

Maps each IDSSE ball, which has a measured height, onto the ground the way a tracker with no height
would, then reports how far balls land from where they are, how many real flights the cleaning gate
deletes, and what happens to contacts above the feet. IDSSE and the printed row only: this never reads
the competition's match. The result is an upper bound: it assumes every high ball is detected and that
nothing compensates for height, and DATA.md says balls in the air are often lost outright.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline.camera import apparent_ground, solve_camera  # noqa: E402
from pipeline.clean import CleanConfig, gate_ball  # noqa: E402
from pipeline.idsse import load_idsse  # noqa: E402

FLIGHT_MIN_Z = 1.0          # metres: above a low bounce
FLIGHT_MIN_FRAMES = 5
PEAK_BINS = ((1.0, 2.5), (2.5, 5.0), (5.0, 10.0), (10.0, 99.0))
HEIGHT_BINS = ((0.5, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 10.0), (10.0, 99.0))
DEPTH_BINS = (("near third", -99.0, -11.33), ("middle third", -11.33, 11.33), ("far third", 11.33, 99.0))
GATES = (40.0, 50.0, 60.0, 80.0, 120.0)
CONTACT_Z = (1.0, 2.5)      # chest and head height
CONTACT_NEAR_M = 1.5        # true distance from the ball to the nearest player
TOUCH_GATES_M = (2.0, 2.5)
RAY_Z_MAX = 3.0             # the highest contact the height-aware distance allows for


def runs(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def nearest_player(point, players_xy):
    """(distance, slot) of the nearest player per frame."""
    delta = players_xy - point[:, None, :]
    distance = np.hypot(delta[..., 0], delta[..., 1])
    slot = np.nanargmin(distance, axis=1)
    return distance[np.arange(len(point)), slot], slot


def nearest_on_ray(point, players_xy, camera, z_max):
    """(distance, slot) of the player nearest to the stretch of camera ray a ball seen at `point` can be on."""
    back = (point - camera.centre[:2]) * (z_max / camera.centre[2])
    start = point - back                      # where the ball would stand at z_max
    rel = players_xy.astype(np.float64) - start[:, None, :]
    length_sq = (back ** 2).sum(axis=1)[:, None]
    t = np.clip((rel * back[:, None, :]).sum(axis=2) / length_sq, 0.0, 1.0)
    closest = start[:, None, :] + t[..., None] * back[:, None, :]
    distance = np.hypot(*(players_xy - closest).transpose(2, 0, 1))
    slot = np.nanargmin(distance, axis=1)
    return distance[np.arange(len(point)), slot], slot


def flights_of(match, live):
    """Starts, ends and peak height of the flights, each inside one half."""
    starts, ends, peaks = [], [], []
    for half in (1, 2):
        rows = np.flatnonzero(match.period == half)
        high = live[rows] & (match.ball_z[rows] >= FLIGHT_MIN_Z)
        for a, b in zip(*runs(high)):
            if b - a >= FLIGHT_MIN_FRAMES:
                starts.append(rows[a])
                ends.append(rows[b - 1] + 1)
                peaks.append(float(match.ball_z[rows[a:b]].max()))
    return np.array(starts), np.array(ends), np.array(peaks)


def analyse(path, camera):
    match = load_idsse(path)
    fps = match.fps
    live = match.labels != "D"
    z = match.ball_z.astype(np.float64)
    true_xy = match.ball_xy.astype(np.float64)
    seen_xy = apparent_ground(true_xy, z, camera.centre)
    shift = np.hypot(*(seen_xy - true_xy).T)
    out = {"match": match.match_id, "live_frames": int(live.sum()),
           "z_share": {str(t): float((z[live] > t).mean()) for t in (1, 2, 3, 5, 10)},
           "no_ground_point": int(np.isnan(seen_xy[:, 0]).sum())}

    out["shift_by_height"] = {}
    for low, high in HEIGHT_BINS:
        pick = live & (z >= low) & (z < high)
        row = {"share_of_live": float(pick.mean() / live.mean())}
        for name, y_low, y_high in DEPTH_BINS:
            part = pick & (true_xy[:, 1] >= y_low) & (true_xy[:, 1] < y_high)
            row[name] = float(np.nanmedian(shift[part])) if part.any() else float("nan")
        row["p90"] = float(np.nanpercentile(shift[pick], 90)) if pick.any() else float("nan")
        out["shift_by_height"]["{}-{}".format(low, high)] = row

    length, width = match.pitch
    outside = live & (np.abs(seen_xy[:, 0]) > length / 2 + 5) | live & (np.abs(seen_xy[:, 1]) > width / 2 + 5)
    true_outside = live & (np.abs(true_xy[:, 0]) > length / 2 + 5) | live & (np.abs(true_xy[:, 1]) > width / 2 + 5)
    out["outside_pitch_5m_share_of_live"] = {"apparent": float(outside.mean() / live.mean()),
                                             "true": float(true_outside.mean() / live.mean())}

    # The gate sees live frames only, so the parked ball's jumps do not count against either path.
    true_live = np.where(live[:, None], true_xy, np.nan)
    seen_live = np.where(live[:, None], seen_xy, np.nan)
    speed = {}
    for name, xy in (("true", true_live), ("apparent", seen_live)):
        step = np.hypot(*np.diff(xy, axis=0).T) * fps
        step[match.period[1:] != match.period[:-1]] = np.nan
        speed[name] = step
    out["speed_over_40"] = {name: float(np.nanmean(step[np.isfinite(step)] > 40.0)) for name, step in speed.items()}

    starts, ends, peaks = flights_of(match, live)
    keep_true = {}
    keep_seen = {}
    for gate in GATES:
        keep_true[gate] = gate_ball(true_live, match.frame, match.period, fps, gate, match.pitch)
        keep_seen[gate] = gate_ball(seen_live, match.frame, match.period, fps, gate, match.pitch)
    usable_true = int(np.isfinite(true_live[:, 0]).sum())
    usable_seen = int(np.isfinite(seen_live[:, 0]).sum())
    out["gate"] = {}
    for gate in GATES:
        gone_true = np.isfinite(true_live[:, 0]) & ~keep_true[gate]
        gone_seen = np.isfinite(seen_live[:, 0]) & ~keep_seen[gate]
        lost_true = np.array([gone_true[a:b].mean() for a, b in zip(starts, ends)])
        lost_seen = np.array([gone_seen[a:b].mean() for a, b in zip(starts, ends)])
        row = {"frames_removed_true": float(gone_true.sum() / usable_true),
               "frames_removed_apparent": float(gone_seen.sum() / usable_seen),
               "flights": int(len(starts)),
               "flights_any_true": float((lost_true > 0).mean()),
               "flights_any_apparent": float((lost_seen > 0).mean()),
               "flights_quarter_apparent": float((lost_seen >= 0.25).mean()),
               "flights_half_apparent": float((lost_seen >= 0.5).mean()),
               "flights_whole_apparent": float((lost_seen >= 0.999).mean())}
        row["by_peak"] = {}
        for low, high in PEAK_BINS:
            pick = (peaks >= low) & (peaks < high)
            if pick.any():
                row["by_peak"]["{}-{}".format(low, high)] = {
                    "flights": int(pick.sum()),
                    "any_true": float((lost_true[pick] > 0).mean()),
                    "any": float((lost_seen[pick] > 0).mean()),
                    "half": float((lost_seen[pick] >= 0.5).mean()),
                    "frames_lost": float(lost_seen[pick].mean())}
        out["gate"][str(int(gate))] = row

    # Contacts above the feet: true nearest player close, ball at chest or head height.
    true_near, true_slot = nearest_player(true_xy, match.players_xy)
    contact = live & (z >= CONTACT_Z[0]) & (z < CONTACT_Z[1]) & (true_near <= CONTACT_NEAR_M)
    seen_near, seen_slot = nearest_player(seen_xy[contact], match.players_xy[contact])
    out["contacts"] = {"frames": int(contact.sum()),
                       "share_of_live": float(contact.sum() / live.sum()),
                       "median_apparent_nearest_m": float(np.median(seen_near)),
                       "other_player_nearest": float((seen_slot != true_slot[contact]).mean()),
                       "other_team_nearest": float((match.slot_team[seen_slot] != match.slot_team[true_slot[contact]]).mean())}
    for gate in TOUCH_GATES_M:
        out["contacts"]["beyond_{}m".format(gate)] = float((seen_near > gate).mean())
    for name, y_low, y_high in DEPTH_BINS:
        part = (true_xy[contact, 1] >= y_low) & (true_xy[contact, 1] < y_high)
        out["contacts"]["beyond_2.5m_" + name] = float((seen_near[part] > 2.5).mean()) if part.any() else float("nan")
    # The same comparison with every live frame near a player, to show how much is height alone.
    grounded = live & (z < 0.5) & (true_near <= CONTACT_NEAR_M)
    grounded_seen, grounded_slot = nearest_player(seen_xy[grounded], match.players_xy[grounded])
    out["contacts"]["grounded_other_team_nearest"] = float(
        (match.slot_team[grounded_slot] != match.slot_team[true_slot[grounded]]).mean())

    # Height-aware distance: the ground point under the ball lies between the apparent point and the
    # camera foot, at most RAY_Z_MAX / H of the way. It needs only the camera, so it works without height.
    ray_near, ray_slot = nearest_on_ray(seen_xy[contact], match.players_xy[contact], camera, RAY_Z_MAX)
    out["contacts"]["ray_beyond_2.5m"] = float((ray_near > 2.5).mean())
    out["contacts"]["ray_other_team_nearest"] = float(
        (match.slot_team[ray_slot] != match.slot_team[true_slot[contact]]).mean())
    _, ray_grounded_slot = nearest_on_ray(seen_xy[grounded], match.players_xy[grounded], camera, RAY_Z_MAX)
    out["contacts"]["ray_grounded_other_team_nearest"] = float(
        (match.slot_team[ray_grounded_slot] != match.slot_team[true_slot[grounded]]).mean())

    # Signatures a label-free check on the real match could look for, with and without parallax.
    out["signature"] = {}
    for name, xy in (("parallax", seen_xy), ("none", true_xy)):
        near, slot = nearest_player(np.nan_to_num(xy, nan=1e6), match.players_xy)
        player = match.players_xy[np.arange(len(xy)), slot].astype(np.float64)
        ray = player - camera.centre[:2]
        along = ((xy - player) * ray).sum(axis=1) / np.hypot(*ray.T)
        pick = live & (near <= 3.0)
        row = {}
        for third, y_low, y_high in DEPTH_BINS:
            part = pick & (player[:, 1] >= y_low) & (player[:, 1] < y_high)
            row[third] = float(np.mean(along[part]))
        with np.errstate(invalid="ignore"):
            row["beyond_far_line"] = float((live & (xy[:, 1] > width / 2 + 2)).sum() / live.sum())
            row["beyond_near_line"] = float((live & (xy[:, 1] < -width / 2 - 2)).sum() / live.sum())
        out["signature"][name] = row

    # Which way fast apparent motion points: along the camera ray, if parallax causes it.
    step = np.diff(seen_live, axis=0)
    ray = seen_live[:-1] - camera.centre[:2]
    norm = np.hypot(*step.T) * np.hypot(*ray.T)
    with np.errstate(divide="ignore", invalid="ignore"):
        cosine = np.abs((step * ray).sum(axis=1)) / norm
    fast = np.isfinite(cosine) & (np.hypot(*step.T) * fps > 40.0)
    out["fast_motion_radial_median_cos"] = float(np.median(cosine[fast])) if fast.any() else float("nan")
    return out


def pooled(results, get):
    """Weighted by the number of flights."""
    weights = np.array([r["gate"]["40"]["flights"] for r in results], dtype=float)
    return float(np.sum([w * get(r) for w, r in zip(weights, results)]) / weights.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--idsse", default=os.path.join(ROOT, "external/processed/idsse"))
    parser.add_argument("--json", help="write the raw measurements here")
    args = parser.parse_args()

    camera = solve_camera()
    print("camera   centre x {:.2f}  y {:.2f}  height {:.2f} m, {:.1f} m behind the near line, focal {:.0f} px, "
          "{:.1f} deg wide".format(*camera.centre, -34.0 - camera.centre[1], camera.focal,
                                    2 * np.degrees(np.arctan(camera.image_size[0] / 2 / camera.focal))))
    print("principal point scan, vertical 460 to 620 px:")
    for cy in (460, 500, 540, 580, 620):
        scan = solve_camera(principal_y=cy)
        print("  cy {:4d}: height {:.2f} m, behind the near line {:.2f} m, cx {:.1f}".format(
            cy, scan.centre[2], -34.0 - scan.centre[1], scan.principal[0]))
    print("\nformula D*z/(H-z), metres along the ray:")
    for z in (1, 2, 5, 10):
        print("  z = {:2d} m: near line {:5.1f}, centre {:5.1f}, far line {:5.1f}".format(z, *[
            np.hypot(0 - camera.centre[0], y - camera.centre[1]) * z / (camera.centre[2] - z) for y in (-34, 0, 34)]))

    results = [analyse(os.path.join(args.idsse, name), camera) for name in sorted(os.listdir(args.idsse))
               if name.endswith(".npz")]
    print("\nlive-frame ball height, share above 1 / 2 / 3 / 5 / 10 m")
    for r in results:
        print("  {}: {}".format(r["match"], " ".join("{:.3f}".format(v) for v in r["z_share"].values())))

    print("\nmedian landing error by height, metres, mean over matches (p90 is over the whole pitch)")
    print("  {:>9} {:>8} {:>7} {:>7} {:>7} {:>7}".format("height", "live %", "near", "middle", "far", "p90"))
    for low, high in HEIGHT_BINS:
        key = "{}-{}".format(low, high)
        rows = [r["shift_by_height"][key] for r in results]
        print("  {:>9} {:8.2f} {:7.1f} {:7.1f} {:7.1f} {:7.1f}".format(
            key, 100 * np.mean([x["share_of_live"] for x in rows]),
            *[np.nanmean([x[name] for x in rows]) for name, _, _ in DEPTH_BINS],
            np.nanmean([x["p90"] for x in rows])))

    print("\nlive frames more than 5 m outside the pitch: apparent {:.2%}, true {:.2%}".format(
        np.mean([r["outside_pitch_5m_share_of_live"]["apparent"] for r in results]),
        np.mean([r["outside_pitch_5m_share_of_live"]["true"] for r in results])))
    print("consecutive live frames over 40 m/s: apparent {:.2%}, true {:.2%}".format(
        np.mean([r["speed_over_40"]["apparent"] for r in results]),
        np.mean([r["speed_over_40"]["true"] for r in results])))
    print("fast apparent motion, median |cos| against the camera ray: {:.2f}".format(
        np.mean([r["fast_motion_radial_median_cos"] for r in results])))

    print("\ngate, all flights over {} m pooled ({} flights)".format(
        FLIGHT_MIN_Z, sum(r["gate"]["40"]["flights"] for r in results)))
    print("  {:>6} {:>13} {:>13} {:>11} {:>11} {:>11} {:>11} {:>11}".format(
        "m/s", "frames, true", "frames, appt", "fl any true", "fl any appt", "fl >=25%", "fl >=50%", "fl whole"))
    for gate in GATES:
        key = str(int(gate))
        print("  {:>6} {:13.2%} {:13.2%} {:11.1%} {:11.1%} {:11.1%} {:11.1%} {:11.1%}".format(
            key,
            np.mean([r["gate"][key]["frames_removed_true"] for r in results]),
            np.mean([r["gate"][key]["frames_removed_apparent"] for r in results]),
            pooled(results, lambda r: r["gate"][key]["flights_any_true"]),
            pooled(results, lambda r: r["gate"][key]["flights_any_apparent"]),
            pooled(results, lambda r: r["gate"][key]["flights_quarter_apparent"]),
            pooled(results, lambda r: r["gate"][key]["flights_half_apparent"]),
            pooled(results, lambda r: r["gate"][key]["flights_whole_apparent"])))
    print("\n40 m/s gate by flight peak")
    print("  {:>9} {:>8} {:>11} {:>11} {:>9} {:>12}".format("peak m", "flights", "any, true", "any, appt", ">=50%", "frames lost"))
    for low, high in PEAK_BINS:
        key = "{}-{}".format(low, high)
        rows = [r["gate"]["40"]["by_peak"][key] for r in results if key in r["gate"]["40"]["by_peak"]]
        weights = np.array([x["flights"] for x in rows], dtype=float)
        print("  {:>9} {:8d} {:11.1%} {:11.1%} {:9.1%} {:12.1%}".format(
            key, int(weights.sum()), *[np.average([x[name] for x in rows], weights=weights)
                                       for name in ("any_true", "any", "half", "frames_lost")]))

    print("\ncontacts: ball at {}-{} m with a player within {} m in truth".format(*CONTACT_Z, CONTACT_NEAR_M))
    print("  share of live frames {:.2%}, median apparent distance to nearest player {:.1f} m".format(
        np.mean([r["contacts"]["share_of_live"] for r in results]),
        np.mean([r["contacts"]["median_apparent_nearest_m"] for r in results])))
    print("  beyond a 2.0 m gate {:.1%}, beyond 2.5 m {:.1%} (near {:.1%}, middle {:.1%}, far {:.1%})".format(
        np.mean([r["contacts"]["beyond_2.0m"] for r in results]),
        np.mean([r["contacts"]["beyond_2.5m"] for r in results]),
        *[np.nanmean([r["contacts"]["beyond_2.5m_" + name] for r in results]) for name, _, _ in DEPTH_BINS]))
    print("  nearest player changes {:.1%}, nearest team changes {:.1%}".format(
        np.mean([r["contacts"]["other_player_nearest"] for r in results]),
        np.mean([r["contacts"]["other_team_nearest"] for r in results])))
    print("  same check on balls under 0.5 m, nearest team changes {:.1%}".format(
        np.mean([r["contacts"]["grounded_other_team_nearest"] for r in results])))
    print("  height-aware distance (ray up to {} m): beyond 2.5 m {:.1%}, nearest team changes {:.1%}, "
          "and {:.1%} on balls under 0.5 m".format(
              RAY_Z_MAX, np.mean([r["contacts"]["ray_beyond_2.5m"] for r in results]),
              np.mean([r["contacts"]["ray_other_team_nearest"] for r in results]),
              np.mean([r["contacts"]["ray_grounded_other_team_nearest"] for r in results])))

    print("\nlabel-free signatures, with parallax / without")
    print("  mean offset of the ball beyond its nearest player (within 3 m) along the camera ray, metres:")
    for third, _, _ in DEPTH_BINS:
        print("    {:>12}: {:5.2f} / {:5.2f}".format(third, *[
            np.mean([r["signature"][name][third] for r in results]) for name in ("parallax", "none")]))
    for side in ("beyond_far_line", "beyond_near_line"):
        print("  live frames {} by over 2 m: {:.2%} / {:.2%}".format(side.replace("_", " "), *[
            np.mean([r["signature"][name][side] for r in results]) for name in ("parallax", "none")]))
    print("\ncleaning default: {} m/s".format(CleanConfig().ball_max_speed))
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(results, handle, indent=1)


if __name__ == "__main__":
    main()
