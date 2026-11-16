"""Self-checks for the twin. Own code only: nothing here imports or runs the challenge repository.

    python -I tests/check_twin.py --twin twin/J03WMX --scorer analysis/spellmetric.py \\
        [--games upstream/games.py] [--json results.json] [--no-kloppy]

Checks the file shape, the timeline and the documented quirks, loads the ATD pair with kloppy's
Metrica EPTS loader, then runs a fresh reimplementation of the public baseline ("nearest tracked
player, D when no ball, majority vote of +/-50 frames") and scores it with the clean-room scorer.
Exit code 1 if any check fails. The published real-match figures are printed for comparison only.
"""
import argparse
import importlib.util
import json
import os
import statistics
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

N_FRAMES = 199319
N_SLOTS = 46
FPS = 25
HALF1 = (11185, 90651)
HALF2 = (110808, 190197)
KICKOFF1 = 11185
KICKOFF2 = 113742
PUBLIC = (11185, 18684)
PITCH = (105.3, 68.0)
STEM = "BAR-ATH_tactical"
SUFFIXES = {"atd_meta": "_atd.xml", "atd_raw": "_atd.txt", "homography": "_homography.csv",
            "smart_tagging": "_smart_tagging.xml"}
CODES = {"PLAYERS IN THE BOX", "BALL IN FINAL THIRD", "BALL IN THE BOX", "PROGRESSION",
         "DEFENDING IN MIDDLE THIRD", "SET PIECES", "CREATING CHANCES", "DEFENDING IN DEFENSIVE THIRD",
         "LONG BALL", "ATTACKING TRANSITION", "DEFENSIVE TRANSITION", "BUILD UP",
         "DEFENDING IN ATTACKING THIRD", "GOALS"}
TAG_VALUES = {
    "Team": {"Barcelona", "Atletic Club", "N/A"},
    "Half": {"1st Half", "2nd Half"},
    "Type": {"Throw-in", "Corner Kick", "Goal Kick", "Free Kick", "Kick Off (After Goal)", "Kick Off (Start)"},
    "Side": {"Central", "Left", "Right"},
    "Direction of ball entry": {"Horizontal", "Vertical", "Diagonal"},
    "Max Players in the box": {"1", "2", "3", "4", "5", "6", "7+"},
}
# Published on the real match in DATA.md and starter/baseline.py. Shown beside the twin, not asserted.
REAL = {"visible_all": 0.569, "visible_graded": 0.593, "mean_tracked_graded": 20.5,
        "baseline": {"raw": (0.6532, 0.0486), 25: (0.6752, 0.1996), 50: (0.6724, 0.2433), 125: (0.6356, 0.2242)}}

results = []


def check(name, ok, detail=""):
    results.append({"name": name, "ok": bool(ok), "detail": str(detail)})
    print("{:4} {:<44} {}".format("PASS" if ok else "FAIL", name, detail), flush=True)


def near(value, target, tolerance):
    return abs(value - target) <= tolerance


def paths_of(twin):
    root = os.path.join(twin, "data", "tactical")
    return {name: os.path.join(root, STEM + suffix) for name, suffix in SUFFIXES.items()}


def check_names_against_games(games_path):
    """Read games.py as text, never import it: the stem, suffixes and blocks should still match."""
    with open(games_path) as handle:
        text = handle.read()
    wanted = ['"stem": "{}"'.format(STEM)] + ['"{}": "{}"'.format(k, v) for k, v in SUFFIXES.items()]
    wanted += ['"n_video_frames": {}'.format(N_FRAMES), '"offsets": {1: 1702057, 2: 1702058}',
               '"halves": {1: (1713242, 1792708), 2: (1812866, 1892255)}']
    missing = [item for item in wanted if item not in text]
    check("names and blocks agree with games.py text", not missing, missing or "stem, suffixes, offsets, halves")
    first = 1713242 - 1702057, 1792708 - 1702057
    second = 1812866 - 1702058, 1892255 - 1702058
    check("graded blocks from games.py offsets", (first, second) == (HALF1, HALF2), "{} {}".format(first, second))


def read_raw(path):
    """Validate every line's shape, then return the numbers as (frames, 142) float32."""
    n_lines = 0
    bad_ids = bad_shape = bad_players = bad_ball = 0
    first_bad = None
    rows = np.empty((N_FRAMES, 1 + 3 * N_SLOTS + 3), dtype=np.float32)
    with open(path) as handle:
        for line in handle:
            n_lines += 1
            if n_lines > N_FRAMES:
                continue
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                bad_shape += 1
                first_bad = first_bad or line[:60]
                continue
            if parts[0] != str(n_lines):
                bad_ids += 1
            groups = parts[1].split(";")
            ball = parts[2].split(",")
            if len(groups) != N_SLOTS or len(ball) != 3:
                bad_shape += 1
                first_bad = first_bad or line[:60]
                continue
            for group in groups:
                fields = group.split(",")
                if len(fields) != 3:
                    bad_shape += 1
                    break
                nan_count = fields.count("NaN")
                if nan_count not in (0, 3):
                    bad_players += 1
            if (ball[0] == "NaN") != (ball[1] == "NaN") or (ball[0] == "NaN" and ball[2] != "NaN"):
                bad_ball += 1
            rows[n_lines - 1] = np.fromstring(
                parts[0] + "," + parts[1].replace(";", ",") + "," + parts[2], sep=",", dtype=np.float32)
    check("raw line count", n_lines == N_FRAMES, n_lines)
    check("raw frame ids 1..199319 in order", bad_ids == 0 and rows[0, 0] == 1 and rows[-1, 0] == N_FRAMES, "bad {}".format(bad_ids))
    check("raw shape: 46 slots, 3 fields each, ball 3", bad_shape == 0, "bad {} {}".format(bad_shape, first_bad or ""))
    check("slot groups are all NaN or all numbers", bad_players == 0, "bad {}".format(bad_players))
    check("ball is x,y,speed or x,y,NaN or NaN,NaN,NaN", bad_ball == 0, "bad {}".format(bad_ball))
    return rows


def check_metadata(path):
    root = ET.parse(path).getroot()
    teams = list(root.iter("Team"))
    check("metadata has two Team elements with id and Name",
          len(teams) == 2 and all(t.get("id") and t.findtext("Name") for t in teams),
          [t.findtext("Name") for t in teams])
    check("first team is TWIN-HOME", teams[0].findtext("Name") == "TWIN-HOME", teams[0].findtext("Name"))
    players = list(root.iter("Player"))
    check("metadata has 46 Player elements", len(players) == N_SLOTS, len(players))
    names_ok = all(p.findtext("Name") == "Track {}".format(i + 1) and p.findtext("ShirtNumber") == str(i + 1)
                   for i, p in enumerate(players))
    check("players are Track N with ShirtNumber N in slot order", names_ok)
    home_id = teams[0].get("id")
    return np.array([0 if p.get("teamId") == home_id else 1 for p in players], dtype=np.int8)


def camera_follow(h, rows, graded_mask):
    """Pitch x under the image centre, against ball x: correlation at a few delays."""
    frames = np.flatnonzero(graded_mask)
    inverse = np.linalg.inv(h[frames])
    centre = np.array([960.0, 540.0, 1.0])
    pitch = inverse @ centre
    aim = pitch[:, 0] / pitch[:, 2]
    ball = rows[frames, 139].astype(np.float64)
    best = (None, -2.0)
    for delay in (0, 25, 50, 75, 100, 150):
        here = aim[delay:]
        there = ball[:len(ball) - delay]
        keep = np.isfinite(there)
        value = float(np.corrcoef(here[keep], there[keep])[0, 1])
        if value > best[1]:
            best = (delay, value)
    return best


def check_homography(path, rows):
    header = "frame,h00,h01,h02,h10,h11,h12,h20,h21,h22,reprojection_error,valid"
    with open(path) as handle:
        first = handle.readline().rstrip("\n")
    check("homography header", first == header, first[:40])
    table = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    check("homography rows", table.shape == (N_FRAMES, 12), table.shape)
    check("homography frames 0..199318", np.array_equal(table[:, 0], np.arange(N_FRAMES)))
    check("homography valid on 100% of rows", np.all(table[:, 11] == 1), float(np.mean(table[:, 11] == 1)))
    check("homography matrices finite, h22 = 1", np.isfinite(table[:, 1:10]).all() and np.all(table[:, 9] == 1))
    h = table[:, 1:10].reshape(-1, 3, 3)
    corners = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
    check("pitch corners stay in front of the camera", np.all(h[:, 2] @ corners.T > 0))
    w = h[:, 2, 0] * 0.5 + h[:, 2, 1] * 0.5 + h[:, 2, 2]
    cx = (h[:, 0, 0] * 0.5 + h[:, 0, 1] * 0.5 + h[:, 0, 2]) / w
    graded_mask = np.zeros(N_FRAMES, dtype=bool)
    graded_mask[HALF1[0]:HALF1[1] + 1] = True
    graded_mask[HALF2[0]:HALF2[1] + 1] = True
    print("     pitch centre x in pixels over graded frames: {:.0f}..{:.0f}".format(cx[graded_mask].min(), cx[graded_mask].max()))
    error = table[:, 10]
    halftime = np.arange(HALF1[1] + 1, HALF2[0])
    print("     reprojection median: graded {:.3f} halftime {:.3f}".format(
        float(np.median(error[graded_mask])), float(np.median(error[halftime]))))
    print("     row at the first kickoff equals DATA.md's printed matrix: {}".format(
        np.allclose(table[KICKOFF1, 1:10], [1822.663, -511.2005, 44.48168, 1.698323, 169.2971, 255.7303, -0.01907188, -0.5269586, 1])))
    delay, correlation = camera_follow(h, rows, graded_mask)
    check("synthetic camera follows the ball with a lag", correlation > 0.8 and delay > 0,
          "correlation {:.3f} at {} frames".format(correlation, delay))


def check_gt(twin):
    def read(path):
        with open(path) as handle:
            header = handle.readline().rstrip("\n")
            lines = handle.read().split()
        labels = np.array([line.split(",")[1] for line in lines])
        frames = np.array([int(line.split(",")[0]) for line in lines])
        return header, frames, labels

    header, frames, labels = read(os.path.join(twin, "gt", "possession_gt_full.csv"))
    check("GT header frame,label", header == "frame,label", header)
    check("GT rows and frame numbers", len(labels) == N_FRAMES and np.array_equal(frames, np.arange(N_FRAMES)), len(labels))
    graded = np.zeros(N_FRAMES, dtype=bool)
    graded[HALF1[0]:HALF1[1] + 1] = True
    graded[HALF2[0]:HALF2[1] + 1] = True
    check("GT graded frames are H/A/D, 158,857 of them",
          np.isin(labels[graded], ("H", "A", "D")).all() and int(graded.sum()) == 158857, int(graded.sum()))
    check("GT is X outside the graded blocks", np.all(labels[~graded] == "X"))
    check("GT 110808..113741 all D", np.all(labels[HALF2[0]:KICKOFF2] == "D"))
    check("GT both kickoffs are live", labels[KICKOFF1] != "D" and labels[KICKOFF2] != "D",
          "{} {}".format(labels[KICKOFF1], labels[KICKOFF2]))
    check("GT both graded blocks end dead (padding)", labels[HALF1[1]] == "D" and labels[HALF2[1]] == "D")

    header, frames, public = read(os.path.join(twin, "gt", "possession_gt_public_like.csv"))
    window = np.zeros(N_FRAMES, dtype=bool)
    window[PUBLIC[0]:PUBLIC[1] + 1] = True
    check("public-like GT labels only 11185..18684",
          len(public) == N_FRAMES and np.all(public[~window] == "X") and np.array_equal(public[window], labels[window]),
          {k: int(np.sum(public[window] == k)) for k in "HAD"})
    return labels, graded


def check_tags(path, labels):
    root = ET.parse(path).getroot()
    instances = list(root.iter("instance"))
    parsed, bad_codes, bad_groups, bad_times = [], 0, [], 0
    for instance in instances:
        code = instance.findtext("code")
        start, end = float(instance.findtext("start")), float(instance.findtext("end"))
        tags = {label.findtext("group"): label.findtext("text") for label in instance.findall("label")}
        bad_codes += code not in CODES
        bad_times += not (0.0 <= start <= end <= (N_FRAMES - 1) / FPS)
        for group, text in tags.items():
            if text not in TAG_VALUES.get(group, ()):
                bad_groups.append((group, text))
        parsed.append((code, start, end, tags))
    print("     {} instances (DATA.md 638)".format(len(parsed)))
    check("tags: codes from DATA.md", bad_codes == 0, bad_codes)
    check("tags: groups and values from DATA.md", not bad_groups, bad_groups[:3])
    check("tags: times are seconds on the video timeline", bad_times == 0, bad_times)
    check("tags: every instance has Team and Half", all("Team" in t and "Half" in t for _, _, _, t in parsed))

    pieces = [p for p in parsed if p[0] == "SET PIECES"]
    starts = [p for p in pieces if p[3].get("Type") == "Kick Off (Start)"]
    contains = {
        truth: any(round(p[1] * FPS) <= truth <= round(p[2] * FPS) and p[3]["Half"] == half for p in starts)
        for truth, half in ((KICKOFF1, "1st Half"), (KICKOFF2, "2nd Half"))}
    check("tags: Kick Off (Start) contain both kickoffs", all(contains.values()), contains)
    throw_ins = [p for p in pieces if p[3].get("Type") == "Throw-in"]
    check("tags: N/A team on every throw-in", throw_ins and all(p[3]["Team"] == "N/A" for p in throw_ins), len(throw_ins))
    n_na = sum(p[3]["Team"] == "N/A" for p in parsed)
    check("tags: N/A on about 36 instances, throw-ins and a few more", near(n_na, 36, 4) and n_na > len(throw_ins), n_na)

    # Opening lead: seconds from tag start to the dead-to-live transition inside the tag.
    restart = np.flatnonzero((labels[1:] != "D") & (labels[1:] != "X") & (labels[:-1] == "D")) + 1
    restart = np.sort(np.concatenate((restart, [KICKOFF1])))
    leads = []
    for _, start, end, _ in pieces:
        low, high = np.searchsorted(restart, [round(start * FPS), round(end * FPS)])
        if high > low:
            leads.append(restart[low] / FPS - start)
    median_lead = statistics.median(leads) if leads else float("nan")
    check("tags: set pieces open about 3.5 s early", near(median_lead, 3.5, 0.5),
          "median {:.2f} s over {} tags".format(median_lead, len(leads)))
    counts = {}
    for code, *_ in parsed:
        counts[code] = counts.get(code, 0) + 1
    return {"instances": len(parsed), "codes": counts}


def runs(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def degradation(rows, labels, graded, team_of_slot, stats_out, expect_dead_share):
    xs = rows[:, 1::3][:, :N_SLOTS]
    ys = rows[:, 2::3][:, :N_SLOTS]
    ss = rows[:, 3::3][:, :N_SLOTS]
    bx, by, bs = rows[:, 139], rows[:, 140], rows[:, 141]

    present = np.isfinite(xs)
    in_range = np.all((xs[present] >= 0) & (xs[present] <= 1)) and np.all((ys[present] >= 0) & (ys[present] <= 1))
    ball_seen = np.isfinite(bx)
    in_range = in_range and np.all((bx[ball_seen] >= 0) & (bx[ball_seen] <= 1)) and np.all((by[ball_seen] >= 0) & (by[ball_seen] <= 1))
    check("coordinates normalised 0-1", in_range)
    check("player speed filled wherever a slot is tracked", np.all(np.isfinite(ss[present])))

    tracked = present.sum(axis=1)
    live = graded & (labels != "D")
    dead = graded & (labels == "D")
    mean_graded = float(tracked[graded].mean())
    check("mean tracked objects near 20.5", near(mean_graded, 20.5, 0.2),
          "graded {:.2f}, live {:.2f}, dead {:.2f}".format(mean_graded, tracked[live].mean(), tracked[dead].mean()))
    over = float(np.mean(tracked[graded] > 22))
    check("occasional frames above 22 objects", 0.001 < over < 0.10, "{:.2%} of graded, max {}".format(over, tracked.max()))
    check("almost no tracking outside the graded blocks", tracked[~graded].mean() < 1.0, "mean {:.2f}".format(tracked[~graded].mean()))

    used_home = int(np.sum(present[:, team_of_slot == 0].any(axis=0)))
    used_away = int(np.sum(present[:, team_of_slot == 1].any(axis=0)))
    segments = 0
    for slot in range(N_SLOTS):
        starts, _ = runs(present[:, slot])
        segments += len(starts)
    print("     slots used home {} away {}, tracklet segments {}, mean occupancy {:.1%}".format(
        used_home, used_away, segments, present[graded].mean()))
    check("slots do not follow a person (many segments per slot)", segments > 2000 and used_home + used_away == N_SLOTS,
          "{} segments, {:.0f} per slot".format(segments, segments / N_SLOTS))

    visible_all = float(ball_seen.mean())
    visible_graded = float(ball_seen[graded].mean())
    check("ball visible 56.9% of all frames", near(visible_all, 0.569, 0.01), "{:.2%}".format(visible_all))
    check("ball visible 59.3% of graded frames", near(visible_graded, 0.593, 0.01), "{:.2%}".format(visible_graded))

    # Gaps, measured inside each graded block.
    gaps = []
    for first, last in (HALF1, HALF2):
        starts, ends = runs(~ball_seen[first:last + 1])
        gaps.extend((int(a) + first, int(b) + first) for a, b in zip(starts, ends))
    seconds = np.array([(b - a) / FPS for a, b in gaps])
    over_5 = seconds[seconds > 5]
    table = {">5": int(len(over_5)), ">10": int(np.sum(seconds > 10)), ">20": int(np.sum(seconds > 20)),
             ">30": int(np.sum(seconds > 30)), ">60": int(np.sum(seconds > 60))}
    # The blackout counts are set in twin_builder/twin.toml by whoever builds the twin (this copy ships
    # none), so the gaps are shown here rather than checked against figures.
    print("     gaps {}, longest {:.1f} s, median over 5 s {:.1f} s".format(
        table, float(seconds.max()), float(np.median(over_5))), flush=True)
    missing_share = float(np.mean(~ball_seen[graded]))
    check("41% of graded frames missing", near(missing_share, 0.41, 0.01), "{:.2%}, {} gaps in all".format(missing_share, len(gaps)))
    missing_dead = float(np.sum(~ball_seen & dead) / np.sum(~ball_seen & graded))
    check("share of missing frames in stoppages is gap_dead_share", near(missing_dead, expect_dead_share, 0.02),
          "{:.3f}, expected {}".format(missing_dead, expect_dead_share))

    # Unparked: consecutive visible dead frames should show a moving ball.
    steps = np.hypot((bx[1:] - bx[:-1]) * PITCH[0], (by[1:] - by[:-1]) * PITCH[1])
    both_dead = dead[1:] & dead[:-1] & ball_seen[1:] & ball_seen[:-1]
    both_live = live[1:] & live[:-1] & ball_seen[1:] & ball_seen[:-1]
    still = float(np.mean(steps[both_dead] < 0.005))
    check("ball is not parked in stoppages", still < 0.05,
          "{:.1%} of dead steps still; {:.2f} m/s dead vs {:.2f} m/s live (noise included)".format(
              still, np.mean(steps[both_dead]) * FPS, np.mean(steps[both_live]) * FPS))
    fill = float(np.mean(np.isfinite(bs[ball_seen])))
    # The share of visible frames with a ball speed is set in twin_builder/twin.toml by whoever builds the
    # twin (this copy ships none), so it is shown rather than checked.
    print("     ball speed on {:.1%} of visible frames".format(fill), flush=True)

    # Player pace in gaps.
    pace = np.full(N_FRAMES, np.nan, dtype=np.float64)
    occupied = present.any(axis=1)
    pace[occupied] = np.nanmean(ss[occupied], axis=1)
    pace_live = float(np.nanmean(pace[live]))
    pace_seen = float(np.nanmean(pace[graded & ball_seen]))
    bins = {"5-10": (5, 10), "10-20": (10, 20), "20-60": (20, 60), "60+": (60, 1e9)}
    table_pace = {}
    print("     pace in gaps  bin: mean m/s, ratio to live pace, ratio to ball-visible pace, dead share")
    for name, (low, high) in bins.items():
        frames = [np.arange(a, b) for a, b in gaps if low < (b - a) / FPS <= high]
        joined = np.concatenate(frames) if frames else np.array([], dtype=np.int64)
        mean = float(np.nanmean(pace[joined])) if len(joined) else float("nan")
        dead_share = float(np.mean(dead[joined])) if len(joined) else float("nan")
        table_pace[name] = {"mps": mean, "ratio_live": mean / pace_live, "ratio_visible": mean / pace_seen,
                            "dead_share": dead_share, "gaps": len(frames)}
        print("       {:>6}: {:.2f}  {:.2f}  {:.2f}  {:.2f}  ({} gaps)".format(
            name, mean, mean / pace_live, mean / pace_seen, dead_share, len(frames)))
    print("     live pace {:.2f} m/s, ball-visible pace {:.2f} m/s, dead pace {:.2f} m/s".format(
        pace_live, pace_seen, float(np.nanmean(pace[dead]))))
    stats_out.update({
        "mean_tracked_graded": mean_graded, "share_over_22": over, "visible_all": visible_all,
        "visible_graded": visible_graded, "gap_counts": table, "longest_gap_s": float(seconds.max()),
        "median_gap_over_5s": float(np.median(over_5)), "missing_share": missing_share,
        "missing_dead_share": missing_dead, "ball_speed_fill": fill, "pace": table_pace,
        "pace_live": pace_live, "pace_visible": pace_seen,
    })


def nearest_team(rows, team_of_slot):
    """Per frame: team of the tracked object nearest the ball in metres, -1 where there is no answer."""
    xs = rows[:, 1::3][:, :N_SLOTS]
    ys = rows[:, 2::3][:, :N_SLOTS]
    bx, by = rows[:, 139], rows[:, 140]
    codes = np.full(N_FRAMES, -1, dtype=np.int8)
    seen = np.flatnonzero(np.isfinite(bx))
    for begin in range(0, len(seen), 50000):
        chunk = seen[begin:begin + 50000]
        dx = (xs[chunk] - bx[chunk, None]) * PITCH[0]
        dy = (ys[chunk] - by[chunk, None]) * PITCH[1]
        distance = np.where(np.isfinite(dx), dx * dx + dy * dy, np.inf)
        closest = np.argmin(distance, axis=1)
        found = np.isfinite(distance[np.arange(len(chunk)), closest])
        codes[chunk[found]] = team_of_slot[closest[found]]
    return codes


def vote(codes, window, home_code, away_code, dead_code):
    """Centred majority of home against away over +/- window frames. Dead where neither is present."""
    if window == 0:
        out = np.full(N_FRAMES, dead_code, dtype=np.int8)
        out[codes == 0] = home_code
        out[codes == 1] = away_code
        return out
    home = np.concatenate(([0], np.cumsum(codes == 0)))
    away = np.concatenate(([0], np.cumsum(codes == 1)))
    index = np.arange(N_FRAMES)
    low = np.maximum(0, index - window)
    high = np.minimum(N_FRAMES, index + window + 1)
    n_home = home[high] - home[low]
    n_away = away[high] - away[low]
    out = np.full(N_FRAMES, dead_code, dtype=np.int8)
    seen = n_home + n_away > 0
    out[seen] = np.where(n_home[seen] >= n_away[seen], home_code, away_code)
    return out


def line_level_agreement(raw_path, codes, team_of_slot, step=20):
    """Parse a sample of raw lines the plain way (split, first character "N" means missing) and
    compare the nearest team with the vectorised answer. This is the reading contract simple
    readers of the format rely on."""
    length, width = PITCH
    compared = wrong = 0
    with open(raw_path) as handle:
        for number, line in enumerate(handle):
            if number % step:
                continue
            parts = line.split(":")
            ball = parts[2]
            answer = -1
            if ball[0] != "N":
                fields = ball.split(",")
                ball_x, ball_y = float(fields[0]) * length, float(fields[1]) * width
                best = None
                for slot, group in enumerate(parts[1].split(";")):
                    if group[0] == "N":
                        continue
                    fields = group.split(",")
                    dx = float(fields[0]) * length - ball_x
                    dy = float(fields[1]) * width - ball_y
                    distance = dx * dx + dy * dy
                    if best is None or distance < best:
                        best, answer = distance, int(team_of_slot[slot])
            compared += 1
            wrong += answer != int(codes[int(parts[0]) - 1])
    # float32 rows against text can split a near tie, so allow a sliver.
    check("plain line parsing agrees with the array reader", wrong <= compared * 0.001,
          "{} of {} sampled frames differ".format(wrong, compared))


def score_baseline(rows, team_of_slot, twin, scorer_path, stats_out, raw_path):
    spec = importlib.util.spec_from_file_location("spellmetric_clean", scorer_path)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    gt = scorer.read_labels(os.path.join(twin, "gt", "possession_gt_full.csv"))
    public = scorer.read_labels(os.path.join(twin, "gt", "possession_gt_public_like.csv"))
    check("scorer reads 199,319 labels", len(gt) == N_FRAMES, len(gt))

    codes = nearest_team(rows, team_of_slot)
    line_level_agreement(raw_path, codes, team_of_slot)
    out = {}
    print("     baseline on the twin: window, spellF1, macroF1, true possessions, predicted, (real spellF1, macroF1)")
    for window, key in ((0, "raw"), (25, 25), (50, 50), (125, 125)):
        pred = vote(codes, window, scorer.H, scorer.A, scorer.D)
        spell, n_true, n_pred, per_threshold = scorer.spell_f1(gt, pred)
        macro = scorer.macro_f1(gt, pred)
        public_spell, public_true, public_pred, _ = scorer.spell_f1(public, pred)
        public_macro = scorer.macro_f1(public, pred)
        real_macro, real_spell = REAL["baseline"][key]
        out[str(key)] = {"spellF1": spell, "macroF1": macro, "true_possessions": n_true, "predicted_possessions": n_pred,
                         "per_iou": per_threshold, "public_like": {"spellF1": public_spell, "macroF1": public_macro,
                                                                   "true": public_true, "predicted": public_pred},
                         "real_spellF1": real_spell, "real_macroF1": real_macro}
        print("       +/-{:<4} {:.4f}  {:.4f}  {:>5}  {:>6}   ({:.4f}, {:.4f})  public-like {:.4f} {:.4f}".format(
            window, spell, macro, n_true, n_pred, real_spell, real_macro, public_spell, public_macro))
    main = out["50"]
    check("baseline +/-50 runs and scores", 0.0 <= main["spellF1"] <= 1.0 and 0.0 <= main["macroF1"] <= 1.0,
          "spellF1 {:.4f} macroF1 {:.4f}, true {} predicted {}".format(
              main["spellF1"], main["macroF1"], main["true_possessions"], main["predicted_possessions"]))
    stats_out["baseline"] = out


def check_kloppy(paths, rows, stats_out):
    from kloppy import metrica

    started = time.time()
    dataset = metrica.load_tracking_epts(meta_data=paths["atd_meta"], raw_data=paths["atd_raw"])
    records = dataset.records
    check("kloppy loads every frame", len(records) == N_FRAMES, "{} records in {:.0f} s".format(len(records), time.time() - started))
    meta = dataset.metadata
    check("kloppy teams", meta.teams[0].name == "TWIN-HOME" and meta.teams[1].name == "TWIN-AWAY",
          [team.name for team in meta.teams])
    n_players = len(meta.teams[0].players) + len(meta.teams[1].players)
    check("kloppy sees 46 tracks", n_players == N_SLOTS, n_players)
    check("kloppy pitch 105 x 68", meta.pitch_dimensions.pitch_length == 105 and meta.pitch_dimensions.pitch_width == 68,
          "{} x {}".format(meta.pitch_dimensions.pitch_length, meta.pitch_dimensions.pitch_width))
    check("kloppy frame rate 25", meta.frame_rate == FPS, meta.frame_rate)
    ids = [int(record.frame_id) for record in records]
    check("kloppy ids are 1..199319", ids == list(range(1, N_FRAMES + 1)))
    in_period = sum(record.period is not None for record in records)
    print("     kloppy periods {}, frames inside a period {}".format([p.id for p in meta.periods], in_period))

    # Spot check: values through kloppy against the raw file.
    players_by_id = {p.player_id: p for team in meta.teams for p in team.players}
    worst = 0.0
    seen_ball = 0
    for video in (12000, 50000, 90000, 120000, 180000):
        record = records[video]
        row = rows[video]
        if np.isfinite(row[139]):
            seen_ball += 1
            worst = max(worst, abs(record.ball_coordinates.x - row[139]), abs(record.ball_coordinates.y - row[140]))
        for slot in range(N_SLOTS):
            data = record.players_data[players_by_id[str(slot + 1)]]
            if np.isfinite(row[1 + 3 * slot]):
                worst = max(worst, abs(data.coordinates.x - row[1 + 3 * slot]), abs(data.coordinates.y - row[2 + 3 * slot]),
                            abs(data.speed - row[3 + 3 * slot]))
    check("kloppy values match the raw file", worst < 2e-3, "worst difference {:.6f}, ball seen on {}/5".format(worst, seen_ball))
    del records, dataset

    small = metrica.load_tracking_epts(meta_data=paths["atd_meta"], raw_data=paths["atd_raw"], limit=11300)
    frame = small.to_df()
    columns = list(frame.columns)
    wanted = {"frame_id", "ball_x", "ball_y", "1_x", "1_y", "1_s", "46_x", "46_y", "46_s"}
    check("to_df has the columns DATA.md names", wanted <= set(columns) and len(frame) == 11300,
          "{} rows, first columns {}".format(len(frame), columns[:8]))
    stats_out["kloppy_periods"] = [p.id for p in meta.periods]

    from kloppy import sportscode

    tags = sportscode.load(paths["smart_tagging"]).to_df()
    wanted = {"code", "timestamp", "end_timestamp", "Team", "Half", "Type", "Side",
              "Direction of ball entry", "Max Players in the box"}
    check("kloppy SportsCode loader reads the tags", wanted <= set(tags.columns), "{} rows, {} columns".format(len(tags), len(tags.columns)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--twin", required=True)
    parser.add_argument("--scorer", required=True, help="analysis/spellmetric.py")
    parser.add_argument("--games", help="upstream games.py, read as text only")
    parser.add_argument("--json", help="write results here")
    parser.add_argument("--no-kloppy", action="store_true")
    parser.add_argument("--expect-dead-share", type=float, default=0.6, help="gap_dead_share the twin was built with")
    args = parser.parse_args()

    paths = paths_of(args.twin)
    for name, path in paths.items():
        check("file exists: {}".format(os.path.basename(path)), os.path.isfile(path))
    check("notice file present", os.path.isfile(os.path.join(args.twin, "TWIN_NOT_REAL_DATA.txt")))
    if args.games:
        check_names_against_games(args.games)

    stats = {}
    rows = read_raw(paths["atd_raw"])
    team_of_slot = check_metadata(paths["atd_meta"])
    check_homography(paths["homography"], rows)
    labels, graded = check_gt(args.twin)
    stats["tags"] = check_tags(paths["smart_tagging"], labels)
    degradation(rows, labels, graded, team_of_slot, stats, args.expect_dead_share)
    score_baseline(rows, team_of_slot, args.twin, args.scorer, stats, paths["atd_raw"])
    if not args.no_kloppy:
        check_kloppy(paths, rows, stats)

    failed = [item for item in results if not item["ok"]]
    print("\n{} checks, {} failed".format(len(results), len(failed)))
    if args.json:
        with open(args.json, "w") as handle:
            json.dump({"checks": results, "stats": stats}, handle, indent=1, sort_keys=True, default=float)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
