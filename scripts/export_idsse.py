"""Export IDSSE (DFL) matches to plain numpy archives with H/A/D frame labels.

    python -I scripts/export_idsse.py external/idsse external/processed/idsse [--stats out.tsv] [IDS...]

Tracking comes through kloppy 3.19.1 with only_alive=False, read from local files. The labels come
from a separate raw pass over the ball FrameSets, because kloppy turns a missing BallPossession into
"away": a frame is D where BallStatus is 0, otherwise BallPossession 1 is H and 2 is A, and any other
raw value stops the export. kloppy's own state and owner are then cross-checked against the raw pass.

Output, one <match>.npz per match:
    frame_id, period            (F,)        DFL frame counter N, half (1 or 2)
    labels                      (F,) "U1"   H, A or D
    ball_status, ball_possession (F,) int8  raw values, -1 where the attribute is absent
    ball_xyz                    (F, 3)      metres, origin at the pitch centre, as recorded
    ball_speed                  (F,)        raw S, km/h per the IDSSE paper
    players_xy                  (F, P, 2)   metres, NaN where that player is not on the pitch
    players_speed               (F, P)      raw S, km/h per the IDSSE paper
    player_team                 (P,) int8   0 home, 1 away
    player_id, player_jersey, player_position (P,)   roster entries, home first
    team_names (2,), pitch (2,) metres, fps

The stats report spells per half (a spell is a maximal run of H or A) and how far the ball moves
during stoppages, because DFL is said to freeze it ("parked").
"""
import argparse
import glob
import math
import os
import time

import numpy as np
from kloppy import sportec
from kloppy.domain import BallState
from lxml import etree

SECTIONS = {"firstHalf": 1, "secondHalf": 2}
PARKED_STEP_M = 0.005  # a per-frame move below this counts as a stationary ball


def raw_ball_frames(positions_path):
    """Per-frame ball attributes straight from the XML, in file order: (period, N, status, poss, x, y, z, s)."""
    rows = []
    period = None
    in_ball = False
    context = etree.iterparse(positions_path, events=("start", "end"),
                              tag=("FrameSet", "Frame"), huge_tree=True)
    for event, elem in context:
        if elem.tag == "FrameSet":
            if event == "start":
                in_ball = elem.get("TeamId") == "BALL"
                if in_ball:
                    section = elem.get("GameSection")
                    if section not in SECTIONS:
                        raise ValueError("unexpected GameSection {!r} on the ball".format(section))
                    period = SECTIONS[section]
            else:
                in_ball = False
                elem.clear()
                while elem.getprevious() is not None:
                    del elem.getparent()[0]
        elif event == "end":
            if in_ball:
                rows.append((period, int(elem.get("N")), int(elem.get("BallStatus", -1)),
                             int(elem.get("BallPossession", -1)), float(elem.get("X")),
                             float(elem.get("Y")), float(elem.get("Z", "nan")), float(elem.get("S", "nan"))))
            elem.clear()
    if not rows:
        raise ValueError("no ball frames in {}".format(positions_path))
    return rows


def labels_from_raw(status, possession):
    """H/A/D. Stops on any status outside {0,1} and any live-ball possession outside {1,2}."""
    bad_status = np.flatnonzero(~np.isin(status, (0, 1)))
    if len(bad_status):
        raise ValueError("BallStatus outside {{0,1}} on {} frames, first index {}, value {}".format(
            len(bad_status), bad_status[0], status[bad_status[0]]))
    alive = status != 0
    bad_possession = np.flatnonzero(alive & ~np.isin(possession, (1, 2)))
    if len(bad_possession):
        raise ValueError("live ball with BallPossession outside {{1,2}} on {} frames, first index {}, value {}".format(
            len(bad_possession), bad_possession[0], possession[bad_possession[0]]))
    labels = np.full(len(status), "D", dtype="<U1")
    labels[alive & (possession == 1)] = "H"
    labels[alive & (possession == 2)] = "A"
    return labels


def spell_lengths(labels, period):
    """Lengths in frames of every maximal H or A run. A half boundary ends a run."""
    out = []
    for half in np.unique(period):
        chunk = labels[period == half]
        edges = np.flatnonzero(chunk[1:] != chunk[:-1]) + 1
        bounds = np.concatenate(([0], edges, [len(chunk)]))
        for start, end in zip(bounds[:-1], bounds[1:]):
            if chunk[start] in ("H", "A"):
                out.append(int(end - start))
    return np.array(out, dtype=np.int64)


def parked_stats(labels, ball_xyz, period):
    """How still the ball is across consecutive dead frames, in metres per frame."""
    same_half = period[1:] == period[:-1]
    dead_step = same_half & (labels[1:] == "D") & (labels[:-1] == "D")
    live_step = same_half & (labels[1:] != "D") & (labels[:-1] != "D")
    steps = np.hypot(np.diff(ball_xyz[:, 0]), np.diff(ball_xyz[:, 1]))
    still = steps < PARKED_STEP_M
    return {
        "dead_frames_still": float(np.mean(still[dead_step])) if dead_step.any() else float("nan"),
        "live_frames_still": float(np.mean(still[live_step])) if live_step.any() else float("nan"),
        "dead_speed_mps": float(np.mean(steps[dead_step]) * 25.0) if dead_step.any() else float("nan"),
        "live_speed_mps": float(np.mean(steps[live_step]) * 25.0) if live_step.any() else float("nan"),
    }


def export_match(match, source_dir, out_dir):
    positions = glob.glob(os.path.join(source_dir, "DFL_04_03_positions_raw_observed_*_DFL-MAT-{}.xml".format(match)))
    info = glob.glob(os.path.join(source_dir, "DFL_02_01_matchinformation_*_DFL-MAT-{}.xml".format(match)))
    if len(positions) != 1 or len(info) != 1:
        raise FileNotFoundError("{}: positions {} matchinformation {}".format(match, positions, info))

    started = time.time()
    raw = raw_ball_frames(positions[0])
    raw_period = np.array([r[0] for r in raw], dtype=np.int8)
    raw_n = np.array([r[1] for r in raw], dtype=np.int64)
    status = np.array([r[2] for r in raw], dtype=np.int8)
    possession = np.array([r[3] for r in raw], dtype=np.int8)
    raw_xyz = np.array([(r[4], r[5], r[6]) for r in raw], dtype=np.float32)
    raw_speed = np.array([r[7] for r in raw], dtype=np.float32)
    labels = labels_from_raw(status, possession)
    del raw

    dataset = sportec.load_tracking(raw_data=positions[0], meta_data=info[0],
                                    only_alive=False, coordinates="sportec")
    frames = dataset.records
    if len(frames) != len(raw_n):
        raise ValueError("{}: kloppy has {} frames, the raw ball has {}".format(match, len(frames), len(raw_n)))

    teams = dataset.metadata.teams
    roster = list(teams[0].players) + list(teams[1].players)
    index_of = {player.player_id: i for i, player in enumerate(roster)}
    n_frames, n_players = len(frames), len(roster)

    players_xy = np.full((n_frames, n_players, 2), np.nan, dtype=np.float32)
    players_speed = np.full((n_frames, n_players), np.nan, dtype=np.float32)
    ball_xyz = np.zeros((n_frames, 3), dtype=np.float32)
    k_period = np.zeros(n_frames, dtype=np.int8)
    k_id = np.zeros(n_frames, dtype=np.int64)
    k_alive = np.zeros(n_frames, dtype=bool)
    k_home = np.zeros(n_frames, dtype=bool)
    for i, frame in enumerate(frames):
        k_id[i] = frame.frame_id
        k_period[i] = frame.period.id
        k_alive[i] = frame.ball_state == BallState.ALIVE
        k_home[i] = frame.ball_owning_team == teams[0]
        point = frame.ball_coordinates
        ball_xyz[i] = (point.x, point.y, point.z)
        for player, data in frame.players_data.items():
            j = index_of[player.player_id]
            players_xy[i, j] = (data.coordinates.x, data.coordinates.y)
            players_speed[i, j] = data.speed

    # kloppy and the raw pass have to tell the same story, frame by frame.
    if not np.array_equal(k_id, raw_n) or not np.array_equal(k_period, raw_period):
        raise ValueError("{}: kloppy frame order differs from the raw file".format(match))
    if not np.array_equal(k_alive, status == 1):
        raise ValueError("{}: kloppy ball state differs from raw BallStatus".format(match))
    alive = status == 1
    if not np.array_equal(k_home[alive], possession[alive] == 1):
        raise ValueError("{}: kloppy owner differs from raw BallPossession".format(match))
    worst = float(np.max(np.abs(ball_xyz[:, :2] - raw_xyz[:, :2])))
    if worst > 1e-3:
        raise ValueError("{}: kloppy coordinates are not raw metres, off by {}".format(match, worst))

    # Frame counter contiguous within each half: the twin relies on one row per 40 ms.
    for half in np.unique(raw_period):
        gaps = np.diff(raw_n[raw_period == half])
        if not np.all(gaps == 1):
            raise ValueError("{}: half {} has frame counter steps {}".format(match, half, np.unique(gaps)))

    pitch = np.array([dataset.metadata.pitch_dimensions.pitch_length,
                      dataset.metadata.pitch_dimensions.pitch_width], dtype=np.float32)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "{}.npz".format(match))
    np.savez_compressed(
        out_path,
        frame_id=raw_n,
        period=raw_period,
        labels=labels,
        ball_status=status,
        ball_possession=possession,
        ball_xyz=ball_xyz,
        ball_speed=raw_speed,
        players_xy=players_xy,
        players_speed=players_speed,
        player_team=np.array([0 if p.team == teams[0] else 1 for p in roster], dtype=np.int8),
        player_id=np.array([p.player_id for p in roster]),
        player_jersey=np.array([p.jersey_no for p in roster], dtype=np.int16),
        player_position=np.array([str(p.starting_position) for p in roster]),
        team_names=np.array([teams[0].name, teams[1].name]),
        pitch=pitch,
        fps=np.array(float(dataset.metadata.frame_rate)),
    )

    spells = spell_lengths(labels, raw_period)
    parked = parked_stats(labels, ball_xyz, raw_period)
    live_missing_possession = int(np.sum((status == 1) & (possession == -1)))
    dead_no_possession = int(np.sum((status == 0) & (possession == -1)))
    row = {
        "match": match,
        "frames": n_frames,
        "frames_h1": int(np.sum(raw_period == 1)),
        "frames_h2": int(np.sum(raw_period == 2)),
        "share_D": float(np.mean(labels == "D")),
        "share_H": float(np.mean(labels == "H")),
        "share_A": float(np.mean(labels == "A")),
        "spells": int(len(spells)),
        "median_spell_s": float(np.median(spells) / 25.0),
        "mean_spell_s": float(np.mean(spells) / 25.0),
        "dead_still": parked["dead_frames_still"],
        "live_still": parked["live_frames_still"],
        "dead_speed_mps": parked["dead_speed_mps"],
        "live_speed_mps": parked["live_speed_mps"],
        "ball_z_present": bool(np.isfinite(ball_xyz[:, 2]).all()),
        "players": n_players,
        "pitch": "{:.2f}x{:.2f}".format(float(pitch[0]), float(pitch[1])),
        "live_missing_possession": live_missing_possession,
        "dead_no_possession": dead_no_possession,
        "seconds": round(time.time() - started, 1),
    }
    return row


def format_value(value):
    if isinstance(value, float):
        return "nan" if math.isnan(value) else "{:.4f}".format(value)
    return str(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source_dir")
    parser.add_argument("out_dir")
    parser.add_argument("matches", nargs="*", help="match ids, default every match in the listing")
    parser.add_argument("--stats", help="write the per-match table here as TSV")
    args = parser.parse_args()

    matches = args.matches or sorted({
        os.path.basename(path).split("DFL-MAT-")[1].split(".")[0]
        for path in glob.glob(os.path.join(args.source_dir, "DFL_04_03_positions_raw_observed_*.xml"))})
    rows = []
    for match in matches:
        row = export_match(match, args.source_dir, args.out_dir)
        rows.append(row)
        print("\t".join("{}={}".format(k, format_value(v)) for k, v in row.items()), flush=True)
        if args.stats:
            with open(args.stats, "w") as handle:
                handle.write("\t".join(rows[0].keys()) + "\n")
                for done in rows:
                    handle.write("\t".join(format_value(v) for v in done.values()) + "\n")


if __name__ == "__main__":
    main()
