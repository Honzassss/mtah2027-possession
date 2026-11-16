"""SkillCorner open data: broadcast tracking at 10 fps, one directory per match.

`ball_xy` and `players_xy` hold detections only, because the competition's tracking does not fill in
what the camera missed. SkillCorner's extrapolated positions stay in extras, for tests that need to
know where an unseen player or ball probably was.

Labels are H or A where `possession.group` names a team and X everywhere else. A null group covers
about half of the in-play time, far more than dead-ball time, because SkillCorner sets it only while
a player holds the ball. It is not the competition's D, so it is not written as D.
"""
import json
import os

import numpy as np

from .schema import AWAY, HOME, FormatError, Match

GROUP_LABEL = {"home team": "H", "away team": "A", None: "X"}
KEEPER_ACRONYM = "GK"
CORNERS = ("top_left", "bottom_left", "bottom_right", "top_right")


def _number(value):
    return float("nan") if value is None else value


def load_skillcorner(match_dir):
    """Read data/matches/<id>/: <id>_match.json and <id>_tracking_extrapolated.jsonl."""
    match_id = os.path.basename(os.path.normpath(match_dir))
    meta_path = os.path.join(match_dir, match_id + "_match.json")
    tracking_path = os.path.join(match_dir, match_id + "_tracking_extrapolated.jsonl")
    for path in (meta_path, tracking_path):
        if not os.path.isfile(path):
            raise FormatError("SkillCorner file not found: {}".format(path))

    with open(meta_path, encoding="utf-8") as handle:
        meta = json.load(handle)
    home_id, away_id = meta["home_team"]["id"], meta["away_team"]["id"]
    home_first = sorted(meta["players"], key=lambda p: (p["team_id"] != home_id, p["id"]))
    if {p["team_id"] for p in home_first} - {home_id, away_id}:
        raise FormatError("SkillCorner roster names a third team")
    slot_of = {p["id"]: i for i, p in enumerate(home_first)}

    with open(tracking_path, encoding="utf-8") as handle:
        n = sum(1 for line in handle if line.strip())
    slots = len(home_first)
    frame = np.zeros(n, dtype=np.int64)
    period = np.zeros(n, dtype=np.int8)
    labels = np.full(n, "X", dtype="<U1")
    ball_all = np.full((n, 3), np.nan, dtype=np.float32)
    ball_detected = np.zeros(n, dtype=bool)
    polygon = np.full((n, 4, 2), np.nan, dtype=np.float32)
    modelled = np.full((n, slots, 2), np.nan, dtype=np.float32)
    detected = np.zeros((n, slots), dtype=bool)
    possessor = np.full(n, -1, dtype=np.int16)

    with open(tracking_path, encoding="utf-8") as handle:
        for i, record in enumerate(json.loads(line) for line in handle if line.strip()):
            frame[i] = record["frame"]
            period[i] = record["period"] or 0
            group = record["possession"]["group"]
            if group not in GROUP_LABEL:
                raise FormatError("SkillCorner possession group outside home team, away team and null")
            labels[i] = GROUP_LABEL[group]
            holder = record["possession"]["player_id"]
            if holder is not None:
                possessor[i] = slot_of[holder]
            ball = record["ball_data"]
            ball_all[i] = (_number(ball["x"]), _number(ball["y"]), _number(ball["z"]))
            ball_detected[i] = ball["is_detected"] is True
            view = record["image_corners_projection"]
            polygon[i] = [(_number(view["x_" + c]), _number(view["y_" + c])) for c in CORNERS]
            for player in record["player_data"]:
                slot = slot_of[player["player_id"]]
                modelled[i, slot] = (player["x"], player["y"])
                detected[i, slot] = player["is_detected"] is True

    players_xy = np.where(detected[..., None], modelled, np.nan)
    ball_xy = np.where(ball_detected[:, None], ball_all[:, :2], np.nan)
    ball_z = np.where(ball_detected, ball_all[:, 2], np.nan)
    # A detection without a position, or a position with only one coordinate, is no observation.
    ball_xy[np.isnan(ball_xy).any(axis=1)] = np.nan
    return Match(
        source="skillcorner",
        match_id=match_id,
        fps=10.0,
        pitch=(float(meta["pitch_length"]), float(meta["pitch_width"])),
        team_names=(meta["home_team"]["name"], meta["away_team"]["name"]),
        frame=frame,
        period=period,
        ball_xy=ball_xy,
        ball_z=ball_z,
        players_xy=players_xy,
        slot_team=np.array([HOME if p["team_id"] == home_id else AWAY for p in home_first], dtype=np.int8),
        slot_id=np.array([str(p["id"]) for p in home_first]),
        labels=labels,
        view_polygon=polygon,
        extras={
            "ball_xy_modelled": ball_all[:, :2],
            "ball_z_modelled": ball_all[:, 2],
            "players_xy_modelled": modelled,
            "possessor_slot": possessor,
            "slot_is_keeper": np.array([p["player_role"]["acronym"] == KEEPER_ACRONYM for p in home_first]),
            "home_attack_direction": tuple(meta["home_team_side"]),
        },
    ).validate()
