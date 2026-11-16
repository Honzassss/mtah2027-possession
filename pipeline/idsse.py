"""IDSSE (DFL) open data, from the archives written by scripts/export_idsse.py."""
import os

import numpy as np

from .schema import FormatError, Match

KEEPER_POSITION = "Goalkeeper"


def load_idsse(path):
    """Read external/processed/idsse/<match>.npz.

    Positions are already metres from the pitch centre. The ball carries height, is present on every
    frame, and is parked in stoppages: that fake cue is left in place here and handled by the
    degrader. Starting keepers are named in extras for the self-checks; a substitute keeper is not.
    """
    if not os.path.isfile(path):
        raise FormatError("IDSSE archive not found: {}".format(path))
    with np.load(path, allow_pickle=False) as data:
        xyz = data["ball_xyz"]
        return Match(
            source="idsse",
            match_id=os.path.splitext(os.path.basename(path))[0],
            fps=float(data["fps"]),
            pitch=tuple(float(v) for v in data["pitch"]),
            team_names=tuple(str(v) for v in data["team_names"]),
            frame=data["frame_id"],
            period=data["period"],
            ball_xy=xyz[:, :2],
            ball_z=xyz[:, 2],
            players_xy=data["players_xy"],
            slot_team=data["player_team"],
            slot_id=data["player_id"],
            labels=data["labels"],
            extras={
                "ball_status": data["ball_status"],
                "ball_possession": data["ball_possession"],
                "ball_speed_raw_kmh": data["ball_speed"],
                "slot_speed_raw_kmh": data["players_speed"],
                "slot_is_keeper": data["player_position"] == KEEPER_POSITION,
            },
        ).validate()
