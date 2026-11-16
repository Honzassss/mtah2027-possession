"""One schema for every tracking source.

Every later stage reads only these fields, so code written and debugged on open data runs unchanged
on the competition bundle.

Coordinates
    Metres, origin at the pitch centre. x runs along the long side. y runs along the short side and
    grows away from the main camera, towards the top of the picture. NaN means no observation, for
    players and ball alike, and x and y are always missing together.

Frames
    One row per source frame, in order, with a frame number that goes up by one inside a half.
    `frame` is the source's own number; in the competition format it is the video frame, which is
    also the submission row. `period` is 1 or 2 inside a half and 0 outside both.

Slots
    `players_xy` has one column per slot, and `slot_team` says whose slot it is: 0 home, 1 away. In
    IDSSE and SkillCorner a slot is a person. In the competition format it is a tracklet holder, not
    a person, and its team comes from the metadata, so no code may assume a range of slots.

Labels
    H, A, D, or X for "no label on this frame". None where the source has no labels at all, which
    includes the competition bundle: the adapter never reads a ground-truth file.

`extras` holds source-specific arrays that the pipeline must not feed to a model, such as the raw
speed fields. They exist for the self-checks.

Errors raised here and in the adapters name fields and counts only, never a value or a frame number
from the file, so a failure on the competition bundle can be reported without showing any of it.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

HOME, AWAY = 0, 1
LABEL_ALPHABET = ("H", "A", "D", "X")


class FormatError(ValueError):
    """A source file is not what its documentation says. The message never quotes the file."""


class SchemaError(ValueError):
    """An adapter built something that breaks the schema. The message never quotes the data."""


def _require(condition, message):
    if not condition:
        raise SchemaError(message)


@dataclass
class Match:
    source: str
    match_id: str
    fps: float
    pitch: tuple                # (length, width) in metres
    team_names: tuple           # (home, away)
    frame: np.ndarray           # (F,) int64
    period: np.ndarray          # (F,) int8, 1 or 2, 0 outside both halves
    ball_xy: np.ndarray         # (F, 2) float32
    players_xy: np.ndarray      # (F, S, 2) float32
    slot_team: np.ndarray       # (S,) int8
    slot_id: np.ndarray         # (S,) source's name for the slot
    ball_z: Optional[np.ndarray] = None         # (F,) float32, None where the source has no height
    labels: Optional[np.ndarray] = None         # (F,) "<U1"
    forced_dead: Optional[np.ndarray] = None    # (F,) bool, published as dead before any modelling
    view_polygon: Optional[np.ndarray] = None   # (F, 4, 2), image corners TL, BL, BR, TR on the ground
    extras: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.forced_dead is None:
            self.forced_dead = np.zeros(len(self.frame), dtype=bool)

    @property
    def n_frames(self):
        return len(self.frame)

    @property
    def n_slots(self):
        return len(self.slot_team)

    def validate(self):
        """Check shapes and conventions. Returns self, so an adapter can end with Match(...).validate()."""
        n, slots = self.n_frames, self.n_slots
        _require(self.fps > 0, "fps must be positive")
        _require(len(self.pitch) == 2 and min(self.pitch) > 0, "pitch must be (length, width) in metres")
        _require(len(self.team_names) == 2, "team_names must be (home, away)")
        _require(n > 0 and slots > 0, "no frames or no slots")
        _require(self.frame.shape == (n,) and self.period.shape == (n,), "frame and period must be (F,)")
        _require(self.ball_xy.shape == (n, 2), "ball_xy must be (F, 2)")
        _require(self.players_xy.shape == (n, slots, 2), "players_xy must be (F, S, 2)")
        _require(len(self.slot_id) == slots, "slot_id must have one entry per slot")
        _require(np.isin(self.slot_team, (HOME, AWAY)).all(), "slot_team must be 0 or 1")
        _require(np.isin(self.period, (0, 1, 2)).all(), "period must be 0, 1 or 2")
        _require((self.period > 0).any(), "no frame falls inside a half")
        _require(self.forced_dead.shape == (n,) and self.forced_dead.dtype == bool, "forced_dead must be (F,) bool")
        for half in (1, 2):
            numbers = self.frame[self.period == half]
            _require(np.all(np.diff(numbers) == 1), "frame numbers must rise by one inside a half")
        for name, xy in (("ball_xy", self.ball_xy), ("players_xy", self.players_xy)):
            missing_x, missing_y = np.isnan(xy[..., 0]), np.isnan(xy[..., 1])
            _require(np.array_equal(missing_x, missing_y), name + ": x and y must be missing together")
            _require(not np.isinf(xy).any(), name + ": infinite coordinates")
        if self.ball_z is not None:
            _require(self.ball_z.shape == (n,), "ball_z must be (F,)")
        if self.labels is not None:
            _require(self.labels.shape == (n,), "labels must be (F,)")
            _require(np.isin(self.labels, LABEL_ALPHABET).all(), "labels must be H, A, D or X")
        if self.view_polygon is not None:
            _require(self.view_polygon.shape == (n, 4, 2), "view_polygon must be (F, 4, 2)")
        return self
