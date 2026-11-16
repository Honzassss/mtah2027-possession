"""FIFA/EPTS pair: the competition's tactical bundle and Metrica's sample game 3.

Own small parser. The metadata decides which field of a raw line belongs to which slot, per
DataFormatSpecification, because Metrica game 3 has a new layout after every substitution. The team
of a slot comes from its teamId, the first Team in the file is home, and no slot range is assumed.
tests/check_adapters.py compares this parser with kloppy 3.19.1 on game 3.

The competition bundle is read through an explicit path. Nothing here consults the environment, and
nothing here opens a ground-truth file. Errors carry counts only, never a value or a frame number.
"""
import os
import warnings
import xml.etree.ElementTree as ET

import numpy as np

from .schema import AWAY, HOME, FormatError, Match

# Published in the challenge's public documents: DATA.md ("Which frames are graded"), games.py (pitch)
# and FAQ.md (the 2,934 graded frames before the second-half kickoff are all D).
COMPETITION_STEM = "BAR-ATH_tactical"
COMPETITION_ID = "BAR-ATH"
COMPETITION_FRAMES = 199319
COMPETITION_PITCH = (105.3, 68.0)
COMPETITION_HALVES = ((1, 11185, 90651), (2, 110808, 190197))
COMPETITION_DEAD_BLOCKS = ((110808, 113741),)
IMAGE_SIZE = (1920, 1080)

COMPETITION_SUFFIXES = {"meta": "_atd.xml", "raw": "_atd.txt", "homography": "_homography.csv"}
HOMOGRAPHY_HEADER = "frame,h00,h01,h02,h10,h11,h12,h20,h21,h22,reprojection_error,valid"
KEEPER_POSITION_TYPE = "Goalkeeper"


def _number(text, kind=float):
    """A number from the metadata, or a FormatError that does not quote it."""
    try:
        return kind(text)
    except (TypeError, ValueError):
        raise FormatError("metadata holds a number that cannot be read") from None


def _parameters(element, tag):
    """{Name: Value} of the provider parameters under one element."""
    block = element.find(tag) if element is not None else None
    if block is None:
        return {}
    return {p.findtext("Name"): p.findtext("Value") for p in block.findall("ProviderParameter")}


def read_metadata(path):
    """The parts of the metadata XML that the reader needs."""
    if not os.path.isfile(path):
        raise FormatError("metadata file not found: {}".format(path))
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        raise FormatError("metadata is not well-formed XML") from None

    teams = list(root.iter("Team"))
    if len(teams) != 2 or not all(t.get("id") for t in teams):
        raise FormatError("metadata must list exactly two teams with ids")
    team_ids = [t.get("id") for t in teams]
    players = list(root.iter("Player"))
    if not players:
        raise FormatError("metadata lists no players")
    if {p.get("teamId") for p in players} - set(team_ids):
        raise FormatError("metadata has a player whose teamId is not one of the two teams")
    slot_ids = [p.get("id") for p in players]
    if len(set(slot_ids)) != len(slot_ids):
        raise FormatError("metadata repeats a player id")

    channels = {}
    for channel in root.iter("PlayerChannel"):
        channels[channel.get("id")] = (channel.get("playerId"), channel.get("channelId"))
    specs = sorted(((_number(s.get("startFrame"), int), _number(s.get("endFrame"), int), s)
                    for s in root.iter("DataFormatSpecification")), key=lambda item: item[0])
    if not specs:
        raise FormatError("metadata has no DataFormatSpecification")
    separators = {e.get("separator") for e in root.iter() if e.get("separator")}

    global_parameters = _parameters(root.find(".//GlobalConfig"), "ProviderGlobalParameters")
    return {
        "team_names": tuple(t.findtext("Name") for t in teams),
        "team_ids": team_ids,
        "attack_direction_first_half": tuple(
            _parameters(t, "ProviderTeamsParameters").get("attack_direction_first_half") for t in teams),
        "slot_ids": slot_ids,
        "slot_team": np.array([HOME if p.get("teamId") == team_ids[0] else AWAY for p in players], dtype=np.int8),
        "slot_is_keeper": np.array([
            _parameters(p, "ProviderPlayerParameters").get("position_type") == KEEPER_POSITION_TYPE for p in players]),
        "channels": channels,
        "specs": specs,
        "separators": separators,
        "fps": _number(root.findtext(".//GlobalConfig/FrameRate")),
        "field_size": (_number(root.findtext(".//FieldSize/Width")), _number(root.findtext(".//FieldSize/Height"))),
        "global_parameters": global_parameters,
    }


def _halves_from_metadata(parameters):
    """((half, first, last), ...) on the frame = frame_id - 1 timeline, or None if the XML has none."""
    halves = []
    for half, name in ((1, "first_half"), (2, "second_half")):
        first, last = parameters.get(name + "_start"), parameters.get(name + "_end")
        if first and last:
            halves.append((half, _number(first, int) - 1, _number(last, int) - 1))
    return tuple(halves) or None


def _map_fields(spec, line, table):
    """Field index of every declared channel, worked out by walking the metadata over one raw line.

    A raw group may hold more fields than the metadata declares, for example a trailing speed on the
    ball; those are skipped. A line with fewer groups than declared cannot be read.
    """
    fields = {"players": {}, "ball": {}, "frame": None}
    cursor = 0

    def tokens(text):
        return text.translate(table).count(",") + 1

    def walk(element, text):
        nonlocal cursor
        children = list(element)
        if not children:
            if element.tag == "PlayerChannelRef":
                fields["players"][element.get("playerChannelId")] = cursor
            elif element.tag == "BallChannelRef":
                fields["ball"][element.get("channelId")] = cursor
            elif element.tag == "StringRegister" and fields["frame"] is None:
                fields["frame"] = cursor
            cursor += tokens(text)
            return
        pieces = text.split(element.get("separator"))
        if len(pieces) < len(children):
            raise FormatError("raw lines hold fewer groups than the metadata declares")
        for child, piece in zip(children, pieces):
            walk(child, piece)
        for piece in pieces[len(children):]:
            cursor += tokens(piece)

    walk(spec, line.rstrip("\r\n"))
    if fields["frame"] is None:
        raise FormatError("metadata declares no frame counter")
    return fields, cursor


class _Layout:
    """Numpy indices for one DataFormatSpecification."""

    def __init__(self, fields, n_fields, channels, slot_of):
        by_channel = {"x": [], "y": [], "s": []}
        for reference, column in fields["players"].items():
            if reference not in channels:
                raise FormatError("metadata references an undeclared player channel")
            player, channel = channels[reference]
            if player not in slot_of:
                raise FormatError("metadata references an undeclared player")
            if channel in by_channel:
                by_channel[channel].append((slot_of[player], column))
        if {s for s, _ in by_channel["x"]} != {s for s, _ in by_channel["y"]}:
            raise FormatError("metadata gives a player x without y, or y without x")
        if "x" not in fields["ball"] or "y" not in fields["ball"]:
            raise FormatError("metadata declares no ball position")
        self.n_fields = n_fields
        self.frame = fields["frame"]
        self.xy_slots = np.array([s for s, _ in sorted(by_channel["x"])], dtype=np.intp)
        self.x = np.array([c for _, c in sorted(by_channel["x"])], dtype=np.intp)
        self.y = np.array([c for _, c in sorted(by_channel["y"])], dtype=np.intp)
        self.s_slots = np.array([s for s, _ in sorted(by_channel["s"])], dtype=np.intp)
        self.s = np.array([c for _, c in sorted(by_channel["s"])], dtype=np.intp)
        self.ball = np.array([fields["ball"]["x"], fields["ball"]["y"]], dtype=np.intp)
        self.ball_s = fields["ball"].get("s")


def read_raw(path, meta):
    """(frame_id, slot_x, slot_y, slot_s, ball_x, ball_y, ball_s), normalised as in the file."""
    if not os.path.isfile(path):
        raise FormatError("raw file not found: {}".format(path))
    slot_of = {slot_id: i for i, slot_id in enumerate(meta["slot_ids"])}
    table = str.maketrans({sep: "," for sep in meta["separators"] if sep != ","})
    specs = meta["specs"]

    with open(path, encoding="ascii", errors="replace") as handle:
        n = sum(1 for line in handle if line.strip())
    if n == 0:
        raise FormatError("raw file is empty")
    slots = len(slot_of)
    frame_id = np.zeros(n, dtype=np.int64)
    slot_x = np.full((n, slots), np.nan, dtype=np.float32)
    slot_y = np.full((n, slots), np.nan, dtype=np.float32)
    slot_s = np.full((n, slots), np.nan, dtype=np.float32)
    ball = np.full((n, 3), np.nan, dtype=np.float32)

    spec_index = -1
    layout = None
    last_frame = 0
    bad_lines = beyond_layouts = 0
    i = 0
    with open(path, encoding="ascii", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                values = np.fromstring(line.translate(table), sep=",", dtype=np.float64)
                number = int(values[layout.frame if layout else 0])
            except (ValueError, IndexError, OverflowError):
                # The messages of these errors can quote the file, so they stop here.
                bad_lines += 1
                i += 1
                continue
            while number > last_frame and spec_index + 1 < len(specs):
                spec_index += 1
                _, last_frame, spec = specs[spec_index]
                layout = _Layout(*_map_fields(spec, line, table), meta["channels"], slot_of)
            if number > last_frame:
                beyond_layouts += 1
            elif len(values) != layout.n_fields:
                bad_lines += 1
            else:
                frame_id[i] = number
                slot_x[i, layout.xy_slots] = values[layout.x]
                slot_y[i, layout.xy_slots] = values[layout.y]
                if layout.s.size:
                    slot_s[i, layout.s_slots] = values[layout.s]
                ball[i, :2] = values[layout.ball]
                if layout.ball_s is not None:
                    ball[i, 2] = values[layout.ball_s]
            i += 1
    if bad_lines:
        raise FormatError("{} raw lines could not be read with the declared layout".format(bad_lines))
    if beyond_layouts:
        raise FormatError("{} raw lines fall after the last declared layout".format(beyond_layouts))
    if not np.all(np.diff(frame_id) == 1):
        raise FormatError("raw frame ids are not contiguous")
    return frame_id, slot_x, slot_y, slot_s, ball[:, 0], ball[:, 1], ball[:, 2]


def read_homography(path, n_frames):
    """(H (F, 3, 3), reprojection_error (F,), valid (F,) bool). H maps normalised pitch to pixels."""
    if not os.path.isfile(path):
        raise FormatError("homography file not found: {}".format(path))
    with open(path, encoding="ascii", errors="replace") as handle:
        header = handle.readline().strip()
    if header != HOMOGRAPHY_HEADER:
        raise FormatError("homography header differs from DATA.md")
    try:
        rows = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64, ndmin=2)
    except ValueError:
        raise FormatError("homography holds a row that is not numeric") from None
    if rows.shape != (n_frames, 12):
        raise FormatError("homography does not have one row of 12 columns per tracking frame")
    if not np.array_equal(rows[:, 0], np.arange(n_frames)):
        raise FormatError("homography frames do not count from 0 in order")
    return rows[:, 1:10].reshape(-1, 3, 3), rows[:, 10], rows[:, 11] == 1


def view_polygon_from_homography(matrices, valid, pitch, image_size=IMAGE_SIZE):
    """Ground footprint of the image, (F, 4, 2) in schema metres, corners TL, BL, BR, TR.

    NaN where the matrix is invalid or singular, and for a corner above the horizon, which has no
    ground point at all.
    """
    length, width = pitch
    image_w, image_h = image_size
    corners = np.array([[0, 0, 1], [0, image_h, 1], [image_w, image_h, 1], [image_w, 0, 1]], dtype=np.float64).T
    polygon = np.full((len(matrices), 4, 2), np.nan, dtype=np.float32)
    usable = valid & np.isfinite(matrices).all(axis=(1, 2))
    usable[usable] = np.abs(np.linalg.det(matrices[usable])) > 1e-12
    if not usable.any():
        return polygon
    ground = np.linalg.inv(matrices[usable]) @ corners
    in_front = ground[:, 2] > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        x_m = (ground[:, 0] / ground[:, 2] - 0.5) * length
        y_m = (0.5 - ground[:, 1] / ground[:, 2]) * width
    polygon[usable, :, 0] = np.where(in_front, x_m, np.nan)
    polygon[usable, :, 1] = np.where(in_front, y_m, np.nan)
    return polygon


def load_epts(meta_path, raw_path, homography_path=None, *, source="epts", match_id=None, pitch=None,
              halves=None, dead_blocks=(), n_frames=None):
    """Read an EPTS pair into the schema.

    pitch        (length, width) in metres, default the metadata's FieldSize.
    halves       ((half, first, last), ...) on frame = frame_id - 1, default the metadata's.
    dead_blocks  ((first, last), ...) published as dead, marked in forced_dead.
    n_frames     stop if the raw file has any other number of frames.
    """
    meta = read_metadata(meta_path)
    frame_id, slot_x, slot_y, slot_s, ball_x, ball_y, ball_s = read_raw(raw_path, meta)
    n = len(frame_id)
    if n_frames is not None and n != n_frames:
        raise FormatError("raw file does not have the expected number of frames")
    frame = frame_id - 1

    length, width = pitch if pitch is not None else meta["field_size"]
    halves = halves if halves is not None else _halves_from_metadata(meta["global_parameters"])
    if not halves:
        raise FormatError("no half boundaries: pass them, or use metadata that has them")
    period = np.zeros(n, dtype=np.int8)
    for half, first, last in halves:
        period[(frame >= first) & (frame <= last)] = half
    forced_dead = np.zeros(n, dtype=bool)
    for first, last in dead_blocks:
        forced_dead[(frame >= first) & (frame <= last)] = True

    # x and y go missing together in the schema, whatever the file does.
    players_missing = np.isnan(slot_x) | np.isnan(slot_y)
    players_xy = np.stack(((slot_x - 0.5) * length, (0.5 - slot_y) * width), axis=-1)
    players_xy[players_missing] = np.nan
    ball_missing = np.isnan(ball_x) | np.isnan(ball_y)
    ball_xy = np.stack(((ball_x - 0.5) * length, (0.5 - ball_y) * width), axis=-1)
    ball_xy[ball_missing] = np.nan

    polygon = None
    extras = {
        "slot_speed_raw": slot_s,
        "ball_speed_raw": ball_s,
        "slot_is_keeper": meta["slot_is_keeper"],
        "attack_direction_first_half": meta["attack_direction_first_half"],
    }
    if homography_path is not None:
        matrices, error, valid = read_homography(homography_path, n)
        polygon = view_polygon_from_homography(matrices, valid, (length, width))
        extras.update(homography=matrices, homography_error=error, homography_valid=valid)

    return Match(
        source=source,
        match_id=match_id or os.path.splitext(os.path.basename(raw_path))[0],
        fps=meta["fps"],
        pitch=(float(length), float(width)),
        team_names=meta["team_names"],
        frame=frame,
        period=period,
        ball_xy=ball_xy,
        players_xy=players_xy,
        slot_team=meta["slot_team"],
        slot_id=np.array(meta["slot_ids"]),
        forced_dead=forced_dead,
        view_polygon=polygon,
        extras=extras,
    ).validate()


def load_competition(tactical_dir, with_homography=True):
    """The tactical bundle in the directory passed in, on the published timeline."""
    paths = {name: os.path.join(tactical_dir, COMPETITION_STEM + suffix) for name, suffix in COMPETITION_SUFFIXES.items()}
    return load_epts(
        paths["meta"],
        paths["raw"],
        paths["homography"] if with_homography else None,
        source="competition",
        match_id=COMPETITION_ID,
        pitch=COMPETITION_PITCH,
        halves=COMPETITION_HALVES,
        dead_blocks=COMPETITION_DEAD_BLOCKS,
        n_frames=COMPETITION_FRAMES,
    )
