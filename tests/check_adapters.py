"""Self-checks for the adapters and the cleaning step, on open data and the twin only.

    python -I tests/check_adapters.py [--twin twin/J03WMX] [--metrica external/metrica/data/Sample_Game_3]
        [--idsse external/processed/idsse] [--skillcorner external/skillcorner/data/matches]
        [--json results.json] [--no-kloppy]

Own code and kloppy only: nothing here imports or runs the challenge repository. This script prints
measurements, so it must never be pointed at the competition's data. Exit code 1 if a check fails.

The numeric floors below come from the first full run on this data. They guard against regressions,
they are not claims about the competition's match.
"""
import argparse
import ast
import dataclasses
import hashlib
import json
import os
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline.clean import (CleanConfig, attack_direction, cap_team_size, clean, fragment_velocity,  # noqa: E402
                            gate_ball, split_tracks, stable_deepest)
from pipeline.epts import (COMPETITION_DEAD_BLOCKS, COMPETITION_HALVES, COMPETITION_PITCH,  # noqa: E402
                           load_competition, load_epts)
from pipeline.idsse import load_idsse  # noqa: E402
from pipeline.schema import AWAY, HOME, FormatError, Match, SchemaError  # noqa: E402
from pipeline.skillcorner import load_skillcorner  # noqa: E402

KEEPER_RECALL_FLOOR = {"epts": 0.95, "idsse": 0.90, "skillcorner": 0.70}
results = []


def check(name, ok, detail=""):
    results.append({"name": name, "ok": bool(ok), "detail": str(detail)})
    print("{:4} {:<62} {}".format("PASS" if ok else "FAIL", name, detail), flush=True)


def note(text):
    print("     " + text, flush=True)


def clean_invariants(tag, match, cleaned):
    """What must hold on every source after cleaning."""
    inside = match.period > 0
    tracked = np.isfinite(cleaned.players_xy[..., 0])
    check("{}: nothing survives outside the halves".format(tag),
          not tracked[~inside].any() and not np.isfinite(cleaned.ball_xy[~inside]).any())
    check("{}: fragment and team exist exactly where a track exists".format(tag),
          np.array_equal(cleaned.fragment >= 0, tracked) and np.array_equal(cleaned.players_team >= 0, tracked))
    per_team = max(int((cleaned.players_team == HOME).sum(axis=1).max()), int((cleaned.players_team == AWAY).sum(axis=1).max()))
    check("{}: at most 11 tracks per team".format(tag), per_team <= 11, per_team)
    speed = np.hypot(cleaned.players_vel[..., 0], cleaned.players_vel[..., 1])
    limit = CleanConfig().player_jump_speed
    top = float(np.nanmax(speed))
    check("{}: recomputed player speed stays under the jump limit".format(tag), top <= limit * 1.01, round(top, 1))
    ball_top = float(np.nanmax(np.hypot(*cleaned.ball_vel.T)))
    check("{}: recomputed ball speed stays under the gate".format(tag), ball_top <= CleanConfig().ball_max_speed * 1.01, round(ball_top, 1))
    keeper = cleaned.keeper_slot
    rows = np.arange(match.n_frames)
    for which in (HOME, AWAY):
        chosen = keeper[:, which] >= 0
        ok = (cleaned.players_team[rows[chosen], keeper[chosen, which]] == which).mean() if chosen.any() else 1.0
        check("{}: keeper slot {} is a track of that team".format(tag, which), ok == 1.0, ok)
    check("{}: the sides swap at half-time".format(tag), cleaned.attack_sign[1] == -cleaned.attack_sign[2] != 0, cleaned.attack_sign.tolist())


def sign_of(direction):
    return {"left_to_right": 1, "right_to_left": -1}[direction]


# ---------------------------------------------------------------- synthetic EPTS

META_TEMPLATE = """<?xml version="1.0" encoding="utf-8"?>
<main><Metadata>
<GlobalConfig><FrameRate>{fps}</FrameRate></GlobalConfig>
<Sessions><Session id="s"><FieldSize><Width>100</Width><Height>50</Height></FieldSize></Session></Sessions>
<Teams><Team id="HH"><Name>Home side</Name></Team><Team id="AA"><Name>Away side</Name></Team></Teams>
<Players>
<Player id="10" teamId="AA"/><Player id="11" teamId="HH"/><Player id="12" teamId="AA"/><Player id="13" teamId="HH"/>
</Players>
<PlayerChannels>{channels}</PlayerChannels>
</Metadata><DataFormatSpecifications>{specs}</DataFormatSpecifications></main>
"""


def small_meta(path, layouts, ball=("x", "y"), player_channels=("x", "y"), fps="25"):
    channels = "".join('<PlayerChannel id="p{0}_{1}" playerId="{0}" channelId="{1}"/>'.format(p, c)
                       for p in (10, 11, 12, 13) for c in player_channels)
    specs = []
    for start, end, order in layouts:
        groups = "".join('<SplitRegister separator=",">{}</SplitRegister>'.format(
            "".join('<PlayerChannelRef playerChannelId="p{}_{}"/>'.format(p, c) for c in player_channels)) for p in order)
        ball_refs = "".join('<BallChannelRef channelId="{}"/>'.format(c) for c in ball)
        specs.append('<DataFormatSpecification startFrame="{}" endFrame="{}" separator=":">'
                     '<StringRegister name="frameCount"/><SplitRegister separator=";">{}</SplitRegister>'
                     '<SplitRegister separator=",">{}</SplitRegister></DataFormatSpecification>'.format(start, end, groups, ball_refs))
    with open(path, "w") as handle:
        handle.write(META_TEMPLATE.format(fps=fps, channels=channels, specs="".join(specs)))


def error_of(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except FormatError as error:
        return error
    return None


def check_synthetic_epts():
    print("--- synthetic EPTS", flush=True)
    with tempfile.TemporaryDirectory() as folder:
        meta, raw = os.path.join(folder, "m.xml"), os.path.join(folder, "r.txt")

        # Two layouts with a different slot order, and a ball field the metadata does not declare.
        small_meta(meta, [(1, 2, (10, 11, 12, 13)), (3, 4, (13, 12))])
        with open(raw, "w") as handle:
            handle.write("1:0.1,0.2;0.3,0.4;0.5,0.6;0.7,0.8:0.25,0.75,9.9\n")
            handle.write("2:0.1,0.2;NaN,NaN;0.5,0.6;0.7,0.8:NaN,NaN,NaN\r\n")
            handle.write("3:0.15,0.25;0.35,0.45:0.5,0.5,NaN\n")
            handle.write("4:0.15,0.25;0.35,0.45:0.5,0.5,NaN\n\n")
        match = load_epts(meta, raw, halves=((1, 0, 1), (2, 2, 3)), dead_blocks=((2, 2),), match_id="small")
        check("team comes from teamId, not from a slot range", match.slot_team.tolist() == [AWAY, HOME, AWAY, HOME], match.slot_team.tolist())
        check("frame is frame_id minus one", match.frame.tolist() == [0, 1, 2, 3])
        check("period and forced_dead follow what was passed", match.period.tolist() == [1, 1, 2, 2] and match.forced_dead.tolist() == [False, False, True, False])
        expected = (0.7 - 0.5) * 100, (0.5 - 0.8) * 50
        check("metres, centred, y grows towards the top", np.allclose(match.players_xy[0, 3], expected, atol=1e-3), match.players_xy[0, 3].tolist())
        layout2 = (0.15 - 0.5) * 100
        check("second layout routes fields by its own order",
              np.allclose(match.players_xy[2, 3, 0], layout2, atol=1e-3) and np.isnan(match.players_xy[2, 0, 0]) and np.isnan(match.players_xy[2, 1, 0]))
        check("undeclared trailing ball field is skipped, not read as a position", np.allclose(match.ball_xy[0], (-25, -12.5), atol=1e-3), match.ball_xy[0].tolist())
        check("CRLF and blank lines are tolerated, NaN is missing", np.isnan(match.ball_xy[1]).all() and np.isnan(match.players_xy[1, 1]).all())
        check("frame rate and field size come from the metadata", match.fps == 25.0 and match.pitch == (100.0, 50.0))

        # Declared channels x, y, s: speed lands in extras and is not confused with a position.
        small_meta(meta, [(1, 1, (10, 11, 12, 13))], ball=("x", "y", "s"), player_channels=("s", "x", "y"))
        with open(raw, "w") as handle:
            handle.write("1:9.0,0.1,0.2;8.0,0.3,0.4;7.0,0.5,0.6;6.0,0.7,0.8:0.25,0.75,5.0\n")
        match = load_epts(meta, raw, halves=((1, 0, 0),))
        check("channel order inside a group comes from the metadata",
              np.allclose(match.players_xy[0, 0], ((0.1 - 0.5) * 100, (0.5 - 0.2) * 50), atol=1e-3)
              and match.extras["slot_speed_raw"][0].tolist() == [9.0, 8.0, 7.0, 6.0] and match.extras["ball_speed_raw"][0] == 5.0)

        # Failures must not quote the file: the sentinel is a value that appears nowhere else.
        sentinel = "0.98765"
        small_meta(meta, [(1, 3, (10, 11, 12, 13))])
        good = "{}:0.1,0.2;0.3,0.4;0.5,0.6;0.7,0.8:0.25,0.75\n"
        cases = {
            "bad token": [good.format(1), "2:abc{},0.2;0.3,0.4;0.5,0.6;0.7,0.8:0.25,0.75\n".format(sentinel), good.format(3)],
            "empty field": [good.format(1), "2:{},;0.3,0.4;0.5,0.6;0.7,0.8:0.25,0.75\n".format(sentinel), good.format(3)],
            "missing group": [good.format(1), "2:{},0.2;0.3,0.4:0.25,0.75\n".format(sentinel), good.format(3)],
            "frame id gap": [good.format(1), "3:{},0.2;0.3,0.4;0.5,0.6;0.7,0.8:0.25,0.75\n".format(sentinel)],
            "beyond the last layout": [good.format(1), good.format(2), good.format(3), "4:{},0.2;0.3,0.4;0.5,0.6;0.7,0.8:0.25,0.75\n".format(sentinel)],
        }
        for name, lines in cases.items():
            with open(raw, "w") as handle:
                handle.writelines(lines)
            error = error_of(load_epts, meta, raw, halves=((1, 0, 2),))
            clean_message = error is not None and sentinel not in str(error) and (error.__context__ is None or error.__suppress_context__)
            check("raw failure names no value: {}".format(name), clean_message, error)

        with open(raw, "w") as handle:
            handle.writelines([good.format(1), good.format(2), good.format(3)])
        homography = os.path.join(folder, "h.csv")
        with open(homography, "w") as handle:
            handle.write("frame,h00,h01,h02,h10,h11,h12,h20,h21,h22,reprojection_error,valid\n")
            handle.write("0,abc{},1,1,1,1,1,1,1,1,0,1\n".format(sentinel))
        error = error_of(load_epts, meta, raw, homography, halves=((1, 0, 2),))
        check("homography failure names no value", error is not None and sentinel not in str(error) and error.__suppress_context__, error)

        small_meta(meta, [(1, 3, (10, 11, 12, 13))], fps="abc" + sentinel)
        error = error_of(load_epts, meta, raw, halves=((1, 0, 2),))
        check("metadata failure names no value", error is not None and sentinel not in str(error) and error.__suppress_context__, error)

        small_meta(meta, [(1, 3, (10, 11, 12, 13))])
        error = error_of(load_epts, meta, raw, halves=((1, 0, 2),), n_frames=4)
        check("a wrong frame count stops the load", error is not None, error)
        error = error_of(load_epts, meta, raw)
        check("no half boundaries anywhere stops the load", error is not None, error)
        error = error_of(load_competition, os.path.join(folder, "nowhere"))
        check("a missing bundle file stops the load", error is not None)

    bad = Match(source="x", match_id="x", fps=25.0, pitch=(100.0, 50.0), team_names=("a", "b"),
                frame=np.arange(3), period=np.array([1, 1, 1], dtype=np.int8),
                ball_xy=np.array([[1.0, np.nan], [0, 0], [0, 0]], dtype=np.float32),
                players_xy=np.zeros((3, 1, 2), dtype=np.float32),
                slot_team=np.zeros(1, dtype=np.int8), slot_id=np.array(["a"]))
    try:
        bad.validate()
        raised = False
    except SchemaError:
        raised = True
    check("schema refuses an x without its y", raised)


HARNESS = ("operate.py", "run")       # the one place pipeline/ may print: the operator's PASS/FAIL lines

# Each holds one violation that the policy must flag; PROSE holds only words, which must not count.
PLANTED = [
    ("environment", {"a.py": "import os\nhome = os.getenv('HOME')\n"}),
    ("upstream", {"a.py": "import games\n"}),
    ("upstream", {"a.py": "path = 'upstream/starter/baseline.py'\n"}),
    ("ground truth", {"a.py": "rows = open('twin/gt/possession_gt_full.csv')\n"}),
    ("prints", {"a.py": "def f(x):\n    print(x)\n"}),
    ("prints", {"a.py": "import sys\nsys.stdout.write('x')\n"}),
    ("prints", {"operate.py": "def run():\n    print('PASS')\n\ndef report(x):\n    print(x)\n"}),
]
PROSE = {
    "a.py": '"""The rule of upstream/starter/baseline.py; never print(), os.environ or /gt/."""\n'
            "# upstream/, games, possession_gt, getenv: a comment\n"
            "def f():\n    '''As upstream/evaluation reads it.'''\n    return 1\n",
    "operate.py": "def run():\n    print('PASS')\n",
}


def code_of(source):
    """The source as it runs: docstrings and comments dropped, so the policy reads code, not prose."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                    and isinstance(first.value.value, str):
                node.body = node.body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def stray_output(name, source):
    """Calls to print() outside the harness, and sys.stdout or sys.stderr outside operate.py."""
    stray = []
    for top in ast.parse(source).body:
        owner = getattr(top, "name", "<module>")
        for node in ast.walk(top):
            if isinstance(node, ast.Call) and (name, owner) != HARNESS and (
                    (isinstance(node.func, ast.Name) and node.func.id == "print")
                    or (isinstance(node.func, ast.Attribute) and node.func.attr == "print")):
                stray.append("{}:{} print".format(name, node.lineno))
            if isinstance(node, ast.Attribute) and node.attr in ("stdout", "stderr") and name != HARNESS[0] \
                    and isinstance(node.value, ast.Name) and node.value.id == "sys":
                stray.append("{}:{} sys.{}".format(name, node.lineno, node.attr))
    return stray


def policy(sources):
    """{rule: (holds, detail)} for {file name: source} of pipeline/."""
    code = "\n".join(code_of(text) for text in sources.values())
    stray = [place for name, text in sources.items() for place in stray_output(name, text)]
    return {
        "environment": ("os.environ" not in code and "getenv" not in code, ""),
        "upstream": ("upstream" not in code and "from games" not in code and "import games" not in code, ""),
        "ground truth": ("possession_gt" not in code and "/gt/" not in code, ""),
        "prints": (not stray, " ".join(stray)),
    }


def check_policy():
    print("--- policy", flush=True)
    sources = {}
    folder = os.path.join(ROOT, "pipeline")
    for name in sorted(os.listdir(folder)):
        if name.endswith(".py"):
            with open(os.path.join(folder, name)) as handle:
                sources[name] = handle.read()
    verdict = policy(sources)
    check("pipeline/ reads nothing from the environment", *verdict["environment"])
    check("pipeline/ does not import or call into upstream/", *verdict["upstream"])
    check("pipeline/ opens no ground-truth file", *verdict["ground truth"])
    check("pipeline/ prints only in operate.run(), the operator's lines", *verdict["prints"])
    check("policy: docstrings and comments do not count", all(ok for ok, _ in policy(PROSE).values()))
    missed = [rule for rule, planted in PLANTED if policy(planted)[rule][0]]
    check("policy: every planted violation is flagged", not missed,
          "{} planted, missed: {}".format(len(PLANTED), " ".join(missed) or "none"))


# ---------------------------------------------------------------- cleaning, unit level

def check_cleaning_units():
    print("--- cleaning units", flush=True)
    fps, pitch = 25.0, (105.0, 68.0)
    n = 1000
    frame = np.arange(n)
    period = np.ones(n, dtype=np.int8)
    path = np.stack((np.linspace(-40, 40, n), np.zeros(n)), axis=1).astype(np.float32)   # 2 m/s

    def gate(ball):
        return gate_ball(ball, frame, period, fps, 40.0, pitch)

    check("gate keeps a slow ball whole", gate(path).all())
    spiked = path.copy()
    spiked[500] += (20.0, 0.0)
    keep = gate(spiked)
    check("gate drops exactly one spike", not keep[500] and keep.sum() == n - 1)
    decoy = path.copy()
    decoy[400:450] = (-50.0, 30.0)
    keep = gate(decoy)
    check("gate drops a 50-frame decoy and keeps the real track", not keep[400:450].any() and keep.sum() >= n - 50 - 10, int(keep.sum()))
    first_is_false = path.copy()
    first_is_false[0] = (50.0, 30.0)
    keep = gate(first_is_false)
    check("gate survives a false first observation", not keep[0] and keep[1:].all())
    gaps = path.copy()
    gaps[300:600] = np.nan
    keep = gate(gaps)
    check("gate leaves missing frames missing", not keep[300:600].any() and keep[:300].all() and keep[600:].all())
    fast = np.stack((np.arange(n) * 1.2 - 600, np.zeros(n)), axis=1).astype(np.float32)   # 30 m/s
    check("gate keeps a real 30 m/s flight", gate(fast).all())
    two_halves = path.copy()
    two_halves[500:] += (30.0, 0.0)
    half_period = np.where(np.arange(n) < 500, 1, 2).astype(np.int8)
    check("gate does not compare across halves", gate_ball(two_halves, frame, half_period, fps, 40.0, pitch).all())

    players = np.full((n, 2, 2), np.nan, dtype=np.float32)
    players[:, 0] = path
    players[600:, 0, 0] += 10.0              # a jump
    players[100:200, 1] = path[100:200]      # a slot that appears and vanishes
    players[300:400, 1] = path[300:400]
    fragment, table, jumps = split_tracks(players, frame, period, fps, 25.0)
    check("split at a jump and at a gap", jumps == 1 and fragment[599, 0] != fragment[600, 0] and fragment[150, 1] != fragment[350, 1] and fragment[250, 1] == -1)
    check("fragment table agrees with the fragment map", all(
        (fragment[table["start"][i]:table["end"][i] + 1, table["slot"][i]] == i).all() for i in range(len(table["slot"]))))
    velocity = fragment_velocity(players[:, 0], fragment[:, 0], table["start"], table["end"], 5, fps)
    check("speed is exact on a straight line and blind to the jump", np.allclose(np.hypot(*velocity.T), 2.0, atol=0.01), round(float(np.nanmax(np.hypot(*velocity.T))), 3))
    lone = np.full((n, 1, 2), np.nan, dtype=np.float32)
    lone[5, 0] = (1.0, 1.0)
    fragment_lone, table_lone, _ = split_tracks(lone, frame, period, fps, 25.0)
    check("a one-frame fragment has no speed", np.isnan(fragment_velocity(lone[:, 0], fragment_lone[:, 0], table_lone["start"], table_lone["end"], 5, fps)).all())
    half_split, _, _ = split_tracks(players[:, :1], frame, half_period, fps, 1000.0)
    check("no fragment crosses the half", half_split[499, 0] != half_split[500, 0])

    # Team size: 13 tracks, the shortest fragments go, the keeper stays however short.
    team = np.zeros((1, 13), dtype=np.int8)
    ids = np.arange(13, dtype=np.int32)[None, :]
    lengths = np.arange(10, 23)
    keeper = np.array([[0, -1]], dtype=np.int16)
    drop = cap_team_size(team, ids, lengths, keeper, 11)
    check("team cap drops the shortest and spares the keeper", np.flatnonzero(drop[0]).tolist() == [1, 2], np.flatnonzero(drop[0]).tolist())

    depth = np.full((20, 3), np.nan)
    depth[:, 0] = 5.0
    depth[:, 1] = 7.0
    depth[10:12, 1] = 4.0                    # leads for 2 frames only
    depth[14:, 1] = 3.0                      # leads for good
    held = stable_deepest(depth, 3)
    check("stable deepest ignores a short lead and takes a long one", held[:16].tolist() == [0] * 16 and held[16:].tolist() == [1] * 4, held.tolist())
    depth2 = depth.copy()
    depth2[5:, 0] = np.nan
    check("stable deepest changes at once when the track ends", stable_deepest(depth2, 3)[5] == 1)

    # Direction from centroids, including a half with no players.
    players2 = np.zeros((4, 8, 2), dtype=np.float32)
    players2[:, :4, 0] = -10.0
    players2[:, 4:, 0] = 10.0
    slot_team = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8)
    players2[2:, :4, 0] = 10.0
    players2[2:, 4:, 0] = -10.0
    sign, _ = attack_direction(players2, slot_team, np.array([1, 1, 2, 2], dtype=np.int8), np.zeros(4, dtype=bool))
    check("direction reads the centroids in both halves", sign.tolist() == [0, 1, -1], sign.tolist())
    players2[2:] = np.nan
    sign, _ = attack_direction(players2, slot_team, np.array([1, 1, 2, 2], dtype=np.int8), np.zeros(4, dtype=bool))
    check("a half without players takes the opposite of the other", sign.tolist() == [0, 1, -1], sign.tolist())


def inject_keeper_reuse(match, seed, share=0.12, low_s=5.0, high_s=120.0):
    """Move stretches of each true keeper into free slots of the other team, a tracker fault the cleaning has to undo."""
    rng = np.random.default_rng(seed)
    players_xy = match.players_xy.copy()
    moved = np.zeros(players_xy.shape[:2], dtype=bool)
    for keeper in np.flatnonzero(match.extras["slot_is_keeper"]):
        others = np.flatnonzero(match.slot_team != match.slot_team[keeper])
        for half in (1, 2):
            rows = np.flatnonzero(match.period == half)
            budget, used = int(share * len(rows)), 0
            for _ in range(300):
                if used >= budget:
                    break
                length = int(rng.uniform(low_s, high_s) * match.fps)
                start = int(rng.integers(rows[0], rows[-1] - length))
                stretch = np.arange(start, start + length)
                present = np.isfinite(players_xy[stretch, keeper, 0])
                if present.mean() < 0.5:
                    continue
                free = [s for s in others if not np.isfinite(players_xy[stretch, s, 0]).any()]
                if not free:
                    continue
                slot = free[int(rng.integers(len(free)))]
                players_xy[stretch, slot] = players_xy[stretch, keeper]
                moved[stretch, slot] = present
                players_xy[stretch, keeper] = np.nan
                used += length
    return dataclasses.replace(match, players_xy=players_xy), moved


def check_keeper_reuse(tag, match, floor):
    damaged, moved = inject_keeper_reuse(match, seed=20261106)
    repaired = clean(damaged)
    labels_before = damaged.slot_team[np.nonzero(moved)[1]]
    labels_after = repaired.players_team[moved]
    right = (labels_after == 1 - labels_before)
    recall = float(right.mean()) if len(right) else float("nan")
    untouched = ~moved & (repaired.players_team >= 0)
    wrong = float(np.mean(repaired.players_team[untouched] != np.broadcast_to(match.slot_team, moved.shape)[untouched]))
    check("{}: reused keeper time goes back to the right team".format(tag), recall >= floor,
          "{:.3f} of {} frames, floor {}".format(recall, int(moved.sum()), floor))
    check("{}: nobody else changes team".format(tag), wrong <= 0.002, "{:.5f} of other track-frames".format(wrong))
    return recall


# ---------------------------------------------------------------- real files

def check_metrica(folder, use_kloppy):
    print("--- Metrica game 3", flush=True)
    meta = os.path.join(folder, "Sample_Game_3_metadata.xml")
    raw = os.path.join(folder, "Sample_Game_3_tracking.txt")
    started = time.time()
    match = load_epts(meta, raw, match_id="metrica3")
    note("loaded in {:.1f} s".format(time.time() - started))
    check("metrica: 143,761 frames and 35 players", match.n_frames == 143761 and match.n_slots == 35, (match.n_frames, match.n_slots))
    check("metrica: halves from the metadata", np.sum(match.period == 1) == 69661 and np.sum(match.period == 2) == 74100)
    check("metrica: pitch and rate from the metadata", match.pitch == (105.0, 68.0) and match.fps == 25.0)
    tracked = np.isfinite(match.players_xy[..., 0]).sum(axis=1)
    check("metrica: 22 people on every frame", tracked.min() == 22 and tracked.max() == 22, (int(tracked.min()), int(tracked.max())))
    check("metrica: first team in the file is home", match.team_names == ("Team A", "Team B"))

    if use_kloppy:
        from kloppy import metrica
        started = time.time()
        dataset = metrica.load_tracking_epts(meta_data=meta, raw_data=raw)
        frames = dataset.records
        note("kloppy loaded in {:.1f} s".format(time.time() - started))
        check("metrica vs kloppy: same frames", len(frames) == match.n_frames
              and [f.frame_id for f in frames[:1]] == [1] and frames[-1].frame_id == match.n_frames, len(frames))
        check("metrica vs kloppy: same home team", dataset.metadata.teams[0].name == match.team_names[0])
        period = np.fromiter((f.period.id for f in frames), dtype=np.int8, count=len(frames))
        check("metrica vs kloppy: same period on every frame", np.array_equal(period, match.period))
        x_norm = match.players_xy[..., 0] / match.pitch[0] + 0.5
        y_norm = 0.5 - match.players_xy[..., 1] / match.pitch[1]
        slot_of = {slot_id: i for i, slot_id in enumerate(match.slot_id.tolist())}
        worst = 0.0
        missing_disagree = 0
        team_ok = True
        for i, frame in enumerate(frames):
            seen = np.zeros(match.n_slots, dtype=bool)
            for player, data in frame.players_data.items():
                # kloppy lists every player on every frame, with no coordinates when out of the layout.
                if data.coordinates is None or np.isnan(data.coordinates.x):
                    continue
                slot = slot_of[player.player_id]
                seen[slot] = True
                worst = max(worst, abs(x_norm[i, slot] - data.coordinates.x), abs(y_norm[i, slot] - data.coordinates.y))
                team_ok = team_ok and (match.slot_team[slot] == (HOME if player.team == dataset.metadata.teams[0] else AWAY))
            missing_disagree += int(np.sum(seen != np.isfinite(x_norm[i])))
        check("metrica vs kloppy: every player position, every frame", worst < 1e-4 and missing_disagree == 0,
              "worst {:.2e} of the pitch, presence disagreements {}".format(worst, missing_disagree))
        check("metrica vs kloppy: every slot on the same team", team_ok)
        ball_x = np.array([f.ball_coordinates.x for f in frames], dtype=np.float64)
        ball_y = np.array([f.ball_coordinates.y for f in frames], dtype=np.float64)
        ours_x = match.ball_xy[:, 0] / match.pitch[0] + 0.5
        ours_y = 0.5 - match.ball_xy[:, 1] / match.pitch[1]
        same_missing = np.array_equal(np.isnan(ball_x), np.isnan(ours_x))
        both = np.isfinite(ball_x) & np.isfinite(ours_x)
        gap = max(float(np.max(np.abs(ball_x[both] - ours_x[both]))), float(np.max(np.abs(ball_y[both] - ours_y[both]))))
        check("metrica vs kloppy: every ball position, every frame", same_missing and gap < 1e-4, "worst {:.2e}".format(gap))
        del frames, dataset

    cleaned = clean(match)
    note("clean report {}".format(cleaned.report))
    clean_invariants("metrica", match, cleaned)
    home_direction = sign_of(match.extras["attack_direction_first_half"][0])
    check("metrica: computed direction equals the metadata's", cleaned.attack_sign[1] == home_direction, (cleaned.attack_sign.tolist(), match.extras["attack_direction_first_half"]))
    check("metrica: nobody relabelled on clean data", cleaned.report["relabelled_fragments"] == 0)
    check_keeper_pick("metrica", match, cleaned, 0.95)
    check_keeper_reuse("metrica", match, KEEPER_RECALL_FLOOR["epts"])


def check_keeper_pick(tag, match, cleaned, floor):
    """Share of frames with the true keeper tracked where the stable deepest track is that keeper."""
    for which in (HOME, AWAY):
        slots = np.flatnonzero(match.extras["slot_is_keeper"] & (match.slot_team == which))
        on_pitch = np.isfinite(match.players_xy[:, slots, 0]).any(axis=1) & (match.period > 0)
        share = float(np.mean(np.isin(cleaned.keeper_slot[on_pitch, which], slots)))
        check("{}: stable deepest is the true keeper, team {}".format(tag, which), share >= floor, "{:.3f}, floor {}".format(share, floor))


def check_twin(twin):
    print("--- twin", flush=True)
    tactical = os.path.join(twin, "data", "tactical")
    started = time.time()
    match = load_competition(tactical)
    note("loaded with homography in {:.1f} s".format(time.time() - started))
    check("twin: 199,319 frames, 46 slots", match.n_frames == 199319 and match.n_slots == 46, (match.n_frames, match.n_slots))
    check("twin: frame is the submission row", np.array_equal(match.frame, np.arange(199319)))
    for half, first, last in COMPETITION_HALVES:
        rows = np.flatnonzero(match.period == half)
        check("twin: half {} is the published block".format(half), rows[0] == first and rows[-1] == last and len(rows) == last - first + 1)
    check("twin: 158,857 frames inside the halves", int(np.sum(match.period > 0)) == 158857)
    (dead_first, dead_last), = COMPETITION_DEAD_BLOCKS
    check("twin: 2,934 forced dead frames, all before the second kickoff",
          int(match.forced_dead.sum()) == 2934 and match.forced_dead[dead_first] and match.forced_dead[dead_last] and not match.forced_dead[dead_last + 1])
    check("twin: published pitch overrides the metadata", match.pitch == COMPETITION_PITCH)
    check("twin: team from teamId", np.bincount(match.slot_team).tolist() == [23, 23], np.bincount(match.slot_team).tolist())
    check("twin: no labels are read", match.labels is None)
    visible = np.isfinite(match.ball_xy[:, 0])
    check("twin: ball visible on 56.9% of all frames", abs(visible.mean() - 0.569) < 0.002, round(float(visible.mean()), 4))
    check("twin: ball visible on 59.3% of graded frames", abs(visible[match.period > 0].mean() - 0.593) < 0.002, round(float(visible[match.period > 0].mean()), 4))
    tracked = np.isfinite(match.players_xy[match.period > 0, :, 0]).sum(axis=1).mean()
    check("twin: 20.5 tracks per graded frame", abs(tracked - 20.5) < 0.05, round(float(tracked), 2))

    # Direct read of a few raw lines, by hand, against the arrays.
    raw_path = os.path.join(tactical, "BAR-ATH_tactical_atd.txt")
    wanted = {12000: None, 50000: None, 150000: None}
    with open(raw_path) as handle:
        for index, line in enumerate(handle):
            if index in wanted:
                wanted[index] = line
            if index > 150000:
                break
    worst = 0.0
    for index, line in wanted.items():
        _, groups, ball = line.rstrip("\n").split(":")
        for slot, group in enumerate(groups.split(";")):
            x, y, _ = group.split(",")
            if x == "NaN":
                worst = max(worst, 0.0 if np.isnan(match.players_xy[index, slot, 0]) else 1.0)
            else:
                worst = max(worst, abs(match.players_xy[index, slot, 0] - (float(x) - 0.5) * 105.3),
                            abs(match.players_xy[index, slot, 1] - (0.5 - float(y)) * 68.0))
    check("twin: arrays match the raw text on sampled lines", worst < 1e-3, "{:.1e} m".format(worst))

    # DATA.md prints the homography row of frame 11185. The ground polygon must map back to the image corners.
    printed = np.array([[1822.663, -511.2005, 44.48168], [1.698323, 169.2971, 255.7303], [-0.01907188, -0.5269586, 1]])
    check("twin: homography row at the first kickoff equals DATA.md", np.allclose(match.extras["homography"][11185], printed, atol=1e-3))
    polygon = match.view_polygon[11185].astype(np.float64)
    homogeneous = np.column_stack((polygon[:, 0] / match.pitch[0] + 0.5, 0.5 - polygon[:, 1] / match.pitch[1], np.ones(4)))
    pixels = (printed @ homogeneous.T).T
    pixels = pixels[:, :2] / pixels[:, 2:]
    corners = np.array([[0, 0], [0, 1080], [1920, 1080], [1920, 0]], dtype=np.float64)
    check("twin: ground polygon maps back to the image corners", np.allclose(pixels, corners, atol=2.0), np.round(pixels).tolist())
    check("twin: polygon top is far from the camera, bottom is near", polygon[0, 1] > polygon[1, 1] and polygon[3, 1] > polygon[2, 1])
    check("twin: polygon exists on every frame", not np.isnan(match.view_polygon).any())
    inside = point_in_quads(match.ball_xy[visible], match.view_polygon[visible].astype(np.float64))
    check("twin: the camera sees the ball it follows", inside.mean() > 0.95, round(float(inside.mean()), 4))

    cleaned = clean(match)
    note("clean report {}".format(cleaned.report))
    clean_invariants("twin", match, cleaned)
    home_direction = sign_of(match.extras["attack_direction_first_half"][0])
    check("twin: computed direction equals the metadata's", cleaned.attack_sign[1] == home_direction, cleaned.attack_sign.tolist())
    removed = cleaned.report["ball_removed"] / cleaned.report["ball_observed"]
    check("twin: the gate removes little", removed < 0.02, round(removed, 4))
    return match


def point_in_quads(points, quads):
    """Each point against its own convex quad, given in order around the edge."""
    points = points.astype(np.float64)
    following = np.roll(quads, -1, axis=1)
    cross = ((following[..., 0] - quads[..., 0]) * (points[:, None, 1] - quads[..., 1])
             - (following[..., 1] - quads[..., 1]) * (points[:, None, 0] - quads[..., 0]))
    return (cross >= 0).all(axis=1) | (cross <= 0).all(axis=1)


def check_twin_against_source(twin_match, idsse_folder):
    """The twin was built from IDSSE J03WMX, so its metres must land on the source's metres."""
    source = load_idsse(os.path.join(idsse_folder, "J03WMX.npz"))
    live_rows = np.flatnonzero((source.period == 1) & (source.labels != "D"))
    first_live = int(live_rows[0])
    # Half 1 of the twin starts on the first live source frame, one row per frame from there.
    offset = 11185 - first_live
    rows = live_rows[(live_rows + offset) <= 90651]
    ours = twin_match.ball_xy[rows + offset]
    theirs = source.ball_xy[rows]
    seen = np.isfinite(ours[:, 0])
    gap = np.hypot(*(ours[seen] - theirs[seen]).T)
    check("twin vs IDSSE source: same metres, same orientation", np.median(gap) < 2.0, "median {:.2f} m on {} live frames".format(np.median(gap), int(seen.sum())))


def check_idsse(folder):
    print("--- IDSSE", flush=True)
    names = sorted(n for n in os.listdir(folder) if n.endswith(".npz"))
    check("idsse: 7 archives", len(names) == 7, len(names))
    for index, name in enumerate(names):
        tag = "idsse " + name[:-4]
        started = time.time()
        match = load_idsse(os.path.join(folder, name))
        cleaned = clean(match)
        note("{}: load and clean {:.1f} s, report {}".format(tag, time.time() - started, cleaned.report))
        status, possession = match.extras["ball_status"], match.extras["ball_possession"]
        labels = np.where(status == 0, "D", np.where(possession == 1, "H", "A"))
        check("{}: labels equal the raw flags".format(tag), np.array_equal(labels, match.labels))
        check("{}: home slots come first as written".format(tag), set(match.slot_team.tolist()) == {HOME, AWAY})
        check("{}: height present".format(tag), np.isfinite(match.ball_z).all())
        clean_invariants(tag, match, cleaned)

        # Truth for the direction: where the starting keepers stand.
        for half in (1, 2):
            rows = match.period == half
            home_keeper = np.nanmedian(match.players_xy[rows][:, match.extras["slot_is_keeper"] & (match.slot_team == HOME), 0])
            check("{}: direction agrees with the keeper, half {}".format(tag, half), np.sign(home_keeper) == -cleaned.attack_sign[half], round(float(home_keeper), 1))

        raw = match.extras["slot_speed_raw_kmh"] / 3.6
        ours = np.hypot(cleaned.players_vel[..., 0], cleaned.players_vel[..., 1])
        both = np.isfinite(raw) & np.isfinite(ours)
        check("{}: recomputed player speed matches the raw km/h field".format(tag), np.mean(np.abs(raw[both] - ours[both])) < 0.05,
              "mean abs diff {:.3f} m/s".format(float(np.mean(np.abs(raw[both] - ours[both])))))
        removed = cleaned.report["ball_removed"] / cleaned.report["ball_observed"]
        check("{}: the gate removes under 1% of the ball".format(tag), removed < 0.01, round(removed, 4))
        check("{}: no false relabelling on clean data".format(tag), cleaned.report["relabelled_fragments"] == 0, cleaned.report["relabelled_fragments"])
        check_keeper_pick(tag, match, cleaned, 0.94)
        if index < 2:
            check_keeper_reuse(tag, match, KEEPER_RECALL_FLOOR["idsse"])


def check_skillcorner(folder):
    print("--- SkillCorner", flush=True)
    names = sorted(n for n in os.listdir(folder) if os.path.isdir(os.path.join(folder, n)))
    check("skillcorner: 20 matches", len(names) == 20, len(names))
    direction_ok = 0
    relabelled = relabelled_seconds = 0
    for index, name in enumerate(names):
        tag = "skillcorner " + name
        started = time.time()
        match = load_skillcorner(os.path.join(folder, name))
        cleaned = clean(match)
        relabelled += cleaned.report["relabelled_fragments"]
        relabelled_seconds += cleaned.report["relabelled_frames"] / match.fps
        note("{}: load and clean {:.1f} s, report {}".format(tag, time.time() - started, cleaned.report))
        check("{}: 10 fps, home first".format(tag), match.fps == 10.0 and match.slot_team[0] == HOME and match.slot_team[-1] == AWAY)
        check("{}: labels are H, A or X, never D".format(tag), set(np.unique(match.labels).tolist()) <= {"H", "A", "X"})
        holder = match.extras["possessor_slot"]
        held = holder >= 0
        same = match.slot_team[holder[held]] == np.where(match.labels[held] == "H", HOME, AWAY)
        check("{}: the holder belongs to the team in the label".format(tag), same.mean() == 1.0, round(float(same.mean()), 4))
        detected_only = np.isnan(match.ball_xy[:, 0]).mean()
        modelled_only = np.isnan(match.extras["ball_xy_modelled"][:, 0]).mean()
        check("{}: unseen ball is NaN, its extrapolation stays in extras".format(tag), detected_only > modelled_only, (round(float(detected_only), 3), round(float(modelled_only), 3)))
        polygon = match.view_polygon
        known = np.isfinite(polygon[:, 0, 1]) & np.isfinite(polygon[:, 1, 1])
        far = np.mean(polygon[known, 0, 1] > polygon[known, 1, 1])
        check("{}: y grows towards the top of the picture".format(tag), far > 0.999, round(float(far), 4))
        clean_invariants(tag, match, cleaned)
        wanted = [sign_of(d) for d in match.extras["home_attack_direction"]]
        direction_ok += wanted == cleaned.attack_sign[1:].tolist()
        if index == 0:
            check_keeper_reuse(tag, match, KEEPER_RECALL_FLOOR["skillcorner"])
    check("skillcorner: computed direction equals home_team_side on every match", direction_ok == len(names), "{} of {}".format(direction_ok, len(names)))
    check("skillcorner: false relabelling stays rare with keepers out of view", relabelled_seconds < 120, "{} fragments, {:.0f} s in {} matches".format(relabelled, relabelled_seconds, len(names)))


def check_clean_contract(match):
    """clean() leaves its input alone and gives the same bytes twice."""
    before = hashlib.sha256(match.players_xy.tobytes() + match.ball_xy.tobytes()).hexdigest()
    first, second = clean(match), clean(match)
    after = hashlib.sha256(match.players_xy.tobytes() + match.ball_xy.tobytes()).hexdigest()
    check("clean leaves the input untouched", before == after)
    same = all(np.array_equal(getattr(first, name), getattr(second, name), equal_nan=True)
               for name in ("ball_xy", "players_xy", "ball_vel", "players_vel", "fragment", "players_team", "keeper_slot", "attack_sign"))
    check("clean gives the same arrays twice", same)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--twin", default=os.path.join(ROOT, "twin", "J03WMX"))
    parser.add_argument("--metrica", default=os.path.join(ROOT, "external", "metrica", "data", "Sample_Game_3"))
    parser.add_argument("--idsse", default=os.path.join(ROOT, "external", "processed", "idsse"))
    parser.add_argument("--skillcorner", default=os.path.join(ROOT, "external", "skillcorner", "data", "matches"))
    parser.add_argument("--json", help="write the results here")
    parser.add_argument("--no-kloppy", action="store_true", help="skip the comparison with kloppy")
    args = parser.parse_args()

    if not os.path.isfile(os.path.join(args.twin, "TWIN_NOT_REAL_DATA.txt")):
        sys.exit("refusing to run: --twin has no TWIN_NOT_REAL_DATA.txt, and this script prints measurements")

    started = time.time()
    check_policy()
    check_synthetic_epts()
    check_cleaning_units()
    twin_match = check_twin(args.twin)
    check_twin_against_source(twin_match, args.idsse)
    check_clean_contract(twin_match)
    del twin_match
    check_metrica(args.metrica, not args.no_kloppy)
    check_idsse(args.idsse)
    check_skillcorner(args.skillcorner)

    failed = [r for r in results if not r["ok"]]
    print("\n{} checks, {} failed, {:.0f} s".format(len(results), len(failed), time.time() - started), flush=True)
    for item in failed:
        print("FAIL " + item["name"] + "  " + item["detail"], flush=True)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(results, handle, indent=1)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
