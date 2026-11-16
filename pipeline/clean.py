"""Cleaning: the same code on every source.

    ball          the largest set of observations with no implied speed above the gate
    tracks        split a slot at every gap and every jump, so nothing is computed across a person change
    direction     which way the home side attacks in each half, from the team centroids
    keepers       a long fragment that behaves like the keeper of the other team goes to that team;
                  each team then gets one stable deepest track per frame
    team size     at most 11 tracks per team on a frame, longest fragments first
    speeds        recomputed from positions, inside a fragment only; every raw speed field is ignored

Outside the halves nothing survives. Published dead frames stay in `forced_dead`, and they are left
out of the direction and keeper estimates, because players warm up during them.

All thresholds are defaults set on open data (IDSSE, SkillCorner, Metrica game 3), never on the
competition's match. tests/check_adapters.py measures them.

`report` on the result is a summary of the input. Scripts run on the competition bundle must not
print it until the organisers have cleared that in writing.
"""
from dataclasses import dataclass, field, fields
from typing import Optional

import numpy as np

from .schema import AWAY, HOME, Match


@dataclass(frozen=True)
class CleanConfig:
    ball_max_speed: float = 40.0            # m/s
    player_jump_speed: float = 25.0         # m/s, 1 m per frame at 25 fps. People stay under 13 m/s
    player_velocity_window_s: float = 0.2   # half-width of the central difference
    ball_velocity_window_s: float = 0.08
    max_per_team: int = 11
    keeper_min_s: float = 5.0               # shorter fragments are never relabelled
    keeper_deepest_share: float = 0.7       # of its frames, the fragment is the deepest of all tracks
    keeper_max_depth_m: float = 40.0        # median distance from that goal line
    keeper_max_abs_y_m: float = 12.0        # median distance from the line through both goals
    keeper_min_gap_m: float = 2.0           # median lead over the nearest track of the team at that end
    keeper_hold_s: float = 1.0              # a new deepest track must lead this long to take over


@dataclass
class CleanMatch(Match):
    ball_removed: Optional[np.ndarray] = None   # (F,) bool, observations the gate dropped
    ball_vel: Optional[np.ndarray] = None       # (F, 2) float32, m/s
    players_vel: Optional[np.ndarray] = None    # (F, S, 2) float32, m/s
    fragment: Optional[np.ndarray] = None       # (F, S) int32, -1 where the slot is empty
    players_team: Optional[np.ndarray] = None   # (F, S) int8, -1 where the slot is empty
    keeper_slot: Optional[np.ndarray] = None    # (F, 2) int16, per team, -1 where it has no track
    attack_sign: Optional[np.ndarray] = None    # (3,) int8 by period, +1 when home attacks towards +x
    fragments: dict = field(default_factory=dict)   # slot, start, end, team, relabelled: one row each
    report: dict = field(default_factory=dict)


def gate_ball(ball_xy, frame, period, fps, max_speed, pitch):
    """True on the observations to keep.

    Dynamic programme over each half: the longest chain of observations in which every consecutive
    pair implies at most max_speed. One spike costs one point and a decoy loses to the longer real
    track, where dropping "whatever follows a jump" would lose the real track for seconds.
    """
    keep = np.zeros(len(ball_xy), dtype=bool)
    # Two observations further apart in time than this are compatible wherever they are.
    horizon = int(np.ceil(np.hypot(*pitch) * 1.5 / max_speed * fps))
    limit = max_speed / fps
    for half in (1, 2):
        rows = np.flatnonzero((period == half) & np.isfinite(ball_xy[:, 0]))
        if not len(rows):
            continue
        x, y = ball_xy[rows, 0].astype(np.float64), ball_xy[rows, 1].astype(np.float64)
        t = frame[rows].astype(np.int64)
        n = len(rows)
        best_len = np.ones(n, dtype=np.int64)
        parent = np.full(n, -1, dtype=np.int64)
        overall = 0             # index of the longest chain so far
        old_end = 0             # rows before this index are past the horizon of the current one
        old_best = -1           # index of the longest chain among them
        for i in range(1, n):
            while t[i] - t[old_end] > horizon:
                if old_best < 0 or best_len[old_end] > best_len[old_best]:
                    old_best = old_end
                old_end += 1
            # Fast path, and exact: no chain can beat extending the longest one so far.
            if (overall == i - 1
                    and (x[i] - x[i - 1]) ** 2 + (y[i] - y[i - 1]) ** 2 <= (limit * (t[i] - t[i - 1])) ** 2):
                best_len[i], parent[i], overall = best_len[i - 1] + 1, i - 1, i
                continue
            allowed = limit * (t[i] - t[old_end:i])
            near = (x[old_end:i] - x[i]) ** 2 + (y[old_end:i] - y[i]) ** 2 <= allowed * allowed
            candidate, score = -1, 0
            if near.any():
                lengths = np.where(near, best_len[old_end:i], 0)
                at = int(np.argmax(lengths))
                candidate, score = old_end + at, int(lengths[at])
            if old_best >= 0 and best_len[old_best] > score:
                candidate, score = old_best, int(best_len[old_best])
            if candidate >= 0:
                best_len[i], parent[i] = score + 1, candidate
            if best_len[i] > best_len[overall]:
                overall = i
        node = overall
        while node >= 0:
            keep[rows[node]] = True
            node = parent[node]
    return keep


def split_tracks(players_xy, frame, period, fps, jump_speed):
    """(fragment (F, S) int32, table, number of jumps): a fragment ends at a half boundary, a gap or a jump."""
    n, slots = players_xy.shape[:2]
    fragment = np.full((n, slots), -1, dtype=np.int32)
    table_slot, table_start, table_end = [], [], []
    next_id = 0
    jumps = 0
    limit = jump_speed / fps
    for slot in range(slots):
        rows = np.flatnonzero(np.isfinite(players_xy[:, slot, 0]) & (period > 0))
        if not len(rows):
            continue
        xy = players_xy[rows, slot].astype(np.float64)
        step = np.hypot(xy[1:, 0] - xy[:-1, 0], xy[1:, 1] - xy[:-1, 1])
        adjacent = np.diff(rows) == 1
        same_half = period[rows[1:]] == period[rows[:-1]]
        jump = adjacent & same_half & (step > limit * np.diff(frame[rows]))
        jumps += int(jump.sum())
        breaks = np.flatnonzero(jump | ~adjacent | ~same_half) + 1
        starts = np.concatenate(([0], breaks))
        ends = np.concatenate((breaks, [len(rows)])) - 1
        ids = np.arange(next_id, next_id + len(starts), dtype=np.int32)
        fragment[rows, slot] = np.repeat(ids, ends - starts + 1)
        table_slot.append(np.full(len(starts), slot, dtype=np.int32))
        table_start.append(rows[starts])
        table_end.append(rows[ends])
        next_id += len(starts)
    empty = np.zeros(0, dtype=np.int64)
    table = {
        "slot": np.concatenate(table_slot) if table_slot else empty,
        "start": np.concatenate(table_start) if table_start else empty,
        "end": np.concatenate(table_end) if table_end else empty,
    }
    return fragment, table, jumps


def attack_direction(players_xy, slot_team, period, forced_dead):
    """(sign (3,) by period, margin (3,) in metres). +1 when home attacks towards +x.

    The defending side stands nearer its own goal than the attacking side, whoever has the ball, so
    the median gap between the team centroids points the right way. The sides swap at half-time:
    if the two halves disagree about that, the half with the larger margin decides.
    """
    sign = np.zeros(3, dtype=np.int8)
    margin = np.zeros(3, dtype=np.float64)
    home, away = slot_team == HOME, slot_team == AWAY
    for half in (1, 2):
        x = players_xy[(period == half) & ~forced_dead][:, :, 0]
        enough = (np.isfinite(x[:, home]).sum(axis=1) >= 3) & (np.isfinite(x[:, away]).sum(axis=1) >= 3)
        if enough.any():
            margin[half] = np.median(np.nanmean(x[enough][:, away], axis=1) - np.nanmean(x[enough][:, home], axis=1))
    sign[1:] = np.sign(margin[1:]).astype(np.int8)
    if sign[1] != -sign[2]:
        strong, weak = (1, 2) if abs(margin[1]) >= abs(margin[2]) else (2, 1)
        sign[weak] = -sign[strong]
    return sign, margin


def _deepest(depth):
    """(slot of the smallest depth per frame, False where the frame has no track). NaN is no track."""
    filled = np.where(np.isnan(depth), np.inf, depth)
    slot = np.argmin(filled, axis=1)
    return slot, np.isfinite(filled[np.arange(len(filled)), slot])


def relabel_keepers(players_xy, table, slot_team, period, forced_dead, attack_sign, pitch, fps, cfg):
    """Team of each fragment. A keeper filed under the other team's slot goes to the team at that end.

    Only labels that disagree with the end are tested. Over its frames the fragment must be long,
    the deepest of all tracks at the goal it stands at, near the line through both goals, and clear
    of every track labelled for the team that defends that goal. A real attacker fails on one of
    these: at a corner flag on the line through the goals, in a goalmouth scramble on the lead.
    """
    length = pitch[0]
    team = slot_team[table["slot"]].astype(np.int8)
    min_frames = int(round(cfg.keeper_min_s * fps))
    for half in (1, 2):
        rows = np.flatnonzero(period == half)
        if not len(rows) or attack_sign[half] == 0:
            continue
        chosen = np.flatnonzero((table["start"] >= rows[0]) & (table["start"] <= rows[-1])
                                & (table["end"] - table["start"] + 1 >= min_frames))
        if not len(chosen):
            continue
        x = players_xy[rows, :, 0].astype(np.float64) * attack_sign[half]
        abs_y = np.abs(players_xy[rows, :, 1].astype(np.float64))
        # Metres from the goal line that each team defends.
        depth = {HOME: x + length / 2, AWAY: length / 2 - x}
        deepest = {which: _deepest(depth[which])[0] for which in depth}
        for index in chosen:
            slot = table["slot"][index]
            first, last = table["start"][index] - rows[0], table["end"][index] - rows[0]
            usable = np.flatnonzero(~forced_dead[rows[first:last + 1]]) + first
            if len(usable) < min_frames:
                continue
            other = AWAY if slot_team[slot] == HOME else HOME
            if np.mean(deepest[other][usable] == slot) < cfg.keeper_deepest_share:
                continue
            own_depth = depth[other][usable, slot]
            if np.median(own_depth) > cfg.keeper_max_depth_m or np.median(abs_y[usable, slot]) > cfg.keeper_max_abs_y_m:
                continue
            defenders = depth[other][usable][:, slot_team == other]
            has_defender = np.isfinite(defenders).any(axis=1)
            if has_defender.any():
                nearest = np.min(np.where(np.isfinite(defenders[has_defender]), defenders[has_defender], np.inf), axis=1)
                if np.median(nearest - own_depth[has_defender]) < cfg.keeper_min_gap_m:
                    continue
            team[index] = other
    return team


def stable_deepest(depth, hold_frames):
    """Slot of the deepest track per frame, with hysteresis: a new one must lead for hold_frames."""
    n = len(depth)
    out = np.full(n, -1, dtype=np.int16)
    best, present = _deepest(depth)
    current = challenger = -1
    count = 0
    for i in range(n):
        if not present[i]:
            current = challenger = -1
            count = 0
            continue
        leader = best[i]
        if current < 0 or np.isnan(depth[i, current]):
            current, challenger, count = leader, -1, 0
        elif leader == current:
            challenger, count = -1, 0
        else:
            count = count + 1 if leader == challenger else 1
            challenger = leader
            if count >= hold_frames:
                current, challenger, count = leader, -1, 0
        out[i] = current
    return out


def keeper_per_frame(players_xy, players_team, period, attack_sign, pitch, fps, cfg):
    """(F, 2) slot of each team's stable deepest track, on team labels after the relabelling."""
    n = len(players_xy)
    keeper = np.full((n, 2), -1, dtype=np.int16)
    length = pitch[0]
    hold = max(1, int(round(cfg.keeper_hold_s * fps)))
    for half in (1, 2):
        rows = np.flatnonzero(period == half)
        if not len(rows) or attack_sign[half] == 0:
            continue
        x = players_xy[rows, :, 0].astype(np.float64) * attack_sign[half]
        for which, depth in ((HOME, x + length / 2), (AWAY, length / 2 - x)):
            keeper[rows, which] = stable_deepest(np.where(players_team[rows] == which, depth, np.nan), hold)
    return keeper


def cap_team_size(players_team, fragment, lengths, keeper, max_per_team):
    """True on (frame, slot) pairs to blank: beyond max_per_team, shortest fragments first, keeper spared."""
    drop = np.zeros(players_team.shape, dtype=bool)
    for which in (HOME, AWAY):
        mine = players_team == which
        for i in np.flatnonzero(mine.sum(axis=1) > max_per_team):
            slots = np.flatnonzero(mine[i])
            size = lengths[fragment[i, slots]].astype(np.float64)
            size[slots == keeper[i, which]] = np.inf
            # Stable, so a tie falls the same way on every run.
            order = np.argsort(-size, kind="stable")
            drop[i, slots[order[max_per_team:]]] = True
    return drop


def fragment_velocity(xy, fragment, starts, ends, window, fps):
    """Central difference over +-window frames, clipped to the fragment. (F, 2), NaN if it has one frame."""
    velocity = np.full(xy.shape, np.nan, dtype=np.float32)
    rows = np.flatnonzero(fragment >= 0)
    if not len(rows):
        return velocity
    low = np.maximum(rows - window, starts[fragment[rows]])
    high = np.minimum(rows + window, ends[fragment[rows]])
    span = (high - low) / fps
    moving = span > 0
    velocity[rows[moving]] = (xy[high[moving]] - xy[low[moving]]) / span[moving, None]
    return velocity


def clean(match, cfg=CleanConfig()):
    """Cleaned copy of a Match. The input is left untouched."""
    n, slots = match.n_frames, match.n_slots
    fps, pitch, period, frame = match.fps, match.pitch, match.period, match.frame
    inside = period > 0

    observed = np.isfinite(match.ball_xy[:, 0]) & inside
    keep = gate_ball(match.ball_xy, frame, period, fps, cfg.ball_max_speed, pitch)
    ball_xy = match.ball_xy.copy()
    ball_xy[~keep] = np.nan
    ball_z = None
    if match.ball_z is not None:
        ball_z = match.ball_z.copy()
        ball_z[~keep] = np.nan
    ball_removed = observed & ~keep

    players_xy = match.players_xy.copy()
    players_xy[~inside] = np.nan
    fragment, table, jumps = split_tracks(players_xy, frame, period, fps, cfg.player_jump_speed)
    lengths = table["end"] - table["start"] + 1

    attack_sign, margin = attack_direction(players_xy, match.slot_team, period, match.forced_dead)
    team_of = relabel_keepers(players_xy, table, match.slot_team, period, match.forced_dead,
                              attack_sign, pitch, fps, cfg)
    relabelled = team_of != match.slot_team[table["slot"]]
    players_team = np.full((n, slots), -1, dtype=np.int8)
    tracked = fragment >= 0
    players_team[tracked] = team_of[fragment[tracked]]
    keeper_slot = keeper_per_frame(players_xy, players_team, period, attack_sign, pitch, fps, cfg)

    player_window = max(1, int(round(cfg.player_velocity_window_s * fps)))
    players_vel = np.full((n, slots, 2), np.nan, dtype=np.float32)
    for slot in range(slots):
        players_vel[:, slot] = fragment_velocity(players_xy[:, slot], fragment[:, slot],
                                                 table["start"], table["end"], player_window, fps)

    # Short fragments may lose frames from here on; the table above still gives their full extent.
    drop = cap_team_size(players_team, fragment, lengths, keeper_slot, cfg.max_per_team)
    players_xy[drop] = np.nan
    players_vel[drop] = np.nan
    fragment[drop] = -1
    players_team[drop] = -1

    # The ball's runs: consecutive kept observations inside one half.
    ball_rows = np.flatnonzero(keep)
    ball_run = np.full(n, -1, dtype=np.int32)
    ball_vel = np.full((n, 2), np.nan, dtype=np.float32)
    if len(ball_rows):
        breaks = np.flatnonzero((np.diff(ball_rows) != 1) | (period[ball_rows[1:]] != period[ball_rows[:-1]])) + 1
        starts = np.concatenate(([0], breaks))
        ends = np.concatenate((breaks, [len(ball_rows)])) - 1
        ball_run[ball_rows] = np.repeat(np.arange(len(starts), dtype=np.int32), ends - starts + 1)
        ball_vel = fragment_velocity(ball_xy, ball_run, ball_rows[starts], ball_rows[ends],
                                     max(1, int(round(cfg.ball_velocity_window_s * fps))), fps)

    report = {
        "ball_observed": int(observed.sum()),
        "ball_removed": int(ball_removed.sum()),
        "fragments": int(len(lengths)),
        "fragment_jumps": jumps,
        "fragment_median_s": float(np.median(lengths) / fps) if len(lengths) else 0.0,
        "relabelled_fragments": int(relabelled.sum()),
        "relabelled_frames": int(lengths[relabelled].sum()),
        "capped_track_frames": int(drop.sum()),
        "attack_margin_m": (float(margin[1]), float(margin[2])),
    }
    base = {f.name: getattr(match, f.name) for f in fields(Match)}
    base.update(ball_xy=ball_xy, ball_z=ball_z, players_xy=players_xy)
    return CleanMatch(
        **base,
        ball_removed=ball_removed,
        ball_vel=ball_vel,
        players_vel=players_vel,
        fragment=fragment,
        players_team=players_team,
        keeper_slot=keeper_slot,
        attack_sign=attack_sign,
        fragments={"slot": table["slot"], "start": table["start"], "end": table["end"],
                   "team": team_of, "relabelled": relabelled},
        report=report,
    ).validate()
