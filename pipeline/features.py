"""Features at 5 Hz, from the clean schema only.

    steps = step_index(cleaned)                        # one step per 0.2 s inside each half
    names, table = features(cleaned, steps)            # (K, n) float32, NaN where a value does not exist
    _, swapped = features(cleaned, steps, swap=True)   # the same with home and away exchanged

A step covers `steps.size` consecutive frames of one half (5 at 25 fps); the last step of a half may be
shorter. Per-frame signals are averaged over the step, then windows look backwards and forwards, never
across the half-time break. Positions are turned into the home side's attacking frame (home attacks
towards +x), so swapping the teams also turns the pitch round: a model trained on both views, and
averaged over both at prediction, treats the two sides the same and carries no home-side prior.

Only these fields are read: frame, period, fps, pitch, ball_xy, ball_vel, players_xy, players_vel,
players_team, keeper_slot, attack_sign and forced_dead. Never extras, view_polygon, labels or ball_z: on
a degraded copy extras is the truth and the polygon follows the true ball. tests/check_features.py
strips them and checks that nothing moves.

The ray distance is test 3's height-aware distance: a ball seen at a ground point may be anywhere
on the stretch of camera ray between that point (height 0) and RAY_Z_MAX metres up, and the distance is
from a player to the nearest point of that stretch. The camera is the one solved from the DATA.md row.
It is a feature beside the plain distance, not a replacement: it is worse for balls on the ground.

Nothing here prints or raises with a value from the data.
"""
from dataclasses import dataclass

import numpy as np

from .camera import solve_camera
from .schema import AWAY, HOME

STEP_HZ = 5.0
RAY_Z_MAX = 3.0         # metres, the highest contact the ray distance allows for
TOUCH_M = 1.5           # a player this close to the ball counts as a touch
RUN_MPS = 4.0           # a player faster than this is running
NEAR_M = (3.0, 10.0)    # players of a team within these distances of the ball
CROWD_M = 5.0           # opponents this close to a player crowd him
CAP_S = 30.0            # times since and until events are capped here
GAP_CAP_S = 60.0
WINDOWS_S = (1.0, 3.0, 10.0)
PAIR_CHUNK = 4096       # steps per chunk for the player-to-player distances

_CENTRE = []


def camera_centre():
    """(3,) metres. The pan camera turns about its centre, so one centre holds for every frame."""
    if not _CENTRE:
        _CENTRE.append(np.asarray(solve_camera().centre, dtype=np.float64))
    return _CENTRE[0]


@dataclass(frozen=True)
class Steps:
    start: np.ndarray   # (K,) first row of the step
    stop: np.ndarray    # (K,) one past its last row
    centre: np.ndarray  # (K,) the row the step stands for
    half: np.ndarray    # (K,) int8, 1 or 2
    size: int           # frames per full step
    fps: float

    @property
    def n_steps(self):
        return len(self.start)

    def bounds(self):
        """((first, stop), ...) step ranges of each half, in step indices."""
        out = []
        for half in (1, 2):
            idx = np.flatnonzero(self.half == half)
            if len(idx):
                out.append((int(idx[0]), int(idx[-1]) + 1))
        return out

    def frame_labels(self, step_values, n_frames, fill):
        """Spread one value per step over its frames; rows outside every step get `fill`."""
        out = np.full(n_frames, fill, dtype=np.asarray(step_values).dtype)
        sizes = self.stop - self.start
        out[_rows(self.start, sizes)] = np.repeat(step_values, sizes)
        return out


def dead_steps(match, steps):
    """(K,) bool: steps with any frame in a published dead block. The decoder keeps them dead."""
    return _Blocks(steps).any(match.forced_dead)


def _rows(start, sizes):
    first = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    return np.repeat(start, sizes) + (np.arange(int(sizes.sum())) - np.repeat(first, sizes))


def step_index(match):
    size = max(1, int(round(match.fps / STEP_HZ)))
    starts, halves = [], []
    for half in (1, 2):
        rows = np.flatnonzero(match.period == half)
        if not len(rows):
            continue
        if rows[-1] - rows[0] + 1 != len(rows):
            raise ValueError("the rows of a half must be contiguous")
        first, stop = int(rows[0]), int(rows[-1]) + 1
        st = np.arange(first, stop, size)
        starts.append(st)
        halves.append(np.full(len(st), half, dtype=np.int8))
    start = np.concatenate(starts).astype(np.int64)
    half = np.concatenate(halves)
    stop = np.empty_like(start)
    for h in (1, 2):
        idx = np.flatnonzero(half == h)
        if len(idx):
            last = int(np.flatnonzero(match.period == h)[-1]) + 1
            stop[idx] = np.minimum(start[idx] + size, last)
    centre = np.minimum(start + size // 2, stop - 1)
    return Steps(start=start, stop=stop, centre=centre, half=half, size=size, fps=float(match.fps))


class _Blocks:
    """Reductions of per-frame arrays over each step's frames."""

    def __init__(self, steps):
        sizes = steps.stop - steps.start
        self.rows = _rows(steps.start, sizes)
        self.first = np.concatenate(([0], np.cumsum(sizes)[:-1]))

    def mean(self, x):
        v = np.asarray(x, dtype=np.float64)[self.rows]
        ok = np.isfinite(v)
        total = np.add.reduceat(np.where(ok, v, 0.0), self.first, axis=0)
        count = np.add.reduceat(ok.astype(np.int32), self.first, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(count > 0, total / np.maximum(count, 1), np.nan).astype(np.float32)

    def min(self, x):
        v = np.asarray(x, dtype=np.float32)[self.rows]
        return np.fmin.reduceat(v, self.first, axis=0)

    def any(self, x):
        v = np.asarray(x, dtype=np.int32)[self.rows]
        return np.add.reduceat(v, self.first, axis=0) > 0


def _window(y, bounds, lo, hi):
    """NaN-aware mean of y over steps [t + lo, t + hi], inside each half."""
    y = np.asarray(y, dtype=np.float64)
    out = np.full(len(y), np.nan, dtype=np.float32)
    for a, b in bounds:
        seg = y[a:b]
        ok = np.isfinite(seg)
        cs = np.concatenate(([0.0], np.cumsum(np.where(ok, seg, 0.0))))
        cn = np.concatenate(([0], np.cumsum(ok)))
        t = np.arange(b - a)
        i0 = np.clip(t + lo, 0, b - a)
        i1 = np.clip(t + hi + 1, 0, b - a)
        n = cn[i1] - cn[i0]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[a:b] = np.where(n > 0, (cs[i1] - cs[i0]) / np.maximum(n, 1), np.nan)
    return out


def _since(event, bounds, dt, cap):
    """Seconds since the last step with the event, the step itself included; cap when there is none."""
    out = np.full(len(event), cap, dtype=np.float32)
    for a, b in bounds:
        t = np.arange(b - a)
        last = np.maximum.accumulate(np.where(event[a:b], t, -1))
        out[a:b] = np.where(last >= 0, np.minimum((t - last) * dt, cap), cap)
    return out


def _until(event, bounds, dt, cap):
    out = np.full(len(event), cap, dtype=np.float32)
    for a, b in bounds:
        out[a:b] = _since(event[a:b][::-1], [(0, b - a)], dt, cap)[::-1]
    return out


def _carry(values, valid, bounds, forward=True):
    """Each step's value at the nearest valid step behind it (forward=True) or ahead, inside a half."""
    values = np.asarray(values, dtype=np.float32)
    out = np.full(values.shape, np.nan, dtype=np.float32)
    for a, b in bounds:
        t = np.arange(b - a)
        v = valid[a:b]
        if forward:
            pick = np.maximum.accumulate(np.where(v, t, -1))
        else:
            pick = np.minimum.accumulate(np.where(v, t, b - a)[::-1])[::-1]
            pick = np.where(pick >= b - a, -1, pick)
        have = pick >= 0
        out[a:b][have] = values[a:b][pick[have]]
    return out


def _team_stats(value, mask):
    """Per frame: mean and max of value over the slots in mask, NaN where the mask is empty."""
    mask = mask & np.isfinite(value)
    count = mask.sum(axis=1)
    total = np.where(mask, value, 0.0).sum(axis=1)
    top = np.where(mask, value, -np.inf).max(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return mean.astype(np.float32), np.where(count > 0, top, np.nan).astype(np.float32), count


def _nearest_two(distance, mask):
    """Per frame: nearest and second nearest distance among the slots in mask (NaN if absent)."""
    d = np.where(mask & np.isfinite(distance), distance, np.inf)
    if d.shape[1] >= 2:
        two = np.partition(d, 1, axis=1)[:, :2]
        two.sort(axis=1)
    else:
        two = np.column_stack((d[:, 0], np.full(len(d), np.inf)))
    two[~np.isfinite(two)] = np.nan
    return two[:, 0].astype(np.float32), two[:, 1].astype(np.float32)


def _ray_distance(ball, players, centre, z_max):
    """(F, S) distance from each player to the stretch of camera ray a ball seen at `ball` can be on."""
    foot, height = centre[:2], float(centre[2])
    back = (ball.astype(np.float64) - foot) * (z_max / height)       # from the ground point to z_max
    rel = players.astype(np.float64) - ball[:, None, :]
    length_sq = (back ** 2).sum(axis=1)[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.clip(-(rel * back[:, None, :]).sum(axis=2) / length_sq, 0.0, 1.0)
    closest = ball[:, None, :] - t[..., None] * back[:, None, :]
    return np.hypot(*(players - closest).transpose(2, 0, 1)).astype(np.float32)


def _last_next_rows(seen, period):
    """Per frame: the row of the last seen ball at or before it and the next at or after it, same half."""
    n = len(seen)
    rows = np.arange(n)
    last = np.full(n, -1, dtype=np.int64)
    nxt = np.full(n, -1, dtype=np.int64)
    for half in (1, 2):
        idx = np.flatnonzero(period == half)
        if not len(idx):
            continue
        a, b = int(idx[0]), int(idx[-1]) + 1
        s = seen[a:b]
        lr = np.maximum.accumulate(np.where(s, rows[a:b], -1))
        nr = np.minimum.accumulate(np.where(s, rows[a:b], n)[::-1])[::-1]
        last[a:b] = lr
        nxt[a:b] = np.where(nr >= n, -1, nr)
    return last, nxt


def _pair_features(px, py, team, centre_rows):
    """At step centres: per team, mean distance to the nearest opponent and the most opponents around
    one player within CROWD_M."""
    k = len(centre_rows)
    out = np.full((k, 4), np.nan, dtype=np.float32)
    for a in range(0, k, PAIR_CHUNK):
        r = centre_rows[a:a + PAIR_CHUNK]
        x, y, tm = px[r], py[r], team[r]
        dist = np.hypot(x[:, :, None] - x[:, None, :], y[:, :, None] - y[:, None, :])
        for j, (mine, theirs) in enumerate(((HOME, AWAY), (AWAY, HOME))):
            me = tm == mine
            them = tm == theirs
            d = np.where(me[:, :, None] & them[:, None, :] & np.isfinite(dist), dist, np.inf)
            near = d.min(axis=2)
            valid = me & np.isfinite(near)
            total = np.where(valid, near, 0.0).sum(axis=1)
            used = valid.sum(axis=1)
            count = (d < CROWD_M).sum(axis=2)
            has = valid.any(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[a:a + PAIR_CHUNK, j] = np.where(has, total / np.maximum(used, 1), np.nan)
            out[a:a + PAIR_CHUNK, 2 + j] = np.where(has, np.where(me, count, 0).max(axis=1), np.nan)
    return out


def features(match, steps, swap=False):
    """(names, table): table is (K, n) float32. See the module docstring."""
    fps = float(match.fps)
    length, width = (float(v) for v in match.pitch)
    period = match.period
    n = len(period)
    inside = period > 0
    blocks = _Blocks(steps)
    bounds = steps.bounds()
    dt = steps.size / fps
    cols = {}

    team = match.players_team.astype(np.int8)
    keeper = match.keeper_slot.astype(np.int64)
    sign_by_period = match.attack_sign.astype(np.float32)
    if swap:
        team = np.where(team >= 0, 1 - team, team).astype(np.int8)
        keeper = keeper[:, ::-1]
        sign_by_period = -sign_by_period
    sign = np.where(inside, sign_by_period[period], 0.0).astype(np.float32)

    # The ball, turned into the home side's attacking frame.
    ball = match.ball_xy.astype(np.float32)
    seen = np.isfinite(ball[:, 0]) & inside
    ball = np.where(seen[:, None], ball, np.nan)
    bxa, bya = sign * ball[:, 0], sign * ball[:, 1]
    bvel = np.where(seen[:, None], match.ball_vel, np.nan).astype(np.float32)
    bvxa = sign * bvel[:, 0]
    bspeed = np.hypot(bvel[:, 0], bvel[:, 1])

    players = np.where(inside[:, None, None], match.players_xy, np.nan).astype(np.float32)
    present = np.isfinite(players[..., 0]) & (team >= 0)
    home, away = present & (team == HOME), present & (team == AWAY)
    pxa, pya = sign[:, None] * players[..., 0], sign[:, None] * players[..., 1]
    pvel = match.players_vel.astype(np.float32)
    pvxa = sign[:, None] * pvel[..., 0]
    pspeed = np.hypot(pvel[..., 0], pvel[..., 1])

    # Keepers and outfield players.
    rows = np.arange(n)
    keeper_mask = np.zeros_like(present)
    keeper_xa = np.full((n, 2), np.nan, dtype=np.float32)
    keeper_speed = np.full((n, 2), np.nan, dtype=np.float32)
    for side in (HOME, AWAY):
        has = keeper[:, side] >= 0
        slot = np.where(has, keeper[:, side], 0)
        ok = has & present[rows, slot] & (team[rows, slot] == side)
        keeper_mask[rows[ok], slot[ok]] = True
        keeper_xa[ok, side] = pxa[rows[ok], slot[ok]]
        keeper_speed[ok, side] = pspeed[rows[ok], slot[ok]]
    outfield_home, outfield_away = home & ~keeper_mask, away & ~keeper_mask

    # Ball against players.
    distance = np.hypot(players[..., 0] - ball[:, None, 0], players[..., 1] - ball[:, None, 1])
    ray = _ray_distance(ball, players, camera_centre(), RAY_Z_MAX)
    rel_speed = np.hypot(pvel[..., 0] - bvel[:, None, 0], pvel[..., 1] - bvel[:, None, 1])
    nearest = {}
    for tag, mask in (("h", home), ("a", away)):
        d1, d2 = _nearest_two(distance, mask)
        r1, _ = _nearest_two(ray, mask)
        nearest[tag] = (d1, r1)
        cols["ball_d1_" + tag] = blocks.mean(d1)
        cols["ball_d2_" + tag] = blocks.mean(d2)
        cols["ball_r1_" + tag] = blocks.mean(r1)
        for radius in NEAR_M:
            with np.errstate(invalid="ignore"):
                count = (mask & (distance < radius)).sum(axis=1).astype(np.float32)
            cols["ball_n{:g}_{}".format(radius, tag)] = blocks.mean(np.where(seen, count, np.nan))
        slot = np.argmin(np.where(mask & np.isfinite(distance), distance, np.inf), axis=1)
        rs = rel_speed[rows, slot]
        cols["ball_relspeed_" + tag] = blocks.mean(np.where(np.isfinite(d1), rs, np.nan))
    dd = nearest["h"][0] - nearest["a"][0]
    rd = nearest["h"][1] - nearest["a"][1]

    cols["ball_seen"] = blocks.mean(seen.astype(np.float32))
    cols["ball_xa"] = blocks.mean(bxa)
    cols["ball_ya"] = blocks.mean(bya)
    cols["ball_y_raw"] = blocks.mean(ball[:, 1])
    cols["ball_out_x"] = blocks.mean(np.abs(bxa) - length / 2)
    cols["ball_out_y"] = blocks.mean(np.abs(bya) - width / 2)
    cols["ball_speed"] = blocks.mean(bspeed)
    cols["ball_vxa"] = blocks.mean(bvxa)
    cols["ball_dd"] = blocks.mean(dd)
    cols["ball_rd"] = blocks.mean(rd)

    # Team shape, ball free. Forward is towards the opponent's goal: +x for home, -x for away.
    for tag, side, mask, outfield, forward in (("h", HOME, home, outfield_home, 1.0),
                                               ("a", AWAY, away, outfield_away, -1.0)):
        cx, _, count = _team_stats(pxa, mask)
        cy, _, _ = _team_stats(pya, mask)
        sx, _, _ = _team_stats((pxa - cx[:, None]) ** 2, mask)
        sy, _, _ = _team_stats((pya - cy[:, None]) ** 2, mask)
        speed_mean, speed_max, _ = _team_stats(pspeed, mask)
        running, _, _ = _team_stats((pspeed > RUN_MPS).astype(np.float32), mask)
        fwd, _, _ = _team_stats(forward * pvxa, mask)
        depth = np.where(outfield, forward * pxa, np.inf).min(axis=1)       # own goal is at -forward * L/2
        reach = np.where(outfield, forward * pxa, -np.inf).max(axis=1)
        top_y = np.where(outfield, pya, -np.inf).max(axis=1)
        low_y = np.where(outfield, pya, np.inf).min(axis=1)
        has_outfield = outfield.any(axis=1)
        cols["n_" + tag] = blocks.mean(np.where(inside, count, np.nan).astype(np.float32))
        cols["cx_" + tag] = blocks.mean(cx)
        cols["sx_" + tag] = blocks.mean(np.sqrt(sx))
        cols["sy_" + tag] = blocks.mean(np.sqrt(sy))
        cols["speed_" + tag] = blocks.mean(speed_mean)
        cols["speedmax_" + tag] = blocks.mean(speed_max)
        cols["running_" + tag] = blocks.mean(running * count)
        cols["fwd_" + tag] = blocks.mean(fwd)
        cols["line_" + tag] = blocks.mean(np.where(has_outfield, depth + length / 2, np.nan))
        cols["reach_" + tag] = blocks.mean(np.where(has_outfield, length / 2 - reach, np.nan))
        cols["span_x_" + tag] = blocks.mean(np.where(has_outfield, reach - depth, np.nan))
        cols["span_y_" + tag] = blocks.mean(np.where(has_outfield, top_y - low_y, np.nan))
        cols["keeper_depth_" + tag] = blocks.mean(forward * keeper_xa[:, side] + length / 2)
        cols["keeper_speed_" + tag] = blocks.mean(keeper_speed[:, side])
    all_speed, _, _ = _team_stats(pspeed, present)
    all_x, _, _ = _team_stats(pxa, present)
    cols["speed_all"] = blocks.mean(all_speed)
    cols["cx_all"] = blocks.mean(all_x)
    cols["cx_diff"] = cols["cx_h"] - cols["cx_a"]
    cols["spread_diff"] = (cols["sx_h"] + cols["sy_h"]) - (cols["sx_a"] + cols["sy_a"])

    pairs = _pair_features(pxa, pya, np.where(present, team, -1), steps.centre)
    cols["press_h"], cols["press_a"], cols["crowd_h"], cols["crowd_a"] = (pairs[:, j] for j in range(4))

    # Touches: the nearest player of a side within TOUCH_M, plainly and along the ray.
    dt_s = dt
    for tag in ("h", "a"):
        d1, r1 = nearest[tag]
        with np.errstate(invalid="ignore"):
            touch = blocks.any(d1 < TOUCH_M)
            touch_ray = blocks.any(r1 < TOUCH_M)
        cols["since_touch_" + tag] = _since(touch, bounds, dt_s, CAP_S)
        cols["until_touch_" + tag] = _until(touch, bounds, dt_s, CAP_S)
        cols["since_touchray_" + tag] = _since(touch_ray, bounds, dt_s, CAP_S)
        cols["until_touchray_" + tag] = _until(touch_ray, bounds, dt_s, CAP_S)
    cols["since_touch_diff"] = cols["since_touch_a"] - cols["since_touch_h"]
    cols["until_touch_diff"] = cols["until_touch_a"] - cols["until_touch_h"]

    # Gaps: how long the ball has been and will be away, and what it looked like at both ends.
    seen_step = blocks.any(seen)
    since_seen = _since(seen_step, bounds, dt_s, GAP_CAP_S)
    until_seen = _until(seen_step, bounds, dt_s, GAP_CAP_S)
    cols["since_seen"] = since_seen
    cols["until_seen"] = until_seen
    cols["gap_len"] = np.where(seen_step, 0.0, np.minimum(since_seen + until_seen, GAP_CAP_S)).astype(np.float32)
    for name in ("ball_xa", "ball_ya", "ball_out_x", "ball_out_y", "ball_dd", "ball_rd", "ball_speed", "ball_vxa"):
        cols["last_" + name[5:]] = _carry(cols[name], seen_step, bounds, forward=True)
        cols["next_" + name[5:]] = _carry(cols[name], seen_step, bounds, forward=False)

    # The ball interpolated across a gap, at step centres, against the players there.
    last_row, next_row = _last_next_rows(seen, period)
    c = steps.centre
    lr, nr = last_row[c], next_row[c]
    ok = (lr >= 0) & (nr >= 0)
    w = np.where(ok & (nr > lr), (c - lr) / np.maximum(nr - lr, 1), 0.0)[:, None]
    guess = np.where(ok[:, None], (1 - w) * ball[np.maximum(lr, 0)] + w * ball[np.maximum(nr, 0)], np.nan)
    guess = np.where(np.isfinite(ball[c]), ball[c], guess)
    gd = np.hypot(players[c, :, 0] - guess[:, None, 0], players[c, :, 1] - guess[:, None, 1])
    g1h, _ = _nearest_two(gd, home[c])
    g1a, _ = _nearest_two(gd, away[c])
    cols["guess_d1_h"], cols["guess_d1_a"], cols["guess_dd"] = g1h, g1a, g1h - g1a
    cols["guess_xa"] = (sign[c] * guess[:, 0]).astype(np.float32)

    # Windows, backwards and forwards, inside each half.
    win = {"seen": cols["ball_seen"], "dd": cols["ball_dd"], "rd": cols["ball_rd"], "gdd": cols["guess_dd"],
           "speed_all": cols["speed_all"], "cx_diff": cols["cx_diff"], "fwd_h": cols["fwd_h"],
           "fwd_a": cols["fwd_a"], "running_h": cols["running_h"], "running_a": cols["running_a"],
           "ball_speed": cols["ball_speed"], "spread_diff": cols["spread_diff"]}
    plan = {"seen": WINDOWS_S, "dd": WINDOWS_S[:2], "rd": WINDOWS_S[:2], "gdd": WINDOWS_S[:2],
            "speed_all": WINDOWS_S, "cx_diff": WINDOWS_S[1:], "fwd_h": WINDOWS_S[:2], "fwd_a": WINDOWS_S[:2],
            "running_h": WINDOWS_S[:2], "running_a": WINDOWS_S[:2], "ball_speed": WINDOWS_S[:1],
            "spread_diff": WINDOWS_S[1:2]}
    for key, spans in plan.items():
        for span in spans:
            k = max(1, int(round(span / dt)))
            cols["{}_past{:g}".format(key, span)] = _window(win[key], bounds, -k + 1, 0)
            cols["{}_next{:g}".format(key, span)] = _window(win[key], bounds, 0, k - 1)
    for key in ("dd", "rd"):
        cols[key + "_mid"] = _window(win[key], bounds, -1, 1)

    # No clock features: where the competition's halves start against the kick-off is not known, and
    # the published dead blocks go to the decoder (dead_steps), not to a model that never saw one.
    names = list(cols)
    table = np.column_stack([np.asarray(cols[k], dtype=np.float32) for k in names])
    return names, table
