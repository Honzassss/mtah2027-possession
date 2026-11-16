"""Training only: turn clean open tracking into a look-alike of the automatic tracking.

    copy = degrade(load_idsse(path), seed=1)                             # every nuisance drawn per copy
    copy = degrade(match, seed=1, cfg=replace(SIMULATORS["plain"], live_share=0.3))
    cleaned = clean(copy)                                                # as on the competition bundle

The labels come through unchanged. What changes is what a model may learn from:

    ball        goes missing in contiguous blocks whose lengths are heavy-tailed, with separate hazards
                for live and dead play and a slow per-block drift in visibility
    stoppages   DFL parks the ball in stoppages and jumps it to the restart. Here it coasts, gets carried
                near a player and arrives at the restart spot, so neither a still ball nor a jump marks
                dead play, and how much of the missing time is live is drawn per copy, so neither does
                absence
    camera      a ball with height lands further along the camera ray, noise is added in pixels, and
                whatever leaves the moving image is lost. See pipeline/camera.py
    identities  people become tracklets in a fixed set of slots: breaks, swaps at occlusions, tracklets on
                the wrong team, keeper time on the other team, duplicates and false tracks

Targets are only what the public documents print: 41% of graded frames without a ball, 20.5 tracked
objects on average, 46 slots and the homography row. Everything else is an assumption, kept wide and
drawn per copy. Figures measured from the match or from the baseline CSVs do not belong in this file
until the organisers have answered: they may be extracts of the dataset, and this repository goes public.

A Match without labels is refused, which includes the competition bundle. `extras` holds the truth for
the self-checks and no source fields, so the raw speed fields are gone; nothing may feed `extras` to a
model.

The camera follows the true ball, so `view_polygon` knows where the ball is inside a gap. Whether the
real camera's aim tells anything is test 9 (scripts/match_test9.py); until it passes, no feature may read the polygon.
"""
import heapq
import math
import zlib
from dataclasses import dataclass, fields, replace
from types import SimpleNamespace

import numpy as np

from .camera import apparent_ground, project, solve_camera
from .schema import AWAY, HOME, Match, SchemaError


@dataclass(frozen=True)
class DegradeConfig:
    """A (low, high) tuple is drawn uniformly per copy. A plain number holds for every copy."""
    # Ball visibility
    missing_share: float = 0.41             # DATA.md: of graded frames
    live_share: tuple = (0.05, 0.55)        # of the missing frames. Unknown, and estimates disagree widely
    short_gap_share: tuple = (0.3, 0.7)     # of the gaps
    short_gap_median_s: tuple = (0.4, 1.2)
    short_gap_sigma: float = 0.8
    long_gap_median_s: tuple = (5.0, 20.0)
    long_gap_sigma: tuple = (0.7, 1.2)
    max_gap_s: float = 300.0
    min_seen_frames: int = 5                # a gap never starts right on the heels of the last one
    block_s: float = 300.0                  # visibility drifts from block to block
    block_sigma: tuple = (0.2, 1.0)         # of the log hazard
    block_rho: float = 0.7
    dead_snap: tuple = (0.2, 0.8)           # chance a gap that starts in a stoppage ends after the restart
    snap_delay_s: float = 1.5               # mean delay of that end
    air_height_m: float = 2.5
    air_loss: tuple = (0.0, 0.6)            # chance a flight is lost while above air_height_m. DATA.md names it
    crowd_radius_m: float = 3.0
    crowd_count: int = 3                    # this many players around the ball count as a crowd
    crowd_factor: tuple = (1.0, 4.0)        # on the live hazard. DATA.md names crowding and occlusion

    # The ball in stoppages
    coast_s: tuple = (0.5, 3.0)             # time constant of the slowdown after the ball goes dead
    carry_segment_s: float = 5.0            # mean time before another player handles it
    carry_tau_s: tuple = (0.4, 2.0)         # how tightly it follows that player
    settle_s: tuple = (1.0, 4.0)            # time it takes to reach the restart spot

    # Ball noise and false detections
    ball_noise_px: tuple = (0.5, 2.0)
    ball_noise_m: tuple = (0.1, 0.4)        # used when camera is False
    ball_noise_tau_s: float = 0.2
    spike_rate_per_min: tuple = (0.0, 2.0)  # of ball time seen: a false point for a few frames
    spike_max_frames: int = 3
    spike_min_m: float = 5.0
    spike_max_m: float = 40.0
    decoy_rate_per_min: tuple = (0.0, 0.5)  # of missing time: a still false ball, in live and dead play alike
    decoy_mean_s: float = 2.0

    # Camera
    camera: bool = True
    parallax_chance: float = 0.5            # chance a copy has no height correction. Unknown for the real tracker
    off_pitch_margin_m: float = 15.0        # an apparent ball further out than this is lost
    pan_gain: tuple = (0.7, 1.0)
    pan_smooth_s: float = 2.0
    pan_lag_s: tuple = (0.5, 1.5)
    pan_wander_deg: tuple = (0.0, 1.5)
    pan_wander_tau_s: float = 5.0
    zoom: tuple = (0.7, 1.0)                # focal length against the printed frame's. The real zoom is unknown,
                                            # and it sets how much of the tracking loss is the view's
    view_margin_px: float = 10.0            # a player this close to the image edge is already lost

    # Players
    n_slots: int = 46                       # DATA.md
    slot_split_spread: int = 2              # the split between the teams is unknown
    tracked_mean: tuple = (20.0, 21.0)      # DATA.md: 20.5 over graded frames
    player_noise_px: tuple = (0.5, 2.0)
    player_noise_m: tuple = (0.1, 0.4)      # used when camera is False
    player_noise_tau_s: float = 1.0
    tracklet_mean_s: tuple = (15.0, 60.0)   # mean time between random breaks
    switch_share: float = 0.15              # of random breaks: only the slot changes, no frames are lost
    lost_mean_s: float = 3.0                # rescaled until tracked_mean holds
    lost_sigma: float = 1.0
    occlusion_radius_m: float = 1.0
    occlusion_break: float = 0.25           # chance an occlusion breaks one of the two tracklets
    occlusion_lost_mean_s: float = 1.0
    swap_share: tuple = (0.0, 0.2)          # chance an occlusion swaps the two people's slots, teams or not
    flicker_share: tuple = (0.0, 0.04)      # of tracklets, on a slot of the wrong team
    keeper_reuse: tuple = (0.0, 0.2)        # the same for a keeper's tracklets
    duplicate_rate_per_min: tuple = (1.0, 6.0)  # DATA.md: the tracker sometimes duplicates someone
    duplicate_mean_s: float = 2.5
    duplicate_offset_m: float = 1.0
    false_rate_per_min: tuple = (0.0, 2.0)  # people who are not players
    false_mean_s: float = 4.0
    min_track_frames: int = 3


# The gate compares changes under two different simulators. Live share and the rest stay drawn per copy.
SIMULATORS = {
    "camera": DegradeConfig(),
    "camera-no-parallax": DegradeConfig(parallax_chance=0.0),
    "camera-parallax": DegradeConfig(parallax_chance=1.0),
    "plain": DegradeConfig(camera=False, parallax_chance=0.0, air_loss=0.0, crowd_factor=1.0),
}


def _stream(seed, match_id, name):
    """A generator for one purpose, so fixing a parameter or changing a step leaves the others alone."""
    return np.random.default_rng([int(seed), zlib.crc32(str(match_id).encode("utf-8")),
                                  zlib.crc32(name.encode("ascii"))])


def _draw_parameters(cfg, seed, match_id):
    values = {}
    for item in fields(cfg):
        spec = getattr(cfg, item.name)
        if isinstance(spec, tuple):
            spec = float(_stream(seed, match_id, "param-" + item.name).uniform(*spec))
        values[item.name] = spec
    values["parallax"] = 1.0 if _stream(seed, match_id, "parallax").random() < cfg.parallax_chance else 0.0
    return SimpleNamespace(**values)


def _runs(mask):
    """(starts, ends) of the True runs, ends exclusive."""
    edges = np.diff(np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def _halves(period):
    out = []
    for half in (1, 2):
        rows = np.flatnonzero(period == half)
        if len(rows):
            if rows[-1] - rows[0] + 1 != len(rows):
                raise SchemaError("degrade: a half is not contiguous")
            out.append((int(rows[0]), int(rows[-1]) + 1))
    return out


def _ar1(rng, n, m, sigma, tau_frames):
    """Stationary AR(1), shape (n, m), float32."""
    phi = math.exp(-1.0 / max(tau_frames, 1e-6))
    innovations = rng.standard_normal((n, m), dtype=np.float32) * (sigma * math.sqrt(1.0 - phi * phi))
    out = np.empty((n, m), dtype=np.float32)
    out[0] = rng.standard_normal(m, dtype=np.float32) * sigma
    for t in range(1, n):
        out[t] = phi * out[t - 1] + innovations[t]
    return out


# ---------------------------------------------------------------- the ball in stoppages

def _handled_path(start, velocity, players, end, p, rng, fps):
    """Ball path over one stoppage: coast from start, follow nearby players, settle on end if given."""
    n = len(players)
    t = np.arange(1, n + 1) / fps
    coast = min(n, int(3.0 * p.coast_s * fps))
    decay = p.coast_s * (1.0 - np.exp(-t[:coast] / p.coast_s))
    path = np.empty((n, 2))
    path[:coast] = start + velocity * decay[:, None]
    x, y = (path[coast - 1] if coast else start).tolist()
    i = coast
    while i < n:
        stop = min(n, i + max(int(fps), int(rng.exponential(p.carry_segment_s * fps))))
        distance = np.hypot(players[i, :, 0] - x, players[i, :, 1] - y)
        order = np.argsort(np.where(np.isfinite(distance), distance, np.inf))[:3]
        order = order[np.isfinite(distance[order])]
        xs, ys = [], []
        if len(order):
            carrier = int(order[rng.integers(len(order))])
            angle = rng.uniform(0.0, 2.0 * math.pi)
            reach = rng.uniform(0.2, 1.0)
            target_x = (players[i:stop, carrier, 0] + reach * math.cos(angle)).tolist()
            target_y = (players[i:stop, carrier, 1] + reach * math.sin(angle)).tolist()
            alpha = 1.0 / (rng.uniform(0.5, 1.5) * p.carry_tau_s * fps)
            for target_xi, target_yi in zip(target_x, target_y):
                if target_xi == target_xi:      # the carrier may have gone off
                    x += alpha * (target_xi - x)
                    y += alpha * (target_yi - y)
                xs.append(x)
                ys.append(y)
        else:
            xs, ys = [x] * (stop - i), [y] * (stop - i)
        path[i:stop, 0], path[i:stop, 1] = xs, ys
        i = stop
    if end is not None:
        settle = min(n, max(2, int(p.settle_s * fps)))
        u = np.arange(1, settle + 1) / (settle + 1.0)
        weight = u * u * (3.0 - 2.0 * u)
        path[n - settle:] += (end - path[n - settle:]) * weight[:, None]
    return path


def unpark_ball(match, p, rng):
    """(xy, z) float64 with every stoppage replaced by a handled ball. Live frames stay as they are."""
    fps = match.fps
    xy = match.ball_xy.astype(np.float64)
    z = match.ball_z.astype(np.float64) if match.ball_z is not None else np.zeros(match.n_frames)
    dead = match.labels == "D"
    for first, last in _halves(match.period):
        starts, ends = _runs(dead[first:last])
        for a, b in zip((starts + first).tolist(), (ends + first).tolist()):
            end = xy[b].copy() if b < last else None
            if a > first:
                back = min(5, a - 1 - first)
                start = xy[a - 1].copy()
                velocity = (xy[a - 1] - xy[a - 1 - back]) * fps / back if back else np.zeros(2)
                speed = float(np.hypot(*velocity))
                if speed > 30.0:
                    velocity *= 30.0 / speed
                start_z = float(z[a - 1])
            else:
                start = (end if end is not None else xy[a]) + rng.uniform(-5.0, 5.0, 2)
                velocity = np.zeros(2)
                start_z = 0.0
            xy[a:b] = _handled_path(start, velocity, match.players_xy[a:b], end, p, rng, fps)
            z[a:b] = start_z * np.exp(-np.arange(1, b - a + 1) / (0.4 * fps))
    return xy, z


# ---------------------------------------------------------------- when the ball is seen

def _gap_lengths(rng, count, p, fps):
    short = rng.random(count) < p.short_gap_share
    normal = rng.standard_normal(count)
    seconds = np.where(short, p.short_gap_median_s * np.exp(p.short_gap_sigma * normal),
                       p.long_gap_median_s * np.exp(p.long_gap_sigma * normal))
    return np.clip(np.round(seconds * fps), 2, p.max_gap_s * fps).astype(np.int64)


def _place_gaps(hazard, dead, next_live, draws, snap_share, min_seen):
    """One half. A gap starts when the hazard accumulated while the ball is seen passes the next wait."""
    n = len(hazard)
    mask = np.zeros(n, dtype=bool)
    cumulative = np.cumsum(hazard)
    level = 0.0
    gaps = 0
    for wait, length, snap_draw, delay in zip(*draws):
        start = int(np.searchsorted(cumulative, level + wait))
        if start >= n:
            break
        if dead[start] and snap_draw < snap_share:
            length = int(next_live[start]) - start + delay
        stop = min(n, start + length)
        mask[start:stop] = True
        gaps += 1
        if stop >= n:
            break
        level = float(cumulative[min(n - 1, stop + min_seen)])
    return mask, gaps


def _odds(value):
    value = min(max(value, 1e-4), 1.0 - 1e-4)
    return value / (1.0 - value)


def ball_visibility(forced_missing, dead, weight, halves, p, rng, fps):
    """(seen mask, statistics). Live and dead hazards are tuned until the two missing shares hold.

    forced_missing are frames lost whatever the hazards do. A gap that starts in a stoppage and runs past
    the restart adds live missing frames, so when a low live share cannot be reached the chance of ending
    at the restart goes up. The share actually reached is in the statistics.
    """
    n = len(dead)
    inside = np.zeros(n, dtype=bool)
    for first, last in halves:
        inside[first:last] = True
    live_frames = int((inside & ~dead).sum())
    dead_frames = int((inside & dead).sum())
    dead_share = dead_frames / max(1, live_frames + dead_frames)
    target_dead = min(0.98, p.missing_share * (1.0 - p.live_share) / max(dead_share, 1e-6))
    target_live = (p.missing_share - target_dead * dead_share) / max(1.0 - dead_share, 1e-6)

    delay = max(1.0, p.snap_delay_s * fps)
    plans = []
    for first, last in halves:
        size = last - first
        count = 6 * size // (p.min_seen_frames + 2) // 5 + 16
        plans.append((first, last, (rng.exponential(1.0, count).tolist(),
                                    _gap_lengths(rng, count, p, fps).tolist(),
                                    rng.random(count).tolist(),
                                    np.round(rng.exponential(delay, count)).astype(np.int64).tolist())))
    next_live = np.arange(n, dtype=np.int64)
    next_live[dead] = n
    for first, last in halves:
        part = next_live[first:last]
        part[part >= n] = last
        next_live[first:last] = np.minimum.accumulate(part[::-1])[::-1] - first
    mean_length = float(np.mean(plans[0][2][1]))

    def simulate(hazard_live, hazard_dead, snap):
        hazard = weight * np.where(dead, hazard_dead, hazard_live)
        mask = forced_missing.copy()
        gaps = 0
        for first, last, draws in plans:
            part, count = _place_gaps(hazard[first:last], dead[first:last], next_live[first:last], draws, snap,
                                      p.min_seen_frames)
            mask[first:last] |= part
            gaps += count
        return mask, gaps

    hazard_live = _odds(target_live) / mean_length
    hazard_dead = _odds(target_dead) / mean_length
    snap = p.dead_snap
    best = None
    tolerance = 0.002
    for _ in range(5):
        reached_live = 1.0
        for _ in range(12):
            mask, gaps = simulate(hazard_live, hazard_dead, snap)
            reached_live = float(mask[inside & ~dead].mean()) if live_frames else 0.0
            reached_dead = float(mask[inside & dead].mean()) if dead_frames else 0.0
            error = abs(reached_live - target_live) * live_frames + abs(reached_dead - target_dead) * dead_frames
            if best is None or error < best[0]:
                best = (error, mask, gaps, snap, reached_live, reached_dead)
            if abs(reached_live - target_live) < tolerance and abs(reached_dead - target_dead) < tolerance:
                break
            hazard_live = min(1.0, max(1e-9, hazard_live * (_odds(target_live) / _odds(reached_live)) ** 0.8))
            hazard_dead = min(1.0, max(1e-9, hazard_dead * (_odds(target_dead) / _odds(reached_dead)) ** 0.8))
        if reached_live <= target_live + 2.0 * tolerance or snap >= 1.0:
            break
        snap = min(1.0, snap + 0.25)
    _, mask, gaps, snap, reached_live, reached_dead = best
    return ~mask, {"gaps": int(gaps), "snap_used": float(snap), "target_live": float(target_live),
                   "target_dead": float(target_dead), "missing_given_live": reached_live,
                   "missing_given_dead": reached_dead}


def _block_weight(n, halves, p, rng, fps):
    """Multiplier on the hazard per frame, drifting from block to block, independent of play."""
    weight = np.ones(n)
    block = max(1, int(p.block_s * fps))
    innovation = math.sqrt(1.0 - p.block_rho * p.block_rho)
    for first, last in halves:
        value = rng.standard_normal()
        for start in range(first, last, block):
            weight[start:min(last, start + block)] = math.exp(p.block_sigma * value)
            value = p.block_rho * value + innovation * rng.standard_normal()
    return weight


def _crowded(truth_xy, players_xy, radius, count):
    delta = players_xy - truth_xy[:, None, :].astype(np.float32)
    with np.errstate(invalid="ignore"):
        near = (delta[..., 0] ** 2 + delta[..., 1] ** 2) < radius * radius
    return near.sum(axis=1) >= count


def _flights_lost(truth_z, live, halves, p, target_live, rng, fps):
    """Frames lost because a high ball is hard to see. Never more than half of the live missing budget."""
    high = live & (truth_z >= p.air_height_m)
    chance = p.air_loss
    if high.any():
        chance = min(chance, 0.5 * target_live * live.sum() / high.sum())
    mask = np.zeros(len(live), dtype=bool)
    edge = int(0.1 * fps)
    for first, last in halves:
        starts, ends = _runs(high[first:last])
        for a, b, draw in zip(starts.tolist(), ends.tolist(), rng.random(len(starts)).tolist()):
            if draw < chance:
                mask[first + max(0, a - edge):first + min(last - first, b + edge)] = True
    return mask


def _false_detections(observed, seen, truth_xy, matrices, halves, pitch, image_size, p, rng, fps):
    """Add spikes on seen frames and still decoys on missing ones, in place.

    Returns (kind per frame, int8: 0 real or none, 1 spike, 2 decoy; spike count; decoy count).
    """
    n = len(seen)
    false_ball = np.zeros(n, dtype=np.int8)

    def in_frame(x, y, frame):
        if matrices is None:
            return True
        point = project(matrices[frame:frame + 1], np.array([[x, y]]))[0]
        return (np.isfinite(point).all() and 0.0 <= point[0] < image_size[0] and 0.0 <= point[1] < image_size[1])

    half_end = np.zeros(n, dtype=np.int64)
    for first, last in halves:
        half_end[first:last] = last

    rows = np.flatnonzero(seen)
    length_m, width_m = pitch
    spikes = 0
    if len(rows):
        count = rng.poisson(p.spike_rate_per_min * len(rows) / fps / 60.0)
        for frame in rng.choice(rows, size=count, replace=False).tolist():
            angle = rng.uniform(0.0, 2.0 * math.pi)
            reach = rng.uniform(p.spike_min_m, p.spike_max_m)
            x = float(np.clip(truth_xy[frame, 0] + reach * math.cos(angle), -length_m / 2, length_m / 2))
            y = float(np.clip(truth_xy[frame, 1] + reach * math.sin(angle), -width_m / 2, width_m / 2))
            if not in_frame(x, y, frame):
                continue
            stop = min(int(half_end[frame]), frame + int(rng.integers(1, p.spike_max_frames + 1)))
            observed[frame:stop] = (x, y)
            false_ball[frame:stop] = 1
            spikes += 1

    rows = np.flatnonzero(~seen)
    decoys = 0
    if len(rows):
        count = rng.poisson(p.decoy_rate_per_min * len(rows) / fps / 60.0)
        for frame in rng.choice(rows, size=count, replace=False).tolist():
            x = float(rng.uniform(-length_m / 2, length_m / 2))
            y = float(rng.uniform(-width_m / 2, width_m / 2))
            if not in_frame(x, y, frame):
                continue
            wanted = max(2, int(rng.exponential(p.decoy_mean_s * fps)))
            stop = min(int(half_end[frame]), frame + wanted)
            # Only over frames with no real ball, so the decoy stays inside its gap.
            hole = np.flatnonzero(seen[frame:stop])
            if len(hole):
                stop = frame + int(hole[0])
            observed[frame:stop] = (x, y)
            false_ball[frame:stop] = 2
            decoys += 1
    return false_ball, spikes, decoys


# ---------------------------------------------------------------- camera path

def _pan_path(camera, truth_xy, halves, p, rng, fps):
    """Pan per frame: a smoothed, delayed, partial follow of the true ball, plus some operator wander."""
    n = len(truth_xy)
    pan = np.zeros(n)
    target = camera.pan_to_face(truth_xy[:, 0], truth_xy[:, 1])
    alpha = 1.0 / (p.pan_smooth_s * fps)
    lag = int(round(p.pan_lag_s * fps))
    for first, last in halves:
        value = float(target[first])
        smooth = []
        for sample in target[first:last].tolist():
            value += alpha * (sample - value)
            smooth.append(value)
        smooth = np.asarray(smooth)
        delayed = np.full(last - first, smooth[0])
        if last - first > lag:
            delayed[lag:] = smooth[:last - first - lag]
        pan[first:last] = p.pan_gain * delayed
    if p.pan_wander_deg > 0:
        wander = _ar1(rng, n, 1, math.radians(p.pan_wander_deg), p.pan_wander_tau_s * fps)[:, 0]
        pan = pan + wander
    return pan


# ---------------------------------------------------------------- identities

def _occlusion_events(players_xy, present, p, rng):
    """{person: [(frame, lost in frames as a fraction of the mean, partner or -1)]}, in frame order.

    An occlusion starts when two tracked people come within the radius. It may break one tracklet, and it
    may swap the two, which keeps both tracks running with each on the other's slot.
    """
    n, people = present.shape
    sampled = np.arange(0, n, 5)
    xy = players_xy[sampled]
    seen = present[sampled]
    limit = p.occlusion_radius_m ** 2
    events = {person: [] for person in range(people)}
    active = [person for person in range(people) if seen[:, person].any()]
    for index, i in enumerate(active):
        for j in active[index + 1:]:
            delta = xy[:, i] - xy[:, j]
            with np.errstate(invalid="ignore"):
                close = ((delta[:, 0] ** 2 + delta[:, 1] ** 2) < limit) & seen[:, i] & seen[:, j]
            starts = np.flatnonzero(close[1:] & ~close[:-1]) + 1
            if not len(starts):
                continue
            swap = rng.random(len(starts)) < p.swap_share
            broken = rng.random(len(starts)) < p.occlusion_break
            first_loses = rng.random(len(starts)) < 0.5
            lost = rng.exponential(1.0, len(starts))
            rows = zip(sampled[starts].tolist(), swap.tolist(), broken.tolist(), first_loses.tolist(), lost.tolist())
            for frame, is_swap, is_broken, first, amount in rows:
                if is_swap:
                    events[i].append((frame, 0.0, j))
                    events[j].append((frame, 0.0, i))
                elif is_broken:
                    events[i if first else j].append((frame, amount, -1))
    for person in events:
        events[person].sort()
    return events


def _person_plan(present, occlusions, halves, p, seed, match_id, fps):
    """Per person, the visible runs and every break in them, with loss times split by cause for rescaling."""
    plan = []
    mean_life = p.tracklet_mean_s * fps
    lost_shape = math.exp(-0.5 * p.lost_sigma * p.lost_sigma)
    for person in range(present.shape[1]):
        rng = _stream(seed, match_id, "tracklets-{}".format(person))
        pieces = []
        for first, last in halves:
            starts, ends = _runs(present[first:last, person])
            for a, b in zip((starts + first).tolist(), (ends + first).tolist()):
                if b - a < p.min_track_frames:
                    continue
                count = int((b - a) / mean_life * 3) + 8
                times = a + np.cumsum(rng.exponential(mean_life, count))
                switch = rng.random(count) < p.switch_share
                lost = rng.lognormal(0.0, p.lost_sigma, count) * lost_shape * p.lost_mean_s * fps
                lost[switch] = 0.0
                breaks = [(int(t), float(amount), 0.0, -1)
                          for t, amount in zip(times.tolist(), lost.tolist()) if t < b]
                occlusion_fps = p.occlusion_lost_mean_s * fps
                for frame, amount, partner in occlusions[person]:
                    if a < frame < b:
                        breaks.append((frame, 0.0, amount * occlusion_fps, partner))
                breaks.sort()
                pieces.append((a, b, breaks))
        plan.append(pieces)
    return plan


def _cut(plan, scale, min_frames):
    """[(person, start, end, partner)]: the same scale always gives the same cuts."""
    out = []
    occlusion_scale = min(scale, 1.0)
    for person, pieces in enumerate(plan):
        for a, b, breaks in pieces:
            current, partner_now = a, -1
            for frame, lost, occlusion_lost, partner in breaks:
                if frame <= current:
                    continue
                if frame - current >= min_frames:
                    out.append((person, current, frame, partner_now))
                current = frame + int(round(lost * scale + occlusion_lost * occlusion_scale))
                partner_now = partner
                if current >= b:
                    break
            if current < b and b - current >= min_frames:
                out.append((person, current, b, partner_now))
    return out


def _tune_loss(plan, wanted_frames, min_frames):
    """Scale on the loss times that brings the tracked frames to wanted_frames, by bisection."""
    def frames(scale):
        return sum(end - start for _, start, end, _ in _cut(plan, scale, min_frames))

    if frames(0.0) <= wanted_frames:
        return 0.0
    low, high = 0.0, 1.0
    while frames(high) > wanted_frames and high < 4096.0:
        low, high = high, high * 2.0
    for _ in range(14):
        middle = 0.5 * (low + high)
        if frames(middle) > wanted_frames:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def _extra_tracks(players_xy, present, halves, pitch, matrices, image_size, p, rng, fps):
    """Duplicates of people and tracks of nobody: [(kind, person, start, end, path (m, 2))]."""
    tracks = []
    minutes = sum(last - first for first, last in halves) / fps / 60.0
    length_m, width_m = pitch
    rows = np.concatenate([np.arange(first, last) for first, last in halves])
    half_end = np.zeros(len(players_xy), dtype=np.int64)
    for first, last in halves:
        half_end[first:last] = last
    for frame in rng.choice(rows, size=rng.poisson(p.duplicate_rate_per_min * minutes)).tolist():
        people = np.flatnonzero(present[frame])
        if not len(people):
            continue
        person = int(people[rng.integers(len(people))])
        stop = min(int(half_end[frame]), frame + max(p.min_track_frames, int(rng.exponential(p.duplicate_mean_s * fps))))
        stop = frame + int(np.argmin(np.r_[present[frame:stop, person], False]))
        if stop - frame < p.min_track_frames:
            continue
        angle = rng.uniform(0.0, 2.0 * math.pi)
        offset = rng.uniform(0.5, 1.0) * p.duplicate_offset_m * np.array([math.cos(angle), math.sin(angle)])
        tracks.append(("duplicate", person, frame, stop, offset))
    for frame in rng.choice(rows, size=rng.poisson(p.false_rate_per_min * minutes)).tolist():
        stop = min(int(half_end[frame]), frame + max(p.min_track_frames, int(rng.exponential(p.false_mean_s * fps))))
        if rng.random() < 0.5:      # someone along a touchline, or someone on the pitch
            start = np.array([rng.uniform(-length_m / 2, length_m / 2),
                              rng.choice([-1.0, 1.0]) * (width_m / 2 + rng.uniform(-1.0, 2.0))])
        else:
            start = np.array([rng.uniform(-length_m / 2, length_m / 2), rng.uniform(-width_m / 2, width_m / 2)])
        angle = rng.uniform(0.0, 2.0 * math.pi)
        speed = rng.uniform(0.3, 2.5)
        steps = np.arange(stop - frame)[:, None] / fps
        path = start + steps * speed * np.array([math.cos(angle), math.sin(angle)])
        if matrices is not None:
            pixels = project(matrices[frame:stop], path)
            with np.errstate(invalid="ignore"):
                seen = ((pixels[:, 0] >= 0) & (pixels[:, 0] < image_size[0])
                        & (pixels[:, 1] >= 0) & (pixels[:, 1] < image_size[1]))
            stop = frame + int(np.argmin(np.r_[seen, False]))
            path = path[:stop - frame]
            if stop - frame < p.min_track_frames:
                continue
        tracks.append(("false", -1, frame, stop, path))
    return tracks


def _assign_slots(tracks, slot_team, rng):
    """Slot per track, or -1 if every slot of its team is taken. tracks: [(start, end, team, partner, person)].

    A track takes a random free slot of its team, except after a swap: then it takes the slot its partner
    has just left, whichever team that slot belongs to.
    """
    free = {HOME: np.flatnonzero(slot_team == HOME).tolist(), AWAY: np.flatnonzero(slot_team == AWAY).tolist()}
    busy = []
    last_slot = {}
    slots = [-1] * len(tracks)
    for i in sorted(range(len(tracks)), key=lambda k: (tracks[k][0], tracks[k][1], k)):
        start, end, team, partner, person = tracks[i]
        while busy and busy[0][0] <= start:
            _, released = heapq.heappop(busy)
            free[int(slot_team[released])].append(released)
        slot = -1
        if partner >= 0 and partner in last_slot:
            held, ended = last_slot[partner]
            pool = free[int(slot_team[held])]
            if ended == start and held in pool:
                pool.remove(held)
                slot = held
        if slot < 0:
            pool = free[team]
            if not pool:
                continue
            slot = pool.pop(int(rng.integers(len(pool))))
        heapq.heappush(busy, (end, slot))
        slots[i] = slot
        if person >= 0:
            last_slot[person] = (slot, end)
    return slots


def degrade_players(match, matrices, inverse, image_size, p, seed, fps):
    """(players_xy (F, S, 2) float32, slot_team, slot_person (F, S) int16, statistics)."""
    n, people = match.n_frames, match.n_slots
    halves = _halves(match.period)
    match_id = match.match_id
    true_xy = match.players_xy
    on_pitch = np.isfinite(true_xy[..., 0])

    if matrices is not None:
        pixels = project(matrices, true_xy)
        margin = p.view_margin_px
        width, height = image_size
        with np.errstate(invalid="ignore"):
            in_frame = ((pixels[..., 0] >= margin) & (pixels[..., 0] < width - margin)
                        & (pixels[..., 1] >= margin) & (pixels[..., 1] < height - margin))
        present = on_pitch & in_frame
        noise = _ar1(_stream(seed, match_id, "player-noise"), n, 2 * people, p.player_noise_px,
                     p.player_noise_tau_s * fps).reshape(n, people, 2)
        positions = project(inverse, pixels + noise).astype(np.float32)
        del pixels, noise
    else:
        present = on_pitch.copy()
        noise = _ar1(_stream(seed, match_id, "player-noise"), n, 2 * people, p.player_noise_m / math.sqrt(2.0),
                     p.player_noise_tau_s * fps).reshape(n, people, 2)
        positions = (true_xy + noise).astype(np.float32)
        del noise
    view_lost = 1.0 - float(present.sum()) / max(1, int(on_pitch.sum()))

    occlusions = _occlusion_events(true_xy, present, p, _stream(seed, match_id, "occlusions"))
    plan = _person_plan(present, occlusions, halves, p, seed, match_id, fps)
    extras = _extra_tracks(true_xy, present, halves, match.pitch, matrices, image_size, p,
                           _stream(seed, match_id, "extra-tracks"), fps)

    inside = sum(last - first for first, last in halves)
    extra_frames = sum(track[3] - track[2] for track in extras)
    wanted = p.tracked_mean * inside - extra_frames
    scale = _tune_loss(plan, wanted, p.min_track_frames)
    person_tracks = _cut(plan, scale, p.min_track_frames)

    # Slot layout: no range of slots means anything, as in the competition format.
    layout = _stream(seed, match_id, "slot-layout")
    home = p.n_slots // 2
    if p.slot_split_spread:
        home += int(layout.integers(-p.slot_split_spread, p.slot_split_spread + 1))
    slot_team = np.full(p.n_slots, AWAY, dtype=np.int8)
    slot_team[layout.permutation(p.n_slots)[:home]] = HOME

    # Which team's slot a track gets.
    team_rng = _stream(seed, match_id, "slot-team")
    keeper = match.extras.get("slot_is_keeper", np.zeros(people, dtype=bool))
    tracks = []
    for person, start, end, partner in person_tracks:
        team = int(match.slot_team[person])
        wrong = p.keeper_reuse if keeper[person] else p.flicker_share
        if team_rng.random() < wrong:
            team = 1 - team
        tracks.append((start, end, team, partner, person))
    for kind, person, start, end, _ in extras:
        team = int(match.slot_team[person]) if person >= 0 else int(team_rng.integers(2))
        tracks.append((start, end, team, -1, -1))
    slots = _assign_slots(tracks, slot_team, _stream(seed, match_id, "slots"))

    out_xy = np.full((n, p.n_slots, 2), np.nan, dtype=np.float32)
    out_person = np.full((n, p.n_slots), -1, dtype=np.int16)
    dropped = 0
    wrong_frames = 0
    person_frames = 0
    keeper_wrong = 0
    keeper_frames = 0
    for i, (person, start, end, _) in enumerate(person_tracks):
        slot = slots[i]
        if slot < 0:
            dropped += 1
            continue
        out_xy[start:end, slot] = positions[start:end, person]
        out_person[start:end, slot] = person
        size = end - start
        person_frames += size
        off = slot_team[slot] != match.slot_team[person]
        wrong_frames += size if off else 0
        if keeper[person]:
            keeper_frames += size
            keeper_wrong += size if off else 0
    offset_index = len(person_tracks)
    for i, (kind, person, start, end, shape) in enumerate(extras):
        slot = slots[offset_index + i]
        if slot < 0:
            dropped += 1
            continue
        if kind == "duplicate":
            out_xy[start:end, slot] = positions[start:end, person] + shape.astype(np.float32)
            out_person[start:end, slot] = -2
        else:
            out_xy[start:end, slot] = shape.astype(np.float32)
            out_person[start:end, slot] = -3
    # A person with no finite noisy position (a point that left the image) is simply not seen.
    lost = ~np.isfinite(out_xy[..., 0]) | ~np.isfinite(out_xy[..., 1])
    out_xy[lost] = np.nan
    out_person[lost] = -1

    count = np.isfinite(out_xy[..., 0]).sum(axis=1)
    inside_mask = match.period > 0
    return out_xy, slot_team, out_person, {
        "view_lost_share": view_lost,
        "loss_scale": float(scale),
        "person_tracks": len(person_tracks),
        "extra_tracks": len(extras),
        "dropped_no_slot": int(dropped),
        "tracked_mean": float(count[inside_mask].mean()),
        "tracked_max": int(count.max()),
        "wrong_team_share": wrong_frames / max(1, person_frames),
        "keeper_wrong_team_share": keeper_wrong / max(1, keeper_frames),
    }


# ---------------------------------------------------------------- the whole copy

def degrade(match, seed, cfg=DegradeConfig()):
    """A degraded copy of a labelled Match, in the same schema. The same inputs give the same bytes."""
    if match.labels is None:
        raise SchemaError("degrade is for labelled training data only")
    if not np.isfinite(match.ball_xy[match.period > 0]).all():
        raise SchemaError("degrade needs a ball position on every frame of a half")

    p = _draw_parameters(cfg, seed, match.match_id)
    fps = match.fps
    n = match.n_frames
    halves = _halves(match.period)
    dead = match.labels == "D"
    live = ~dead & (match.period > 0)
    dead = dead & (match.period > 0)

    truth_xy, truth_z = unpark_ball(match, p, _stream(seed, match.match_id, "unpark"))

    camera, matrices, inverse, polygons = None, None, None, None
    if p.camera:
        camera = solve_camera()
        camera = replace(camera, focal=camera.focal * p.zoom)
        pan = _pan_path(camera, truth_xy, halves, p, _stream(seed, match.match_id, "pan"), fps)
        matrices = camera.matrices(pan)
        inverse = np.linalg.inv(matrices)
        polygons = camera.view_polygons(pan)

    # What the tracker would report for the ball if it saw it, before any visibility decision.
    if p.camera:
        height = p.parallax * truth_z
        ground = apparent_ground(truth_xy, height, camera.centre)
        pixels = project(matrices, ground)
        width, tall = camera.image_size
        with np.errstate(invalid="ignore"):
            out_of_frame = ~((pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < tall))
        noise = _ar1(_stream(seed, match.match_id, "ball-noise"), n, 2, p.ball_noise_px, p.ball_noise_tau_s * fps)
        observed = project(inverse, pixels + noise)
        margin = p.off_pitch_margin_m
        with np.errstate(invalid="ignore"):
            off_pitch = ((np.abs(ground[:, 0]) > match.pitch[0] / 2 + margin)
                         | (np.abs(ground[:, 1]) > match.pitch[1] / 2 + margin))
        forced_missing = out_of_frame | off_pitch
    else:
        noise = _ar1(_stream(seed, match.match_id, "ball-noise"), n, 2, p.ball_noise_m / math.sqrt(2.0),
                     p.ball_noise_tau_s * fps)
        observed = truth_xy + noise
        forced_missing = np.zeros(n, dtype=bool)
    forced_share_live = float(forced_missing[live].mean())

    weight = _block_weight(n, halves, p, _stream(seed, match.match_id, "blocks"), fps)
    if p.crowd_factor != 1.0:
        crowded = _crowded(truth_xy, match.players_xy, p.crowd_radius_m, p.crowd_count)
        weight = weight * np.where(live & crowded, p.crowd_factor, 1.0)

    dead_share = float(dead.sum()) / max(1, int(dead.sum() + live.sum()))
    live_budget = p.missing_share * p.live_share / max(1.0 - dead_share, 1e-6)
    air = _flights_lost(truth_z, live, halves, p, live_budget, _stream(seed, match.match_id, "flights"), fps)
    forced_missing = forced_missing | air

    seen, ball_stats = ball_visibility(forced_missing, dead, weight, halves, p,
                                       _stream(seed, match.match_id, "gaps"), fps)
    observed = observed.copy()
    observed[~seen] = np.nan
    false_ball, spikes, decoys = _false_detections(observed, seen, truth_xy, matrices, halves, match.pitch,
                                                   camera.image_size if camera else None, p,
                                                   _stream(seed, match.match_id, "false-ball"), fps)

    players_xy, slot_team, slot_person, player_stats = degrade_players(
        match, matrices, inverse, camera.image_size if camera else None, p, seed, fps)

    observed_finite = np.isfinite(observed[:, 0])
    inside = match.period > 0
    missing = inside & ~observed_finite
    statistics = {
        "seed": int(seed),
        "parameters": {k: v for k, v in vars(p).items()},
        "missing_share": float(missing[inside].mean()),
        "live_share_of_missing": float(missing[live].sum() / max(1, missing.sum())),
        "forced_missing_of_live": forced_share_live,
        "air_lost_of_live": float(air[live].mean()),
        "spikes": spikes,
        "decoys": decoys,
    }
    statistics.update(ball_stats)
    statistics.update(player_stats)

    return Match(
        source=match.source + "+degraded",
        match_id=match.match_id,
        fps=fps,
        pitch=tuple(match.pitch),
        team_names=tuple(match.team_names),
        frame=match.frame.copy(),
        period=match.period.copy(),
        ball_xy=observed.astype(np.float32),
        players_xy=players_xy,
        slot_team=slot_team,
        slot_id=np.array(["Track {}".format(i + 1) for i in range(p.n_slots)]),
        ball_z=None,
        labels=match.labels.copy(),
        forced_dead=match.forced_dead.copy(),
        view_polygon=polygons,
        extras={
            "truth_ball_xy": truth_xy.astype(np.float32),
            "truth_ball_z": truth_z.astype(np.float32),
            "false_ball": false_ball,
            "slot_person": slot_person,
            "person_team": match.slot_team.copy(),
            "person_is_keeper": match.extras.get("slot_is_keeper", np.zeros(match.n_slots, dtype=bool)).copy(),
            "degrade": statistics,
        },
    ).validate()
