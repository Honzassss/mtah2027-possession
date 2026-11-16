"""Turn known people into anonymous tracker output.

Each player's trajectory is cut into tracklets at random times and at occlusions, with short losses
between them, and each tracklet goes to a random free slot of its team, so a slot does not follow a
person. Duplicates and position noise are added on top. Loss rates are rescaled in a closed loop
until the mean number of tracked objects over graded frames hits the target.
"""
import heapq

import numpy as np

from common import ar1, stream, true_runs, walk


def build_base(source, timeline, cfg):
    """Metre positions per roster player on every video frame, NaN where not on the pitch.

    Pre-kickoff and padding frames get a slow walk out of the first or last real position.
    """
    n_players = source["players_xy"].shape[1]
    base = np.full((timeline.n, n_players, 2), np.nan, dtype=np.float32)
    mapped = timeline.src_index >= 0
    base[mapped] = source["players_xy"][timeline.src_index[mapped]]

    p = cfg["players"]
    length, width = (float(v) for v in source["pitch"])
    limit = np.array([length / 2 + 1.0, width / 2 + 1.0], dtype=np.float32)
    rng = stream(cfg["seed"], "base-walk")
    for half, first, last in timeline.blocks:
        video = np.flatnonzero((timeline.half == half) & mapped)
        head, tail = int(video[0]), int(video[-1])
        if head > first:
            who = np.flatnonzero(np.isfinite(base[head, :, 0]))
            offset = walk(rng, head - first, len(who), p["pad_walk_speed_mps"], p["pad_walk_corr_s"], timeline.fps)
            # offset[k] is k + 1 frames before the first real frame
            for k in range(head - first):
                base[head - 1 - k, who] = np.clip(base[head, who] + offset[k], -limit, limit)
        if tail < last:
            who = np.flatnonzero(np.isfinite(base[tail, :, 0]))
            offset = walk(rng, last - tail, len(who), p["pad_walk_speed_mps"], p["pad_walk_corr_s"], timeline.fps)
            base[tail + 1:last + 1][:, who] = np.clip(base[tail, who] + offset, -limit, limit)
    return base


def occlusion_events(base, timeline, cfg):
    """{player: sorted frames} where that player loses out in an occlusion that breaks a tracklet."""
    p = cfg["players"]
    rng = stream(cfg["seed"], "occlusion")
    sampled = np.flatnonzero(timeline.graded)[::5]
    xy = base[sampled]
    n_players = xy.shape[1]
    limit = p["occlusion_radius_m"] ** 2
    events = {player: [] for player in range(n_players)}
    total = 0
    for i in range(n_players):
        if not np.isfinite(xy[:, i, 0]).any():
            continue
        for j in range(i + 1, n_players):
            if not np.isfinite(xy[:, j, 0]).any():
                continue
            delta = xy[:, i] - xy[:, j]
            with np.errstate(invalid="ignore"):
                close = (delta[:, 0] ** 2 + delta[:, 1] ** 2) < limit
            starts = np.flatnonzero(close[1:] & ~close[:-1]) + 1
            total += len(starts)
            keep = rng.random(len(starts)) < p["occlusion_break_prob"]
            loser = rng.random(len(starts)) < 0.5
            for start, kept, first in zip(sampled[starts].tolist(), keep.tolist(), loser.tolist()):
                if kept:
                    events[i if first else j].append(start)
    for player in events:
        events[player].sort()
    return events, total


def cut_tracklets(present, occlusions, lost_scale, cfg, fps):
    """[(player, start, end)] for everyone. The same lost_scale always gives the same cuts."""
    p = cfg["players"]
    out = []
    for player in range(present.shape[1]):
        rng = stream(cfg["seed"], "tracklets-{}".format(player))
        times = np.asarray(occlusions[player], dtype=np.int64)
        starts, ends = true_runs(present[:, player])
        for a, b in zip(starts.tolist(), ends.tolist()):
            count = int((b - a) / (p["tracklet_mean_s"] * fps) * 2.5) + 8
            random_times = a + np.cumsum(rng.exponential(p["tracklet_mean_s"] * fps, count))
            draws = rng.random(count)
            lost_random = rng.exponential(p["lost_mean_s"] * fps, count)
            in_run = times[(times > a) & (times < b)]
            lost_occlusion = rng.exponential(p["occlusion_lost_mean_s"] * fps, len(in_run))
            merged = []
            for time, draw, lost in zip(random_times.tolist(), draws.tolist(), lost_random.tolist()):
                if time >= b:
                    break
                merged.append((int(time), 0.0 if draw < p["switch_without_loss_share"] else lost * lost_scale))
            merged.extend(zip(in_run.tolist(), (lost_occlusion * min(lost_scale, 1.0)).tolist()))
            merged.sort()
            current = a
            for time, lost in merged:
                if time <= current:
                    continue
                out.append((player, current, time))
                current = time + int(round(lost))
                if current >= b:
                    break
            if current < b:
                out.append((player, current, b))
    return out


def make_duplicates(present, timeline, cfg):
    """[(player, start, end)] short extra objects that shadow someone already on the pitch."""
    p = cfg["players"]
    rng = stream(cfg["seed"], "duplicates")
    graded = np.flatnonzero(timeline.graded)
    count = rng.poisson(p["duplicate_rate_per_min"] * len(graded) / (timeline.fps * 60.0))
    out = []
    for frame in np.sort(rng.choice(graded, size=count, replace=False)).tolist():
        here = np.flatnonzero(present[frame])
        if not len(here):
            continue
        player = int(here[rng.integers(len(here))])
        length = max(1, int(rng.exponential(p["duplicate_mean_s"] * timeline.fps)))
        half = int(timeline.half[frame])
        last = next(block[2] for block in timeline.blocks if block[0] == half)
        end = min(frame + length, last + 1)
        stops = np.flatnonzero(~present[frame:end, player])
        if len(stops):
            end = frame + int(stops[0])
        if end > frame:
            out.append((player, frame, end))
    return out


def tune_loss(present, occlusions, duplicates, timeline, cfg):
    """Rescale the random-loss length until mean tracked objects over graded frames hits the target."""
    target = cfg["players"]["target_mean_tracked"]
    graded_frames = float(np.sum(timeline.graded))
    duplicate_frames = sum(end - start for _, start, end in duplicates)

    def measure(scale):
        cuts = cut_tracklets(present, occlusions, scale, cfg, timeline.fps)
        return cuts, (sum(end - start for _, start, end in cuts) + duplicate_frames) / graded_frames

    scale = 1.0
    cuts, mean = measure(scale)
    # A loss-length scale moves mean tracked roughly linearly, so a few secant steps are plenty.
    previous_scale, previous_mean = None, None
    for _ in range(8):
        if abs(mean - target) < 0.01:
            break
        if previous_scale is None:
            next_scale = scale * max(0.1, (22.0 - target) / max(22.0 - mean, 0.05))
        else:
            slope = (mean - previous_mean) / (scale - previous_scale)
            next_scale = scale + (target - mean) / slope if slope else scale * 1.1
        previous_scale, previous_mean = scale, mean
        scale = max(0.01, next_scale)
        cuts, mean = measure(scale)
    return cuts, scale, mean


def assign_slots(tracks, cfg):
    """Slot per track, 0-based. tracks: [(start, end, team, order)], handed out in time order.

    A track takes a random free slot of its own team, home slots first in the layout. Running out of
    slots would mean more than 23 simultaneous objects on one team, which is a bug, so it stops.
    """
    n_home, n_away = cfg["slots"]["home"], cfg["slots"]["away"]
    rng = stream(cfg["seed"], "slots")
    free = {0: list(range(n_home)), 1: list(range(n_home, n_home + n_away))}
    busy = []
    slots = [None] * len(tracks)
    order = sorted(range(len(tracks)), key=lambda i: tracks[i][:2] + (tracks[i][3],))
    for i in order:
        start, end, team, _ = tracks[i]
        while busy and busy[0][0] <= start:
            _, released, released_team = heapq.heappop(busy)
            free[released_team].append(released)
        pool = free[team]
        if not pool:
            raise RuntimeError("team {} ran out of slots at frame {}".format(team, start))
        slot = pool.pop(int(rng.integers(len(pool))))
        heapq.heappush(busy, (end, slot, team))
        slots[i] = slot
    return slots


def player_speeds(track_xy, timeline, window):
    """m/s per player per frame from the noisy track, over +/- window frames inside each block."""
    speed = np.full(track_xy.shape[:2], np.nan, dtype=np.float32)
    for _, first, last in timeline.blocks:
        frames = np.arange(first, last + 1)
        low = np.maximum(frames - window, first)
        high = np.minimum(frames + window, last)
        delta = track_xy[high] - track_xy[low]
        seconds = ((high - low) / float(timeline.fps))[:, None]
        speed[frames] = np.hypot(delta[:, :, 0], delta[:, :, 1]) / seconds
    return speed


def build_slots(source, base, timeline, cfg):
    """Everything the writer needs: per-slot x, y (normalised) and speed, plus realised statistics."""
    p = cfg["players"]
    fps = timeline.fps
    norm_x, norm_y = cfg["pitch"]["normalise_m"]
    n_slots = cfg["slots"]["home"] + cfg["slots"]["away"]
    team = source["player_team"]

    present = np.isfinite(base[:, :, 0]) & timeline.graded[:, None]
    occlusions, n_occlusions = occlusion_events(base, timeline, cfg)
    duplicates = make_duplicates(present, timeline, cfg)
    cuts, loss_scale, tuned_mean = tune_loss(present, occlusions, duplicates, timeline, cfg)

    # Position noise per person, in metres, so a re-acquired player keeps a continuous error.
    rng_noise = stream(cfg["seed"], "position-noise")
    axis_sigma = p["position_noise_m"] / np.sqrt(2.0)
    n_players = base.shape[1]
    noise = ar1(rng_noise, timeline.n, 2 * n_players, axis_sigma, p["position_noise_corr_s"] * fps)
    noise = noise.reshape(timeline.n, n_players, 2)
    noisy = base + noise
    speeds = player_speeds(noisy, timeline, p["speed_window_frames"])
    # A player who only just came on has no window on one side; call that standing still.
    speeds[~np.isfinite(speeds) & present] = 0.0

    # Sparse stray objects across ungraded frames: almost no tracking, but not literally none.
    rng_stray = stream(cfg["seed"], "stray")
    ungraded = np.flatnonzero(~timeline.graded)
    mean_frames = p["ungraded_track_mean_s"] * fps
    n_stray = rng_stray.poisson(len(ungraded) * p["ungraded_mean_objects"] / mean_frames)
    stray = []
    for frame in np.sort(rng_stray.integers(0, len(ungraded), n_stray)).tolist():
        start = int(ungraded[frame])
        end = start + max(1, int(rng_stray.exponential(mean_frames)))
        stops = np.flatnonzero(timeline.graded[start:end])
        if len(stops):
            end = start + int(stops[0])
        stray.append((start, min(end, timeline.n), int(rng_stray.integers(2))))

    tracks = []
    for order, (player, start, end) in enumerate(cuts):
        tracks.append((start, end, int(team[player]), order))
    first_duplicate = len(tracks)
    for order, (player, start, end) in enumerate(duplicates):
        tracks.append((start, end, int(team[player]), first_duplicate + order))
    first_stray = len(tracks)
    for order, (start, end, side) in enumerate(stray):
        tracks.append((start, end, side, first_stray + order))
    slots = assign_slots(tracks, cfg)

    slot_x = np.full((timeline.n, n_slots), np.nan, dtype=np.float32)
    slot_y = np.full((timeline.n, n_slots), np.nan, dtype=np.float32)
    slot_s = np.full((timeline.n, n_slots), np.nan, dtype=np.float32)

    def put(slot, start, end, xy, speed):
        slot_x[start:end, slot] = np.clip(0.5 + xy[:, 0] / norm_x, 0.0, 1.0)
        slot_y[start:end, slot] = np.clip(0.5 - xy[:, 1] / norm_y, 0.0, 1.0)
        slot_s[start:end, slot] = speed

    for index, (player, start, end) in enumerate(cuts):
        put(slots[index], start, end, noisy[start:end, player], speeds[start:end, player])

    rng_dup = stream(cfg["seed"], "duplicate-offset")
    for index, (player, start, end) in enumerate(duplicates):
        angle = rng_dup.uniform(0.0, 2.0 * np.pi)
        offset = p["duplicate_offset_m"] * np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        own = ar1(rng_dup, end - start, 2, axis_sigma, p["position_noise_corr_s"] * fps)
        put(slots[first_duplicate + index], start, end, base[start:end, player] + offset + own,
            speeds[start:end, player])

    rng_place = stream(cfg["seed"], "stray-place")
    for index, (start, end, _) in enumerate(stray):
        anchor = np.array([rng_place.uniform(-52.0, 52.0), rng_place.uniform(-34.0, 34.0)], dtype=np.float32)
        path = anchor + walk(rng_place, end - start, 1, 0.8, 3.0, fps)[:, 0, :]
        put(slots[first_stray + index], start, end, path, np.full(end - start, 0.8, dtype=np.float32))

    tracked = np.sum(np.isfinite(slot_x), axis=1)
    graded = timeline.graded
    live = graded & (timeline.labels != "D")
    lengths = np.array([end - start for _, start, end in cuts], dtype=np.float64) / fps
    people = {slot: set() for slot in range(n_slots)}
    for index, (player, _, _) in enumerate(cuts):
        people[slots[index]].add(player)
    people_per_slot = [len(people[slot]) for slot in range(n_slots)]
    used = np.isfinite(base[:, :, 0]) & graded[:, None]
    stats = {
        "mean_tracked_graded": float(np.mean(tracked[graded])),
        "mean_tracked_live": float(np.mean(tracked[live])),
        "mean_tracked_ungraded": float(np.mean(tracked[~graded])),
        "share_graded_over_22": float(np.mean(tracked[graded] > 22)),
        "max_tracked": int(np.max(tracked)),
        "occlusions_seen": int(n_occlusions),
        "tracklets": len(cuts),
        "tracklet_median_s": float(np.median(lengths)),
        "tracklet_mean_s": float(np.mean(lengths)),
        "duplicates": len(duplicates),
        "strays": len(stray),
        "loss_scale": float(loss_scale),
        "tuned_mean_tracked": float(tuned_mean),
        "people_per_slot_mean": float(np.mean(people_per_slot)),
        "position_noise_rms_m": float(np.sqrt(np.mean(np.sum(noise[used] ** 2, axis=1)))),
        "player_pace_live_mps": float(np.nanmean(slot_s[live])),
        "player_pace_dead_mps": float(np.nanmean(slot_s[graded & (timeline.labels == "D")])),
    }
    return slot_x, slot_y, slot_s, stats
