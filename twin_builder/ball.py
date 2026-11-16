"""The ball: unparked while dead, blacked out in contiguous blocks, speed on a minority of frames."""
import numpy as np

from common import ar1, stream, true_runs, walk


def true_path(source, timeline, cfg):
    """Ball position in metres on every video frame, before any degradation.

    DFL freezes the ball during stoppages, which would teach a false cue, so every dead run becomes
    a slow random walk bridging where play stopped to where it restarted.
    """
    b = cfg["ball"]
    fps = timeline.fps
    length, width = (float(v) for v in source["pitch"])
    rng = stream(cfg["seed"], "ball-walk")
    limit = np.array([length / 2 + 2.0, width / 2 + 2.0], dtype=np.float32)
    path = np.zeros((timeline.n, 2), dtype=np.float32)
    mapped = timeline.src_index >= 0
    path[mapped] = source["ball_xyz"][timeline.src_index[mapped], :2]

    for _, first, last in timeline.blocks:
        live = timeline.labels[first:last + 1] != "D"
        starts, ends = true_runs(~live)
        for start, end in zip((starts + first).tolist(), (ends + first).tolist()):
            n = end - start
            wander = walk(rng, n, 1, b["dead_walk_speed_mps"], b["dead_walk_corr_s"], fps)[:, 0, :]
            before = path[start - 1] if start > first else None
            after = path[end] if end <= last else None
            if before is not None and after is not None:
                fraction = (np.arange(1, n + 1, dtype=np.float32) / (n + 1))[:, None]
                run = before + wander - fraction * wander[-1] + fraction * (after - before)
            elif after is not None:
                run = after + wander - wander[-1]
            else:
                run = before + wander
            path[start:end] = np.clip(run, -limit, limit)

    # Outside the graded blocks the "ball" is a slow wanderer, seen some of the time.
    starts, ends = true_runs(~timeline.graded)
    for start, end in zip(starts.tolist(), ends.tolist()):
        anchor = path[start - 1] if start > 0 else np.zeros(2, dtype=np.float32)
        wander = walk(rng, end - start, 1, b["ungraded_walk_speed_mps"], b["dead_walk_corr_s"], fps)[:, 0, :]
        path[start:end] = np.clip(anchor + wander, -limit, limit)
    return path


def gap_lengths(cfg, rng, fps, missing_frames):
    """Blackout lengths in frames: exact counts above 5 s from twin.toml [gaps], the rest short."""
    g = cfg["gaps"]
    counts = [g["count_over_5s"] - g["count_over_10s"], g["count_over_10s"] - g["count_over_20s"],
              g["count_over_20s"] - g["count_over_30s"], g["count_over_30s"] - g["count_over_60s"],
              g["count_over_60s"] - 1]
    bounds = [(5.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, 60.0), (60.0, g["upper_bin_s"])]
    long_gaps = []
    for count, (low, high), alpha in zip(counts, bounds, g["alpha"]):
        # Stratified quantiles of a truncated Pareto keep the configured counts exact.
        u = (np.arange(count) + rng.random(count)) / count
        tail = 1.0 - u * (1.0 - (low / high) ** alpha)
        seconds = low * tail ** (-1.0 / alpha)
        frames = np.clip(np.round(seconds * fps), low * fps + 1, high * fps).astype(np.int64)
        long_gaps.extend(frames.tolist())
    long_gaps.append(int(round(g["longest_s"] * fps)))
    used = int(np.sum(long_gaps))
    if used >= missing_frames:
        raise ValueError("gaps over 5 s use {} frames but only {} may be missing".format(used, missing_frames))

    short_gaps = []
    room = missing_frames - used
    low, high = 1, int(5 * fps)
    while room > 0:
        draw = rng.lognormal(np.log(g["short_median_s"] * fps), g["short_sigma"])
        frames = int(np.clip(round(draw), low, high))
        frames = min(frames, room)
        short_gaps.append(frames)
        room -= frames
    return long_gaps, short_gaps


def dead_targets(lengths, g, gap_dead_share, rng, fps):
    """Per-gap target for the fraction that lies in stoppages; the targets total gap_dead_share."""
    edges = np.array([5.0, 10.0, 20.0, 30.0, 60.0])
    weights = np.asarray(g["dead_weight"], dtype=np.float64)
    seconds = lengths / float(fps)
    which = np.searchsorted(edges, seconds, side="left")
    weight = weights[which]
    low, high = 0.0, 200.0
    for _ in range(60):
        scale = 0.5 * (low + high)
        share = np.sum(lengths * np.minimum(1.0, weight * scale)) / np.sum(lengths)
        if share < gap_dead_share:
            low = scale
        else:
            high = scale
    mean = np.clip(weight * scale, 0.001, 0.999)
    kappa = np.asarray(g["dead_concentration"], dtype=np.float64)[which]
    draws = rng.beta(mean * kappa, (1.0 - mean) * kappa)
    # Random draws on a handful of long gaps can miss the total, so pull them back onto it.
    want = gap_dead_share * np.sum(lengths)
    for _ in range(30):
        draws = np.clip(draws * want / np.sum(draws * lengths), 0.0, 1.0)
    return draws


def place_gaps(timeline, cfg):
    """Boolean per frame: ball missing on a graded frame. Longest gaps first, exhaustive search.

    Each gap goes where its overlap with dead frames is closest to its target, corrected by the
    running error so the total lands on gap_dead_share. Ties are broken at random.
    """
    g = cfg["gaps"]
    b = cfg["ball"]
    fps = timeline.fps
    rng = stream(cfg["seed"], "gaps")
    graded_frames = int(np.sum(timeline.graded))
    missing_frames = int(round(b["missing_share"] * graded_frames))
    long_gaps, short_gaps = gap_lengths(cfg, rng, fps, missing_frames)
    lengths = np.array(sorted(long_gaps + short_gaps, reverse=True), dtype=np.int64)
    targets = dead_targets(lengths, g, b["gap_dead_share"], rng, fps)

    dead = (timeline.labels == "D").astype(np.int64)
    cum_dead = np.concatenate(([0], np.cumsum(dead)))
    occupied = np.zeros(timeline.n, dtype=np.int64)
    margin = g["min_separation_frames"]
    error = 0.0
    placed = []
    for length, target in zip(lengths.tolist(), targets.tolist()):
        cum_occupied = np.concatenate(([0], np.cumsum(occupied)))
        candidates, overlaps = [], []
        for _, first, last in timeline.blocks:
            starts = np.arange(first, last - length + 2)
            low = np.maximum(starts - margin, first)
            high = np.minimum(starts + length + margin, last + 1)
            free = (cum_occupied[high] - cum_occupied[low]) == 0
            starts = starts[free]
            candidates.append(starts)
            overlaps.append(cum_dead[starts + length] - cum_dead[starts])
        candidates = np.concatenate(candidates)
        overlaps = np.concatenate(overlaps).astype(np.float64)
        if not len(candidates):
            raise RuntimeError("no room left for a gap of {} frames".format(length))
        desired = np.clip(target * length - error, 0.0, float(length))
        score = np.abs(overlaps - desired) / length
        near = np.flatnonzero(score <= score.min() + 0.02)
        pick = int(near[rng.integers(len(near))])
        start = int(candidates[pick])
        occupied[start:start + length] = 1
        error += overlaps[pick] - target * length
        placed.append((start, length))

    missing = occupied.astype(bool)
    return missing, {
        "gaps_total": len(lengths),
        "gaps_over_5s_planned": len(long_gaps),
        "gaps_short_planned": len(short_gaps),
        "short_gap_seconds": float(np.sum(short_gaps) / fps),
        "long_gap_seconds": float(np.sum(long_gaps) / fps),
    }


def ungraded_visible(timeline, cfg):
    """Boolean per frame: ball seen on an ungraded frame, in alternating blocks, exact share overall."""
    b = cfg["ball"]
    rng = stream(cfg["seed"], "ungraded-ball")
    frames = np.flatnonzero(~timeline.graded)
    want_on = int(round(b["ungraded_visible_share"] * len(frames)))
    mean = b["ungraded_block_mean_s"] * timeline.fps
    n_blocks = int(len(frames) / mean) + 2
    on = rng.exponential(1.0, n_blocks)
    off = rng.exponential(1.0, n_blocks)
    on = np.round(on / on.sum() * want_on).astype(np.int64)
    off = np.round(off / off.sum() * (len(frames) - want_on)).astype(np.int64)
    flags = []
    for on_length, off_length in zip(on.tolist(), off.tolist()):
        flags.extend([True] * on_length + [False] * off_length)
    flags = (flags + [False] * len(frames))[:len(frames)]
    flags = np.array(flags, dtype=bool)
    # Rounding can leave a few frames over or under; settle them at the tail of the last blocks.
    delta = want_on - int(flags.sum())
    if delta > 0:
        flags[np.flatnonzero(~flags)[:delta]] = True
    elif delta < 0:
        flags[np.flatnonzero(flags)[delta:]] = False
    out = np.zeros(timeline.n, dtype=bool)
    out[frames] = flags
    return out


def speed_filled(timeline, cfg):
    """Boolean per frame: ball speed written, in short runs, for about speed_fill_share of frames."""
    b = cfg["ball"]
    rng = stream(cfg["seed"], "ball-speed")
    mean_on = float(b["speed_block_mean_frames"])
    mean_off = mean_on * (1.0 - b["speed_fill_share"]) / b["speed_fill_share"]
    out = np.zeros(timeline.n, dtype=bool)
    position = int(rng.exponential(mean_off))
    while position < timeline.n:
        on = max(1, int(round(rng.exponential(mean_on))))
        out[position:position + on] = True
        position += on + max(1, int(round(rng.exponential(mean_off))))
    return out


def build(path, timeline, cfg):
    """(x, y, speed) normalised and written per frame, NaN where missing, plus realised statistics."""
    b = cfg["ball"]
    fps = timeline.fps
    norm_x, norm_y = cfg["pitch"]["normalise_m"]
    rng = stream(cfg["seed"], "ball-noise")

    missing, gap_stats = place_gaps(timeline, cfg)
    visible = (timeline.graded & ~missing) | ungraded_visible(timeline, cfg)

    axis_sigma = b["noise_m"] / np.sqrt(2.0)
    noisy = path + ar1(rng, timeline.n, 2, axis_sigma, b["noise_corr_s"] * fps)
    x = np.clip(0.5 + noisy[:, 0] / norm_x, 0.0, 1.0)
    y = np.clip(0.5 - noisy[:, 1] / norm_y, 0.0, 1.0)

    # Speed from the true path over +/- 2 frames.
    low = np.maximum(np.arange(timeline.n) - 2, 0)
    high = np.minimum(np.arange(timeline.n) + 2, timeline.n - 1)
    delta = path[high] - path[low]
    speed = np.hypot(delta[:, 0], delta[:, 1]) / ((high - low) / float(fps))
    filled = speed_filled(timeline, cfg) & visible
    speed = np.where(filled, speed, np.nan).astype(np.float32)

    x = np.where(visible, x, np.nan).astype(np.float32)
    y = np.where(visible, y, np.nan).astype(np.float32)

    graded = timeline.graded
    dead = timeline.labels == "D"
    stats = dict(gap_stats)
    stats.update({
        "visible_all": float(np.mean(visible)),
        "visible_graded": float(np.mean(visible[graded])),
        "visible_ungraded": float(np.mean(visible[~graded])),
        "missing_graded_frames": int(np.sum(missing)),
        "gap_dead_share_realised": float(np.sum(missing & dead) / max(int(np.sum(missing)), 1)),
        "speed_fill_share_realised": float(np.mean(filled[visible])),
    })
    return x, y, speed, stats
