"""Label-free measurements behind the scripts the operator runs on the competition match.

    test 7           what the ball gaps hide: player speed by gap length, visibility against movement
                     per 5 minutes, what the camera shows during a gap          gap_table, visibility_windows, camera_by_gap
    test 8           integrity: the baseline's nearest-player rule from our parser, the kickoffs on
                     the published frames, attack direction, homography rows   nearest_player_codes, kickoff_check, ...
    test 9           camera aim against the visible ball                        aim_measurements
    parallax check   do fast ball steps run along the camera ray, and does the
                     ball land beyond the far touchline                         parallax_measurements

Every function takes a Match in the schema, raw or cleaned, so the same code measures the competition
match, the twin and degraded IDSSE copies, which give the reference values (simulated_copies). With
labels (copies only) some functions also return the truth they are compared with.

Nothing here prints, and nothing returns a frame index or a row of the data: results are counts,
durations, shares, medians and ratios. scripts/match_*.py decide what reaches the screen (PASS/FAIL
only) and what goes into the operator's report.
"""
import glob
import os

import numpy as np

from .camera import DATA_MD_ROW, project, solve_camera
from .clean import attack_direction, clean
from .epts import COMPETITION_DEAD_BLOCKS, COMPETITION_HALVES, IMAGE_SIZE
from .schema import AWAY, HOME, FormatError, SchemaError

# Published frames (DATA.md "Which frames are graded", FAQ.md): the first half is graded from its
# kickoff, the second kicks off on the frame after the published dead block.
KICKOFFS = ((1, COMPETITION_HALVES[0][1]), (2, COMPETITION_DEAD_BLOCKS[0][1] + 1))

GAP_BUCKETS_S = ((0.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, 60.0), (60.0, np.inf))
LONG_GAP_S = 10.0
MIN_TRACKED = 6             # fewer tracked players on a frame and its mean speed is not used
LIVE_BALL_SPEED = 2.0       # m/s: "confident live" is a ball that is seen and moving at least this fast
WINDOW_S = 300.0            # test 7: visibility per 5 minutes
WINDOW_MIN_S = 60.0         # a shorter tail joins the window before it
AIM_SPEED_HALF_S = 0.24     # half-width of the central difference for the speed of the aim point
EDGE_SHARE = 0.1            # of the image width: a ball this close to the left or right edge is leaving it
FAST_STEP_MPS = 40.0        # pipeline/clean.py gate
BEYOND_LINE_M = 2.0         # test 3's signature: the ball this far beyond a touchline
RAY_ALIGNED_COS = 0.9       # |cos| above this counts as along the ray; 0.287 of random directions do
RANDOM_MEDIAN_COS = float(np.sqrt(0.5))
AIM_GAP_MIN_S = 1.0         # test 9 looks at gaps at least this long
AIM_GAP_CLASSES_S = ((1.0, 5.0), (5.0, 20.0), (20.0, np.inf))
LAGS_S = tuple(round(k / 5.0, 1) for k in range(-10, 16))     # -2.0 .. +3.0 s in 0.2 s steps
NEAR_TIE_M = 1e-3           # nearest home and away player this close: float32 storage may flip the answer
SPOT_M = 2.0                # kickoff: the ball this close to the centre spot on some frame near the published one
SPOT_BEFORE = 12            # frames before the published frame that count for that (the ball waits on the spot)
LEAVE_M = 2.0               # a ball seen after the kickoff and never this far from the spot within
LEAVE_S = 2.0               # LEAVE_S has not been kicked
KICKOFF_AFTER = 3           # frames after the published frame for the ball and the formation
FORMATION_SHARE = 0.9       # of the tracked players stand in their own half at the kickoff
FORMATION_MARGIN_M = 1.0    # a player sprinting from the line covers under 1 m in KICKOFF_AFTER frames
SHIFTS = (-25, -5, -1, 0, 1, 5, 25)
ALIGNED_FLOOR = 0.95        # share of the seen ball inside its own frame's image. A camera that lags
                            # the ball shows it slightly more often in later images, so no shift test


# ---------------------------------------------------------------- small helpers

def play_mask(match):
    """Frames inside a half and outside every published dead block."""
    return (match.period > 0) & ~match.forced_dead


def masked_mean(values, mask):
    """(mean of the finite values under mask, how many there were)."""
    pick = mask & np.isfinite(values)
    count = int(pick.sum())
    return (float(values[pick].mean()) if count else float("nan")), count


def median(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else float("nan")


def quantile(values, q):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, q)) if len(values) else float("nan")


def share(mask, within):
    total = int(within.sum())
    return float((mask & within).sum() / total) if total else float("nan")


def correlation(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def rank_correlation(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    ranks = [np.argsort(np.argsort(v[ok])).astype(np.float64) for v in (a, b)]
    return correlation(*ranks) if ok.sum() >= 3 else float("nan")


def shifted(values, period, shift):
    """values[t + shift] on row t, NaN where t + shift leaves the array or the half of t."""
    n = len(values)
    out = np.full(values.shape, np.nan, dtype=np.float64)
    rows = np.arange(n)
    source = rows + shift
    ok = (source >= 0) & (source < n) & (period > 0)
    ok[ok] = period[source[ok]] == period[rows[ok]]
    out[ok] = values[source[ok]]
    return out


def runs(mask):
    """(starts, ends) of the True runs, ends exclusive."""
    edges = np.diff(np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


# ---------------------------------------------------------------- geometry

def metres_to_normalised(pitch):
    """3x3 from schema metres to the bundle's normalised pitch coordinates (pipeline/epts.py)."""
    length, width = pitch
    return np.array([[1.0 / length, 0.0, 0.5], [0.0, -1.0 / width, 0.5], [0.0, 0.0, 1.0]])


def metres_matrices(match):
    """(F, 3, 3) from pitch metres to pixels, from the bundle's homography. NaN where it is not valid."""
    normalised = match.extras.get("homography")
    if normalised is None:
        raise SchemaError("the match carries no homography: load it with with_homography=True")
    out = np.asarray(normalised, dtype=np.float64) @ metres_to_normalised(match.pitch)
    valid = match.extras.get("homography_valid")
    if valid is not None:
        out[~np.asarray(valid, dtype=bool)] = np.nan
    return out


def matrices_from_polygon(polygon, image_size=IMAGE_SIZE):
    """(F, 3, 3) from pitch metres to pixels that put the view polygon's corners on the image corners.

    Match.view_polygon holds the ground points of the image corners TL, BL, BR, TR, and four
    correspondences fix a homography. Degraded copies carry the polygon but not the matrices. NaN
    where a corner is missing or the corners are degenerate.
    """
    width, height = image_size
    pixels = ((0.0, 0.0), (0.0, height), (width, height), (width, 0.0))
    ground = np.asarray(polygon, dtype=np.float64)
    n = len(ground)
    a = np.zeros((n, 8, 8))
    b = np.zeros((n, 8))
    for k, (u, v) in enumerate(pixels):
        x, y = ground[:, k, 0], ground[:, k, 1]
        a[:, 2 * k, 0], a[:, 2 * k, 1], a[:, 2 * k, 2] = x, y, 1.0
        a[:, 2 * k, 6], a[:, 2 * k, 7], b[:, 2 * k] = -u * x, -u * y, u
        a[:, 2 * k + 1, 3], a[:, 2 * k + 1, 4], a[:, 2 * k + 1, 5] = x, y, 1.0
        a[:, 2 * k + 1, 6], a[:, 2 * k + 1, 7], b[:, 2 * k + 1] = -v * x, -v * y, v
    out = np.full((n, 3, 3), np.nan)
    ok = np.isfinite(a).all(axis=(1, 2))
    ok[ok] = np.abs(np.linalg.det(a[ok])) > 1e-9
    if ok.any():
        h = np.linalg.solve(a[ok], b[ok][..., None])[..., 0]
        out[ok] = np.concatenate((h, np.ones((len(h), 1))), axis=1).reshape(-1, 3, 3)
    return out


def safe_inverse(matrices):
    out = np.full(matrices.shape, np.nan)
    ok = np.isfinite(matrices).all(axis=(1, 2))
    ok[ok] = np.abs(np.linalg.det(matrices[ok])) > 1e-12
    out[ok] = np.linalg.inv(matrices[ok])
    return out


def ground_of_pixel(inverse, u, v):
    """(F, 2) ground point of the pixel (u, v) on every frame, NaN above the horizon."""
    g = inverse @ np.array([u, v, 1.0])
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = g[:, :2] / g[:, 2:3]
    xy[~(g[:, 2] > 0)] = np.nan
    return xy


def x_on_line(p, q, y):
    """x where the ground line through p and q crosses depth y, per row."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (y - p[:, 1]) / (q[:, 1] - p[:, 1])
    return p[:, 0] + t * (q[:, 0] - p[:, 0])


WELL_CONDITIONED = 0.3      # |h20^2 - h21^2| / (h20^2 + h21^2): the equal-norm condition holds the focal length


def _principal_x(h, cy, image_size):
    """Per frame, solve_camera's search: the horizontal principal point at which the equal-norm focal
    length makes the first two columns orthogonal. Only meaningful where WELL_CONDITIONED holds."""
    n = len(h)

    def fit(cx):
        centred = h.copy()
        centred[:, 0] = h[:, 0] - cx[:, None] * h[:, 2]
        centred[:, 1] = h[:, 1] - cy * h[:, 2]
        first, second = centred[:, :, 0], centred[:, :, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            focal = np.sqrt(-((first[:, :2] ** 2).sum(axis=1) - (second[:, :2] ** 2).sum(axis=1))
                            / (first[:, 2] ** 2 - second[:, 2] ** 2))
            scaled = centred.copy()
            scaled[:, :2] /= focal[:, None, None]
            c1, c2 = scaled[:, :, 0], scaled[:, :, 1]
            cosine = (c1 * c2).sum(axis=1) / (np.linalg.norm(c1, axis=1) * np.linalg.norm(c2, axis=1))
        return scaled, cosine

    cx_a = np.full(n, image_size[0] / 2.0)
    cx_b = cx_a - 40.0
    _, res_a = fit(cx_a)
    _, res_b = fit(cx_b)
    for _ in range(12):
        denominator = res_b - res_a
        step = np.abs(denominator) >= 1e-15
        with np.errstate(divide="ignore", invalid="ignore"):
            new_b = np.where(step, cx_b - res_b * (cx_b - cx_a) / np.where(step, denominator, 1.0), cx_b)
        cx_a, res_a = np.where(step, cx_b, cx_a), np.where(step, res_b, res_a)
        cx_b = new_b
        _, res_b = fit(cx_b)
    return cx_b


def camera_centres(matrices, image_size=IMAGE_SIZE, principal_x=None):
    """(F, 3) camera centre in metres per frame, from its homography, as pipeline/camera.solve_camera
    solves one row: square pixels, no skew, the vertical principal point at the image centre.

    Two steps. The horizontal principal point does not move, so it is solve_camera's per-frame answer
    taken as a median over the frames where its equal-norm condition is well conditioned (or passed
    in). Then each frame's focal length comes from both rotation conditions, equal norms and
    orthogonal columns, by least squares: the equal-norm condition alone, which solve_camera uses,
    degenerates where the camera looks about 45 degrees across the pitch. Zoom does not move the
    answer. NaN where a frame has no solution. Returns (centres, principal_x).
    """
    h = np.asarray(matrices, dtype=np.float64)
    ok = np.isfinite(h).all(axis=(1, 2))
    h = np.where(ok[:, None, None], h, np.eye(3))
    cy = image_size[1] / 2.0
    if principal_x is None:
        with np.errstate(divide="ignore", invalid="ignore"):
            conditioning = np.abs(h[:, 2, 0] ** 2 - h[:, 2, 1] ** 2) / (h[:, 2, 0] ** 2 + h[:, 2, 1] ** 2)
        good = ok & (conditioning > WELL_CONDITIONED)
        found = _principal_x(h[good], cy, image_size) if good.any() else np.zeros(0)
        found = found[np.isfinite(found)]
        principal_x = float(np.median(found)) if len(found) else image_size[0] / 2.0

    centred = h.copy()
    centred[:, 0] = h[:, 0] - principal_x * h[:, 2]
    centred[:, 1] = h[:, 1] - cy * h[:, 2]
    c1, c2 = centred[:, :, 0], centred[:, :, 1]
    a1 = (c1[:, :2] ** 2).sum(axis=1) - (c2[:, :2] ** 2).sum(axis=1)
    b1 = c1[:, 2] ** 2 - c2[:, 2] ** 2
    a2 = (c1[:, :2] * c2[:, :2]).sum(axis=1)
    b2 = c1[:, 2] * c2[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        inverse_focal_sq = -(a1 * b1 + a2 * b2) / (a1 ** 2 + a2 ** 2)
        focal = 1.0 / np.sqrt(inverse_focal_sq)          # NaN where the conditions want a negative square
        scaled = centred.copy()
        scaled[:, :2] /= focal[:, None, None]
        scale = 2.0 / (np.linalg.norm(scaled[:, :, 0], axis=1) + np.linalg.norm(scaled[:, :, 1], axis=1))
    scale = np.where(scaled[:, 2, 2] * scale < 0, -scale, scale)
    r1 = scaled[:, :, 0] * scale[:, None]
    r2 = scaled[:, :, 1] * scale[:, None]
    t = scaled[:, :, 2] * scale[:, None]
    rotation = np.stack((r1, r2, np.cross(r1, r2)), axis=2)
    centre = -np.einsum("nji,nj->ni", rotation, t)
    centre[~ok | ~np.isfinite(centre).all(axis=1)] = np.nan
    return centre, principal_x


# ---------------------------------------------------------------- test 7: what the gaps hide

def bucket_name(i):
    low, high = GAP_BUCKETS_S[i]
    if low == 0:
        return "under {:g} s".format(high)
    if not np.isfinite(high):
        return "{:g} s and over".format(low)
    return "{:g}-{:g} s".format(low, high)


def bucket_index(seconds):
    edges = np.array([high for _, high in GAP_BUCKETS_S[:-1]])
    return np.searchsorted(edges, seconds, side="right")


def ball_gaps(period, forced_dead, seen):
    """Runs without a ball inside one half; a run never crosses a half or the edge of a dead block.

    {start, end (exclusive), half, block}: block is True for a run inside a published dead block.
    """
    key = np.where((period > 0) & ~seen, period.astype(np.int16) * 2 + forced_dead.astype(np.int16), -1)
    change = np.flatnonzero(np.diff(key) != 0) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(key)]))
    keep = key[starts] >= 0
    starts, ends, key = starts[keep], ends[keep], key[starts[keep]]
    return {"start": starts, "end": ends, "half": key // 2, "block": (key % 2).astype(bool)}


IN_BLOCK = -2


def gap_bucket_of_frame(gaps, n, fps):
    """(F,) bucket of the gap a frame lies in, -1 off every gap, IN_BLOCK inside a dead block's gap."""
    out = np.full(n, -1, dtype=np.int8)
    index = bucket_index((gaps["end"] - gaps["start"]) / fps)
    for start, end, b, block in zip(gaps["start"], gaps["end"], index, gaps["block"]):
        out[start:end] = IN_BLOCK if block else b
    return out


def player_speed(cleaned):
    """(F,) mean speed of the tracked players in m/s, from the cleaned velocities; NaN with few tracked."""
    speed = np.hypot(cleaned.players_vel[..., 0], cleaned.players_vel[..., 1]).astype(np.float64)
    finite = np.isfinite(speed)
    count = finite.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(finite, speed, 0.0).sum(axis=1) / count
    mean[count < MIN_TRACKED] = np.nan
    return mean


def confident_live(cleaned):
    """Frames whose ball is seen and moving: almost always live play."""
    speed = np.hypot(cleaned.ball_vel[:, 0], cleaned.ball_vel[:, 1])
    with np.errstate(invalid="ignore"):
        return play_mask(cleaned) & np.isfinite(cleaned.ball_xy[:, 0]) & (speed >= LIVE_BALL_SPEED)


def gap_table(match, cleaned, labels=None):
    """Ball gaps by length with the players' speed in them, against speeds where the ball is seen.

    match is the raw Match (its ball defines the gaps), cleaned is clean(match) (speeds). Each bucket's
    ratio is its gap speed over the speed on confident-live frames. With labels (a simulated copy),
    each bucket also carries the truth: the live share of its gap frames and the ratios of its live and
    its dead frames.
    """
    fps, n = match.fps, match.n_frames
    seen = np.isfinite(match.ball_xy[:, 0])
    play = play_mask(match)
    block = match.forced_dead & (match.period > 0)
    gaps = ball_gaps(match.period, match.forced_dead, seen)
    bucket = gap_bucket_of_frame(gaps, n, fps)
    index = bucket_index((gaps["end"] - gaps["start"]) / fps)
    speed = player_speed(cleaned)
    conf_speed, conf_frames = masked_mean(speed, confident_live(cleaned))
    missing = play & ~seen
    live = None if labels is None else np.isin(labels, ("H", "A"))
    out = {
        "play_s": float(play.sum() / fps),
        "seen_share": share(seen, play),
        "gaps": int((~gaps["block"]).sum()),
        "missing_s": float(missing.sum() / fps),
        "confident_live_speed": conf_speed,
        "confident_live_s": conf_frames / fps,
        "seen_speed": masked_mean(speed, play & seen)[0],
        "block_s": float(block.sum() / fps),
        "block_seen_share": share(seen, block),
        "block_speed": masked_mean(speed, block)[0],
        "buckets": [],
    }
    with np.errstate(divide="ignore", invalid="ignore"):
        out["seen_ratio"] = out["seen_speed"] / conf_speed
        out["block_ratio"] = out["block_speed"] / conf_speed
    for b in range(len(GAP_BUCKETS_S)):
        frames = bucket == b
        gap_speed, speed_frames = masked_mean(speed, frames)
        row = {"name": bucket_name(b), "gaps": int(((index == b) & ~gaps["block"]).sum()),
               "seconds": float(frames.sum() / fps), "share_of_missing": share(frames, missing),
               "speed": gap_speed, "speed_frames": speed_frames, "ratio": gap_speed / conf_speed}
        if live is not None:
            live_speed, live_frames = masked_mean(speed, frames & live)
            dead_speed, dead_frames = masked_mean(speed, frames & ~live)
            row.update(live_frames=live_frames, dead_frames=dead_frames,
                       live_share=live_frames / max(1, live_frames + dead_frames),
                       live_ratio=live_speed / conf_speed, dead_ratio=dead_speed / conf_speed)
        out["buckets"].append(row)
    if live is not None:
        out["live_ratio"] = masked_mean(speed, play & live)[0] / conf_speed
        out["dead_ratio"] = masked_mean(speed, play & ~live)[0] / conf_speed
    return out


def gap_references(tables):
    """Per bucket (live ratio, dead ratio), pooled over simulated copies' gap tables, weighted by frames."""
    refs = []
    for b in range(len(GAP_BUCKETS_S)):
        pair = []
        for kind in ("live", "dead"):
            weights = np.array([t["buckets"][b][kind + "_frames"] for t in tables], dtype=np.float64)
            ratios = np.array([t["buckets"][b][kind + "_ratio"] for t in tables], dtype=np.float64)
            ok = np.isfinite(ratios) & (weights > 0)
            pair.append(float(np.average(ratios[ok], weights=weights[ok])) if ok.any() else float("nan"))
        refs.append(tuple(pair))
    return refs


def unmix(ratio, live_ratio, dead_ratio):
    """Live share p with ratio = p * live_ratio + (1 - p) * dead_ratio, clipped to [0, 1]."""
    if not all(np.isfinite(v) for v in (ratio, live_ratio, dead_ratio)) or abs(live_ratio - dead_ratio) < 1e-6:
        return float("nan")
    return float(np.clip((ratio - dead_ratio) / (live_ratio - dead_ratio), 0.0, 1.0))


def estimate_live_share(table, refs, dead_ratio=None):
    """Per bucket and over all gap time: the live share that explains each bucket's speed ratio.

    refs from gap_references; dead_ratio, when given, replaces every bucket's dead reference (the
    match's own published dead block).
    """
    shares = [unmix(row["ratio"], live, dead if dead_ratio is None else dead_ratio)
              for row, (live, dead) in zip(table["buckets"], refs)]
    weights = np.array([row["speed_frames"] for row in table["buckets"]], dtype=np.float64)
    values = np.array(shares)
    ok = np.isfinite(values) & (weights > 0)
    overall = float(np.average(values[ok], weights=weights[ok])) if ok.any() else float("nan")
    return {"buckets": shares, "overall": overall}


def true_live_share(table):
    live = sum(row["live_frames"] for row in table["buckets"])
    dead = sum(row["dead_frames"] for row in table["buckets"])
    return live / max(1, live + dead)


def validate_estimator(tables):
    """Leave one copy out: estimate its live share in gaps from the others' references, against its truth."""
    if len(tables) < 2:
        return {"copies": len(tables)}
    errors, truths = [], []
    for i, table in enumerate(tables):
        refs = gap_references(tables[:i] + tables[i + 1:])
        truths.append(true_live_share(table))
        errors.append(estimate_live_share(table, refs)["overall"] - truths[-1])
    errors = np.array(errors)
    return {"copies": len(tables), "mean_abs_error": float(np.nanmean(np.abs(errors))),
            "max_abs_error": float(np.nanmax(np.abs(errors))), "bias": float(np.nanmean(errors)),
            "truth_low": float(min(truths)), "truth_high": float(max(truths))}


def visibility_windows(match, speed):
    """Per 5 minutes of play from each half's first live frame: visibility and movement.

    Returns the windows (half, minutes from that frame, visibility, mean player speed, the speed with
    the ball seen and missing, the share of time in gaps of LONG_GAP_S or more) and the correlations
    across windows.
    """
    fps = match.fps
    seen = np.isfinite(match.ball_xy[:, 0])
    gaps = ball_gaps(match.period, match.forced_dead, seen)
    long_gap = np.zeros(match.n_frames, dtype=bool)
    for start, end in zip(gaps["start"], gaps["end"]):
        if (end - start) / fps >= LONG_GAP_S:
            long_gap[start:end] = True
    size = int(round(WINDOW_S * fps))
    windows = []
    for half in (1, 2):
        rows = np.flatnonzero((match.period == half) & ~match.forced_dead)
        if not len(rows):
            continue
        starts = list(range(0, len(rows), size))
        if len(starts) > 1 and len(rows) - starts[-1] < WINDOW_MIN_S * fps:
            starts.pop()
        bounds = starts + [len(rows)]
        for a, b in zip(bounds[:-1], bounds[1:]):
            mask = np.zeros(match.n_frames, dtype=bool)
            mask[rows[a:b]] = True
            windows.append({
                "half": half, "from_min": a / fps / 60.0, "to_min": b / fps / 60.0,
                "seen": share(seen, mask), "speed": masked_mean(speed, mask)[0],
                "speed_seen": masked_mean(speed, mask & seen)[0],
                "speed_missing": masked_mean(speed, mask & ~seen)[0],
                "long_gap_share": share(long_gap, mask)})
    seen_share = [w["seen"] for w in windows]
    out = {"windows": windows,
           "corr_seen_speed": correlation(seen_share, [w["speed"] for w in windows]),
           "rank_corr_seen_speed": rank_correlation(seen_share, [w["speed"] for w in windows]),
           "corr_seen_missing_speed": correlation(seen_share, [w["speed_missing"] for w in windows]),
           "corr_long_gap_speed": correlation([w["long_gap_share"] for w in windows], [w["speed"] for w in windows])}
    return out


def camera_state(match, matrices):
    """(F,) arrays of what the camera does: zoom (ground width of the middle image row, metres), the
    speed of its aim point (m/s), the ball's horizontal place in the image (0 left edge, 1 right edge)
    and the reprojection error where the bundle has one."""
    width, height = IMAGE_SIZE
    inverse = safe_inverse(matrices)
    left = ground_of_pixel(inverse, 0.0, height / 2.0)
    right = ground_of_pixel(inverse, width, height / 2.0)
    aim = ground_of_pixel(inverse, width / 2.0, height / 2.0)
    k = max(1, int(round(AIM_SPEED_HALF_S * match.fps)))
    ahead, behind = shifted(aim, match.period, k), shifted(aim, match.period, -k)
    error = match.extras.get("homography_error")
    return {"zoom": np.hypot(*(right - left).T),
            "aim_speed": np.hypot(*(ahead - behind).T) * match.fps / (2 * k),
            "ball_u": project(matrices, match.ball_xy)[:, 0] / width,
            "error": None if error is None else np.asarray(error, dtype=np.float64)}


def _region_edges(match, gaps, i):
    """(row before the gap, row after it), each None when it leaves the gap's half or region."""
    start, end = gaps["start"][i], gaps["end"][i]
    region = (match.period[start], match.forced_dead[start])
    before = start - 1 if start > 0 and (match.period[start - 1], match.forced_dead[start - 1]) == region else None
    after = end if end < match.n_frames and (match.period[end], match.forced_dead[end]) == region else None
    return before, after


def camera_by_gap(match, state):
    """Per gap bucket: median zoom, aim speed and reprojection error over gap frames, and the share of
    gaps whose ball was last seen, or is next seen, near the left or right image edge."""
    fps = match.fps
    seen = np.isfinite(match.ball_xy[:, 0])
    gaps = ball_gaps(match.period, match.forced_dead, seen)
    bucket = gap_bucket_of_frame(gaps, match.n_frames, fps)
    index = bucket_index((gaps["end"] - gaps["start"]) / fps)
    u = state["ball_u"]
    with np.errstate(invalid="ignore"):
        at_edge = (np.minimum(u, 1.0 - u) < EDGE_SHARE)
    edge_before = np.full(len(index), np.nan)
    edge_after = np.full(len(index), np.nan)
    for i in range(len(index)):
        before, after = _region_edges(match, gaps, i)
        if before is not None and np.isfinite(u[before]):
            edge_before[i] = at_edge[before]
        if after is not None and np.isfinite(u[after]):
            edge_after[i] = at_edge[after]

    def summary(mask):
        error = state["error"]
        return {"seconds": float(mask.sum() / fps), "zoom": median(state["zoom"][mask]),
                "aim_speed": median(state["aim_speed"][mask]),
                "error": median(error[mask]) if error is not None else float("nan")}

    play = play_mask(match)
    rows = []
    for b in range(len(GAP_BUCKETS_S)):
        row = summary(bucket == b)
        pick = (index == b) & ~gaps["block"]
        row.update(name=bucket_name(b), edge_before=float(np.nanmean(edge_before[pick])) if np.isfinite(edge_before[pick]).any() else float("nan"),
                   edge_after=float(np.nanmean(edge_after[pick])) if np.isfinite(edge_after[pick]).any() else float("nan"))
        rows.append(row)
    seen_row = summary(play & seen)
    seen_row["edge_seen"] = share(at_edge, play & seen & np.isfinite(u))
    return {"buckets": rows, "seen": seen_row, "block": summary(match.forced_dead & (match.period > 0))}


# ---------------------------------------------------------------- test 9: camera aim

def aim_measurements(cleaned, matrices):
    """Where the camera points against where the visible ball is, on the cleaned ball.

    The camera's aim is the image centre's column: on the ground a line through the centre pixel's
    ground point (the aim point) and the bottom centre pixel's. Errors are taken along the pitch at
    the ball's depth (x error), as the full distance from the aim point (which sits at one depth), and
    in pixels; the centroid of the tracked players is the comparator a ball proxy has to beat.
    """
    fps, period = cleaned.fps, cleaned.period
    width, height = IMAGE_SIZE
    ball = cleaned.ball_xy.astype(np.float64)
    play = play_mask(cleaned)
    seen = play & np.isfinite(ball[:, 0])
    moving = confident_live(cleaned)
    inverse = safe_inverse(matrices)
    aim = ground_of_pixel(inverse, width / 2.0, height / 2.0)
    foot = ground_of_pixel(inverse, width / 2.0, height)
    err_x = x_on_line(aim, foot, ball[:, 1]) - ball[:, 0]
    dist = np.hypot(*(aim - ball).T)
    offset_px = project(matrices, ball)[:, 0] - width / 2.0
    players = cleaned.players_xy.astype(np.float64)
    count = np.isfinite(players[..., 0]).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        centroid = np.where(np.isfinite(players), players, 0.0).sum(axis=1) / count[:, None]
    centroid[count < MIN_TRACKED] = np.nan
    err_cx = centroid[:, 0] - ball[:, 0]
    dist_c = np.hypot(*(centroid - ball).T)

    def summary(mask):
        x_med, c_med = median(np.abs(err_x[mask])), median(np.abs(err_cx[mask]))
        return {"seconds": float(mask.sum() / fps), "x_median": x_med, "x_p90": quantile(np.abs(err_x[mask]), 0.9),
                "dist_median": median(dist[mask]), "offset_px_median": median(np.abs(offset_px[mask])),
                "offset_px_bias": median(offset_px[mask]), "centroid_x_median": c_med,
                "centroid_dist_median": median(dist_c[mask]),
                "ratio": x_med / c_med if np.isfinite(c_med) and c_med > 0 else float("nan")}

    out = {"seen": summary(seen), "moving": summary(moving)}

    lags = []
    for lag_s in LAGS_S:
        lag = int(round(lag_s * fps))
        aim_l, foot_l = shifted(aim, period, lag), shifted(foot, period, lag)
        err = np.abs(x_on_line(aim_l, foot_l, ball[:, 1]) - ball[:, 0])
        lags.append((float(lag_s), median(err[seen])))
    best = min((v, s) for s, v in lags if np.isfinite(v)) if any(np.isfinite(v) for _, v in lags) else (float("nan"), float("nan"))
    out["lags"] = lags
    out["best_lag_s"], out["best_lag_x_median"] = best[1], best[0]

    gaps = ball_gaps(period, cleaned.forced_dead, np.isfinite(ball[:, 0]))
    edge_errors, gap_rows = [], []
    for i in range(len(gaps["start"])):
        if gaps["block"][i] or (gaps["end"][i] - gaps["start"][i]) / fps < AIM_GAP_MIN_S:
            continue
        before, after = _region_edges(cleaned, gaps, i)
        if before is None or after is None:
            continue
        last = gaps["end"][i] - 1
        edge_errors += [abs(err_x[before]), abs(err_x[after])]
        aim_end = x_on_line(aim[last:last + 1], foot[last:last + 1], ball[before:before + 1, 1])[0]
        aim_start = x_on_line(aim[gaps["start"][i]:gaps["start"][i] + 1], foot[gaps["start"][i]:gaps["start"][i] + 1],
                              ball[before:before + 1, 1])[0]
        gap_rows.append(((gaps["end"][i] - gaps["start"][i]) / fps, ball[after, 0], ball[before, 0], aim_end,
                         centroid[last, 0], aim_end - aim_start))
    out["edge_x_median"] = median(edge_errors)
    out["edge_points"] = len(edge_errors)
    rows = np.array(gap_rows, dtype=np.float64).reshape(-1, 6)
    out["reappear"] = []
    for low, high in ((AIM_GAP_MIN_S, np.inf),) + AIM_GAP_CLASSES_S:
        pick = (rows[:, 0] >= low) & (rows[:, 0] < high)
        target = rows[pick, 1]
        out["reappear"].append({
            "low": low, "high": high, "gaps": int(pick.sum()),
            "last_seen": median(np.abs(rows[pick, 2] - target)),
            "aim": median(np.abs(rows[pick, 3] - target)),
            "centroid": median(np.abs(rows[pick, 4] - target))})
    moved = np.abs(rows[:, 1] - rows[:, 2]) > 10.0
    out["pan_follows_share"] = float(np.mean(np.sign(rows[moved, 5]) == np.sign(rows[moved, 1] - rows[moved, 2]))) \
        if moved.any() else float("nan")
    out["pan_follows_gaps"] = int(moved.sum())
    out["pan_corr"] = correlation(rows[:, 5], rows[:, 1] - rows[:, 2])
    return out


# ---------------------------------------------------------------- parallax

def parallax_measurements(match, cleaned, matrices):
    """Label-free signatures of a ball tracked without height (test 3 found them on IDSSE):

    fast steps   ball steps over FAST_STEP_MPS run along the camera ray: median |cos| near 1, against
                 RANDOM_MEDIAN_COS for directions at random. The ray runs from the camera's foot,
                 solved per frame from the homography, to the ball.
    far line     the ball more than BEYOND_LINE_M beyond the touchline away from the camera, far more
                 often than beyond the near one
    gate         the share of ball observations the 40 m/s cleaning gate removes

    match is the raw Match (the raw ball), cleaned is clean(match) (the gate).
    """
    fps, (length, width) = match.fps, match.pitch
    play = play_mask(match)
    centres, principal_x = camera_centres(np.where(play[:, None, None], matrices, np.nan))
    usable = play & np.isfinite(centres[:, 0])
    out = {"camera_usable": share(usable, play), "principal_x": principal_x}
    if usable.any():
        middle = np.median(centres[usable], axis=0)
        out["camera_median"] = tuple(float(v) for v in middle)
        out["camera_iqr"] = tuple(float(v) for v in np.subtract(*np.quantile(centres[usable], [0.75, 0.25], axis=0)))
    else:
        middle = np.full(3, np.nan)
        out["camera_median"] = out["camera_iqr"] = (float("nan"),) * 3
    far = 1.0 if not np.isfinite(middle[1]) or middle[1] < 0 else -1.0

    ball = match.ball_xy.astype(np.float64)
    seen = play & np.isfinite(ball[:, 0])
    pair = seen[:-1] & seen[1:] & (match.period[1:] == match.period[:-1])
    step = ball[1:] - ball[:-1]
    with np.errstate(invalid="ignore"):
        fast = pair & (np.hypot(*step.T) * fps > FAST_STEP_MPS)
    out["steps"] = int(pair.sum())
    out["fast_steps"] = int(fast.sum())
    out["fast_share"] = share(fast, pair)
    out["rays"] = {}
    for name, foot in (("per-frame camera", centres[:-1, :2]), ("median camera", middle[:2]),
                       ("DATA.md camera", solve_camera(row=DATA_MD_ROW).centre[:2])):
        ray = ball[:-1] - foot
        with np.errstate(invalid="ignore", divide="ignore"):
            cosine = np.abs((step * ray).sum(axis=1)) / (np.hypot(*step.T) * np.hypot(*ray.T))
        valid = fast & np.isfinite(cosine)
        out["rays"][name] = {"steps": int(valid.sum()), "median_cos": median(cosine[valid]),
                             "aligned_share": float(np.mean(cosine[valid] > RAY_ALIGNED_COS)) if valid.any() else float("nan")}
    depth = far * ball[:, 1]
    with np.errstate(invalid="ignore"):
        out["beyond_far"] = share(depth > width / 2 + BEYOND_LINE_M, seen)
        out["beyond_near"] = share(depth < -(width / 2 + BEYOND_LINE_M), seen)
    out["gate_removed"] = share(cleaned.ball_removed, seen)
    out["seen_s"] = float(seen.sum() / fps)
    return out


# ---------------------------------------------------------------- test 8: integrity

def read_label_csv(path, n_frames):
    """(F,) labels of a frame,label CSV laid out like a submission. Errors name a rule, never a value."""
    if not os.path.isfile(path):
        raise FormatError("label file not found: {}".format(path))
    try:
        with open(path, "rb") as handle:
            text = handle.read().decode("utf-8")
    except UnicodeDecodeError:
        raise FormatError("the label file is not UTF-8 text") from None
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if not lines or lines[0].rstrip("\r") != "frame,label":
        raise FormatError("the label file's header is not frame,label")
    if len(lines) - 1 != n_frames:
        raise FormatError("the label file does not have one row per frame")
    labels = np.full(n_frames, "", dtype="<U1")
    bad = 0
    for i, line in enumerate(lines[1:]):
        parts = line.rstrip("\r").split(",")
        if len(parts) != 2 or not parts[0].isdigit() or int(parts[0]) != i or parts[1] not in ("H", "A", "D", "X"):
            bad += 1
            continue
        labels[i] = parts[1]
    if bad:
        raise FormatError("{} rows of the label file are not <frame>,<H|A|D|X> in order".format(bad))
    return labels


def nearest_player_codes(match, chunk=20000):
    """The rule of upstream/starter/baseline.py, step 1, on our parser's output.

    Per frame, the team (HOME, AWAY) of the tracked slot nearest to the ball, -1 where the ball or every
    slot is missing; ties go to the lowest slot, as the baseline's strict < does. Also the gap between
    the nearest home and the nearest away player (metres, NaN when a side has nobody), which marks the
    frames where float32 storage could flip the answer.
    """
    n = match.n_frames
    team = match.slot_team
    codes = np.full(n, -1, dtype=np.int8)
    tie_gap = np.full(n, np.nan)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        ball = match.ball_xy[a:b].astype(np.float64)
        players = match.players_xy[a:b].astype(np.float64)
        d2 = ((players - ball[:, None, :]) ** 2).sum(axis=2)
        d2 = np.where(np.isfinite(d2), d2, np.inf)
        has = np.isfinite(d2).any(axis=1)
        best = np.argmin(d2, axis=1)
        part = codes[a:b]
        part[has] = team[best[has]]
        home = np.where(team == HOME, d2, np.inf).min(axis=1)
        away = np.where(team == AWAY, d2, np.inf).min(axis=1)
        both = np.isfinite(home) & np.isfinite(away)
        gap = tie_gap[a:b]
        gap[both] = np.abs(np.sqrt(home[both]) - np.sqrt(away[both]))
    return codes, tie_gap


def baseline_labels(codes):
    """H, A, or D where nobody has an answer: baseline_raw.csv's labels (no smoothing)."""
    return np.where(codes == HOME, "H", np.where(codes == AWAY, "A", "D"))


def compare_labels(ours, theirs, tie_gap, regions):
    """Per region {name: mask}: frames, frames that differ, of those near-ties and the rest."""
    differ = ours != theirs
    with np.errstate(invalid="ignore"):
        tie = np.isfinite(tie_gap) & (tie_gap < NEAR_TIE_M)
    presence = (ours == "D") != (theirs == "D")
    out = {}
    for name, mask in regions.items():
        wrong = mask & differ
        out[name] = {"frames": int(mask.sum()), "differ": int(wrong.sum()), "near_ties": int((wrong & tie).sum()),
                     "unexplained": int((wrong & ~tie).sum()), "ball_presence": int((wrong & presence).sum())}
    return out


def kickoff_check(match, half, frame, attack_sign):
    """Does the tracking show a kickoff on the published frame?

    The published frames are the kick itself (games.py's anchors agree to a frame), so on that frame
    and the KICKOFF_AFTER frames after it, inside the half, the teams stand in their own halves:
    FORMATION_SHARE of the tracked players, with a margin. Where the ball is seen from SPOT_BEFORE
    frames before to KICKOFF_AFTER after, it comes within SPOT_M of the centre spot, and a ball seen
    in the LEAVE_S after the frame does not stay on the spot all that time. Where the ball is not
    seen, the players alone decide. This finds a timeline off by a second or more, not by a frame.
    Returns (ok, reason, details); reason names a rule only.
    """
    details = {"ball_seen": False, "ball_seen_after": False}
    rows = np.flatnonzero(match.frame == frame)
    if not len(rows) or match.period[rows[0]] != half:
        return False, "the published kickoff frame is not in that half", details
    row = int(rows[0])
    sign = int(attack_sign[half])
    if sign == 0:
        return False, "the tracking gives no attack direction for that half", details
    end = min(match.n_frames, row + KICKOFF_AFTER + 1)
    frames = np.arange(row, end)
    frames = frames[match.period[frames] == half]
    x = match.players_xy[frames][..., 0].astype(np.float64) * sign
    tracked = np.isfinite(x)
    with np.errstate(invalid="ignore"):
        own = np.where(match.slot_team == HOME, x <= FORMATION_MARGIN_M, x >= -FORMATION_MARGIN_M)
    details["formation_share"] = share(own, tracked)
    details["tracked_per_frame"] = float(tracked.sum() / max(1, x.shape[0]))
    spot = np.hypot(*match.ball_xy[max(0, row - SPOT_BEFORE):end].astype(np.float64).T)
    if np.isfinite(spot).any():
        details["ball_seen"] = True
        details["ball_spot_m"] = float(np.nanmin(spot))
    after = np.hypot(*match.ball_xy[row + 1:row + 1 + int(round(LEAVE_S * match.fps))].astype(np.float64).T)
    after = after[np.isfinite(after)]
    if len(after):
        details["ball_seen_after"] = True
        details["ball_leaves"] = bool(np.any(after > LEAVE_M))
    if not np.isfinite(details["formation_share"]) or details["formation_share"] < FORMATION_SHARE:
        return False, "the teams are not in their own halves at the published frame", details
    if details["ball_seen"] and details["ball_spot_m"] > SPOT_M:
        return False, "the ball is not on the centre spot at the published frame", details
    if details["ball_seen_after"] and not details["ball_leaves"]:
        return False, "the ball stays on the centre spot for two seconds after the published frame", details
    return True, None, details


def metadata_attack_sign(directions):
    """(sign, reason): +1 when the metadata says home attacks towards +x in the first half, -1 for -x.

    sign is None when the metadata does not say (reason None) or contradicts itself (reason says so).
    Metrica writes left_to_right for +x, as Sample Game 3's tracking confirms.
    """
    table = {"left_to_right": 1, "right_to_left": -1}
    home, away = (tuple(directions) + (None, None))[:2]
    if home not in table or away not in table:
        return None, None
    if table[home] == table[away]:
        return None, "the metadata gives both teams the same attack direction"
    return table[home], None


def ball_in_image_by_shift(match, matrices, shifts=SHIFTS):
    """{shift: share of the seen ball that falls inside the image of the frame `shift` later}."""
    width, height = IMAGE_SIZE
    ball = match.ball_xy.astype(np.float64)
    seen = play_mask(match) & np.isfinite(ball[:, 0])
    flat = matrices.reshape(len(matrices), 9)
    out = {}
    for shift in shifts:
        moved = shifted(flat, match.period, shift).reshape(-1, 3, 3)
        valid = seen & np.isfinite(moved).all(axis=(1, 2))
        pixels = project(np.where(valid[:, None, None], moved, np.eye(3)), ball)
        with np.errstate(invalid="ignore"):
            inside = (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
        out[shift] = share(inside, valid)
    return out


def homography_aligned(shares):
    """(ok, reason): the ball the tracker saw lies inside the image of its own frame.

    A gross misalignment, rows of another time or another half, puts it outside. One frame cannot be
    told apart this way: the shares by shift are for the report only.
    """
    at_zero = shares.get(0, float("nan"))
    if not np.isfinite(at_zero) or at_zero < ALIGNED_FLOOR:
        return False, "the seen ball is often outside the image of its frame"
    return True, None


# ---------------------------------------------------------------- references from simulated copies

def simulated_copies(idsse_dir, simulator, limit=None, seed=1):
    """Yield (name, copy, cleaned, matrices) for degraded copies of the IDSSE matches in idsse_dir.

    matrices come from the copy's view polygon (None for a simulator without a camera). The copies
    carry labels, which is what makes them references.
    """
    from .degrade import SIMULATORS, degrade
    from .idsse import load_idsse

    paths = sorted(glob.glob(os.path.join(idsse_dir, "*.npz")))
    for path in paths[:limit]:
        copy = degrade(load_idsse(path), seed=seed, cfg=SIMULATORS[simulator])
        cleaned = clean(copy)
        matrices = matrices_from_polygon(copy.view_polygon) if copy.view_polygon is not None else None
        yield os.path.splitext(os.path.basename(path))[0], copy, cleaned, matrices


def pooled(rows, key, weight_key=None):
    """Mean of rows[key] over the rows where it is finite, weighted by rows[weight_key] if given."""
    values = np.array([r[key] for r in rows], dtype=np.float64)
    weights = np.ones(len(rows)) if weight_key is None else np.array([r[weight_key] for r in rows], dtype=np.float64)
    ok = np.isfinite(values) & (weights > 0)
    return float(np.average(values[ok], weights=weights[ok])) if ok.any() else float("nan")


def attack_signs(match):
    return attack_direction(match.players_xy, match.slot_team, match.period, match.forced_dead)[0]
