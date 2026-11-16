"""Synthetic homography: the matrix printed in DATA.md, panned to follow a lagged, smoothed ball x.

This camera is invented. It cannot say whether camera aim is a useful signal on the real match,
and because it follows the undegraded ball it knows where the ball is during blackouts.
"""
import numpy as np

from common import stream


def smoothed_target(x_norm, timeline, cfg):
    """Ball x the camera aims at: 0.5 outside the graded blocks, smoothed, then delayed."""
    c = cfg["camera"]
    fps = timeline.fps
    target = np.where(timeline.graded, x_norm, 0.5)
    alpha = 1.0 / (c["smooth_s"] * fps)
    smooth = np.empty(timeline.n, dtype=np.float64)
    value = 0.5
    for i, sample in enumerate(target.tolist()):
        value += alpha * (sample - value)
        smooth[i] = value
    lag = int(round(c["lag_s"] * fps))
    out = np.full(timeline.n, 0.5, dtype=np.float64)
    out[lag:] = smooth[:-lag]
    return out


def matrices(aim, cfg):
    """(n, 3, 3), each row normalised to h22 = 1 like the printed one."""
    c = cfg["camera"]
    start = np.asarray(c["start_matrix"], dtype=np.float64).reshape(3, 3)
    focal = c["focal_px"]
    cx, cy = c["principal_px"]
    k = np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]])
    k_inverse = np.linalg.inv(k)

    def pixel_x(x):
        w = start[2, 0] * x + start[2, 1] * 0.5 + start[2, 2]
        return (start[0, 0] * x + start[0, 1] * 0.5 + start[0, 2]) / w

    # Turn the camera by the angle that would bring the aim point back to where the pitch centre is,
    # scaled by pan_gain so the ball is approached rather than centred.
    centre_px = pixel_x(0.5)
    aim_px = pixel_x(aim)
    theta = c["pan_gain"] * (np.arctan2(aim_px - cx, focal) - np.arctan2(centre_px - cx, focal))
    cosine, sine = np.cos(theta), np.sin(theta)

    rotation = np.zeros((len(aim), 3, 3), dtype=np.float64)
    rotation[:, 0, 0] = cosine
    rotation[:, 0, 2] = -sine
    rotation[:, 1, 1] = 1.0
    rotation[:, 2, 0] = sine
    rotation[:, 2, 2] = cosine
    out = k @ rotation @ k_inverse @ start
    return out / out[:, 2, 2][:, None, None], theta


def reprojection_error(timeline, cfg):
    c = cfg["camera"]
    rng = stream(cfg["seed"], "reprojection")
    median = np.full(timeline.n, c["reproj_median_other"], dtype=np.float64)
    median[timeline.graded] = c["reproj_median_play"]
    first_half_end = timeline.blocks[0][2]
    second_half_start = timeline.blocks[1][1]
    median[first_half_end + 1:second_half_start] = c["reproj_median_halftime"]
    return median * np.exp(rng.normal(0.0, c["reproj_sigma"], timeline.n))


def write_csv(path, h, error):
    """frame,h00..h22,reprojection_error,valid with DATA.md's number shape (7 significant digits)."""
    names = ",".join("h{}{}".format(r, c) for r in range(3) for c in range(3))
    rows = h.reshape(-1, 9).tolist()
    with open(path, "w") as handle:
        handle.write("frame,{},reprojection_error,valid\n".format(names))
        for frame, (row, err) in enumerate(zip(rows, error.tolist())):
            handle.write("{},{:.7g},{:.7g},{:.7g},{:.7g},{:.7g},{:.7g},{:.7g},{:.7g},{:.7g},{:.7g},1\n".format(
                frame, *row, err))


def build(path_m, timeline, cfg, csv_path):
    norm_x = cfg["pitch"]["normalise_m"][0]
    x_norm = 0.5 + path_m[:, 0] / norm_x
    aim = smoothed_target(x_norm, timeline, cfg)
    h, theta = matrices(aim, cfg)
    error = reprojection_error(timeline, cfg)
    write_csv(csv_path, h, error)

    # Every pitch corner has to stay in front of the camera for project() to mean anything.
    corners = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
    depth = h[:, 2, :] @ corners.T
    if np.any(depth <= 0.0):
        raise RuntimeError("a pan puts a pitch corner behind the camera; reduce pan_gain")
    return {
        "pan_deg_min": float(np.degrees(theta.min())),
        "pan_deg_max": float(np.degrees(theta.max())),
        "min_corner_depth": float(depth.min()),
    }
