"""A pan camera solved from the one homography row printed in DATA.md.

The row maps normalised pitch coordinates to pixels. Assuming square pixels and no roll, it fixes the
focal length and where the camera stands, and those explain two things a flat homography hides:

    parallax    Tracking on the ground plane puts a ball at height z further along the camera ray, by
                D * z / (H - z) for a ball D metres from the foot of the camera, which stands H high.
    noise       One pixel covers far more metres in depth than along the pitch, and more on the far side.

Only the row is public, so this camera says nothing about how the real one moves. Panning is modelled
as a turn about the vertical axis through the camera centre, which leaves the parallax unchanged.

World coordinates are schema metres: x along the pitch, y away from the camera, z up.
"""
from dataclasses import dataclass

import numpy as np

from .epts import IMAGE_SIZE

DATA_MD_ROW = (1822.663, -511.2005, 44.48168, 1.698323, 169.2971, 255.7303, -0.01907188, -0.5269586, 1.0)
DATA_MD_PITCH = (105.3, 68.0)   # games.py, the pitch the normalised coordinates refer to
MIN_RAY_CLEARANCE_M = 0.5       # a point closer than this to camera height has no usable ground point


@dataclass(frozen=True)
class PanCamera:
    focal: float
    principal: tuple
    rotation: np.ndarray    # (3, 3) world to camera, at pan 0
    centre: np.ndarray      # (3,) metres
    image_size: tuple

    def intrinsics(self):
        return np.array([[self.focal, 0.0, self.principal[0]],
                         [0.0, self.focal, self.principal[1]],
                         [0.0, 0.0, 1.0]])

    def world_to_camera(self, pan):
        """(F, 3, 3). A positive pan turns the camera from +y towards +x."""
        pan = np.atleast_1d(np.asarray(pan, dtype=np.float64))
        cosine, sine = np.cos(pan), np.sin(pan)
        turn = np.zeros((len(pan), 3, 3))
        turn[:, 0, 0] = cosine
        turn[:, 0, 1] = -sine
        turn[:, 1, 0] = sine
        turn[:, 1, 1] = cosine
        turn[:, 2, 2] = 1.0
        return self.rotation @ turn

    def matrices(self, pan):
        """(F, 3, 3) from ground metres to pixels, normalised to h22 = 1."""
        offset = np.column_stack(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), -self.centre))
        out = self.intrinsics() @ self.world_to_camera(pan) @ offset
        return out / out[:, 2, 2][:, None, None]

    def pixels_3d(self, xyz, pan):
        """Pixels of world points, xyz (F, 3), one pan per point. NaN behind the camera."""
        camera = np.einsum("nij,nj->ni", self.world_to_camera(pan), xyz - self.centre)
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.focal * camera[:, 0] / camera[:, 2] + self.principal[0]
            v = self.focal * camera[:, 1] / camera[:, 2] + self.principal[1]
        out = np.column_stack((u, v))
        out[camera[:, 2] <= 0] = np.nan
        return out

    def heading_of_point(self, x, y):
        """Angle about the vertical from +y towards +x of the ground direction from the camera to (x, y)."""
        return np.arctan2(x - self.centre[0], y - self.centre[1])

    def pan_to_face(self, x, y):
        """Pan that puts (x, y) on the image column through the principal point, close to the centre."""
        return self.heading_of_point(x, y) - self.heading_at_rest()

    def heading_at_rest(self):
        """Heading of the optical axis at pan 0."""
        axis = self.rotation.T @ np.array([0.0, 0.0, 1.0])
        return float(np.arctan2(axis[0], axis[1]))

    def view_polygons(self, pan):
        """(F, 4, 2) ground footprint of the image, corners TL, BL, BR, TR. NaN above the horizon."""
        width, height = self.image_size
        corners = np.array([[0, 0, 1], [0, height, 1], [width, height, 1], [width, 0, 1]], dtype=np.float64).T
        ground = np.linalg.inv(self.matrices(pan)) @ corners
        with np.errstate(divide="ignore", invalid="ignore"):
            xy = ground[:, :2, :] / ground[:, 2:3, :]
        xy = np.where(ground[:, 2:3, :] > 0, xy, np.nan)
        return np.swapaxes(xy, 1, 2).astype(np.float32)


def solve_camera(row=DATA_MD_ROW, pitch=DATA_MD_PITCH, principal_y=None, image_size=IMAGE_SIZE):
    """PanCamera that reproduces one homography row.

    The two conditions on a rotation (first two columns orthogonal, equal length) leave a one-parameter
    family, so the vertical principal point is fixed at the image centre, or at `principal_y` for a
    sensitivity scan, and the focal length and horizontal principal point are solved. The horizontal
    principal point does not move the camera's distance or height, only its position along the pitch.
    """
    length, width = pitch
    normalised_to_metres = np.array([[1.0 / length, 0.0, 0.5], [0.0, -1.0 / width, 0.5], [0.0, 0.0, 1.0]])
    h = np.asarray(row, dtype=np.float64).reshape(3, 3) @ normalised_to_metres
    cy = image_size[1] / 2.0 if principal_y is None else float(principal_y)

    def fit(cx):
        centred = np.vstack((h[0] - cx * h[2], h[1] - cy * h[2], h[2]))
        first, second = centred[:, 0], centred[:, 1]
        focal_sq = -((first[:2] @ first[:2]) - (second[:2] @ second[:2])) / (first[2] ** 2 - second[2] ** 2)
        focal = float(np.sqrt(focal_sq))
        scaled = centred * np.array([[1.0 / focal], [1.0 / focal], [1.0]])
        return focal, scaled, float(scaled[:, 0] @ scaled[:, 1] / (np.linalg.norm(scaled[:, 0]) * np.linalg.norm(scaled[:, 1])))

    # The cosine between the columns is almost linear in cx, so a few secant steps are exact.
    cx_a, cx_b = image_size[0] / 2.0, image_size[0] / 2.0 - 40.0
    _, _, residual_a = fit(cx_a)
    _, _, residual_b = fit(cx_b)
    for _ in range(12):
        if abs(residual_b - residual_a) < 1e-15:
            break
        cx_a, residual_a, cx_b = cx_b, residual_b, cx_b - residual_b * (cx_b - cx_a) / (residual_b - residual_a)
        _, _, residual_b = fit(cx_b)
    focal, scaled, _ = fit(cx_b)

    scale = 2.0 / (np.linalg.norm(scaled[:, 0]) + np.linalg.norm(scaled[:, 1]))
    if scaled[2, 2] * scale < 0:    # the pitch centre has to be in front of the camera
        scale = -scale
    r1, r2, t = scaled[:, 0] * scale, scaled[:, 1] * scale, scaled[:, 2] * scale
    rotation = np.column_stack((r1, r2, np.cross(r1, r2)))
    centre = -rotation.T @ t
    return PanCamera(focal=focal, principal=(cx_b, cy), rotation=rotation, centre=centre, image_size=tuple(image_size))


def apparent_ground(xy, z, centre):
    """Where a point at height z lands when its pixel is mapped back to the pitch as if z were 0.

    xy (..., 2), z (...), centre (3,) in metres. NaN for points too near or above camera height, which
    have no usable ground point.
    """
    height = float(centre[2])
    z = np.asarray(z, dtype=np.float64)
    usable = z < height - MIN_RAY_CLEARANCE_M
    factor = np.where(usable, height / np.where(usable, height - z, 1.0), np.nan)
    foot = np.asarray(centre[:2], dtype=np.float64)
    return foot + (np.asarray(xy, dtype=np.float64) - foot) * factor[..., None]


def project(matrices, xy):
    """Apply (F, 3, 3) homographies to (F, 2) or (F, S, 2) points, returning float64 of the same shape."""
    x, y = xy[..., 0].astype(np.float64), xy[..., 1].astype(np.float64)
    h = matrices if xy.ndim == 2 else matrices[:, None, :, :]
    w = h[..., 2, 0] * x + h[..., 2, 1] * y + h[..., 2, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = (h[..., 0, 0] * x + h[..., 0, 1] * y + h[..., 0, 2]) / w
        v = (h[..., 1, 0] * x + h[..., 1, 1] * y + h[..., 1, 2]) / w
    out = np.stack((u, v), axis=-1)
    out[w <= 0] = np.nan
    return out
