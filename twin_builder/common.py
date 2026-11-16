"""Helpers shared by the twin builder: named random streams, runs, smooth noise."""
import math
import zlib

import numpy as np


def stream(seed, name):
    """A random generator for one named purpose. Stable across runs and independent of call order."""
    return np.random.default_rng([int(seed), zlib.crc32(name.encode("ascii"))])


def true_runs(mask):
    """(starts, ends) of the True runs of a boolean array, ends exclusive."""
    edges = np.diff(np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def ar1(rng, n, m, sigma, tau_frames):
    """Stationary AR(1) noise, shape (n, m), with std sigma and correlation time tau_frames."""
    phi = math.exp(-1.0 / max(tau_frames, 1e-6))
    scale = sigma * math.sqrt(1.0 - phi * phi)
    innovations = rng.standard_normal((n, m), dtype=np.float32) * scale
    out = np.zeros((n, m), dtype=np.float32)
    if n:
        out[0] = rng.standard_normal(m, dtype=np.float32) * sigma
    for t in range(1, n):
        out[t] = phi * out[t - 1] + innovations[t]
    return out


def walk(rng, n, m, speed_mps, tau_s, fps):
    """Smooth random walk, offsets in metres, shape (n, m, 2), with RMS speed speed_mps."""
    velocity = ar1(rng, n, 2 * m, speed_mps / math.sqrt(2.0), tau_s * fps)
    position = np.cumsum(velocity / float(fps), axis=0)
    return position.reshape(n, m, 2)
