"""Lay one IDSSE match onto the competition's video timeline.

Half 1 starts on the first kickoff frame, half 2 on the second-half kickoff, both at the first live
source frame. The real halves are shorter than the graded blocks, so each block ends in dead-ball
padding. 110808..113741 is pre-kickoff and all D. Everything outside the blocks is X.
"""
import numpy as np


class Timeline:
    def __init__(self, n, blocks, kickoffs, fps):
        self.n = n
        self.blocks = blocks          # [(half, first, last)] inclusive
        self.kickoffs = kickoffs      # {half: video frame}
        self.fps = fps
        self.labels = np.full(n, "X", dtype="<U1")
        self.src_index = np.full(n, -1, dtype=np.int32)
        self.half = np.zeros(n, dtype=np.int8)
        self.info = {}

    @property
    def graded(self):
        return self.half > 0

    @property
    def dead(self):
        return self.labels == "D"


def build(source, cfg):
    t = cfg["timeline"]
    blocks = [(1, t["half1"][0], t["half1"][1]), (2, t["half2"][0], t["half2"][1])]
    kickoffs = {1: t["kickoff1"], 2: t["kickoff2"]}
    timeline = Timeline(t["n_video_frames"], blocks, kickoffs, t["fps"])

    for half, first, last in blocks:
        timeline.half[first:last + 1] = half
        timeline.labels[first:last + 1] = "D"
        rows = np.flatnonzero(source["period"] == half)
        live = np.flatnonzero(source["labels"][rows] != "D")
        if not len(live):
            raise ValueError("half {} has no live frames".format(half))
        video = kickoffs[half] + np.arange(len(rows)) - live[0]
        if video[-1] > last:
            raise ValueError("half {} is {} frames too long for its graded block".format(half, video[-1] - last))
        keep = video >= first
        timeline.src_index[video[keep]] = rows[keep]
        timeline.labels[video[keep]] = source["labels"][rows[keep]]
        timeline.info["half{}".format(half)] = {
            "source_frames": int(len(rows)),
            "source_frames_dropped": int(np.sum(~keep)),
            "first_mapped": int(video[keep][0]),
            "last_mapped": int(video[-1]),
            "pre_frames": int(video[keep][0] - first),
            "pad_frames": int(last - video[-1]),
        }

    for half, kickoff in kickoffs.items():
        if timeline.labels[kickoff] == "D":
            raise ValueError("kickoff frame of half {} is not live".format(half))
    return timeline


def write_gt(timeline, cfg, full_path, public_path):
    first, last = cfg["timeline"]["public_window"]
    labels = timeline.labels
    with open(full_path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels.tolist()))
    with open(public_path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label if first <= i <= last else "X")
                          for i, label in enumerate(labels.tolist()))
