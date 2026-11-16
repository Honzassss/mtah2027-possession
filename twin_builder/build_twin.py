"""Build the stand-in ("twin") for the competition's tactical bundle from one open DFL match.

    python -I twin_builder/build_twin.py --config twin_builder/twin.toml \\
        --source external/processed/idsse --out twin

Output goes to <out>/<match>/. Code written against the competition's file format runs unchanged
when its data directory points at <out>/<match>/data. There is no video and no broadcast source.
Everything random comes from the seed in the config, and the build is byte-for-byte reproducible.
"""
import argparse
import hashlib
import json
import os
import sys
import tomllib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atd  # noqa: E402
import ball  # noqa: E402
import camera  # noqa: E402
import players  # noqa: E402
import tags  # noqa: E402
import timeline as timeline_module  # noqa: E402

NOTICE = """THIS IS NOT REAL DATA AND NOT COMPETITION DATA.

This folder is a synthetic stand-in built from open DFL tracking data (IDSSE, match {match},
Bassek et al. 2025, CC BY 4.0), degraded by twin_builder/ with seed {seed}. It only imitates the
file format and documented quirks of the More than a Hack 2027 tactical bundle, so code can be
tested against that format without the gated data.

Team names, tag values, file names and timeline follow the public challenge repository. The match
in these files is not FC Barcelona vs Athletic Club. The homography and the Smart Tagging are
invented. Point the data directory at the data/ folder next to this file.
"""


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--source", required=True, help="directory holding <match>.npz")
    parser.add_argument("--out", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                        help="override a number in the config, for experiments")
    args = parser.parse_args()

    with open(args.config, "rb") as handle:
        cfg = tomllib.load(handle)
    for item in args.set:
        key, value = item.split("=", 1)
        section, name = key.split(".", 1)
        if name not in cfg[section]:
            raise KeyError("no such setting: {}".format(key))
        cfg[section][name] = float(value)
    match = cfg["match"]
    with np.load(os.path.join(args.source, "{}.npz".format(match))) as archive:
        source = {key: archive[key] for key in archive.files}

    root = os.path.join(args.out, match)
    tactical = os.path.join(root, "data", "tactical")
    gt_dir = os.path.join(root, "gt")
    os.makedirs(tactical, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    stem = os.path.join(tactical, "BAR-ATH_tactical")

    timeline = timeline_module.build(source, cfg)
    base = players.build_base(source, timeline, cfg)
    slot_x, slot_y, slot_s, player_stats = players.build_slots(source, base, timeline, cfg)
    path = ball.true_path(source, timeline, cfg)
    ball_x, ball_y, ball_s, ball_stats = ball.build(path, timeline, cfg)

    context = tags.Context(timeline, path, base, source["player_team"], source["pitch"], cfg)
    instances = tags.build(context)
    tags.write_xml(stem + "_smart_tagging.xml", instances)

    camera_stats = camera.build(path, timeline, cfg, stem + "_homography.csv")
    atd.write_metadata(stem + "_atd.xml", timeline, cfg, context.sign[(1, "H")] > 0)
    atd.write_raw(stem + "_atd.txt", slot_x, slot_y, slot_s, ball_x, ball_y, ball_s)
    timeline_module.write_gt(timeline, cfg, os.path.join(gt_dir, "possession_gt_full.csv"),
                             os.path.join(gt_dir, "possession_gt_public_like.csv"))

    with open(os.path.join(root, "TWIN_NOT_REAL_DATA.txt"), "w") as handle:
        handle.write(NOTICE.format(match=match, seed=cfg["seed"]))

    graded = timeline.graded
    labels = timeline.labels
    stats = {
        "match": match,
        "seed": cfg["seed"],
        "timeline": timeline.info,
        "graded_label_share": {key: float(np.mean(labels[graded] == key)) for key in "HAD"},
        "players": player_stats,
        "ball": ball_stats,
        "tags": tags.summary(instances),
        "camera": camera_stats,
    }
    with open(os.path.join(root, "build_stats.json"), "w") as handle:
        json.dump(stats, handle, indent=2, sort_keys=True)
        handle.write("\n")

    outputs = []
    for folder, _, names in os.walk(root):
        for name in names:
            if name != "MANIFEST.sha256":
                outputs.append(os.path.relpath(os.path.join(folder, name), root))
    outputs.sort()
    with open(os.path.join(root, "MANIFEST.sha256"), "w") as handle:
        for name in outputs:
            handle.write("{}  {}\n".format(sha256_of(os.path.join(root, name)), name))
    print(json.dumps(stats, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
