"""Synthetic Smart Tagging, in the SportsCode XML shape the public loaders read.

Phases are derived from the twin's labels and the undegraded ball and players, then thinned to the
counts in DATA.md and perturbed: boundary jitter, team flips and tags moved to a random time. Every
throw-in carries team N/A, and set pieces open about 3.5 s before the restart.

Which tag groups go on which code is not stated in the public docs, and neither is the rule behind
each phase. These definitions are plausible guesses, so the twin cannot judge how informative the
real tags are. It only guarantees the file shape and the documented quirks.
"""
from xml.sax.saxutils import escape

import numpy as np

from common import stream, true_runs

BOX_DEPTH_M = 16.5
BOX_HALF_WIDTH_M = 20.16
THIRD = 1.0 / 3.0


class Context:
    def __init__(self, timeline, ball_m, base, player_team, pitch, cfg):
        self.timeline = timeline
        self.labels = timeline.labels
        self.fps = timeline.fps
        self.ball = ball_m
        self.base = base
        self.team = player_team
        self.length, self.width = (float(v) for v in pitch)
        self.cfg = cfg
        self.sign = self.attack_signs()

    def attack_signs(self):
        """{(half, side): +1 if that side attacks towards +x}, from team shape at each kickoff."""
        signs = {}
        for half, kickoff in self.timeline.kickoffs.items():
            home = np.nanmean(self.base[kickoff, self.team == 0, 0])
            away = np.nanmean(self.base[kickoff, self.team == 1, 0])
            signs[(half, "H")] = 1.0 if home < away else -1.0
            signs[(half, "A")] = -signs[(half, "H")]
        return signs


def spells_of(ctx):
    """[(start, end, side)] for each maximal H or A run inside the graded blocks."""
    out = []
    for _, first, last in ctx.timeline.blocks:
        chunk = ctx.labels[first:last + 1]
        edges = np.flatnonzero(chunk[1:] != chunk[:-1]) + 1
        bounds = np.concatenate(([0], edges, [len(chunk)])) + first
        for start, end in zip(bounds[:-1].tolist(), bounds[1:].tolist()):
            if ctx.labels[start] in ("H", "A"):
                out.append((start, end, str(ctx.labels[start])))
    return out


def other(side):
    return "A" if side == "H" else "H"


def side_label(y_mean, sign, width):
    if abs(y_mean) < width / 6.0:
        return "Central"
    return "Left" if sign * y_mean > 0 else "Right"


def entry_label(ball, start, fps):
    """Direction of ball entry from the second before it reached the box."""
    before = max(0, start - fps)
    dx, dy = np.abs(ball[start] - ball[before])
    if dy > 2.0 * dx:
        return "Horizontal"
    if dx > 2.0 * dy:
        return "Vertical"
    return "Diagonal"


def phase_candidates(ctx):
    """{code: [instance]} for the open-play phases, before thinning."""
    fps = ctx.fps
    length, width = ctx.length, ctx.width
    out = {code: [] for code in ctx.cfg["tags"]["code_counts"]}
    spells = spells_of(ctx)

    def add(code, start, end, team, **labels):
        out[code].append({"start": start / fps, "end": end / fps, "team": team, "labels": labels})

    for index, (a, b, side) in enumerate(spells):
        half = int(ctx.timeline.half[a])
        sign = ctx.sign[(half, side)]
        u = 0.5 + sign * ctx.ball[a:b, 0] / length
        y = ctx.ball[a:b, 1]
        side_here = lambda s, e: side_label(float(np.mean(y[s:e])), sign, width)

        if u[0] < THIRD:
            passed = np.flatnonzero(u >= THIRD)
            stop = int(passed[0]) if len(passed) else b - a
            if stop >= 3 * fps:
                add("BUILD UP", a, a + stop, side, Side=side_here(0, stop))

        starts, ends = true_runs((u >= THIRD) & (u < 2 * THIRD))
        for s, e in zip(starts.tolist(), ends.tolist()):
            if e - s >= 2 * fps and u[e - 1] - u[s] > 0.08:
                add("PROGRESSION", a + s, a + e, side, Side=side_here(s, e))

        starts, ends = true_runs(u >= 2 * THIRD)
        for s, e in zip(starts.tolist(), ends.tolist()):
            if e - s >= 1.5 * fps:
                add("BALL IN FINAL THIRD", a + s, a + e, side, Side=side_here(s, e))

        depth = 1.0 - BOX_DEPTH_M / length
        starts, ends = true_runs((u >= depth) & (np.abs(y) < BOX_HALF_WIDTH_M))
        for s, e in zip(starts.tolist(), ends.tolist()):
            if e - s >= 0.2 * fps:
                add("BALL IN THE BOX", a + s, a + e, side, Side=side_here(s, e),
                    **{"Direction of ball entry": entry_label(ctx.ball, a + s, fps)})

        near_goal = np.flatnonzero(u >= 0.6)
        if len(near_goal):
            own = ctx.team == (0 if side == "H" else 1)
            px = ctx.base[a:b, :, 0][:, own]
            py = ctx.base[a:b, :, 1][:, own]
            with np.errstate(invalid="ignore"):
                in_box = (0.5 + sign * px / length >= depth) & (np.abs(py) < BOX_HALF_WIDTH_M)
            count = np.sum(in_box, axis=1)
            starts, ends = true_runs(count >= 1)
            for s, e in zip(starts.tolist(), ends.tolist()):
                if e - s >= fps:
                    peak = int(np.max(count[s:e]))
                    add("PLAYERS IN THE BOX", a + s, a + e, side, Side=side_here(s, e),
                        **{"Max Players in the box": "7+" if peak >= 7 else str(peak)})

        # A chance: the spell dies close to goal after the ball moved fast there.
        tail = max(0, (b - a) - fps)
        if b - a >= fps and u[-1] >= 0.8 and abs(float(y[-1])) < 25.0:
            run = np.hypot(*(ctx.ball[b - 1] - ctx.ball[a + tail]))
            if run >= 8.0:
                add("CREATING CHANCES", max(a, b - 6 * fps), b, side, Side=side_here(tail, b - a))

        # A long ball: at least 30 m forward inside 2 s.
        gap = 2 * fps
        if b - a > gap:
            forward = (u[gap:] - u[:-gap]) * length
            hits = np.flatnonzero(forward >= 30.0)
            last = -10 * fps
            for hit in hits.tolist():
                if hit - last >= 5 * fps:
                    add("LONG BALL", a + hit, a + hit + int(2.5 * fps), side, Side=side_here(hit, hit + gap))
                    last = hit

        # A turnover without a stoppage.
        if index and spells[index - 1][1] == a and spells[index - 1][2] == other(side):
            before_a = spells[index - 1][0]
            window = min(b - a, 5 * fps)
            if a - before_a >= 2 * fps and window >= 2.5 * fps and u[window - 1] - u[0] > 0.05:
                add("ATTACKING TRANSITION", a - fps // 2, a + window, side, Side=side_here(0, window))
                add("DEFENSIVE TRANSITION", a - fps // 2, a + window, other(side), Side=side_here(0, window))

        if b - a >= 4 * fps:
            zones = (("DEFENDING IN DEFENSIVE THIRD", u >= 2 * THIRD),
                     ("DEFENDING IN MIDDLE THIRD", (u >= THIRD) & (u < 2 * THIRD)),
                     ("DEFENDING IN ATTACKING THIRD", u < THIRD))
            for code, mask in zones:
                starts, ends = true_runs(mask)
                for s, e in zip(starts.tolist(), ends.tolist()):
                    if e - s >= 3 * fps:
                        add(code, a + s, a + e, other(side), Side=side_here(s, e))
    return out


def restart_candidates(ctx):
    """([set piece], [goal]) from each dead-to-live transition, typed by where the ball restarted."""
    fps = ctx.fps
    half_length, half_width = ctx.length / 2.0, ctx.width / 2.0
    set_pieces, goals = [], []
    kickoffs = set(ctx.timeline.kickoffs.values())
    for half, first, last in ctx.timeline.blocks:
        chunk = ctx.labels[first:last + 1]
        restarts = (np.flatnonzero((chunk[1:] != "D") & (chunk[:-1] == "D")) + 1 + first).tolist()
        if chunk[0] != "D":
            restarts.insert(0, first)
        for frame in restarts:
            x, y = (float(v) for v in ctx.ball[frame])
            team = str(ctx.labels[frame])
            if frame in kickoffs:
                kind = "Kick Off (Start)"
            elif np.hypot(x, y) < 2.5:
                kind = "Kick Off (After Goal)"
            elif abs(x) > half_length - 3.0 and abs(y) > half_width - 3.0:
                kind = "Corner Kick"
            elif abs(y) > half_width - 2.0:
                kind = "Throw-in"
            elif abs(x) > half_length - 7.0 and abs(y) < 11.0:
                kind = "Goal Kick"
            else:
                kind = "Free Kick"
            sign = ctx.sign[(half, team)]
            set_pieces.append({"frame": frame, "team": team, "kind": kind,
                               "side": side_label(y, sign, ctx.width)})
            if kind == "Kick Off (After Goal)":
                # The goal came where the stoppage before this kickoff began.
                dead = np.flatnonzero(ctx.labels[first:frame] != "D")
                stop = first + int(dead[-1]) + 1 if len(dead) else first
                goals.append({"start": (stop - 8 * fps) / fps, "end": (stop + 2 * fps) / fps,
                              "team": other(team), "labels": {}})
    return set_pieces, goals


def thin(items, count, rng):
    """Keep count items at random, in their original order."""
    if len(items) <= count:
        return list(items)
    keep = np.sort(rng.choice(len(items), size=count, replace=False)).tolist()
    return [items[i] for i in keep]


def build(ctx):
    t = ctx.cfg["tags"]
    fps = ctx.fps
    rng = stream(ctx.cfg["seed"], "tags")
    candidates = phase_candidates(ctx)
    set_pieces, goals = restart_candidates(ctx)
    candidates["GOALS"] = goals

    instances = []
    for code in sorted(t["code_counts"]):
        for item in thin(candidates[code], t["code_counts"][code], rng):
            instances.append({"code": code, **item})

    by_type = {}
    for piece in set_pieces:
        by_type.setdefault(piece["kind"], []).append(piece)
    kept_pieces = []
    for kind in sorted(t["type_counts"]):
        kept_pieces.extend(thin(by_type.get(kind, []), t["type_counts"][kind], rng))

    # Noise on phases: jittered boundaries, a few wrong teams, a few tags somewhere else entirely.
    graded = np.flatnonzero(ctx.timeline.graded)
    for item in instances:
        length = max(0.2, item["end"] - item["start"])
        item["start"] += float(rng.normal(0.0, t["boundary_jitter_s"]))
        item["end"] = item["start"] + max(0.2, length + float(rng.normal(0.0, t["boundary_jitter_s"])))
        if rng.random() < t["team_flip_prob"]:
            item["team"] = other(item["team"])
        if rng.random() < t["false_positive_share"]:
            moved = float(rng.choice(graded)) / fps
            item["end"] = moved + (item["end"] - item["start"])
            item["start"] = moved

    # Set pieces open early and close soon after the restart. Team is N/A on every throw-in and on
    # a few others, as in DATA.md.
    others = [i for i, piece in enumerate(kept_pieces) if piece["kind"] != "Throw-in"]
    extra_na = set(np.sort(rng.choice(others, size=min(t["team_na_extra"], len(others)), replace=False)).tolist())
    for i, piece in enumerate(kept_pieces):
        opening = max(0.5, float(rng.normal(t["early_open_s"], t["early_open_sd_s"])))
        closing = max(0.5, float(rng.normal(t["late_close_s"], t["late_close_sd_s"])))
        seconds = piece["frame"] / fps
        team = "N/A" if piece["kind"] == "Throw-in" or i in extra_na else piece["team"]
        instances.append({"code": "SET PIECES", "start": seconds - opening, "end": seconds + closing,
                          "team": team, "labels": {"Type": piece["kind"], "Side": piece["side"]},
                          "restart_frame": piece["frame"]})

    limit = (ctx.timeline.n - 1) / fps
    for item in instances:
        item["start"] = float(np.clip(item["start"], 0.0, limit))
        item["end"] = float(np.clip(item["end"], item["start"], limit))
    instances.sort(key=lambda item: (item["start"], item["end"], item["code"]))

    names = {"H": t["team_home"], "A": t["team_away"], "N/A": "N/A"}
    half_two_start = ctx.timeline.blocks[1][1]
    half_one_end = ctx.timeline.blocks[0][2]
    for item in instances:
        item["team_name"] = names[item["team"]]
        frame = round(item["start"] * fps)
        # Tags opening just before a half still belong to it.
        item["half"] = "1st Half" if frame <= half_one_end else "2nd Half"
        if half_one_end < frame < half_two_start:
            item["half"] = "2nd Half" if frame - half_one_end > half_two_start - frame else "1st Half"
    return instances


def write_xml(path, instances):
    lines = ['<?xml version="1.0" encoding="utf-8"?>', "<file>", "  <ALL_INSTANCES>"]
    for number, item in enumerate(instances, start=1):
        lines.append("    <instance>")
        lines.append("      <ID>{}</ID>".format(number))
        lines.append("      <start>{:.3f}</start>".format(item["start"]))
        lines.append("      <end>{:.3f}</end>".format(item["end"]))
        lines.append("      <code>{}</code>".format(escape(item["code"])))
        labels = [("Team", item["team_name"]), ("Half", item["half"])] + list(item["labels"].items())
        for group, text in labels:
            lines.append("      <label>")
            lines.append("        <group>{}</group>".format(escape(group)))
            lines.append("        <text>{}</text>".format(escape(text)))
            lines.append("      </label>")
        lines.append("    </instance>")
    lines.extend(["  </ALL_INSTANCES>", "</file>", ""])
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def summary(instances):
    codes, types, teams = {}, {}, {}
    for item in instances:
        codes[item["code"]] = codes.get(item["code"], 0) + 1
        teams[item["team_name"]] = teams.get(item["team_name"], 0) + 1
        kind = item["labels"].get("Type")
        if kind:
            types[kind] = types.get(kind, 0) + 1
    return {"instances": len(instances),
            "codes": {key: codes[key] for key in sorted(codes)},
            "types": {key: types[key] for key in sorted(types)},
            "teams": {key: teams[key] for key in sorted(teams)}}
