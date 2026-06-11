#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


MOTION_RE = re.compile(
    r"MOTION pos=\((?P<x>[+-]?\d+\.\d+),(?P<y>[+-]?\d+\.\d+)\) "
    r"yaw=(?P<yaw>[+-]?\d+\.\d+) v=(?P<v>\d+\.\d+) "
    r"step_err=(?P<step>\d+\.\d+)m/(?P<yaw_step>\d+\.\d+)rad "
    r"track_err=(?P<track>\d+\.\d+)m .*? route=(?P<route>\S+) "
    r"cmd_vel=\((?P<cmd_v>[+-]?\d+\.\d+),(?P<cmd_w>[+-]?\d+\.\d+)\)"
)
JITTER_RE = re.compile(r"JITTER events=(?P<count>\d+) reasons=(?P<reasons>.*?) pos=.*? route=(?P<route>\S+)")
SEGMENT_RE = re.compile(r"route segment=(?P<segment>\S+); previous=(?P<previous>\S+); .*?decision=(?P<decision>.*)$")
REJOIN_RE = re.compile(
    r"ROUTE_REJOIN teleport actual=\((?P<x>[+-]?\d+\.\d+),(?P<y>[+-]?\d+\.\d+)\) "
    r"cmd_error=(?P<cmd_error>\d+\.\d+)m .*? route_dist=(?P<route_dist>\d+\.\d+)m "
    r"segment=(?P<segment>\S+)"
)
DIRECTION_RE = re.compile(r"SAFE_ROUTE_VERSION=.*? direction=(?P<direction>.*?);")


@dataclass
class MotionEvent:
    route: str
    x: float
    y: float
    yaw: float
    v: float
    step_err_m: float
    yaw_step_rad: float
    track_err_m: float
    cmd_v: float
    cmd_w: float
    line: str


@dataclass
class JitterEvent:
    route: str
    count: int
    reasons: str
    line: str


@dataclass
class RejoinEvent:
    segment: str
    x: float
    y: float
    cmd_error_m: float
    route_dist_m: float
    line: str


def read_watch_text(path: Path, watch_seconds: float, from_end: bool) -> str:
    deadline = time.monotonic() + max(0.0, watch_seconds)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        if from_end:
            handle.seek(0, 2)
        chunks: list[str] = []
        while time.monotonic() < deadline:
            chunk = handle.read()
            if chunk:
                chunks.append(chunk)
            time.sleep(0.25)
        chunk = handle.read()
        if chunk:
            chunks.append(chunk)
    return "".join(chunks)


def analyze_text(text: str) -> dict:
    motions: list[MotionEvent] = []
    jitters: list[JitterEvent] = []
    rejoins: list[RejoinEvent] = []
    segment_changes: list[dict] = []
    direction = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if direction is None:
            match = DIRECTION_RE.search(line)
            if match:
                direction = match.group("direction")
        match = MOTION_RE.search(line)
        if match:
            motions.append(
                MotionEvent(
                    route=match.group("route"),
                    x=float(match.group("x")),
                    y=float(match.group("y")),
                    yaw=float(match.group("yaw")),
                    v=float(match.group("v")),
                    step_err_m=float(match.group("step")),
                    yaw_step_rad=float(match.group("yaw_step")),
                    track_err_m=float(match.group("track")),
                    cmd_v=float(match.group("cmd_v")),
                    cmd_w=float(match.group("cmd_w")),
                    line=line,
                )
            )
            continue
        match = JITTER_RE.search(line)
        if match:
            jitters.append(
                JitterEvent(
                    route=match.group("route"),
                    count=int(match.group("count")),
                    reasons=match.group("reasons"),
                    line=line,
                )
            )
            continue
        match = REJOIN_RE.search(line)
        if match:
            rejoins.append(
                RejoinEvent(
                    segment=match.group("segment"),
                    x=float(match.group("x")),
                    y=float(match.group("y")),
                    cmd_error_m=float(match.group("cmd_error")),
                    route_dist_m=float(match.group("route_dist")),
                    line=line,
                )
            )
            continue
        match = SEGMENT_RE.search(line)
        if match:
            segment_changes.append(match.groupdict())

    route_counts = Counter(event.route for event in motions)
    jitter_routes = Counter()
    jitter_reason_counts = Counter()
    for event in jitters:
        jitter_routes[event.route] += event.count
        for reason in event.reasons.split(","):
            reason_name = reason.split("=", 1)[0].strip()
            if reason_name:
                jitter_reason_counts[reason_name] += event.count

    spin_events = [
        event for event in motions
        if abs(event.cmd_w) >= 1.0 and (abs(event.cmd_v) <= 0.08 or event.v <= 0.08)
    ]
    high_yaw_events = [event for event in motions if event.yaw_step_rad >= 0.20 or abs(event.cmd_w) >= 1.6]
    high_track_events = [event for event in motions if event.track_err_m >= 0.045 or event.step_err_m >= 0.030]

    max_motion = {
        "max_abs_cmd_w": max((abs(event.cmd_w) for event in motions), default=0.0),
        "max_track_err_m": max((event.track_err_m for event in motions), default=0.0),
        "max_step_err_m": max((event.step_err_m for event in motions), default=0.0),
        "max_yaw_step_rad": max((event.yaw_step_rad for event in motions), default=0.0),
    }

    problem_score_by_route: dict[str, float] = defaultdict(float)
    for event in spin_events:
        problem_score_by_route[event.route] += 8.0
    for event in high_yaw_events:
        problem_score_by_route[event.route] += 3.0
    for event in high_track_events:
        problem_score_by_route[event.route] += 2.0
    for event in jitters:
        problem_score_by_route[event.route] += event.count * 0.5
    for event in rejoins:
        problem_score_by_route[event.segment] += 12.0

    top_problem_routes = sorted(
        ({"route": route, "score": round(score, 2)} for route, score in problem_score_by_route.items()),
        key=lambda item: item["score"],
        reverse=True,
    )[:12]

    return {
        "direction": direction,
        "counts": {
            "motion": len(motions),
            "jitter": len(jitters),
            "jitter_events_total": sum(event.count for event in jitters),
            "rejoin": len(rejoins),
            "segment_changes": len(segment_changes),
            "spin_like_motion": len(spin_events),
            "high_yaw_motion": len(high_yaw_events),
            "high_track_error_motion": len(high_track_events),
        },
        "max_motion": max_motion,
        "top_motion_routes": route_counts.most_common(12),
        "top_jitter_routes": jitter_routes.most_common(12),
        "jitter_reasons": jitter_reason_counts.most_common(),
        "top_problem_routes": top_problem_routes,
        "rejoins": [asdict(event) for event in rejoins],
        "spin_samples": [asdict(event) for event in spin_events[:20]],
        "high_yaw_samples": [asdict(event) for event in high_yaw_events[:20]],
        "high_track_samples": [asdict(event) for event in high_track_events[:20]],
        "segment_changes": segment_changes[-40:],
    }


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# Route Run Log Analysis",
        "",
        f"- direction: `{report.get('direction')}`",
        f"- motion lines: `{report['counts']['motion']}`",
        f"- jitter groups/events: `{report['counts']['jitter']}` / `{report['counts']['jitter_events_total']}`",
        f"- rejoin teleports: `{report['counts']['rejoin']}`",
        f"- spin-like motions: `{report['counts']['spin_like_motion']}`",
        f"- high-yaw motions: `{report['counts']['high_yaw_motion']}`",
        f"- high-track-error motions: `{report['counts']['high_track_error_motion']}`",
        "",
        "## Max Motion",
        "",
    ]
    for key, value in report["max_motion"].items():
        lines.append(f"- {key}: `{value:.4f}`")
    lines.extend(["", "## Problem Routes", ""])
    for item in report["top_problem_routes"]:
        lines.append(f"- `{item['route']}` score `{item['score']}`")
    lines.extend(["", "## Jitter Reasons", ""])
    for reason, count in report["jitter_reasons"]:
        lines.append(f"- `{reason}`: `{count}`")
    lines.extend(["", "## Rejoins", ""])
    if report["rejoins"]:
        for event in report["rejoins"]:
            lines.append(
                f"- `{event['segment']}` cmd_error=`{event['cmd_error_m']:.3f}` "
                f"route_dist=`{event['route_dist_m']:.3f}`"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Spin Samples", ""])
    if report["spin_samples"]:
        for event in report["spin_samples"][:8]:
            lines.append(
                f"- `{event['route']}` pos=({event['x']:+.3f},{event['y']:+.3f}) "
                f"v=`{event['v']:.3f}` cmd=({event['cmd_v']:+.3f},{event['cmd_w']:+.3f}) "
                f"track_err=`{event['track_err_m']:.3f}`"
            )
    else:
        lines.append("- none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze or watch gazebo_track_car launch logs for route instability.")
    parser.add_argument("log_file", type=Path)
    parser.add_argument("--watch-seconds", type=float, default=0.0)
    parser.add_argument("--from-end", action="store_true", help="When watching, ignore existing log content.")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.watch_seconds > 0.0:
        text = read_watch_text(args.log_file, args.watch_seconds, args.from_end)
    else:
        text = args.log_file.read_text(encoding="utf-8", errors="replace")
    report = analyze_text(text)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(report, args.output_md)
    print(json.dumps(report["counts"], indent=2))
    if report["top_problem_routes"]:
        print("Top problem routes:")
        for item in report["top_problem_routes"][:8]:
            print(f"  {item['route']}: {item['score']}")
    if report["jitter_reasons"]:
        print("Jitter reasons:")
        for reason, count in report["jitter_reasons"][:8]:
            print(f"  {reason}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
