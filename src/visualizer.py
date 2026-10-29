"""Frame overlays for detections, tracks, hands, and traffic stats."""

from __future__ import annotations

import cv2
import numpy as np

import config
from src.counter import TrafficCounter
from src.hands import HandDetection
from src.tracker import Track

# MediaPipe hand landmark connections (simplified finger chains)
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def draw_tracks(
    frame: np.ndarray,
    tracks: list[Track],
    counter: TrafficCounter,
    fps: float,
    hands: list[HandDetection] | None = None,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    a, b = counter.line_points(w, h)

    # Counting line
    cv2.line(out, a, b, config.COLORS["line"], 2, cv2.LINE_AA)
    mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2 - 10)
    cv2.putText(
        out,
        "COUNT LINE",
        mid,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        config.COLORS["line"],
        1,
        cv2.LINE_AA,
    )

    for track in tracks:
        color = config.COLORS.get(track.label, (200, 200, 200))
        x1, y1, x2, y2 = track.box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        tag = f"ID {track.track_id} {track.label} {track.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            out,
            tag,
            (x1 + 3, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

        pts = list(track.path)
        for i in range(1, len(pts)):
            thickness = max(1, int(2 * (i / len(pts))))
            cv2.line(out, pts[i - 1], pts[i], color, thickness, cv2.LINE_AA)

        cx, cy = track.centroid
        cv2.circle(out, (cx, cy), 3, color, -1)

    if hands:
        _draw_hands(out, hands)

    _draw_panel(out, counter, fps, len(tracks), len(hands or []))
    return out


def _draw_hands(frame: np.ndarray, hands: list[HandDetection]) -> None:
    color = config.COLORS["hand"]
    for hand in hands:
        cv2.rectangle(frame, (hand.x1, hand.y1), (hand.x2, hand.y2), color, 2)
        tag = f"{hand.label} {hand.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(
            frame,
            (hand.x1, hand.y1 - th - 8),
            (hand.x1 + tw + 6, hand.y1),
            color,
            -1,
        )
        cv2.putText(
            frame,
            tag,
            (hand.x1 + 3, hand.y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

        pts = hand.landmarks
        for a, b in _HAND_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(frame, pts[a], pts[b], color, 1, cv2.LINE_AA)
        for p in pts:
            cv2.circle(frame, p, 2, (255, 255, 255), -1, cv2.LINE_AA)


def _draw_panel(
    frame: np.ndarray,
    counter: TrafficCounter,
    fps: float,
    active: int,
    hand_count: int,
) -> None:
    panel_h = 130
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (340, 10 + panel_h), config.COLORS["panel"], -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    lines = [
        f"Total: {counter.total}",
        f"Up: {counter.up}   Down: {counter.down}",
        f"Active: {active}   Hands: {hand_count}   FPS: {fps:.1f}",
    ]
    class_bits = [f"{k}:{v}" for k, v in sorted(counter.by_class.items())]
    if class_bits:
        lines.append("  ".join(class_bits))

    y = 35
    for text in lines:
        cv2.putText(
            frame,
            text,
            (22, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            config.COLORS["text"],
            1,
            cv2.LINE_AA,
        )
        y += 24
