"""Line-crossing traffic / pedestrian counter."""

from __future__ import annotations

from collections import defaultdict

import config
from src.tracker import Track


class TrafficCounter:
    """Counts unique tracks that cross a horizontal (or angled) counting line."""

    def __init__(
        self,
        line_frac: tuple[float, float, float, float] = config.COUNT_LINE,
        countable_labels: set[str] | None = None,
    ) -> None:
        self.line_frac = line_frac
        self.countable_labels = countable_labels or config.COUNTABLE_LABELS
        self.total = 0
        self.by_class: dict[str, int] = defaultdict(int)
        self.up = 0
        self.down = 0

    def line_points(self, frame_w: int, frame_h: int) -> tuple[tuple[int, int], tuple[int, int]]:
        x1 = int(self.line_frac[0] * frame_w)
        y1 = int(self.line_frac[1] * frame_h)
        x2 = int(self.line_frac[2] * frame_w)
        y2 = int(self.line_frac[3] * frame_h)
        return (x1, y1), (x2, y2)

    @staticmethod
    def _side(px: int, py: int, a: tuple[int, int], b: tuple[int, int]) -> float:
        """Signed cross product: which side of line AB is point P on."""
        return (b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0])

    def update(
        self,
        tracks: list[Track],
        frame_w: int,
        frame_h: int,
    ) -> list[Track]:
        a, b = self.line_points(frame_w, frame_h)
        newly_counted: list[Track] = []

        for track in tracks:
            if track.counted or len(track.path) < 2:
                continue
            if track.label not in self.countable_labels:
                continue

            prev = track.path[-2]
            curr = track.path[-1]
            s0 = self._side(prev[0], prev[1], a, b)
            s1 = self._side(curr[0], curr[1], a, b)

            # Crossed the line when sign flips (and not exactly on the line)
            if s0 == 0 or s1 == 0:
                continue
            if (s0 > 0) == (s1 > 0):
                continue

            # Also require the segment to be near the line span in x
            min_x, max_x = min(a[0], b[0]), max(a[0], b[0])
            if not (min_x - 20 <= curr[0] <= max_x + 20):
                continue

            track.counted = True
            if curr[1] > prev[1]:
                track.direction = "down"
                self.down += 1
            else:
                track.direction = "up"
                self.up += 1

            self.total += 1
            self.by_class[track.label] += 1
            newly_counted.append(track)

        return newly_counted
