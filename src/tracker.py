"""Custom multi-object tracker (centroid + IoU) – no DeepSORT / ByteTrack deps."""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field

import numpy as np

import config
from src.detector import Detection


def iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def centroid_distance(
    c1: tuple[int, int], c2: tuple[int, int]
) -> float:
    return float(np.hypot(c1[0] - c2[0], c1[1] - c2[1]))


@dataclass
class Track:
    track_id: int
    box: tuple[int, int, int, int]
    centroid: tuple[int, int]
    label: str
    confidence: float
    disappeared: int = 0
    hits: int = 1
    path: deque = field(default_factory=lambda: deque(maxlen=32))
    counted: bool = False
    direction: str | None = None  # "up" | "down"

    def update(self, detection: Detection) -> None:
        self.box = detection.box
        self.centroid = detection.centroid
        self.label = detection.label
        self.confidence = detection.confidence
        self.disappeared = 0
        self.hits += 1
        self.path.append(self.centroid)


class CentroidTracker:
    """
    Assigns stable IDs across frames using IoU first, then centroid distance.
    Similar end result to YOLO+tracker demos, implemented from scratch.
    """

    def __init__(
        self,
        iou_threshold: float = config.IOU_THRESHOLD,
        max_disappeared: int = config.MAX_DISAPPEARED,
        max_distance: float = config.MAX_DISTANCE,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.next_id = 1
        self.tracks: OrderedDict[int, Track] = OrderedDict()

    def update(self, detections: list[Detection]) -> list[Track]:
        if not self.tracks:
            for det in detections:
                self._register(det)
            return list(self.tracks.values())

        if not detections:
            to_remove: list[int] = []
            for track_id, track in self.tracks.items():
                track.disappeared += 1
                if track.disappeared > self.max_disappeared:
                    to_remove.append(track_id)
            for track_id in to_remove:
                del self.tracks[track_id]
            return list(self.tracks.values())

        track_ids = list(self.tracks.keys())
        track_boxes = [self.tracks[tid].box for tid in track_ids]
        det_boxes = [d.box for d in detections]

        iou_matrix = np.zeros((len(track_ids), len(detections)), dtype=np.float32)
        for i, tbox in enumerate(track_boxes):
            for j, dbox in enumerate(det_boxes):
                iou_matrix[i, j] = iou(tbox, dbox)

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()

        # Greedy match by highest IoU
        while True:
            if iou_matrix.size == 0:
                break
            idx = np.unravel_index(int(np.argmax(iou_matrix)), iou_matrix.shape)
            best = float(iou_matrix[idx])
            if best < self.iou_threshold:
                break
            ti, dj = int(idx[0]), int(idx[1])
            if ti in matched_tracks or dj in matched_dets:
                iou_matrix[ti, dj] = -1
                continue
            track = self.tracks[track_ids[ti]]
            track.update(detections[dj])
            matched_tracks.add(ti)
            matched_dets.add(dj)
            iou_matrix[ti, :] = -1
            iou_matrix[:, dj] = -1

        # Fallback: centroid distance for remaining
        unmatched_track_idxs = [i for i in range(len(track_ids)) if i not in matched_tracks]
        unmatched_det_idxs = [j for j in range(len(detections)) if j not in matched_dets]

        if unmatched_track_idxs and unmatched_det_idxs:
            dist_matrix = np.full(
                (len(unmatched_track_idxs), len(unmatched_det_idxs)),
                np.inf,
                dtype=np.float32,
            )
            for a, ti in enumerate(unmatched_track_idxs):
                tc = self.tracks[track_ids[ti]].centroid
                for b, dj in enumerate(unmatched_det_idxs):
                    dist_matrix[a, b] = centroid_distance(tc, detections[dj].centroid)

            used_a: set[int] = set()
            used_b: set[int] = set()
            while True:
                if dist_matrix.size == 0:
                    break
                idx = np.unravel_index(int(np.argmin(dist_matrix)), dist_matrix.shape)
                best = float(dist_matrix[idx])
                if best > self.max_distance:
                    break
                a, b = int(idx[0]), int(idx[1])
                if a in used_a or b in used_b:
                    dist_matrix[a, b] = np.inf
                    continue
                ti = unmatched_track_idxs[a]
                dj = unmatched_det_idxs[b]
                self.tracks[track_ids[ti]].update(detections[dj])
                matched_tracks.add(ti)
                matched_dets.add(dj)
                used_a.add(a)
                used_b.add(b)
                dist_matrix[a, :] = np.inf
                dist_matrix[:, b] = np.inf

        # Mark unmatched tracks as disappeared
        to_remove = []
        for i, track_id in enumerate(track_ids):
            if i not in matched_tracks:
                self.tracks[track_id].disappeared += 1
                if self.tracks[track_id].disappeared > self.max_disappeared:
                    to_remove.append(track_id)
        for track_id in to_remove:
            del self.tracks[track_id]

        # Register new detections
        for j, det in enumerate(detections):
            if j not in matched_dets:
                self._register(det)

        return [t for t in self.tracks.values() if t.disappeared == 0]

    def _register(self, detection: Detection) -> None:
        track = Track(
            track_id=self.next_id,
            box=detection.box,
            centroid=detection.centroid,
            label=detection.label,
            confidence=detection.confidence,
        )
        track.path.append(detection.centroid)
        self.tracks[self.next_id] = track
        self.next_id += 1
