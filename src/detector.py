"""Custom YOLO detector using OpenCV DNN (no Ultralytics).

Detects person + vehicles from COCO. Hands are handled separately (MediaPipe).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

import config


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    label: str

    @property
    def centroid(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


class YOLODetector:
    """YOLOv4-tiny inference through OpenCV's DNN module."""

    def __init__(
        self,
        cfg_path: Path | str = config.YOLO_CFG,
        weights_path: Path | str = config.YOLO_WEIGHTS,
        names_path: Path | str = config.YOLO_NAMES,
        input_size: tuple[int, int] = config.INPUT_SIZE,
        conf_threshold: float = config.CONFIDENCE_THRESHOLD,
        nms_threshold: float = config.NMS_THRESHOLD,
        detect_classes: dict[int, str] | None = None,
        vehicle_classes: dict[int, str] | None = None,  # alias
    ) -> None:
        self.cfg_path = Path(cfg_path)
        self.weights_path = Path(weights_path)
        self.names_path = Path(names_path)
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.detect_classes = (
            detect_classes or vehicle_classes or config.DETECT_CLASSES
        )
        self.vehicle_classes = self.detect_classes  # back-compat

        self._ensure_model_files()
        self.class_names = self._load_names()
        self.net = cv2.dnn.readNetFromDarknet(
            str(self.cfg_path), str(self.weights_path)
        )
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        layer_names = self.net.getLayerNames()
        # Stubs type getUnconnectedOutLayers() as Sequence[int] (no .flatten);
        # wrap with np.asarray so 1-based indices work across OpenCV versions.
        unconnected = np.asarray(self.net.getUnconnectedOutLayers()).reshape(-1)
        self.output_layers = [layer_names[int(i) - 1] for i in unconnected]

    def _ensure_model_files(self) -> None:
        missing = [
            p.name
            for p in (self.cfg_path, self.weights_path, self.names_path)
            if not p.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing YOLO model files: "
                + ", ".join(missing)
                + ". Run: python scripts/download_models.py"
            )

    def _load_names(self) -> list[str]:
        return [
            line.strip()
            for line in self.names_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def detect(self, frame: np.ndarray) -> list[Detection]:
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1 / 255.0,
            size=self.input_size,
            mean=(0, 0, 0),
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        return self._decode(outputs, width, height)

    def _decode(
        self,
        outputs: Sequence[np.ndarray],
        width: int,
        height: int,
    ) -> list[Detection]:
        boxes: list[list[int]] = []
        confidences: list[float] = []
        class_ids: list[int] = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])

                if class_id not in self.vehicle_classes:
                    continue
                if confidence < self.conf_threshold:
                    continue

                cx = detection[0] * width
                cy = detection[1] * height
                w = detection[2] * width
                h = detection[3] * height
                x1 = int(cx - w / 2)
                y1 = int(cy - h / 2)
                boxes.append([x1, y1, int(w), int(h)])
                confidences.append(confidence)
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, self.conf_threshold, self.nms_threshold
        )

        results: list[Detection] = []
        if len(indices) == 0:
            return results

        for i in np.array(indices).flatten():
            x, y, w, h = boxes[i]
            class_id = class_ids[i]
            label = self.vehicle_classes.get(
                class_id, self.class_names[class_id] if class_id < len(self.class_names) else str(class_id)
            )
            results.append(
                Detection(
                    x1=max(0, x),
                    y1=max(0, y),
                    x2=min(width - 1, x + w),
                    y2=min(height - 1, y + h),
                    confidence=confidences[i],
                    class_id=class_id,
                    label=label,
                )
            )
        return results


def filter_by_min_area(
    detections: Iterable[Detection], min_area: int = 400
) -> list[Detection]:
    return [d for d in detections if d.area >= min_area]
