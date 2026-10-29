"""Hand detection via MediaPipe Hand Landmarker (COCO YOLO has no hand class)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import config

try:
    import mediapipe as mp

    # Public surface: mp.tasks is mediapipe.tasks.python
    _BaseOptions = mp.tasks.BaseOptions
    _HandLandmarker = mp.tasks.vision.HandLandmarker
    _HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    _RunningMode = mp.tasks.vision.RunningMode
    _MpImage = mp.Image
    _ImageFormat = mp.ImageFormat
    _MP_OK = True
except ImportError:
    mp = None  # type: ignore[assignment]
    _MP_OK = False


@dataclass
class HandDetection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    handedness: str  # "Left" | "Right"
    landmarks: list[tuple[int, int]]

    @property
    def label(self) -> str:
        return f"hand-{self.handedness.lower()}"


class HandDetector:
    """Detects hands and 21 landmarks per hand."""

    def __init__(
        self,
        model_path: Path | str = config.HAND_MODEL,
        max_hands: int = 4,
        min_confidence: float = 0.5,
    ) -> None:
        if not _MP_OK:
            raise ImportError(
                "mediapipe is required for hand detection. "
                "Install with: pip install mediapipe"
            )
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing hand model: {model_path}. "
                "Run: python scripts/download_models.py"
            )

        options = _HandLandmarkerOptions(
            base_options=_BaseOptions(model_asset_path=str(model_path)),
            running_mode=_RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._landmarker: Any = _HandLandmarker.create_from_options(options)
        self._frame_ts_ms = 0

    def detect(self, frame_bgr: np.ndarray) -> list[HandDetection]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = _MpImage(image_format=_ImageFormat.SRGB, data=rgb)

        self._frame_ts_ms += 33  # ~30 FPS clock for VIDEO mode
        result = self._landmarker.detect_for_video(mp_image, self._frame_ts_ms)

        hands: list[HandDetection] = []
        if not result.hand_landmarks:
            return hands

        handedness_list = result.handedness or []
        for i, landmarks in enumerate(result.hand_landmarks):
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            pad = 12
            x1 = max(0, min(xs) - pad)
            y1 = max(0, min(ys) - pad)
            x2 = min(w - 1, max(xs) + pad)
            y2 = min(h - 1, max(ys) + pad)

            label = "Right"
            score = 0.5
            if i < len(handedness_list) and handedness_list[i]:
                cat = handedness_list[i][0]
                label = cat.category_name
                score = float(cat.score)

            hands.append(
                HandDetection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=score,
                    handedness=label,
                    landmarks=pts,
                )
            )
        return hands

    def close(self) -> None:
        self._landmarker.close()
