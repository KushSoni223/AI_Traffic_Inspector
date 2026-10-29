"""
AI Traffic Assistant
--------------------
YOLO-style detection + tracking, implemented without Ultralytics:
  • Detection: YOLOv4-tiny via OpenCV DNN (person + vehicles)
  • Hands:     MediaPipe Hand Landmarker
  • Tracking:  custom centroid + IoU tracker
  • Counting:  line-crossing counter
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

import config
from src.counter import TrafficCounter
from src.detector import YOLODetector, filter_by_min_area
from src.hands import HandDetector
from src.phone_camera import list_adb_devices, open_phone_camera, phone_help_text
from src.tracker import CentroidTracker
from src.visualizer import draw_tracks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Custom YOLO traffic detection, tracking, counting + hands"
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Video path, RTSP/HTTP URL, camera index, or 'phone' (ADB USB)",
    )
    parser.add_argument(
        "--camera-facing",
        choices=("back", "front"),
        default="back",
        help="Phone camera facing when --source phone (scrcpy mode)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save annotated video",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=config.CONFIDENCE_THRESHOLD,
        help="Detection confidence threshold",
    )
    parser.add_argument(
        "--no-hands",
        action="store_true",
        help="Disable MediaPipe hand detection",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run headless (useful when saving output only)",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=0,
        help="Process every Nth frame only (0 = all frames)",
    )
    return parser.parse_args()


def open_capture(source: str, camera_facing: str = "back"):
    """Open camera/video/phone stream; on macOS try AVFoundation first for webcams."""
    if source.lower() == "phone":
        return open_phone_camera(facing=camera_facing)

    if source.isdigit():
        index = int(source)
        backends = []
        if hasattr(cv2, "CAP_AVFOUNDATION"):
            backends.append(cv2.CAP_AVFOUNDATION)
        backends.extend([cv2.CAP_ANY])
        last_err = None
        for backend in backends:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                # Warm up a few frames — some Mac cams need this
                for _ in range(5):
                    ok, _ = cap.read()
                    if ok:
                        return cap
                cap.release()
            last_err = backend

        hint = (
            f"Could not open camera index {source} "
            f"(last backend={last_err}). "
            "This Mac has no usable local camera."
        )
        if list_adb_devices():
            hint += "\n" + phone_help_text()
        else:
            hint += (
                "\nGrant Camera permission, plug in a webcam, "
                "or pass a video: --source path/to/video.mp4"
            )
        raise RuntimeError(hint)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    return cap


def main() -> int:
    args = parse_args()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        detector = YOLODetector(conf_threshold=args.conf)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    hand_detector: HandDetector | None = None
    if not args.no_hands:
        try:
            hand_detector = HandDetector()
            print("Hand detection: ON (MediaPipe)")
        except Exception as exc:  # noqa: BLE001
            print(f"Hand detection disabled: {exc}", file=sys.stderr)

    tracker = CentroidTracker()
    counter = TrafficCounter()

    try:
        cap = open_capture(args.source, camera_facing=args.camera_facing)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    writer = None
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, src_fps, (width, height))

    window = "AI Traffic Assistant"
    if not args.no_display:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    frame_idx = 0
    fps_smooth = 0.0
    last_tracks = []
    last_hands = []

    print(
        "Detecting: person, bicycle, car, motorcycle, bus, truck"
        + (" + hands" if hand_detector else "")
    )
    print("Running. Press Q / Esc to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1
        t0 = time.perf_counter()

        run_detect = args.skip_frames <= 0 or (frame_idx % (args.skip_frames + 1) == 1)
        if run_detect:
            detections = filter_by_min_area(detector.detect(frame))
            last_tracks = tracker.update(detections)
            counter.update(last_tracks, frame.shape[1], frame.shape[0])
            if hand_detector is not None:
                last_hands = hand_detector.detect(frame)

        annotated = draw_tracks(
            frame, last_tracks, counter, fps_smooth, hands=last_hands
        )

        dt = time.perf_counter() - t0
        fps = 1.0 / dt if dt > 0 else 0.0
        fps_smooth = fps if fps_smooth == 0 else (0.85 * fps_smooth + 0.15 * fps)

        if writer is not None:
            writer.write(annotated)

        if not args.no_display:
            cv2.imshow(window, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

    cap.release()
    if hand_detector is not None:
        hand_detector.close()
    if writer is not None:
        writer.release()
        print(f"Saved: {args.output}")
    if not args.no_display:
        cv2.destroyAllWindows()

    print(
        f"Done. Frames={frame_idx}  Total counted={counter.total}  "
        f"Up={counter.up}  Down={counter.down}  By class={dict(counter.by_class)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
