"""Runtime settings for the traffic assistant."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "output"

# YOLOv4-tiny via OpenCV DNN (not Ultralytics)
YOLO_CFG = MODELS_DIR / "yolov4-tiny.cfg"
YOLO_WEIGHTS = MODELS_DIR / "yolov4-tiny.weights"
YOLO_NAMES = MODELS_DIR / "coco.names"
HAND_MODEL = MODELS_DIR / "hand_landmarker.task"

INPUT_SIZE = (416, 416)
CONFIDENCE_THRESHOLD = 0.35
NMS_THRESHOLD = 0.4

# COCO classes we care about (YOLO has no "hand" class — hands use MediaPipe)
DETECT_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Back-compat alias
VEHICLE_CLASSES = DETECT_CLASSES

# Classes that increment the traffic counter when crossing the line
COUNTABLE_LABELS = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
}

# Custom tracker
IOU_THRESHOLD = 0.3
MAX_DISAPPEARED = 30
MAX_DISTANCE = 80

# Counting line as fractions of frame size: (x1, y1, x2, y2)
COUNT_LINE = (0.05, 0.55, 0.95, 0.55)

# Colors BGR
COLORS = {
    "person": (50, 200, 255),
    "bicycle": (200, 160, 80),
    "car": (66, 185, 131),
    "motorcycle": (255, 178, 102),
    "bus": (80, 140, 255),
    "truck": (180, 105, 255),
    "hand": (0, 165, 255),
    "line": (0, 220, 255),
    "text": (240, 240, 240),
    "panel": (28, 28, 28),
}
