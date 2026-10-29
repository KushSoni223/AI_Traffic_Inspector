"""Download YOLOv4-tiny Darknet weights + cfg + COCO names for OpenCV DNN."""

from __future__ import annotations

import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config

FILES = {
    "yolov4-tiny.weights": "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights",
    "yolov4-tiny.cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
    "coco.names": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names",
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
}


def download(url: str, dest: Path) -> None:
    print(f"Downloading {dest.name} ...")
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        done = 0
        with dest.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1024 * 256):
                if not chunk:
                    continue
                fh.write(chunk)
                done += len(chunk)
                if total:
                    pct = done * 100 / total
                    print(f"\r  {pct:5.1f}% ({done / 1e6:.1f} MB)", end="", flush=True)
        print()


def main() -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dest = config.MODELS_DIR / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"Skip (exists): {dest.name}")
            continue
        download(url, dest)
    print("Models ready in", config.MODELS_DIR)


if __name__ == "__main__":
    main()
