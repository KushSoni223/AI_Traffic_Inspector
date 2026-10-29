# AI Traffic Assistant

YOLO-style **vehicle detection + multi-object tracking + counting**, built **without Ultralytics**.

| Piece | Implementation |
|-------|----------------|
| Detection | YOLOv4-tiny through **OpenCV DNN** — person, bicycle, car, motorcycle, bus, truck |
| Hands | **MediaPipe** Hand Landmarker (YOLO/COCO has no hand class) |
| Tracking | Custom **centroid + IoU** tracker |
| Counting | Custom **line-crossing** counter |

## Setup

```bash
cd AI_Traffic_Assistant
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_models.py
```

## Run

**Android phone over USB debugging** (your Mac Mini has no camera):

```bash
# Recommended — uses ADB + scrcpy camera (no phone app)
brew install scrcpy
python main.py --source phone
python main.py --source phone --camera-facing front
```

Or install **IP Webcam** on the phone → tap **Start server**, then:

```bash
python main.py --source phone
```

Video file:

```bash
python main.py --source path/to/traffic.mp4 --output output/tracked.mp4
```

Local webcam (if available):

```bash
python main.py --source 0
```

Options:

```text
--conf 0.4          confidence threshold
--skip-frames 1     detect every other frame (faster)
--no-hands          disable hand landmark detection
--no-display        headless (write output only)
```

**No camera?** Your Mac Mini reported no video input. Use a video file instead, or enable Camera permission for Terminal/Cursor under System Settings → Privacy & Security → Camera.

Press **Q** or **Esc** to quit.

## Project layout

```text
main.py                 entry point
config.py               thresholds, paths, colors
scripts/download_models.py
src/
  detector.py           OpenCV DNN YOLO
  tracker.py            custom ID tracker
  counter.py            line crossing
  visualizer.py         overlays
models/                 yolov4-tiny + coco.names (downloaded)
```

## How tracking works

1. YOLO predicts vehicle boxes each frame.
2. The tracker matches boxes to existing IDs with **IoU**, then **centroid distance**.
3. Lost IDs stay alive for a few frames, then drop.
4. When a track’s path crosses the yellow count line, it increments **once**.
