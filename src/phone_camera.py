"""Use an Android phone as the camera over USB debugging (ADB).

USB debugging alone does NOT create a Mac webcam device. This module:

1. Prefer scrcpy camera mode (no phone app) if `scrcpy` is on PATH
2. Else ADB-reverse + MJPEG apps: IP Webcam / DroidCam
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np


# Common phone streaming apps over adb reverse
MJPEG_CANDIDATES = (
    "http://127.0.0.1:8080/video",  # IP Webcam
    "http://127.0.0.1:8080/video.mjpg",
    "http://127.0.0.1:4747/video",  # DroidCam
    "http://127.0.0.1:4747/mjpegfeed?640x480",
)

REVERSE_PORTS = (8080, 4747, 8081)


@dataclass
class PhoneCapture:
    """Wrapper so main can treat phone streams like cv2.VideoCapture."""

    cap: cv2.VideoCapture
    kind: str
    cleanup: list

    def read(self):
        return self.cap.read()

    def isOpened(self) -> bool:
        return self.cap.isOpened()

    def get(self, prop: int):
        return self.cap.get(prop)

    def release(self) -> None:
        self.cap.release()
        for fn in self.cleanup:
            try:
                fn()
            except Exception:  # noqa: BLE001
                pass


def adb_bin() -> str | None:
    return shutil.which("adb")


def list_adb_devices() -> list[str]:
    adb = adb_bin()
    if not adb:
        return []
    try:
        out = subprocess.check_output(
            [adb, "devices"], text=True, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    devices = []
    for line in out.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            devices.append(parts[0])
    return devices


def adb_reverse_ports(ports: tuple[int, ...] = REVERSE_PORTS) -> None:
    adb = adb_bin()
    if not adb:
        return
    for port in ports:
        subprocess.run(
            [adb, "reverse", f"tcp:{port}", f"tcp:{port}"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def _port_open(host: str, port: int, timeout: float = 0.2) -> bool:
    import socket

    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _try_mjpeg_urls(urls: tuple[str, ...] = MJPEG_CANDIDATES) -> cv2.VideoCapture | None:
    # Skip URLs whose port is closed to avoid OpenCV spam
    live_ports = {p for p in REVERSE_PORTS if _port_open("127.0.0.1", p)}
    if not live_ports:
        return None

    for url in urls:
        port = 8080 if ":8080" in url else 4747 if ":4747" in url else None
        if port is not None and port not in live_ports:
            continue
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            cap.release()
            continue
        ok, frame = cap.read()
        if ok and frame is not None:
            return cap
        cap.release()
    return None


def _open_scrcpy_camera(
    width: int = 1280,
    height: int = 720,
    facing: str = "back",
) -> PhoneCapture | None:
    """Stream phone camera via scrcpy → ffmpeg → raw BGR pipe (no extra phone app)."""
    scrcpy = shutil.which("scrcpy")
    ffmpeg = shutil.which("ffmpeg")
    if not scrcpy or not ffmpeg:
        return None

    scrcpy_cmd = [
        scrcpy,
        "--video-source=camera",
        f"--camera-facing={facing}",
        f"--camera-size={width}x{height}",
        "--no-audio",
        "--no-playback",
        "--record=-",
        "--record-format=mkv",
    ]
    ffmpeg_cmd = [
        ffmpeg,
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-an",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "pipe:1",
    ]

    try:
        scrcpy_proc = subprocess.Popen(
            scrcpy_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=scrcpy_proc.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        if scrcpy_proc.stdout:
            scrcpy_proc.stdout.close()
    except OSError:
        return None

    frame_bytes = width * height * 3

    class _ScrcpyCap:
        def __init__(self) -> None:
            self._w = width
            self._h = height

        def isOpened(self) -> bool:
            return ffmpeg_proc.poll() is None and scrcpy_proc.poll() is None

        def read(self):
            if ffmpeg_proc.stdout is None:
                return False, None
            raw = ffmpeg_proc.stdout.read(frame_bytes)
            if not raw or len(raw) < frame_bytes:
                return False, None
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3)).copy()
            return True, frame

        def get(self, prop: int):
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return float(width)
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return float(height)
            if prop == cv2.CAP_PROP_FPS:
                return 25.0
            return 0.0

        def release(self) -> None:
            for proc in (ffmpeg_proc, scrcpy_proc):
                try:
                    proc.terminate()
                    proc.wait(timeout=2)
                except Exception:  # noqa: BLE001
                    try:
                        proc.kill()
                    except Exception:  # noqa: BLE001
                        pass

    # Warm up a frame
    cap = _ScrcpyCap()
    time.sleep(1.5)
    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None

    return PhoneCapture(cap=cap, kind="scrcpy", cleanup=[])  # type: ignore[arg-type]


def open_phone_camera(facing: str = "back") -> PhoneCapture:
    devices = list_adb_devices()
    if not devices:
        raise RuntimeError(
            "No Android device in `adb devices`.\n"
            "Enable USB debugging, plug in the phone, and accept the RSA prompt."
        )

    print(f"ADB device: {devices[0]}", file=sys.stderr)

    # 1) scrcpy path (best — uses USB debugging only, no phone app)
    phone = _open_scrcpy_camera(facing=facing)
    if phone is not None:
        print("Phone camera via scrcpy (USB).", file=sys.stderr)
        return phone

    # 2) MJPEG apps over adb reverse
    adb_reverse_ports()
    print("Waiting for phone MJPEG stream on localhost:8080 / :4747 ...", file=sys.stderr)
    for _ in range(8):
        cap = _try_mjpeg_urls()
        if cap is not None:
            print("Phone camera via MJPEG (IP Webcam / DroidCam).", file=sys.stderr)
            return PhoneCapture(cap=cap, kind="mjpeg", cleanup=[])
        time.sleep(0.5)

    scrcpy_hint = ""
    if not shutil.which("scrcpy"):
        scrcpy_hint = (
            "\n  Option A (recommended): brew install scrcpy\n"
            "    then re-run: python main.py --source phone\n"
        )

    raise RuntimeError(
        "Phone is on ADB, but no camera stream was found."
        + scrcpy_hint
        + "\n  Option B: install 'IP Webcam' on the phone → Start server,\n"
        "    then re-run: python main.py --source phone\n"
        "    (ADB reverse is already set for ports 8080 / 4747)\n"
    )


def phone_help_text() -> str:
    return (
        "This Mac has no local camera. Your phone is on USB debugging — use:\n"
        "  python main.py --source phone\n"
        "If that fails: brew install scrcpy   OR   start IP Webcam on the phone."
    )
