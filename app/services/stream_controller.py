from __future__ import annotations

from dataclasses import dataclass

import cv2
import mss
import numpy as np
from PySide6.QtCore import QObject, QTimer


@dataclass
class StreamReadResult:
    kind: str
    frame_bgr: cv2.typing.MatLike | None = None
    message: str | None = None


class StreamController(QObject):
    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.video_capture: cv2.VideoCapture | None = None
        self.screen_capture: mss.base.MSSBase | None = None
        self.screen_region: dict[str, int] | None = None

        self.video_timer = QTimer(self)
        self.video_interval_ms = 33

    def open_video(self, file_path: str) -> bool:
        capture = cv2.VideoCapture(file_path)
        if not capture.isOpened():
            capture.release()
            return False
        self.video_capture = capture
        return True

    def open_camera(self, index: int = 0) -> bool:
        capture = cv2.VideoCapture(index)
        if not capture.isOpened():
            capture.release()
            return False
        self.video_capture = capture
        return True

    def start_screen_capture(self, region: dict[str, int]) -> tuple[bool, str | None]:
        try:
            self.screen_capture = mss.mss()
        except Exception as exc:
            self.screen_capture = None
            return False, str(exc)
        self.screen_region = region
        return True, None

    def start_playback(self) -> None:
        fps = 30.0
        if self.video_capture is not None:
            capture_fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            if 0 < capture_fps <= 240:
                fps = capture_fps

        self.video_interval_ms = max(1, int(1000 / fps))
        self.video_timer.start(self.video_interval_ms)

    def stop_playback(self) -> None:
        self.video_timer.stop()

    def read_next_frame(self, source_type: str) -> StreamReadResult:
        if source_type in {"video", "camera"}:
            if self.video_capture is None:
                return StreamReadResult("inactive")
            success, frame_bgr = self.video_capture.read()
            if not success or frame_bgr is None:
                return StreamReadResult("eof")
            return StreamReadResult("ok", frame_bgr=frame_bgr)

        if source_type == "screen":
            return self._read_next_screen_frame()

        return StreamReadResult("inactive")

    def _read_next_screen_frame(self) -> StreamReadResult:
        if self.screen_capture is None or self.screen_region is None:
            return StreamReadResult("inactive")

        try:
            raw_frame = self.screen_capture.grab(self.screen_region)
        except Exception as exc:
            return StreamReadResult("error", message=f"Screen capture failed: {exc}")

        frame_bgra = np.asarray(raw_frame)
        if frame_bgra.size == 0:
            return StreamReadResult("error", message="Screen capture failed: empty frame")

        return StreamReadResult("ok", frame_bgr=cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR))

    def release(self) -> tuple[bool, bool]:
        had_video = self.video_capture is not None
        had_screen = self.screen_capture is not None

        self.video_timer.stop()

        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None

        if self.screen_capture is not None:
            self.screen_capture.close()
            self.screen_capture = None
            self.screen_region = None

        return had_video, had_screen
