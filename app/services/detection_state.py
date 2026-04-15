from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import cv2


@dataclass
class StreamDetectionState:
    current_source_type: str = "none"
    video_detection_enabled: bool = False
    camera_detection_enabled: bool = False
    screen_detection_enabled: bool = False
    video_paused: bool = False
    camera_inference_failures: int = 0

    stream_session_id: int = 0
    stream_frame_id: int = 0
    last_accepted_result_frame_id: int = -1
    last_stream_frame_bgr: cv2.typing.MatLike | None = None
    last_stream_frame_id: int = -1

    displayed_frames: int = 0
    fps_window_start: float = perf_counter()

    def reset_detection_toggles(self) -> None:
        self.video_detection_enabled = False
        self.camera_detection_enabled = False
        self.screen_detection_enabled = False
        self.camera_inference_failures = 0
        self.stream_frame_id = 0
        self.last_accepted_result_frame_id = -1

    def next_session(self) -> None:
        self.stream_session_id += 1
        self.stream_frame_id = 0
        self.last_accepted_result_frame_id = -1
        self.last_stream_frame_bgr = None
        self.last_stream_frame_id = -1

    def is_detection_enabled_for_source(self, source_type: str) -> bool:
        if source_type == "video":
            return self.video_detection_enabled
        if source_type == "camera":
            return self.camera_detection_enabled
        if source_type == "screen":
            return self.screen_detection_enabled
        return False

    def runtime_detection_text(self) -> str:
        return "ON" if self.is_detection_enabled_for_source(self.current_source_type) else "OFF"

    def start_fps_window(self) -> None:
        self.displayed_frames = 0
        self.fps_window_start = perf_counter()

    def mark_frame_displayed(self) -> float | None:
        self.displayed_frames += 1
        elapsed = perf_counter() - self.fps_window_start
        if elapsed < 0.5:
            return None

        fps = self.displayed_frames / elapsed
        self.displayed_frames = 0
        self.fps_window_start = perf_counter()
        return fps
