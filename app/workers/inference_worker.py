from __future__ import annotations

import threading

import cv2
from PySide6.QtCore import QObject, Signal, Slot

from app.services import YoloService


class InferenceWorker(QObject):
    inference_ready = Signal(object, object, object, str, int)

    def __init__(self, model_path: str) -> None:
        super().__init__()
        self._service = YoloService(model_path)
        self._lock = threading.Lock()
        self._pending_frame: cv2.typing.MatLike | None = None
        self._pending_source_type = "none"
        self._pending_request_id = 0
        self._processing = False

    @Slot(object, str, int)
    def submit_frame(self, frame_bgr: cv2.typing.MatLike, source_type: str, request_id: int) -> None:
        with self._lock:
            self._pending_frame = frame_bgr
            self._pending_source_type = source_type
            self._pending_request_id = request_id
            if self._processing:
                return
            self._processing = True

        while True:
            with self._lock:
                frame = self._pending_frame
                source = self._pending_source_type
                current_request_id = self._pending_request_id
                self._pending_frame = None

            if frame is None:
                with self._lock:
                    self._processing = False
                return

            if not self._service.has_loaded_model():
                loaded, status_message, confidence_message = self._service.load_model()
                if not loaded:
                    self.inference_ready.emit(
                        None,
                        status_message,
                        confidence_message,
                        source,
                        current_request_id,
                    )
                    continue

            prediction, status_message, confidence_message = self._service.predict_single_frame(frame)
            self.inference_ready.emit(
                prediction,
                status_message,
                confidence_message,
                source,
                current_request_id,
            )

    @Slot()
    def reset_pending(self) -> None:
        with self._lock:
            self._pending_frame = None
