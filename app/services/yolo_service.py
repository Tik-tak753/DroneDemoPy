from pathlib import Path
from typing import NamedTuple

import cv2
from ultralytics import YOLO


class PredictionResult(NamedTuple):
    annotated_bgr: cv2.typing.MatLike
    detection_count: int
    top_confidence: float


class YoloService:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self._model: YOLO | None = None

    def model_file_exists(self) -> bool:
        return Path(self.model_path).exists()

    def has_loaded_model(self) -> bool:
        return self._model is not None

    def load_model(self) -> tuple[bool, str | None, str | None]:
        try:
            self._model = YOLO(self.model_path)
        except Exception as exc:
            return False, f"Model load failed: {exc}", "Confidence: model load failed"

        return True, None, None

    def predict_single_frame(
        self, frame_bgr: cv2.typing.MatLike
    ) -> tuple[PredictionResult | None, str | None, str | None]:
        if self._model is None:
            return None, "Inference failed: model not loaded", None

        try:
            results = self._model.predict(frame_bgr, verbose=False)
        except Exception as exc:
            return None, f"Inference failed: {exc}", "Confidence: inference failed"

        if not results:
            return None, "Inference completed: no result returned", "Confidence: no result"

        result = results[0]
        annotated_bgr = result.plot()
        boxes = result.boxes
        detection_count = int(len(boxes)) if boxes is not None else 0
        top_confidence = 0.0
        if boxes is not None and boxes.conf is not None and len(boxes.conf) > 0:
            top_confidence = float(boxes.conf.max().item())

        return PredictionResult(annotated_bgr, detection_count, top_confidence), None, None
