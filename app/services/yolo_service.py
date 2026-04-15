from pathlib import Path
from typing import NamedTuple

import cv2
from ultralytics import YOLO


class PredictionResult(NamedTuple):
    annotated_bgr: cv2.typing.MatLike
    detection_count: int
    top_confidence: float
    detections: list[tuple[float, float, float, float, str, float]]


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
        self, frame_bgr: cv2.typing.MatLike, conf_threshold: float = 0.25
    ) -> tuple[PredictionResult | None, str | None, str | None]:
        if self._model is None:
            return None, "Inference failed: model not loaded", None

        try:
            results = self._model.predict(frame_bgr, verbose=False, conf=conf_threshold)
        except Exception as exc:
            return None, f"Inference failed: {exc}", "Confidence: inference failed"

        if not results:
            return None, "Inference completed: no result returned", "Confidence: no result"

        result = results[0]
        annotated_bgr = result.plot()
        boxes = result.boxes
        detection_count = int(len(boxes)) if boxes is not None else 0
        top_confidence = 0.0
        detections: list[tuple[float, float, float, float, str, float]] = []
        if boxes is not None and boxes.conf is not None and len(boxes.conf) > 0:
            top_confidence = float(boxes.conf.max().item())
            class_names = result.names or {}
            xyxy_values = boxes.xyxy.cpu().tolist()
            confidence_values = boxes.conf.cpu().tolist()
            class_values = boxes.cls.cpu().tolist() if boxes.cls is not None else [None] * len(xyxy_values)

            for i, coords in enumerate(xyxy_values):
                if len(coords) != 4:
                    continue
                class_id = int(class_values[i]) if class_values[i] is not None else -1
                class_name = class_names.get(class_id, str(class_id)) if class_id >= 0 else "object"
                detections.append(
                    (
                        float(coords[0]),
                        float(coords[1]),
                        float(coords[2]),
                        float(coords[3]),
                        class_name,
                        float(confidence_values[i]),
                    )
                )
        return PredictionResult(annotated_bgr, detection_count, top_confidence, detections), None, None
