from pathlib import Path
from time import perf_counter

import cv2
from PySide6.QtCore import QThread, QTimer, Signal
from PySide6.QtGui import QCloseEvent, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.services import YoloService, bgr_to_qpixmap
from app.widgets import ImageView
from app.workers import InferenceWorker

# Simple first-step model config for still-image detection.
YOLO_MODEL_PATH = r"C:\projects\DroneDemoPy\models\best.pt"


class MainWindow(QMainWindow):
    request_inference = Signal(object, str, int, int, float)
    reset_worker_pending = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DroneDemoPy")
        self.resize(1000, 700)

        self.source_label: QLabel | None = None
        self.confidence_label: QLabel | None = None
        self.mode_state_label: QLabel | None = None
        self.fps_label: QLabel | None = None
        self.confidence_spinbox: QDoubleSpinBox | None = None

        self._video_capture: cv2.VideoCapture | None = None
        self._video_timer = QTimer(self)
        self._video_timer.timeout.connect(self._read_next_video_frame)

        self._current_source_type = "none"
        self._current_image_bgr: cv2.typing.MatLike | None = None
        self._video_detection_enabled = False
        self._camera_detection_enabled = False
        self._camera_inference_failures = 0

        self._stream_session_id = 0
        self._stream_frame_id = 0
        self._last_accepted_result_frame_id = -1
        self._last_stream_frame_bgr: cv2.typing.MatLike | None = None
        self._last_stream_frame_id = -1
        self._displayed_frames = 0
        self._fps_window_start = perf_counter()

        self._yolo_service = YoloService(YOLO_MODEL_PATH)
        self._setup_inference_worker()

        self._create_central_widget()
        self._create_menu_bar()
        self.statusBar().showMessage("Ready")

    def _setup_inference_worker(self) -> None:
        self._inference_thread = QThread(self)
        self._inference_worker = InferenceWorker(YOLO_MODEL_PATH)
        self._inference_worker.moveToThread(self._inference_thread)

        self.request_inference.connect(self._inference_worker.submit_frame)
        self.reset_worker_pending.connect(self._inference_worker.reset_pending)
        self._inference_worker.inference_ready.connect(self._handle_stream_inference_result)

        self._inference_thread.start()

    def _create_central_widget(self) -> None:
        root = QWidget(self)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        self.image_view = ImageView(root)
        layout.addWidget(self.image_view, stretch=3)

        control_panel = self._create_control_panel(root)
        layout.addWidget(control_panel, stretch=1)

        self.setCentralWidget(root)

    def _create_control_panel(self, parent: QWidget) -> QWidget:
        panel = QFrame(parent)
        panel.setFrameShape(QFrame.StyledPanel)

        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(12, 12, 12, 12)
        panel_layout.setSpacing(8)

        open_image_button = QPushButton("Open Image", panel)
        open_image_button.clicked.connect(self._open_image)
        panel_layout.addWidget(open_image_button)

        open_video_button = QPushButton("Open Video", panel)
        open_video_button.clicked.connect(self._open_video)
        panel_layout.addWidget(open_video_button)

        start_camera_button = QPushButton("Start Camera", panel)
        start_camera_button.clicked.connect(self._start_camera)
        panel_layout.addWidget(start_camera_button)

        run_detection_button = QPushButton("Run Detection", panel)
        run_detection_button.clicked.connect(self._run_detection)
        panel_layout.addWidget(run_detection_button)

        stop_button = QPushButton("Stop", panel)
        stop_button.clicked.connect(self._stop_active_stream)
        panel_layout.addWidget(stop_button)

        panel_layout.addSpacing(16)

        confidence_threshold_label = QLabel("Confidence Threshold:", panel)
        panel_layout.addWidget(confidence_threshold_label)

        self.confidence_spinbox = QDoubleSpinBox(panel)
        self.confidence_spinbox.setRange(0.05, 0.95)
        self.confidence_spinbox.setSingleStep(0.05)
        self.confidence_spinbox.setDecimals(2)
        self.confidence_spinbox.setValue(0.25)
        panel_layout.addWidget(self.confidence_spinbox)

        self.confidence_label = QLabel("Confidence: -", panel)
        panel_layout.addWidget(self.confidence_label)

        self.source_label = QLabel("Source: none", panel)
        panel_layout.addWidget(self.source_label)

        self.mode_state_label = QLabel("Mode: none | Detection: OFF", panel)
        panel_layout.addWidget(self.mode_state_label)

        self.fps_label = QLabel("FPS: -", panel)
        panel_layout.addWidget(self.fps_label)

        panel_layout.addStretch()
        panel.setMaximumWidth(260)
        panel.setMinimumWidth(220)

        return panel

    def _create_menu_bar(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction("Exit", self.close)

        help_menu = self.menuBar().addMenu("Help")
        help_menu.addAction("About", self._show_about)

    def _show_about(self) -> None:
        QMessageBox.about(self, "About", "DroneDemoPy - PySide6 demo application")

    def _open_image(self) -> None:
        self._release_video_resources()
        self._reset_detection_state()

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)",
        )

        if not file_path:
            self._set_status("Image open canceled")
            return

        image_bgr = cv2.imread(file_path)
        if image_bgr is None:
            self._set_status("Failed to load image")
            return

        pixmap = QPixmap(file_path)
        if pixmap.isNull():
            self._set_status("Failed to display image")
            return

        self._current_source_type = "image"
        self._current_image_bgr = image_bgr

        self.image_view.set_pixmap(pixmap)
        self._mark_frame_displayed()
        if self.source_label is not None:
            self.source_label.setText(f"Source: {Path(file_path).name} (image)")
        if self.confidence_label is not None:
            self.confidence_label.setText("Confidence: -")
        self._update_runtime_state_label()
        self._set_fps_idle()

        self._set_status(f"Loaded image: {file_path}")

    def _open_video(self) -> None:
        self._release_video_resources()
        self._reset_detection_state()

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video",
            "",
            "Videos (*.mp4 *.avi *.mov *.mkv);;All Files (*)",
        )

        if not file_path:
            self._set_status("Video open canceled")
            return

        capture = cv2.VideoCapture(file_path)
        if not capture.isOpened():
            capture.release()
            self._set_status("Failed to load video")
            return

        self._current_source_type = "video"
        self._current_image_bgr = None

        self._video_capture = capture
        if self.source_label is not None:
            self.source_label.setText(f"Source: {Path(file_path).name} (video)")
        if self.confidence_label is not None:
            self.confidence_label.setText("Confidence: -")
        self._update_runtime_state_label()
        self._set_fps_idle()

        self._set_status(f"Video loaded: {file_path}")
        self._start_video_playback()

    def _start_camera(self) -> None:
        # Simple behavior: always restart camera cleanly when button is pressed.
        self._release_video_resources()
        self._reset_detection_state()

        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            capture.release()
            self._set_status("Camera failed to open")
            return

        self._current_source_type = "camera"
        self._current_image_bgr = None

        self._video_capture = capture
        if self.source_label is not None:
            self.source_label.setText("Source: camera 0")
        if self.confidence_label is not None:
            self.confidence_label.setText("Confidence: -")
        self._update_runtime_state_label()
        self._set_fps_idle()

        self._set_status("Camera started")
        self._start_video_playback()

    def _run_detection(self) -> None:
        if self._current_source_type == "image":
            self._run_image_detection()
            return

        if self._current_source_type == "video":
            self._toggle_video_detection()
            return

        if self._current_source_type == "camera":
            self._toggle_camera_detection()
            return

        self._set_status("No active source to run detection")

    def _run_image_detection(self) -> None:
        if self._current_image_bgr is None:
            self._set_status("No image is loaded")
            return

        if not self._ensure_model_loaded():
            return

        self._set_status("Running inference on image...")
        prediction = self._predict_single_frame(
            self._current_image_bgr,
            conf_threshold=self._current_confidence_threshold(),
        )
        if prediction is None:
            return

        self._set_display_from_bgr(prediction.annotated_bgr)
        self._update_confidence_label(prediction.detection_count, prediction.top_confidence)
        self._set_status(
            f"Detection complete: {prediction.detection_count} detection(s), top confidence {prediction.top_confidence:.3f}"
        )

    def _toggle_video_detection(self) -> None:
        if self._video_capture is None:
            self._set_status("No video is loaded")
            return

        if not self._video_detection_enabled:
            if not self._stream_model_is_available():
                self._video_detection_enabled = False
                return
            self._video_detection_enabled = True
            self._update_runtime_state_label()
            self._set_status("Video detection enabled")
        else:
            self._video_detection_enabled = False
            self._update_runtime_state_label()
            self._set_status("Video detection disabled")
            if self.confidence_label is not None:
                self.confidence_label.setText("Confidence: -")

    def _toggle_camera_detection(self) -> None:
        if self._video_capture is None or self._current_source_type != "camera":
            self._set_status("Camera is not active")
            return

        if not self._camera_detection_enabled:
            if not self._stream_model_is_available():
                self._camera_detection_enabled = False
                return
            self._camera_detection_enabled = True
            self._camera_inference_failures = 0
            self._update_runtime_state_label()
            self._set_status("Camera detection enabled")
        else:
            self._camera_detection_enabled = False
            self._camera_inference_failures = 0
            self._update_runtime_state_label()
            self._set_status("Camera detection disabled")
            if self.confidence_label is not None:
                self.confidence_label.setText("Confidence: -")

    def _stream_model_is_available(self) -> bool:
        if self._yolo_service.model_file_exists():
            return True

        self._set_status(f"Model file not found: {YOLO_MODEL_PATH}")
        if self.confidence_label is not None:
            self.confidence_label.setText("Confidence: model not found")
        return False

    def _ensure_model_loaded(self) -> bool:
        if not self._yolo_service.model_file_exists():
            self._set_status(f"Model file not found: {YOLO_MODEL_PATH}")
            if self.confidence_label is not None:
                self.confidence_label.setText("Confidence: model not found")
            return False

        if self._yolo_service.has_loaded_model():
            return True

        self._set_status(f"Loading model: {YOLO_MODEL_PATH}")
        loaded, status_message, confidence_message = self._yolo_service.load_model()
        if status_message is not None:
            self._set_status(status_message)
        if confidence_message is not None and self.confidence_label is not None:
            self.confidence_label.setText(confidence_message)
        return loaded

    def _predict_single_frame(self, frame_bgr: cv2.typing.MatLike, conf_threshold: float):
        prediction, status_message, confidence_message = self._yolo_service.predict_single_frame(
            frame_bgr,
            conf_threshold=conf_threshold,
        )
        if status_message is not None:
            self._set_status(status_message)
        if confidence_message is not None and self.confidence_label is not None:
            self.confidence_label.setText(confidence_message)
        return prediction

    def _update_confidence_label(self, detection_count: int, top_confidence: float) -> None:
        if self.confidence_label is not None:
            self.confidence_label.setText(
                f"Confidence: top={top_confidence:.3f}, detections={detection_count}"
            )

    def _set_display_from_bgr(self, image_bgr: cv2.typing.MatLike) -> None:
        self.image_view.set_pixmap(bgr_to_qpixmap(image_bgr))
        self._mark_frame_displayed()

    def _start_video_playback(self) -> None:
        if self._video_capture is None:
            return

        fps = self._video_capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 240:
            fps = 30.0

        interval_ms = max(1, int(1000 / fps))
        self._reset_fps_counter()
        self._video_timer.start(interval_ms)
        self._set_status("Playback started")

    def _read_next_video_frame(self) -> None:
        if self._video_capture is None:
            self._video_timer.stop()
            return

        success, frame_bgr = self._video_capture.read()
        if not success or frame_bgr is None:
            self._handle_stream_end_of_stream()
            return

        source_type = self._current_source_type
        stream_detection_enabled = (
            source_type == "video" and self._video_detection_enabled
        ) or (
            source_type == "camera" and self._camera_detection_enabled
        )

        if stream_detection_enabled:
            frame_id = self._stream_frame_id
            self._stream_frame_id += 1
            self._last_stream_frame_bgr = frame_bgr.copy()
            self._last_stream_frame_id = frame_id
            self.request_inference.emit(
                frame_bgr.copy(),
                source_type,
                self._stream_session_id,
                frame_id,
                self._current_confidence_threshold(),
            )
            return

        self._set_display_from_bgr(frame_bgr)

    def _handle_stream_inference_result(
        self,
        prediction,
        status_message,
        confidence_message,
        source_type: str,
        session_id: int,
        frame_id: int,
    ) -> None:
        if session_id != self._stream_session_id:
            return

        if frame_id < self._last_accepted_result_frame_id:
            return

        if source_type == "video" and not self._video_detection_enabled:
            return

        if source_type == "camera" and not self._camera_detection_enabled:
            return

        if status_message is not None:
            self._set_status(status_message)
        if confidence_message is not None and self.confidence_label is not None:
            self.confidence_label.setText(confidence_message)

        if source_type == "video":
            if prediction is None:
                self._video_detection_enabled = False
                self._update_runtime_state_label()
                self._set_status("Inference failure: video detection disabled")
                return
            self._set_display_from_bgr(prediction.annotated_bgr)
            self._last_accepted_result_frame_id = frame_id
            self._update_confidence_label(prediction.detection_count, prediction.top_confidence)
            return

        if source_type == "camera":
            if prediction is None:
                self._camera_inference_failures += 1
                if self._camera_inference_failures >= 3:
                    self._camera_detection_enabled = False
                    self._camera_inference_failures = 0
                    self._update_runtime_state_label()
                    self._set_status("Inference failure: camera detection disabled")
                    if self.confidence_label is not None:
                        self.confidence_label.setText("Confidence: camera detection disabled")
                return

            self._camera_inference_failures = 0
            self._set_display_from_bgr(prediction.annotated_bgr)
            self._last_accepted_result_frame_id = frame_id
            self._update_confidence_label(prediction.detection_count, prediction.top_confidence)

    def _handle_stream_end_of_stream(self) -> None:
        if self._video_capture is None:
            self._video_timer.stop()
            return

        is_video_source = self._current_source_type == "video"
        video_detection_was_enabled = is_video_source and self._video_detection_enabled
        if video_detection_was_enabled:
            self._video_detection_enabled = False

        self._camera_detection_enabled = False
        self._camera_inference_failures = 0
        self._update_runtime_state_label()

        # Keep no-flicker playback behavior, but avoid ending on a stale annotated frame
        # by showing the final decoded frame once when inference is lagging behind.
        if (
            is_video_source
            and self._last_stream_frame_bgr is not None
            and self._last_stream_frame_id > self._last_accepted_result_frame_id
        ):
            self._set_display_from_bgr(self._last_stream_frame_bgr)

        self._release_video_resources(set_status=False)
        self._current_source_type = "none"
        self._current_image_bgr = None
        self._update_runtime_state_label()

        if self.source_label is not None:
            self.source_label.setText("Source: none")
        if self.confidence_label is not None:
            self.confidence_label.setText("Confidence: -")
        self._set_status("Playback finished")

    def _release_video_resources(self, *, set_status: bool = True) -> None:
        self._video_timer.stop()

        self._stream_session_id += 1
        self._stream_frame_id = 0
        self._last_accepted_result_frame_id = -1
        self._last_stream_frame_bgr = None
        self._last_stream_frame_id = -1
        self.reset_worker_pending.emit()

        if self._video_capture is not None:
            self._video_capture.release()
            self._video_capture = None
            if set_status:
                self._set_status("Camera/video stopped")
        self._set_fps_idle()

    def _reset_detection_state(self) -> None:
        self._video_detection_enabled = False
        self._camera_detection_enabled = False
        self._camera_inference_failures = 0
        self._stream_frame_id = 0
        self._last_accepted_result_frame_id = -1
        self._update_runtime_state_label()

    def _current_confidence_threshold(self) -> float:
        if self.confidence_spinbox is None:
            return 0.25
        return float(self.confidence_spinbox.value())

    def _stop_active_stream(self) -> None:
        if self._current_source_type not in {"video", "camera"} or self._video_capture is None:
            self._set_status("No active stream to stop")
            return

        stopped_source = self._current_source_type
        self._release_video_resources()
        self._reset_detection_state()
        self._current_source_type = "none"
        self._current_image_bgr = None
        self._update_runtime_state_label()

        if self.source_label is not None:
            self.source_label.setText("Source: none")
        if self.confidence_label is not None:
            self.confidence_label.setText("Confidence: -")

        # Keep the last rendered frame on screen after stop for demo friendliness.
        self._set_status(f"{stopped_source.capitalize()} stopped (last frame kept)")

    def _update_runtime_state_label(self) -> None:
        if self.mode_state_label is None:
            return

        detection_enabled = False
        if self._current_source_type == "video":
            detection_enabled = self._video_detection_enabled
        elif self._current_source_type == "camera":
            detection_enabled = self._camera_detection_enabled

        detection_text = "ON" if detection_enabled else "OFF"
        self.mode_state_label.setText(
            f"Mode: {self._current_source_type} | Detection: {detection_text}"
        )

    def _reset_fps_counter(self) -> None:
        self._displayed_frames = 0
        self._fps_window_start = perf_counter()
        self._set_fps_idle()

    def _mark_frame_displayed(self) -> None:
        self._displayed_frames += 1
        elapsed = perf_counter() - self._fps_window_start
        if elapsed < 0.5:
            return

        fps = self._displayed_frames / elapsed
        if self.fps_label is not None:
            self.fps_label.setText(f"FPS: {fps:.1f}")
        self._displayed_frames = 0
        self._fps_window_start = perf_counter()

    def _set_fps_idle(self) -> None:
        if self.fps_label is not None:
            self.fps_label.setText("FPS: -")

    def _set_status(self, message: str) -> None:
        self.statusBar().showMessage(message)

    def closeEvent(self, event: QCloseEvent) -> None:
        self._release_video_resources()
        self._inference_thread.quit()
        self._inference_thread.wait(1000)
        super().closeEvent(event)
