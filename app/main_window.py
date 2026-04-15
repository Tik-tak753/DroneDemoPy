from pathlib import Path

import cv2
from PySide6.QtCore import QTimer
from PySide6.QtGui import QCloseEvent, QImage, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.widgets import ImageView


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DroneDemoPy")
        self.resize(1000, 700)

        self.source_label: QLabel | None = None
        self._video_capture: cv2.VideoCapture | None = None
        self._video_timer = QTimer(self)
        self._video_timer.timeout.connect(self._read_next_video_frame)

        self._create_central_widget()
        self._create_menu_bar()
        self.statusBar().showMessage("Ready")

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
        start_camera_button.clicked.connect(lambda: self._set_status("Start Camera clicked"))
        panel_layout.addWidget(start_camera_button)

        run_detection_button = QPushButton("Run Detection", panel)
        run_detection_button.clicked.connect(lambda: self._set_status("Run Detection clicked"))
        panel_layout.addWidget(run_detection_button)

        panel_layout.addSpacing(16)

        confidence_label = QLabel("Confidence: -", panel)
        panel_layout.addWidget(confidence_label)

        self.source_label = QLabel("Source: none", panel)
        panel_layout.addWidget(self.source_label)

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

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)",
        )

        if not file_path:
            self._set_status("Image open canceled")
            return

        pixmap = QPixmap(file_path)
        if pixmap.isNull():
            self._set_status("Failed to load image")
            return

        self.image_view.set_pixmap(pixmap)
        if self.source_label is not None:
            self.source_label.setText(f"Source: {Path(file_path).name}")
        self._set_status(f"Loaded image: {file_path}")

    def _open_video(self) -> None:
        self._release_video_resources()

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

        self._video_capture = capture
        if self.source_label is not None:
            self.source_label.setText(f"Source: {Path(file_path).name}")

        self._set_status(f"Video loaded: {file_path}")
        self._start_video_playback()

    def _start_video_playback(self) -> None:
        if self._video_capture is None:
            return

        fps = self._video_capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 240:
            fps = 30.0

        interval_ms = max(1, int(1000 / fps))
        self._video_timer.start(interval_ms)
        self._set_status("Playback started")

    def _read_next_video_frame(self) -> None:
        if self._video_capture is None:
            self._video_timer.stop()
            return

        success, frame_bgr = self._video_capture.read()
        if not success or frame_bgr is None:
            self._video_timer.stop()
            self._release_video_resources()
            self._set_status("Playback finished")
            return

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width, channels = frame_rgb.shape
        bytes_per_line = channels * width
        image = QImage(
            frame_rgb.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888,
        ).copy()

        self.image_view.set_pixmap(QPixmap.fromImage(image))

    def _release_video_resources(self) -> None:
        self._video_timer.stop()

        if self._video_capture is not None:
            self._video_capture.release()
            self._video_capture = None

    def _set_status(self, message: str) -> None:
        self.statusBar().showMessage(message)

    def closeEvent(self, event: QCloseEvent) -> None:
        self._release_video_resources()
        super().closeEvent(event)
