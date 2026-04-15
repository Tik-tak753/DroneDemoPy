from PySide6.QtWidgets import (
    QFrame,
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
        open_image_button.clicked.connect(lambda: self._set_status("Open Image clicked"))
        panel_layout.addWidget(open_image_button)

        open_video_button = QPushButton("Open Video", panel)
        open_video_button.clicked.connect(lambda: self._set_status("Open Video clicked"))
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

        source_label = QLabel("Source: none", panel)
        panel_layout.addWidget(source_label)

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

    def _set_status(self, message: str) -> None:
        self.statusBar().showMessage(message)
