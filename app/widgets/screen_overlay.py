from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QWidget


@dataclass(frozen=True)
class OverlayDetection:
    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    confidence: float


class ScreenOverlayWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.Tool
            | Qt.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._detections: list[OverlayDetection] = []

    def set_region_geometry(self, region: dict[str, int]) -> None:
        self.setGeometry(
            region["left"],
            region["top"],
            region["width"],
            region["height"],
        )

    def set_detections(self, detections: Sequence[OverlayDetection]) -> None:
        self._detections = list(detections)
        self.update()

    def clear_detections(self) -> None:
        self._detections = []
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        _ = event
        if not self._detections:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)

        box_pen = QPen(QColor(0, 255, 120, 230))
        box_pen.setWidth(2)
        painter.setPen(box_pen)
        font = QFont()
        font.setPointSize(10)
        painter.setFont(font)

        for det in self._detections:
            rect = QRect(
                int(det.x1),
                int(det.y1),
                max(1, int(det.x2 - det.x1)),
                max(1, int(det.y2 - det.y1)),
            )
            painter.drawRect(rect)

            label_text = f"{det.label} {det.confidence:.2f}"
            text_width = max(100, int(len(label_text) * 7.5))
            text_rect = QRect(rect.x(), max(0, rect.y() - 22), text_width, 20)
            painter.fillRect(text_rect, QColor(0, 0, 0, 170))
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(text_rect.adjusted(5, 0, -5, 0), Qt.AlignVCenter | Qt.AlignLeft, label_text)
            painter.setPen(box_pen)
