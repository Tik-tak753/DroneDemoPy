from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QColor, QGuiApplication, QMouseEvent, QPaintEvent, QPainter
from PySide6.QtWidgets import QDialog, QRubberBand, QWidget


class ScreenRegionSelector(QDialog):
    """Simple full-screen overlay used to pick a screen capture rectangle."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select Screen Region")
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.Dialog
            | Qt.WindowStaysOnTopHint
        )
        self.setModal(True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setCursor(Qt.CrossCursor)

        geometry = QRect()
        for screen in QGuiApplication.screens():
            geometry = geometry.united(screen.geometry())
        if geometry.isNull():
            geometry = self.screen().geometry() if self.screen() is not None else QRect(0, 0, 1280, 720)
        self.setGeometry(geometry)

        self._origin = QPoint()
        self._selection = QRect()
        self._rubber_band = QRubberBand(QRubberBand.Rectangle, self)

    @classmethod
    def select_region(cls, parent: QWidget | None = None) -> dict[str, int] | None:
        selector = cls(parent)
        if selector.exec() != QDialog.Accepted:
            return None

        rect = selector._selection.normalized()
        if rect.width() < 2 or rect.height() < 2:
            return None

        return {
            "left": rect.x(),
            "top": rect.y(),
            "width": rect.width(),
            "height": rect.height(),
        }

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.LeftButton:
            return
        self._origin = event.globalPosition().toPoint()
        self._selection = QRect(self._origin, self._origin)
        self._rubber_band.setGeometry(self._selection)
        self._rubber_band.show()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._rubber_band.isHidden():
            return
        current = event.globalPosition().toPoint()
        self._selection = QRect(self._origin, current).normalized()
        self._rubber_band.setGeometry(self._selection)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.LeftButton:
            return
        if self._rubber_band.isHidden():
            return
        self._selection = self._rubber_band.geometry().normalized()
        self.accept()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self.reject()
            return
        super().keyPressEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:
        _ = event
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 70))
