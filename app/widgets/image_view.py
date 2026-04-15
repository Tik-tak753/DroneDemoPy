from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPaintEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget


class ImageView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self._pixmap: QPixmap | None = None

    def set_pixmap(self, pixmap: QPixmap) -> None:
        self._pixmap = QPixmap(pixmap)
        self.update()

    def clear_pixmap(self) -> None:
        self._pixmap = None
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        rect = self.rect().adjusted(12, 12, -12, -12)
        painter.fillRect(rect, QColor("#f3f5f7"))

        pen = QPen(QColor("#7a7d80"))
        pen.setWidth(2)
        pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.drawRect(rect)

        if self._pixmap is None or self._pixmap.isNull():
            painter.setPen(QColor("#2f3133"))
            painter.drawText(rect, Qt.AlignCenter, "Image / Video view")
            return

        scaled = self._pixmap.scaled(
            rect.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        x = rect.x() + (rect.width() - scaled.width()) // 2
        y = rect.y() + (rect.height() - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)
