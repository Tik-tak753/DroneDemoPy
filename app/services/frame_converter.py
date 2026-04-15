import cv2
from PySide6.QtGui import QImage, QPixmap


def bgr_to_qpixmap(image_bgr: cv2.typing.MatLike) -> QPixmap:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width, channels = image_rgb.shape
    bytes_per_line = channels * width
    image = QImage(
        image_rgb.data,
        width,
        height,
        bytes_per_line,
        QImage.Format_RGB888,
    ).copy()
    return QPixmap.fromImage(image)
