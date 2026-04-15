import sys

from PySide6.QtWidgets import QApplication

from app.main_window import MainWindow


def run() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()
