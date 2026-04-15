import sys
from PySide6.QtWidgets import QApplication, QWidget

app = QApplication(sys.argv)

window = QWidget()
window.resize(800, 600)
window.setWindowTitle("DroneDemoPy")
window.show()

sys.exit(app.exec())