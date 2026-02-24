import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont
from main_window import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    font = QFont("Inter", 10)
    app.setFont(font)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())