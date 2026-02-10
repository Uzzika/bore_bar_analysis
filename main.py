import sys
from PyQt5.QtWidgets import QApplication
from borebar_gui import BoreBarGUI

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BoreBarGUI()
    window.show()
    sys.exit(app.exec_())