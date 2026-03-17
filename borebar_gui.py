"""
borebar_gui.py

Совместимый адаптер для старого имени GUI-модуля.

После перехода на многооконную архитектуру на базе MainWindow поддерживается 
одна GUI-ветка. Этот существует только для обратной совместимости
со старыми импортами, отчётом и пользовательскими сценариями запуска.

- фактический рабочий интерфейс реализован в main_window.py и *_page.py;
- класс BoreBarGUI оставлен как обёртка над новым MainWindow;
- за счёт этого исключается рассинхронизация старого GUI с API модели.
"""

from __future__ import annotations

import sys

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication

from main_window import MainWindow


class BoreBarGUI(MainWindow):
    """
    Совместимое имя главного окна для старых импортов.

    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "Моделирование колебаний борштанги "
        )



def main() -> int:
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = BoreBarGUI()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
