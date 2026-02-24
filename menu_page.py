from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


class MenuPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(20)

        # ---------- Заголовок ----------
        title = QLabel("МОДЕЛИРОВАНИЕ КОЛЕБАНИЙ БОРШТАНГИ")
        title.setAlignment(Qt.AlignCenter)

        font = QFont()
        font.setPointSize(16)
        font.setBold(True)
        title.setFont(font)

        layout.addWidget(title)
        layout.addSpacing(30)

        # ---------- Кнопки ----------
        btn_torsional = self._create_button(
            "Крутильные колебания",
            lambda: main_window.switch(main_window.torsional)
        )

        btn_longitudinal = self._create_button(
            "Продольные колебания",
            lambda: main_window.switch(main_window.longitudinal)
        )

        btn_transverse = self._create_button(
            "Поперечные колебания",
            lambda: main_window.switch(main_window.transverse)
        )

        btn_help = self._create_button(
            "Справка",
            lambda: main_window.switch(main_window.help_page)
        )

        btn_algo = self._create_button(
            "Алгоритм работы",
            lambda: main_window.switch(main_window.algorithm_page)
        )

        menu_font = QFont("Inter", 12)

        for btn in [btn_torsional, btn_longitudinal, btn_transverse, btn_help, btn_algo]:
            btn.setFont(menu_font)

        layout.addWidget(btn_torsional)
        layout.addWidget(btn_longitudinal)
        layout.addWidget(btn_transverse)
        layout.addSpacing(10)
        layout.addWidget(btn_help)
        layout.addWidget(btn_algo)

        self.setLayout(layout)

    def _create_button(self, text, slot):
        btn = QPushButton(text)
        btn.clicked.connect(slot)
        btn.setMinimumWidth(350)
        btn.setMinimumHeight(50)
        btn.setFont(QFont("", 11))
        return btn