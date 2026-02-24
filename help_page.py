from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import Qt


class HelpPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window

        layout = QVBoxLayout()

        title = QLabel("СПРАВКА")
        title.setAlignment(Qt.AlignCenter)

        text = QLabel(
            "<b>Анализ колебаний борштанги</b><br><br>"
            "Программа реализует расчёт крутильных, продольных и поперечных "
            "колебаний борштанги на основе математических моделей из курсовой работы "
            "и строит соответствующие кривые D-разбиения и годографы.<br><br>"
            "Интерфейс позволяет:<br>"
            "• задавать физические и геометрические параметры системы;<br>"
            "• анализировать устойчивость при разных значениях δ₁, μ, τ и K;<br>"
            "• в интерактивном режиме исследовать влияние параметров на форму кривых.<br><br>"
            "Курсовая работа, ФИИТ-4 курс."
        )
        text.setAlignment(Qt.AlignCenter)

        back_btn = QPushButton("Назад в меню")
        back_btn.clicked.connect(
            lambda: main_window.switch(main_window.menu)
        )

        layout.addWidget(title)
        layout.addWidget(text)
        layout.addWidget(back_btn)
        layout.addStretch()

        self.setLayout(layout)