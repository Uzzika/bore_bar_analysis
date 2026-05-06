from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QGridLayout,
    QFrame,
    QSizePolicy,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


class MenuPage(QWidget):
    def close_application(self):
        self.main_window.close()

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window

        root = QVBoxLayout(self)
        root.setAlignment(Qt.AlignTop)
        root.setContentsMargins(24, 16, 24, 16)
        root.setSpacing(10)

        hero = QFrame()
        hero.setObjectName("heroCard")
        hero.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        hero_layout = QVBoxLayout(hero)
        hero_layout.setContentsMargins(18, 16, 18, 16)
        hero_layout.setSpacing(6)

        title = QLabel("МОДЕЛИРОВАНИЕ КОЛЕБАНИЙ БОРШТАНГИ")
        title.setAlignment(Qt.AlignCenter)
        title.setObjectName("heroTitle")
        title.setFont(QFont("Segoe UI", 15, QFont.Bold))

        subtitle = QLabel(
            "Программный комплекс для анализа устойчивости крутильных, продольных "
            "и поперечных колебаний борштанги"
        )
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setObjectName("heroSubtitle")
        subtitle.setWordWrap(True)

        hero_layout.addWidget(title)
        hero_layout.addWidget(subtitle)
        root.addWidget(hero)

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)

        cards = [
            (
                "Крутильные колебания",
                "Анализ устойчивости и поиск характерных точек.",
                lambda: main_window.switch(main_window.torsional),
            ),
            (
                "Продольные колебания",
                "Расчёт продольных колебаний борштанги и построение графика.",
                lambda: main_window.switch(main_window.longitudinal),
            ),
            (
                "Поперечные колебания",
                "Расчёт поперечных колебаний и построение годографа.",
                lambda: main_window.switch(main_window.transverse),
            ),
            (
                "Диаграммы устойчивости",
                "Сравнение границы устойчивости для крутильных и поперечных колебаний.",
                lambda: main_window.switch(main_window.stability_diagram_page),
            ),
            (
                "Справка",
                "Краткое описание программы и её возможностей.",
                lambda: main_window.switch(main_window.help_page),
            ),
            (
                "Алгоритм работы",
                "Пошаговая инструкция по вводу данных, расчёту и экспорту.",
                lambda: main_window.switch(main_window.algorithm_page),
            ),
            ("Выход", "Закрыть программу.", self.close_application),
        ]

        for idx, (header, description, slot) in enumerate(cards):
            grid.addWidget(self._create_menu_card(header, description, slot), idx // 2, idx % 2)

        root.addLayout(grid)
        root.addSpacing(4)

    def _create_menu_card(self, title: str, description: str, slot):
        card = QFrame()
        card.setObjectName("menuCard")
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)

        title_label = QLabel(title)
        title_label.setObjectName("menuCardTitle")
        title_label.setWordWrap(True)

        desc_label = QLabel(description)
        desc_label.setObjectName("menuCardDescription")
        desc_label.setWordWrap(True)

        btn = QPushButton("Открыть" if title != "Выход" else "Завершить")
        btn.setMinimumHeight(34)
        btn.clicked.connect(slot)

        layout.addWidget(title_label)
        layout.addWidget(desc_label)
        layout.addWidget(btn)
        return card
