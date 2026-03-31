from PyQt5.QtWidgets import QSizePolicy, QWidget, QVBoxLayout, QLabel, QPushButton, QFrame
from PyQt5.QtCore import Qt


class AlgorithmPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window

        root = QVBoxLayout(self)
        root.setContentsMargins(36, 28, 36, 28)
        root.setSpacing(16)

        card = QFrame()
        card.setObjectName("card")
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(14)

        title = QLabel("АЛГОРИТМ РАБОТЫ ПРОГРАММЫ")
        title.setAlignment(Qt.AlignCenter)
        title.setObjectName("sectionTitle")

        text = QLabel(
            "<b>Как работать с программой</b><br><br>"

            "<b>1. Выберите раздел</b><br>"
            "В главном меню выберите, какие колебания хотите исследовать: "
            "крутильные, продольные или поперечные.<br><br>"

            "<b>2. Введите параметры</b><br>"
            "Заполните поля вручную или выберите готовый пресет. "
            "После этого при необходимости можно изменить отдельные значения.<br><br>"

            "<b>3. Запустите расчёт</b><br>"
            "Нажмите кнопку выполнения анализа. Программа рассчитает модель "
            "и построит график.<br><br>"

            "<b>4. Посмотрите результат</b><br>"
            "После расчёта на странице появятся график, найденные точки "
            "и краткая сводка результатов.<br><br>"

            "<b>5. Сравните варианты</b><br>"
            "Меняйте параметры и запускайте расчёт ещё раз, чтобы увидеть, "
            "как меняется поведение системы.<br><br>"

            "<b>6. Сохраните результат</b><br>"
            "При необходимости экспортируйте данные в JSON или CSV.<br><br>"

            "Таким образом, работа в программе сводится к простому порядку: "
            "выбрать раздел, ввести параметры, выполнить расчёт, посмотреть график и сохранить результат."
        )
        text.setAlignment(Qt.AlignLeft)
        text.setWordWrap(True)
        text.setObjectName("resultsLabel")

        back_btn = QPushButton("Назад в меню")
        back_btn.clicked.connect(lambda: main_window.switch(main_window.menu))

        layout.addWidget(title)
        layout.addWidget(text)
        layout.addWidget(back_btn, alignment=Qt.AlignRight)
        root.addWidget(card)
        root.addStretch()
