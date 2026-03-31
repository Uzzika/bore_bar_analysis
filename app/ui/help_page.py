from PyQt5.QtWidgets import QSizePolicy, QWidget, QVBoxLayout, QLabel, QPushButton, QFrame
from PyQt5.QtCore import Qt


class HelpPage(QWidget):
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

        title = QLabel("СПРАВКА")
        title.setAlignment(Qt.AlignCenter)
        title.setObjectName("sectionTitle")

        text = QLabel(
            "<b>Анализ колебаний борштанги</b><br><br>"
            "Программа нужна для расчёта и просмотра колебаний борштанги. "
            "В ней можно выполнить анализ трёх видов колебаний: крутильных, продольных "
            "и поперечных.<br><br>"

            "Пользователь вводит параметры, запускает расчёт, смотрит график и при необходимости "
            "сохраняет результат в файл.<br><br>"

            "<b>Что умеет программа</b><br>"
            "• рассчитывать крутильные, продольные и поперечные колебания;<br>"
            "• строить графики по результатам расчёта;<br>"
            "• показывать найденные характерные точки;<br>"
            "• подставлять готовые пресеты для быстрого заполнения полей;<br>"
            "• сохранять результаты в JSON и CSV.<br><br>"

            "<b>Какие модели используются</b><br>"
            "• <b>Крутильные колебания</b> — строится кривая устойчивости по параметрам борштанги. "
            "Для удобства просмотра график может отображаться и для отрицательных частот, "
            "но основной расчёт выполняется по физической ветви.<br><br>"

            "• <b>Продольные колебания</b> — рассчитывается зависимость параметров системы "
            "при продольных смещениях борштанги. Модель сделана в удобной и устойчивой для расчёта форме.<br><br>"

            "• <b>Поперечные колебания</b> — рассчитывается годограф поперечных колебаний борштанги "
            "с учётом её формы, размеров и параметров резания.<br><br>"

            "<b>Как работать с программой</b><br>"
            "• в главном меню выберите нужный раздел;<br>"
            "• введите параметры или выберите пресет;<br>"
            "• нажмите кнопку расчёта;<br>"
            "• посмотрите график и краткие результаты;<br>"
            "• при необходимости сохраните результат в файл.<br><br>"

            "Программа подходит для учебной работы, экспериментов с параметрами "
            "и наглядного анализа поведения системы."
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
