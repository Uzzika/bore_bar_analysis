from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QFrame
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
        layout = QVBoxLayout(card)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(14)

        title = QLabel("АЛГОРИТМ РАБОТЫ ПРОГРАММЫ")
        title.setAlignment(Qt.AlignCenter)
        title.setObjectName("sectionTitle")

        text = QLabel(
            "<b>Порядок работы</b><br><br>"
            "<b>1. Выбор раздела</b><br>"
            "В главном меню выберите нужный тип анализа: крутильные, продольные или поперечные колебания.<br><br>"
            "<b>2. Ввод параметров</b><br>"
            "Заполните поля с исходными данными. При необходимости можно выбрать готовый пресет, "
            "а затем изменить отдельные значения вручную.<br><br>"
            "<b>3. Запуск расчёта</b><br>"
            "Нажмите кнопку запуска анализа. Программа выполнит вычисления по выбранной модели "
            "и построит соответствующий график.<br><br>"
            "<b>4. Просмотр результатов</b><br>"
            "После расчёта на странице отображаются кривая или годограф, характерные точки, "
            "а также краткая сводка по рассчитанным параметрам.<br><br>"
            "<b>5. Сравнение вариантов</b><br>"
            "Изменяйте исходные данные и повторяйте расчёт, чтобы посмотреть, "
            "как параметры влияют на поведение системы и форму графиков.<br><br>"
            "<b>6. Экспорт</b><br>"
            "При необходимости сохраните результаты в файл. Поддерживается экспорт в форматы JSON и CSV.<br><br>"
            "Таким образом, программа позволяет последовательно перейти от задания параметров "
            "к расчёту, визуальному анализу и сохранению результатов."
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
