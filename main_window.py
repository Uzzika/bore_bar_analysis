from PyQt5.QtWidgets import QMainWindow, QStackedWidget

from menu_page import MenuPage
from torsional_page import TorsionalPage
from longitudinal_page import LongitudinalPage
from transverse_page import TransversePage
from help_page import HelpPage
from algorithm_page import AlgorithmPage


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Моделирование колебаний борштанги")
        self.resize(900, 600)

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # Создаем страницы
        self.menu = MenuPage(self)
        self.torsional = TorsionalPage(self)
        self.longitudinal = LongitudinalPage(self)
        self.transverse = TransversePage(self)
        self.help_page = HelpPage(self)
        self.algorithm_page = AlgorithmPage(self)

        # Добавляем в стек
        self.stack.addWidget(self.menu)
        self.stack.addWidget(self.torsional)
        self.stack.addWidget(self.longitudinal)
        self.stack.addWidget(self.transverse)
        self.stack.addWidget(self.help_page)
        self.stack.addWidget(self.algorithm_page)

        self.stack.setCurrentWidget(self.menu)

    def switch(self, widget):
        self.stack.setCurrentWidget(widget)