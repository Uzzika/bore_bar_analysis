from PyQt5.QtCore import QTimer, QSize
from PyQt5.QtWidgets import QApplication, QMainWindow, QStackedWidget, QStatusBar, QSizePolicy

from app.ui.menu_page import MenuPage
from app.ui.torsional_page import TorsionalPage
from app.ui.longitudinal_page import LongitudinalPage
from app.ui.transverse_page import TransversePage
from app.ui.help_page import HelpPage
from app.ui.algorithm_page import AlgorithmPage
from app.ui.stability_diagram_page import StabilityDiagramPage


APP_STYLESHEET = """
QMainWindow, QWidget {
    background: #f4f7fb;
    color: #1f2a37;
    font-family: 'Segoe UI';
    font-size: 10.5pt;
}
QLabel#heroTitle {
    color: #123b73;
    font-weight: 700;
}
QLabel#heroSubtitle {
    color: #4d6078;
}
QFrame#heroCard, QFrame#menuCard, QFrame#card {
    background: #ffffff;
    border: 1px solid #dbe4f0;
    border-radius: 16px;
}
QFrame#heroCard {
    border: 1px solid #d5e2f6;
}
QLabel#menuCardTitle, QLabel#sectionTitle {
    color: #143d73;
    font-weight: 700;
}
QLabel#menuCardDescription, QLabel#resultsLabel {
    color: #506174;
    line-height: 1.35em;
}
QLabel#fieldLabel {
    color: #2d3e53;
    font-weight: 600;
}
QPushButton {
    background: #2e6fcb;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 14px;
    font-weight: 600;
    min-height: 18px;
}
QPushButton:hover { background: #255fb0; }
QPushButton:pressed { background: #1e4f94; }
QLineEdit, QComboBox, QAbstractSpinBox, QTextEdit, QPlainTextEdit {
    background: #fbfcfe;
    border: 1px solid #cfd8e6;
    border-radius: 10px;
    padding: 8px 10px;
    min-height: 20px;
}
QLineEdit:focus, QComboBox:focus, QAbstractSpinBox:focus, QTextEdit:focus, QPlainTextEdit:focus {
    border: 1px solid #2e6fcb;
}
QStatusBar {
    background: #eaf1fa;
    color: #3f566f;
    border-top: 1px solid #d1deee;
}
QScrollArea#leftPanelScroll {
    background: transparent;
}
"""


class MainWindow(QMainWindow):
    MIN_WIDTH = 1080
    MIN_HEIGHT = 400
    CHROME_W = 56
    CHROME_H = 94

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Моделирование колебаний борштанги")
        self.setStyleSheet(APP_STYLESHEET)

        self.stack = QStackedWidget()
        self.stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setCentralWidget(self.stack)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Готово к работе")

        self.menu = MenuPage(self)
        self.torsional = TorsionalPage(self)
        self.longitudinal = LongitudinalPage(self)
        self.transverse = TransversePage(self)
        self.help_page = HelpPage(self)
        self.algorithm_page = AlgorithmPage(self)
        self.stability_diagram_page = StabilityDiagramPage(self)

        self.pages = [
            self.menu,
            self.torsional,
            self.longitudinal,
            self.transverse,
            self.help_page,
            self.algorithm_page,
            self.stability_diagram_page,
        ]

        for page in self.pages:
            page.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.stack.addWidget(page)

        self.stack.setCurrentWidget(self.menu)
        self.setMinimumSize(self.MIN_WIDTH, self.MIN_HEIGHT)

        QTimer.singleShot(0, lambda: self._fit_window_to_page(self.menu, force=True, use_maximum=False))

    def switch(self, widget):
        self.stack.setCurrentWidget(widget)
        names = {
            self.menu: "Главное меню",
            self.torsional: "Раздел: крутильные колебания",
            self.longitudinal: "Раздел: продольные колебания",
            self.transverse: "Раздел: поперечные колебания",
            self.help_page: "Раздел: справка",
            self.algorithm_page: "Раздел: алгоритм работы",
            self.stability_diagram_page: "Раздел: диаграммы устойчивости",
        }
        self.status.showMessage(names.get(widget, "Готово"))
        self._fit_window_to_page(widget, force=True)

    def _fit_window_to_page(self, page, *, force: bool, use_maximum: bool = False):
        if self.isMaximized() or self.isFullScreen():
            return

        pages = self.pages if use_maximum else [page]
        target_w = self.MIN_WIDTH
        target_h = self.MIN_HEIGHT
        for p in pages:
            p.ensurePolished()
            p.adjustSize()
            hint = p.sizeHint().expandedTo(p.minimumSizeHint())
            target_w = max(target_w, hint.width() + self.CHROME_W)
            target_h = max(target_h, hint.height() + self.CHROME_H)

        available = self._available_screen_size()
        target = QSize(min(target_w, available.width()), min(target_h, available.height()))

        self.setMinimumSize(min(target.width(), available.width()), min(target.height(), available.height()))
        if force or self.width() < target.width() or self.height() < target.height():
            self.resize(target)

    @staticmethod
    def _available_screen_size() -> QSize:
        app = QApplication.instance()
        screen = app.primaryScreen() if app is not None else None
        if screen is None:
            return QSize(1440, 600)
        geom = screen.availableGeometry()
        return QSize(geom.width(), geom.height())
