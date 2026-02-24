from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from borebar_model import BoreBarModel
import numpy as np


class TransversePage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.model = BoreBarModel()

        layout = QHBoxLayout()

        # ---------- Левая панель ----------
        left = QVBoxLayout()

        self.E_input = QLineEdit("2.1e11")
        self.rho_input = QLineEdit("7800")
        self.length_input = QLineEdit("2.7")
        self.mu_input = QLineEdit("0.6")
        self.tau_input = QLineEdit("0.1")
        self.R_input = QLineEdit("0.04")
        self.r_input = QLineEdit("0.035")
        self.K_input = QLineEdit("6e5")
        self.beta_input = QLineEdit("0.3")
        self.omega_start_input = QLineEdit("0")
        self.omega_end_input = QLineEdit("220")
        self.omega_step_input = QLineEdit("0.1")

        analyze_btn = QPushButton("Выполнить анализ")
        analyze_btn.clicked.connect(self.run_analysis)

        back_btn = QPushButton("Назад в меню")
        back_btn.clicked.connect(
            lambda: main_window.switch(main_window.menu)
        )

        left.addWidget(QLabel("Модуль Юнга (E, Па)"))
        left.addWidget(self.E_input)

        left.addWidget(QLabel("Плотность материала (ρ, кг/м³)"))
        left.addWidget(self.rho_input)

        left.addWidget(QLabel("Длина борштанги (L, м)"))
        left.addWidget(self.length_input)

        left.addWidget(QLabel("Внешний радиус борштанги (R, м)"))
        left.addWidget(self.R_input)

        left.addWidget(QLabel("Внутренний радиус борштанги (r, м)"))
        left.addWidget(self.r_input)

        left.addWidget(QLabel("Динамическая жёсткость резания (K, Н/м)"))
        left.addWidget(self.K_input)

        left.addWidget(QLabel("Коэффициент вязкого демпфирования (β)"))
        left.addWidget(self.beta_input)

        left.addWidget(QLabel("Коэффициент регенеративной связи (μ)"))
        left.addWidget(self.mu_input)

        left.addWidget(QLabel("Время запаздывания резания (τ, с)"))
        left.addWidget(self.tau_input)

        left.addWidget(analyze_btn)
        left.addWidget(back_btn)
        left.addStretch()
        left.addWidget(QLabel("Начальная частота ω₀ (рад/с)"))
        left.addWidget(self.omega_start_input)

        left.addWidget(QLabel("Конечная частота ω (рад/с)"))
        left.addWidget(self.omega_end_input)

        left.addWidget(QLabel("Шаг Δω"))
        left.addWidget(self.omega_step_input)

        # ---------- Правая часть ----------
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        layout.addLayout(left, 1)
        layout.addWidget(self.canvas, 3)

        self.setLayout(layout)

    def run_analysis(self):
        params = {
            "E": float(self.E_input.text()),
            "rho": float(self.rho_input.text()),
            "length": float(self.length_input.text()),
            "mu": float(self.mu_input.text()),
            "tau": float(self.tau_input.text()),
            "R": float(self.R_input.text()),
            "r": float(self.r_input.text()),
            "K_cut": float(self.K_input.text()),
            "beta": float(self.beta_input.text()),
            "omega_start": float(self.omega_start_input.text()),
            "omega_end": float(self.omega_end_input.text()),
            "omega_step": float(self.omega_step_input.text()),
        }

        result = self.model.calculate_transverse(params)

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        ax.plot(result["W_real"], result["W_imag"])

        # === Поиск пересечений Im(W) = 0 ===
        im0 = self.model.find_transverse_im0_points(params)
        points = im0.get("points", [])
        crit = im0.get("critical")

        # Все пересечения
        if points:
            ax.plot(
                [p["re"] for p in points],
                [0] * len(points),
                "o",
                markersize=5,
                label="Im(W)=0"
            )

        # Критическая точка (минимальный Re)
        if crit:
            ax.plot(
                crit["re"],
                0,
                "o",
                markersize=9,
                label="Критическая"
            )

        ax.legend()

        ax.axhline(0, linestyle="--")
        ax.axvline(0, linestyle="--")

        ax.set_title("Поперечные колебания: годограф W(p)")
        ax.set_xlabel("Re(W)")
        ax.set_ylabel("Im(W)")
        ax.set_aspect("equal")
        ax.grid(True)

        self.canvas.draw()