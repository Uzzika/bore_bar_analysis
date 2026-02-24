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


class LongitudinalPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.model = BoreBarModel()

        layout = QHBoxLayout()

        # ---------- Левая панель ----------
        left = QVBoxLayout()

        self.E_input = QLineEdit("2e11")
        self.rho_input = QLineEdit("7800")
        self.S_input = QLineEdit("2e-4")
        self.length_input = QLineEdit("2.5")
        self.mu_input = QLineEdit("0.1")
        self.tau_input = QLineEdit("0.06")
        self.omega_start_input = QLineEdit("0.001")
        self.omega_end_input = QLineEdit("400")
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

        left.addWidget(QLabel("Площадь поперечного сечения (S, м²)"))
        left.addWidget(self.S_input)

        left.addWidget(QLabel("Длина борштанги (L, м)"))
        left.addWidget(self.length_input)

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

        # ---------- Правая часть (график) ----------
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        layout.addLayout(left, 1)
        layout.addWidget(self.canvas, 3)

        self.setLayout(layout)

    def run_analysis(self):
        params = {
            "E": float(self.E_input.text()),
            "rho": float(self.rho_input.text()),
            "S": float(self.S_input.text()),
            "length": float(self.length_input.text()),
            "mu": float(self.mu_input.text()),
            "tau": float(self.tau_input.text()),
            "omega_start": float(self.omega_start_input.text()),
            "omega_end": float(self.omega_end_input.text()),
            "omega_step": float(self.omega_step_input.text()),
        }

        result = self.model.calculate_longitudinal(params)

        K1 = np.array(result["K1"]) / 1e6
        delta = np.array(result["delta"]) / 1e3

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        ax.plot(K1, delta)

        # === Поиск пересечений δ = 0 ===
        im0 = self.model.find_longitudinal_im0_points(params)
        points = im0.get("points", [])
        crit = im0.get("critical")

        # Все пересечения
        if points:
            ax.plot(
                [p["re"] / 1e6 for p in points],     # K1
                [0] * len(points),            # δ = 0
                "o",
                markersize=5,
                label="δ = 0"
            )

        # Критическая точка (минимальный K1)
        if crit:
            ax.plot(
                crit["re"] / 1e6,
                0,
                "o",
                markersize=9,
                label="Критическая"
            )

        ax.legend()
        ax.set_title("Продольные колебания: кривая K₁–δ")
        ax.set_xlabel("K₁ (МН/м)")
        ax.set_ylabel("δ (кН·с/м)")
        ax.grid(True)

        self.canvas.draw()