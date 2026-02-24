from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from borebar_model import BoreBarModel


class TorsionalPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.model = BoreBarModel()

        layout = QHBoxLayout()

        # --- Левая панель параметров ---
        left = QVBoxLayout()

        self.rho_input = QLineEdit("7800")
        self.G_input = QLineEdit("8e10")
        self.length_input = QLineEdit("2.5")
        self.delta_input = QLineEdit("3.44e-6")
        self.Jr_input = QLineEdit("2.57e-2")
        self.Jp_input = QLineEdit("1.9e-5") 
        self.omega_start_input = QLineEdit("1000")
        self.omega_end_input = QLineEdit("15000")
        self.omega_step_input = QLineEdit("1")

        analyze_btn = QPushButton("Выполнить анализ")
        analyze_btn.clicked.connect(self.run_analysis)

        back_btn = QPushButton("Назад в меню")
        back_btn.clicked.connect(lambda: main_window.switch(main_window.menu))

        left.addWidget(QLabel("Плотность материала (ρ, кг/м³)"))
        left.addWidget(self.rho_input)

        left.addWidget(QLabel("Модуль сдвига (G, Па)"))
        left.addWidget(self.G_input)

        left.addWidget(QLabel("Длина борштанги (L, м)"))
        left.addWidget(self.length_input)

        left.addWidget(QLabel("Коэффициент внутреннего демпфирования (δ₁, с)"))
        left.addWidget(self.delta_input)

        left.addWidget(QLabel("Момент инерции режущей головки (Jr, кг·м²)"))
        left.addWidget(self.Jr_input)

        left.addWidget(QLabel("Полярный момент инерции (Jp, м⁴)"))
        left.addWidget(self.Jp_input)

        left.addWidget(analyze_btn)
        left.addWidget(back_btn)
        left.addStretch()
        left.addWidget(QLabel("Начальная частота ω₀ (рад/с)"))
        left.addWidget(self.omega_start_input)

        left.addWidget(QLabel("Конечная частота ω (рад/с)"))
        left.addWidget(self.omega_end_input)

        left.addWidget(QLabel("Шаг Δω"))
        left.addWidget(self.omega_step_input)

        # --- Правая часть (график) ---
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        layout.addLayout(left, 1)
        layout.addWidget(self.canvas, 3)

        self.setLayout(layout)

    def run_analysis(self):
        params = {
            "rho": float(self.rho_input.text()),
            "G": float(self.G_input.text()),
            "Jr": 2.57e-2,
            "Jp": 1.9e-5,
            "delta1": float(self.delta_input.text()),
            "length": float(self.length_input.text()),
            "multiplier": 1,    
            "omega_start": float(self.omega_start_input.text()),
            "omega_end": float(self.omega_end_input.text()),
            "omega_step": float(self.omega_step_input.text()),
        }

        result = self.model.calculate_torsional(params)

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(result["sigma_real"], result["sigma_imag"])
        ax.set_title("Крутильные колебания: σ(p)")
        ax.set_xlabel("Re(σ)")
        ax.set_ylabel("Im(σ)")
        ax.grid(True)

        im0 = self.model.find_torsional_im0_points(params)
        points = im0.get("points", [])
        crit = im0.get("critical")

        # все пересечения
        if points:
            ax.plot(
                [p["re"] for p in points],
                [0]*len(points),
                "o",
                markersize=5,
                label="Im(σ)=0"
            )

        # критическая
        if crit:
            ax.plot(
                crit["re"], 0,
                "o",
                markersize=9,
                label="Критическая"
            )

        ax.legend()

        self.canvas.draw()