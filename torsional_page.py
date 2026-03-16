from PyQt5.QtWidgets import (
    QComboBox,
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
from presets import get_presets
import numpy as np


class TorsionalPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.model = BoreBarModel()

        layout = QHBoxLayout()
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
        back_btn.clicked.connect(
            lambda: main_window.switch(main_window.menu)
        )

        # --- параметры ---
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
        
        self.export_button = QPushButton("Экспорт результатов")
        self.export_button.clicked.connect(self.export_results)
        left.addWidget(self.export_button)

        # --- пресеты ---
        self.presets = get_presets()

        self.preset_combo = QComboBox()
        self.preset_combo.addItem("Выберите пресет")
        self.preset_combo.addItems(self.presets.keys())
        self.preset_combo.currentTextChanged.connect(self.apply_preset)

        left.addWidget(QLabel("Типовые режимы:"))
        left.addWidget(self.preset_combo)

        left.addStretch()

        left.addWidget(QLabel("Начальная частота ω₀ (рад/с)"))
        left.addWidget(self.omega_start_input)

        left.addWidget(QLabel("Конечная частота ω (рад/с)"))
        left.addWidget(self.omega_end_input)

        left.addWidget(QLabel("Шаг Δω"))
        left.addWidget(self.omega_step_input)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        layout.addLayout(left, 1)
        layout.addWidget(self.canvas, 3)
        self.setLayout(layout)

    # -----------------------------------------------------------------
    def get_parameters(self) -> dict:
        """Возвращает текущие параметры страницы для экспорта."""
        return self._get_current_parameters()

    def run_analysis(self):
        omega_step = float(self.omega_step_input.text())
        params = {
            "rho": float(self.rho_input.text()),
            "G": float(self.G_input.text()),
            "Jr": float(self.Jr_input.text()),
            "Jp": float(self.Jp_input.text()),
            "delta1": float(self.delta_input.text()),
            "length": float(self.length_input.text()),
            "multiplier": 1,
            "omega_start": float(self.omega_start_input.text()),
            "omega_end": float(self.omega_end_input.text()),
            "omega_step": omega_step,
            "arg_min": 0.2,
        }

        result = self.model.calculate_torsional(params)

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        omega = result["omega"]
        re = result["sigma_real"]
        im = result["sigma_imag"]

        mask = np.isfinite(re) & np.isfinite(im)

        omega_start = float(self.omega_start_input.text())

        # 1) Если диапазон только положительный — НИКАКИХ зеркал
        if omega_start >= 0:
            ax.plot(re[mask], im[mask], label="ω > 0")

        # 2) Если пользователь реально затронул отрицательные — дозеркаливаем ТОЛЬКО относительно оси Re
        else:
            # рисуем положительную ветвь
            ax.plot(re[mask], im[mask], label="ω > 0")
            # и зеркальную (сопряжение): Re тот же, Im меняет знак
            ax.plot(re[mask], -im[mask], label="ω < 0 (сопряж.)")

        ax.set_title("Крутильные колебания: σ(p)")
        ax.set_xlabel("Re(σ)")
        ax.set_ylabel("Im(σ)")
        ax.grid(True)

        im0 = self.model.find_torsional_im0_points(params)
        inter = self.model.find_intersection(params)
        points = im0.get("points", [])
        crit = im0.get("critical")

        if points:
            ax.plot([p["re"] for p in points], [0]*len(points),
                    "o", markersize=5, label="Im(σ)=0")

        if crit:
            ax.plot(crit["re"], 0,
                    "o", markersize=9, label="Критическая")

        ax.legend()
        self.canvas.draw()

    # -----------------------------------------------------------------

    def apply_preset(self, name):
        if name not in self.presets:
            return

        preset = self.presets[name]

        mapping = {
            "rho": self.rho_input,
            "length": self.length_input,
            "delta1": self.delta_input,
        }

        for key, widget in mapping.items():
            if key in preset:
                widget.setText(str(preset[key]))

    def export_results(self):
        from PyQt5.QtWidgets import QFileDialog
        import json
        import csv
        import numpy as np

        params = self.get_parameters()
        im0 = self.model.find_torsional_im0_points(params)
        pts = im0["points"]
        crit = im0["critical"]
        curve = self.model.calculate_torsional(params)

        filename, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Сохранить результаты",
            "",
            "JSON (*.json);;CSV (*.csv)"
        )

        if not filename:
            return

        file_format = "json" if selected_filter.startswith("JSON") else "csv"

        if file_format == "json":
            data = {
                "params": params,
                "curve": curve,
                "im0_points": pts,
                "critical_point": crit
            }

            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)

        else:
            with open(filename, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)

                writer.writerow(["omega", "Re", "Im"])

                for row in curve:
                    writer.writerow(row)

                writer.writerow([])
                writer.writerow(["omega", "Re", "Im=0"])

                for p in pts:
                    writer.writerow([p["omega"], p["re"], 0.0])