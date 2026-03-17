from PyQt5.QtWidgets import (
    QComboBox,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QFileDialog,
    QMessageBox,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from borebar_model import BoreBarModel
import numpy as np
from presets import get_presets


class LongitudinalPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.model = BoreBarModel()

        layout = QHBoxLayout()
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

    def get_parameters(self):
        return {
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

    def _show_error(self, text: str):
        QMessageBox.critical(self, "Ошибка параметров", text)

    def _validate_parameters(self, params: dict):
        if params["E"] <= 0:
            raise ValueError("Модуль Юнга E должен быть > 0.")
        if params["rho"] <= 0:
            raise ValueError("Плотность ρ должна быть > 0.")
        if params["S"] <= 0:
            raise ValueError("Площадь сечения S должна быть > 0.")
        if params["length"] <= 0:
            raise ValueError("Длина L должна быть > 0.")
        if params["omega_step"] <= 0:
            raise ValueError("Шаг частоты Δω должен быть > 0.")
        if params["omega_end"] <= params["omega_start"]:
            raise ValueError("Конечная частота должна быть больше начальной.")

    # -----------------------------------------------------------------

    def run_analysis(self):
        try:
            params = self.get_parameters()
            self._validate_parameters(params)
        except ValueError as e:
            self._show_error(str(e))
            return
        except Exception as e:
            self._show_error(f"Не удалось прочитать параметры: {e}")
            return

        result = self.model.calculate_longitudinal(params)

        K1 = np.array(result["K1"]) / 1e6
        delta = np.array(result["delta"]) / 1e3

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(K1, delta)
        im0 = self.model.find_longitudinal_im0_points(params)
        points = im0.get("points", [])
        crit = im0.get("critical")

        if points:
            ax.plot([p["re"]/1e6 for p in points],
                    [0]*len(points),
                    "o", markersize=5, label="δ=0")

        if crit:
            ax.plot(crit["re"]/1e6, 0,
                    "o", markersize=9, label="Критическая")

        ax.legend()
        ax.set_title("Продольные колебания: кривая K₁–δ")
        ax.set_xlabel("K₁ (МН/м)")
        ax.set_ylabel("δ (кН·с/м)")
        ax.grid(True)

        self.canvas.draw()

    # -----------------------------------------------------------------

    def apply_preset(self, name):
        if name not in self.presets:
            return

        preset = self.presets[name]

        mapping = {
            "length": self.length_input,
            "mu": self.mu_input,
            "tau": self.tau_input,
        }

        for key, widget in mapping.items():
            if key in preset:
                widget.setText(str(preset[key]))

    def export_results(self):
        import json
        import csv
        import numpy as np

        try:
            params = self.get_parameters()
            self._validate_parameters(params)
        except ValueError as e:
            self._show_error(str(e))
            return
        except Exception as e:
            self._show_error(f"Не удалось прочитать параметры: {e}")
            return

        # ---- расчёты ----
        omega = np.linspace(1, 5000, 5000)
        K1, delta = self.model.compute_longitudinal_curve(params, omega)

        im0 = self.model.find_longitudinal_im0_points(params)
        points = im0.get("points", [])
        critical = im0.get("critical")

        filename, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Сохранить продольные результаты",
            "",
            "JSON (*.json);;CSV (*.csv)"
        )

        if not filename:
            return

        file_format = "json" if selected_filter.startswith("JSON") else "csv"

        # ================= JSON =================
        if file_format == "json":
            data = {
                "params": params,
                "curve": [
                    {"omega": float(o), "K1": float(k), "delta": float(d)}
                    for o, k, d in zip(omega, K1, delta)
                ],
                "delta0_points": points,
                "critical_point": critical
            }

            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)

        # ================= CSV =================
        else:
            with open(filename, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)

                writer.writerow(["omega", "K1", "delta"])
                for o, k, d in zip(omega, K1, delta):
                    writer.writerow([o, k, d])

                writer.writerow([])
                writer.writerow(["# Точки пересечения δ = 0"])
                writer.writerow(["omega*", "K1*", "delta"])

                for p in points:
                    writer.writerow([
                        p["omega"],
                        p["K1"],
                        0.0
                    ])

                if critical:
                    writer.writerow([])
                    writer.writerow(["# Критическая точка"])
                    writer.writerow([
                        critical["omega"],
                        critical["K1"],
                        0.0
                    ])