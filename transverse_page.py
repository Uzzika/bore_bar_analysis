from PyQt5.QtWidgets import (
    QComboBox,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QMessageBox,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from borebar_model import BoreBarModel
import numpy as np
from presets import get_presets


class TransversePage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.model = BoreBarModel()

        layout = QHBoxLayout()

        left = QVBoxLayout()

        self.E_input = QLineEdit("2.1e11")
        self.rho_input = QLineEdit("7800")
        self.length_input = QLineEdit("2.7")
        self.mu_input = QLineEdit("0.6")
        self.tau_input = QLineEdit("0.1")
        self.R_input = QLineEdit("0.04")
        self.r_input = QLineEdit("0.035")
        self.K_input = QLineEdit("6e5")
        self.h_input = QLineEdit("3.02141544835e-05")
        self.omega_start_input = QLineEdit("0")
        self.omega_end_input = QLineEdit("220")
        self.omega_step_input = QLineEdit("0.1")

        analyze_btn = QPushButton("Выполнить анализ")
        analyze_btn.clicked.connect(self.run_analysis)

        back_btn = QPushButton("Назад в меню")
        back_btn.clicked.connect(lambda: main_window.switch(main_window.menu))

        self.export_button = QPushButton("Экспорт результатов")
        self.export_button.clicked.connect(self.export_results)

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

        left.addWidget(QLabel("Коэффициент внутреннего трения (h, с)"))
        left.addWidget(self.h_input)

        left.addWidget(QLabel("Коэффициент регенеративной связи (μ)"))
        left.addWidget(self.mu_input)

        left.addWidget(QLabel("Время запаздывания резания (τ, с)"))
        left.addWidget(self.tau_input)

        left.addWidget(analyze_btn)
        left.addWidget(back_btn)
        left.addWidget(self.export_button)

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

    def get_parameters(self) -> dict:
        """Возвращает текущие параметры страницы для анализа и экспорта."""
        return {
            "E": float(self.E_input.text()),
            "rho": float(self.rho_input.text()),
            "length": float(self.length_input.text()),
            "mu": float(self.mu_input.text()),
            "tau": float(self.tau_input.text()),
            "R": float(self.R_input.text()),
            "r": float(self.r_input.text()),
            "K_cut": float(self.K_input.text()),
            "h": float(self.h_input.text()),
            "omega_start": float(self.omega_start_input.text()),
            "omega_end": float(self.omega_end_input.text()),
            "omega_step": float(self.omega_step_input.text()),
        }

    def _get_current_parameters(self) -> dict:
        return self.get_parameters()

    def _show_error(self, text: str):
        QMessageBox.critical(self, "Ошибка параметров", text)

    def _validate_parameters(self, params: dict):
        if params["E"] <= 0:
            raise ValueError("Модуль Юнга E должен быть > 0.")
        if params["rho"] <= 0:
            raise ValueError("Плотность ρ должна быть > 0.")
        if params["length"] <= 0:
            raise ValueError("Длина борштанги L должна быть > 0.")
        if params["R"] <= 0:
            raise ValueError("Внешний радиус R должен быть > 0.")
        if params["r"] < 0:
            raise ValueError("Внутренний радиус r не может быть отрицательным.")
        if params["r"] >= params["R"]:
            raise ValueError("Должно выполняться R > r.")
        if params["h"] < 0:
            raise ValueError("Коэффициент внутреннего трения h не может быть отрицательным.")
        if params["omega_step"] <= 0:
            raise ValueError("Шаг частоты Δω должен быть > 0.")
        if params["omega_end"] <= params["omega_start"]:
            raise ValueError("Конечная частота должна быть больше начальной.")

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

        result = self.model.calculate_transverse(params)

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(result["W_real"], result["W_imag"], label="W(iω)")

        im0 = self.model.find_transverse_im0_points(params)
        points = im0.get("points", [])
        crit = im0.get("critical")

        if points:
            ax.plot([p["re"] for p in points], [0] * len(points), "o", markersize=5, label="Im(W)=0")

        if crit:
            ax.plot(crit["re"], 0, "o", markersize=9, label="Критическая")

        ax.legend()
        ax.axhline(0, linestyle="--")
        ax.axvline(0, linestyle="--")
        ax.set_title("Поперечные колебания: годограф W(p), β = hγ")
        ax.set_xlabel("Re(W)")
        ax.set_ylabel("Im(W)")
        ax.set_aspect("equal")
        ax.grid(True)
        self.canvas.draw()

    def apply_preset(self, name):
        if name not in self.presets:
            return

        preset = self.presets[name]
        mapping = {
            "E": self.E_input,
            "rho": self.rho_input,
            "length": self.length_input,
            "mu": self.mu_input,
            "tau": self.tau_input,
            "R": self.R_input,
            "r": self.r_input,
            "K_cut": self.K_input,
            "h": self.h_input,
            "omega_start": self.omega_start_input,
            "omega_end": self.omega_end_input,
            "omega_step": self.omega_step_input,
        }

        for key, widget in mapping.items():
            if key in preset:
                widget.setText(str(preset[key]))

    def export_results(self):
        from PyQt5.QtWidgets import QFileDialog
        import json
        import csv

        try:
            params = self.get_parameters()
            self._validate_parameters(params)
        except ValueError as e:
            self._show_error(str(e))
            return
        except Exception as e:
            self._show_error(f"Не удалось прочитать параметры: {e}")
            return

        omega = np.linspace(
            float(params["omega_start"]),
            float(params["omega_end"]),
            max(2, int((params["omega_end"] - params["omega_start"]) / params["omega_step"]) + 1)
        )

        result = self.model.calculate_transverse({**params, "omega_override": omega})
        re_w = np.asarray(result["W_real"], dtype=float)
        im_w = np.asarray(result["W_imag"], dtype=float)
        omega_valid = np.asarray(result["omega"], dtype=float)

        im0 = self.model.find_transverse_im0_points(params)
        points = im0.get("points", [])
        critical = im0.get("critical")

        filename, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Сохранить поперечные результаты",
            "",
            "JSON (*.json);;CSV (*.csv)"
        )
        if not filename:
            return

        file_format = "json" if selected_filter.startswith("JSON") else "csv"

        metadata = {
            "transverse_model": "Galerkin one-mode model based on EJ*y'''' + EJ*h*y''''_t + m*y¨ = 0",
            "damping_relation": "beta = h * gamma = EJ * h * integral(phi''^2 dx)",
            "alpha": float(result["alpha"]),
            "beta": float(result["beta"]),
            "gamma": float(result["gamma"]),
            "h": float(result["h"]),
            "damping_source": result["damping_source"],
            "modal_mass_integral": float(result["modal_mass_integral"]),
            "modal_curvature_integral": float(result["modal_curvature_integral"]),
        }

        if file_format == "json":
            data = {
                "params": params,
                "metadata": metadata,
                "curve": [
                    {"omega": float(o), "Re(W)": float(r), "Im(W)": float(i)}
                    for o, r, i in zip(omega_valid, re_w, im_w)
                ],
                "im0_points": points,
                "critical_point": critical,
            }

            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        else:
            with open(filename, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["# Поперечная модель"])
                for key, value in metadata.items():
                    writer.writerow([key, value])

                writer.writerow([])
                writer.writerow(["omega", "Re(W)", "Im(W)"])
                for o, r, i in zip(omega_valid, re_w, im_w):
                    writer.writerow([o, r, i])

                writer.writerow([])
                writer.writerow(["# Точки пересечения Im(W) = 0"])
                writer.writerow(["omega*", "Re(W)*", "Im(W)"])
                for p in points:
                    writer.writerow([p["omega"], p["re"], 0.0])

                if critical:
                    writer.writerow([])
                    writer.writerow(["# Критическая точка"])
                    writer.writerow([critical["omega"], critical["re"], 0.0])
