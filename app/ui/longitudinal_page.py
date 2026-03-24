from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, QFileDialog
import numpy as np

from app.ui.analysis_page_base import AnalysisPageBase
from app.core.borebar_model import BoreBarModel
from app.utils.presets import get_longitudinal_presets
from app.utils.export_utils import finite_curve_rows, curve_summary, export_analysis_data
from app.utils.summary_builders import build_longitudinal_summary


class LongitudinalPage(AnalysisPageBase):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.model = BoreBarModel()
        self.current_preset_name = "custom"

        left_card, left = self._make_card("Параметры продольной модели")

        self.E_input = QLineEdit("2e11")
        self.rho_input = QLineEdit("7800")
        self.S_input = QLineEdit("2e-4")
        self.length_input = QLineEdit("2.5")
        self.mu_input = QLineEdit("0.1")
        self.tau_input = QLineEdit("0.06")
        self.omega_start_input = QLineEdit("0.001")
        self.omega_end_input = QLineEdit("400")
        self.omega_step_input = QLineEdit("0.1")

        result_card, self.results_label = self._make_result_card("После расчёта здесь появится краткая сводка по параметрической кривой и критическим точкам.")

        analyze_btn = QPushButton("Выполнить анализ")
        analyze_btn.clicked.connect(self.run_analysis)

        back_btn = QPushButton("Назад в меню")
        back_btn.clicked.connect(lambda: main_window.switch(main_window.menu))

        self.export_button = QPushButton("Экспорт результатов")
        self.export_button.clicked.connect(self.export_results)

        for label_text, widget in [
            ("Модуль Юнга (E, Па)", self.E_input),
            ("Плотность материала (ρ, кг/м³)", self.rho_input),
            ("Площадь поперечного сечения (S, м²)", self.S_input),
            ("Длина борштанги (L, м)", self.length_input),
            ("Коэффициент регенеративной связи (μ)", self.mu_input),
            ("Время запаздывания резания (τ, с)", self.tau_input),
        ]:
            label = QLabel(label_text)
            label.setObjectName("fieldLabel")
            left.addWidget(label)
            left.addWidget(widget)

        left.addWidget(analyze_btn)
        left.addWidget(back_btn)
        left.addWidget(self.export_button)

        self._setup_preset_selector(left, get_longitudinal_presets(), self.apply_preset)
        self._add_frequency_controls(left, self.omega_start_input, self.omega_end_input, self.omega_step_input)
        left.addStretch()

        self._build_analysis_layout(left_card, result_card)

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

    def _update_result_summary(self, result: dict):
        self._set_results_text(build_longitudinal_summary(result))

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
        self._update_result_summary(result)

        K1 = np.asarray(result["K1"], dtype=float) / 1e6
        delta = np.asarray(result["delta"], dtype=float) / 1e3

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(K1, delta, linewidth=2.0, label="K₁(ω)–δ(ω)")

        im0 = self.model.find_longitudinal_im0_points(params)
        points = im0.get("points", [])
        crit = im0.get("critical")

        if points:
            ax.plot([p["re"] / 1e6 for p in points], [0] * len(points), "o", markersize=5, label="δ=0")
        if crit:
            ax.plot(crit["re"] / 1e6, 0, "o", markersize=9, label="Критическая")

        ax.legend()
        self._style_plot_axes(ax, "Продольные колебания: кривая K₁–δ", "K₁ (МН/м)", "δ (кН·с/м)")
        self._finalize_plot()
        if self.main_window is not None and hasattr(self.main_window, "status"):
            self.main_window.status.showMessage("Продольный анализ выполнен")

    def apply_preset(self, name):
        if name not in self.presets:
            return
        self.current_preset_name = name
        preset = self.presets[name]
        mapping = {
            "E": self.E_input,
            "rho": self.rho_input,
            "S": self.S_input,
            "length": self.length_input,
            "mu": self.mu_input,
            "tau": self.tau_input,
            "omega_start": self.omega_start_input,
            "omega_end": self.omega_end_input,
            "omega_step": self.omega_step_input,
        }
        for key, widget in mapping.items():
            if key in preset:
                widget.setText(str(preset[key]))

    def _build_export_data(self, params: dict) -> dict:
        omega = self.model.build_frequency_grid(params, include_endpoint=True)
        result = self.model.calculate_longitudinal({**params, "omega_override": omega})

        K1 = np.asarray(result["K1"], dtype=float)
        delta = np.asarray(result["delta"], dtype=float)

        curve_rows = finite_curve_rows(omega, K1, delta)
        finite_mask = np.isfinite(omega) & np.isfinite(K1) & np.isfinite(delta)

        im0 = self.model.find_longitudinal_im0_points(params)
        critical = im0.get("critical")

        invalid_nonfinite = int(np.size(omega) - np.count_nonzero(finite_mask))

        return {
            "export_schema_version": 4,
            "analysis_type": "longitudinal",
            "preset_name": self.current_preset_name or "custom",
            "params": params,
            "model_info": {
                "model_variant": "si_wave_speed",
                "curve_semantics": "curve stores omega, K1(omega), delta(omega)",
                "wave_speed_a": float(result.get("a", 0.0)),
                "x_definition": "omega * L / a",
                "source_curve_for_special_points": im0.get("source_curve", "direct_longitudinal_curve"),
            },
            "numerics": {
                "solver_variant": "longitudinal_direct_curve_sampling_with_zero_crossing_detection",
                "export_variant": "compact_unified_v4",
                "omega_step": float(params["omega_step"]),
                "invalid_point_count": invalid_nonfinite,
                "invalid_reason_counts": {
                    "arg_too_small": 0,
                    "near_coth_pole": 0,
                    "sigma_clip": 0,
                    "nonfinite": invalid_nonfinite,
                    "other": 0,
                },
                "curve_saved_kind": "direct_curve",
            },
            "curve_summary": curve_summary(omega, K1, delta, include_total_count=False),
            "special_points": {
                "im0_points": im0.get("points", []),
                "critical_point": critical,
            },
            "curve": curve_rows,
        }

    def export_results(self):
        try:
            params = self.get_parameters()
            self._validate_parameters(params)
        except ValueError as e:
            self._show_error(str(e))
            return
        except Exception as e:
            self._show_error(f"Не удалось прочитать параметры: {e}")
            return

        data = self._build_export_data(params)

        filename, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Сохранить продольные результаты",
            "",
            "JSON (*.json);;CSV (*.csv)"
        )
        if not filename:
            return

        file_format = "json" if selected_filter.startswith("JSON") else "csv"
        export_analysis_data(data, filename, file_format)
        if self.main_window is not None and hasattr(self.main_window, "status"):
            self.main_window.status.showMessage("Продольные результаты экспортированы")
