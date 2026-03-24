from PyQt5.QtWidgets import (
    QVBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QFileDialog,
)
import numpy as np

from app.ui.analysis_page_base import AnalysisPageBase
from app.core.borebar_model import BoreBarModel
from app.utils.presets import get_torsional_presets
from app.utils.export_utils import curve_rows_with_gaps, curve_summary, export_analysis_data
from app.utils.summary_builders import build_torsional_summary


class TorsionalPage(AnalysisPageBase):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.model = BoreBarModel()
        self.current_preset_name = "custom"

        left_card, left = self._make_card("Параметры крутильной модели")

        self.rho_input = QLineEdit("7800")
        self.G_input = QLineEdit("8e10")
        self.length_input = QLineEdit("2.5")
        self.delta_input = QLineEdit("3.44e-6")
        self.multiplier_input = QLineEdit("1")
        self.Jr_input = QLineEdit("2.57e-2")
        self.Jp_input = QLineEdit("1.9e-5")
        self.omega_start_input = QLineEdit("1000")
        self.omega_end_input = QLineEdit("15000")
        self.omega_step_input = QLineEdit("1")

        result_card, self.results_label = self._make_result_card("После расчёта здесь появится краткая сводка по критическим точкам и параметрам модели.")

        analyze_btn = QPushButton("Выполнить анализ")
        analyze_btn.clicked.connect(self.run_analysis)

        back_btn = QPushButton("Назад в меню")
        back_btn.clicked.connect(lambda: main_window.switch(main_window.menu))

        export_btn = QPushButton("Экспорт результатов")
        export_btn.clicked.connect(self.export_results)

        for label_text, widget in [
            ("Плотность материала (ρ, кг/м³)", self.rho_input),
            ("Модуль сдвига (G, Па)", self.G_input),
            ("Длина борштанги (L, м)", self.length_input),
            ("Коэффициент внутреннего демпфирования (δ₁, с)", self.delta_input),
            ("Множитель демпфирования m", self.multiplier_input),
            ("Момент инерции режущей головки (Jr, кг·м²)", self.Jr_input),
            ("Полярный момент инерции (Jp, м⁴)", self.Jp_input),
        ]:
            label = QLabel(label_text)
            label.setObjectName("fieldLabel")
            left.addWidget(label)
            left.addWidget(widget)

        left.addWidget(analyze_btn)
        left.addWidget(back_btn)
        left.addWidget(export_btn)

        self._setup_preset_selector(left, get_torsional_presets(), self.apply_preset)
        self._add_frequency_controls(left, self.omega_start_input, self.omega_end_input, self.omega_step_input)
        left.addStretch()

        self._build_analysis_layout(left_card, result_card)

    def get_parameters(self):
        return {
            "rho": float(self.rho_input.text()),
            "G": float(self.G_input.text()),
            "length": float(self.length_input.text()),
            "delta1": float(self.delta_input.text()),
            "multiplier": float(self.multiplier_input.text()),
            "Jr": float(self.Jr_input.text()),
            "Jp": float(self.Jp_input.text()),
            "omega_start": float(self.omega_start_input.text()),
            "omega_end": float(self.omega_end_input.text()),
            "omega_step": float(self.omega_step_input.text()),
        }

    def _validate_parameters(self, params: dict):
        for key in ("rho", "G", "length", "Jr", "Jp"):
            if params[key] <= 0:
                raise ValueError(f"Параметр {key} должен быть > 0.")
        if params["delta1"] < 0:
            raise ValueError("Параметр δ₁ не может быть отрицательным.")
        if params["multiplier"] <= 0:
            raise ValueError("Множитель демпфирования должен быть > 0.")
        if params["omega_step"] <= 0:
            raise ValueError("Шаг частоты Δω должен быть > 0.")
        if params["omega_end"] <= params["omega_start"]:
            raise ValueError("Конечная частота должна быть больше начальной.")

    def _update_result_summary(self, result: dict, critical: dict | None):
        self._set_results_text(build_torsional_summary(result, critical))

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

        result = self.model.calculate_torsional(params)
        im0 = self.model.find_torsional_im0_points(params)
        points = im0.get("points", [])
        critical = im0.get("critical")
        self._update_result_summary(result, critical)

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(result["display_sigma_real"], result["display_sigma_imag"], linewidth=2.0, label="σ(iω)")

        if points:
            ax.plot([p["re"] for p in points], [0.0] * len(points), "o", markersize=5, label="Im(σ)=0")
        if critical:
            ax.plot(critical["re"], 0.0, "o", markersize=9, label="Критическая")

        ax.legend()
        ax.axhline(0, linestyle="--", linewidth=0.8, alpha=0.45)
        ax.axvline(0, linestyle="--", linewidth=0.8, alpha=0.45)
        self._style_plot_axes(ax, "Крутильные колебания: кривая D-разбиения σ(iω)", "Re(σ)", "Im(σ)")
        self._finalize_plot()
        if self.main_window is not None and hasattr(self.main_window, "status"):
            self.main_window.status.showMessage("Крутильный анализ выполнен")

    def apply_preset(self, name):
        if name not in self.presets:
            return
        self.current_preset_name = name
        preset = self.presets[name]
        mapping = {
            "rho": self.rho_input,
            "G": self.G_input,
            "length": self.length_input,
            "delta1": self.delta_input,
            "multiplier": self.multiplier_input,
            "Jr": self.Jr_input,
            "Jp": self.Jp_input,
            "omega_start": self.omega_start_input,
            "omega_end": self.omega_end_input,
            "omega_step": self.omega_step_input,
        }
        for key, widget in mapping.items():
            if key in preset:
                widget.setText(str(preset[key]))

    def _build_export_data(self, params: dict) -> dict:
        result = self.model.calculate_torsional(params)
        im0 = self.model.find_torsional_im0_points(params)
        critical = im0.get("critical")

        curve_omega = np.asarray(result["physical_omega"], dtype=float)
        curve_re = np.asarray(result["physical_sigma_real"], dtype=float)
        curve_im = np.asarray(result["physical_sigma_imag"], dtype=float)

        return {
            "export_schema_version": 4,
            "analysis_type": "torsional",
            "preset_name": self.current_preset_name or "custom",
            "params": params,
            "model_info": {
                "model_variant": result.get("model_variant", "torsional_physical_positive_plus_model_display_symmetry"),
                "curve_semantics": "curve stores the physical torsional branch: omega, Re(sigma), Im(sigma)",
                "delta1_effective": float(result.get("delta1_effective", params["delta1"] * params.get("multiplier", 1.0))),
                "negative_frequency_policy": result.get(
                    "negative_frequency_policy",
                    "display curve is a conjugate mirror of the positive physical branch",
                ),
                "source_curve_for_special_points": im0.get("source_curve", "physical_positive_branch"),
            },
            "numerics": {
                "solver_variant": "torsional_direct_curve_sampling_with_im0_detection",
                "export_variant": "compact_unified_v4",
                "omega_step": float(params["omega_step"]),
                "invalid_point_count": int(result.get("invalid_point_count", 0)),
                "invalid_reason_counts": dict(result.get("invalid_reason_counts", {})),
                "numerics_metadata": dict(result.get("numerics_metadata", {})),
                "curve_saved_kind": "physical_only",
            },
            "curve_summary": curve_summary(curve_omega, curve_re, curve_im, include_total_count=True),
            "special_points": {
                "im0_points": im0.get("points", []),
                "critical_point": critical,
            },
            "curve": curve_rows_with_gaps(curve_omega, curve_re, curve_im),
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
            "Сохранить крутильные результаты",
            "",
            "JSON (*.json);;CSV (*.csv)",
        )
        if not filename:
            return

        file_format = "json" if selected_filter.startswith("JSON") else "csv"
        export_analysis_data(data, filename, file_format)
        if self.main_window is not None and hasattr(self.main_window, "status"):
            self.main_window.status.showMessage("Крутильные результаты экспортированы")
