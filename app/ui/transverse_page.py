from PyQt5.QtCore import Qt
from time import perf_counter

from PyQt5.QtWidgets import (
    QPushButton,
    QLabel,
    QLineEdit,
    QFileDialog,
    QApplication,
    QComboBox,
    QSizePolicy,
)
import numpy as np

from app.ui.analysis_page_base import AnalysisPageBase
from app.core.borebar_model import BoreBarModel
from app.utils.presets import get_transverse_presets
from app.utils.export_utils import curve_rows_with_gaps, curve_summary, export_analysis_data
from app.utils.summary_builders import build_transverse_summary


class TransversePage(AnalysisPageBase):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.model = BoreBarModel()
        self.current_preset_name = "custom"

        left_card, left = self._make_card("Параметры поперечной модели")

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

        for widget in (
            self.E_input,
            self.rho_input,
            self.length_input,
            self.mu_input,
            self.tau_input,
            self.R_input,
            self.r_input,
            self.K_input,
            self.h_input,
            self.omega_start_input,
            self.omega_end_input,
            self.omega_step_input,
        ):
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            widget.setMinimumWidth(0)

        self.modal_shape_combo = QComboBox()
        # Важно: Ignored позволяет combo сжиматься внутри узкой прокручиваемой панели
        # и не раздувать минимальную ширину всей левой колонки.
        self.modal_shape_combo.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.modal_shape_combo.setMinimumWidth(0)
        self.modal_shape_combo.setMaximumWidth(16777215)
        self.modal_shape_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.modal_shape_combo.setMinimumContentsLength(1)

        verified_short = "Первая собственная форма консольной балки"
        project_short = "Project approximation"

        self.modal_shape_combo.addItem(
            verified_short,
            "verified_cantilever_first_mode_phi",
        )
        self.modal_shape_combo.setItemData(
            0,
            "Первая собственная форма консольной балки (верифицированная)",
            role=Qt.ToolTipRole,
        )
        self.modal_shape_combo.addItem(
            project_short,
            "project_maple_compatible_phi",
        )
        self.modal_shape_combo.setItemData(
            1,
            "Project approximation (Maple-compatible)",
            role=Qt.ToolTipRole,
        )
        self.modal_shape_combo.setToolTip(
            "Выбор варианта формы φ(x). Полные названия доступны во всплывающей подсказке."
        )
        self.modal_shape_combo.currentIndexChanged.connect(self._refresh_modal_shape_tooltip)
        self._refresh_modal_shape_tooltip()

        result_card, self.results_label = self._make_result_card(
            "После расчёта здесь будут показаны α, β, γ и сведения о форме φ(x)."
        )

        analyze_btn = QPushButton("Выполнить анализ")
        analyze_btn.clicked.connect(self.run_analysis)

        back_btn = QPushButton("Назад в меню")
        back_btn.clicked.connect(lambda: main_window.switch(main_window.menu))

        export_btn = QPushButton("Экспорт результатов")
        export_btn.clicked.connect(self.export_results)

        for btn in (analyze_btn, back_btn, export_btn):
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.setMinimumWidth(0)

        for label_text, widget in [
            ("Модуль Юнга (E, Па)", self.E_input),
            ("Плотность материала (ρ, кг/м³)", self.rho_input),
            ("Длина борштанги (L, м)", self.length_input),
            ("Внешний радиус борштанги (R, м)", self.R_input),
            ("Внутренний радиус борштанги (r, м)", self.r_input),
            ("Динамическая жёсткость резания (K, Н/м)", self.K_input),
            ("Коэффициент внутреннего трения (h, с)", self.h_input),
            ("Коэффициент регенеративной связи (μ)", self.mu_input),
            ("Время запаздывания резания (τ, с)", self.tau_input),
        ]:
            label = QLabel(label_text)
            label.setObjectName("fieldLabel")
            left.addWidget(label)
            left.addWidget(widget)

        label = QLabel("Вариант формы φ(x)")
        label.setObjectName("fieldLabel")
        left.addWidget(label)
        left.addWidget(self.modal_shape_combo)

        left.addWidget(analyze_btn)
        left.addWidget(back_btn)
        left.addWidget(export_btn)

        self._setup_preset_selector(left, get_transverse_presets(), self.apply_preset)
        if self.preset_combo is not None:
            self.preset_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.preset_combo.setMinimumWidth(0)
        self._add_frequency_controls(left, self.omega_start_input, self.omega_end_input, self.omega_step_input)
        left.addStretch()

        self._build_analysis_layout(left_card, result_card)

    def _refresh_modal_shape_tooltip(self):
        idx = self.modal_shape_combo.currentIndex()
        if idx < 0:
            self.modal_shape_combo.setToolTip("")
            return
        full_text = self.modal_shape_combo.itemData(idx, Qt.ToolTipRole) or self.modal_shape_combo.currentText()
        self.modal_shape_combo.setToolTip(str(full_text))

    def get_parameters(self) -> dict:
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
            "transverse_modal_shape_variant": self.modal_shape_combo.currentData(),
        }

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

    def _update_result_summary(self, result: dict, display_curve: dict | None = None, elapsed_seconds: float | None = None):
        if display_curve is None:
            omega = np.asarray(result.get("omega", []), dtype=float)
            display_curve = {
                "display_point_count": int(omega.size),
                "base_point_count": int(omega.size),
            }
        text = build_transverse_summary(result, display_curve)
        if elapsed_seconds is not None:
            text += f"\nВремя расчёта и построения графика: {elapsed_seconds:.3f} с"
        self._set_results_text(text)

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

        started_at = perf_counter()
        result = self.model.calculate_transverse(params)
        display_curve = self.model.build_transverse_display_curve(params)

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(display_curve["W_real"], display_curve["W_imag"], linewidth=2.0, label="W(iω)")

        im0 = self.model.find_transverse_im0_points(params)
        points = im0.get("points", [])
        crit = im0.get("critical")

        if points:
            ax.plot([p["re"] for p in points], [0.0] * len(points), "o", markersize=5, label="Im(W)=0")
        if crit:
            ax.plot(crit["re"], 0.0, "o", markersize=9, label="Критическая")

        ax.legend()
        ax.axhline(0, linestyle="--", linewidth=0.8, alpha=0.45)
        ax.axvline(0, linestyle="--", linewidth=0.8, alpha=0.45)
        self._style_plot_axes(ax, "Поперечные колебания: годограф W(p)", "Re(W)", "Im(W)", equal=True)
        self.canvas.draw()
        QApplication.processEvents()
        elapsed_seconds = perf_counter() - started_at
        self._update_result_summary(result, display_curve, elapsed_seconds)
        if self.main_window is not None and hasattr(self.main_window, "status"):
            self.main_window.status.showMessage(f"Поперечный анализ выполнен за {elapsed_seconds:.3f} с")

    def apply_preset(self, name: str):
        if name not in self.presets:
            return
        self.current_preset_name = name

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

        variant = preset.get("transverse_modal_shape_variant")
        if variant is not None:
            idx = self.modal_shape_combo.findData(variant)
            if idx >= 0:
                self.modal_shape_combo.setCurrentIndex(idx)

    def _build_export_data(self, params: dict) -> dict:
        result = self.model.calculate_transverse(params)
        curve_omega = np.asarray(result["omega"], dtype=float)
        curve_re = np.asarray(result["W_real"], dtype=float)
        curve_im = np.asarray(result["W_imag"], dtype=float)

        im0 = self.model.find_transverse_im0_points(params)
        critical = im0.get("critical")

        return {
            "export_schema_version": 4,
            "analysis_type": "transverse",
            "preset_name": self.current_preset_name or "custom",
            "params": params,
            "model_info": {
                "model_variant": result.get("model_variant", "galerkin_one_mode_unknown"),
                "curve_semantics": "curve stores omega, Re(W), Im(W) on the full physical grid; invalid points are kept as null/NaN gaps",
                "transverse_model": "Galerkin one-mode model",
                "modal_shape_variant": result.get("modal_shape_variant"),
                "modal_shape_source": result.get("modal_shape_source"),
                "modal_shape_description": result.get("modal_shape_description"),
                "shape_normalization": result.get("shape_normalization"),
                "shape_scale_C": float(result["shape_scale_C"]),
                "k1": float(result["k1"]),
                "lambda1": float(result["lambda1"]),
                "shape_eta": result.get("shape_eta"),
                "alpha": float(result["alpha"]),
                "beta": float(result["beta"]),
                "gamma": float(result["gamma"]),
                "h": float(result["h"]),
                "damping_source": result["damping_source"],
                "modal_mass_integral": float(result["modal_mass_integral"]),
                "modal_curvature_integral": float(result["modal_curvature_integral"]),
            },
            "numerics": {
                "solver_variant": "transverse_direct_curve_sampling_with_zero_crossing_detection",
                "export_variant": "compact_unified_v4",
                "omega_step": float(params["omega_step"]),
                "invalid_point_count": int(result.get("invalid_point_count", 0)),
                "invalid_reason_counts": dict(result.get("invalid_reason_counts", {})),
                "numerics_metadata": dict(result.get("numerics_metadata", {})),
                "curve_saved_kind": "full_curve_with_nan_gaps",
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
            "Сохранить поперечные результаты",
            self._default_export_path("transverse_results.json"),
            "JSON (*.json);;CSV (*.csv)"
        )
        if not filename:
            return

        file_format = "json" if selected_filter.startswith("JSON") else "csv"
        export_analysis_data(data, filename, file_format)
        if self.main_window is not None and hasattr(self.main_window, "status"):
            self.main_window.status.showMessage("Поперечные результаты экспортированы")
