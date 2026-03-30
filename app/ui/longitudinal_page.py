from time import perf_counter

from PyQt5.QtWidgets import QApplication, QPushButton, QLabel, QLineEdit, QFileDialog
import numpy as np

from app.ui.analysis_page_base import AnalysisPageBase
from app.core.borebar_model import BoreBarModel
from app.utils.presets import get_longitudinal_presets
from app.utils.export_utils import finite_curve_rows, curve_summary, export_analysis_data


class LongitudinalPage(AnalysisPageBase):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.model = BoreBarModel()
        self.current_preset_name = "custom"
        self._cached_signature = None
        self._cached_analysis = None

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

        result_card, self.results_label = self._make_result_card(
            "После расчёта здесь появится краткая сводка по параметрической кривой и критическим точкам."
        )

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

    @staticmethod
    def _params_signature(params: dict):
        return tuple(sorted((key, repr(value)) for key, value in params.items()))

    def _get_or_compute_analysis(self, params: dict) -> tuple[np.ndarray, dict, dict]:
        signature = self._params_signature(params)
        if self._cached_signature == signature and self._cached_analysis is not None:
            cached = self._cached_analysis
            return cached["omega"], cached["result"], cached["im0"]

        omega = self.model.build_frequency_grid(params, include_endpoint=True)
        result = self.model.calculate_longitudinal({**params, "omega_override": omega})
        im0 = self.model.find_longitudinal_im0_points_from_result(params, result)

        self._cached_signature = signature
        self._cached_analysis = {
            "omega": np.asarray(omega, dtype=float),
            "result": result,
            "im0": im0,
        }
        return np.asarray(omega, dtype=float), result, im0

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
        self.model.validate_longitudinal_params(params)


    @staticmethod
    def _format_nonzero_reason_counts(reason_counts: dict) -> str:
        items = []
        for key, value in (reason_counts or {}).items():
            try:
                ivalue = int(value)
            except Exception:
                continue
            if ivalue > 0:
                items.append(f"{key}={ivalue}")
        return ", ".join(items) if items else "нет"

    def _update_result_summary(self, result: dict, im0: dict | None = None, elapsed_seconds: float | None = None):
        lines = [
            "Продольная модель",
            "",
            "Ключевые параметры:",
            f"a = √(E/ρ) = {float(result.get('a', float('nan'))):.6g} м/с",
            f"K₁(0) = {float(result.get('K1_0', float('nan'))):.6g}",
            f"δ(0) = {float(result.get('delta_0', float('nan'))):.6g}",
            f"ω₁ ≈ πa/L = {float(result.get('omega_main', float('nan'))):.6g} рад/с",
        ]

        if im0 is not None:
            points = im0.get('points', []) or []
            research_critical = im0.get('research_critical_point') or im0.get('critical')
            minimum_k1_point = im0.get('minimum_re_critical_point')
            policy = im0.get('critical_selection_policy', {}) or {}
            lines += [
                "",
                "Исследовательские special points:",
                f"Найдено точек δ(ω)=0: {len(points)}",
                f"Политика выбора критической точки: {policy.get('kind', 'minimum_K1_on_delta_zero_set')}",
            ]
            if research_critical is not None:
                lines += [
                    "Исследовательская критическая точка:",
                    f"ω* = {float(research_critical.get('omega', float('nan'))):.6g} рад/с",
                    f"f* = {float(research_critical.get('frequency', float('nan'))):.6g} Гц",
                    f"K₁* = {float(research_critical.get('K1', research_critical.get('re', float('nan')))):.6g}",
                ]
                crit_value = float(research_critical.get('K1', research_critical.get('re', float('nan'))))
                sign_text = 'K₁ < 0' if np.isfinite(crit_value) and crit_value < 0.0 else 'K₁ ≥ 0'
                lines.append(f"Положение критической точки: {sign_text}")
            else:
                lines.append("Исследовательская критическая точка: не найдена")

            if minimum_k1_point is not None and research_critical is not None:
                mk = float(minimum_k1_point.get('K1', minimum_k1_point.get('re', float('nan'))))
                rk = float(research_critical.get('K1', research_critical.get('re', float('nan'))))
                mo = float(minimum_k1_point.get('omega', float('nan')))
                ro = float(research_critical.get('omega', float('nan')))
                if not (np.isclose(mk, rk, equal_nan=True) and np.isclose(mo, ro, equal_nan=True)):
                    lines += [
                        "",
                        "Диагностическая самая левая точка:",
                        f"ω = {mo:.6g} рад/с",
                        f"K₁ = {mk:.6g}",
                    ]

        invalid_count = int(result.get('invalid_point_count', 0))
        lines += [
            "",
            "Паспорт расчёта:",
            f"Отбраковано точек: {invalid_count}",
            f"Причины: {self._format_nonzero_reason_counts(result.get('invalid_reason_counts', {}))}",
        ]
        if elapsed_seconds is not None:
            lines.append(f"Время расчёта и построения графика: {elapsed_seconds:.3f} с")
        self._set_results_text("\n".join(lines))


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
        omega, result, im0 = self._get_or_compute_analysis(params)

        K1 = np.asarray(result["K1"], dtype=float) / 1e6
        delta = np.asarray(result["delta"], dtype=float) / 1e3

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(K1, delta, linewidth=2.0, label="K₁(ω)–δ(ω)")

        points = im0.get("points", [])
        crit = im0.get("research_critical_point") or im0.get("critical")

        if points:
            ax.plot([p["re"] / 1e6 for p in points], [0.0] * len(points), "o", markersize=5, label="δ=0")
        if crit:
            ax.plot(crit["re"] / 1e6, 0.0, "o", markersize=9, label="Критическая")

        ax.legend()
        self._style_plot_axes(ax, "Продольные колебания: кривая K₁–δ", "K₁ (МН/м)", "δ (кН·с/м)")
        self.canvas.draw()
        QApplication.processEvents()
        elapsed_seconds = perf_counter() - started_at
        self._update_result_summary(result, im0, elapsed_seconds)
        if self.main_window is not None and hasattr(self.main_window, "status"):
            self.main_window.status.showMessage(f"Продольный анализ выполнен за {elapsed_seconds:.3f} с")

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
        omega, result, im0 = self._get_or_compute_analysis(params)

        K1 = np.asarray(result["K1"], dtype=float)
        delta = np.asarray(result["delta"], dtype=float)
        curve_rows = finite_curve_rows(omega, K1, delta)
        research_critical = im0.get("research_critical_point") or im0.get("critical")

        return {
            "export_schema_version": 4,
            "analysis_type": "longitudinal",
            "preset_name": self.current_preset_name or "custom",
            "params": params,
            "model_info": {
                "model_variant": result.get("model_variant", "si_wave_speed"),
                "model_regime": result.get("longitudinal_model_regime"),
                "model_regime_label": result.get("longitudinal_model_regime_label"),
                "model_scope": result.get("longitudinal_model_scope"),
                "research_alignment_status": result.get("research_alignment_status"),
                "model_note": result.get("longitudinal_model_note"),
                "curve_semantics": "curve stores omega, K1(omega), delta(omega)",
                "curve_parameterization": result.get("curve_parameterization", "omega -> (K1(omega), delta(omega))"),
                "wave_speed_a": float(result.get("a", 0.0)),
                "x_definition": "omega * L / a",
                "zero_frequency_limit_policy": result.get("zero_frequency_limit_policy"),
                "source_curve_for_special_points": im0.get("source_curve", "direct_longitudinal_curve"),
            },
            "numerics": {
                "solver_variant": "longitudinal_direct_curve_sampling_with_zero_crossing_detection",
                "export_variant": "compact_unified_v4",
                "omega_step": float(params["omega_step"]),
                "invalid_point_count": int(result.get("invalid_point_count", 0)),
                "invalid_reason_counts": dict(result.get("invalid_reason_counts", {})),
                "numerics_metadata": dict(result.get("numerics_metadata", {})),
                "curve_saved_kind": "direct_curve",
            },
            "curve_summary": curve_summary(omega, K1, delta, include_total_count=False),
            "special_points": {
                "im0_points": im0.get("points", []),
                "research_critical_point": research_critical,
                "minimum_re_critical_point": im0.get("minimum_re_critical_point"),
                "critical_selection_policy": im0.get("critical_selection_policy"),
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
            self._default_export_path("longitudinal_results.json"),
            "JSON (*.json);;CSV (*.csv)",
        )
        if not filename:
            return

        file_format = "json" if selected_filter.startswith("JSON") else "csv"
        export_analysis_data(data, filename, file_format)
        if self.main_window is not None and hasattr(self.main_window, "status"):
            self.main_window.status.showMessage("Продольные результаты экспортированы")
