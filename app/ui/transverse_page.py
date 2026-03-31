
from time import perf_counter

from PyQt5.QtWidgets import (
    QPushButton,
    QLabel,
    QLineEdit,
    QFileDialog,
    QApplication,
    QSizePolicy,
)

from app.ui.analysis_page_base import AnalysisPageBase
from app.core.borebar_model import BoreBarModel
from app.utils.presets import get_transverse_presets
from app.utils.export_utils import export_analysis_data
from app.utils.analysis_presenters import (
    build_transverse_export_data,
    build_transverse_summary_text,
)


class TransversePage(AnalysisPageBase):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.model = BoreBarModel()
        self.current_preset_name = "custom"
        self._cached_signature = None
        self._cached_analysis = None

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
            self.E_input, self.rho_input, self.length_input, self.mu_input, self.tau_input,
            self.R_input, self.r_input, self.K_input, self.h_input,
            self.omega_start_input, self.omega_end_input, self.omega_step_input,
        ):
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            widget.setMinimumWidth(0)

        result_card, self.results_label = self._make_result_card(
            "После расчёта здесь будут показаны α, β, γ и сведения о верифицированной поперечной модели."
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

    @staticmethod
    def _params_signature(params: dict):
        return tuple(sorted((key, repr(value)) for key, value in params.items()))

    def _get_or_compute_analysis(self, params: dict) -> tuple[dict, dict]:
        signature = self._params_signature(params)
        if self._cached_signature == signature and self._cached_analysis is not None:
            cached = self._cached_analysis
            return cached["result"], cached["im0"]

        result = self.model.calculate_transverse(params)
        im0 = self.model.find_transverse_im0_points_from_result(params, result)
        self._cached_signature = signature
        self._cached_analysis = {"result": result, "im0": im0}
        return result, im0

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
            "transverse_modal_shape_variant": "verified_cantilever_first_mode_phi",
        }

    def _validate_parameters(self, params: dict):
        self.model.validate_transverse_params(params)

    def _update_result_summary(self, result: dict, im0: dict | None = None, display_curve: dict | None = None, elapsed_seconds: float | None = None):
        self._set_results_text(
            build_transverse_summary_text(
                result=result,
                im0=im0,
                display_curve=display_curve,
                elapsed_seconds=elapsed_seconds,
            )
        )

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
        result, im0 = self._get_or_compute_analysis(params)
        display_curve = self.model.build_transverse_display_curve_from_result(params, result)
        plot_im0 = self.model.build_transverse_plot_im0_from_result(
            params,
            result,
            display_curve=display_curve,
            semantic_im0=im0,
        )

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(display_curve["W_real"], display_curve["W_imag"], linewidth=2.0, label="W(iω)")

        plot_points = plot_im0.get("points", [])
        plot_research_critical = plot_im0.get("research_critical_point")

        if plot_points:
            ax.plot(
                [p["re"] for p in plot_points],
                [0.0] * len(plot_points),
                "o",
                color="orange",
                markersize=5,
                label="Im(W)=0",
                zorder=6,
                markeredgecolor="white",
                markeredgewidth=0.7,
            )
        if plot_research_critical:
            ax.plot(
                plot_research_critical["re"],
                0.0,
                "o",
                color="green",
                markersize=9,
                label="Критическая",
                zorder=7,
                markeredgecolor="white",
                markeredgewidth=0.9,
            )

        ax.legend()
        ax.axhline(0, linestyle="--", linewidth=0.8, alpha=0.45)
        ax.axvline(0, linestyle="--", linewidth=0.8, alpha=0.45)
        self._style_plot_axes(ax, "Поперечные колебания: годограф W(p)", "Re(W)", "Im(W)", equal=True)
        self.canvas.draw()
        QApplication.processEvents()
        elapsed_seconds = perf_counter() - started_at
        self._update_result_summary(result, im0, display_curve, elapsed_seconds)
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

    def _build_export_data(self, params: dict) -> dict:
        result, im0 = self._get_or_compute_analysis(params)
        return build_transverse_export_data(
            params=params,
            preset_name=self.current_preset_name,
            result=result,
            im0=im0,
        )

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
            "JSON (*.json);;CSV (*.csv)",
        )
        if not filename:
            return

        file_format = "json" if selected_filter.startswith("JSON") else "csv"
        export_analysis_data(data, filename, file_format)
        if self.main_window is not None and hasattr(self.main_window, "status"):
            self.main_window.status.showMessage("Поперечные результаты экспортированы")
