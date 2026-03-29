from time import perf_counter

from PyQt5.QtWidgets import (
    QPushButton,
    QLabel,
    QLineEdit,
    QFileDialog,
    QApplication,
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
        self._cached_signature = None
        self._cached_analysis = None

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

        result_card, self.results_label = self._make_result_card(
            "После расчёта здесь появится краткая сводка по критическим точкам и параметрам модели."
        )

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


    @staticmethod
    def _params_signature(params: dict):
        return tuple(sorted((key, repr(value)) for key, value in params.items()))

    def _compute_im0_from_result(self, result: dict, params: dict) -> dict:
        omega = np.asarray(result.get("physical_omega", result.get("omega", [])), dtype=float)
        sig_re = np.asarray(result.get("physical_sigma_real", result.get("sigma_real", [])), dtype=float)
        sig_im = np.asarray(result.get("physical_sigma_imag", result.get("sigma_imag", [])), dtype=float)

        points = []
        eps = float(params.get("im0_eps_torsional", 1e-9))
        finite_zero = np.isfinite(omega) & np.isfinite(sig_re) & np.isfinite(sig_im) & (np.abs(sig_im) <= eps)
        for idx in np.where(finite_zero)[0]:
            w = float(omega[idx])
            points.append({
                "omega": w,
                "re": float(sig_re[idx]),
                "im": 0.0,
                "frequency": float(w / (2.0 * np.pi)),
            })

        for i, j in BoreBarModel._sign_change_intervals(omega, sig_im):
            w1, w2 = omega[i], omega[j]
            y1, y2 = sig_im[i], sig_im[j]
            w0 = BoreBarModel._linear_root(w1, y1, w2, y2)
            re0 = np.interp(w0, [w1, w2], [sig_re[i], sig_re[j]])
            points.append({
                "omega": float(w0),
                "re": float(re0),
                "im": 0.0,
                "frequency": float(w0 / (2.0 * np.pi)),
            })

        dedup = []
        omega_tol = float(params.get("im0_omega_tol_torsional", max(float(params.get("omega_step", 1.0)) * 0.5, 1e-9)))
        re_tol = float(params.get("im0_re_tol_torsional", 1e-6))
        for p in sorted(points, key=lambda item: (item["omega"], item["re"])):
            if dedup and abs(p["omega"] - dedup[-1]["omega"]) <= omega_tol and abs(p["re"] - dedup[-1]["re"]) <= re_tol:
                continue
            dedup.append(p)

        research_critical_point, policy_meta = BoreBarModel._select_torsional_research_critical_point(dedup, params)
        minimum_re_critical_point = min(dedup, key=lambda p: p["re"]) if dedup else None
        return {
            "all_im0_points": dedup,
            "research_critical_point": research_critical_point,
            "minimum_re_critical_point": minimum_re_critical_point,
            "points": dedup,
            "critical": research_critical_point,
            "source_curve": "physical_positive_branch",
            "critical_selection_policy": policy_meta,
        }

    def _get_or_compute_analysis(self, params: dict) -> tuple[dict, dict]:
        signature = self._params_signature(params)
        if self._cached_signature == signature and self._cached_analysis is not None:
            cached = self._cached_analysis
            return cached["result"], cached["im0"]

        result = self.model.calculate_torsional(params)
        im0 = self._compute_im0_from_result(result, params)
        self._cached_signature = signature
        self._cached_analysis = {"result": result, "im0": im0}
        return result, im0

    @staticmethod
    def _build_display_curve_from_physical(result: dict, params: dict) -> dict:
        omega_pos = np.asarray(result.get("physical_omega", result.get("omega", [])), dtype=float)
        re_pos = np.asarray(result.get("physical_sigma_real", result.get("sigma_real", [])), dtype=float)
        im_pos = np.asarray(result.get("physical_sigma_imag", result.get("sigma_imag", [])), dtype=float)

        if omega_pos.size == 0:
            empty = np.array([], dtype=float)
            return {
                "omega": empty,
                "re": empty,
                "im": empty,
                "policy": "display_rebuilt_from_physical_branch_no_points",
            }

        if float(params.get("omega_start", 0.0)) < 0.0:
            omega = np.concatenate([-omega_pos[::-1], omega_pos])
            re = np.concatenate([re_pos[::-1], re_pos])
            im = np.concatenate([-im_pos[::-1], im_pos])
            policy = "display_rebuilt_as_exact_conjugate_mirror_of_physical_positive_branch"
        else:
            omega = omega_pos.copy()
            re = re_pos.copy()
            im = im_pos.copy()
            policy = "display_rebuilt_from_physical_positive_branch"

        return {
            "omega": omega,
            "re": re,
            "im": im,
            "policy": policy,
        }

    @staticmethod
    def _compute_plot_policy(display_curve: dict, points=None, critical=None) -> dict:
        re = np.asarray(display_curve.get("re", []), dtype=float)
        im = np.asarray(display_curve.get("im", []), dtype=float)
        finite = np.isfinite(re) & np.isfinite(im)

        if np.count_nonzero(finite) < 2:
            return {
                "xlim": (-1.0, 1.0),
                "ylim": (-1.0, 1.0),
                "y_clip": (-1.0, 1.0),
                "origin_band_left": 0.0,
                "mode": "fallback_empty",
            }

        x_all = re[finite]
        y_all = im[finite]

        if points:
            px = np.asarray([p["re"] for p in points if np.isfinite(p.get("re", np.nan))], dtype=float)
            if px.size:
                x_all = np.concatenate([x_all, px])
        if critical and np.isfinite(critical.get("re", np.nan)):
            x_all = np.concatenate([x_all, np.asarray([float(critical["re"])], dtype=float)])

        xmin = float(np.min(x_all))
        xmax = float(np.max(x_all))
        xspan = max(xmax - xmin, 1e-9)
        xpad = xspan * 0.08
        xlim = (xmin - xpad, xmax + xpad)

        # Разрешаем резать только узкую правую полосу около начала координат.
        # Это соответствует вертикальным выбросам около Re≈0, а не всей верхней/нижней дуге.
        origin_band_fraction = 0.06
        origin_band_left = xmax - xspan * origin_band_fraction

        non_origin_mask = finite & (re < origin_band_left)
        if np.count_nonzero(non_origin_mask) >= 8:
            y_for_view = np.abs(im[non_origin_mask])
            mode = "origin_only_segment_clip"
        else:
            y_for_view = np.abs(y_all)
            mode = "fallback_full_curve_no_origin_separation"

        ymax_visible = float(np.max(y_for_view)) if y_for_view.size else float(np.max(np.abs(y_all)))
        ymax_visible = max(ymax_visible, 1e-9)

        y_clip_max = ymax_visible * 1.03
        y_window_max = ymax_visible * 1.12

        return {
            "xlim": xlim,
            "ylim": (-y_window_max, y_window_max),
            "y_clip": (-y_clip_max, y_clip_max),
            "origin_band_left": origin_band_left,
            "mode": mode,
        }

    @staticmethod
    def _clip_segment_to_horizontal_strip(x1, y1, x2, y2, ylo, yhi):
        eps = 1e-15
        dy = y2 - y1

        if abs(dy) <= eps:
            if ylo <= y1 <= yhi:
                return [(x1, y1), (x2, y2)], False
            return [], True

        t_a = (ylo - y1) / dy
        t_b = (yhi - y1) / dy
        t_enter = max(0.0, min(t_a, t_b))
        t_exit = min(1.0, max(t_a, t_b))

        if t_enter > t_exit:
            return [], True

        xa = x1 + (x2 - x1) * t_enter
        ya = y1 + dy * t_enter
        xb = x1 + (x2 - x1) * t_exit
        yb = y1 + dy * t_exit

        partial = not (np.isclose(t_enter, 0.0) and np.isclose(t_exit, 1.0))
        return [(xa, ya), (xb, yb)], partial

    @staticmethod
    def _build_plot_curve(display_curve: dict, plot_policy: dict) -> dict:
        re = np.asarray(display_curve.get("re", []), dtype=float)
        im = np.asarray(display_curve.get("im", []), dtype=float)
        n = re.size

        ylo, yhi = plot_policy["y_clip"]
        origin_band_left = float(plot_policy["origin_band_left"])

        out_x = []
        out_y = []
        clipped_segments = 0

        def append_point(xv, yv):
            if out_x and np.isfinite(out_x[-1]) and np.isfinite(out_y[-1]):
                if np.isclose(out_x[-1], xv) and np.isclose(out_y[-1], yv):
                    return
            out_x.append(float(xv))
            out_y.append(float(yv))

        def append_gap():
            if not out_x or np.isfinite(out_x[-1]) or np.isfinite(out_y[-1]):
                out_x.append(np.nan)
                out_y.append(np.nan)

        if n == 0:
            return {
                "re": np.array([], dtype=float),
                "im": np.array([], dtype=float),
                "clipped_count": 0,
                "policy": "origin_only_segment_clip_no_points",
            }

        for i in range(n - 1):
            x1, y1 = re[i], im[i]
            x2, y2 = re[i + 1], im[i + 1]

            if not (np.isfinite(x1) and np.isfinite(y1) and np.isfinite(x2) and np.isfinite(y2)):
                append_gap()
                continue

            near_origin_spike_zone = max(x1, x2) >= origin_band_left

            if not near_origin_spike_zone:
                append_point(x1, y1)
                append_point(x2, y2)
                continue

            kept, partial = TorsionalPage._clip_segment_to_horizontal_strip(x1, y1, x2, y2, ylo, yhi)
            if not kept:
                clipped_segments += 1
                append_gap()
                continue

            if partial:
                clipped_segments += 1
                append_point(*kept[0])
                append_point(*kept[-1])
                append_gap()
            else:
                append_point(*kept[0])
                append_point(*kept[-1])

        return {
            "re": np.asarray(out_x, dtype=float),
            "im": np.asarray(out_y, dtype=float),
            "clipped_count": int(clipped_segments),
            "policy": "segment_clip_only_for_spikes_near_origin",
        }

    def _update_result_summary(
        self,
        result: dict,
        critical: dict | None,
        display_curve: dict | None = None,
        plot_policy: dict | None = None,
        plot_curve: dict | None = None,
        elapsed_seconds: float | None = None,
    ):
        text = build_torsional_summary(result, critical)
        if display_curve is not None:
            text += "\n\nОтображение графика:"
            text += f"\nDisplay-ветвь: {display_curve.get('policy', 'unknown')}"
        if plot_policy is not None:
            text += f"\nРежим показа: {plot_policy.get('mode', 'unknown')}"
        if plot_curve is not None:
            text += f"\nОбрезано сегментов-шпилей у начала координат: {int(plot_curve.get('clipped_count', 0))}"
            text += (
                "\nВажно: физическая кривая, special points, критическая точка и экспорт не изменялись; "
                "отрисовка меняется только для шпилей около Re≈0."
            )
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
        result, im0 = self._get_or_compute_analysis(params)
        points = im0.get("points", [])
        critical = im0.get("critical")

        display_curve = self._build_display_curve_from_physical(result, params)
        plot_policy = self._compute_plot_policy(display_curve, points=points, critical=critical)
        plot_curve = self._build_plot_curve(display_curve, plot_policy)

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(plot_curve["re"], plot_curve["im"], linewidth=2.0, label="σ(iω)")

        if points:
            ax.plot([p["re"] for p in points], [0.0] * len(points), "o", markersize=5, label="Im(σ)=0")
        if critical:
            ax.plot(critical["re"], 0.0, "o", markersize=9, label="Критическая")

        ax.legend()
        ax.axhline(0, linestyle="--", linewidth=0.8, alpha=0.45)
        ax.axvline(0, linestyle="--", linewidth=0.8, alpha=0.45)
        ax.set_xlim(*plot_policy["xlim"])
        ax.set_ylim(*plot_policy["ylim"])
        self._style_plot_axes(ax, "Крутильные колебания: кривая D-разбиения σ(iω)", "Re(σ)", "Im(σ)")
        self.canvas.draw()
        QApplication.processEvents()
        elapsed_seconds = perf_counter() - started_at
        self._update_result_summary(result, critical, display_curve, plot_policy, plot_curve, elapsed_seconds)
        if self.main_window is not None and hasattr(self.main_window, "status"):
            self.main_window.status.showMessage(f"Крутильный анализ выполнен за {elapsed_seconds:.3f} с")

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
        result, im0 = self._get_or_compute_analysis(params)
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
            self._default_export_path("torsional_results.json"),
            "JSON (*.json);;CSV (*.csv)",
        )
        if not filename:
            return

        file_format = "json" if selected_filter.startswith("JSON") else "csv"
        export_analysis_data(data, filename, file_format)
        if self.main_window is not None and hasattr(self.main_window, "status"):
            self.main_window.status.showMessage("Крутильные результаты экспортированы")
