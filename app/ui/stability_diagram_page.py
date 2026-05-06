from PyQt5.QtWidgets import (
    QPushButton,
    QLabel,
    QLineEdit,
    QCheckBox,
    QComboBox,
    QWidget,
    QVBoxLayout,
    QSizePolicy,
)
import numpy as np

from app.ui.analysis_page_base import AnalysisPageBase
from app.core.borebar_model import BoreBarModel


class StabilityDiagramPage(AnalysisPageBase):
    """Страница диаграмм устойчивости.

    Поддерживаются два режима:
    - крутильные колебания: прежняя диаграмма Re(σ(ω*)) от δ₁;
    - поперечные колебания: зависимость предельной глубины резания bкр
      от длины борштанги L или от времени запаздывания τ.

    Для поперечной модели используется уже реализованный в BoreBarModel
    годограф W(iω). Критической считается нетривиальная точка пересечения
    годографа с отрицательной действительной осью. Предельная глубина
    оценивается как bкр = -1 / Re(W*).
    """

    def __init__(self, main_window):
        super().__init__(main_window)
        self.model = BoreBarModel()

        left_card, left = self._make_card("Параметры диаграммы устойчивости")

        self.mode_label = QLabel(
            "Выберите вид колебаний. Для каждого режима показываются только нужные поля."
        )
        self.mode_label.setWordWrap(True)
        self.mode_label.setObjectName("resultsLabel")

        self.vibration_type_combo = QComboBox()
        self.vibration_type_combo.addItems([
            "Крутильные колебания",
            "Поперечные колебания",
        ])
        self.vibration_type_combo.currentTextChanged.connect(self._update_visible_parameter_groups)

        left.addWidget(self.mode_label)
        self._add_labeled_widget(left, "Вид колебаний", self.vibration_type_combo)

        self.torsional_group = self._build_torsional_group()
        self.transverse_group = self._build_transverse_group()
        left.addWidget(self.torsional_group)
        left.addWidget(self.transverse_group)

        result_card, self.results_label = self._make_result_card(
            "Здесь будет краткая сводка по построенной диаграмме устойчивости."
        )

        plot_btn = QPushButton("Построить диаграмму")
        plot_btn.clicked.connect(self.run_analysis)

        back_btn = QPushButton("Назад в меню")
        back_btn.clicked.connect(lambda: main_window.switch(main_window.menu))

        left.addWidget(plot_btn)
        left.addWidget(back_btn)
        left.addStretch()
        self._build_analysis_layout(left_card, result_card)
        self._update_visible_parameter_groups()

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    def _add_labeled_widget(self, layout, label_text: str, widget):
        label = QLabel(label_text)
        label.setObjectName("fieldLabel")
        layout.addWidget(label)
        layout.addWidget(widget)
        return label, widget

    def _make_group(self) -> tuple[QWidget, QVBoxLayout]:
        group = QWidget()
        group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        return group, layout

    def _build_torsional_group(self) -> QWidget:
        group, layout = self._make_group()

        section = QLabel("Крутильная диаграмма")
        section.setObjectName("sectionTitle")
        layout.addWidget(section)

        self.torsion_rho_input = QLineEdit("7800")
        self.torsion_G_input = QLineEdit("8e10")
        self.torsion_Jr_input = QLineEdit("2.57e-2")
        self.torsion_Jp_input = QLineEdit("1.9e-5")
        self.torsion_delta1_input = QLineEdit("3.44e-6")
        self.torsion_length_input = QLineEdit("2.5")
        self.torsion_lengths_input = QLineEdit("2.5, 3, 4, 5, 6")
        self.torsion_compare_lengths_cb = QCheckBox("Сравнить несколько длин L")
        self.torsion_compare_lengths_cb.setChecked(True)
        self.torsion_length_input.setEnabled(False)
        self.torsion_compare_lengths_cb.toggled.connect(self._toggle_torsion_length_mode)

        self.torsion_omega_start_input = QLineEdit("1000")
        self.torsion_omega_end_input = QLineEdit("15000")
        self.torsion_omega_step_input = QLineEdit("1")

        for label_text, widget in [
            ("Плотность материала (ρ, кг/м³)", self.torsion_rho_input),
            ("Модуль сдвига (G, Па)", self.torsion_G_input),
            ("Момент инерции режущей головки (Jr, кг·м²)", self.torsion_Jr_input),
            ("Полярный момент инерции (Jp, м⁴)", self.torsion_Jp_input),
            ("Базовое δ₁ (с)", self.torsion_delta1_input),
        ]:
            self._add_labeled_widget(layout, label_text, widget)

        layout.addWidget(self.torsion_compare_lengths_cb)
        self._add_labeled_widget(layout, "Одна длина L (м)", self.torsion_length_input)
        self._add_labeled_widget(layout, "Список длин L (м)", self.torsion_lengths_input)
        self._add_frequency_controls(
            layout,
            self.torsion_omega_start_input,
            self.torsion_omega_end_input,
            self.torsion_omega_step_input,
        )
        return group

    def _build_transverse_group(self) -> QWidget:
        group, layout = self._make_group()

        section = QLabel("Поперечная диаграмма")
        section.setObjectName("sectionTitle")
        layout.addWidget(section)

        self.transverse_dependency_combo = QComboBox()
        self.transverse_dependency_combo.addItems([
            "bкр от длины L",
            "bкр от запаздывания τ",
        ])
        self.transverse_dependency_combo.currentTextChanged.connect(self._update_transverse_dependency_fields)

        self.trans_E_input = QLineEdit("2.1e11")
        self.trans_rho_input = QLineEdit("7800")
        self.trans_R_input = QLineEdit("0.04")
        self.trans_r_input = QLineEdit("0.035")
        self.trans_K_input = QLineEdit("6e5")
        self.trans_h_input = QLineEdit("3.02141544835e-05")
        self.trans_mu_input = QLineEdit("0.6")
        self.trans_length_input = QLineEdit("2.7")
        self.trans_lengths_input = QLineEdit("1.8, 2.0, 2.3, 2.7, 3.0, 3.5")
        self.trans_tau_input = QLineEdit("0.1")
        self.trans_taus_input = QLineEdit("0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3")
        self.trans_omega_start_input = QLineEdit("0")
        self.trans_omega_end_input = QLineEdit("220")
        self.trans_omega_step_input = QLineEdit("0.1")

        self._add_labeled_widget(layout, "Зависимость", self.transverse_dependency_combo)
        for label_text, widget in [
            ("Модуль Юнга (E, Па)", self.trans_E_input),
            ("Плотность материала (ρ, кг/м³)", self.trans_rho_input),
            ("Внешний радиус борштанги (R, м)", self.trans_R_input),
            ("Внутренний радиус борштанги (r, м)", self.trans_r_input),
            ("Динамическая жёсткость резания (K, Н/м)", self.trans_K_input),
            ("Коэффициент внутреннего трения (h, с)", self.trans_h_input),
            ("Коэффициент регенеративной связи (μ)", self.trans_mu_input),
        ]:
            self._add_labeled_widget(layout, label_text, widget)

        self.trans_length_label, _ = self._add_labeled_widget(
            layout, "Фиксированная длина L (м)", self.trans_length_input
        )
        self.trans_lengths_label, _ = self._add_labeled_widget(
            layout, "Список длин L (м)", self.trans_lengths_input
        )
        self.trans_tau_label, _ = self._add_labeled_widget(
            layout, "Фиксированное запаздывание τ (с)", self.trans_tau_input
        )
        self.trans_taus_label, _ = self._add_labeled_widget(
            layout, "Список значений τ (с)", self.trans_taus_input
        )

        self._add_frequency_controls(
            layout,
            self.trans_omega_start_input,
            self.trans_omega_end_input,
            self.trans_omega_step_input,
        )
        return group

    def _toggle_torsion_length_mode(self, checked: bool):
        self.torsion_lengths_input.setEnabled(checked)
        self.torsion_length_input.setEnabled(not checked)

    def _update_visible_parameter_groups(self):
        is_torsional = self.vibration_type_combo.currentText().startswith("Крутильные")
        self.torsional_group.setVisible(is_torsional)
        self.transverse_group.setVisible(not is_torsional)
        if not is_torsional:
            self._update_transverse_dependency_fields()

    def _update_transverse_dependency_fields(self):
        by_length = self.transverse_dependency_combo.currentText() == "bкр от длины L"

        self.trans_lengths_label.setVisible(by_length)
        self.trans_lengths_input.setVisible(by_length)
        self.trans_tau_label.setVisible(by_length)
        self.trans_tau_input.setVisible(by_length)

        self.trans_length_label.setVisible(not by_length)
        self.trans_length_input.setVisible(not by_length)
        self.trans_taus_label.setVisible(not by_length)
        self.trans_taus_input.setVisible(not by_length)

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_positive_list(text: str, value_name: str) -> list[float]:
        raw = text.strip().replace(";", ",")
        parts = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
        values = []
        for part in parts:
            try:
                value = float(part.replace(" ", ""))
            except ValueError:
                continue
            if np.isfinite(value) and value > 0.0:
                values.append(value)
        if not values:
            raise ValueError(f"Введите хотя бы одно корректное значение: {value_name}.")
        return values

    @staticmethod
    def _require_positive(value: float, value_name: str) -> float:
        value = float(value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"Параметр {value_name} должен быть > 0.")
        return value

    # ------------------------------------------------------------------
    # Torsional diagram
    # ------------------------------------------------------------------

    def _torsion_base_params(self) -> dict:
        return {
            "rho": float(self.torsion_rho_input.text()),
            "G": float(self.torsion_G_input.text()),
            "Jr": float(self.torsion_Jr_input.text()),
            "Jp": float(self.torsion_Jp_input.text()),
            "delta1": float(self.torsion_delta1_input.text()),
            "omega_start": float(self.torsion_omega_start_input.text()),
            "omega_end": float(self.torsion_omega_end_input.text()),
            "omega_step": float(self.torsion_omega_step_input.text()),
        }

    def _torsion_lengths(self) -> list[float]:
        if self.torsion_compare_lengths_cb.isChecked():
            return self._parse_positive_list(self.torsion_lengths_input.text(), "длины L")
        return [self._require_positive(float(self.torsion_length_input.text()), "L")]

    @staticmethod
    def _delta1_multipliers() -> list[int]:
        return [1, 2, 3, 4, 6, 10]

    def _find_torsional_diagram_critical_point(self, params: dict) -> dict | None:
        """Найти точку для крутильной диаграммы устойчивости.

        Для диаграммы устойчивости нельзя ограничиваться пользовательской
        частотной сеткой omega_start..omega_end. В исходном Matlab-расчёте
        корень для Sigma(I,J) ищется функцией fzero в исследовательском окне:
        [500; 2000] для L < 5.5 м и [500; 1000] для длинной борштанги.

        Поэтому здесь корень Im(σ(iω))=0 ищется напрямую в этом окне.
        Поля omega_start, omega_end и omega_step остаются параметрами
        отображения/дискретизации, но не должны отрезать физически нужный
        корень диаграммы устойчивости.
        """
        params = self.model.validate_torsional_params(params)
        policy = self.model._torsional_research_policy(params)
        w_low = float(policy["omega_low"])
        w_high = float(policy["omega_high"])
        if not (np.isfinite(w_low) and np.isfinite(w_high)) or w_high <= w_low:
            return None

        def _eval_sigma_at(w: float) -> tuple[float, float]:
            _, re_v, im_v, _ = BoreBarModel._evaluate_torsional_sigma_positive(
                params,
                np.asarray([float(w)], dtype=float),
            )
            return float(re_v[0]), float(im_v[0])

        def im_func(w: float) -> float:
            return _eval_sigma_at(w)[1]

        def re_func(w: float) -> float:
            return _eval_sigma_at(w)[0]

        # Сначала проверяем прямой интервал, как в fzero(f,[a b]).
        y_low = im_func(w_low)
        y_high = im_func(w_high)
        if np.isfinite(y_low) and np.isfinite(y_high) and y_low * y_high <= 0.0:
            w_root = BoreBarModel._refine_root_on_interval(im_func, w_low, y_low, w_high, y_high)
            re_root = re_func(w_root)
            if np.isfinite(w_root) and np.isfinite(re_root):
                return {
                    "omega": float(w_root),
                    "re": float(re_root),
                    "im": 0.0,
                    "frequency": float(w_root / (2.0 * np.pi)),
                    "selection_policy": policy,
                }

        # Если на концах окна нет смены знака, ищем первый локальный интервал
        # смены знака внутри окна. Это защищает GUI от грубого шага сетки.
        search_grid = np.linspace(w_low, w_high, 1001, dtype=float)
        im_values = np.asarray([im_func(w) for w in search_grid], dtype=float)
        for i, j in BoreBarModel._sign_change_intervals(search_grid, im_values):
            w1, w2 = float(search_grid[i]), float(search_grid[j])
            y1, y2 = float(im_values[i]), float(im_values[j])
            w_root = BoreBarModel._refine_root_on_interval(im_func, w1, y1, w2, y2)
            re_root = re_func(w_root)
            if np.isfinite(w_root) and np.isfinite(re_root):
                return {
                    "omega": float(w_root),
                    "re": float(re_root),
                    "im": 0.0,
                    "frequency": float(w_root / (2.0 * np.pi)),
                    "selection_policy": policy,
                }
        return None

    def _build_torsional_diagram_payload(self) -> dict:
        base_params = self._torsion_base_params()
        lengths = self._torsion_lengths()
        multipliers = self._delta1_multipliers()
        delta1_scaled = np.asarray([base_params["delta1"] * m * 1e6 for m in multipliers], dtype=float)

        series = []
        total_valid = 0
        for length in lengths:
            x_values = []
            critical_points = []
            points_found = 0
            for multiplier in multipliers:
                params = {
                    **base_params,
                    "length": float(length),
                    "multiplier": float(multiplier),
                }
                crit = self._find_torsional_diagram_critical_point(params)
                critical_points.append(crit)
                if crit is None or not np.isfinite(float(crit.get("re", np.nan))):
                    x_values.append(np.nan)
                    continue
                x_values.append(float(crit["re"]))
                points_found += 1
                total_valid += 1

            series.append({
                "length": float(length),
                "delta1_scaled": delta1_scaled.copy(),
                "re_sigma": np.asarray(x_values, dtype=float),
                "critical_points": critical_points,
                "valid_points": points_found,
            })

        return {
            "kind": "torsional",
            "series": series,
            "total_valid_points": int(total_valid),
            "multipliers": multipliers,
        }

    def _plot_torsional_diagram(self, payload: dict):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        styles = ["-o", "-.x", "--d", "--s", "--*"]
        for idx, item in enumerate(payload["series"]):
            ax.plot(
                item["re_sigma"],
                item["delta1_scaled"],
                styles[idx % len(styles)],
                linewidth=1.6,
                markersize=5,
                label=f"L = {item['length']:g} м",
            )

        if len(payload["series"]) > 1:
            ax.legend()

        self._style_plot_axes(
            ax,
            "Крутильная диаграмма устойчивости",
            "Re(σ(ω*))",
            "δ₁ (×10⁻⁶ с)",
        )
        self.canvas.draw()

    # ------------------------------------------------------------------
    # Transverse diagram
    # ------------------------------------------------------------------

    def _transverse_base_params(self) -> dict:
        return {
            "E": float(self.trans_E_input.text()),
            "rho": float(self.trans_rho_input.text()),
            "R": float(self.trans_R_input.text()),
            "r": float(self.trans_r_input.text()),
            "K_cut": float(self.trans_K_input.text()),
            "h": float(self.trans_h_input.text()),
            "mu": float(self.trans_mu_input.text()),
            "omega_start": float(self.trans_omega_start_input.text()),
            "omega_end": float(self.trans_omega_end_input.text()),
            "omega_step": float(self.trans_omega_step_input.text()),
            "transverse_modal_shape_variant": "verified_cantilever_first_mode_phi",
        }

    @staticmethod
    def _critical_depth_from_transverse_point(critical: dict | None) -> float:
        if critical is None:
            return np.nan
        re_value = float(critical.get("re", np.nan))
        if not np.isfinite(re_value) or re_value >= 0.0:
            return np.nan
        return -1.0 / re_value

    def _calculate_transverse_depth_point(self, params: dict) -> tuple[float, dict | None, int]:
        self.model.validate_transverse_params(params)
        result = self.model.calculate_transverse(params)
        im0 = self.model.find_transverse_im0_points_from_result(params, result)
        critical = im0.get("research_critical_point")
        b_critical = self._critical_depth_from_transverse_point(critical)
        return b_critical, critical, len(im0.get("points", []) or [])

    def _build_transverse_diagram_payload(self) -> dict:
        base = self._transverse_base_params()
        by_length = self.transverse_dependency_combo.currentText() == "bкр от длины L"

        x_values = []
        b_values_m = []
        point_counts = []
        critical_points = []

        if by_length:
            tau = self._require_positive(float(self.trans_tau_input.text()), "τ")
            variable_values = self._parse_positive_list(self.trans_lengths_input.text(), "длины L")
            variable_name = "length"
            x_label = "Длина борштанги L (м)"
            title = "Поперечная диаграмма устойчивости: bкр(L)"
            fixed_description = f"τ = {tau:g} с"
        else:
            length = self._require_positive(float(self.trans_length_input.text()), "L")
            variable_values = self._parse_positive_list(self.trans_taus_input.text(), "значения τ")
            variable_name = "tau"
            x_label = "Время запаздывания τ (с)"
            title = "Поперечная диаграмма устойчивости: bкр(τ)"
            fixed_description = f"L = {length:g} м"

        for value in variable_values:
            params = dict(base)
            if by_length:
                params["length"] = float(value)
                params["tau"] = tau
            else:
                params["length"] = length
                params["tau"] = float(value)

            b_critical, critical, point_count = self._calculate_transverse_depth_point(params)
            x_values.append(float(value))
            b_values_m.append(float(b_critical) if np.isfinite(b_critical) else np.nan)
            critical_points.append(critical)
            point_counts.append(point_count)

        b_values_m = np.asarray(b_values_m, dtype=float)
        return {
            "kind": "transverse",
            "variable_name": variable_name,
            "x": np.asarray(x_values, dtype=float),
            "b_critical_m": b_values_m,
            "b_critical_mm": b_values_m * 1000.0,
            "critical_points": critical_points,
            "point_counts": point_counts,
            "total_valid_points": int(np.count_nonzero(np.isfinite(b_values_m))),
            "title": title,
            "x_label": x_label,
            "fixed_description": fixed_description,
        }

    def _plot_transverse_diagram(self, payload: dict):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(
            payload["x"],
            payload["b_critical_mm"],
            "-o",
            linewidth=1.8,
            markersize=5,
            label="bкр",
        )
        ax.legend()
        self._style_plot_axes(
            ax,
            payload["title"],
            payload["x_label"],
            "Предельная глубина резания bкр (мм)",
        )
        self.canvas.draw()

    # ------------------------------------------------------------------
    # Summary and run
    # ------------------------------------------------------------------

    def _update_result_summary(self, payload: dict):
        if payload.get("kind") == "torsional":
            lines = [
                "Диаграмма устойчивости: крутильные колебания",
                "",
                f"Построено кривых: {len(payload.get('series', []))}",
                f"Количество найденных точек: {int(payload.get('total_valid_points', 0))}",
                "Ось X: Re(σ(ω*))",
                "Ось Y: δ₁ · 10⁻⁶ с",
            ]
        else:
            finite_b = np.asarray(payload.get("b_critical_mm", []), dtype=float)
            finite_b = finite_b[np.isfinite(finite_b)]
            if finite_b.size:
                range_line = f"Диапазон bкр: {np.min(finite_b):.6g} ... {np.max(finite_b):.6g} мм"
            else:
                range_line = "Предельные глубины не найдены: нет критических пересечений с отрицательной действительной осью."

            lines = [
                "Диаграмма устойчивости: поперечные колебания",
                "",
                payload.get("fixed_description", ""),
                f"Количество рассчитанных режимов: {len(payload.get('x', []))}",
                f"Количество найденных предельных глубин: {int(payload.get('total_valid_points', 0))}",
                range_line,
                "Ось X: варьируемый параметр L или τ",
                "Ось Y: bкр = -1 / Re(W*)",
            ]
        self._set_results_text("\n".join(lines))

    def run_analysis(self):
        try:
            if self.vibration_type_combo.currentText().startswith("Крутильные"):
                payload = self._build_torsional_diagram_payload()
                self._plot_torsional_diagram(payload)
            else:
                payload = self._build_transverse_diagram_payload()
                self._plot_transverse_diagram(payload)
        except ValueError as e:
            self._show_error(str(e))
            return
        except Exception as e:
            self._show_error(f"Не удалось построить диаграмму: {e}")
            return

        self._update_result_summary(payload)
        if self.main_window is not None and hasattr(self.main_window, "status"):
            self.main_window.status.showMessage("Диаграмма устойчивости построена")
