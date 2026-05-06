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

        self.torsion_show_stable_region_cb = QCheckBox("Показывать область устойчивости D(0)")
        self.torsion_show_stable_region_cb.setChecked(True)
        self.torsion_show_crosses_cb = QCheckBox("Показывать крестики внутри устойчивой области")
        self.torsion_show_crosses_cb.setChecked(True)
        layout.addWidget(self.torsion_show_stable_region_cb)
        layout.addWidget(self.torsion_show_crosses_cb)
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

        self.trans_show_stable_region_cb = QCheckBox("Показывать область устойчивости D(0)")
        self.trans_show_stable_region_cb.setChecked(True)
        self.trans_show_crosses_cb = QCheckBox("Показывать крестики внутри устойчивой области")
        self.trans_show_crosses_cb.setChecked(True)
        layout.addWidget(self.trans_show_stable_region_cb)
        layout.addWidget(self.trans_show_crosses_cb)
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
        plotted_items = []
        for idx, item in enumerate(payload["series"]):
            line, = ax.plot(
                item["re_sigma"],
                item["delta1_scaled"],
                styles[idx % len(styles)],
                linewidth=1.6,
                markersize=5,
                label=f"L = {item['length']:g} м",
            )
            plotted_items.append((line, item))

        stable_label = None
        if self.torsion_show_stable_region_cb.isChecked():
            stable_label = self._draw_torsional_stable_region(ax, payload, plotted_items)

        handles, labels = ax.get_legend_handles_labels()
        if stable_label is not None:
            handles.append(stable_label[0])
            labels.append(stable_label[1])
        if handles:
            ax.legend(handles, labels)

        self._style_plot_axes(
            ax,
            "Крутильная диаграмма устойчивости",
            "Re(σ(ω*))",
            "δ₁ (×10⁻⁶ с)",
        )
        self.canvas.draw()

    def _draw_torsional_stable_region(self, ax, payload: dict, plotted_items: list[tuple]):
        if not plotted_items:
            return None

        y = np.asarray(payload["series"][0]["delta1_scaled"], dtype=float)
        x_rows = [np.asarray(item["re_sigma"], dtype=float) for _, item in plotted_items]
        x_matrix = np.vstack(x_rows)
        envelope = np.nanmax(x_matrix, axis=0)
        finite_mask = np.isfinite(y) & np.isfinite(envelope)
        if np.count_nonzero(finite_mask) < 2:
            return None

        y_f = y[finite_mask]
        env_f = envelope[finite_mask]
        x_min = float(np.nanmin(x_matrix))
        x_max = float(np.nanmax(x_matrix))
        span = max(abs(x_max - x_min), 1.0)
        x_right = max(0.0, x_max + 0.18 * span)

        hatch_artist = ax.fill_betweenx(
            y_f,
            env_f,
            x_right,
            facecolor="#d8ecff",
            edgecolor="#6a90b6",
            linewidth=0.0,
            alpha=0.22,
            hatch="///",
            zorder=0,
        )

        if self.torsion_show_crosses_cb.isChecked():
            y_sorted_idx = np.argsort(y_f)
            y_sorted = y_f[y_sorted_idx]
            env_sorted = env_f[y_sorted_idx]
            y_grid = np.linspace(float(np.min(y_sorted)), float(np.max(y_sorted)), 7)
            cross_x = []
            cross_y = []
            pad_x = 0.05 * span
            for y0 in y_grid:
                x_left = float(np.interp(y0, y_sorted, env_sorted))
                inner_left = x_left + pad_x
                inner_right = x_right - pad_x
                if inner_right <= inner_left:
                    continue
                for x0 in np.linspace(inner_left, inner_right, 4):
                    cross_x.append(float(x0))
                    cross_y.append(float(y0))
            if cross_x:
                ax.scatter(
                    cross_x,
                    cross_y,
                    marker="x",
                    s=28,
                    linewidths=1.0,
                    color="#4e647b",
                    alpha=0.9,
                    zorder=1,
                )

        y_span = max(float(np.max(y_f) - np.min(y_f)), 1e-6)
        text_y = float(np.max(y_f) - 0.12 * y_span)
        text_x = float(np.nanmax(env_f) + 0.22 * (x_right - np.nanmax(env_f)))
        ax.text(
            text_x,
            text_y,
            "Область устойчивости D(0)",
            fontsize=9.5,
            color="#27496d",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#c7d9eb", "alpha": 0.9},
            zorder=2,
        )

        left_bound = x_min - 0.12 * span
        y_pad = max(0.08 * y_span, 0.2)
        ax.set_xlim(left_bound, x_right)
        ax.set_ylim(float(np.min(y_f)) - y_pad, float(np.max(y_f)) + y_pad)

        descriptor = "Устойчивая область D(0)"
        if len(plotted_items) > 1:
            descriptor = "Общая устойчивая область D(0)"
        return hatch_artist, descriptor

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

        x = np.asarray(payload["x"], dtype=float)
        b_mm = np.asarray(payload["b_critical_mm"], dtype=float)
        finite = np.isfinite(x) & np.isfinite(b_mm) & (b_mm > 0.0)

        stable_label = None
        unstable_label = None
        if np.count_nonzero(finite) >= 2:
            x_f = x[finite]
            b_f = b_mm[finite]
            order = np.argsort(x_f)
            x_f = x_f[order]
            b_f = b_f[order]

            ymax = float(np.max(b_f))
            y_top = ymax * 1.22 if ymax > 0.0 else 1.0

            if self.trans_show_stable_region_cb.isChecked():
                stable_artist = ax.fill_between(
                    x_f,
                    0.0,
                    b_f,
                    facecolor="#d8ecff",
                    edgecolor="#6a90b6",
                    linewidth=0.0,
                    alpha=0.25,
                    hatch="///",
                    zorder=0,
                )
                stable_label = (stable_artist, "Устойчивая область D(0)")

            if self.trans_show_crosses_cb.isChecked():
                cross_x = []
                cross_y = []
                for x0, b0 in zip(x_f, b_f):
                    for frac in (0.25, 0.50, 0.75):
                        cross_x.append(float(x0))
                        cross_y.append(float(b0 * frac))
                if cross_x:
                    ax.scatter(
                        cross_x,
                        cross_y,
                        marker="x",
                        s=30,
                        linewidths=1.0,
                        color="#4e647b",
                        alpha=0.9,
                        zorder=1,
                    )

            text_x = float(x_f[len(x_f) // 2])
            text_y = float(np.interp(text_x, x_f, b_f) * 0.45)
            ax.text(
                text_x,
                text_y,
                "D(0): устойчиво",
                fontsize=9.5,
                color="#27496d",
                ha="center",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#c7d9eb", "alpha": 0.9},
                zorder=2,
            )

        ax.plot(
            x,
            b_mm,
            "-o",
            linewidth=1.8,
            markersize=5,
            label="граница bкр",
            zorder=3,
        )

        handles, labels = ax.get_legend_handles_labels()
        for item in (stable_label, unstable_label):
            if item is not None:
                handles.append(item[0])
                labels.append(item[1])
        if handles:
            ax.legend(handles, labels)

        self._style_plot_axes(
            ax,
            payload["title"],
            payload["x_label"],
            "Предельная глубина резания bкр (мм)",
        )
        self.canvas.draw()

    def _update_result_summary(self, payload: dict):
        if payload.get("kind") == "torsional":
            lines = [
                "Диаграмма устойчивости: крутильные колебания",
                "",
                f"Построено кривых: {len(payload.get('series', []))}",
                f"Количество найденных точек: {int(payload.get('total_valid_points', 0))}",
                "Ось X: Re(σ(ω*))",
                "Ось Y: δ₁ · 10⁻⁶ с",
                "Каждая кривая является границей устойчивости.",
                "Заштрихованная область справа показывает устойчивую область D(0).",
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
                "Кривая bкр является границей устойчивости.",
                "Область ниже кривой соответствует устойчивым режимам, область выше — неустойчивым.",
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
