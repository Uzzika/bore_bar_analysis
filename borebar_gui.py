"""
borebar_gui.py

Упрощённый и аккуратно оформленный графический интерфейс
для анализа колебаний борштанги:

- крутильные колебания;
- продольные колебания;
- поперечные колебания;
- диаграмма устойчивости по δ₁.

Сделан максимально понятным для чтения и проверки в рамках курсовой.
"""

import sys
import json
import csv

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QTabWidget,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QDoubleSpinBox,
    QComboBox,
    QPushButton,
    QStatusBar,
    QFileDialog,
    QMessageBox,
    QDialog,
    QSlider,
)
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

from borebar_model import BoreBarModel


class BoreBarGUI(QMainWindow):
    """
    Главное окно программы.

    Слева — панель параметров,
    справа — вкладки с графиками и блок кнопок управления.
    """

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Анализ колебаний борштанги")
        self.setGeometry(100, 100, 1600, 900)

        self.model = BoreBarModel()

        self._init_ui()
        self._init_parameters()

    # ----------------------------------------------------------------------
    # Инициализация интерфейса
    # ----------------------------------------------------------------------

    def _init_ui(self):
        """Создание основной компоновки окна."""
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(10)

        # Левая панель с параметрами
        self._create_parameters_panel()
        main_layout.addWidget(self.params_panel, stretch=1)

        # Правая часть: вкладки + кнопки управления
        self._create_right_panel()
        main_layout.addWidget(self.right_panel, stretch=3)

        # Строка состояния
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    # ----------------------------------------------------------------------
    # Левая панель: параметры
    # ----------------------------------------------------------------------

    def _create_parameters_panel(self):
        self.params_panel = QGroupBox("Параметры системы")
        self.params_layout = QVBoxLayout(self.params_panel)
        self.params_layout.setSpacing(8)

        self._create_material_group()
        self._create_geometry_group()
        self._create_transverse_group()
        self._create_friction_group()
        self._create_params_buttons()
        self._create_intersection_panel()

    def _create_material_group(self):
        group = QGroupBox("Материальные свойства")
        layout = QVBoxLayout(group)

        # Все спинбоксы сохраняем как self.<имя>_spin для удобного доступа
        layout.addWidget(
            self._create_labeled_spinbox(
                "Плотность ρ (кг/м³):",
                "rho",
                default=7800,
                minimum=1000,
                maximum=20000,
            )
        )
        layout.addWidget(
            self._create_labeled_spinbox(
                "Модуль сдвига G (Па):",
                "G",
                default=8e10,
                minimum=1e9,
                maximum=1e12,
            )
        )
        layout.addWidget(
            self._create_labeled_spinbox(
                "Модуль Юнга E (Па):",
                "E",
                default=200e9,
                minimum=1e9,
                maximum=1e12,
            )
        )

        self.params_layout.addWidget(group)

    def _create_geometry_group(self):
        group = QGroupBox("Геометрические параметры")
        layout = QVBoxLayout(group)

        # Длина борштанги — выпадающий список
        length_row = QWidget()
        length_layout = QHBoxLayout(length_row)
        length_layout.setContentsMargins(0, 0, 0, 0)

        length_label = QLabel("Длина борштанги L (м):")
        self.length_combo = QComboBox()
        self.length_combo.addItems(["2.5", "3", "4", "5", "6"])

        length_layout.addWidget(length_label)
        length_layout.addWidget(self.length_combo)
        layout.addWidget(length_row)

        # Площадь, моменты
        layout.addWidget(
            self._create_labeled_spinbox(
                "Площадь сечения S (м²):",
                "S",
                default=2e-4,
                minimum=1e-6,
                maximum=1e-2,
            )
        )
        layout.addWidget(
            self._create_labeled_spinbox(
                "Момент инерции головки J_r (кг·м²):",
                "Jr",
                default=2.57e-2,
                minimum=1e-5,
                maximum=1,
            )
        )
        layout.addWidget(
            self._create_labeled_spinbox(
                "Полярный момент инерции J_p (м⁴):",
                "Jp",
                default=1.9e-5,
                minimum=1e-8,
                maximum=1e-2,
            )
        )

        self.params_layout.addWidget(group)

    def _create_transverse_group(self):
        group = QGroupBox("Параметры поперечных колебаний")
        layout = QVBoxLayout(group)

        layout.addWidget(
            self._create_labeled_spinbox(
                "Внешний радиус R (м):",
                "R",
                default=0.04,
                minimum=0.001,
                maximum=0.2,
            )
        )
        layout.addWidget(
            self._create_labeled_spinbox(
                "Внутренний радиус r (м):",
                "r",
                default=0.035,
                minimum=0.0,
                maximum=0.2,
            )
        )
        layout.addWidget(
            self._create_labeled_spinbox(
                "Дин. жёсткость резания K (Н/м):",
                "K_cut",
                default=6e5,
                minimum=1e3,
                maximum=1e8,
            )
        )
        layout.addWidget(
            self._create_labeled_spinbox(
                "Коэф. демпфирования β:",
                "beta",
                default=0.3,
                minimum=0.0,
                maximum=10.0,
            )
        )
        # h пока не используется в модели, но оставляем в интерфейсе
        layout.addWidget(
            self._create_labeled_spinbox(
                "Коэф. внутр. трения h:",
                "h",
                default=0.0,
                minimum=0.0,
                maximum=10.0,
            )
        )

        self.params_layout.addWidget(group)

    def _create_friction_group(self):
        group = QGroupBox("Параметры трения и запаздывания")
        layout = QVBoxLayout(group)

        # Крутильное внутреннее трение δ₁ и множитель
        layout.addWidget(
            self._create_labeled_spinbox(
                "Базовый коэф. δ₁ (с):",
                "delta1",
                default=3.44e-6,
                minimum=1e-8,
                maximum=1e-4,
            )
        )

        mult_row = QWidget()
        mult_layout = QHBoxLayout(mult_row)
        mult_layout.setContentsMargins(0, 0, 0, 0)
        mult_label = QLabel("Множитель для δ₁:")
        self.multiplier_combo = QComboBox()
        self.multiplier_combo.addItems(["1", "2", "3", "4", "6", "10"])
        mult_layout.addWidget(mult_label)
        mult_layout.addWidget(self.multiplier_combo)
        layout.addWidget(mult_row)

        # Параметры μ и τ для запаздывания (продольные и поперечные)
        layout.addWidget(
            self._create_labeled_spinbox(
                "Коэф. μ (безразм.):",
                "mu",
                default=0.1,
                minimum=0.0,
                maximum=1.0,
            )
        )
        layout.addWidget(
            self._create_labeled_spinbox(
                "Время запаздывания τ (с):",
                "tau",
                default=60e-3,
                minimum=1e-3,
                maximum=1.0,
            )
        )

        self.params_layout.addWidget(group)

    def _create_params_buttons(self):
        row = QHBoxLayout()
        row.setSpacing(6)

        self.save_params_btn = QPushButton("Сохранить параметры")
        self.load_params_btn = QPushButton("Загрузить параметры")
        self.reset_params_btn = QPushButton("Сбросить")

        row.addWidget(self.save_params_btn)
        row.addWidget(self.load_params_btn)
        row.addWidget(self.reset_params_btn)

        self.params_layout.addLayout(row)

        self.save_params_btn.clicked.connect(self._save_parameters)
        self.load_params_btn.clicked.connect(self._load_parameters)
        self.reset_params_btn.clicked.connect(self._init_parameters)

    def _create_intersection_panel(self):
        """Панель с текстовым отображением точки пересечения σ(p) с осью Im σ = 0."""
        group = QGroupBox("Точка пересечения Im(σ) = 0 (крутильные колебания)")
        layout = QVBoxLayout(group)

        self.intersection_label = QLabel("Пересечение не вычислено.")
        self.intersection_label.setWordWrap(True)

        layout.addWidget(self.intersection_label)
        self.params_layout.addWidget(group)

    # ----------------------------------------------------------------------
    # Вспомогательная функция создания пары (Label + SpinBox)
    # ----------------------------------------------------------------------

    def _create_labeled_spinbox(
        self,
        label_text: str,
        name: str,
        default: float,
        minimum: float,
        maximum: float,
        step: float = 0.1,
        decimals: int = 6,
    ) -> QWidget:
        """
        Создаёт горизонтальную строку:
            [Label] [QDoubleSpinBox]
        и сохраняет spinbox в self.<name>_spin.
        """
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)

        label = QLabel(label_text)
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(default)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setMaximumWidth(160)

        layout.addWidget(label, stretch=2)
        layout.addWidget(spin, stretch=1)

        setattr(self, f"{name}_spin", spin)
        return row

    # ----------------------------------------------------------------------
    # Правая часть: вкладки и кнопки управления
    # ----------------------------------------------------------------------

    def _create_right_panel(self):
        self.right_panel = QWidget()
        layout = QVBoxLayout(self.right_panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Вкладки с графиками
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, stretch=1)

        self._create_torsional_tab()
        self._create_longitudinal_tab()
        self._create_transverse_tab()
        self._create_stability_tab()

        # Кнопки под вкладками
        buttons_row = QHBoxLayout()
        buttons_row.setSpacing(8)

        self.analyze_btn = QPushButton("Выполнить анализ")
        self.export_btn = QPushButton("Экспорт результатов")
        self.interactive_btn = QPushButton("Интерактивный режим")

        buttons_row.addWidget(self.analyze_btn)
        buttons_row.addWidget(self.export_btn)
        buttons_row.addWidget(self.interactive_btn)

        layout.addLayout(buttons_row)

        # Привязка сигналов
        self.analyze_btn.clicked.connect(self._run_analysis)
        self.export_btn.clicked.connect(self._export_results)
        self.interactive_btn.clicked.connect(self._show_interactive_dialog)

    def _create_torsional_tab(self):
        self.torsional_tab = QWidget()
        layout = QVBoxLayout(self.torsional_tab)

        self.torsional_figure = Figure()
        self.torsional_canvas = FigureCanvas(self.torsional_figure)

        layout.addWidget(self.torsional_canvas)
        self.tabs.addTab(self.torsional_tab, "Крутильные")

    def _create_longitudinal_tab(self):
        self.longitudinal_tab = QWidget()
        layout = QVBoxLayout(self.longitudinal_tab)

        self.longitudinal_figure = Figure()
        self.longitudinal_canvas = FigureCanvas(self.longitudinal_figure)

        layout.addWidget(self.longitudinal_canvas)
        self.tabs.addTab(self.longitudinal_tab, "Продольные")

    def _create_transverse_tab(self):
        self.transverse_tab = QWidget()
        layout = QVBoxLayout(self.transverse_tab)

        self.transverse_figure = Figure()
        self.transverse_canvas = FigureCanvas(self.transverse_figure)

        layout.addWidget(self.transverse_canvas)
        self.tabs.addTab(self.transverse_tab, "Поперечные")

    def _create_stability_tab(self):
        self.stability_tab = QWidget()
        layout = QVBoxLayout(self.stability_tab)

        self.stability_figure = Figure()
        self.stability_canvas = FigureCanvas(self.stability_figure)

        layout.addWidget(self.stability_canvas)

        self.plot_stability_btn = QPushButton("Построить диаграмму устойчивости")
        self.plot_stability_btn.clicked.connect(self._plot_stability_diagram)

        layout.addWidget(self.plot_stability_btn)
        self.tabs.addTab(self.stability_tab, "Диаграмма устойчивости")

    # ----------------------------------------------------------------------
    # Работа с параметрами
    # ----------------------------------------------------------------------

    def _init_parameters(self):
        """Установка значений по умолчанию для всех спинбоксов и комбобоксов."""
        self.rho_spin.setValue(7800)
        self.G_spin.setValue(8e10)
        self.E_spin.setValue(200e9)

        self.S_spin.setValue(2e-4)
        self.Jr_spin.setValue(2.57e-2)
        self.Jp_spin.setValue(1.9e-5)

        self.R_spin.setValue(0.04)
        self.r_spin.setValue(0.035)
        self.K_cut_spin.setValue(6e5)
        self.beta_spin.setValue(0.3)
        self.h_spin.setValue(0.0)

        self.delta1_spin.setValue(3.44e-6)
        self.mu_spin.setValue(0.1)
        self.tau_spin.setValue(60e-3)

        self.length_combo.setCurrentIndex(0)
        self.multiplier_combo.setCurrentIndex(0)

        self.intersection_label.setText("Пересечение не вычислено.")

    def _get_current_parameters(self) -> dict:
        """Собрать все параметры в один словарь для передачи в модель."""
        return {
            "rho": self.rho_spin.value(),
            "G": self.G_spin.value(),
            "E": self.E_spin.value(),
            "S": self.S_spin.value(),
            "Jr": self.Jr_spin.value(),
            "Jp": self.Jp_spin.value(),
            "delta1": self.delta1_spin.value(),
            "mu": self.mu_spin.value(),
            "tau": self.tau_spin.value(),
            "length": float(self.length_combo.currentText()),
            "multiplier": int(self.multiplier_combo.currentText()),
            "R": self.R_spin.value(),
            "r": self.r_spin.value(),
            "K_cut": self.K_cut_spin.value(),
            "h": self.h_spin.value(),
            "beta": self.beta_spin.value(),
        }

    def _save_parameters(self):
        """Сохранить текущие параметры в JSON-файл."""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Сохранить параметры", "", "JSON Files (*.json)"
        )
        if not filename:
            return

        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(self._get_current_parameters(), f, indent=4, ensure_ascii=False)
            self.status_bar.showMessage("Параметры успешно сохранены", 3000)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить параметры:\n{e}")

    def _load_parameters(self):
        """Загрузить параметры из JSON-файла."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Загрузить параметры", "", "JSON Files (*.json)"
        )
        if not filename:
            return

        try:
            with open(filename, "r", encoding="utf-8") as f:
                params = json.load(f)

            # Подставляем значения, если они есть, иначе остаются прежние
            self.rho_spin.setValue(params.get("rho", self.rho_spin.value()))
            self.G_spin.setValue(params.get("G", self.G_spin.value()))
            self.E_spin.setValue(params.get("E", self.E_spin.value()))

            self.S_spin.setValue(params.get("S", self.S_spin.value()))
            self.Jr_spin.setValue(params.get("Jr", self.Jr_spin.value()))
            self.Jp_spin.setValue(params.get("Jp", self.Jp_spin.value()))

            self.R_spin.setValue(params.get("R", self.R_spin.value()))
            self.r_spin.setValue(params.get("r", self.r_spin.value()))
            self.K_cut_spin.setValue(params.get("K_cut", self.K_cut_spin.value()))
            self.beta_spin.setValue(params.get("beta", self.beta_spin.value()))
            self.h_spin.setValue(params.get("h", self.h_spin.value()))

            self.delta1_spin.setValue(params.get("delta1", self.delta1_spin.value()))
            self.mu_spin.setValue(params.get("mu", self.mu_spin.value()))
            self.tau_spin.setValue(params.get("tau", self.tau_spin.value()))

            # Длина и множитель — строки
            length_str = str(params.get("length", float(self.length_combo.currentText())))
            idx_len = self.length_combo.findText(length_str)
            if idx_len >= 0:
                self.length_combo.setCurrentIndex(idx_len)

            mult_str = str(params.get("multiplier", int(self.multiplier_combo.currentText())))
            idx_mult = self.multiplier_combo.findText(mult_str)
            if idx_mult >= 0:
                self.multiplier_combo.setCurrentIndex(idx_mult)

            self.status_bar.showMessage("Параметры успешно загружены", 3000)

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить параметры:\n{e}")

    # ----------------------------------------------------------------------
    # Запуск анализа по текущей вкладке
    # ----------------------------------------------------------------------

    def _run_analysis(self):
        """Определить активную вкладку и выполнить соответствующий анализ."""
        params = self._get_current_parameters()
        idx = self.tabs.currentIndex()

        try:
            if idx == 0:
                self._analyze_torsional(params)
            elif idx == 1:
                self._analyze_longitudinal(params)
            elif idx == 2:
                self._analyze_transverse(params)
            elif idx == 3:
                self._plot_stability_diagram()

            self.status_bar.showMessage("Анализ успешно выполнен", 3000)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка анализа", f"Ошибка при выполнении анализа:\n{e}")
            self.status_bar.showMessage("Ошибка при выполнении анализа", 3000)

    # ----------------------------------------------------------------------
    # Крутильные колебания
    # ----------------------------------------------------------------------

    def _analyze_torsional(self, params: dict):
        """Построение кривой D-разбиения σ(p) для крутильных колебаний."""
        self.torsional_figure.clear()
        ax = self.torsional_figure.add_subplot(111)

        result = self.model.calculate_torsional(params)

        if len(result["sigma_real"]) == 0:
            ax.text(
                0.5,
                0.5,
                "Нет данных для построения.\nПроверьте параметры.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        else:
            ax.plot(result["sigma_real"], result["sigma_imag"], "b-", linewidth=1.5)
            ax.axhline(0, color="red", linestyle="--", linewidth=0.7)
            ax.axvline(0, color="red", linestyle="--", linewidth=0.7)

            ax.set_title("Крутильные колебания: кривая D-разбиения σ(p)")
            ax.set_xlabel("Re(σ)")
            ax.set_ylabel("Im(σ)")
            ax.grid(True, linestyle=":", alpha=0.7)

            # Точка пересечения Im σ = 0
            intersection = self.model.find_intersection(params)
            if intersection is not None:
                ax.plot(
                    intersection["re_sigma"],
                    0,
                    "ro",
                    markersize=8,
                    label="Пересечение Im(σ)=0",
                )
                ax.legend()

                self.intersection_label.setText(
                    f"ω* = {intersection['omega']:.2f} рад/с\n"
                    f"Re(σ(ω*)) = {intersection['re_sigma']:.2f}\n"
                    f"f* = {intersection['frequency']:.2f} Гц"
                )
            else:
                self.intersection_label.setText(
                    "Пересечение с осью Im(σ) = 0 не найдено\n"
                    "(в выбранном диапазоне частот)."
                )

        self.torsional_canvas.draw()

    # ----------------------------------------------------------------------
    # Продольные колебания
    # ----------------------------------------------------------------------

    def _analyze_longitudinal(self, params: dict):
        """Построение D-разбиения K₁–δ для продольных колебаний."""
        self.longitudinal_figure.clear()
        ax = self.longitudinal_figure.add_subplot(111)

        result = self.model.calculate_longitudinal(params)

        if len(result["K1"]) == 0:
            ax.text(
                0.5,
                0.5,
                "Нет данных для построения.\nПроверьте параметры.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        else:
            # Переводим в удобные единицы
            K1_MN = result["K1"] / 1e6        # МН/м
            delta_kNs = result["delta"] / 1e3  # кН·с/м

            ax.plot(K1_MN, delta_kNs, "b-", linewidth=1.5)
            ax.axhline(0, color="black", linestyle="--", linewidth=0.7)

            ax.set_title("Продольные колебания: кривая D-разбиения K₁–δ")
            ax.set_xlabel("K₁, МН/м")
            ax.set_ylabel("δ, кН·с/м")
            ax.grid(True, linestyle=":", alpha=0.7)

        self.longitudinal_canvas.draw()

    # ----------------------------------------------------------------------
    # Поперечные колебания
    # ----------------------------------------------------------------------

    def _analyze_transverse(self, params: dict):
        """Построение годографа W(p) для поперечных колебаний."""
        self.transverse_figure.clear()
        ax = self.transverse_figure.add_subplot(111)

        result = self.model.calculate_transverse(params)

        if len(result["W_real"]) == 0:
            ax.text(
                0.5,
                0.5,
                "Нет данных для построения.\nПроверьте параметры.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        else:
            ax.plot(result["W_real"], result["W_imag"], "b-", linewidth=1.5)
            ax.axhline(0, color="red", linestyle="--", linewidth=0.7)
            ax.axvline(0, color="red", linestyle="--", linewidth=0.7)

            ax.set_title("Поперечные колебания: годограф W(p)")
            ax.set_xlabel("Re(W)")
            ax.set_ylabel("Im(W)")
            ax.grid(True, linestyle=":", alpha=0.7)

        self.transverse_canvas.draw()

    # ----------------------------------------------------------------------
    # Диаграмма устойчивости по δ₁
    # ----------------------------------------------------------------------

    def _plot_stability_diagram(self):
        """
        Построение диаграммы устойчивости:
        зависимость Re σ(ω*) от δ₁ с разными множителями.
        """
        self.stability_figure.clear()
        ax = self.stability_figure.add_subplot(111)

        base_params = self._get_current_parameters()
        base_delta1 = base_params["delta1"]
        multipliers = [1, 2, 3, 4, 6, 10]

        delta_values = []
        re_sigma_values = []

        for m in multipliers:
            params = dict(base_params)
            params["multiplier"] = m

            intersection = self.model.find_intersection(params)
            delta_values.append(base_delta1 * m * 1e6)  # Перевод δ₁ в ×10⁻⁶ с

            if intersection is not None:
                re_sigma_values.append(intersection["re_sigma"])
            else:
                re_sigma_values.append(np.nan)

        ax.plot(delta_values, re_sigma_values, "o-", linewidth=1.5)
        ax.set_xlabel("δ₁ (×10⁻⁶ с)")
        ax.set_ylabel("Re(σ(ω*))")
        ax.set_title(
            f"Диаграмма устойчивости по δ₁\n"
            f"L = {base_params['length']} м"
        )
        ax.grid(True, linestyle=":", alpha=0.7)

        self.stability_canvas.draw()

    # ----------------------------------------------------------------------
    # Экспорт результатов
    # ----------------------------------------------------------------------

    def _export_results(self):
        """
        Экспорт текущих результатов анализа в JSON/CSV.
        Для простоты экспортируем:
            - параметры;
            - результаты крутильных и продольных расчётов.
        """
        formats = ["JSON (*.json)", "CSV (*.csv)"]
        filename, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Экспорт результатов",
            "",
            ";;".join(formats),
        )
        if not filename:
            return

        file_format = "json" if selected_filter.startswith("JSON") else "csv"
        params = self._get_current_parameters()

        # Считаем результаты для экспорта
        torsional = self.model.calculate_torsional(params)
        longitudinal = self.model.calculate_longitudinal(params)

        data = {
            "parameters": params,
            "torsional": {
                "omega": torsional["omega"].tolist(),
                "sigma_real": torsional["sigma_real"].tolist(),
                "sigma_imag": torsional["sigma_imag"].tolist(),
            },
            "longitudinal": {
                "omega": longitudinal["omega"].tolist(),
                "K1": longitudinal["K1"].tolist(),
                "delta": longitudinal["delta"].tolist(),
            },
        }

        try:
            if file_format == "json":
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
            else:
                with open(filename, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    # Параметры
                    writer.writerow(["# Параметры"])
                    for key, value in params.items():
                        writer.writerow([key, value])

                    writer.writerow([])
                    writer.writerow(["# Крутильные колебания"])
                    writer.writerow(["omega", "Re(sigma)", "Im(sigma)"])
                    for w, sr, si in zip(
                        torsional["omega"],
                        torsional["sigma_real"],
                        torsional["sigma_imag"],
                    ):
                        writer.writerow([w, sr, si])

                    writer.writerow([])
                    writer.writerow(["# Продольные колебания"])
                    writer.writerow(["omega", "K1", "delta"])
                    for w, k1, dlt in zip(
                        longitudinal["omega"],
                        longitudinal["K1"],
                        longitudinal["delta"],
                    ):
                        writer.writerow([w, k1, dlt])

            self.status_bar.showMessage(f"Результаты экспортированы в {filename}", 5000)
            QMessageBox.information(self, "Экспорт завершён", "Результаты успешно сохранены.")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка экспорта", f"Не удалось экспортировать результаты:\n{e}")
            self.status_bar.showMessage("Ошибка при экспорте результатов", 3000)

    # ----------------------------------------------------------------------
    # Интерактивный режим
    # ----------------------------------------------------------------------

    def _show_interactive_dialog(self):
        """
        Небольшое диалоговое окно с вкладками и слайдерами
        для интерактивного изменения ключевых параметров
        и мгновенного обновления графиков.
        """
        try:
            dialog = QDialog(self)
            dialog.setWindowTitle("Интерактивный режим")
            dialog.resize(1200, 800)
            layout = QVBoxLayout(dialog)

            tabs = QTabWidget()
            layout.addWidget(tabs)

            # --- Крутильные колебания ---
            tors_tab = QWidget()
            tors_layout = QVBoxLayout(tors_tab)

            tors_fig = Figure()
            tors_canvas = FigureCanvas(tors_fig)
            tors_toolbar = NavigationToolbar(tors_canvas, tors_tab)

            tors_ax = tors_fig.add_subplot(111)
            tors_layout.addWidget(tors_toolbar)
            tors_layout.addWidget(tors_canvas)

            # Слайдер длины и δ₁
            tors_controls = QHBoxLayout()
            tors_controls.setSpacing(10)

            length_slider = QSlider(Qt.Horizontal)
            length_slider.setRange(25, 60)  # 2.5 .. 6.0 м (делим на 10)
            length_slider.setValue(int(self.length_combo.currentText().replace(".", "")))

            delta_slider = QSlider(Qt.Horizontal)
            delta_slider.setRange(1, 100)  # 1e-6 .. 1e-4
            delta_slider.setValue(int(self.delta1_spin.value() * 1e6))

            tors_controls.addWidget(QLabel("L (м) × 0.1:"))
            tors_controls.addWidget(length_slider)
            tors_controls.addWidget(QLabel("δ₁ (×10⁻⁶ с):"))
            tors_controls.addWidget(delta_slider)

            tors_layout.addLayout(tors_controls)

            # Обновление графика
            def update_torsional():
                params = self._get_current_parameters()
                params["length"] = length_slider.value() / 10.0
                params["delta1"] = delta_slider.value() * 1e-6

                tors_fig.clear()
                ax = tors_fig.add_subplot(111)
                result = self.model.calculate_torsional(params)

                if len(result["sigma_real"]) > 0:
                    ax.plot(result["sigma_real"], result["sigma_imag"], "b-", linewidth=1.5)
                    ax.axhline(0, color="red", linestyle="--", linewidth=0.7)
                    ax.axvline(0, color="red", linestyle="--", linewidth=0.7)
                    ax.set_title(
                        f"Крутильные колебания\nL={params['length']:.2f} м, δ₁={params['delta1']:.1e} с"
                    )
                    ax.set_xlabel("Re(σ)")
                    ax.set_ylabel("Im(σ)")
                    ax.grid(True, linestyle=":", alpha=0.7)
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "Нет данных для построения.\nПроверьте параметры.",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )

                tors_canvas.draw_idle()

            length_slider.valueChanged.connect(update_torsional)
            delta_slider.valueChanged.connect(update_torsional)

            update_torsional()
            tabs.addTab(tors_tab, "Крутильные")

            # --- Продольные колебания ---
            long_tab = QWidget()
            long_layout = QVBoxLayout(long_tab)

            long_fig = Figure()
            long_canvas = FigureCanvas(long_fig)
            long_toolbar = NavigationToolbar(long_canvas, long_tab)

            long_ax = long_fig.add_subplot(111)
            long_layout.addWidget(long_toolbar)
            long_layout.addWidget(long_canvas)

            long_controls = QHBoxLayout()
            long_controls.setSpacing(10)

            mu_slider = QSlider(Qt.Horizontal)
            mu_slider.setRange(1, 50)  # 0.01..0.50
            mu_slider.setValue(int(self.mu_spin.value() * 100))

            tau_slider = QSlider(Qt.Horizontal)
            tau_slider.setRange(10, 200)  # 0.01..0.20 c
            tau_slider.setValue(int(self.tau_spin.value() * 1000))

            long_controls.addWidget(QLabel("μ (×0.01):"))
            long_controls.addWidget(mu_slider)
            long_controls.addWidget(QLabel("τ (мс):"))
            long_controls.addWidget(tau_slider)

            long_layout.addLayout(long_controls)

            def update_longitudinal():
                params = self._get_current_parameters()
                params["mu"] = mu_slider.value() / 100.0
                params["tau"] = tau_slider.value() / 1000.0

                long_fig.clear()
                ax = long_fig.add_subplot(111)
                result = self.model.calculate_longitudinal(params)

                if len(result["K1"]) > 0:
                    K1_MN = result["K1"] / 1e6
                    delta_kNs = result["delta"] / 1e3

                    ax.plot(K1_MN, delta_kNs, "b-", linewidth=1.5)
                    ax.axhline(0, color="black", linestyle="--", linewidth=0.7)

                    ax.set_title(
                        f"Продольные колебания\nμ={params['mu']:.2f}, τ={params['tau']*1e3:.0f} мс"
                    )
                    ax.set_xlabel("K₁, МН/м")
                    ax.set_ylabel("δ, кН·с/м")
                    ax.grid(True, linestyle=":", alpha=0.7)
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "Нет данных для построения.\nПроверьте параметры.",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )

                long_canvas.draw_idle()

            mu_slider.valueChanged.connect(update_longitudinal)
            tau_slider.valueChanged.connect(update_longitudinal)

            update_longitudinal()
            tabs.addTab(long_tab, "Продольные")

            # --- Поперечные колебания ---
            trans_tab = QWidget()
            trans_layout = QVBoxLayout(trans_tab)

            trans_fig = Figure()
            trans_canvas = FigureCanvas(trans_fig)
            trans_toolbar = NavigationToolbar(trans_canvas, trans_tab)

            trans_ax = trans_fig.add_subplot(111)
            trans_layout.addWidget(trans_toolbar)
            trans_layout.addWidget(trans_canvas)

            trans_controls = QHBoxLayout()
            trans_controls.setSpacing(10)

            K_slider = QSlider(Qt.Horizontal)
            K_slider.setRange(1, 10)  # 1e5..1e6
            K_slider.setValue(int(self.K_cut_spin.value() / 1e5))

            mu2_slider = QSlider(Qt.Horizontal)
            mu2_slider.setRange(1, 50)  # 0.01..0.50
            mu2_slider.setValue(int(self.mu_spin.value() * 100))

            tau2_slider = QSlider(Qt.Horizontal)
            tau2_slider.setRange(10, 200)  # 0.01..0.20 c
            tau2_slider.setValue(int(self.tau_spin.value() * 1000))

            trans_controls.addWidget(QLabel("K (×10⁵ Н/м):"))
            trans_controls.addWidget(K_slider)
            trans_controls.addWidget(QLabel("μ (×0.01):"))
            trans_controls.addWidget(mu2_slider)
            trans_controls.addWidget(QLabel("τ (мс):"))
            trans_controls.addWidget(tau2_slider)

            trans_layout.addLayout(trans_controls)

            def update_transverse():
                params = self._get_current_parameters()
                params["K_cut"] = K_slider.value() * 1e5
                params["mu"] = mu2_slider.value() / 100.0
                params["tau"] = tau2_slider.value() / 1000.0

                trans_fig.clear()
                ax = trans_fig.add_subplot(111)
                result = self.model.calculate_transverse(params)

                if len(result["W_real"]) > 0:
                    ax.plot(result["W_real"], result["W_imag"], "b-", linewidth=1.5)
                    ax.axhline(0, color="red", linestyle="--", linewidth=0.7)
                    ax.axvline(0, color="red", linestyle="--", linewidth=0.7)

                    ax.set_title(
                        "Поперечные колебания\n"
                        f"K={params['K_cut']/1e5:.1f}·10⁵ Н/м, "
                        f"μ={params['mu']:.2f}, τ={params['tau']*1e3:.0f} мс"
                    )
                    ax.set_xlabel("Re(W)")
                    ax.set_ylabel("Im(W)")
                    ax.grid(True, linestyle=":", alpha=0.7)
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "Нет данных для построения.\nПроверьте параметры.",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )

                trans_canvas.draw_idle()

            K_slider.valueChanged.connect(update_transverse)
            mu2_slider.valueChanged.connect(update_transverse)
            tau2_slider.valueChanged.connect(update_transverse)

            update_transverse()
            tabs.addTab(trans_tab, "Поперечные")

            # Кнопка закрытия
            close_btn = QPushButton("Закрыть")
            close_btn.clicked.connect(dialog.accept)
            layout.addWidget(close_btn, alignment=Qt.AlignRight)

            dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось запустить интерактивный режим:\n{e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BoreBarGUI()
    window.show()
    sys.exit(app.exec_())
