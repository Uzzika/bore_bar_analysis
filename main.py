import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, 
                            QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
                            QDoubleSpinBox, QComboBox, QPushButton, 
                            QStatusBar, QFileDialog, QMessageBox, QDialog, QSlider)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import Cursor
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
import json
import csv

import logging
logging.basicConfig(filename='borebar_analysis.log', 
                   level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

class BoreBarGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Анализ колебаний борштанги")
        self.setGeometry(100, 100, 1800, 800)
        
        # Центральный виджет и основной layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        
        # Панель параметров слева
        self.params_panel = QGroupBox("Параметры системы")
        self.params_layout = QVBoxLayout()
        self.setup_parameters_ui()
        self.params_panel.setLayout(self.params_layout)
        self.main_layout.addWidget(self.params_panel, stretch=1)
        
        # Правая часть с вкладками и графиками
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)
        
        # Вкладки для разных типов анализа
        self.tabs = QTabWidget()
        self.setup_analysis_tabs()
        self.right_layout.addWidget(self.tabs, stretch=5)
        
        # Кнопки управления внизу
        self.control_buttons = self.setup_control_buttons()
        self.right_layout.addLayout(self.control_buttons, stretch=1)
        
        self.main_layout.addWidget(self.right_panel, stretch=4)
        
        # Статус бар
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Инициализация параметров
        self.init_parameters()
        
    def setup_parameters_ui(self):
        """Настройка UI для параметров системы"""
        # Материальные свойства
        material_group = QGroupBox("Материальные свойства")
        material_layout = QVBoxLayout()
        
        self.rho_input = self.create_parameter_input("Плотность материала (кг/м³)", 7800, 1000, 20000)
        self.G_input = self.create_parameter_input("Модуль сдвига (Па)", 8e10, 1e9, 1e12)
        self.E_input = self.create_parameter_input("Модуль Юнга (Па)", 200e9, 1e9, 1e12)
        
        material_layout.addWidget(self.rho_input)
        material_layout.addWidget(self.G_input)
        material_layout.addWidget(self.E_input)
        material_group.setLayout(material_layout)
        self.params_layout.addWidget(material_group)
        
        # Геометрические параметры
        geometry_group = QGroupBox("Геометрические параметры")
        geometry_layout = QVBoxLayout()
        
        self.length_combo = QComboBox()
        self.length_combo.addItems(["2.5", "3", "4", "5", "6"])
        geometry_layout.addWidget(QLabel("Длина борштанги (м):"))
        geometry_layout.addWidget(self.length_combo)
        
        self.S_input = self.create_parameter_input("Площадь сечения (м²)", 2e-4, 1e-6, 1e-2)
        self.Jr_input = self.create_parameter_input("Момент инерции стержня (кг·м²)", 2.57e-2, 1e-5, 1)
        self.Jp_input = self.create_parameter_input("Полярный момент инерции (м⁴)", 1.9e-5, 1e-8, 1e-2)
        
        geometry_layout.addWidget(self.S_input)
        geometry_layout.addWidget(self.Jr_input)
        geometry_layout.addWidget(self.Jp_input)
        geometry_group.setLayout(geometry_layout)
        self.params_layout.addWidget(geometry_group)
        
        # Параметры трения
        friction_group = QGroupBox("Параметры трения")
        friction_layout = QVBoxLayout()
        
        self.delta1_input = self.create_parameter_input("Коэф. трения δ₁ (с)", 3.44e-6, 1e-8, 1e-4)
        self.multiplier_combo = QComboBox()
        self.multiplier_combo.addItems(["1", "2", "3", "4", "6", "10"])
        friction_layout.addWidget(QLabel("Множитель для δ₁:"))
        friction_layout.addWidget(self.multiplier_combo)
        
        self.mu_input = self.create_parameter_input("Коэф. внутр. трения μ", 0.1, 0, 1)
        self.tau_input = self.create_parameter_input("Время запаздывания τ (с)", 60e-3, 1e-3, 1)
        
        friction_layout.addWidget(self.delta1_input)
        friction_layout.addWidget(self.mu_input)
        friction_layout.addWidget(self.tau_input)
        friction_group.setLayout(friction_layout)
        self.params_layout.addWidget(friction_group)
        
        # Кнопки управления параметрами
        self.params_buttons = QHBoxLayout()
        self.save_params_btn = QPushButton("Сохранить параметры")
        self.load_params_btn = QPushButton("Загрузить параметры")
        self.reset_params_btn = QPushButton("Сбросить к умолчаниям")
        
        self.params_buttons.addWidget(self.save_params_btn)
        self.params_buttons.addWidget(self.load_params_btn)
        self.params_buttons.addWidget(self.reset_params_btn)
        
        self.params_layout.addLayout(self.params_buttons)
        
        # Подключение сигналов
        self.save_params_btn.clicked.connect(self.save_parameters)
        self.load_params_btn.clicked.connect(self.load_parameters)
        self.reset_params_btn.clicked.connect(self.init_parameters)
        
    def setup_analysis_tabs(self):
        """Настройка вкладок для разных типов анализа"""
        # Вкладка крутильных колебаний
        self.torsional_tab = QWidget()
        torsional_layout = QVBoxLayout(self.torsional_tab)
        
        self.torsional_figure = Figure()
        self.torsional_canvas = FigureCanvas(self.torsional_figure)
        torsional_layout.addWidget(self.torsional_canvas)
        
        # Вкладка продольных колебаний
        self.longitudinal_tab = QWidget()
        longitudinal_layout = QVBoxLayout(self.longitudinal_tab)
        
        self.longitudinal_figure = Figure()
        self.longitudinal_canvas = FigureCanvas(self.longitudinal_figure)
        longitudinal_layout.addWidget(self.longitudinal_canvas)
        
        # Вкладка сравнительного анализа
        self.comparative_tab = QWidget()
        comparative_layout = QVBoxLayout(self.comparative_tab)
        
        self.comparative_figure = Figure()
        self.comparative_canvas = FigureCanvas(self.comparative_figure)
        comparative_layout.addWidget(self.comparative_canvas)
        
        # Добавление вкладок
        self.tabs.addTab(self.torsional_tab, "Крутильные колебания")
        self.tabs.addTab(self.longitudinal_tab, "Продольные колебания")
        self.tabs.addTab(self.comparative_tab, "Сравнительный анализ")

    def show_interactive(self):
        """Интерактивная визуализация с отдельным окном"""
        try:
            # Создаем диалоговое окно для интерактивного режима
            dialog = QDialog(self)
            dialog.setWindowTitle("Интерактивный режим")
            dialog.setModal(True)
            dialog.resize(800, 600)
            
            layout = QVBoxLayout(dialog)
            
            # Создаем вкладки для разных типов интерактивного анализа
            tabs = QTabWidget()
            
            # 1. Вкладка крутильных колебаний
            torsional_tab = QWidget()
            torsional_layout = QVBoxLayout(torsional_tab)
            
            # Добавляем слайдеры для параметров
            length_slider = QSlider(Qt.Horizontal)
            length_slider.setRange(2, 6)
            length_slider.setValue(3)
            length_slider.setSingleStep(1)
            length_slider.setTickInterval(1)
            length_slider.setTickPosition(QSlider.TicksBelow)
            
            delta_slider = QSlider(Qt.Horizontal)
            delta_slider.setRange(1, 100)
            delta_slider.setValue(34)
            
            torsional_layout.addWidget(QLabel("Длина борштанги (м):"))
            torsional_layout.addWidget(length_slider)
            torsional_layout.addWidget(QLabel("Коэффициент трения δ₁ (x1e-6):"))
            torsional_layout.addWidget(delta_slider)
            
            # 2. Вкладка продольных колебаний
            longitudinal_tab = QWidget()
            longitudinal_layout = QVBoxLayout(longitudinal_tab)
            
            # Добавляем слайдеры для параметров
            mu_slider = QSlider(Qt.Horizontal)
            mu_slider.setRange(1, 50)
            mu_slider.setValue(10)
            
            tau_slider = QSlider(Qt.Horizontal)
            tau_slider.setRange(10, 200)
            tau_slider.setValue(60)
            
            longitudinal_layout.addWidget(QLabel("Коэффициент трения μ (x0.01):"))
            longitudinal_layout.addWidget(mu_slider)
            longitudinal_layout.addWidget(QLabel("Время запаздывания τ (мс):"))
            longitudinal_layout.addWidget(tau_slider)
            
            # Добавляем вкладки
            tabs.addTab(torsional_tab, "Крутильные колебания")
            tabs.addTab(longitudinal_tab, "Продольные колебания")
            
            layout.addWidget(tabs)
            
            # Кнопка закрытия
            close_btn = QPushButton("Закрыть")
            close_btn.clicked.connect(dialog.close)
            layout.addWidget(close_btn)
            
            # Показываем диалоговое окно
            dialog.exec_()
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось запустить интерактивный режим: {str(e)}")
            
    def setup_control_buttons(self):
        """Настройка кнопок управления"""
        buttons_layout = QHBoxLayout()
        
        self.analyze_btn = QPushButton("Выполнить анализ")
        self.export_btn = QPushButton("Экспорт результатов")
        self.interactive_btn = QPushButton("Интерактивный режим")
        
        buttons_layout.addWidget(self.analyze_btn)
        buttons_layout.addWidget(self.export_btn)
        buttons_layout.addWidget(self.interactive_btn)
        
        # Подключение сигналов
        self.analyze_btn.clicked.connect(self.run_analysis)
        self.export_btn.clicked.connect(self.export_results)
        self.interactive_btn.clicked.connect(self.show_interactive)
        
        return buttons_layout
        
    def create_parameter_input(self, label, default, min_val, max_val):
        """Создание элемента ввода для параметра"""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        
        param_label = QLabel(label)
        param_input = QDoubleSpinBox()
        param_input.setRange(min_val, max_val)
        param_input.setValue(default)
        param_input.setDecimals(6)
        param_input.setSingleStep(0.1)
        
        layout.addWidget(param_label, stretch=2)
        layout.addWidget(param_input, stretch=1)
        
        return container
    
    def init_parameters(self):
        """Инициализация параметров по умолчанию"""
        self.rho_input.findChild(QDoubleSpinBox).setValue(7800)
        self.G_input.findChild(QDoubleSpinBox).setValue(8e10)
        self.E_input.findChild(QDoubleSpinBox).setValue(200e9)
        self.S_input.findChild(QDoubleSpinBox).setValue(2e-4)
        self.Jr_input.findChild(QDoubleSpinBox).setValue(2.57e-2)
        self.Jp_input.findChild(QDoubleSpinBox).setValue(1.9e-5)
        self.delta1_input.findChild(QDoubleSpinBox).setValue(3.44e-6)
        self.mu_input.findChild(QDoubleSpinBox).setValue(0.1)
        self.tau_input.findChild(QDoubleSpinBox).setValue(60e-3)
        self.length_combo.setCurrentIndex(0)
        self.multiplier_combo.setCurrentIndex(0)
        
    def get_current_parameters(self):
        """Получение текущих параметров из UI"""
        params = {
            'rho': self.rho_input.findChild(QDoubleSpinBox).value(),
            'G': self.G_input.findChild(QDoubleSpinBox).value(),
            'E': self.E_input.findChild(QDoubleSpinBox).value(),
            'S': self.S_input.findChild(QDoubleSpinBox).value(),
            'Jr': self.Jr_input.findChild(QDoubleSpinBox).value(),
            'Jp': self.Jp_input.findChild(QDoubleSpinBox).value(),
            'delta1': self.delta1_input.findChild(QDoubleSpinBox).value(),
            'mu': self.mu_input.findChild(QDoubleSpinBox).value(),
            'tau': self.tau_input.findChild(QDoubleSpinBox).value(),
            'length': float(self.length_combo.currentText()),
            'multiplier': int(self.multiplier_combo.currentText())
        }
        return params
    
    def save_parameters(self):
        """Сохранение параметров в файл"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Сохранить параметры", "", "JSON Files (*.json)"
        )
        if filename:
            params = self.get_current_parameters()
            try:
                with open(filename, 'w') as f:
                    json.dump(params, f, indent=4)
                self.status_bar.showMessage("Параметры успешно сохранены", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить параметры: {str(e)}")

    def load_parameters(self):
        """Загрузка параметров из файла"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Загрузить параметры", "", "JSON Files (*.json)"
        )
        if filename:
            try:
                with open(filename, 'r') as f:
                    params = json.load(f)
                # Установка загруженных параметров в UI
                self.rho_input.findChild(QDoubleSpinBox).setValue(params.get('rho', 7800))
                self.G_input.findChild(QDoubleSpinBox).setValue(params.get('G', 8e10))
                self.E_input.findChild(QDoubleSpinBox).setValue(params.get('E', 200e9))
                self.S_input.findChild(QDoubleSpinBox).setValue(params.get('S', 2e-4))
                self.Jr_input.findChild(QDoubleSpinBox).setValue(params.get('Jr', 2.57e-2))
                self.Jp_input.findChild(QDoubleSpinBox).setValue(params.get('Jp', 1.9e-5))
                self.delta1_input.findChild(QDoubleSpinBox).setValue(params.get('delta1', 3.44e-6))
                self.mu_input.findChild(QDoubleSpinBox).setValue(params.get('mu', 0.1))
                self.tau_input.findChild(QDoubleSpinBox).setValue(params.get('tau', 60e-3))
                self.length_combo.setCurrentIndex(params.get('length', 0))
                self.multiplier_combo.setCurrentIndex(params.get('multiplier', 0))
                self.status_bar.showMessage("Параметры успешно загружены", 3000)

            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить параметры: {str(e)}")
    
    def run_analysis(self):
        """Выполнение выбранного анализа"""
        current_tab = self.tabs.currentIndex()
        params = self.get_current_parameters()
        
        try:
            if current_tab == 0:  # Крутильные колебания
                self.analyze_torsional(params)
            elif current_tab == 1:  # Продольные колебания
                self.analyze_longitudinal(params)
            elif current_tab == 2:  # Сравнительный анализ
                self.analyze_comparative(params)
                
            self.status_bar.showMessage("Анализ успешно завершен", 3000)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка анализа", f"Ошибка при выполнении анализа: {str(e)}")
            self.status_bar.showMessage("Ошибка при выполнении анализа", 3000)

    def analyze_torsional(self, params):
        """Анализ крутильных колебаний с построением D-разбиения и диаграммы устойчивости"""
        self.torsional_figure.clear()
        
        # Создаем сетку графиков 1x2
        gs = self.torsional_figure.add_gridspec(1, 2)
        ax1 = self.torsional_figure.add_subplot(gs[0, 0])  # Кривая D-разбиения
        ax2 = self.torsional_figure.add_subplot(gs[0, 1])  # Диаграмма устойчивости
        
        # Вычисляем константы из параметров
        lambda1 = np.sqrt(params['rho'] * params['G']) * params['Jp'] / params['Jr']
        lambda2 = params['length'] * np.sqrt(params['rho'] / params['G'])
        delta1 = params['delta1'] * params['multiplier']
        
        # 1. Кривая D-разбиения
        omega = np.linspace(1000, 15000, 1000)
        p = 1j * omega
        with np.errstate(all='ignore'):
            expr = np.sqrt(1 + delta1 * p)
            sigma = -p - lambda1 * expr * (1 / np.tanh(lambda2 * p / expr))
            sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e10, neginf=-1e10)
        
        ax1.plot(sigma.real, sigma.imag, 'b-', linewidth=1.5)
        ax1.axhline(0, color='red', linestyle='--', linewidth=0.7)
        ax1.axvline(0, color='red', linestyle='--', linewidth=0.7)
        ax1.set_title('Кривая D-разбиения для крутильных колебаний', fontsize=10)
        ax1.set_xlabel('Re(σ)', fontsize=8)
        ax1.set_ylabel('Im(σ)', fontsize=8)
        ax1.grid(True, which='both', linestyle=':', alpha=0.7)
        ax1.set_xlim(-15000, 500)
        ax1.set_ylim(-8000, 8000)
        
        # 2. Диаграмма устойчивости для разных длин
        lengths = np.array([2.5, 3, 4, 5, 6])
        multipliers = np.array([1, 2, 3, 4, 6, 10])
        delta1_values = params['delta1'] * multipliers
        lambda2_values = lengths * np.sqrt(params['rho'] / params['G'])
        colors = plt.cm.viridis(np.linspace(0, 1, len(lengths)))
        
        for i, l in enumerate(lengths):
            Sigma = np.zeros(len(delta1_values))
            
            for j, delta in enumerate(delta1_values):
                def im_sigma(omega):
                    p = 1j * omega
                    with np.errstate(all='ignore'):
                        sqrt_expr = np.sqrt(1 + delta * p)
                        cth = 1 / np.tanh(lambda2_values[i] * p / sqrt_expr)
                        val = -p - lambda1 * sqrt_expr * cth
                        return val.imag
                
                try:
                    omega_sol = root_scalar(im_sigma, bracket=[500, 2000], method='brentq').root
                    p = 1j * omega_sol
                    with np.errstate(all='ignore'):
                        sqrt_expr = np.sqrt(1 + delta * p)
                        cth = 1 / np.tanh(lambda2_values[i] * p / sqrt_expr)
                        Sigma[j] = (-p - lambda1 * sqrt_expr * cth).real
                except:
                    Sigma[j] = np.nan
            
            valid = ~np.isnan(Sigma)
            ax2.plot(delta1_values[valid], Sigma[valid], 'o-', color=colors[i], 
                    label=f'L={l} м', markersize=4, linewidth=1.5)
        
        ax2.axhline(0, color='k', linestyle='--', linewidth=0.8)
        ax2.fill_between(delta1_values, -1500, 0, color='green', alpha=0.15)
        ax2.set_xscale('log')
        ax2.set_ylim(-1500, 100)
        ax2.set_xlabel('Коэффициент внутреннего трения δ₁ (с)', fontsize=8)
        ax2.set_ylabel('Re(σ)', fontsize=8)
        ax2.set_title('Диаграмма устойчивости крутильных колебаний', fontsize=10)
        ax2.legend(fontsize=8)
        ax2.grid(True, which='both', linestyle=':', alpha=0.7)
        
        self.torsional_figure.tight_layout()
        self.torsional_canvas.draw()

    def analyze_longitudinal(self, params):
        self.longitudinal_figure.clear()
        
        # Создаем сетку графиков
        gs = self.longitudinal_figure.add_gridspec(1, 2)
        ax1 = self.longitudinal_figure.add_subplot(gs[0, 0])  # D-разбиение
        ax2 = self.longitudinal_figure.add_subplot(gs[0, 1])  # Устойчивость

        # Параметры системы
        E = params['E']          # Модуль Юнга [Па]
        rho = params['rho']      # Плотность [кг/м³]
        S = params['S']          # Площадь сечения [м²]
        mu = params['mu']        # Коэффициент трения
        tau = params['tau']      # Время запаздывания [с]
        L = params['length']     # Длина [м]

        # Расчетные параметры
        a = np.sqrt(E/rho)  # Скорость волны
        
        # Оптимальный диапазон частот (подобран экспериментально)
        omega = np.linspace(0.01, 2*np.pi*100, 5000)  # Более узкий и плотный диапазон
        
        # 1. Улучшенный расчет D-разбиения
        with np.errstate(all='ignore'):
            x = omega * L / a
            # Избегаем точек сингулярности cot(x)
            mask = (np.abs(np.sin(x)) > 1e-6)
            cot = np.zeros_like(x)
            cot[mask] = 1/np.tan(x[mask])
            
            # Избегаем деления на ноль в знаменателе
            denom = 1 - mu * np.cos(omega * tau)
            denom_mask = np.abs(denom) > 1e-6
            
            # Комбинированная маска валидных точек
            valid = mask & denom_mask
            
            K1 = np.full_like(omega, np.nan)
            delta = np.full_like(omega, np.nan)
            
            K1[valid] = (E*S/a) * omega[valid] * cot[valid] / denom[valid]
            delta[valid] = -(E*S*mu/a) * cot[valid] * np.sin(omega[valid]*tau) / denom[valid]
            
            # Дополнительная фильтрация физически возможных значений
            valid = valid & (K1 > 0) & (K1 < 1e10) & (np.abs(delta) < 1e6)
            K1 = K1[valid]
            delta = delta[valid]
            omega = omega[valid]

        # Построение D-разбиения с правильными пределами
        ax1.plot(K1/1e6, delta/1e3, 'b-', linewidth=1)
        ax1.set_title('Кривая D-разбиения для продольных колебаний', fontsize=10)
        ax1.set_xlabel('K₁, МН/м', fontsize=8)
        ax1.set_ylabel('δ, кН·с/м', fontsize=8)
        ax1.grid(True, which='both', linestyle=':', alpha=0.7)
        
        # Установка правильных пределов для осей
        ax1.set_xlim(0, 20)  # Ограничение по K1
        ax1.set_ylim(-150, 50)  # Ограничение по δ
        
        # 2. Анализ устойчивости для разных длин
        lengths = np.array([2.5, 3, 4, 5, 6])
        mu_values = np.linspace(0.05, 0.5, 20)
        colors = plt.cm.viridis(np.linspace(0, 1, len(lengths)))
        
        for i, l in enumerate(lengths):
            critical_deltas = []
            for mu_val in mu_values:
                try:
                    # Основная частота с поправкой
                    omega_crit = (np.pi * a) / (2 * l) * 0.99  # Коэффициент 0.99 для избежания сингулярности
                    
                    # Безопасный расчет
                    x_crit = omega_crit * l / a
                    if np.abs(np.sin(x_crit)) < 1e-6:
                        critical_deltas.append(np.nan)
                        continue
                        
                    cot_crit = 1/np.tan(x_crit)
                    denom_crit = 1 - mu_val * np.cos(omega_crit * tau)
                    
                    if np.abs(denom_crit) < 1e-6:
                        critical_deltas.append(np.nan)
                    else:
                        delta_crit = - (E*S*mu_val/a) * cot_crit * \
                                    np.sin(omega_crit*tau) / denom_crit
                        critical_deltas.append(delta_crit)
                except:
                    critical_deltas.append(np.nan)
            
            # Отображаем только валидные значения
            valid = ~np.isnan(critical_deltas)
            ax2.plot(mu_values[valid], np.array(critical_deltas)[valid]/1e3, 
                    'o-', color=colors[i], markersize=3, label=f'L={l}м')

        ax2.set_title('Границы устойчивости', fontsize=10)
        ax2.set_xlabel('μ', fontsize=8)
        ax2.set_ylabel('Критическое δ, кН·с/м', fontsize=8)
        ax2.legend(fontsize=8)
        ax2.grid(True, which='both', linestyle=':', alpha=0.7)
        ax2.set_ylim(-10, 15)  # Ограничение по разумным значениям
        
        self.longitudinal_figure.tight_layout()
        self.longitudinal_canvas.draw()
        
        # Проверочные вычисления
        print("\n=== Проверочные вычисления ===")
        print(f"Скорость волны: {a:.2f} м/с")
        print(f"Основная частота для L={L}m: {a/(2*L):.2f} рад/с")
        print(f"Диапазон K1: {np.min(K1)/1e6:.2f} - {np.max(K1)/1e6:.2f} МН/м")
        print(f"Диапазон δ: {np.min(delta)/1e3:.2f} - {np.max(delta)/1e3:.2f} кН·с/м")


        # Проверка крайних значений
        print(f"\nКрайние точки D-разбиения:")
        print(f"Первая точка: K1={K1[0]/1e6:.2f} МН/м, δ={delta[0]/1e3:.2f} кН·с/м")
        print(f"Последняя точка: K1={K1[-1]/1e6:.2f} МН/м, δ={delta[-1]/1e3:.2f} кН·с/м")

        # Проверка устойчивости для L=2.5m
        print(f"\nКритические значения для L=2.5m:")
        print(f"При μ=0.1: δ={critical_deltas[1]/1e3:.2f} кН·с/м")

    def analyze_comparative(self, params):
        """Сравнительный анализ крутильных и продольных колебаний"""
        self.comparative_figure.clear()
        ax1 = self.comparative_figure.add_subplot(121)  # Сравнение частот
        ax2 = self.comparative_figure.add_subplot(122)  # Сравнение устойчивости
        
        # 1. Сравнение частот
        lengths = np.linspace(2, 6, 20)
        
        # Крутильные частоты
        torsional_freq = np.sqrt(params['G']/params['rho']) * np.pi / lengths
        
        # Продольные частоты
        longitudinal_freq = np.sqrt(params['E']/params['rho']) * np.pi / lengths
        
        ax1.plot(lengths, torsional_freq/1000, 'b-o', label='Крутильные', linewidth=1.5, markersize=4)
        ax1.plot(lengths, longitudinal_freq/1000, 'r-s', label='Продольные', linewidth=1.5, markersize=4)
        ax1.set_xlabel('Длина борштанги, м', fontsize=10)
        ax1.set_ylabel('Частота (кГц)', fontsize=10)
        ax1.set_title('Сравнение собственных частот', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, linestyle=':', alpha=0.7)
        
        # 2. Сравнение устойчивости
        stability_ratio = 1 / lengths**2
        ax2.plot(lengths, stability_ratio/stability_ratio.max(), 'g-^', linewidth=2, markersize=6)
        ax2.set_xlabel('Длина борштанги, м', fontsize=10)
        ax2.set_ylabel('Относительная устойчивость', fontsize=10)
        ax2.set_title('Влияние длины на устойчивость', fontsize=12)
        ax2.grid(True, linestyle=':', alpha=0.7)
        ax2.text(3.5, 0.7, 'Уменьшение длины\nувеличивает устойчивость', 
                fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        
        self.comparative_figure.tight_layout()
        self.comparative_canvas.draw()

    def export_results(self):
        """Экспорт результатов анализа в файл"""
        formats = ["JSON (*.json)", "CSV (*.csv)"]
        filename, selected_filter = QFileDialog.getSaveFileName(
            self, "Экспорт результатов", "", ";;".join(formats)
        )
        
        if not filename:
            return
        
        # Определяем формат по выбранному фильтру
        file_format = 'json' if selected_filter == "JSON (*.json)" else 'csv'
        
        try:
            # Собираем все результаты из всех анализаторов
            results = {
                'parameters': self.get_current_parameters(),
                'analysis': {
                    'torsional': self._get_torsional_results(),
                    'longitudinal': self._get_longitudinal_results(),
                    'comparative': self._get_comparative_results()
                }
            }
            
            if file_format == 'json':
                with open(filename, 'w') as f:
                    json.dump(self._convert_to_serializable(results), f, indent=4)
            else:  # CSV
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    # Записываем параметры
                    for key, value in results['parameters'].items():
                        writer.writerow([f"param_{key}", value])
                    # Записываем результаты анализов
                    for analysis_type, analysis_data in results['analysis'].items():
                        if analysis_data:
                            for result_name, values in analysis_data.items():
                                if isinstance(values, (list, np.ndarray)):
                                    writer.writerow([f"{analysis_type}_{result_name}"] + list(values))
                                elif isinstance(values, dict):
                                    for k, v in values.items():
                                        writer.writerow([f"{analysis_type}_{result_name}_{k}", v])
                                else:
                                    writer.writerow([f"{analysis_type}_{result_name}", values])
            
            self.status_bar.showMessage(f"Результаты успешно экспортированы в {filename}", 5000)
            QMessageBox.information(self, "Экспорт завершен", "Результаты успешно сохранены в файл.")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка экспорта", f"Не удалось экспортировать результаты: {str(e)}")
            self.status_bar.showMessage("Ошибка при экспорте результатов", 3000)

    def _get_torsional_results(self):
        """Получение результатов крутильного анализа"""
        params = self.get_current_parameters()
        lambda1 = np.sqrt(params['rho'] * params['G']) * params['Jp'] / params['Jr']
        lambda2 = params['length'] * np.sqrt(params['rho'] / params['G'])
        delta1 = params['delta1'] * params['multiplier']
        
        omega = np.linspace(1000, 15000, 1000)
        p = 1j * omega
        with np.errstate(all='ignore'):
            expr = np.sqrt(1 + delta1 * p)
            sigma = -p - lambda1 * expr * (1 / np.tanh(lambda2 * p / expr))
            sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e10, neginf=-1e10)
        
        return {
            'lambda1': lambda1,
            'lambda2': lambda2,
            'delta1': delta1,
            'sigma_real': sigma.real,
            'sigma_imag': sigma.imag
        }

    def _convert_to_serializable(self, data):
        """Рекурсивное преобразование данных в сериализуемый формат"""
        if isinstance(data, (np.ndarray)):
            return data.tolist()
        elif isinstance(data, dict):
            return {k: self._convert_to_serializable(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._convert_to_serializable(item) for item in data]
        elif isinstance(data, (int, float, str, bool)) or data is None:
            return data
        else:
            return str(data)
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BoreBarGUI()
    window.show()
    sys.exit(app.exec_())