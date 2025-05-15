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
        """Анализ продольных колебаний с построением D-разбиения"""
        self.longitudinal_figure.clear()
        ax = self.longitudinal_figure.add_subplot(111)
        
        try:
            # 1. Получение и проверка параметров
            E = params.get('E', 200e9)       # Модуль Юнга (Па)
            S = params.get('S', 2e-4)        # Площадь сечения (м²) - уточнено по исследованию
            rho = params.get('rho', 7800)    # Плотность (кг/м³)
            mu = max(0.01, min(params.get('mu', 0.1), 0.99))  # Коэф. трения 0 < μ < 1
            tau = params.get('tau', 60e-3)   # Время запаздывания (с)
            L = params.get('length', 4.0)    # Длина (м)
            
            # Проверка особого случая mu ≈ 1
            if abs(1 - mu) < 1e-10:
                raise ValueError("Коэффициент трения mu слишком близок к 1, решение не определено")
            
            # 2. Расчетные параметры
            a = np.sqrt(E/rho)  # Скорость волны (м/с)
            
            # 3. Диапазон частот (рад/с) - как в MATLAB-коде исследования
            omega = np.linspace(0.001, 0.4, 5000)  
            
            # 4. Расчет D-кривой с защитой от особых точек
            with np.errstate(all='ignore'):
                x = omega * L / a
                
                # Безопасный расчет котангенса
                cot = np.zeros_like(x)
                sin_x = np.sin(x)
                cos_x = np.cos(x)
                cot_mask = np.abs(sin_x) > 1e-10
                cot[cot_mask] = cos_x[cot_mask] / sin_x[cot_mask]
                
                # Расчет знаменателя с защитой
                denom = 1 - mu * np.cos(omega * tau)
                denom_mask = np.abs(denom) > 1e-10
                
                # Основные формулы из исследования
                K1 = (E*S/a) * omega * cot / denom
                delta = -(E*S*mu/a) * cot * np.sin(omega*tau) / denom
                
                # Комбинированная маска валидных значений
                valid = cot_mask & denom_mask & ~np.isnan(K1) & ~np.isnan(delta)
                K1 = np.where(valid, K1, np.nan)
                delta = np.where(valid, delta, np.nan)
            
            # 5. Расчет особой точки (ω→0)
            K1_0 = -E*S/(L*(1 - mu)) if abs(1 - mu) > 1e-10 else np.nan
            delta_0 = tau*E*S*mu/(L*(1 - mu)) if abs(1 - mu) > 1e-10 else np.nan
            
            # 6. Построение графика
            if np.any(valid):
                # Основная кривая D-разбиения
                line, = ax.plot(K1[valid]/1e6, delta[valid]/1e6, 'b-', 
                            linewidth=1.5, label='Кривая D-разбиения')
                
                # Область устойчивости (δ > 0 и K₁ < 0)
                stable_mask = (delta[valid] > 0) & (K1[valid] < 0)
                if np.any(stable_mask):
                    ax.fill_between(K1[valid][stable_mask]/1e6, 
                                delta[valid][stable_mask]/1e6, 0,
                                color='green', alpha=0.2, 
                                label='Область устойчивости')
                
                # Особые точки
                if not np.isnan(K1_0) and not np.isnan(delta_0):
                    ax.plot(K1_0/1e6, delta_0/1e6, 'ro', markersize=8,
                        label='Особая точка (ω→0)')
                    ax.annotate(f'({K1_0/1e6:.1f}, {delta_0/1e6:.1f})',
                            xy=(K1_0/1e6, delta_0/1e6), 
                            xytext=(10, 10),
                            textcoords='offset points',
                            bbox=dict(boxstyle='round', 
                                    fc='yellow', alpha=0.7))
            else:
                ax.text(0.5, 0.5, "Недостаточно данных для построения", 
                    ha='center', va='center', transform=ax.transAxes)
            
            # 7. Настройка графика
            ax.set_xlabel('Динамическая жесткость K₁, МН/м', fontsize=10)
            ax.set_ylabel('Коэффициент демпфирования δ, МН·с/м', fontsize=10)
            ax.set_title('D-разбиение для продольных колебаний', fontsize=12)
            ax.grid(True, linestyle=':', alpha=0.5)
            ax.legend(loc='best', fontsize=9)
            
            # Автомасштабирование с защитой
            if np.any(valid):
                x_min, x_max = np.nanmin(K1[valid]/1e6), np.nanmax(K1[valid]/1e6)
                y_min, y_max = np.nanmin(delta[valid]/1e6), np.nanmax(delta[valid]/1e6)
                
                ax.set_xlim(x_min - 0.1*abs(x_min), x_max + 0.1*abs(x_max))
                ax.set_ylim(y_min - 0.1*abs(y_min), y_max + 0.1*abs(y_max))

            self.longitudinal_figure.tight_layout()
            self.longitudinal_canvas.draw()

        except Exception as e:
            logging.error(f"Ошибка в analyze_longitudinal: {str(e)}", exc_info=True)
            ax.clear()
            ax.text(0.5, 0.5, f"Ошибка: {str(e)}", 
                ha='center', va='center', transform=ax.transAxes,
                bbox=dict(facecolor='red', alpha=0.2))
            self.longitudinal_canvas.draw()

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