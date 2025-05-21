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
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import SpanSelector

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
        """Интерактивная визуализация с улучшенным масштабированием"""
        try:
            dialog = QDialog(self)
            dialog.setWindowTitle("Интерактивный режим с масштабированием")
            dialog.setModal(True)
            dialog.resize(1200, 900)
            
            layout = QVBoxLayout(dialog)
            
            tabs = QTabWidget()
            
            # 1. Вкладка крутильных колебаний
            torsional_tab = QWidget()
            torsional_layout = QVBoxLayout(torsional_tab)
            
            # Создаем фигуру и canvas для крутильных колебаний
            torsional_fig = Figure(figsize=(10, 6))
            torsional_canvas = FigureCanvas(torsional_fig)
            
            # Теперь используем только один график (кривая D-разбиения)
            torsional_ax = torsional_fig.add_subplot(111)
            
            # Добавляем слайдеры
            length_slider = QSlider(Qt.Horizontal)
            length_slider.setRange(2, 6)
            length_slider.setValue(3)
            length_slider.setSingleStep(1)
            length_slider.setTickInterval(1)
            length_slider.setTickPosition(QSlider.TicksBelow)
            
            delta_slider = QSlider(Qt.Horizontal)
            delta_slider.setRange(1, 100)
            delta_slider.setValue(34)
            
            # Инициализация графика
            line, = torsional_ax.plot([], [], 'b-', linewidth=1.5)
            torsional_ax.grid(True, which='both', linestyle=':', alpha=0.7)
            torsional_ax.set_title('Кривая D-разбиения для крутильных колебаний')
            torsional_ax.set_xlabel('Re(σ)')
            torsional_ax.set_ylabel('Im(σ)')
            torsional_ax.axhline(0, color='red', linestyle='--', linewidth=0.7)
            torsional_ax.axvline(0, color='red', linestyle='--', linewidth=0.7)
            
            # Для хранения текущих пределов
            torsional_current_xlim = None
            torsional_current_ylim = None
            torsional_press = None
            
            def on_scroll_torsional(event):
                """Масштабирование колесом мыши"""
                if event.inaxes != torsional_ax:
                    return
                    
                scale_factor = 1.2 if event.button == 'up' else 1/1.2
                
                # Масштабирование относительно положения курсора
                xdata = event.xdata
                ydata = event.ydata
                
                x_left = xdata - (xdata - torsional_current_xlim[0]) * scale_factor
                x_right = xdata + (torsional_current_xlim[1] - xdata) * scale_factor
                y_bottom = ydata - (ydata - torsional_current_ylim[0]) * scale_factor
                y_top = ydata + (torsional_current_ylim[1] - ydata) * scale_factor
                
                torsional_current_xlim[:] = [x_left, x_right]
                torsional_current_ylim[:] = [y_bottom, y_top]
                
                torsional_ax.set_xlim(torsional_current_xlim)
                torsional_ax.set_ylim(torsional_current_ylim)
                torsional_canvas.draw()
            
            def on_press_torsional(event):
                """Начало панорамирования"""
                nonlocal torsional_press
                if event.inaxes != torsional_ax or event.button != MouseButton.LEFT:
                    return
                torsional_press = event.xdata, event.ydata
            
            def on_release_torsional(event):
                """Конец панорамирования"""
                nonlocal torsional_press
                torsional_press = None
                torsional_canvas.draw()
            
            def on_motion_torsional(event):
                """Панорамирование"""
                nonlocal torsional_press, torsional_current_xlim, torsional_current_ylim
                if torsional_press is None or event.inaxes != torsional_ax:
                    return
                
                xpress, ypress = torsional_press
                dx = event.xdata - xpress
                dy = event.ydata - ypress
                
                torsional_current_xlim[0] -= dx
                torsional_current_xlim[1] -= dx
                torsional_current_ylim[0] -= dy
                torsional_current_ylim[1] -= dy
                
                torsional_ax.set_xlim(torsional_current_xlim)
                torsional_ax.set_ylim(torsional_current_ylim)
                torsional_press = event.xdata, event.ydata
                torsional_canvas.draw()
            
            def reset_torsional_zoom():
                """Сброс масштаба к исходному"""
                if hasattr(self, 'torsional_original_xlim'):
                    torsional_current_xlim[:] = list(self.torsional_original_xlim)
                    torsional_current_ylim[:] = list(self.torsional_original_ylim)
                    torsional_ax.set_xlim(torsional_current_xlim)
                    torsional_ax.set_ylim(torsional_current_ylim)
                    torsional_canvas.draw()
            
            def update_torsional():
                try:
                    # Получаем значения из слайдеров с правильным масштабированием
                    l = length_slider.value()  # 2-6 метров
                    delta = delta_slider.value() * 1e-6  # 1e-6 до 1e-4
                    
                    params = self.get_current_parameters()
                    rho = params['rho']
                    G = params['G']
                    Jr = params['Jr']
                    Jp = params['Jp']
                    
                    lambda1 = np.sqrt(rho * G) * Jp / Jr
                    lambda2 = l * np.sqrt(rho / G)
                    
                    # Вывод параметров в консоль для отладки
                    print(f"\n=== Крутильные колебания (L={l} м, δ₁={delta:.2e} с) ===")
                    print(f"λ₁ = {lambda1:.2e}, λ₂ = {lambda2:.2e}")
                    
                    # Генерируем диапазон частот
                    omega = np.linspace(1000, 15000, 1000)
                    p = 1j * omega
                    
                    with np.errstate(all='ignore'):
                        # Вычисление как в консольной версии
                        expr = np.sqrt(1 + delta * p)
                        cth = 1 / np.tanh(lambda2 * p / expr)
                        sigma = -p - lambda1 * expr * cth
                        sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e10, neginf=-1e10)
                    
                    # Фильтрация аномальных значений
                    valid = (np.abs(sigma.real) < 1e8) & (np.abs(sigma.imag) < 1e8)
                    sigma = sigma[valid]
                    
                    # Обновление графика с фиксированными пределами как в консоли
                    line.set_data(sigma.real, sigma.imag)
                    torsional_ax.set_xlim(-15000, 500)
                    torsional_ax.set_ylim(-8000, 8000)
                    torsional_canvas.draw_idle()
                    
                    # Автоматическая настройка масштаба
                    if len(sigma) > 0:
                        x_pad = 0.1 * (np.nanmax(sigma.real) - np.nanmin(sigma.real))
                        y_pad = 0.1 * (np.nanmax(sigma.imag) - np.nanmin(sigma.imag))
                        
                        x_min = np.nanmin(sigma.real) - x_pad
                        x_max = np.nanmax(sigma.real) + x_pad
                        y_min = np.nanmin(sigma.imag) - y_pad
                        y_max = np.nanmax(sigma.imag) + y_pad
                        
                        torsional_ax.set_xlim(x_min, x_max)
                        torsional_ax.set_ylim(y_min, y_max)
                        
                        # Сохраняем текущие пределы для масштабирования
                        torsional_current_xlim = [x_min, x_max]
                        torsional_current_ylim = [y_min, y_max]
                    
                    # Обновляем заголовок с текущими параметрами
                    torsional_ax.set_title(
                        f'Крутильные колебания: L={l} м, δ₁={delta:.1e} с\n'
                        f'Re(σ)∈[{np.nanmin(sigma.real):.1f}, {np.nanmax(sigma.real):.1f}], '
                        f'Im(σ)∈[{np.nanmin(sigma.imag):.1f}, {np.nanmax(sigma.imag):.1f}]',
                        fontsize=10
                    )
                    
                    torsional_canvas.draw_idle()
                    
                    # Вывод контрольных точек
                    if len(sigma) > 10:
                        print(f"Первая точка: Re(σ)={sigma.real[0]:.1f}, Im(σ)={sigma.imag[0]:.1f}")
                        print(f"Средняя точка: Re(σ)={sigma.real[len(sigma)//2]:.1f}, Im(σ)={sigma.imag[len(sigma)//2]:.1f}")
                        print(f"Последняя точка: Re(σ)={sigma.real[-1]:.1f}, Im(σ)={sigma.imag[-1]:.1f}")
                    else:
                        print("Недостаточно точек для построения графика!")
                        
                except Exception as e:
                    print(f"Ошибка при обновлении графика: {str(e)}")
                    # В случае ошибки показываем сообщение на графике
                    torsional_ax.clear()
                    torsional_ax.text(0.5, 0.5, 'Ошибка при построении графика', 
                                    ha='center', va='center')
                    torsional_canvas.draw_idle()
                    
            # Подключаем обработчики событий
            torsional_canvas.mpl_connect('scroll_event', on_scroll_torsional)
            torsional_canvas.mpl_connect('button_press_event', on_press_torsional)
            torsional_canvas.mpl_connect('button_release_event', on_release_torsional)
            torsional_canvas.mpl_connect('motion_notify_event', on_motion_torsional)
            
            length_slider.valueChanged.connect(update_torsional)
            delta_slider.valueChanged.connect(update_torsional)
            
            torsional_layout.addWidget(torsional_canvas)
            torsional_layout.addWidget(QLabel("Длина борштанги (м):"))
            torsional_layout.addWidget(length_slider)
            torsional_layout.addWidget(QLabel("Коэффициент трения δ₁ (x1e-6):"))
            torsional_layout.addWidget(delta_slider)
                
            # 2. Вкладка продольных колебаний
            longitudinal_tab = QWidget()
            longitudinal_layout = QVBoxLayout(longitudinal_tab)
            
            # Создаем фигуру и canvas для продольных колебаний
            longitudinal_fig = Figure(figsize=(10, 6))
            longitudinal_canvas = FigureCanvas(longitudinal_fig)
            longitudinal_ax = longitudinal_fig.add_subplot(111)
            
            # Добавляем слайдеры
            mu_slider = QSlider(Qt.Horizontal)
            mu_slider.setRange(1, 50)
            mu_slider.setValue(10)
            
            tau_slider = QSlider(Qt.Horizontal)
            tau_slider.setRange(10, 200)
            tau_slider.setValue(60)
            
            # Инициализация графика
            line_long, = longitudinal_ax.plot([], [], 'b-', linewidth=1.5)
            longitudinal_ax.grid(True)
            longitudinal_ax.set_title('Продольные колебания (Колесо мыши - масштаб, ЛКМ - панорамирование)')
            longitudinal_ax.set_xlabel('K₁, МН/м')
            longitudinal_ax.set_ylabel('δ, кН·с/м')
            
            # Сохраняем исходные пределы
            longitudinal_original_xlim = (0, 20)
            longitudinal_original_ylim = (-150, 50)
            longitudinal_ax.set_xlim(longitudinal_original_xlim)
            longitudinal_ax.set_ylim(longitudinal_original_ylim)
            
            # Для хранения текущих пределов
            longitudinal_current_xlim = list(longitudinal_original_xlim)
            longitudinal_current_ylim = list(longitudinal_original_ylim)
            longitudinal_press = None
            
            def on_scroll_longitudinal(event):
                """Масштабирование колесом мыши"""
                if event.inaxes != longitudinal_ax:
                    return
                    
                scale_factor = 1.2 if event.button == 'up' else 1/1.2
                
                # Масштабирование относительно положения курсора
                xdata = event.xdata
                ydata = event.ydata
                
                x_left = xdata - (xdata - longitudinal_current_xlim[0]) * scale_factor
                x_right = xdata + (longitudinal_current_xlim[1] - xdata) * scale_factor
                y_bottom = ydata - (ydata - longitudinal_current_ylim[0]) * scale_factor
                y_top = ydata + (longitudinal_current_ylim[1] - ydata) * scale_factor
                
                longitudinal_current_xlim[:] = [x_left, x_right]
                longitudinal_current_ylim[:] = [y_bottom, y_top]
                
                longitudinal_ax.set_xlim(longitudinal_current_xlim)
                longitudinal_ax.set_ylim(longitudinal_current_ylim)
                longitudinal_canvas.draw()
            
            def on_press_longitudinal(event):
                """Начало панорамирования"""
                nonlocal longitudinal_press
                if event.inaxes != longitudinal_ax or event.button != MouseButton.LEFT:
                    return
                longitudinal_press = event.xdata, event.ydata
            
            def on_release_longitudinal(event):
                """Конец панорамирования"""
                nonlocal longitudinal_press
                longitudinal_press = None
                longitudinal_canvas.draw()
            
            def on_motion_longitudinal(event):
                """Панорамирование"""
                nonlocal longitudinal_press, longitudinal_current_xlim, longitudinal_current_ylim
                if longitudinal_press is None or event.inaxes != longitudinal_ax:
                    return
                
                xpress, ypress = longitudinal_press
                dx = event.xdata - xpress
                dy = event.ydata - ypress
                
                longitudinal_current_xlim[0] -= dx
                longitudinal_current_xlim[1] -= dx
                longitudinal_current_ylim[0] -= dy
                longitudinal_current_ylim[1] -= dy
                
                longitudinal_ax.set_xlim(longitudinal_current_xlim)
                longitudinal_ax.set_ylim(longitudinal_current_ylim)
                longitudinal_press = event.xdata, event.ydata
                longitudinal_canvas.draw()
            
            def reset_longitudinal_zoom():
                """Сброс масштаба к исходному"""
                longitudinal_current_xlim[:] = list(longitudinal_original_xlim)
                longitudinal_current_ylim[:] = list(longitudinal_original_ylim)
                longitudinal_ax.set_xlim(longitudinal_current_xlim)
                longitudinal_ax.set_ylim(longitudinal_current_ylim)
                longitudinal_canvas.draw()
            
            def update_longitudinal():
                mu = mu_slider.value() / 100  # Преобразуем 1-50 в 0.01-0.5
                tau = tau_slider.value() * 1e-3  # Преобразуем 10-200 в 0.01-0.2
                
                params = self.get_current_parameters()
                E = params['E']
                S = params['S']
                rho = params['rho']
                L = params['length']
                
                a = np.sqrt(E/rho)
                omega_main = np.pi*a/L
                
                print("\n=== Параметры системы ===")
                print(f"E = {E:.2e} Па, S = {S:.2e} м², ρ = {rho:.1f} кг/м³")
                print(f"L = {L} м, μ = {mu}, τ = {tau:.3f} с")
                
                print(f"\n1. Скорость волны:")
                print(f"a = sqrt(E/ρ) = sqrt({E:.2e}/{rho:.1f}) = {a:.2f} м/с")
                
                print(f"\n2. Основная частота:")
                print(f"ω_main = π*a/L = π*{a:.2f}/{L} = {omega_main:.2f} рад/с ({omega_main/(2*np.pi):.2f} Гц)")
                
                omega = np.linspace(0.01, 2*np.pi*100, 5000)
                
                with np.errstate(all='ignore'):
                    x = omega * L / a
                    mask = (np.abs(np.sin(x)) > 1e-6)
                    cot = np.zeros_like(x)
                    cot[mask] = 1/np.tan(x[mask])
                    
                    denom = 1 - mu * np.cos(omega * tau)
                    denom_mask = np.abs(denom) > 1e-6
                    
                    valid = mask & denom_mask
                    
                    K1 = np.full_like(omega, np.nan)
                    delta = np.full_like(omega, np.nan)
                    
                    K1[valid] = (E*S/a) * omega[valid] * cot[valid] / denom[valid]
                    delta[valid] = -(E*S*mu/a) * cot[valid] * np.sin(omega[valid]*tau) / denom[valid]
                    
                    valid = valid & (K1 > 0) & (K1 < 1e10) & (np.abs(delta) < 1e6)
                    K1 = K1[valid]
                    delta = delta[valid]
                    omega_valid = omega[valid]
                    
                    print(f"\n3. Диапазон x = ω*L/a: от {x.min():.2f} до {x.max():.2f}")
                    print(f"Условия при sin(x)≈0: {np.sum(~mask)} точек из {len(x)}")
                    print(f"Условия при denom≈0: {np.sum(~denom_mask)} точек из {len(x)}")
                    print(f"\n4. Результаты после фильтрации:")
                    print(f"Осталось {len(K1)} точек из {len(x)}")
                    
                    if len(K1) > 0:
                        print("\n5. Крайние точки D-разбиения:")
                        print(f"Первая точка: ω={omega_valid[0]:.2f}, K1={K1[0]/1e6:.2f} МН/м, δ={delta[0]/1e3:.2f} кН·с/м")
                        print(f"Последняя точка: ω={omega_valid[-1]:.2f}, K1={K1[-1]/1e6:.2f} МН/м, δ={delta[-1]/1e3:.2f} кН·с/м")
                        
                        K1_0 = (E*S)/(L*(1 - mu))
                        delta_0 = - (E*S*mu*tau)/(L*(1 - mu))
                        print("\n6. Проверка асимптотики при ω→0:")
                        print(f"K1(ω→0) = {K1_0/1e6:.2f} МН/м (ожидается ~17-18 МН/м)")
                        print(f"δ(ω→0) = {delta_0/1e3:.2f} кН·с/м (ожидается ~-100 кН·с/м)")
                
                line_long.set_data(K1/1e6, delta/1e3)
                longitudinal_canvas.draw()
            
            # Подключаем обработчики событий
            longitudinal_canvas.mpl_connect('scroll_event', on_scroll_longitudinal)
            longitudinal_canvas.mpl_connect('button_press_event', on_press_longitudinal)
            longitudinal_canvas.mpl_connect('button_release_event', on_release_longitudinal)
            longitudinal_canvas.mpl_connect('motion_notify_event', on_motion_longitudinal)
            
            mu_slider.valueChanged.connect(update_longitudinal)
            tau_slider.valueChanged.connect(update_longitudinal)
            
            longitudinal_layout.addWidget(longitudinal_canvas)
            longitudinal_layout.addWidget(QLabel("Коэффициент трения μ (x0.01):"))
            longitudinal_layout.addWidget(mu_slider)
            longitudinal_layout.addWidget(QLabel("Время запаздывания τ (мс):"))
            longitudinal_layout.addWidget(tau_slider)
            
            # Добавляем кнопки управления масштабированием
            control_buttons = QHBoxLayout()
            
            reset_zoom_torsional_btn = QPushButton("Сбросить масштаб (Крутильные)")
            reset_zoom_torsional_btn.clicked.connect(reset_torsional_zoom)
            
            reset_zoom_longitudinal_btn = QPushButton("Сбросить масштаб (Продольные)")
            reset_zoom_longitudinal_btn.clicked.connect(reset_longitudinal_zoom)
            
            control_buttons.addWidget(reset_zoom_torsional_btn)
            control_buttons.addWidget(reset_zoom_longitudinal_btn)
            
            # Добавляем вкладки
            tabs.addTab(torsional_tab, "Крутильные колебания")
            tabs.addTab(longitudinal_tab, "Продольные колебания")
            
            layout.addWidget(tabs)
            layout.addLayout(control_buttons)
            
            # Кнопка закрытия
            close_btn = QPushButton("Закрыть")
            close_btn.clicked.connect(dialog.close)
            layout.addWidget(close_btn)
            
            # Первоначальное обновление графиков
            update_torsional()
            update_longitudinal()
            
            dialog.exec_()

            try:
                update_torsional()
            except Exception as e:
                print(f"Ошибка при построении крутильных колебаний: {str(e)}")
                torsional_ax.clear()
                torsional_ax.text(0.5, 0.5, 'Ошибка при построении', 
                                ha='center', va='center')
                torsional_canvas.draw()
                
            try:
                update_longitudinal()
            except Exception as e:
                print(f"Ошибка при построении продольных колебаний: {str(e)}")
                longitudinal_ax.clear()
                longitudinal_ax.text(0.5, 0.5, 'Ошибка при построении', 
                                ha='center', va='center')
                longitudinal_canvas.draw()
                
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
        
        # Получаем параметры из UI
        rho = params['rho']
        G = params['G']
        Jp = params['Jp']
        Jr = params['Jr']
        delta1 = params['delta1'] * params['multiplier']
        length = params['length']
        
        # Вычисляем константы
        lambda1 = np.sqrt(rho * G) * Jp / Jr
        lambda2 = length * np.sqrt(rho / G)
        
        print("\n=== Параметры системы для крутильных колебаний ===")
        print(f"ρ = {rho:.1f} кг/м³, G = {G:.2e} Па")
        print(f"Jp = {Jp:.2e} м⁴, Jr = {Jr:.2e} кг·м²")
        print(f"L = {length} м, δ₁ = {delta1:.2e} с")
        print(f"\n1. Вычисленные константы:")
        print(f"λ₁ = sqrt(ρ*G)*Jp/Jr = {lambda1:.2e}")
        print(f"λ₂ = L*sqrt(ρ/G) = {lambda2:.2e}")
        print(f"δ₁ (с учетом множителя) = {delta1:.2e} с")

        # 1. Кривая D-разбиения
        omega = np.linspace(1000, 15000, 1000)
        p = 1j * omega
        
        with np.errstate(all='ignore'):
            # Улучшенное вычисление с масштабированием
            expr = np.sqrt(1 + delta1 * p)
            
            # Вычисление coth с защитой от переполнения
            coth_arg = lambda2 * p / expr
            # Масштабирование аргумента для избежания переполнения
            scaled_arg = np.where(np.abs(coth_arg) > 100, 
                            100 * np.sign(coth_arg), 
                            coth_arg)
            coth = (np.exp(2*scaled_arg) + 1) / (np.exp(2*scaled_arg) - 1)
            coth = np.nan_to_num(coth, nan=1.0, posinf=1.0, neginf=-1.0)
            
            sigma = -p - lambda1 * expr * coth
            sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Фильтрация аномально больших значений
        valid = (np.abs(sigma.real) < 1e6) & (np.abs(sigma.imag) < 1e6)
        sigma = sigma[valid]
        omega = omega[valid]
        
        print("\n2. Проверка кривой D-разбиения:")
        print(f"Первая точка: ω={omega[0]:.2f}, Re(σ)={sigma.real[0]:.2f}, Im(σ)={sigma.imag[0]:.2f}")
        print(f"Последняя точка: ω={omega[-1]:.2f}, Re(σ)={sigma.real[-1]:.2f}, Im(σ)={sigma.imag[-1]:.2f}")
        
        # Проверка асимптотики при ω→0
        if len(omega) > 0:
            omega_small = omega[0]
            sigma_small = sigma[0]
            print("\n3. Проверка асимптотики при ω→0:")
            print(f"σ(ω→{omega_small:.2f}) ≈ {sigma_small.real:.2f}+{sigma_small.imag:.2f}j")
        
        ax1.plot(sigma.real, sigma.imag, 'b-', linewidth=1.5)
        ax1.axhline(0, color='red', linestyle='--', linewidth=0.7)
        ax1.axvline(0, color='red', linestyle='--', linewidth=0.7)
        ax1.set_title('Кривая D-разбиения для крутильных колебаний', fontsize=10)
        ax1.set_xlabel('Re(σ)', fontsize=8)
        ax1.set_ylabel('Im(σ)', fontsize=8)
        ax1.grid(True, which='both', linestyle=':', alpha=0.7)
        
        # Автоматическая настройка масштаба с учетом данных
        if len(sigma.real) > 0:
            x_pad = 0.1 * (np.max(sigma.real) - np.min(sigma.real))
            y_pad = 0.1 * (np.max(sigma.imag) - np.min(sigma.imag))
            ax1.set_xlim(np.min(sigma.real)-x_pad, np.max(sigma.real)+x_pad)
            ax1.set_ylim(np.min(sigma.imag)-y_pad, np.max(sigma.imag)+y_pad)
        
        # 2. Диаграмма устойчивости для разных длин
        lengths = np.array([2.5, 3, 4, 5, 6])
        multipliers = np.array([1, 2, 3, 4, 6, 10])
        delta1_values = params['delta1'] * multipliers
        lambda2_values = lengths * np.sqrt(rho / G)
        colors = plt.cm.viridis(np.linspace(0, 1, len(lengths)))
        
        print("\n4. Вычисление диаграммы устойчивости:")
        
        for i, l in enumerate(lengths):
            Sigma = np.zeros(len(delta1_values))
            
            for j, delta in enumerate(delta1_values):
                def im_sigma(omega_val):
                    p_val = 1j * omega_val
                    with np.errstate(all='ignore'):
                        sqrt_expr = np.sqrt(1 + delta * p_val)
                        coth_arg = lambda2_values[i] * p_val / sqrt_expr
                        # Масштабирование аргумента
                        scaled_arg = np.where(np.abs(coth_arg) > 100, 
                                        100 * np.sign(coth_arg), 
                                        coth_arg)
                        coth = (np.exp(2*scaled_arg) + 1) / (np.exp(2*scaled_arg) - 1)
                        coth = np.nan_to_num(coth, nan=1.0, posinf=1.0, neginf=-1.0)
                        val = -p_val - lambda1 * sqrt_expr * coth
                        return val.imag
                
                try:
                    # Адаптивный поиск корня
                    omega_min = 100
                    omega_max = 2000
                    
                    # Для больших длин уменьшаем верхнюю границу
                    if l >= 5.5:
                        omega_max = 1000
                    
                    # Проверка знаков на границах
                    f_min = im_sigma(omega_min)
                    f_max = im_sigma(omega_max)
                    
                    if f_min * f_max < 0:
                        sol = root_scalar(im_sigma, 
                                        bracket=[omega_min, omega_max],
                                        method='brentq',
                                        xtol=1e-6,
                                        rtol=1e-6)
                        omega_sol = sol.root
                        
                        p_sol = 1j * omega_sol
                        with np.errstate(all='ignore'):
                            sqrt_expr = np.sqrt(1 + delta * p_sol)
                            coth_arg = lambda2_values[i] * p_sol / sqrt_expr
                            scaled_arg = np.where(np.abs(coth_arg) > 100, 
                                            100 * np.sign(coth_arg), 
                                            coth_arg)
                            coth = (np.exp(2*scaled_arg) + 1) / (np.exp(2*scaled_arg) - 1)
                            coth = np.nan_to_num(coth, nan=1.0, posinf=1.0, neginf=-1.0)
                            Sigma[j] = (-p_sol - lambda1 * sqrt_expr * coth).real
                        
                        print(f"L={l} м, δ₁={delta:.2e}: ω={omega_sol:.1f}, Re(σ)={Sigma[j]:.1f}")
                    else:
                        Sigma[j] = np.nan
                        print(f"Ошибка для L={l} м, δ₁={delta:.2e}: f(a) and f(b) must have different signs")
                        
                except Exception as e:
                    Sigma[j] = np.nan
                    print(f"Ошибка для L={l} м, δ₁={delta:.2e}: {str(e)}")
            
            valid = ~np.isnan(Sigma)
            if np.any(valid):
                ax2.plot(delta1_values[valid], Sigma[valid], 'o-', color=colors[i], 
                        label=f'L={l} м', markersize=4, linewidth=1.5)
            else:
                ax2.plot([], [], 'o-', color=colors[i], label=f'L={l} м (нет решений)')
        
        ax2.axhline(0, color='k', linestyle='--', linewidth=0.8)
        ax2.fill_between([3.44e-6, 3.44e-5], -1500, 0, color='green', alpha=0.15)
        ax2.set_xscale('log')
        ax2.set_ylim(-1500, 100)
        ax2.set_xlabel('Коэффициент внутреннего трения δ₁ (с)', fontsize=8)
        ax2.set_ylabel('Re(σ)', fontsize=8)
        ax2.set_title('Диаграмма устойчивости крутильных колебаний', fontsize=10)
        ax2.legend(fontsize=8)
        ax2.grid(True, which='both', linestyle=':', alpha=0.7)
        
        self.torsional_figure.tight_layout()
        self.torsional_canvas.draw()

    def save_torsional_data(self):
        """Сохранение данных крутильных колебаний в файл"""
        try:
            if not hasattr(self, 'torsional_data'):
                return
                
            # Создаем имя файла с текущей датой и временем
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"torsional_data_{timestamp}.csv"
            
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Записываем основные параметры
                writer.writerow(['Parameter', 'Value'])
                writer.writerow(['lambda1', self.torsional_data['lambda1']])
                writer.writerow(['lambda2', self.torsional_data['lambda2']])
                writer.writerow(['delta1', self.torsional_data['delta1']])
                
                # Записываем данные D-разбиения
                writer.writerow([])  # Пустая строка для разделения
                writer.writerow(['D-partition data'])
                writer.writerow(['omega', 'sigma_real', 'sigma_imag'])
                for i in range(len(self.torsional_data['omega'])):
                    writer.writerow([
                        self.torsional_data['omega'][i],
                        self.torsional_data['sigma_real'][i],
                        self.torsional_data['sigma_imag'][i]
                    ])
                
                # Записываем данные устойчивости
                writer.writerow([])  # Пустая строка для разделения
                writer.writerow(['Stability data'])
                for stability in self.torsional_data['stability']:
                    writer.writerow([f'Length = {stability["length"]} m'])
                    writer.writerow(['delta1', 'sigma'])
                    for i in range(len(stability['delta1_values'])):
                        writer.writerow([
                            stability['delta1_values'][i],
                            stability['sigma_values'][i]
                        ])
                    writer.writerow([])  # Пустая строка между разными длинами
            
            print(f"\nДанные сохранены в файл: {filename}")
            
        except Exception as e:
            print(f"Ошибка при сохранении данных: {str(e)}")

    def analyze_longitudinal(self, params):
        """Анализ продольных колебаний с построением D-разбиения и проверочными вычислениями"""
        self.longitudinal_figure.clear()
        ax = self.longitudinal_figure.add_subplot(111)
        
        # Получаем параметры из UI
        E = params['E']
        S = params['S']
        rho = params['rho']
        L = params['length']
        mu = params['mu']
        tau = params['tau']
        
        # ===== ПРОВЕРОЧНЫЕ ВЫЧИСЛЕНИЯ =====
        print("\n=== Параметры системы ===")
        print(f"E = {E:.2e} Па, S = {S:.2e} м², ρ = {rho:.1f} кг/м³")
        print(f"L = {L} м, μ = {mu}, τ = {tau:.3f} с")
        
        # 1. Проверка скорости волны
        a = np.sqrt(E/rho)
        print(f"\n1. Скорость волны:")
        print(f"a = sqrt(E/ρ) = sqrt({E:.2e}/{rho:.1f}) = {a:.2f} м/с")
        
        # 2. Проверка основной частоты
        omega_main = np.pi*a/L
        print(f"\n2. Основная частота:")
        print(f"ω_main = π*a/L = π*{a:.2f}/{L} = {omega_main:.2f} рад/с ({omega_main/(2*np.pi):.2f} Гц)")
        
        # Генерируем диапазон частот
        omega = np.linspace(0.01, 2*np.pi*100, 5000)
        
        with np.errstate(all='ignore'):
            # 3. Проверка вычисления x = ω*L/a
            x = omega * L / a
            print(f"\n3. Диапазон x = ω*L/a: от {x.min():.2f} до {x.max():.2f}")
            
            # 4. Проверка cot(x)
            mask = (np.abs(np.sin(x)) > 1e-6)
            cot = np.zeros_like(x)
            cot[mask] = 1/np.tan(x[mask])
            print(f"Условия при sin(x)≈0: {np.sum(~mask)} точек из {len(x)}")
            
            # 5. Проверка знаменателя
            denom = 1 - mu * np.cos(omega * tau)
            denom_mask = np.abs(denom) > 1e-6
            print(f"Условия при denom≈0: {np.sum(~denom_mask)} точек из {len(x)}")
            
            valid = mask & denom_mask
            
            # 6. Вычисление K1 и delta
            K1 = np.full_like(omega, np.nan)
            delta = np.full_like(omega, np.nan)
            
            K1[valid] = (E*S/a) * omega[valid] * cot[valid] / denom[valid]
            delta[valid] = -(E*S*mu/a) * cot[valid] * np.sin(omega[valid]*tau) / denom[valid]
            
            # Фильтрация недопустимых значений
            valid = valid & (K1 > 0) & (K1 < 1e10) & (np.abs(delta) < 1e6)
            K1 = K1[valid]
            delta = delta[valid]
            omega_valid = omega[valid]
            
            print(f"\n4. Результаты после фильтрации:")
            print(f"Осталось {len(K1)} точек из {len(x)}")
            
            # 7. Проверка крайних точек
            if len(K1) > 0:
                print("\n5. Крайние точки D-разбиения:")
                print(f"Первая точка: ω={omega_valid[0]:.2f}, K1={K1[0]/1e6:.2f} МН/м, δ={delta[0]/1e3:.2f} кН·с/м")
                print(f"Последняя точка: ω={omega_valid[-1]:.2f}, K1={K1[-1]/1e6:.2f} МН/м, δ={delta[-1]/1e3:.2f} кН·с/м")
                
                # 8. Проверка вблизи ω=0 (асимптотика)
                print("\n6. Проверка асимптотики при ω→0:")
                K1_0 = (E*S)/(L*(1 - mu))
                delta_0 = - (E*S*mu*tau)/(L*(1 - mu))
                print(f"K1(ω→0) = {K1_0/1e6:.2f} МН/м (ожидается ~17-18 МН/м)")
                print(f"δ(ω→0) = {delta_0/1e3:.2f} кН·с/м (ожидается ~-100 кН·с/м)")
        
        # ===== ВИЗУАЛИЗАЦИЯ =====
        if len(K1) > 0:
            ax.plot(K1/1e6, delta/1e3, 'b-', linewidth=1.5)
            ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
            ax.fill_between([0, 20], -150, 0, color='green', alpha=0.15)
            
            ax.set_title('D-разбиение для продольных колебаний', fontsize=10)
            ax.set_xlabel('K₁, МН/м', fontsize=8)
            ax.set_ylabel('δ, кН·с/м', fontsize=8)
            ax.grid(True, which='both', linestyle=':', alpha=0.7)
            ax.set_xlim(0, 20)
            ax.set_ylim(-150, 50)
            
            # Добавляем пояснительные надписи с проверочными значениями
            info_text = (
                f"Проверочные значения (L={L} м):\n"
                f"a = {a:.2f} м/с\n"
                f"ω_main = {omega_main:.2f} рад/с\n"
                f"K1(0) = {K1_0/1e6:.2f} МН/м\n"
                f"δ(0) = {delta_0/1e3:.2f} кН·с/м"
            )
            ax.text(12, -120, info_text, bbox=dict(facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'Нет данных для построения графика\nПроверьте параметры', 
                ha='center', va='center', transform=ax.transAxes)
            print("\nОШИБКА: Нет данных для построения графика!")
        
        self.longitudinal_figure.tight_layout()
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