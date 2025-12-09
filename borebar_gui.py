import sys
import json
import csv
import logging
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, 
                            QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
                            QDoubleSpinBox, QComboBox, QPushButton, 
                            QStatusBar, QFileDialog, QMessageBox, QDialog, QSlider)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backend_bases import MouseButton
from borebar_model import BoreBarModel
from scipy.optimize import root_scalar

logging.basicConfig(filename='borebar_analysis.log', 
                   level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

class BoreBarGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Анализ колебаний борштанги")
        self.setGeometry(100, 100, 1800, 800)
        self.model = BoreBarModel()
        self.init_ui()
        
    def init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        
        self.setup_parameters_panel()
        self.setup_right_panel()
        
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.init_parameters()
        
    def setup_parameters_panel(self):
        self.params_panel = QGroupBox("Параметры системы")
        self.params_layout = QVBoxLayout()
        self.setup_parameters_ui()
        self.params_panel.setLayout(self.params_layout)
        self.main_layout.addWidget(self.params_panel, stretch=1)
        
    def setup_right_panel(self):
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)
        
        self.tabs = QTabWidget()
        self.setup_analysis_tabs()
        self.right_layout.addWidget(self.tabs, stretch=5)
        
        self.control_buttons = self.setup_control_buttons()
        self.right_layout.addLayout(self.control_buttons, stretch=1)
        
        self.main_layout.addWidget(self.right_panel, stretch=4)
        
    def setup_parameters_ui(self):
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

        transverse_group = QGroupBox("Параметры поперечных колебаний")
        transverse_layout = QVBoxLayout()

        self.R_input = self.create_parameter_input("Внешний радиус R (м)", 0.04, 0.001, 0.2)
        self.r_input = self.create_parameter_input("Внутренний радиус r (м)", 0.035, 0.0, 0.2)
        self.K_cut_input = self.create_parameter_input("Дин. жёсткость резания K (Н/м)", 6e5, 1e3, 1e8)
        self.h_input = self.create_parameter_input("Коэф. внутр. трения h", 0.0, 0.0, 10.0)
        self.beta_input = self.create_parameter_input("Коэф. демпфирования β", 0.3, 0.0, 10.0)

        transverse_layout.addWidget(self.R_input)
        transverse_layout.addWidget(self.r_input)
        transverse_layout.addWidget(self.K_cut_input)
        transverse_layout.addWidget(self.h_input)
        transverse_layout.addWidget(self.beta_input)

        transverse_group.setLayout(transverse_layout)
        self.params_layout.addWidget(transverse_group)

        
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
        
        self.params_buttons = QHBoxLayout()
        self.save_params_btn = QPushButton("Сохранить параметры")
        self.load_params_btn = QPushButton("Загрузить параметры")
        self.reset_params_btn = QPushButton("Сбросить к умолчаниям")
        
        self.params_buttons.addWidget(self.save_params_btn)
        self.params_buttons.addWidget(self.load_params_btn)
        self.params_buttons.addWidget(self.reset_params_btn)
        
        self.params_layout.addLayout(self.params_buttons)
        
        self.save_params_btn.clicked.connect(self.save_parameters)
        self.load_params_btn.clicked.connect(self.load_parameters)
        self.reset_params_btn.clicked.connect(self.init_parameters)
        
        self.setup_intersection_panel()
        
    def setup_intersection_panel(self):
        self.intersection_panel = QGroupBox("Точка пересечения с осью")
        self.intersection_layout = QVBoxLayout()
        
        self.intersection_label = QLabel("Точка не найдена")
        self.intersection_label.setStyleSheet("font-size: 12px;")
        self.intersection_layout.addWidget(self.intersection_label)

        self.intersection_panel.setStyleSheet("""
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                border: 1px solid gray;
                border-radius: 5px;
                margin-top: 10px;
            }
            QLabel {
                font-size: 12px;
                font-weight: normal;
            }
        """)
        
        self.intersection_panel.setLayout(self.intersection_layout)
        self.params_layout.addWidget(self.intersection_panel)
        
    def setup_analysis_tabs(self):
        self.torsional_tab = QWidget()
        self.longitudinal_tab = QWidget()
        self.transverse_tab = QWidget()
        self.stability_tab = QWidget()
        
        self.setup_torsional_tab()
        self.setup_longitudinal_tab()
        self.setup_transverse_tab()
        self.setup_stability_tab()
        
        self.tabs.addTab(self.torsional_tab, "Крутильные колебания")
        self.tabs.addTab(self.longitudinal_tab, "Продольные колебания")
        self.tabs.addTab(self.transverse_tab, "Поперечные колебания")
        self.tabs.addTab(self.stability_tab, "Диаграмма устойчивости")
        
    def setup_torsional_tab(self):
        layout = QVBoxLayout(self.torsional_tab)
        self.torsional_figure = Figure()
        self.torsional_canvas = FigureCanvas(self.torsional_figure)
        layout.addWidget(self.torsional_canvas)
        
    def setup_longitudinal_tab(self):
        layout = QVBoxLayout(self.longitudinal_tab)
        self.longitudinal_figure = Figure()
        self.longitudinal_canvas = FigureCanvas(self.longitudinal_figure)
        layout.addWidget(self.longitudinal_canvas)
    
    def setup_transverse_tab(self):
        layout = QVBoxLayout(self.transverse_tab)
        self.transverse_figure = Figure()
        self.transverse_canvas = FigureCanvas(self.transverse_figure)
        layout.addWidget(self.transverse_canvas)
        
    def setup_control_buttons(self):
        buttons_layout = QHBoxLayout()
        
        self.analyze_btn = QPushButton("Выполнить анализ")
        self.export_btn = QPushButton("Экспорт результатов")
        self.interactive_btn = QPushButton("Интерактивный режим")
        
        buttons_layout.addWidget(self.analyze_btn)
        buttons_layout.addWidget(self.export_btn)
        buttons_layout.addWidget(self.interactive_btn)
        
        self.analyze_btn.clicked.connect(self.run_analysis)
        self.export_btn.clicked.connect(self.export_results)
        self.interactive_btn.clicked.connect(self.show_interactive)
        
        return buttons_layout
        
    def create_parameter_input(self, label, default, min_val, max_val):
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
        self.R_input.findChild(QDoubleSpinBox).setValue(0.04)
        self.r_input.findChild(QDoubleSpinBox).setValue(0.035)
        self.K_cut_input.findChild(QDoubleSpinBox).setValue(6e5)
        self.h_input.findChild(QDoubleSpinBox).setValue(0.0)
        self.beta_input.findChild(QDoubleSpinBox).setValue(0.3)

        
    def get_current_parameters(self):
        return {
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
            'multiplier': int(self.multiplier_combo.currentText()),
            'R': self.R_input.findChild(QDoubleSpinBox).value(),
            'r': self.r_input.findChild(QDoubleSpinBox).value(),
            'K_cut': self.K_cut_input.findChild(QDoubleSpinBox).value(),
            'h': self.h_input.findChild(QDoubleSpinBox).value(),
            'beta': self.beta_input.findChild(QDoubleSpinBox).value()
        }
    
    def save_parameters(self):
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
        filename, _ = QFileDialog.getOpenFileName(
            self, "Загрузить параметры", "", "JSON Files (*.json)"
        )
        if filename:
            try:
                with open(filename, 'r') as f:
                    params = json.load(f)
                
                self.rho_input.findChild(QDoubleSpinBox).setValue(params.get('rho', 7800))
                self.G_input.findChild(QDoubleSpinBox).setValue(params.get('G', 8e10))
                self.E_input.findChild(QDoubleSpinBox).setValue(params.get('E', 200e9))
                self.S_input.findChild(QDoubleSpinBox).setValue(params.get('S', 2e-4))
                self.Jr_input.findChild(QDoubleSpinBox).setValue(params.get('Jr', 2.57e-2))
                self.Jp_input.findChild(QDoubleSpinBox).setValue(params.get('Jp', 1.9e-5))
                self.delta1_input.findChild(QDoubleSpinBox).setValue(params.get('delta1', 3.44e-6))
                self.mu_input.findChild(QDoubleSpinBox).setValue(params.get('mu', 0.1))
                self.tau_input.findChild(QDoubleSpinBox).setValue(params.get('tau', 60e-3))
                self.length_combo.setCurrentIndex(self.length_combo.findText(str(params.get('length', '2.5'))))
                self.multiplier_combo.setCurrentIndex(self.multiplier_combo.findText(str(params.get('multiplier', '1'))))
                self.R_input.findChild(QDoubleSpinBox).setValue(params.get('R', 0.04))
                self.r_input.findChild(QDoubleSpinBox).setValue(params.get('r', 0.035))
                self.K_cut_input.findChild(QDoubleSpinBox).setValue(params.get('K_cut', 6e5))
                self.h_input.findChild(QDoubleSpinBox).setValue(params.get('h', 0.0))
                self.beta_input.findChild(QDoubleSpinBox).setValue(params.get('beta', 0.3))
                
                self.status_bar.showMessage("Параметры успешно загружены", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить параметры: {str(e)}")
    
    def run_analysis(self):
        current_tab = self.tabs.currentIndex()
        params = self.get_current_parameters()
        
        try:
            if current_tab == 0:
                self.analyze_torsional(params)
            elif current_tab == 1:
                self.analyze_longitudinal(params)
            elif current_tab == 2:
                self.analyze_transverse(params)
            self.status_bar.showMessage("Анализ успешно завершен", 3000)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка анализа", f"Ошибка при выполнении анализа: {str(e)}")
            self.status_bar.showMessage("Ошибка при выполнении анализа", 3000)

    def analyze_torsional(self, params):
        self.torsional_figure.clear()
        ax = self.torsional_figure.add_subplot(111)

        result = self.model.calculate_torsional(params)

        print("=== Torsional analysis ===")
        print(f"Points count: {len(result['sigma_real'])}")
        print(f"L: {params['length']}")

        if result['sigma_real'].any():
            print(f"Re(σ) range: {min(result['sigma_real']):.4e} … {max(result['sigma_real']):.4e}")
        if result['sigma_imag'].any():
            print(f"Im(σ) range: {min(result['sigma_imag']):.4e} … {max(result['sigma_imag']):.4e}")

        ax.plot(result['sigma_real'], result['sigma_imag'], 'b-', linewidth=1.5)
        ax.axhline(0, color='red', linestyle='--', linewidth=0.7)
        ax.axvline(0, color='red', linestyle='--', linewidth=0.7)

        ax.relim()
        ax.autoscale_view()
        ax.margins(x=0.05, y=0.05)

        print(f"Auto xlim: {ax.get_xlim()}")
        print(f"Auto ylim: {ax.get_ylim()}")

        ax.set_title('Кривая D-разбиения для крутильных колебаний', fontsize=10)
        ax.set_xlabel('Re(σ)', fontsize=8)
        ax.set_ylabel('Im(σ)', fontsize=8)
        ax.grid(True, which='both', linestyle=':', alpha=0.7)

        intersection = self.model.find_intersection(params)
        if intersection is not None:
            ax.plot(intersection['re_sigma'], 0, 'ro', markersize=8, label='Пересечение')
            ax.legend()
            self.intersection_label.setText(
                f"ω = {intersection['omega']:.2f} рад/с\n"
                f"Re(σ) = {intersection['re_sigma']:.2f}\n"
                f"Частота: {intersection['frequency']:.2f} Гц"
            )
        else:
            self.intersection_label.setText("Пересечение с Im(σ)=0 не найдено.")

        self.torsional_canvas.draw()

    def analyze_longitudinal(self, params):
        self.longitudinal_figure.clear()
        ax = self.longitudinal_figure.add_subplot(111)

        result = self.model.calculate_longitudinal(params)

        if len(result['K1']) > 0:
            K1_MN   = result['K1']   / 1e6
            delta_k = result['delta'] / 1e3

            print("=== Longitudinal analysis ===")
            print(f"Points count: {len(K1_MN)}")
            print(f"K₁ range (МН/м): {min(K1_MN):.4f} … {max(K1_MN):.4f}")
            print(f"δ  range (кН·с/м): {min(delta_k):.4f} … {max(delta_k):.4f}")

            ax.plot(K1_MN, delta_k, 'b-', linewidth=1.5)
            ax.axhline(0, color='k', linestyle='--', linewidth=0.8)

            ax.relim()
            ax.autoscale_view()
            ax.margins(x=0.05, y=0.05)

            print(f"Auto xlim: {ax.get_xlim()}")
            print(f"Auto ylim: {ax.get_ylim()}")

            ax.set_title('D-разбиение для продольных колебаний', fontsize=10)
            ax.set_xlabel('K₁, МН/м', fontsize=8)
            ax.set_ylabel('δ, кН·с/м', fontsize=8)
            ax.grid(True, which='both', linestyle=':', alpha=0.7)
        else:
            ax.text(0.5, 0.5, 'Нет данных для построения графика\nПроверьте параметры',
                    ha='center', va='center', transform=ax.transAxes)

        self.longitudinal_canvas.draw()

    def analyze_transverse(self, params):
        """Построение кривой D-разбиения для поперечных колебаний (годограф W(p))."""
        self.transverse_figure.clear()
        ax = self.transverse_figure.add_subplot(111)

        result = self.model.calculate_transverse(params)

        if len(result['W_real']) > 0:
            ax.plot(result['W_real'], result['W_imag'], 'b-', linewidth=1.5)
            ax.axhline(0, color='red', linestyle='--', linewidth=0.7)
            ax.axvline(0, color='red', linestyle='--', linewidth=0.7)

            ax.set_title('Кривая D-разбиения для поперечных колебаний', fontsize=10)
            ax.set_xlabel('Re(W)', fontsize=8)
            ax.set_ylabel('Im(W)', fontsize=8)
            ax.grid(True, which='both', linestyle=':', alpha=0.7)

            ax.relim()
            ax.autoscale_view()
            ax.margins(x=0.05, y=0.05)
        else:
            ax.text(0.5, 0.5,
                    'Нет данных для построения графика\nПроверьте параметры',
                    ha='center', va='center', transform=ax.transAxes)

        self.transverse_canvas.draw()


    def export_results(self):
        formats = ["JSON (*.json)", "CSV (*.csv)"]
        filename, selected_filter = QFileDialog.getSaveFileName(
            self, "Экспорт результатов", "", ";;".join(formats)
        )
        
        if not filename:
            return
        
        file_format = 'json' if selected_filter == "JSON (*.json)" else 'csv'
        
        try:
            results = {
                'parameters': self.get_current_parameters(),
                'analysis': {
                    'torsional': self._get_torsional_results(),
                    'longitudinal': self._get_longitudinal_results(),
                }
            }
            
            if file_format == 'json':
                with open(filename, 'w') as f:
                    json.dump(self._convert_to_serializable(results), f, indent=4)
            else:
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    for key, value in results['parameters'].items():
                        writer.writerow([f"param_{key}", value])
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
        params = self.get_current_parameters()
        result = self.model.calculate_torsional(params)
        return {
            'omega': result['omega'],
            'sigma_real': result['sigma_real'],
            'sigma_imag': result['sigma_imag'],
            'lambda1': result['lambda1'],
            'lambda2': result['lambda2'],
            'delta1': result['delta1']
        }

    def _get_longitudinal_results(self):
        params = self.get_current_parameters()
        result = self.model.calculate_longitudinal(params)
        return {
            'omega': result['omega'],
            'K1': result['K1'],
            'delta': result['delta'],
            'a': result['a'],
            'omega_main': result['omega_main'],
            'K1_0': result['K1_0'],
            'delta_0': result['delta_0']
        }

    def _convert_to_serializable(self, data):
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
        
    def setup_stability_tab(self):
        layout = QVBoxLayout(self.stability_tab)
        self.stability_figure = Figure()
        self.stability_canvas = FigureCanvas(self.stability_figure)
        layout.addWidget(self.stability_canvas)

        self.plot_stability_btn = QPushButton("Построить диаграмму устойчивости")
        self.plot_stability_btn.clicked.connect(self.plot_stability_diagram)
        layout.addWidget(self.plot_stability_btn)

    def plot_stability_diagram(self):
        self.stability_figure.clear()
        ax = self.stability_figure.add_subplot(111)

        params = self.get_current_parameters()
        L = params['length']
        delta_base = params['delta1']
        multiplier_list = [1, 2, 3, 4, 6, 10]
        delta1_values = [delta_base * m for m in multiplier_list]

        rho = params['rho']
        G = params['G']
        Jr = params['Jr']
        Jp = params['Jp']
        lambda1 = np.sqrt(rho * G) * Jp / Jr
        lambda2 = L * np.sqrt(rho / G)

        re_sigma_values = []
        for delta in delta1_values:
            def im_sigma(omega):
                p = 1j * omega
                expr = np.sqrt(1 + delta * p)
                coth_arg = lambda2 * p / expr
                coth_arg = np.clip(coth_arg, -100, 100)
                coth = (np.exp(2 * coth_arg) + 1) / (np.exp(2 * coth_arg) - 1)
                sigma = -p - lambda1 * expr * coth
                return sigma.imag

            omega_cross = None
            for bracket in [(500, 2000), (2000, 5000), (5000, 10000), (10000, 20000)]:
                try:
                    sol = root_scalar(im_sigma, bracket=bracket, method='brentq')
                    if sol.converged:
                        omega_cross = sol.root
                        break
                except:
                    continue

            if omega_cross is not None:
                p_cross = 1j * omega_cross
                expr = np.sqrt(1 + delta * p_cross)
                coth_arg = lambda2 * p_cross / expr
                coth_arg = np.clip(coth_arg, -100, 100)
                coth = (np.exp(2 * coth_arg) + 1) / (np.exp(2 * coth_arg) - 1)
                sigma = -p_cross - lambda1 * expr * coth
                re_sigma_values.append(sigma.real)
            else:
                re_sigma_values.append(np.nan)

        ax.plot(np.array(multiplier_list) * delta_base * 1e6, re_sigma_values, marker='o')
        ax.set_xlabel('δ₁ (×10⁻⁶ с)')
        ax.set_ylabel('Re(σ)')
        ax.set_title(f'Диаграмма устойчивости при L = {L} м')
        ax.grid(True)
        self.stability_canvas.draw()
            
    def show_interactive(self):
        """Интерактивная визуализация с улучшенным масштабированием"""
        try:
            dialog = QDialog(self)
            dialog.setWindowTitle("Интерактивный режим с масштабированием")
            dialog.setModal(True)
            dialog.resize(1200, 900)
            
            layout = QVBoxLayout(dialog)
            tabs = QTabWidget()
            
            # Вкладка крутильных колебаний
            torsional_tab = QWidget()
            torsional_layout = QVBoxLayout(torsional_tab)
            torsional_fig = Figure(figsize=(10, 6))
            torsional_canvas = FigureCanvas(torsional_fig)
            torsional_ax = torsional_fig.add_subplot(111)
            
            # Слайдеры
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
                    l = length_slider.value()
                    delta = delta_slider.value() * 1e-6
                    
                    params = self.get_current_parameters()
                    params['length'] = l
                    params['delta1'] = delta
                    
                    result = self.model.calculate_torsional(params)
                    
                    line.set_data(result['sigma_real'], result['sigma_imag'])
                    torsional_ax.set_xlim(-15000, 500)
                    torsional_ax.set_ylim(-8000, 8000)
                    
                    if len(result['sigma_real']) > 0:
                        x_pad = 0.1 * (np.nanmax(result['sigma_real']) - np.nanmin(result['sigma_real']))
                        y_pad = 0.1 * (np.nanmax(result['sigma_imag']) - np.nanmin(result['sigma_imag']))
                        
                        x_min = np.nanmin(result['sigma_real']) - x_pad
                        x_max = np.nanmax(result['sigma_real']) + x_pad
                        y_min = np.nanmin(result['sigma_imag']) - y_pad
                        y_max = np.nanmax(result['sigma_imag']) + y_pad
                        
                        torsional_ax.set_xlim(x_min, x_max)
                        torsional_ax.set_ylim(y_min, y_max)
                        
                        torsional_current_xlim = [x_min, x_max]
                        torsional_current_ylim = [y_min, y_max]
                    
                    torsional_ax.set_title(
                        f'Крутильные колебания: L={l} м, δ₁={delta:.1e} с\n'
                        f'Re(σ)∈[{np.nanmin(result['sigma_real']):.1f}, {np.nanmax(result['sigma_real']):.1f}], '
                        f'Im(σ)∈[{np.nanmin(result['sigma_imag']):.1f}, {np.nanmax(result['sigma_imag']):.1f}]',
                        fontsize=10
                    )
                    
                    torsional_canvas.draw_idle()
                    
                except Exception as e:
                    print(f"Ошибка при обновлении графика: {str(e)}")
                    torsional_ax.clear()
                    torsional_ax.text(0.5, 0.5, 'Ошибка при построении графика', 
                                    ha='center', va='center')
                    torsional_canvas.draw_idle()
            
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
            
            # Вкладка продольных колебаний
            longitudinal_tab = QWidget()
            longitudinal_layout = QVBoxLayout(longitudinal_tab)
            longitudinal_fig = Figure(figsize=(10, 6))
            longitudinal_canvas = FigureCanvas(longitudinal_fig)
            longitudinal_ax = longitudinal_fig.add_subplot(111)

            # --- Вкладка поперечных колебаний ---
            transverse_tab = QWidget()
            transverse_layout = QVBoxLayout(transverse_tab)
            transverse_fig = Figure(figsize=(10, 6))
            transverse_canvas = FigureCanvas(transverse_fig)
            transverse_ax = transverse_fig.add_subplot(111)

            # Слайдеры для поперечных колебаний
            trans_length_slider = QSlider(Qt.Horizontal)
            trans_length_slider.setRange(25, 60)     # 2.5 ... 6.0 м (умножим на 0.1)
            trans_length_slider.setValue(25)
            trans_length_slider.setTickInterval(5)
            trans_length_slider.setTickPosition(QSlider.TicksBelow)

            trans_K_slider = QSlider(Qt.Horizontal)
            trans_K_slider.setRange(1, 10)          # 1e5 .. 1e6 Н/м
            trans_K_slider.setValue(6)

            trans_mu_slider = QSlider(Qt.Horizontal)
            trans_mu_slider.setRange(1, 30)         # 0.01 .. 0.30
            trans_mu_slider.setValue(10)

            trans_tau_slider = QSlider(Qt.Horizontal)
            trans_tau_slider.setRange(10, 200)      # 0.01 .. 0.20 c
            trans_tau_slider.setValue(60)

            # Линия графика
            trans_line, = transverse_ax.plot([], [], 'b-', linewidth=1.5)
            transverse_ax.grid(True, which='both', linestyle=':', alpha=0.7)
            transverse_ax.set_title('Поперечные колебания: D-разбиение')
            transverse_ax.set_xlabel('Re(W)')
            transverse_ax.set_ylabel('Im(W)')
            
            mu_slider = QSlider(Qt.Horizontal)
            mu_slider.setRange(1, 50)
            mu_slider.setValue(10)
            
            tau_slider = QSlider(Qt.Horizontal)
            tau_slider.setRange(10, 200)
            tau_slider.setValue(60)
            
            line_long, = longitudinal_ax.plot([], [], 'b-', linewidth=1.5)
            longitudinal_ax.grid(True)
            longitudinal_ax.set_title('Кривая D-разбиения для продольных колебаний')
            longitudinal_ax.set_xlabel('K₁, МН/м')
            longitudinal_ax.set_ylabel('δ, кН·с/м')
            
            longitudinal_original_xlim = (0, 20)
            longitudinal_original_ylim = (-150, 50)
            longitudinal_ax.set_xlim(longitudinal_original_xlim)
            longitudinal_ax.set_ylim(longitudinal_original_ylim)
            
            longitudinal_current_xlim = list(longitudinal_original_xlim)
            longitudinal_current_ylim = list(longitudinal_original_ylim)
            longitudinal_press = None

            def update_transverse():
                """Обновление кривой D-разбиения для поперечных колебаний."""
                # переводим значения слайдеров в физические
                L = trans_length_slider.value() / 10.0          # 2.5..6.0 м
                K_cut = trans_K_slider.value() * 1e5            # 1e5..1e6 Н/м
                mu = trans_mu_slider.value() / 100.0            # 0.01..0.30
                tau = trans_tau_slider.value() / 1000.0         # 0.01..0.20 c

                # Берём остальные параметры из текущей формы
                params = self.get_current_parameters()
                params['length'] = L
                params['K_cut'] = K_cut
                params['mu'] = mu
                params['tau'] = tau

                try:
                    result = self.model.calculate_transverse(params)

                    if len(result['W_real']) > 0:
                        trans_line.set_data(result['W_real'], result['W_imag'])

                        # Автомасштаб
                        transverse_ax.relim()
                        transverse_ax.autoscale_view()
                        transverse_ax.margins(x=0.05, y=0.05)

                        transverse_ax.set_title(
                            f'Поперечные колебания: L={L:.2f} м, '
                            f'K={K_cut/1e5:.1f}·10⁵ Н/м, '
                            f'μ={mu:.2f}, τ={tau*1e3:.0f} мс',
                            fontsize=10
                        )
                    else:
                        transverse_ax.clear()
                        transverse_ax.set_title('Поперечные колебания: нет данных')
                        transverse_ax.text(
                            0.5, 0.5,
                            'Нет данных для построения\nПроверьте параметры',
                            ha='center', va='center',
                            transform=transverse_ax.transAxes
                        )

                    transverse_canvas.draw_idle()
                except Exception as e:
                    print(f"[ERROR] transverse update: {e}")
                    transverse_ax.clear()
                    transverse_ax.text(
                        0.5, 0.5,
                        'Ошибка при вычислениях',
                        ha='center', va='center',
                        transform=transverse_ax.transAxes
                    )
                    transverse_canvas.draw_idle()
            
            def on_scroll_longitudinal(event):
                if event.inaxes != longitudinal_ax:
                    return
                    
                scale_factor = 1.2 if event.button == 'up' else 1/1.2
                
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
                nonlocal longitudinal_press
                if event.inaxes != longitudinal_ax or event.button != MouseButton.LEFT:
                    return
                longitudinal_press = event.xdata, event.ydata
            
            def on_release_longitudinal(event):
                nonlocal longitudinal_press
                longitudinal_press = None
                longitudinal_canvas.draw()
            
            def on_motion_longitudinal(event):
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
                longitudinal_current_xlim[:] = list(longitudinal_original_xlim)
                longitudinal_current_ylim[:] = list(longitudinal_original_ylim)
                longitudinal_ax.set_xlim(longitudinal_current_xlim)
                longitudinal_ax.set_ylim(longitudinal_current_ylim)
                longitudinal_canvas.draw()
            
            def update_longitudinal():
                mu = mu_slider.value() / 100
                tau = tau_slider.value() * 1e-3

                params = self.get_current_parameters()
                params['mu'] = mu
                params['tau'] = tau

                result = self.model.calculate_longitudinal(params)

                if len(result['K1']) > 0:
                    K1_MN = result['K1'] / 1e6  # МН/м
                    delta_kNs_per_m = result['delta'] / 1e3  # кН·с/м

                    line_long.set_data(K1_MN, delta_kNs_per_m)

                    # --- Заголовок с параметрами и диапазонами ---
                    longitudinal_ax.set_title(
                        f'Продольные колебания: μ={mu:.2f}, τ={tau*1e3:.0f} мс\n'
                        f'K₁ ∈ [{np.nanmin(K1_MN):.2f}, {np.nanmax(K1_MN):.2f}] МН/м, '
                        f'δ ∈ [{np.nanmin(delta_kNs_per_m):.2f}, {np.nanmax(delta_kNs_per_m):.2f}] кН·с/м',
                        fontsize=10
                    )

                    longitudinal_canvas.draw()
                else:
                    longitudinal_ax.clear()
                    longitudinal_ax.set_title('Продольные колебания: нет данных для построения')
                    longitudinal_ax.text(0.5, 0.5, 'Нет данных для построения', 
                                        ha='center', va='center', transform=longitudinal_ax.transAxes)
                    longitudinal_canvas.draw()
            
            longitudinal_canvas.mpl_connect('scroll_event', on_scroll_longitudinal)
            longitudinal_canvas.mpl_connect('button_press_event', on_press_longitudinal)
            longitudinal_canvas.mpl_connect('button_release_event', on_release_longitudinal)
            longitudinal_canvas.mpl_connect('motion_notify_event', on_motion_longitudinal)
            
            mu_slider.valueChanged.connect(update_longitudinal)
            tau_slider.valueChanged.connect(update_longitudinal)

            trans_length_slider.valueChanged.connect(update_transverse)
            trans_K_slider.valueChanged.connect(update_transverse)
            trans_mu_slider.valueChanged.connect(update_transverse)
            trans_tau_slider.valueChanged.connect(update_transverse)

            transverse_layout.addWidget(transverse_canvas)
            transverse_layout.addWidget(QLabel("Длина борштанги L (м, ×0.1):"))
            transverse_layout.addWidget(trans_length_slider)
            transverse_layout.addWidget(QLabel("Дин. жёсткость резания K (×10⁵ Н/м):"))
            transverse_layout.addWidget(trans_K_slider)
            transverse_layout.addWidget(QLabel("Коэффициент μ (×0.01):"))
            transverse_layout.addWidget(trans_mu_slider)
            transverse_layout.addWidget(QLabel("Время запаздывания τ (мс):"))
            transverse_layout.addWidget(trans_tau_slider)
            
            longitudinal_layout.addWidget(longitudinal_canvas)
            longitudinal_layout.addWidget(QLabel("Коэффициент трения μ (x0.01):"))
            longitudinal_layout.addWidget(mu_slider)
            longitudinal_layout.addWidget(QLabel("Время запаздывания τ (мс):"))
            longitudinal_layout.addWidget(tau_slider)
            
            control_buttons = QHBoxLayout()
            
            reset_zoom_torsional_btn = QPushButton("Сбросить масштаб (Крутильные)")
            reset_zoom_torsional_btn.clicked.connect(reset_torsional_zoom)
            
            reset_zoom_longitudinal_btn = QPushButton("Сбросить масштаб (Продольные)")
            reset_zoom_longitudinal_btn.clicked.connect(reset_longitudinal_zoom)
            
            control_buttons.addWidget(reset_zoom_torsional_btn)
            control_buttons.addWidget(reset_zoom_longitudinal_btn)
            
            tabs.addTab(torsional_tab, "Крутильные колебания")
            tabs.addTab(longitudinal_tab, "Продольные колебания")
            tabs.addTab(transverse_tab, "Поперечные колебания")

            layout.addWidget(tabs)
            layout.addLayout(control_buttons)
            
            close_btn = QPushButton("Закрыть")
            close_btn.clicked.connect(dialog.close)
            layout.addWidget(close_btn)
            
            update_torsional()
            update_transverse()
            update_longitudinal()
            
            dialog.exec_()

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось запустить интерактивный режим: {str(e)}")