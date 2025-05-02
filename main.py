import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.optimize import root_scalar
from matplotlib.gridspec import GridSpec
from colorama import init, Fore, Back, Style
import sys
import time
import os
from functools import lru_cache
import unittest
import json
import csv

# Инициализация colorama для цветного вывода
init(autoreset=True)

class SystemParameters:
    """Класс для хранения параметров системы"""
    def __init__(self):
        # Материальные свойства
        self.rho = 7800       # Плотность материала, кг/м³
        self.G = 8e10         # Модуль сдвига, Па
        self.E = 200e9        # Модуль Юнга, Па
        self.Jr = 2.57e-2     # Момент инерции стержня, кг·м²
        self.Jp = 1.9e-5      # Полярный момент инерции поперечного сечения, м⁴
        
        # Геометрические параметры
        self.lengths = np.array([2.5, 3, 4, 5, 6])  # Длины борштанги, м
        self.S = 2e-4         # Площадь поперечного сечения, м²
        self.delta1_base = 3.44e-6  # Базовое значение коэффициента трения, с
        self.multipliers = np.array([1, 2, 3, 4, 6, 10])  # Множители для delta1
        self.mu_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # Коэф. внутреннего трения
        self.tau = 60e-3      # Время запаздывания, с

    def save_to_json(self, filename):
        """Сохранение параметров в JSON файл"""
        params_dict = {
            'rho': self.rho,
            'G': self.G,
            'E': self.E,
            'Jr': self.Jr,
            'Jp': self.Jp,
            'lengths': self.lengths.tolist(),
            'S': self.S,
            'delta1_base': self.delta1_base,
            'multipliers': self.multipliers.tolist(),
            'mu_values': self.mu_values.tolist(),
            'tau': self.tau
        }
        with open(filename, 'w') as f:
            json.dump(params_dict, f, indent=4)

    @classmethod
    def load_from_json(cls, filename):
        """Загрузка параметров из JSON файла"""
        with open(filename, 'r') as f:
            params_dict = json.load(f)
        
        params = cls()
        params.rho = params_dict['rho']
        params.G = params_dict['G']
        params.E = params_dict['E']
        params.Jr = params_dict['Jr']
        params.Jp = params_dict['Jp']
        params.lengths = np.array(params_dict['lengths'])
        params.S = params_dict['S']
        params.delta1_base = params_dict['delta1_base']
        params.multipliers = np.array(params_dict['multipliers'])
        params.mu_values = np.array(params_dict['mu_values'])
        params.tau = params_dict['tau']
        
        return params

class BoreBarAnalysis:
    """Базовый класс для анализа колебаний борштанги"""
    def __init__(self, params):
        self.params = params
        self.results = {}
        self.figures = {}
    
    @lru_cache(maxsize=100)
    def _cached_computation(self, *args):
        """Кэшируемые вычисления для оптимизации"""
        pass
    
    def validate_parameters(self):
        """Проверка корректности параметров"""
        if np.any(self.params.lengths <= 0):
            raise ValueError("Длины должны быть положительными")
        if self.params.rho <= 0 or self.params.G <= 0 or self.params.E <= 0:
            raise ValueError("Материальные параметры должны быть положительными")
        return True
    
    def export_results(self, filename, format='json'):
        """Сохранение результатов в файл с обработкой numpy массивов"""
        try:
            # Преобразуем результаты в сериализуемый формат
            serializable_results = {}
            for key, value in self.results.items():
                if isinstance(value, dict):
                    serializable_results[key] = {
                        k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in value.items()
                    }
                else:
                    serializable_results[key] = value.tolist() if isinstance(value, np.ndarray) else value

            if format == 'json':
                with open(f"{filename}.json", 'w') as f:
                    json.dump(serializable_results, f, indent=4)
            elif format == 'csv':
                # Для CSV сохраняем только основные параметры
                with open(f"{filename}.csv", 'w', newline='') as f:
                    writer = csv.writer(f)
                    for key, value in serializable_results.items():
                        if isinstance(value, dict):
                            for subkey, subvalue in value.items():
                                if isinstance(subvalue, list):
                                    writer.writerow([f"{key}_{subkey}"] + subvalue)
                                else:
                                    writer.writerow([f"{key}_{subkey}", subvalue])
                        else:
                            writer.writerow([key, value])
            else:
                raise ValueError("Неподдерживаемый формат экспорта")
                
            print(Fore.GREEN + f"Результаты сохранены в {filename}.{format}")
        except Exception as e:
            print(Fore.RED + f"Ошибка при сохранении: {str(e)}")
            raise

class TorsionalAnalysis(BoreBarAnalysis):
    """Анализ крутильных колебаний"""
    def compute(self):
        """
        Анализ крутильных колебаний борштанги
        
        Реализует математическую модель:
        ∂²φ/∂t² = (G/ρ)∂²φ/∂x² + δ₁∂³φ/∂x²∂t
        
        Возвращает:
        - fig : matplotlib.figure.Figure
            Фигура с графиками анализа
        """
        self.validate_parameters()
        
        # Вычисляем константы
        lambda1 = np.sqrt(self.params.rho * self.params.G) * self.params.Jp / self.params.Jr
        lambda2 = self.params.lengths * np.sqrt(self.params.rho / self.params.G)
        delta1_values = self.params.delta1_base * self.params.multipliers
        
        # Создаем фигуру
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 2, figure=fig)
        
        # 1. Кривая D-разбиения
        ax1 = fig.add_subplot(gs[0, 0])
        omega = np.linspace(1000, 15000, 1000)
        p = 1j * omega
        with np.errstate(all='ignore'):
            expr = np.sqrt(1 + delta1_values[0] * p)
            sigma = -p - lambda1 * expr * (1 / np.tanh(lambda2[0] * p / expr))
            sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e10, neginf=-1e10)
        
        ax1.plot(sigma.real, sigma.imag, 'b-', linewidth=1.5)
        ax1.axhline(0, color='red', linestyle='--', linewidth=0.7)
        ax1.axvline(0, color='red', linestyle='--', linewidth=0.7)
        ax1.set_title('Кривая D-разбиения для крутильных колебаний (L=2.5 м)', fontsize=12)
        ax1.set_xlabel('Re(σ)', fontsize=10)
        ax1.set_ylabel('Im(σ)', fontsize=10)
        ax1.grid(True, which='both', linestyle=':', alpha=0.7)
        ax1.set_xlim(-15000, 500)
        ax1.set_ylim(-8000, 8000)
        
        # 2. Диаграмма устойчивости
        ax2 = fig.add_subplot(gs[0, 1])
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.params.lengths)))
        
        for i, l in enumerate(self.params.lengths):
            Sigma = np.zeros(len(delta1_values))
            
            for j, delta in enumerate(delta1_values):
                def im_sigma(omega):
                    p = 1j * omega
                    with np.errstate(all='ignore'):
                        sqrt_expr = np.sqrt(1 + delta * p)
                        cth = 1 / np.tanh(lambda2[i] * p / sqrt_expr)
                        val = -p - lambda1 * sqrt_expr * cth
                        return val.imag
                
                try:
                    omega_sol = root_scalar(im_sigma, bracket=[500, 2000], method='brentq').root
                    p = 1j * omega_sol
                    with np.errstate(all='ignore'):
                        sqrt_expr = np.sqrt(1 + delta * p)
                        cth = 1 / np.tanh(lambda2[i] * p / sqrt_expr)
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
        ax2.set_xlabel('Коэффициент внутреннего трения δ₁ (с)', fontsize=10)
        ax2.set_ylabel('Re(σ)', fontsize=10)
        ax2.set_title('Диаграмма устойчивости крутильных колебаний', fontsize=12)
        ax2.legend(fontsize=9)
        ax2.grid(True, which='both', linestyle=':', alpha=0.7)
        ax2.text(1e-5, -1000, 'Устойчивая область', color='green', fontsize=10)
        
        self.results['torsional'] = {
            'lambda1': lambda1,
            'lambda2': lambda2,
            'delta1_values': delta1_values,
            'sigma': sigma
        }
        self.figures['torsional'] = fig
        return fig
    
    def interactive_plot(self):
        """Интерактивная визуализация крутильных колебаний"""
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.subplots_adjust(bottom=0.25)
        
        # Создать слайдеры
        ax_length = plt.axes([0.2, 0.1, 0.6, 0.03])
        ax_delta = plt.axes([0.2, 0.05, 0.6, 0.03])
        
        length_slider = Slider(ax_length, 'Длина (м)', 2, 6, valinit=2.5)
        delta_slider = Slider(ax_delta, 'δ₁', 1e-6, 1e-4, valinit=3.44e-6)
        
        # Инициализация графика
        omega = np.linspace(1000, 15000, 1000)
        line, = ax.plot([], [], 'b-', linewidth=1.5)
        ax.set_xlim(-15000, 500)
        ax.set_ylim(-8000, 8000)
        ax.grid(True)
        
        def update(val):
            l = length_slider.val
            delta = delta_slider.val
            
            lambda1 = np.sqrt(self.params.rho * self.params.G) * self.params.Jp / self.params.Jr
            lambda2 = l * np.sqrt(self.params.rho / self.params.G)
            
            p = 1j * omega
            with np.errstate(all='ignore'):
                expr = np.sqrt(1 + delta * p)
                sigma = -p - lambda1 * expr * (1 / np.tanh(lambda2 * p / expr))
                sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e10, neginf=-1e10)
            
            line.set_data(sigma.real, sigma.imag)
            fig.canvas.draw_idle()
        
        length_slider.on_changed(update)
        delta_slider.on_changed(update)
        
        plt.show()
        return fig

class LongitudinalAnalysis(BoreBarAnalysis):
    """Анализ продольных колебаний с интерактивной визуализацией"""
    
    def compute(self, interactive=False):
        """
        Анализ продольных колебаний с возможностью интерактивного режима
        
        Parameters:
        -----------
        interactive : bool
            Если True, создает интерактивный виджет
        """
        if interactive:
            return self.interactive_plot()
        return self.static_plot()
    
    def static_plot(self):
        """Статическая визуализация продольных колебаний"""
        # Параметры системы
        E = 200e9
        S = 2
        rho = 7874
        mu = 0.1
        tau = 60
        l = 4
        
        a = np.sqrt(E/rho)
        omega = np.linspace(0.001, 0.4, 10000)
        
        with np.errstate(all='ignore'):
            K1 = (E*S/a**2) * omega * (1/np.tan(omega*l/a)) / (1 - mu*np.cos(omega*tau))
            delta = -(E*S*mu/a**2) * (1/np.tan(omega*l/a)) * np.sin(omega*tau) / (1 - mu*np.cos(omega*tau))
            
            valid = ~(np.isnan(K1) | np.isinf(K1) | np.isnan(delta) | np.isinf(delta))
            K1 = K1[valid]
            delta = delta[valid]
        
        K1_0 = -E*S/(-l + l*mu)
        delta_0 = tau*E*S*mu/(-l + l*mu)

        # Создаем фигуру
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Построение кривой
        ax.plot(K1/1e6, delta/1e6, 'b-', linewidth=1.5, label='Кривая D-разбиения')
        ax.plot(K1_0/1e6, delta_0/1e6, 'r*', markersize=12, label='Особая точка (ω→0)')
        
        # Настройка осей с запасом
        self._auto_scale_plot(ax, K1/1e6, delta/1e6)
        
        # Подписи и оформление
        ax.set_xlabel('K₁, МН/м', fontsize=12)
        ax.set_ylabel('δ, МН·с/м', fontsize=12) 
        ax.set_title('Кривая D-разбиения для продольных колебаний', fontsize=14, pad=20)
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(fontsize=10, loc='upper right')
        ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
        ax.axvline(0, color='k', linestyle='--', linewidth=0.5)
        
        # Сохраняем результаты
        self.results['longitudinal'] = {
            'K1': K1,
            'delta': delta,
            'K1_0': K1_0,
            'delta_0': delta_0
        }
        self.figures['longitudinal'] = fig
        
        return fig
    
    def interactive_plot(self):
        """Интерактивная визуализация с регуляторами параметров"""
        fig = plt.figure(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.3)
        
        # Основная область для графика
        ax = fig.add_subplot(111)
        
        # Создаем слайдеры
        ax_mu = plt.axes([0.25, 0.25, 0.6, 0.03])
        ax_tau = plt.axes([0.25, 0.20, 0.6, 0.03])
        ax_length = plt.axes([0.25, 0.15, 0.6, 0.03])
        ax_E = plt.axes([0.25, 0.10, 0.6, 0.03])
        
        mu_slider = Slider(ax_mu, 'Коэф. трения μ', 0.01, 0.5, valinit=0.1)
        tau_slider = Slider(ax_tau, 'Время запаздывания τ (с)', 10, 200, valinit=60)
        length_slider = Slider(ax_length, 'Длина L (м)', 0.5, 10, valinit=4)
        E_slider = Slider(ax_E, 'Модуль Юнга E (ГПа)', 50, 400, valinit=200)
        
        # Инициализация графика
        line, = ax.plot([], [], 'b-', linewidth=1.5, label='Кривая D-разбиения')
        point, = ax.plot([], [], 'ro', markersize=8, label='Особая точка')
        ax.set_xlabel('K₁, МН/м', fontsize=12)
        ax.set_ylabel('δ, МН·с/м', fontsize=12)
        ax.set_title('Интерактивная кривая D-разбиения', fontsize=14)
        ax.grid(True)
        ax.legend()
        
        def update(val):
            """Обновление графика при изменении параметров"""
            params = {
                'mu': mu_slider.val,
                'tau': tau_slider.val,
                'L': length_slider.val,
                'E': E_slider.val * 1e9
            }
            
            K1, delta, K1_0, delta_0 = self._calculate(params)
            
            line.set_data(K1/1e6, delta/1e6)
            point.set_data([K1_0/1e6], [delta_0/1e6])
            self._auto_scale_plot(ax, K1/1e6, delta/1e6)
            fig.canvas.draw_idle()
        
        # Привязываем обновление
        for slider in [mu_slider, tau_slider, length_slider, E_slider]:
            slider.on_changed(update)
        
        # Первоначальное обновление
        update(None)
        return fig
    
    def _calculate(self, params):
        """Вычисляет параметры для продольных колебаний"""
        E = params['E']
        S = 2
        rho = 7874
        mu = params['mu']
        tau = params['tau']
        L = params['L']
        
        a = np.sqrt(E/rho)
        omega = np.linspace(0.001, 0.4, 2000)
        
        with np.errstate(all='ignore'):
            K1 = (E*S/a**2) * omega * (1/np.tan(omega*L/a)) / (1 - mu*np.cos(omega*tau))
            delta = -(E*S*mu/a**2) * (1/np.tan(omega*L/a)) * np.sin(omega*tau) / (1 - mu*np.cos(omega*tau))
            valid = ~(np.isnan(K1) | np.isinf(K1) | np.isnan(delta) | np.isinf(delta))
            K1 = K1[valid]
            delta = delta[valid]
        
        K1_0 = -E*S/(-L + L*mu)
        delta_0 = tau*E*S*mu/(-L + L*mu)
        return K1, delta, K1_0, delta_0
    
    def _auto_scale_plot(self, ax, x_data, y_data):
        """Автоматическое масштабирование графика"""
        if len(x_data) == 0 or len(y_data) == 0:
            return
        
        x_pad = 0.1 * (np.max(x_data) - np.min(x_data)) if (np.max(x_data) - np.min(x_data)) > 0 else 1
        y_pad = 0.1 * (np.max(y_data) - np.min(y_data)) if (np.max(y_data) - np.min(y_data)) > 0 else 1
        
        ax.set_xlim(np.min(x_data)-x_pad, np.max(x_data)+x_pad)
        ax.set_ylim(np.min(y_data)-y_pad, np.max(y_data)+y_pad)
        ax.relim()
        ax.autoscale_view()
    
    def show(self, interactive=False):
        """Отображение графика"""
        fig = self.compute(interactive)
        plt.show()
        return fig

class ComparativeAnalysis(BoreBarAnalysis):
    """Сравнительный анализ колебаний"""
    def compute(self):
        """
        Сравнительный анализ крутильных и продольных колебаний
        
        Возвращает:
        - fig : matplotlib.figure.Figure
            Фигура с графиками сравнения
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. Сравнение частот
        lengths = np.linspace(2, 6, 20)
        
        # Крутильные частоты
        torsional_freq = np.sqrt(self.params.G/self.params.rho) * np.pi / lengths
        
        # Продольные частоты
        longitudinal_freq = np.sqrt(self.params.E/self.params.rho) * np.pi / lengths
        
        ax1.plot(lengths, torsional_freq/1000, 'b-o', label='Крутильные', linewidth=1.5, markersize=4)
        ax1.plot(lengths, longitudinal_freq/1000, 'r-s', label='Продольные', linewidth=1.5, markersize=4)
        ax1.set_xlabel('Длина борштанги, м', fontsize=12)
        ax1.set_ylabel('Частота (кГц)', fontsize=12)
        ax1.set_title('Сравнение собственных частот колебаний', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, linestyle=':', alpha=0.7)
        
        # 2. Сравнение устойчивости
        stability_ratio = 1 / lengths**2
        ax2.plot(lengths, stability_ratio/stability_ratio.max(), 'g-^', linewidth=2, markersize=6)
        ax2.set_xlabel('Длина борштанги, м', fontsize=12)
        ax2.set_ylabel('Относительная устойчивость', fontsize=12)
        ax2.set_title('Влияние длины на устойчивость системы', fontsize=14)
        ax2.grid(True, linestyle=':', alpha=0.7)
        ax2.text(3.5, 0.7, 'Уменьшение длины\nувеличивает устойчивость', 
                 fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        self.results['comparative'] = {
            'torsional_freq': torsional_freq,
            'longitudinal_freq': longitudinal_freq,
            'stability_ratio': stability_ratio
        }
        self.figures['comparative'] = fig
        return fig

class BoreBarUI:
    """Класс пользовательского интерфейса"""
    def __init__(self):
        self.params = SystemParameters()
        self.results = {}
        self.analyzers = {
            'torsional': TorsionalAnalysis(self.params),
            'longitudinal': LongitudinalAnalysis(self.params),
            'comparative': ComparativeAnalysis(self.params)
        }
        
    def display_header(self):
        """Отображение заголовка программы"""
        print(Fore.YELLOW + "="*60)
        print(Fore.CYAN + " ИССЛЕДОВАТЕЛЬСКАЯ СИСТЕМА АНАЛИЗА БОРШТАНГИ")
        print(Fore.YELLOW + "="*60)
        print(Fore.GREEN + "Автоматический анализ крутильных и продольных колебаний")
        print(Style.RESET_ALL)
        
    def show_menu(self):
        """Отображение главного меню"""
        while True:
            print("\n" + Fore.BLUE + "ГЛАВНОЕ МЕНЮ:")
            print(Fore.WHITE + "1. Исследование крутильных колебаний")
            print("2. Исследование продольных колебаний")
            print("3. Сравнительный анализ")
            print("4. Показать параметры системы")
            print("5. Сохранить результаты")
            print("6. Интерактивная визуализация")
            print("7. Выход")
            
            choice = input(Fore.YELLOW + "Выберите действие (1-7): " + Style.RESET_ALL)
            
            if choice == "1":
                self.run_analysis("torsional")
            elif choice == "2":
                self.run_analysis("longitudinal")
            elif choice == "3":
                self.run_analysis("comparative")
            elif choice == "4":
                self.show_parameters()
            elif choice == "5":
                self.save_results()
            elif choice == "6":
                self.interactive_visualization()
            elif choice == "7":
                self.exit_program()
            else:
                print(Fore.RED + "Неверный ввод! Пожалуйста, выберите 1-7")
    
    def run_analysis(self, analysis_type):
        """Запуск выбранного анализа"""
        try:
            print(Fore.GREEN + f"\nЗапуск {analysis_type} анализа..." + Style.RESET_ALL)
            
            # Простой прогресс-бар
            def simple_progress():
                for i in range(10):
                    time.sleep(0.05)
                    print(Fore.BLUE + "■" * (i+1) + " " * (9-i) + f" {10*(i+1)}%", end='\r')
                print(Fore.GREEN + "Анализ завершен!          ")
            
            simple_progress()
            
            analyzer = self.analyzers[analysis_type]
            fig = analyzer.compute()
            self.results[analysis_type] = analyzer.results[analysis_type]
            
            print(Fore.GREEN + "Отображение результатов...")
            plt.show()
            
        except KeyboardInterrupt:
            print(Fore.RED + "\nАнализ прерван пользователем!")
        except Exception as e:
            print(Fore.RED + f"\nОшибка: {str(e)}")
    
    def show_parameters(self):
        """Показать параметры системы"""
        print(Fore.CYAN + "\nТЕКУЩИЕ ПАРАМЕТРЫ СИСТЕМЫ:")
        print(Fore.WHITE + f"1. Плотность материала: {self.params.rho} кг/м³")
        print(f"2. Модуль сдвига: {self.params.G} Па")
        print(f"3. Модуль Юнга: {self.params.E} Па")
        print(f"4. Площадь сечения: {self.params.S} м²")
        print(f"5. Длины борштанги: {self.params.lengths} м")
        print(f"6. Коэффициент трения: {self.params.delta1_base} (базовый)")
        print(f"7. Множители трения: {self.params.multipliers}")
    
    def save_results(self):
        """Сохранение результатов всех анализов"""
        if not self.results:
            print(Fore.RED + "Нет результатов для сохранения!")
            return
            
        try:
            format_choice = input(Fore.YELLOW + "Выберите формат (csv/json): " + Style.RESET_ALL).lower()
            if format_choice not in ['csv', 'json']:
                raise ValueError("Поддерживаются только csv и json форматы")
                
            filename = input(Fore.YELLOW + "Введите имя файла (без расширения): " + Style.RESET_ALL)
            
            # Собираем все результаты из всех анализаторов
            all_results = {}
            for name, analyzer in self.analyzers.items():
                if hasattr(analyzer, 'results'):
                    all_results[name] = analyzer.results
            
            # Сохраняем все результаты
            if format_choice == 'json':
                with open(f"{filename}.json", 'w') as f:
                    json.dump(
                        {
                            k: {
                                k2: v2.tolist() if isinstance(v2, np.ndarray) else v2
                                for k2, v2 in v.items()
                            } 
                            for k, v in all_results.items()
                        }, 
                        f, 
                        indent=4
                    )
            else:  # CSV
                with open(f"{filename}.csv", 'w', newline='') as f:
                    writer = csv.writer(f)
                    for analyzer_name, results in all_results.items():
                        for result_name, values in results.items():
                            if isinstance(values, dict):
                                for key, val in values.items():
                                    if isinstance(val, (list, np.ndarray)):
                                        writer.writerow([f"{analyzer_name}_{result_name}_{key}"] + list(val))
                                    else:
                                        writer.writerow([f"{analyzer_name}_{result_name}_{key}", val])
                            else:
                                writer.writerow([f"{analyzer_name}_{result_name}", values])
            
            print(Fore.GREEN + f"Все результаты сохранены в {filename}.{format_choice}")
        except Exception as e:
            print(Fore.RED + f"Ошибка при сохранении: {str(e)}")
            
    def interactive_visualization(self):
        """Интерактивная визуализация"""
        print("\n" + Fore.CYAN + "ИНТЕРАКТИВНАЯ ВИЗУАЛИЗАЦИЯ")
        print(Fore.WHITE + "1. Крутильные колебания")
        print("2. Продольные колебания")
        
        choice = input(Fore.YELLOW + "Выберите тип анализа (1-2): " + Style.RESET_ALL)
        
        if choice == "1":
            fig = self.analyzers['torsional'].interactive_plot()
            plt.show()  # Изменено здесь
        elif choice == "2":
            fig = self.analyzers['longitudinal'].interactive_plot()
            plt.show()  # Изменено здесь
        else:
            print(Fore.RED + "Неверный выбор!")
    
    def exit_program(self):
        """Выход из программы"""
        print(Fore.YELLOW + "\nЗавершение работы программы...")
        sys.exit(0)

class TestBoreBarAnalysis(unittest.TestCase):
    """Тесты для анализа борштанги"""
    def setUp(self):
        self.params = SystemParameters()
        self.torsional = TorsionalAnalysis(self.params)
        self.longitudinal = LongitudinalAnalysis(self.params)
    
    def test_parameters_validation(self):
        """Тест проверки параметров"""
        self.assertTrue(self.torsional.validate_parameters())
        
        with self.assertRaises(ValueError):
            self.params.rho = -1
            self.torsional.validate_parameters()
    
    def test_torsional_computation(self):
        """Тест вычислений для крутильных колебаний"""
        result = self.torsional.compute()
        self.assertIsNotNone(result)
        self.assertIn('torsional', self.torsional.results)
    
    def test_longitudinal_computation(self):
        """Тест вычислений для продольных колебаний"""
        result = self.longitudinal.compute()
        self.assertIsNotNone(result)
        self.assertIn('longitudinal', self.longitudinal.results)

if __name__ == "__main__":
    try:
        # Запуск тестов
        if len(sys.argv) > 1 and sys.argv[1] == 'test':
            unittest.main(argv=sys.argv[:1])
        else:
            ui = BoreBarUI()
            ui.display_header()
            ui.show_menu()
    except KeyboardInterrupt:
        print(Fore.RED + "\nПрограмма прервана пользователем!")
        sys.exit(1)
    except Exception as e:
        print(Fore.RED + f"\nКритическая ошибка: {str(e)}")
        sys.exit(1)