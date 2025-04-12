import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

# Параметры системы
rho = 7800       # Плотность материала, кг/м³
G = 8e10         # Модуль сдвига, Па
Jr = 2.57e-2     # Момент инерции стержня, кг·м²
Jp = 1.9e-5      # Полярный момент инерции поперечного сечения, м⁴
delta1_base = 3.44e-6  # Базовое значение коэффициента трения, с
lengths = np.array([2.5, 3, 4, 5, 6])  # Длины борштанги, м
multipliers = np.array([1, 2, 3, 4, 6, 10])  # Множители для delta1

def stability_analysis():
    # Вычисляем константы
    lambda1 = np.sqrt(rho * G) * Jp / Jr
    lambda2 = lengths * np.sqrt(rho / G)
    delta1_values = delta1_base * multipliers
    
    # 1. Построение кривой D-разбиения для L=2.5 м
    omega = np.linspace(1000, 15000, 1000)
    p = 1j * omega
    expr = np.sqrt(1 + delta1_values[0] * p)
    sigma = -p - lambda1 * expr * (1 / np.tanh(lambda2[0] * p / expr))
    
    plt.figure(figsize=(10, 6))
    plt.plot(sigma.real, sigma.imag, 'b-')
    plt.axhline(0, color='red', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='red', linestyle='--', linewidth=0.5)
    plt.title('Кривая D-разбиения для крутильных колебаний (L=2.5 м)', fontsize=14)
    plt.xlabel('Re(σ)', fontsize=12)
    plt.ylabel('Im(σ)', fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.show()
    
    # 2. Построение диаграммы устойчивости
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(lengths)))
    
    for i, l in enumerate(lengths):
        Sigma = np.zeros(len(delta1_values))
        
        for j, delta in enumerate(delta1_values):
            # Функция для нахождения корня Im(σ) = 0
            def im_sigma(omega):
                p = 1j * omega
                sqrt_expr = np.sqrt(1 + delta * p)
                cth = 1 / np.tanh(lambda2[i] * p / sqrt_expr)
                val = -p - lambda1 * sqrt_expr * cth
                return val.imag
            
            # Находим omega, где Im(σ) = 0
            try:
                omega_sol = root_scalar(im_sigma, bracket=[500, 2000], method='brentq').root
                p = 1j * omega_sol
                sqrt_expr = np.sqrt(1 + delta * p)
                cth = 1 / np.tanh(lambda2[i] * p / sqrt_expr)
                Sigma[j] = (-p - lambda1 * sqrt_expr * cth).real
            except:
                Sigma[j] = np.nan
        
        # Построение кривой для текущей длины
        valid = ~np.isnan(Sigma)
        plt.plot(delta1_values[valid], Sigma[valid], 'o-', color=colors[i], 
                label=f'L={l} м', markersize=4, linewidth=1.5)
    
    # Оформление графика устойчивости
    plt.axhline(0, color='k', linestyle='--', linewidth=0.8)
    plt.fill_between(delta1_values, -1500, 0, color='green', alpha=0.1)

    plt.xscale('log')
    plt.ylim(-1500, 100)
    plt.xlabel('Коэффициент внутреннего трения δ₁ (с)', fontsize=12)
    plt.ylabel('Re(σ)', fontsize=12)
    plt.title('Диаграмма устойчивости крутильных колебаний', fontsize=14)
    plt.legend()
    plt.grid(True, which='both', linestyle=':')
    plt.tight_layout()
    
    # Область устойчивости (Re(σ) < 0)
    plt.fill_betweenx([-2000, 0], delta1_values[0], delta1_values[-1], 
                     color='green', alpha=0.1)
    plt.text(1e-5, -1000, 'Устойчивая область', color='green', fontsize=12)
    
    plt.xscale('log')
    plt.ylim(-1500, 100)
    plt.tight_layout()
    plt.show()

stability_analysis()