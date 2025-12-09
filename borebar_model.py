"""
borebar_model.py

Модуль с математическими моделями колебаний борштанги:
- крутильные колебания (torsional);
- продольные колебания (longitudinal);
- поперечные колебания (transverse, модальная аппроксимация).

Файл максимально приближен к формулам из курсовой работы и исследований.
"""

import numpy as np
from scipy.optimize import root_scalar


class BoreBarModel:
    """
    Класс, инкапсулирующий математические модели для анализа устойчивости борштанги.

    Методы:
        - calculate_torsional(params):  крутильные колебания, кривая D-разбиения σ(p).
        - find_intersection(params):    поиск пересечения Im σ(p) = 0 (граница устойчивости).
        - calculate_longitudinal(params): продольные колебания, кривая K1–δ.
        - calculate_comparative(params): простой сравнительный анализ частот от длины.
        - calculate_transverse(params):  поперечные колебания, годограф W(p).
    """

    # -------------------------------------------------------------------------
    # Вспомогательные функции
    # -------------------------------------------------------------------------

    @staticmethod
    def _coth(z: np.ndarray) -> np.ndarray:
        """
        Численно устойчивое вычисление гиперболического котангенса:
            coth(z) = 1 / tanh(z)

        Работает и для комплексных аргументов.
        """
        with np.errstate(all="ignore"):
            return 1.0 / np.tanh(z)

    # -------------------------------------------------------------------------
    # Крутильные колебания
    # -------------------------------------------------------------------------

    @staticmethod
    def calculate_torsional(params: dict) -> dict:
        """
        Расчёт кривой D-разбиения для крутильных колебаний.

        Используется формула из теоретической части:
            σ(p) = -p - λ₁ * sqrt(1 + δ₁ p) * coth( λ₂ p / sqrt(1 + δ₁ p) ),  p = i ω

        где:
            λ₁ = sqrt(ρ G) * J_p / J_r,
            λ₂ = L * sqrt(ρ / G),
            δ₁ — коэффициент внутреннего трения (с учётом множителя).

        Параметры словаря params:
            rho      — плотность материала (кг/м³)
            G        — модуль сдвига (Па)
            Jp       — полярный момент инерции (м⁴)
            Jr       — момент инерции режущей головки (кг·м²)
            delta1   — базовый коэффициент δ₁ (с)
            multiplier — множитель для δ₁ (1, 2, 3, 4, 6, 10)
            length   — длина борштанги L (м)
        """
        rho = params["rho"]
        G = params["G"]
        Jp = params["Jp"]
        Jr = params["Jr"]
        delta1 = params["delta1"] * params["multiplier"]
        length = params["length"]

        # Параметры λ₁ и λ₂, как в работе
        lambda1 = np.sqrt(rho * G) * Jp / Jr
        lambda2 = length * np.sqrt(rho / G)

        # Диапазон частот для построения σ(p)
        omega = np.linspace(1000.0, 15000.0, 5000)  # рад/с
        p = 1j * omega

        with np.errstate(all="ignore"):
            expr = np.sqrt(1.0 + delta1 * p)   # sqrt(1 + δ₁ p)
            arg = lambda2 * p / expr           # аргумент coth
            cth = BoreBarModel._coth(arg)      # coth(arg)
            sigma = -p - lambda1 * expr * cth  # σ(p)

        # Убираем точки с явно «взлетевшими» значениями
        valid = (
            np.isfinite(sigma)
            & (np.abs(sigma.real) < 1e8)
            & (np.abs(sigma.imag) < 1e8)
        )

        return {
            "omega": omega[valid],
            "sigma_real": sigma.real[valid],
            "sigma_imag": sigma.imag[valid],
            "lambda1": lambda1,
            "lambda2": lambda2,
            "delta1": delta1,
        }

    @staticmethod
    def find_intersection(params: dict) -> dict | None:
        """
        Поиск точки пересечения кривой σ(p) с осью Im(σ) = 0 для крутильных колебаний.

        Это соответствует одному из предельных значений при D-разбиении
        и даёт:
            - критическую частоту ω*
            - Re(σ(ω*)), по которой можно судить об устойчивости.
        """
        rho = params["rho"]
        G = params["G"]
        Jp = params["Jp"]
        Jr = params["Jr"]
        delta1 = params["delta1"] * params["multiplier"]
        length = params["length"]

        lambda1 = np.sqrt(rho * G) * Jp / Jr
        lambda2 = length * np.sqrt(rho / G)

        def im_sigma(omega_val: float) -> float:
            """
            Вспомогательная функция: мнимая часть σ(i ω)
            для фиксированной частоты ω.
            """
            p_val = 1j * omega_val
            with np.errstate(all="ignore"):
                expr = np.sqrt(1.0 + delta1 * p_val)
                arg = lambda2 * p_val / expr
                cth = BoreBarModel._coth(arg)
                sigma_val = -p_val - lambda1 * expr * cth
                return float(np.imag(sigma_val))

        try:
            # Подбираем корень Im σ(ω) = 0 на нескольких интервалах частот
            brackets = [(500, 2000), (2000, 5000), (5000, 10000), (10000, 20000)]
            omega_cross = None

            for a, b in brackets:
                try:
                    sol = root_scalar(im_sigma, bracket=(a, b), method="brentq")
                    if sol.converged:
                        omega_cross = sol.root
                        break
                except Exception:
                    continue

            if omega_cross is None:
                return None

            # Вычисляем Re σ(ω*) в найденной точке
            p_cross = 1j * omega_cross
            expr = np.sqrt(1.0 + delta1 * p_cross)
            arg = lambda2 * p_cross / expr
            cth = BoreBarModel._coth(arg)
            sigma_cross = -p_cross - lambda1 * expr * cth

            return {
                "omega": omega_cross,
                "re_sigma": float(np.real(sigma_cross)),
                "frequency": float(omega_cross / (2.0 * np.pi)),  # Гц
            }
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # Продольные колебания
    # -------------------------------------------------------------------------

    @staticmethod
    def calculate_longitudinal(params: dict) -> dict:
        """
        Расчёт D-разбиения для продольных колебаний.

        Используем формулы из теоретической части / Maple:

            a² = E / ρ
            x  = ω L / a²

            K₁(ω) = (E S / a²) * ω cot(x) / (1 - μ cos(ω τ))
            δ(ω)  = -(E S μ / a²) * cot(x) sin(ω τ) / (1 - μ cos(ω τ))

        Здесь:
            E     — модуль Юнга,
            S     — площадь сечения,
            ρ     — плотность,
            L     — длина,
            μ     — коэффициент трения (внешний/запаздывания),
            τ     — время запаздывания.
        """
        E = params["E"]
        S = params["S"]
        rho = params["rho"]
        L = params["length"]
        mu = params["mu"]
        tau = params["tau"]

        # a² = E/ρ (в теории обычно a — скорость продольной волны, но здесь достаточно a²)
        a2 = E / rho
        # Основная собственная частота (формально, но пригодится для справки)
        omega_main = np.pi * a2 / L

        # Диапазон частот
        omega = np.linspace(0.01, 2.0 * np.pi * 100.0, 5000)

        with np.errstate(all="ignore"):
            x = omega * L / a2
            sin_x = np.sin(x)
            cos_x = np.cos(x)

            # Вычисляем cot(x) = cos(x)/sin(x) с защитой от деления на 0
            mask_cot = np.abs(sin_x) > 1e-6
            cot = np.zeros_like(x)
            cot[mask_cot] = cos_x[mask_cot] / sin_x[mask_cot]

            # Знаменатель (1 - μ cos(ω τ)) с мягкой регуляризацией
            denom = 1.0 - mu * np.cos(omega * tau)
            eps = 1e-6
            denom_safe = np.where(np.abs(denom) < eps, eps * np.sign(denom + 1e-12), denom)

            K1 = (E * S / a2) * omega * cot / denom_safe
            delta = -(E * S * mu / a2) * cot * np.sin(omega * tau) / denom_safe

            # Фильтруем некорректные/слишком большие значения
            valid = (
                mask_cot
                & np.isfinite(K1)
                & np.isfinite(delta)
                & (K1 > 0)
                & (K1 < 1e10)
                & (np.abs(delta) < 1e6)
            )

        return {
            "omega": omega[valid],
            "K1": K1[valid],
            "delta": delta[valid],
            "a": a2,
            "omega_main": omega_main,
            # Пределы при ω → 0
            "K1_0": (E * S) / (L * (1.0 - mu)),
            "delta_0": -(E * S * mu * tau) / (L * (1.0 - mu)),
        }

    # -------------------------------------------------------------------------
    # Сравнительный анализ (зависимости от длины)
    # -------------------------------------------------------------------------

    @staticmethod
    def calculate_comparative(params: dict) -> dict:
        """
        Простой сравнительный анализ:
        - как меняются собственные частоты крутильных и продольных колебаний
          в зависимости от длины L;
        - условная "степень устойчивости" 1 / L².

        Носит качественный характер и используется для иллюстрации.
        """
        lengths = np.linspace(2.0, 6.0, 20)

        torsional_freq = np.sqrt(params["G"] / params["rho"]) * np.pi / lengths
        longitudinal_freq = np.sqrt(params["E"] / params["rho"]) * np.pi / lengths
        stability_ratio = 1.0 / lengths**2

        return {
            "lengths": lengths,
            "torsional_freq": torsional_freq,
            "longitudinal_freq": longitudinal_freq,
            "stability_ratio": stability_ratio,
        }

    # -------------------------------------------------------------------------
    # Поперечные колебания (модальная аппроксимация)
    # -------------------------------------------------------------------------

    @staticmethod
    def calculate_transverse(params: dict) -> dict:
        """
        Расчёт поперечных колебаний в модальной аппроксимации.

        Используем формулу:
            W(p) = φ(L)² * K_cut * (1 - μ e^{-p τ}) / (α p² + β p + γ),  p = i ω

        где:
            α = m ∫₀ᴸ φ(x)² dx
            β = коэффициент вязкого демпфирования (beta)
            γ = E J ∫₀ᴸ φ''(x)² dx

        φ(x) — первая собственная форма консольной балки.
        Мы используем классическую аналитическую форму, эквивалентную
        выражению через функции Крылова из теоретической части, но с другой
        нормировкой (масштаб φ сокращается в W(p)).
        """
        E = params["E"]
        rho = params["rho"]
        L = params["length"]
        mu = params["mu"]
        tau = params["tau"]

        # Геометрические параметры борштанги
        R = params.get("R", 0.04)          # внешний радиус, м
        r = params.get("r", 0.035)         # внутренний радиус, м
        K_cut = params.get("K_cut", 6e5)   # динамическая жёсткость резания, Н/м
        beta = params.get("beta", 0.3)     # коэффициент вязкого демпфирования β

        S = np.pi * (R**2 - r**2)          # площадь поперечного сечения
        m = rho * S                        # погонная масса
        J = np.pi * (R**4 - r**4) / 4.0    # изгибной момент инерции

        # ----- Собственная форма φ(x) для консольной балки -----
        # alpha1_const — первый корень уравнения для консольной балки.
        alpha1_const = 1.875104068711961
        lam = alpha1_const / L

        def phi(x: np.ndarray) -> np.ndarray:
            """
            Первая собственная форма консольной балки:
                φ(x) = cosh(λx) - cos(λx) - A (sinh(λx) - sin(λx)),
            где A зависит от значений в точке x = L.
            """
            x = np.asarray(x)
            cosh_lx = np.cosh(lam * x)
            cos_lx = np.cos(lam * x)
            sinh_lx = np.sinh(lam * x)
            sin_lx = np.sin(lam * x)

            denom = np.sinh(lam * L) + np.sin(lam * L)
            num = np.cosh(lam * L) + np.cos(lam * L)
            A = num / denom

            return cosh_lx - cos_lx - A * (sinh_lx - sin_lx)

        def phi_pp(x: np.ndarray) -> np.ndarray:
            """
            Вторая производная φ''(x).
            Нужна для вычисления изгибной энергии и коэффициента γ.
            """
            x = np.asarray(x)
            cosh_lx = np.cosh(lam * x)
            cos_lx = np.cos(lam * x)
            sinh_lx = np.sinh(lam * x)
            sin_lx = np.sin(lam * x)

            denom = np.sinh(lam * L) + np.sin(lam * L)
            num = np.cosh(lam * L) + np.cos(lam * L)
            A = num / denom

            return lam**2 * (cosh_lx + cos_lx - A * (sinh_lx + sin_lx))

        # ----- Численное вычисление α и γ через интегралы -----
        x_grid = np.linspace(0.0, L, 1000)
        phi_vals = phi(x_grid)
        phi_pp_vals = phi_pp(x_grid)

        alpha = m * np.trapz(phi_vals**2, x_grid)
        gamma = E * J * np.trapz(phi_pp_vals**2, x_grid)

        # Диапазон частот (как в Maple-примере, 0..220 рад/с)
        omega = np.linspace(0.1, 220.0, 2000)
        p = 1j * omega
        phi_L = phi(L)

        with np.errstate(all="ignore"):
            numerator = (phi_L**2) * K_cut * (1.0 - mu * np.exp(-p * tau))
            denom = alpha * p**2 + beta * p + gamma

            # Мягкая защита от деления на 0
            eps = 1e-9
            denom_safe = np.where(
                np.abs(denom) < eps,
                eps * np.sign(denom + 1e-12),
                denom,
            )
            W = numerator / denom_safe

        # Фильтрация аномальных значений
        mask = (
            np.isfinite(W)
            & (np.abs(W.real) < 1e8)
            & (np.abs(W.imag) < 1e8)
        )

        W_valid = W[mask]
        omega_valid = omega[mask]

        return {
            "omega": omega_valid,
            "W_real": W_valid.real,
            "W_imag": W_valid.imag,
            "alpha": alpha,
            "gamma": gamma,
            "beta": beta,
            "phi_L": phi_L,
            "K_cut": K_cut,
            "R": R,
            "r": r,
        }
