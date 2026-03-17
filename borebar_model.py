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
    
    def _find_zero_crossings(self, y: np.ndarray):
        """
        Возвращает индексы, где функция меняет знак.
        Используется для поиска пересечений Im = 0.
        """
        y = np.asarray(y)
        sign = np.sign(y)
        sign[sign == 0] = 1
        return np.where(np.diff(sign) != 0)[0]
    
    def _coth(z):
        """
        Численно устойчивая coth(z).
        Для |z| -> 0 используем разложение:
            coth(z) ≈ 1/z + z/3
        """
        z = np.asarray(z)
        eps = 1e-8
        small = np.abs(z) < eps

        out = np.empty_like(z, dtype=complex)

        # Малые аргументы — разложение
        out[small] = 1.0 / z[small] + z[small] / 3.0

        # Остальные — стандартная формула
        out[~small] = 1.0 / np.tanh(z[~small])

        return out
        
    @staticmethod
    def _sign_change_intervals(x: np.ndarray, y: np.ndarray) -> list[tuple[int, int]]:
        """
        Возвращает пары индексов (i, i+1) В ИСХОДНЫХ массивах,
        где y меняет знак (пересечение нуля).
        NaN/Inf пропускаем корректно, без потери соответствия индексов.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if x.size < 2 or y.size < 2:
            return []

        mask = np.isfinite(x) & np.isfinite(y)
        idx = np.where(mask)[0]          # индексы в исходных массивах
        if idx.size < 2:
            return []

        yy = y[idx]
        s = np.sign(yy).astype(float)
        s[s == 0.0] = np.nan

        intervals = []
        for k in range(len(yy) - 1):
            if not (np.isfinite(s[k]) and np.isfinite(s[k + 1])):
                continue
            if s[k] * s[k + 1] < 0:
                # маппим обратно в исходные индексы
                intervals.append((int(idx[k]), int(idx[k + 1])))
        return intervals

    @staticmethod
    def _linear_root(x1: float, y1: float, x2: float, y2: float) -> float:
        """
        Линейная интерполяция корня y(x)=0 на отрезке [x1,x2].
        """
        if y2 == y1:
            return 0.5 * (x1 + x2)
        return x1 - y1 * (x2 - x1) / (y2 - y1)
    # -------------------------------------------------------------------------
    # Крутильные колебания
    # -------------------------------------------------------------------------
    
    def calculate_torsional(self, params: dict):
        """
        Крутильные колебания.

        ВАЖНО:
        Численная модель рассчитывается только для положительных частот ω > 0.
        Если пользователь задаёт диапазон с отрицательной частью, то
        отрицательная ветвь должна отображаться на уровне GUI как сопряжённое
        отражение положительной ветви:
            Re(σ(-iω)) = Re(σ(iω)),
            Im(σ(-iω)) = -Im(σ(iω)).
        """
        rho = float(params["rho"])
        G = float(params["G"])
        Jp = float(params["Jp"])
        Jr = float(params["Jr"])
        L = float(params["length"])
        delta1 = float(params["delta1"])
        multiplier = float(params.get("multiplier", 1.0))
        d1 = delta1 * multiplier

        omega_start = float(params.get("omega_start", 1.0))
        omega_end = float(params.get("omega_end", 15000.0))
        omega_step = float(params.get("omega_step", 1.0))

        if omega_step <= 0:
            raise ValueError("omega_step должен быть > 0")

        omega_full = np.arange(
            omega_start,
            omega_end,
            omega_step,
            dtype=float
        )

        # Модель считаем только для положительных частот.
        # Отрицательная ветвь при необходимости отображается в GUI как сопряжённая.
        omega = omega_full[omega_full > 0]

        if omega.size == 0:
            return {
                "omega": np.array([], dtype=float),
                "sigma_real": np.array([], dtype=float),
                "sigma_imag": np.array([], dtype=float),
                "delta1_effective": d1,
            }

        p = 1j * omega

        lam1 = np.sqrt(rho * G) * Jp / Jr
        lam2 = L * np.sqrt(rho / G)

        expr = np.sqrt(1.0 + d1 * p)
        arg = lam2 * p / expr

        arg_min = float(params.get("arg_min", 0.0))
        if arg_min > 0:
            bad = np.abs(arg) < arg_min
        else:
            bad = np.zeros_like(arg, dtype=bool)

        def stable_coth(z):
            z = np.asarray(z)
            out = np.empty_like(z, dtype=complex)
            small = np.abs(z) < 1e-8

            out[small] = 1.0 / z[small] + z[small] / 3.0
            out[~small] = np.cosh(z[~small]) / np.sinh(z[~small])

            return out

        sigma = -p - lam1 * expr * stable_coth(arg)
        sigma[bad] = np.nan + 1j * np.nan

        return {
            "omega": omega,
            "sigma_real": sigma.real,
            "sigma_imag": sigma.imag,
            "delta1_effective": d1,
        }
    
    @staticmethod
    def find_torsional_im0_points(params: dict) -> dict:
        """
        Находит ВСЕ пересечения Im(σ(iω))=0 на той же кривой,
        которая строится методом calculate_torsional(params).

        Поэтому и график, и точки Im=0, и критическая точка
        используют одно и то же эффективное демпфирование:
            d1 = delta1 * multiplier
        """
        # 1) посчитать кривую
        model = BoreBarModel()
        res = model.calculate_torsional(params)
        omega = np.asarray(res["omega"], dtype=float)
        sig_re = np.asarray(res["sigma_real"], dtype=float)
        sig_im = np.asarray(res["sigma_imag"], dtype=float)

        # 2) интервалы смены знака Im
        intervals = BoreBarModel._sign_change_intervals(omega, sig_im)

        points = []
        for i, j in intervals:
            w1, w2 = omega[i], omega[j]
            y1, y2 = sig_im[i], sig_im[j]

            w0 = BoreBarModel._linear_root(w1, y1, w2, y2)

            # линейно оценим Re в этой точке (для маркера и выбора критической)
            re0 = np.interp(w0, [w1, w2], [sig_re[i], sig_re[j]])

            points.append({
                "omega": float(w0),
                "re": float(re0),
                "im": 0.0,
                "frequency": float(w0 / (2.0 * np.pi)),
            })

        # 3) критическая = min Re
        critical = None
        if points:
            critical = min(points, key=lambda p: p["re"])

        return {
            "points": points,        # все пересечения
            "critical": critical,    # точка с минимальным Re
        }

    @staticmethod
    def find_intersection(params: dict) -> dict | None:
        """
        Поиск точки пересечения Im σ(iω) = 0.

        ВАЖНО:
        Используется то же эффективное демпфирование, что и в calculate_torsional:
            d1 = delta1 * multiplier
        """
        rho = float(params["rho"])
        G = float(params["G"])
        Jr = float(params["Jr"])
        Jp = float(params["Jp"])
        delta1 = float(params["delta1"])
        multiplier = float(params.get("multiplier", 1.0))
        length = float(params["length"])

        lam1 = np.sqrt(rho * G) * Jp / Jr
        lam2 = length * np.sqrt(rho / G)
        d1 = delta1 * multiplier

        def im_sigma(w: float) -> float:
            p = 1j * w
            expr = np.sqrt(1.0 + d1 * p)
            arg = lam2 * p / expr
            s = -p - lam1 * expr * BoreBarModel._coth(arg)
            return float(np.imag(s))

        # строгие брекеты как в исследовании
        bracket = (500.0, 1000.0) if abs(length - 6.0) < 1e-12 else (500.0, 2000.0)

        sol = root_scalar(im_sigma, bracket=bracket, method="brentq")
        if not sol.converged:
            return None

        omega_cross = float(sol.root)
        p = 1j * omega_cross
        expr = np.sqrt(1.0 + d1 * p)
        arg = lam2 * p / expr
        sigma_val = -p - lam1 * expr * BoreBarModel._coth(arg)

        return {
            "omega": omega_cross,
            "re_sigma": float(np.real(sigma_val)),
            "frequency": omega_cross / (2.0 * np.pi),
        }
        
    @staticmethod
    def find_critical_delta1(params: dict) -> dict | None:
        """
        Поиск критического значения δ₁,кр, при котором Re σ(iω*) = 0
        (граница устойчивости для крутильных колебаний).

        Параметры:
          params: словарь, как для calculate_torsional/find_intersection.
                  Используется поле 'delta1' как базовое значение δ₁.

        Возвращает dict:
            'delta_crit'         — критическое δ₁ (с),
            'delta_crit_scaled'  — δ₁ × 1e6 (для удобства графиков),
            'omega'              — ω* (рад/с),
            'frequency'          — f* (Гц),
        либо None, если найти не удалось.
        """
        base_delta = params["delta1"]
        if base_delta <= 0:
            return None

        # Вспомогательная функция: Re σ(iω*) для заданного δ₁
        def re_sigma_for_delta(delta_val: float) -> float:
            local_params = dict(params)
            local_params["delta1"] = delta_val
            inter = BoreBarModel.find_intersection(local_params)
            if inter is None:
                return np.nan
            return inter["re_sigma"]

        # Подбираем несколько точек по δ₁ вокруг базового значения
        factors = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
        deltas = []
        values = []

        for k in factors:
            d_val = base_delta * k
            val = re_sigma_for_delta(d_val)
            if not (np.isnan(val) or np.isinf(val)):
                deltas.append(d_val)
                values.append(val)

        if len(deltas) < 2:
            return None

        # Ищем пару точек со сменой знака Re σ
        bracket = None
        for i in range(len(values) - 1):
            if values[i] == 0.0:
                bracket = (deltas[i], deltas[i])
                break
            if values[i] * values[i + 1] < 0:
                bracket = (deltas[i], deltas[i + 1])
                break

        if bracket is None:
            return None

        # Если ровно попали в корень — корень найден
        if bracket[0] == bracket[1]:
            delta_crit = bracket[0]
        else:
            # Используем тот же root_scalar, что уже импортирован в модуле
            def f(delta_val: float) -> float:
                return re_sigma_for_delta(delta_val)

            sol = root_scalar(f, bracket=bracket, method="brentq")
            if not sol.converged:
                return None
            delta_crit = sol.root

        # Считаем характеристики в критической точке
        local_params = dict(params)
        local_params["delta1"] = delta_crit
        inter = BoreBarModel.find_intersection(local_params)
        if inter is None:
            return None

        return {
            "delta_crit": float(delta_crit),
            "delta_crit_scaled": float(delta_crit * 1e6),
            "omega": inter["omega"],
            "frequency": inter["frequency"],
        }

    # -------------------------------------------------------------------------
    # Продольные колебания
    # -------------------------------------------------------------------------

    @staticmethod
    def calculate_longitudinal(params: dict) -> dict:
        """
        Продольные колебания: кривая D-разбиения (K₁, δ) в плоскости параметров.

        Формулы как в Matlab-листинге исследования:
            a² = E / ρ
            K₁(ω) = (E S / a²) * ω * cot( ω L / a² ) / (1 - μ cos(ω τ))
            δ(ω)  = -(E S μ / a²) * cot( ω L / a² ) * sin(ω τ) / (1 - μ cos(ω τ))

        Примечание по единицам:
        В листинге τ=60 и ω=0..0.4 очень похоже на миллисекундную шкалу.
        В GUI τ обычно вводят в секундах (0.06 для 60 мс),
        поэтому для аргумента cot(·) по умолчанию используется множитель time_scale=1000
        (сек → мс), чтобы форма графика была ближе к исследованию.

        Переопределения (если захочешь):
            longitudinal_time_scale (по умолчанию 1000),
            omega_max_longitudinal (по умолчанию 400 рад/с),
            omega_points_longitudinal (по умолчанию 12000).
        """
        E = float(params["E"])
        rho = float(params["rho"])
        S = float(params["S"])

        # длина
        if "length" in params:
            L = float(params["length"])
        elif "L" in params:
            L = float(params["L"])
        else:
            raise KeyError("length")

        # μ и τ (GUI даёт mu/tau; старые версии могли давать mu_long/tau_long)
        if "mu" in params:
            mu = float(params["mu"])
        elif "mu_long" in params:
            mu = float(params["mu_long"])
        else:
            raise KeyError("mu")

        if "tau" in params:
            tau = float(params["tau"])
        elif "tau_long" in params:
            tau = float(params["tau_long"])
        else:
            raise KeyError("tau")

        a = np.sqrt(E / rho)

        omega_start = float(params.get("omega_start", 1e-3))
        omega_end = float(params.get("omega_end", 400.0))
        omega_step = float(params.get("omega_step", 0.1))

        if omega_step <= 0:
            omega_step = 0.1

        omega = np.arange(omega_start, omega_end + omega_step, omega_step)

        # x = ω L / a² (с учётом масштаба)
        x = omega * L / a

        eps = 1e-9
        sin_x = np.sin(x)
        cos_x = np.cos(x)

        cot_x = np.full_like(x, np.nan, dtype=float)
        mask_cot = np.abs(sin_x) > eps
        cot_x[mask_cot] = cos_x[mask_cot] / sin_x[mask_cot]

        denom = 1.0 - mu * np.cos(omega * tau)
        mask_denom = np.abs(denom) > eps

        valid = mask_cot & mask_denom

        K1 = np.full_like(omega, np.nan, dtype=float)
        delta = np.full_like(omega, np.nan, dtype=float)

        K1[valid] = (E * S / a) * omega[valid] * cot_x[valid] / denom[valid]
        delta[valid] = -(E * S * mu / a) * cot_x[valid] * np.sin(omega[valid] * tau) / denom[valid]

        # Ограничим выбросы (NaN оставляем — matplotlib сам разорвёт линию)
        K1_max = float(params.get("K1_max_longitudinal", 1e10))
        delta_max = float(params.get("delta_max_longitudinal", 1e7))
        bad = (np.abs(K1) > K1_max) | (np.abs(delta) > delta_max)
        K1[bad] = np.nan
        delta[bad] = np.nan

        # Пределы при ω→0 (в справочный блок)
        K1_0 = (E * S) / (L * (1.0 - mu))
        delta_0 = -(E * S * mu * tau) / (L * (1.0 - mu))

        # Ориентир: режим x=π
        omega_main = float(np.pi * a / L)

        return {
            "omega": omega,
            "K1": K1,
            "delta": delta,
            "omega_main": omega_main,
            "K1_0": K1_0,
            "delta_0": delta_0,
            "a": a,
        }

    def compute_longitudinal_curve(self, params: dict, omega: np.ndarray):
        """Вернуть массивы (K1(ω), δ(ω)) для заданного omega.

        Нужен для экспорта и для поиска пересечений δ(ω)=0.
        Формулы совпадают с calculate_longitudinal().
        """
        E = float(params["E"])
        rho = float(params["rho"])
        S = float(params["S"])
        L = float(params.get("length", params.get("L", 0.0)))
        if L <= 0:
            raise KeyError("length")

        mu = float(params.get("mu", 0.0))

        # время запаздывания
        if "tau" in params:
            tau = float(params["tau"])
        elif "tau_long" in params:
            tau = float(params["tau_long"])
        else:
            raise KeyError("tau")

        a = np.sqrt(E / rho)

        omega = np.asarray(omega, dtype=float)

        x = omega * L / a

        eps = 1e-9
        sin_x = np.sin(x)
        cos_x = np.cos(x)

        cot_x = np.full_like(x, np.nan, dtype=float)
        mask_cot = np.abs(sin_x) > eps
        cot_x[mask_cot] = cos_x[mask_cot] / sin_x[mask_cot]

        denom = 1.0 - mu * np.cos(omega * tau)
        mask_denom = np.abs(denom) > eps
        valid = mask_cot & mask_denom

        K1 = np.full_like(omega, np.nan, dtype=float)
        delta = np.full_like(omega, np.nan, dtype=float)

        K1[valid] = (E * S / a) * omega[valid] * cot_x[valid] / denom[valid]
        delta[valid] = -(E * S * mu / a) * cot_x[valid] * np.sin(omega[valid] * tau) / denom[valid]

        # Ограничим выбросы
        K1_max = float(params.get("K1_max_longitudinal", 1e10))
        delta_max = float(params.get("delta_max_longitudinal", 1e7))
        bad = (np.abs(K1) > K1_max) | (np.abs(delta) > delta_max)
        K1[bad] = np.nan
        delta[bad] = np.nan

        return K1, delta

    def find_longitudinal_im0_points(self, params, wmin=1.0, wmax=5000.0, n=5000) -> dict:
        """Найти все пересечения δ(ω)=0 и критическую точку (min K1 среди них).

        Возвращает словарь единого формата:
            {
                "points": [...],
                "critical": {...} | None
            }
        """
        omega_start = float(params.get("omega_start", 1.0))
        omega_end = float(params.get("omega_end", 5000.0))
        omega = np.linspace(omega_start, omega_end, int(n))
        K1, delta = self.compute_longitudinal_curve(params, omega)

        idx = self._find_zero_crossings(delta)
        points = []

        for i in idx:
            j = i + 1

            w1, w2 = omega[i], omega[j]
            d1, d2 = delta[i], delta[j]

            w0 = self._linear_root(w1, d1, w2, d2)
            k0 = np.interp(w0, [w1, w2], [K1[i], K1[j]])

            points.append({
                "omega": float(w0),
                "K1": float(k0),
                "delta": 0.0,
                "re": float(k0),
            })

        critical = None
        if points:
            critical = min(points, key=lambda p: p["K1"])

        return {
            "points": points,
            "critical": critical,
        }

    def compute_transverse_curve(self, params: dict, omega: np.ndarray):
        """Вернуть (Re(W(iω)), Im(W(iω))) на заданной сетке omega.

        Для совместимости со старым экспортом.
        """
        res = self.calculate_transverse(params)
        om_ref = np.asarray(res["omega"], dtype=float)
        re_ref = np.asarray(res["W_real"], dtype=float)
        im_ref = np.asarray(res["W_imag"], dtype=float)

        omega = np.asarray(omega, dtype=float)

        # интерполяция по рассчитанной сетке
        re_w = np.interp(omega, om_ref, re_ref)
        im_w = np.interp(omega, om_ref, im_ref)
        return re_w, im_w

    def calculate_transverse(self, params: dict) -> dict:
        """
        Расчёт поперечных колебаний в модальной аппроксимации.

        Используем формулу:
            W(p) = φ(L)² * K_cut * (1 - μ e^{-p τ}) / (α p² + β p + γ),  p = i ω

        где:
            α = m ∫₀ᴸ φ(x)² dx
            β = коэффициент вязкого демпфирования (beta)
            γ = E J ∫₀ᴸ φ''(x)² dx

        φ(x) — первая собственная форма (координатная функция) из исследования.

        В приложении (Maple) собственная форма задана в виде комбинации
        гиперболических/тригонометрических функций с численным коэффициентом 0.734
        и множителем 1/2 (см. фрагмент Maple-скрипта в исследовании).

        Важно: цель этой функции — воспроизвести годограф W(p) так, как он
        построен в исследовании (рис. 3.1), поэтому форму φ(x) и производную
        берём в той же записи, что и в исходных численных экспериментах.
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
        # В модели программы используется модальный коэффициент демпфирования β.
        # Если извне передан параметр h, трактуем его как алиас для β,
        # чтобы не ломать совместимость с теоретическими обозначениями.
        beta = float(params.get("beta", params.get("h", 0.3)))

        S = np.pi * (R**2 - r**2)          # площадь поперечного сечения
        m = rho * S                        # погонная масса
        J = np.pi * (R**4 - r**4) / 4.0    # изгибной момент инерции

        # ----- Собственная форма φ(x) (как в исследовании / Maple) -----
        # В Maple использовано: k1 := 1.875 / l; (см. исследование)
        k1 = 1.875 / L
        A = 0.734
        C = (1.0 - A) / 2.0  # 1/2 - 0.734/2

        # φ(x) из Maple-скрипта
        def phi(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x)
            return C * (np.sinh(k1 * x) - np.sin(k1 * x))

        # φ''(x) для этой формы
        def phi_pp(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x)
            return C * (k1**2) * (np.sinh(k1 * x) + np.sin(k1 * x))

        # ----- Численное вычисление α и γ через интегралы -----
        x_grid = np.linspace(0.0, L, 1000)
        phi_vals = phi(x_grid)
        phi_pp_vals = phi_pp(x_grid)

        alpha = m * np.trapezoid(phi_vals**2, x_grid)
        gamma = E * J * np.trapezoid(phi_pp_vals**2, x_grid)

        # Диапазон частот (как в Maple-примере, 0..220 рад/с)
        omega_start = float(params.get("omega_start", 0.1))
        omega_end = float(params.get("omega_end", 500.0))
        omega_step = float(params.get("omega_step", 0.5))

        omega = np.arange(omega_start, omega_end + omega_step, omega_step)
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
    
    def find_transverse_im0_points(self, params: dict) -> dict:
        """
        Находит ВСЕ пересечения Im(W(iω))=0 и критическую точку: min Re среди них.
        """
        res = self.calculate_transverse(params)
        omega = np.asarray(res["omega"], dtype=float)
        Wre = np.asarray(res["W_real"], dtype=float)
        Wim = np.asarray(res["W_imag"], dtype=float)

        intervals = BoreBarModel._sign_change_intervals(omega, Wim)

        points = []
        for i, j in intervals:
            w1, w2 = omega[i], omega[j]
            y1, y2 = Wim[i], Wim[j]

            w0 = BoreBarModel._linear_root(w1, y1, w2, y2)
            re0 = np.interp(w0, [w1, w2], [Wre[i], Wre[j]])

            points.append({
                "omega": float(w0),
                "re": float(re0),
                "im": 0.0,
            })

        critical = None
        if points:
            critical = min(points, key=lambda p: p["re"])

        return {"points": points, "critical": critical}