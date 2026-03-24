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

    Основные методы:
        - calculate_torsional(params):      крутильные колебания, кривая D-разбиения σ(p);
        - find_intersection(params):        поиск пересечения Im σ(p) = 0;
        - calculate_longitudinal(params):   продольные колебания, кривая K1–δ;
        - calculate_transverse(params):     поперечные колебания, годограф W(p).
    """

    # -------------------------------------------------------------------------
    # Вспомогательные функции
    # -------------------------------------------------------------------------

    @staticmethod
    def build_frequency_grid(params: dict, include_endpoint: bool = False) -> np.ndarray:
        """Построить единую сетку частот по параметрам проекта.

        Используется анализом, экспортом и поиском специальных точек, чтобы
        все эти шаги опирались на один и тот же диапазон ω.
        """
        omega_start = float(params.get("omega_start", 0.001))
        omega_end = float(params.get("omega_end", omega_start))
        omega_step = float(params.get("omega_step", 0.1))

        if not np.isfinite(omega_start) or not np.isfinite(omega_end):
            raise ValueError("omega_start и omega_end должны быть конечными числами")
        if not np.isfinite(omega_step) or omega_step <= 0.0:
            raise ValueError("omega_step должен быть > 0")
        if omega_end < omega_start:
            raise ValueError("omega_end должен быть >= omega_start")

        if np.isclose(omega_end, omega_start):
            return np.array([omega_start], dtype=float)

        stop = omega_end + (omega_step if include_endpoint else 0.0)
        omega = np.arange(omega_start, stop, omega_step, dtype=float)

        if include_endpoint:
            omega = omega[omega <= omega_end]
            if omega.size == 0 or not np.isclose(omega[-1], omega_end):
                omega = np.append(omega, omega_end)
        else:
            omega = omega[omega < omega_end]
            if omega.size == 0:
                omega = np.array([omega_start], dtype=float)

        return np.asarray(omega, dtype=float)

    @staticmethod
    def _coth(z: np.ndarray, small_eps: float = 1e-8, sinh_eps: float = 1e-10) -> np.ndarray:
        """
        Численно устойчивая coth(z).

        Идея:
        1) Для |z| -> 0 используем разложение:
              coth(z) ≈ 1/z + z/3 - z^3/45
        2) Для остальных точек считаем через cosh(z)/sinh(z),
           но помечаем как NaN точки, где sinh(z) слишком мал,
           т.к. это окрестности полюсов.

        Возвращает массив complex той же формы.
        """
        z = np.asarray(z, dtype=complex)
        out = np.full(z.shape, np.nan + 1j * np.nan, dtype=complex)

        finite_mask = np.isfinite(z.real) & np.isfinite(z.imag)
        if not np.any(finite_mask):
            return out

        zz = z[finite_mask]

        small_mask = np.abs(zz) < small_eps
        regular_mask = ~small_mask

        out_local = np.full(zz.shape, np.nan + 1j * np.nan, dtype=complex)

        if np.any(small_mask):
            zs = zz[small_mask]
            out_local[small_mask] = 1.0 / zs + zs / 3.0 - (zs ** 3) / 45.0

        if np.any(regular_mask):
            zr = zz[regular_mask]
            sh = np.sinh(zr)
            safe_mask = np.abs(sh) >= sinh_eps

            tmp = np.full(zr.shape, np.nan + 1j * np.nan, dtype=complex)
            if np.any(safe_mask):
                tmp[safe_mask] = np.cosh(zr[safe_mask]) / sh[safe_mask]

            out_local[regular_mask] = tmp

        out[finite_mask] = out_local
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



    @staticmethod
    def _require_finite_scalar(value, name: str) -> float:
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"Параметр {name} должен быть конечным числом.")
        return value

    @staticmethod
    def _validate_frequency_params(params: dict) -> None:
        omega_start = BoreBarModel._require_finite_scalar(params.get("omega_start", 0.001), "omega_start")
        omega_end = BoreBarModel._require_finite_scalar(params.get("omega_end", omega_start), "omega_end")
        omega_step = BoreBarModel._require_finite_scalar(params.get("omega_step", 0.1), "omega_step")

        if omega_step <= 0.0:
            raise ValueError("Шаг частоты Δω должен быть > 0.")
        if omega_end < omega_start:
            raise ValueError("Конечная частота должна быть не меньше начальной.")

    @staticmethod
    def validate_torsional_params(params: dict) -> dict:
        validated = dict(params)
        for key, label in (
            ("rho", "ρ"),
            ("G", "G"),
            ("length", "L"),
            ("Jr", "Jr"),
            ("Jp", "Jp"),
        ):
            value = BoreBarModel._require_finite_scalar(validated[key], key)
            if value <= 0.0:
                raise ValueError(f"Параметр {label} должен быть > 0.")
            validated[key] = value

        delta1 = BoreBarModel._require_finite_scalar(validated["delta1"], "delta1")
        if delta1 < 0.0:
            raise ValueError("Параметр δ₁ не может быть отрицательным.")
        validated["delta1"] = delta1

        multiplier = BoreBarModel._require_finite_scalar(validated.get("multiplier", 1.0), "multiplier")
        if multiplier <= 0.0:
            raise ValueError("Множитель демпфирования должен быть > 0.")
        validated["multiplier"] = multiplier

        BoreBarModel._validate_frequency_params(validated)
        return validated

    @staticmethod
    def validate_longitudinal_params(params: dict) -> dict:
        validated = dict(params)
        for key, label in (
            ("E", "E"),
            ("rho", "ρ"),
            ("S", "S"),
        ):
            value = BoreBarModel._require_finite_scalar(validated[key], key)
            if value <= 0.0:
                raise ValueError(f"Параметр {label} должен быть > 0.")
            validated[key] = value

        if "length" in validated:
            length = BoreBarModel._require_finite_scalar(validated["length"], "length")
            validated["length"] = length
        elif "L" in validated:
            length = BoreBarModel._require_finite_scalar(validated["L"], "L")
            validated["L"] = length
        else:
            raise KeyError("length")
        if length <= 0.0:
            raise ValueError("Длина L должна быть > 0.")

        mu_key = "mu" if "mu" in validated else "mu_long" if "mu_long" in validated else None
        if mu_key is None:
            raise KeyError("mu")
        mu = BoreBarModel._require_finite_scalar(validated[mu_key], mu_key)
        validated[mu_key] = mu

        tau_key = "tau" if "tau" in validated else "tau_long" if "tau_long" in validated else None
        if tau_key is None:
            raise KeyError("tau")
        tau = BoreBarModel._require_finite_scalar(validated[tau_key], tau_key)
        if tau < 0.0:
            raise ValueError("Время запаздывания τ не может быть отрицательным.")
        validated[tau_key] = tau

        BoreBarModel._validate_frequency_params(validated)

        mu_singularity_eps = float(validated.get("mu_singularity_eps_longitudinal", 1e-9))
        if abs(1.0 - mu) <= mu_singularity_eps:
            raise ValueError(
                "Продольная модель вырождена при μ≈1: знаменатель 1-μ обращается в ноль. "
                "Уменьшите μ или задайте значение вдали от единицы."
            )
        return validated

    @staticmethod
    def validate_transverse_params(params: dict) -> dict:
        validated = dict(params)
        for key, label in (
            ("E", "E"),
            ("rho", "ρ"),
            ("length", "L"),
            ("R", "R"),
            ("K_cut", "K_cut"),
        ):
            value = BoreBarModel._require_finite_scalar(validated[key], key)
            if value <= 0.0:
                raise ValueError(f"Параметр {label} должен быть > 0.")
            validated[key] = value

        r = BoreBarModel._require_finite_scalar(validated.get("r", 0.0), "r")
        if r < 0.0:
            raise ValueError("Внутренний радиус r не может быть отрицательным.")
        validated["r"] = r
        if validated["R"] <= r:
            raise ValueError("Должно выполняться R > r.")

        mu = BoreBarModel._require_finite_scalar(validated.get("mu", 0.0), "mu")
        tau = BoreBarModel._require_finite_scalar(validated.get("tau", 0.0), "tau")
        if tau < 0.0:
            raise ValueError("Время запаздывания τ не может быть отрицательным.")
        validated["mu"] = mu
        validated["tau"] = tau

        if "h" in validated and validated.get("h") is not None:
            h = BoreBarModel._require_finite_scalar(validated["h"], "h")
            if h < 0.0:
                raise ValueError("Коэффициент внутреннего трения h не может быть отрицательным.")
            validated["h"] = h
        if "beta" in validated and validated.get("beta") is not None:
            beta = BoreBarModel._require_finite_scalar(validated["beta"], "beta")
            if beta < 0.0:
                raise ValueError("Коэффициент β не может быть отрицательным.")
            validated["beta"] = beta

        variant = str(validated.get("transverse_modal_shape_variant", "verified_cantilever_first_mode_phi"))
        if variant not in {"verified_cantilever_first_mode_phi", "project_maple_compatible_phi"}:
            raise ValueError(f"Неизвестный вариант поперечной формы: {variant}")
        validated["transverse_modal_shape_variant"] = variant

        BoreBarModel._validate_frequency_params(validated)
        return validated

    @staticmethod
    def _count_true_map(mask_map: dict[str, np.ndarray]) -> dict[str, int]:
        """Свернуть словарь булевых масок в словарь количества True."""
        counts = {}
        for key, mask in mask_map.items():
            counts[key] = int(np.count_nonzero(np.asarray(mask, dtype=bool)))
        return counts

    @staticmethod
    def _refine_frequency_grid(omega: np.ndarray, factor: int = 1, max_points: int = 25000) -> np.ndarray:
        """Уплотнить сетку частот для отображения, не меняя экспортную физическую сетку."""
        omega = np.asarray(omega, dtype=float)
        if omega.size < 2:
            return omega.copy()

        factor = int(max(1, factor))
        max_points = int(max(omega.size, max_points))

        if factor <= 1:
            return omega.copy()

        max_factor = max(1, max_points // max(1, omega.size - 1))
        factor = min(factor, max_factor)
        if factor <= 1:
            return omega.copy()

        chunks = []
        for i in range(omega.size - 1):
            chunks.append(np.linspace(omega[i], omega[i + 1], factor, endpoint=False, dtype=float))
        chunks.append(np.array([omega[-1]], dtype=float))
        return np.concatenate(chunks)
    
    @staticmethod
    def _torsional_singularity_mask(
        arg: np.ndarray,
        sigma: np.ndarray | None = None,
        arg_min: float = 0.0,
        sinh_eps: float = 1e-10,
        sigma_clip: float | None = None,
    ) -> dict:
        """
        Построить диагностику невалидных точек крутильной ветви.

        Возвращает словарь с:
        - invalid_mask: общая маска невалидных точек;
        - invalid_reason_masks: маски по причинам;
        - invalid_reason_counts: количества по причинам.
        """
        arg = np.asarray(arg, dtype=complex)
        shape = arg.shape

        arg_nonfinite = ~(np.isfinite(arg.real) & np.isfinite(arg.imag))

        if arg_min > 0.0:
            arg_too_small = ~arg_nonfinite & (np.abs(arg) < arg_min)
        else:
            arg_too_small = np.zeros(shape, dtype=bool)

        with np.errstate(all="ignore"):
            sh = np.sinh(arg)
        sh_nonfinite = ~(np.isfinite(sh.real) & np.isfinite(sh.imag))
        near_coth_pole = ~arg_nonfinite & ~sh_nonfinite & (np.abs(sh) < sinh_eps)

        invalid_reason_masks = {
            "arg_nonfinite": np.asarray(arg_nonfinite, dtype=bool),
            "arg_too_small": np.asarray(arg_too_small, dtype=bool),
            "near_coth_pole": np.asarray(near_coth_pole, dtype=bool),
        }

        if sigma is not None:
            sigma = np.asarray(sigma, dtype=complex)
            sigma_nonfinite = ~(np.isfinite(sigma.real) & np.isfinite(sigma.imag))
            invalid_reason_masks["sigma_nonfinite"] = np.asarray(sigma_nonfinite, dtype=bool)

            if sigma_clip is not None and sigma_clip > 0.0:
                sigma_clip_mask = (
                    ~sigma_nonfinite
                    & ((np.abs(sigma.real) > sigma_clip) | (np.abs(sigma.imag) > sigma_clip))
                )
            else:
                sigma_clip_mask = np.zeros(shape, dtype=bool)

            invalid_reason_masks["sigma_clip"] = np.asarray(sigma_clip_mask, dtype=bool)
        else:
            invalid_reason_masks["sigma_nonfinite"] = np.zeros(shape, dtype=bool)
            invalid_reason_masks["sigma_clip"] = np.zeros(shape, dtype=bool)

        invalid_mask = np.zeros(shape, dtype=bool)
        for mask in invalid_reason_masks.values():
            invalid_mask |= np.asarray(mask, dtype=bool)

        return {
            "invalid_mask": invalid_mask,
            "invalid_reason_masks": invalid_reason_masks,
            "invalid_reason_counts": BoreBarModel._count_true_map(invalid_reason_masks),
        }

    # -------------------------------------------------------------------------
    # Крутильные колебания
    # -------------------------------------------------------------------------

    @staticmethod
    def _torsional_positive_omega(params: dict) -> np.ndarray:
        """
        Каноническая физическая сетка для крутильной модели:
        считаем только на ω > 0.

        Важно: именно эта сетка должна использоваться для:
        - расчёта физической кривой;
        - поиска Im(σ)=0;
        - выбора критической точки.

        Если пользователь задаёт диапазон, заходящий в отрицательные частоты,
        физическая положительная ветвь должна строиться независимо от того,
        с какого отрицательного значения стартовал np.arange(). Иначе
        положительная часть сетки сдвигается на накопленную ошибку шага:
            -20000, -19999.9, ... , 0.099997...
        вместо канонических:
            0.1, 0.2, 0.3, ...
        Из-за этого один и тот же физический расчёт даёт разные значения для
        почти нулевого демпфирования. Поэтому при omega_start <= 0 используем
        сетку, привязанную к нулю и построенную через целые номера шагов.
        """
        omega_start = float(params.get("omega_start", 0.001))
        omega_end = float(params.get("omega_end", omega_start))
        omega_step = float(params.get("omega_step", 0.1))

        if omega_end <= 0.0:
            return np.array([], dtype=float)

        if omega_start > 0.0:
            omega_pos = BoreBarModel.build_frequency_grid(params, include_endpoint=False)
            return np.asarray(omega_pos[omega_pos > 0.0], dtype=float)

        # Ищем наибольший целый n, для которого n * omega_step < omega_end.
        # Делать это через (omega_end - eps) / omega_step ненадёжно: например,
        # для 20000 / 0.1 из-за двоичной арифметики легко получить ровно 200000.0
        # и ошибочно включить точку omega_end, хотя физическая сетка строится как
        # np.arange(step, omega_end, step), то есть БЕЗ правого конца.
        ratio = float(omega_end / omega_step)
        n_last = int(np.floor(np.nextafter(ratio, -np.inf)))
        if n_last < 1:
            return np.array([], dtype=float)

        return omega_step * np.arange(1, n_last + 1, dtype=float)

    @staticmethod
    def _evaluate_torsional_sigma_positive(params: dict, omega_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Вычисляет σ(iω) только на физической положительной ветви ω > 0.
        Возвращает:
            omega_pos, sigma_real_pos, sigma_imag_pos, delta1_effective
        """
        params = BoreBarModel.validate_torsional_params(params)

        params = self.validate_torsional_params(params)

        rho = float(params["rho"])
        G = float(params["G"])
        Jp = float(params["Jp"])
        Jr = float(params["Jr"])
        L = float(params["length"])
        delta1 = float(params["delta1"])
        multiplier = float(params.get("multiplier", 1.0))
        d1 = delta1 * multiplier

        omega_pos = np.asarray(omega_pos, dtype=float)

        if omega_pos.size == 0:
            return (
                np.array([], dtype=float),
                np.array([], dtype=float),
                np.array([], dtype=float),
                d1,
            )

        p = 1j * omega_pos

        lam1 = np.sqrt(rho * G) * Jp / Jr
        lam2 = L * np.sqrt(rho / G)

        expr = np.sqrt(1.0 + d1 * p)
        arg = lam2 * p / expr

        arg_min = float(params.get("arg_min", 0.0))
        if arg_min > 0.0:
            bad = np.abs(arg) < arg_min
        else:
            bad = np.zeros_like(arg, dtype=bool)

        sigma = -p - lam1 * expr * BoreBarModel._coth(arg)
        sigma = np.asarray(sigma, dtype=complex)

        if np.any(bad):
            sigma = sigma.copy()
            sigma[bad] = np.nan + 1j * np.nan

        return omega_pos, sigma.real.astype(float), sigma.imag.astype(float), d1

    @staticmethod
    def _build_torsional_display_curve(
        params: dict,
        omega_pos: np.ndarray,
        sigma_real_pos: np.ndarray,
        sigma_imag_pos: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Строит отображаемую кривую для GUI/экспорта.

        Физическая модель считается только на ω > 0.
        Если пользователь запросил диапазон с отрицательной частью,
        отображаемая ветвь достраивается как сопряжённое отражение:

            Re(σ(-iω)) = Re(σ(iω))
            Im(σ(-iω)) = -Im(σ(iω))
        """
        omega_pos = np.asarray(omega_pos, dtype=float)
        sigma_real_pos = np.asarray(sigma_real_pos, dtype=float)
        sigma_imag_pos = np.asarray(sigma_imag_pos, dtype=float)

        omega_start = float(params.get("omega_start", 0.0))

        if omega_pos.size == 0:
            return (
                np.array([], dtype=float),
                np.array([], dtype=float),
                np.array([], dtype=float),
            )

        if omega_start < 0.0:
            omega_neg = -omega_pos[::-1]
            re_neg = sigma_real_pos[::-1]
            im_neg = -sigma_imag_pos[::-1]

            omega_display = np.concatenate([omega_neg, omega_pos])
            sigma_real_display = np.concatenate([re_neg, sigma_real_pos])
            sigma_imag_display = np.concatenate([im_neg, sigma_imag_pos])
            return omega_display, sigma_real_display, sigma_imag_display

        return omega_pos.copy(), sigma_real_pos.copy(), sigma_imag_pos.copy()

    def calculate_torsional(self, params: dict) -> dict:
        """
        Крутильные колебания.

        Физическая модель считается только на ω > 0.
        Если пользователь задал отрицательный диапазон, отображаемая ветвь
        достраивается как сопряжённое отражение положительной.

        Вся диагностика невалидных точек формируется в модели.
        """
        params = self.validate_torsional_params(params)

        rho = float(params["rho"])
        G = float(params["G"])
        Jp = float(params["Jp"])
        Jr = float(params["Jr"])
        L = float(params["length"])
        delta1 = float(params["delta1"])
        multiplier = float(params.get("multiplier", 1.0))
        d1 = delta1 * multiplier

        omega_pos = self._torsional_positive_omega(params)
        if omega_pos.size == 0:
            empty = np.array([], dtype=float)
            empty_bool = np.array([], dtype=bool)
            empty_reason_masks = {
                "arg_nonfinite": empty_bool,
                "arg_too_small": empty_bool,
                "near_coth_pole": empty_bool,
                "sigma_nonfinite": empty_bool,
                "sigma_clip": empty_bool,
            }
            return {
                "physical_omega": empty,
                "physical_sigma_real": empty,
                "physical_sigma_imag": empty,
                "display_omega": empty,
                "display_sigma_real": empty,
                "display_sigma_imag": empty,
                "omega": empty,
                "sigma_real": empty,
                "sigma_imag": empty,
                "delta1_effective": d1,
                "invalid_mask": empty_bool,
                "invalid_reason_masks": empty_reason_masks,
                "invalid_reason_counts": {k: 0 for k in empty_reason_masks},
                "invalid_point_count": 0,
                "numerics_metadata": {
                    "coth_small_eps": float(params.get("coth_small_eps", 1e-8)),
                    "coth_sinh_eps": float(params.get("coth_sinh_eps", 1e-10)),
                    "arg_min": float(params.get("arg_min", 0.0)),
                    "sigma_clip": None,
                },
                "model_variant": "torsional_physical_positive_plus_model_display_symmetry",
                "negative_frequency_policy": "display_curve_is_built_as_conjugate_mirror_of_positive_branch",
            }

        p = 1j * omega_pos
        lam1 = np.sqrt(rho * G) * Jp / Jr
        lam2 = L * np.sqrt(rho / G)

        expr = np.sqrt(1.0 + d1 * p)
        arg = lam2 * p / expr
        coth_small_eps = float(params.get("coth_small_eps", 1e-8))
        coth_sinh_eps = float(params.get("coth_sinh_eps", 1e-10))
        coth_arg = self._coth(arg, small_eps=coth_small_eps, sinh_eps=coth_sinh_eps)
        sigma = -p - lam1 * expr * coth_arg

        arg_min = float(params.get("arg_min", 0.0))
        sigma_clip = float(params.get("sigma_clip", 0.0))
        if sigma_clip <= 0.0:
            sigma_clip = None

        diagnostics = self._torsional_singularity_mask(
            arg=arg,
            sigma=sigma,
            arg_min=arg_min,
            sinh_eps=coth_sinh_eps,
            sigma_clip=sigma_clip,
        )
        invalid_mask = np.asarray(diagnostics["invalid_mask"], dtype=bool)

        sigma_clean = np.asarray(sigma, dtype=complex).copy()
        sigma_clean[invalid_mask] = np.nan + 1j * np.nan

        sig_re_pos = sigma_clean.real.astype(float)
        sig_im_pos = sigma_clean.imag.astype(float)

        omega_start = float(params.get("omega_start", 0.0))
        if omega_start < 0.0:
            omega_neg = -omega_pos[::-1]
            re_neg = sig_re_pos[::-1]
            im_neg = -sig_im_pos[::-1]
            omega_display = np.concatenate([omega_neg, omega_pos])
            sig_re_display = np.concatenate([re_neg, sig_re_pos])
            sig_im_display = np.concatenate([im_neg, sig_im_pos])
        else:
            omega_display = omega_pos.copy()
            sig_re_display = sig_re_pos.copy()
            sig_im_display = sig_im_pos.copy()

        return {
            "physical_omega": omega_pos,
            "physical_sigma_real": sig_re_pos,
            "physical_sigma_imag": sig_im_pos,
            "display_omega": omega_display,
            "display_sigma_real": sig_re_display,
            "display_sigma_imag": sig_im_display,
            "omega": omega_pos,
            "sigma_real": sig_re_pos,
            "sigma_imag": sig_im_pos,
            "delta1_effective": d1,
            "invalid_mask": invalid_mask,
            "invalid_reason_masks": diagnostics["invalid_reason_masks"],
            "invalid_reason_counts": diagnostics["invalid_reason_counts"],
            "invalid_point_count": int(np.count_nonzero(invalid_mask)),
            "numerics_metadata": {
                "coth_small_eps": coth_small_eps,
                "coth_sinh_eps": coth_sinh_eps,
                "arg_min": arg_min,
                "sigma_clip": sigma_clip,
            },
            "model_variant": "torsional_physical_positive_plus_model_display_symmetry",
            "negative_frequency_policy": "display_curve_is_built_as_conjugate_mirror_of_positive_branch",
        }

    @staticmethod
    def find_torsional_im0_points(params: dict) -> dict:
        """
        Ищет все пересечения Im(σ(iω)) = 0 по физической положительной ветви.

        Сначала учитываются точные нули, попавшие в узлы сетки,
        затем добавляются нули между узлами по смене знака.
        """
        model = BoreBarModel()
        res = model.calculate_torsional(params)

        omega = np.asarray(res["physical_omega"], dtype=float)
        sig_re = np.asarray(res["physical_sigma_real"], dtype=float)
        sig_im = np.asarray(res["physical_sigma_imag"], dtype=float)

        points = []
        eps = float(params.get("im0_eps_torsional", 1e-9))
        finite_zero = np.isfinite(omega) & np.isfinite(sig_re) & np.isfinite(sig_im) & (np.abs(sig_im) <= eps)
        for idx in np.where(finite_zero)[0]:
            w = float(omega[idx])
            points.append({
                "omega": w,
                "re": float(sig_re[idx]),
                "im": 0.0,
                "frequency": float(w / (2.0 * np.pi)),
            })

        for i, j in BoreBarModel._sign_change_intervals(omega, sig_im):
            w1, w2 = omega[i], omega[j]
            y1, y2 = sig_im[i], sig_im[j]
            w0 = BoreBarModel._linear_root(w1, y1, w2, y2)
            re0 = np.interp(w0, [w1, w2], [sig_re[i], sig_re[j]])
            points.append({
                "omega": float(w0),
                "re": float(re0),
                "im": 0.0,
                "frequency": float(w0 / (2.0 * np.pi)),
            })

        dedup = []
        omega_tol = float(params.get("im0_omega_tol_torsional", max(float(params.get("omega_step", 1.0)) * 0.5, 1e-9)))
        re_tol = float(params.get("im0_re_tol_torsional", 1e-6))
        for p in sorted(points, key=lambda item: (item["omega"], item["re"])):
            if dedup and abs(p["omega"] - dedup[-1]["omega"]) <= omega_tol and abs(p["re"] - dedup[-1]["re"]) <= re_tol:
                continue
            dedup.append(p)

        critical = min(dedup, key=lambda p: p["re"]) if dedup else None
        return {
            "points": dedup,
            "critical": critical,
            "source_curve": "physical_positive_branch",
        }

    @staticmethod
    def find_intersection(params: dict) -> dict | None:
        """
        Совместимый интерфейс: возвращает критическую точку из find_torsional_im0_points(),
        не решая отдельную задачу root_scalar.
        """
        im0 = BoreBarModel.find_torsional_im0_points(params)
        critical = im0.get("critical")
        if not critical:
            return None

        return {
            "omega": float(critical["omega"]),
            "re_sigma": float(critical["re"]),
            "frequency": float(critical["frequency"]),
        }
    
    # -------------------------------------------------------------------------
    # Продольные колебания
    # -------------------------------------------------------------------------

    @staticmethod
    def calculate_longitudinal(params: dict) -> dict:
        """
        Продольные колебания: кривая D-разбиения (K₁, δ) в плоскости параметров.

        В текущей версии проекта используется единая физически согласованная
        SI-модель продольных колебаний:
            a = sqrt(E / ρ),
            x = ω L / a,
            K₁(ω) = (E S / a) * ω * cot(x) / (1 - μ cos(ω τ)),
            δ(ω)  = -(E S μ / a) * cot(x) * sin(ω τ) / (1 - μ cos(ω τ)).

        Частотная сетка строится через build_frequency_grid(), поэтому анализ,
        экспорт и поиск специальных точек опираются на один и тот же диапазон.
        При необходимости пользовательская сетка может быть подана напрямую
        через params["omega_override"].
        """
        params = BoreBarModel.validate_longitudinal_params(params)

        E = float(params["E"])
        rho = float(params["rho"])
        S = float(params["S"])

        if "length" in params:
            L = float(params["length"])
        elif "L" in params:
            L = float(params["L"])
        else:
            raise KeyError("length")

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

        mu_singularity_eps = float(params.get("mu_singularity_eps_longitudinal", 1e-9))
        if abs(1.0 - mu) <= mu_singularity_eps:
            raise ValueError(
                "Продольная модель вырождена при μ≈1: знаменатель 1-μ обращается в ноль. "
                "Уменьшите μ или задайте значение вдали от единицы."
            )

        a = np.sqrt(E / rho)

        omega_override = params.get("omega_override")
        if omega_override is not None:
            omega = np.asarray(omega_override, dtype=float)
        else:
            omega = BoreBarModel.build_frequency_grid(params, include_endpoint=True)

        K1, delta = BoreBarModel().compute_longitudinal_curve(params, omega)

        K1_0 = (E * S) / (L * (1.0 - mu))
        delta_0 = -(E * S * mu * tau) / (L * (1.0 - mu))
        omega_main = float(np.pi * a / L)

        return {
            "omega": np.asarray(omega, dtype=float),
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
        params = self.validate_longitudinal_params(params)

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

        mu_singularity_eps = float(params.get("mu_singularity_eps_longitudinal", 1e-9))
        if abs(1.0 - mu) <= mu_singularity_eps:
            raise ValueError(
                "Продольная модель вырождена при μ≈1: знаменатель 1-μ обращается в ноль. "
                "Уменьшите μ или задайте значение вдали от единицы."
            )

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

    def find_longitudinal_im0_points(self, params) -> dict:
        """Найти все пересечения δ(ω)=0 и критическую точку на единой сетке.

        Используется та же частотная сетка, что и для анализа/экспорта.
        Сначала учитываются точные нули, попавшие в узлы сетки, затем —
        интервалы смены знака между соседними узлами.
        """
        omega = BoreBarModel.build_frequency_grid(params, include_endpoint=True)
        K1, delta = self.compute_longitudinal_curve(params, omega)

        omega = np.asarray(omega, dtype=float)
        K1 = np.asarray(K1, dtype=float)
        delta = np.asarray(delta, dtype=float)

        points = []
        eps = float(params.get("im0_eps_longitudinal", 1e-9))

        finite_zero = np.isfinite(omega) & np.isfinite(K1) & np.isfinite(delta) & (np.abs(delta) <= eps)
        for idx in np.where(finite_zero)[0]:
            w = float(omega[idx])
            k = float(K1[idx])
            points.append({
                "omega": w,
                "K1": k,
                "delta": 0.0,
                "re": k,
                "im": 0.0,
                "frequency": float(w / (2.0 * np.pi)),
            })

        for i, j in self._sign_change_intervals(omega, delta):
            w1, w2 = omega[i], omega[j]
            d1, d2 = delta[i], delta[j]
            w0 = self._linear_root(w1, d1, w2, d2)
            k0 = np.interp(w0, [w1, w2], [K1[i], K1[j]])
            points.append({
                "omega": float(w0),
                "K1": float(k0),
                "delta": 0.0,
                "re": float(k0),
                "im": 0.0,
                "frequency": float(w0 / (2.0 * np.pi)),
            })

        dedup = []
        omega_tol = float(params.get("im0_omega_tol_longitudinal", max(float(params.get("omega_step", 1.0)) * 0.5, 1e-9)))
        re_tol = float(params.get("im0_re_tol_longitudinal", 1e-6))
        for p in sorted(points, key=lambda item: (item["omega"], item["re"])):
            if dedup and abs(p["omega"] - dedup[-1]["omega"]) <= omega_tol and abs(p["re"] - dedup[-1]["re"]) <= re_tol:
                continue
            dedup.append(p)

        critical = min(dedup, key=lambda p: p["K1"]) if dedup else None
        return {"points": dedup, "critical": critical}

    def _get_transverse_modal_data(self, params: dict) -> dict:
        """
        Вычислить модальные коэффициенты поперечной модели.

        Поддерживаются два явных варианта формы:
        - verified_cantilever_first_mode_phi: верифицированная 1-я собственная форма консольной балки;
        - project_maple_compatible_phi: старая проектная аппроксимация для совместимости.
        """
        params = self.validate_transverse_params(params)

        E = float(params["E"])
        rho = float(params["rho"])
        L = float(params["length"])
        R = float(params.get("R", 0.04))
        r = float(params.get("r", 0.035))

        S = np.pi * (R ** 2 - r ** 2)
        m = rho * S
        J = np.pi * (R ** 4 - r ** 4) / 4.0

        variant = str(params.get("transverse_modal_shape_variant", "verified_cantilever_first_mode_phi"))

        if variant == "verified_cantilever_first_mode_phi":
            lambda1 = float(params.get("transverse_lambda1", 1.875104068711961))
            k1 = lambda1 / L
            eta = (np.cosh(lambda1) + np.cos(lambda1)) / (np.sinh(lambda1) + np.sin(lambda1))

            def raw_phi(x: np.ndarray) -> np.ndarray:
                x = np.asarray(x, dtype=float)
                z = k1 * x
                return np.cosh(z) - np.cos(z) - eta * (np.sinh(z) - np.sin(z))

            def raw_phi_pp(x: np.ndarray) -> np.ndarray:
                x = np.asarray(x, dtype=float)
                z = k1 * x
                return (k1 ** 2) * (np.cosh(z) + np.cos(z) - eta * (np.sinh(z) + np.sin(z)))

            phi_L_raw = float(raw_phi(L))
            if not np.isfinite(phi_L_raw) or abs(phi_L_raw) < 1e-14:
                raise ValueError("Не удалось нормировать verified_cantilever_first_mode_phi: phi(L) слишком мало.")
            C = 1.0 / phi_L_raw
            modal_shape_source = "verified_cantilever_first_mode_phi"
            modal_shape_description = "Первая собственная форма консольной балки Эйлера–Бернулли, нормировка phi(L)=1"
            shape_normalization = "phi(L)=1"
        elif variant == "project_maple_compatible_phi":
            lambda1 = 1.875
            k1 = lambda1 / L
            eta = None
            C = float(params.get("transverse_shape_C", 0.7340955137589128))

            def raw_phi(x: np.ndarray) -> np.ndarray:
                x = np.asarray(x, dtype=float)
                z = k1 * x
                return np.sinh(z) - np.sin(z)

            def raw_phi_pp(x: np.ndarray) -> np.ndarray:
                x = np.asarray(x, dtype=float)
                z = k1 * x
                return (k1 ** 2) * (np.sinh(z) + np.sin(z))

            modal_shape_source = "project_maple_compatible_phi"
            modal_shape_description = "Старая проектная Maple-совместимая аппроксимация, оставлена как режим совместимости"
            shape_normalization = "fixed_C"
        else:
            raise ValueError(f"Неизвестный вариант поперечной формы: {variant}")

        def phi(x: np.ndarray) -> np.ndarray:
            return C * raw_phi(x)

        def phi_pp(x: np.ndarray) -> np.ndarray:
            return C * raw_phi_pp(x)

        x_grid = np.linspace(0.0, L, 2000)
        phi_vals = phi(x_grid)
        phi_pp_vals = phi_pp(x_grid)

        modal_mass_integral = float(np.trapezoid(phi_vals ** 2, x_grid))
        modal_curvature_integral = float(np.trapezoid(phi_pp_vals ** 2, x_grid))

        alpha = float(m * modal_mass_integral)
        gamma = float(E * J * modal_curvature_integral)

        h_explicit = params.get("h", None)
        beta_legacy = params.get("beta", None)
        if h_explicit is not None:
            h = float(h_explicit)
            beta = float(h * gamma)
            damping_source = "h"
        elif beta_legacy is not None:
            beta = float(beta_legacy)
            h = float(beta / gamma) if gamma != 0.0 else np.nan
            damping_source = "legacy_beta"
        else:
            h = 3.0214154483500606e-05
            beta = float(h * gamma)
            damping_source = "default_h"

        return {
            "E": E,
            "rho": rho,
            "length": L,
            "R": R,
            "r": r,
            "S": float(S),
            "m": float(m),
            "J": float(J),
            "phi": phi,
            "phi_pp": phi_pp,
            "phi_L": float(phi(L)),
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "h": float(h),
            "k1": float(k1),
            "shape_scale_C": float(C),
            "modal_mass_integral": modal_mass_integral,
            "modal_curvature_integral": modal_curvature_integral,
            "modal_shape_variant": variant,
            "modal_shape_source": modal_shape_source,
            "modal_shape_description": modal_shape_description,
            "shape_normalization": shape_normalization,
            "lambda1": float(lambda1),
            "shape_eta": None if eta is None else float(eta),
            "damping_source": damping_source,
            "model_variant": (
                "galerkin_one_mode_verified_shape"
                if variant == "verified_cantilever_first_mode_phi"
                else "galerkin_one_mode_project_shape"
            ),
        }

    def compute_transverse_curve(self, params: dict, omega: np.ndarray):
        """Вернуть (Re(W(iω)), Im(W(iω))) на заданной сетке omega."""
        result = self.calculate_transverse({**params, "omega_override": np.asarray(omega, dtype=float)})
        return np.asarray(result["W_real"], dtype=float), np.asarray(result["W_imag"], dtype=float)

    def build_transverse_display_curve(self, params: dict) -> dict:
        """
        Построить более плотную сетку только для отображения годографа.
        Экспортная/физическая сетка при этом не меняется.
        """
        base = self.calculate_transverse(params)
        base_omega = np.asarray(base["omega"], dtype=float)
        factor = int(params.get("display_refinement_factor_transverse", params.get("display_refinement_factor", 8)))
        max_points = int(params.get("display_max_points_transverse", 25000))
        display_omega = self._refine_frequency_grid(base_omega, factor=factor, max_points=max_points)

        if display_omega.size == base_omega.size:
            return {
                "omega": base_omega,
                "W_real": np.asarray(base["W_real"], dtype=float),
                "W_imag": np.asarray(base["W_imag"], dtype=float),
                "refined": False,
                "base_point_count": int(base_omega.size),
                "display_point_count": int(base_omega.size),
            }

        dense = self.calculate_transverse({**params, "omega_override": display_omega})
        return {
            "omega": np.asarray(dense["omega"], dtype=float),
            "W_real": np.asarray(dense["W_real"], dtype=float),
            "W_imag": np.asarray(dense["W_imag"], dtype=float),
            "refined": True,
            "base_point_count": int(base_omega.size),
            "display_point_count": int(display_omega.size),
        }

    def calculate_transverse(self, params: dict) -> dict:
        """
        Расчёт поперечных колебаний в одномодовой аппроксимации на полной сетке.

        Невалидные точки не удаляются, а помечаются через NaN, чтобы график
        имел корректные разрывы, а поиск пересечений не склеивал разные участки.
        """
        params = self.validate_transverse_params(params)

        mu = float(params["mu"])
        tau = float(params["tau"])
        K_cut = float(params.get("K_cut", 6e5))

        modal = self._get_transverse_modal_data(params)
        alpha = modal["alpha"]
        beta = modal["beta"]
        gamma = modal["gamma"]
        phi_L = modal["phi_L"]

        omega_override = params.get("omega_override")
        if omega_override is not None:
            omega = np.asarray(omega_override, dtype=float)
        else:
            omega = BoreBarModel.build_frequency_grid(params, include_endpoint=True)

        p = 1j * omega
        denom_eps = float(params.get("transverse_denom_eps", 1e-12))
        response_clip = float(params.get("transverse_response_clip", 1e8))
        if response_clip <= 0.0:
            response_clip = None

        with np.errstate(all="ignore"):
            numerator = (phi_L ** 2) * K_cut * (1.0 - mu * np.exp(-p * tau))
            denom = alpha * p ** 2 + beta * p + gamma
            denom_too_small = np.isfinite(denom.real) & np.isfinite(denom.imag) & (np.abs(denom) < denom_eps)
            denom_safe = np.where(denom_too_small, np.nan + 1j * np.nan, denom)
            W = numerator / denom_safe

        omega_nonfinite = ~np.isfinite(omega)
        response_nonfinite = ~(np.isfinite(W.real) & np.isfinite(W.imag))
        if response_clip is None:
            response_clip_mask = np.zeros(W.shape, dtype=bool)
        else:
            response_clip_mask = (
                ~response_nonfinite
                & ((np.abs(W.real) > response_clip) | (np.abs(W.imag) > response_clip))
            )

        invalid_reason_masks = {
            "omega_nonfinite": np.asarray(omega_nonfinite, dtype=bool),
            "denom_too_small": np.asarray(denom_too_small, dtype=bool),
            "response_nonfinite": np.asarray(response_nonfinite, dtype=bool),
            "response_clip": np.asarray(response_clip_mask, dtype=bool),
        }
        invalid_mask = np.zeros(W.shape, dtype=bool)
        for mask in invalid_reason_masks.values():
            invalid_mask |= np.asarray(mask, dtype=bool)

        W_clean = np.asarray(W, dtype=complex).copy()
        W_clean[invalid_mask] = np.nan + 1j * np.nan

        return {
            "omega": np.asarray(omega, dtype=float),
            "W_real": np.asarray(W_clean.real, dtype=float),
            "W_imag": np.asarray(W_clean.imag, dtype=float),
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "h": modal["h"],
            "phi_L": phi_L,
            "K_cut": K_cut,
            "R": modal["R"],
            "r": modal["r"],
            "J": modal["J"],
            "S": modal["S"],
            "k1": modal["k1"],
            "shape_scale_C": modal["shape_scale_C"],
            "modal_shape_variant": modal["modal_shape_variant"],
            "modal_shape_source": modal["modal_shape_source"],
            "modal_shape_description": modal["modal_shape_description"],
            "shape_normalization": modal["shape_normalization"],
            "lambda1": modal["lambda1"],
            "shape_eta": modal["shape_eta"],
            "modal_mass_integral": modal["modal_mass_integral"],
            "modal_curvature_integral": modal["modal_curvature_integral"],
            "damping_source": modal["damping_source"],
            "model_variant": modal["model_variant"],
            "invalid_mask": invalid_mask,
            "invalid_reason_masks": invalid_reason_masks,
            "invalid_reason_counts": self._count_true_map(invalid_reason_masks),
            "invalid_point_count": int(np.count_nonzero(invalid_mask)),
            "numerics_metadata": {
                "transverse_denom_eps": denom_eps,
                "transverse_response_clip": response_clip,
            },
        }

    def find_transverse_im0_points(self, params: dict) -> dict:
        """
        Находит все пересечения Im(W(iω))=0 и критическую точку.

        Сначала учитываются точные нули, попавшие прямо в узлы сетки,
        затем добавляются пересечения между соседними узлами со сменой
        знака. После этого близкие точки дедуплицируются, а critical
        выбирается как точка с минимальным Re.
        """
        res = self.calculate_transverse(params)
        omega = np.asarray(res["omega"], dtype=float)
        Wre = np.asarray(res["W_real"], dtype=float)
        Wim = np.asarray(res["W_imag"], dtype=float)

        points = []
        eps = float(params.get("im0_eps_transverse", 1e-9))

        finite_zero = np.isfinite(omega) & np.isfinite(Wre) & np.isfinite(Wim) & (np.abs(Wim) <= eps)
        for idx in np.where(finite_zero)[0]:
            w = float(omega[idx])
            points.append({
                "omega": w,
                "re": float(Wre[idx]),
                "im": 0.0,
                "frequency": float(w / (2.0 * np.pi)),
            })

        for i, j in BoreBarModel._sign_change_intervals(omega, Wim):
            w1, w2 = omega[i], omega[j]
            y1, y2 = Wim[i], Wim[j]
            w0 = BoreBarModel._linear_root(w1, y1, w2, y2)
            re0 = np.interp(w0, [w1, w2], [Wre[i], Wre[j]])
            points.append({
                "omega": float(w0),
                "re": float(re0),
                "im": 0.0,
                "frequency": float(w0 / (2.0 * np.pi)),
            })

        dedup = []
        omega_tol = float(params.get("im0_omega_tol_transverse", max(float(params.get("omega_step", 1.0)) * 0.5, 1e-9)))
        re_tol = float(params.get("im0_re_tol_transverse", 1e-6))
        for p in sorted(points, key=lambda item: (item["omega"], item["re"])):
            if dedup and abs(p["omega"] - dedup[-1]["omega"]) <= omega_tol and abs(p["re"] - dedup[-1]["re"]) <= re_tol:
                continue
            dedup.append(p)

        critical = min(dedup, key=lambda p: p["re"]) if dedup else None
        return {"points": dedup, "critical": critical}
