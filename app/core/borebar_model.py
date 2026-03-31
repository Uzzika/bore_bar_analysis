"""Математические модели крутильных, продольных и поперечных колебаний борштанги."""

import numpy as np
from scipy.optimize import root_scalar


class BoreBarModel:
    """Модельный слой проекта: расчёт кривых, special points и служебной диагностики."""

    # -------------------------------------------------------------------------
    # Вспомогательные функции
    # -------------------------------------------------------------------------

    @staticmethod
    def build_frequency_grid(params: dict, include_endpoint: bool = False) -> np.ndarray:
        """Построить единую частотную сетку для анализа, special points и экспорта."""
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
        """Устойчивая coth(z): около нуля — ряд, возле полюсов — NaN."""
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
        """Вернуть пары соседних индексов, между которыми есть смена знака."""
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
        """Линейно оценить корень y(x)=0 на отрезке [x1, x2]."""
        if y2 == y1:
            return 0.5 * (x1 + x2)
        return x1 - y1 * (x2 - x1) / (y2 - y1)

    @staticmethod
    def _refine_root_on_interval(func, x1: float, y1: float, x2: float, y2: float, *, xtol: float = 1e-10) -> float:
        """Уточнить корень методом Брента; при сбое вернуть линейную оценку."""
        x1 = float(x1)
        x2 = float(x2)
        y1 = float(y1)
        y2 = float(y2)

        if not (np.isfinite(x1) and np.isfinite(x2) and np.isfinite(y1) and np.isfinite(y2)):
            return BoreBarModel._linear_root(x1, y1, x2, y2)
        if x2 <= x1:
            return BoreBarModel._linear_root(x1, y1, x2, y2)
        if abs(y1) <= xtol:
            return x1
        if abs(y2) <= xtol:
            return x2
        if y1 * y2 >= 0.0:
            return BoreBarModel._linear_root(x1, y1, x2, y2)

        try:
            sol = root_scalar(func, bracket=[x1, x2], method='brentq', xtol=xtol)
            if sol.converged and np.isfinite(sol.root):
                return float(sol.root)
        except Exception:
            pass
        return BoreBarModel._linear_root(x1, y1, x2, y2)

    @staticmethod
    def _deduplicate_special_points(points: list[dict], *, omega_tol: float, re_tol: float) -> list[dict]:
        dedup = []
        for p in sorted(points, key=lambda item: (item['omega'], item['re'])):
            if dedup and abs(p['omega'] - dedup[-1]['omega']) <= omega_tol and abs(p['re'] - dedup[-1]['re']) <= re_tol:
                continue
            dedup.append(p)
        return dedup

    @staticmethod
    def _build_zero_crossing_points(x: np.ndarray, re_values: np.ndarray, im_values: np.ndarray, *,
                                    zero_eps: float, omega_tol: float, re_tol: float,
                                    refine_func=None, re_eval_func=None, re_key: str = 're') -> list[dict]:
        x = np.asarray(x, dtype=float)
        re_values = np.asarray(re_values, dtype=float)
        im_values = np.asarray(im_values, dtype=float)

        points = []
        finite_zero = np.isfinite(x) & np.isfinite(re_values) & np.isfinite(im_values) & (np.abs(im_values) <= zero_eps)
        for idx in np.where(finite_zero)[0]:
            w = float(x[idx])
            r = float(re_values[idx])
            points.append({
                'omega': w,
                re_key: r,
                're': r,
                'im': 0.0,
                'frequency': float(w / (2.0 * np.pi)),
            })

        for i, j in BoreBarModel._sign_change_intervals(x, im_values):
            w1, w2 = float(x[i]), float(x[j])
            y1, y2 = float(im_values[i]), float(im_values[j])
            if refine_func is not None:
                w0 = BoreBarModel._refine_root_on_interval(refine_func, w1, y1, w2, y2)
            else:
                w0 = BoreBarModel._linear_root(w1, y1, w2, y2)
            if re_eval_func is not None:
                r0 = float(re_eval_func(w0))
            else:
                r0 = float(np.interp(w0, [w1, w2], [re_values[i], re_values[j]]))
            points.append({
                'omega': float(w0),
                re_key: r0,
                're': r0,
                'im': 0.0,
                'frequency': float(w0 / (2.0 * np.pi)),
            })

        return BoreBarModel._deduplicate_special_points(points, omega_tol=omega_tol, re_tol=re_tol)

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
        variant = str(validated.get("transverse_modal_shape_variant", "verified_cantilever_first_mode_phi"))
        if variant != "verified_cantilever_first_mode_phi":
            raise ValueError(
                "Поперечная модель поддерживает только verified_cantilever_first_mode_phi; "
                f"получено: {variant}"
            )
        validated["transverse_modal_shape_variant"] = variant

        BoreBarModel._validate_frequency_params(validated)
        return validated

    @staticmethod
    def _count_true_map(mask_map: dict[str, np.ndarray]) -> dict[str, int]:
        """Преобразовать набор булевых масок в счётчики True."""
        counts = {}
        for key, mask in mask_map.items():
            counts[key] = int(np.count_nonzero(np.asarray(mask, dtype=bool)))
        return counts

    @staticmethod
    def _refine_frequency_grid(omega: np.ndarray, factor: int = 1, max_points: int = 25000) -> np.ndarray:
        """Уплотнить сетку только для display-построения."""
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
        """Собрать маски и счётчики невалидных точек крутильной ветви."""
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
        """Построить физическую крутильную сетку только для ω > 0."""
        omega_start = float(params.get("omega_start", 0.001))
        omega_end = float(params.get("omega_end", omega_start))
        omega_step = float(params.get("omega_step", 0.1))

        if omega_end <= 0.0:
            return np.array([], dtype=float)

        if omega_start > 0.0:
            omega_pos = BoreBarModel.build_frequency_grid(params, include_endpoint=False)
            return np.asarray(omega_pos[omega_pos > 0.0], dtype=float)

        ratio = float(omega_end / omega_step)
        n_last = int(np.floor(np.nextafter(ratio, -np.inf)))
        if n_last < 1:
            return np.array([], dtype=float)

        return omega_step * np.arange(1, n_last + 1, dtype=float)

    @staticmethod
    def _evaluate_torsional_sigma_positive(params: dict, omega_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Вычислить σ(iω) на физической положительной ветви."""
        params = BoreBarModel.validate_torsional_params(params)

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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Собрать display-ветвь для GUI из физической положительной ветви."""
        omega_pos = np.asarray(omega_pos, dtype=float)
        sigma_real_pos = np.asarray(sigma_real_pos, dtype=float)
        sigma_imag_pos = np.asarray(sigma_imag_pos, dtype=float)

        if omega_pos.size == 0:
            empty = np.array([], dtype=float)
            return empty, empty, empty, {
                "display_outlier_count": 0,
                "display_outlier_threshold_re": None,
                "display_outlier_threshold_im": None,
                "display_outlier_quantile": None,
                "display_outlier_expand": None,
                "display_outlier_policy": "no_points",
            }

        re_display_pos = sigma_real_pos.copy()
        im_display_pos = sigma_imag_pos.copy()

        finite_mask = (
            np.isfinite(re_display_pos)
            & np.isfinite(im_display_pos)
        )

        robust_q = float(params.get("display_outlier_quantile", 0.995))
        robust_expand = float(params.get("display_outlier_expand", 4.0))
        use_robust_filter = bool(params.get("display_filter_outliers", False))

        display_outlier_mask = np.zeros_like(re_display_pos, dtype=bool)
        thr_re = None
        thr_im = None

        if use_robust_filter and np.count_nonzero(finite_mask) >= 8:
            robust_q = min(max(robust_q, 0.90), 0.9999)
            robust_expand = max(1.0, robust_expand)

            finite_re = np.abs(re_display_pos[finite_mask])
            finite_im = np.abs(im_display_pos[finite_mask])

            thr_re = float(np.quantile(finite_re, robust_q)) * robust_expand
            thr_im = float(np.quantile(finite_im, robust_q)) * robust_expand

            # Если вся кривая сама по себе крупная, пороги должны быть не меньше max,
            # чтобы не скрыть нормальную ветвь из-за почти константного диапазона.
            if np.isfinite(thr_re):
                thr_re = max(thr_re, float(np.nanmedian(finite_re)) * robust_expand)
            if np.isfinite(thr_im):
                thr_im = max(thr_im, float(np.nanmedian(finite_im)) * robust_expand)

            display_outlier_mask = (
                finite_mask
                & (
                    ((np.abs(re_display_pos) > thr_re) if thr_re is not None else False)
                    | ((np.abs(im_display_pos) > thr_im) if thr_im is not None else False)
                )
            )

            if np.any(display_outlier_mask):
                re_display_pos[display_outlier_mask] = np.nan
                im_display_pos[display_outlier_mask] = np.nan

        omega_start = float(params.get("omega_start", 0.0))
        if omega_start < 0.0:
            omega_neg = -omega_pos[::-1]
            re_neg = re_display_pos[::-1]
            im_neg = -im_display_pos[::-1]

            omega_display = np.concatenate([omega_neg, omega_pos])
            sigma_real_display = np.concatenate([re_neg, re_display_pos])
            sigma_imag_display = np.concatenate([im_neg, im_display_pos])
        else:
            omega_display = omega_pos.copy()
            sigma_real_display = re_display_pos.copy()
            sigma_imag_display = im_display_pos.copy()

        return omega_display, sigma_real_display, sigma_imag_display, {
            "display_outlier_count": int(np.count_nonzero(display_outlier_mask)),
            "display_outlier_threshold_re": thr_re,
            "display_outlier_threshold_im": thr_im,
            "display_outlier_quantile": robust_q if use_robust_filter else None,
            "display_outlier_expand": robust_expand if use_robust_filter else None,
            "display_outlier_policy": "robust_nan_gap_filter" if use_robust_filter else "disabled",
        }


    @staticmethod
    def build_torsional_plot_im0_from_result(result: dict, params: dict, semantic_im0: dict | None = None) -> dict:
        """Подогнать plot-маркеры к той же sampled physical-ветви, что рисуется в GUI."""
        omega = np.asarray(result.get("physical_omega", result.get("omega", [])), dtype=float)
        sig_re = np.asarray(result.get("physical_sigma_real", result.get("sigma_real", [])), dtype=float)
        sig_im = np.asarray(result.get("physical_sigma_imag", result.get("sigma_imag", [])), dtype=float)

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
            w1, w2 = float(omega[i]), float(omega[j])
            y1, y2 = float(sig_im[i]), float(sig_im[j])
            w0 = float(BoreBarModel._linear_root(w1, y1, w2, y2))
            re0 = float(np.interp(w0, [w1, w2], [sig_re[i], sig_re[j]]))
            points.append({
                "omega": w0,
                "re": re0,
                "im": 0.0,
                "frequency": float(w0 / (2.0 * np.pi)),
            })

        omega_tol = float(params.get("im0_omega_tol_torsional", max(float(params.get("omega_step", 1.0)) * 0.5, 1e-9)))
        re_tol = float(params.get("im0_re_tol_torsional", 1e-6))
        dedup = BoreBarModel._deduplicate_special_points(points, omega_tol=omega_tol, re_tol=re_tol)

        semantic = semantic_im0 or {}
        research_sem = semantic.get("research_critical_point")
        minre_sem = semantic.get("minimum_re_critical_point")

        def _match_plot_point(target: dict | None) -> dict | None:
            if not target or not dedup:
                return None
            tw = float(target.get("omega", np.nan))
            if not np.isfinite(tw):
                return None
            return min(
                dedup,
                key=lambda p: (
                    abs(float(p["omega"]) - tw),
                    abs(float(p["re"]) - float(target.get("re", p["re"]))),
                ),
            )

        research_plot = _match_plot_point(research_sem)
        minre_plot = _match_plot_point(minre_sem)

        return {
            "points": dedup,
            "critical": research_plot,
            "research_critical_point": research_plot,
            "minimum_re_critical_point": minre_plot,
            "source_curve": "physical_sampled_branch_for_plot",
        }

    @staticmethod
    def _augment_torsional_positive_branch_with_special_points(result: dict, params: dict, points: list[dict] | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Вставить physical special points в положительную ветвь по порядку ω."""
        omega_pos = np.asarray(result.get("physical_omega", result.get("omega", [])), dtype=float)
        re_pos = np.asarray(result.get("physical_sigma_real", result.get("sigma_real", [])), dtype=float)
        im_pos = np.asarray(result.get("physical_sigma_imag", result.get("sigma_imag", [])), dtype=float)

        if omega_pos.size == 0 or not points:
            return omega_pos.copy(), re_pos.copy(), im_pos.copy(), 0

        step = float(params.get("omega_step", 1.0))
        omega_tol = float(max(1e-12, step * 1e-9))

        items: list[tuple[float, float, float, int]] = []
        for w, r, i in zip(omega_pos, re_pos, im_pos):
            if np.isfinite(w) and np.isfinite(r) and np.isfinite(i):
                items.append((float(w), float(r), float(i), 0))

        inserted_count = 0
        for p in points:
            w = float(p.get("omega", np.nan))
            r = float(p.get("re", np.nan))
            i = float(p.get("im", 0.0))
            if not (np.isfinite(w) and np.isfinite(r) and np.isfinite(i)):
                continue
            if w <= 0.0:
                continue
            items.append((w, r, i, 1))
            inserted_count += 1

        if not items:
            return omega_pos.copy(), re_pos.copy(), im_pos.copy(), 0

        items.sort(key=lambda t: (t[0], t[3]))

        merged: list[tuple[float, float, float]] = []
        for w, r, i, is_special in items:
            if merged and abs(w - merged[-1][0]) <= omega_tol:
                if is_special:
                    merged[-1] = (w, r, i)
                continue
            merged.append((w, r, i))

        omega_aug = np.asarray([t[0] for t in merged], dtype=float)
        re_aug = np.asarray([t[1] for t in merged], dtype=float)
        im_aug = np.asarray([t[2] for t in merged], dtype=float)
        return omega_aug, re_aug, im_aug, inserted_count

    @staticmethod
    def build_torsional_display_curve_from_result(result: dict, params: dict, points: list[dict] | None = None) -> dict:
        omega_pos, re_pos, im_pos, inserted_count = BoreBarModel._augment_torsional_positive_branch_with_special_points(result, params, points)

        if omega_pos.size == 0:
            empty = np.array([], dtype=float)
            return {
                "omega": empty,
                "re": empty,
                "im": empty,
                "inserted_special_points": 0,
                "policy": "display_rebuilt_from_physical_branch_no_points",
            }

        if float(params.get("omega_start", 0.0)) < 0.0:
            omega = np.concatenate([-omega_pos[::-1], omega_pos])
            re = np.concatenate([re_pos[::-1], re_pos])
            im = np.concatenate([-im_pos[::-1], im_pos])
            policy = "display_rebuilt_as_exact_conjugate_mirror_of_physical_positive_branch"
        else:
            omega = omega_pos.copy()
            re = re_pos.copy()
            im = im_pos.copy()
            policy = "display_rebuilt_from_physical_positive_branch"

        return {
            "omega": omega,
            "re": re,
            "im": im,
            "inserted_special_points": int(inserted_count),
            "policy": policy,
        }

    @staticmethod
    def build_torsional_plot_policy(display_curve: dict, points=None, critical=None) -> dict:
        re = np.asarray(display_curve.get("re", []), dtype=float)
        im = np.asarray(display_curve.get("im", []), dtype=float)
        finite = np.isfinite(re) & np.isfinite(im)

        if np.count_nonzero(finite) < 2:
            return {
                "xlim": (-1.0, 1.0),
                "ylim": (-1.0, 1.0),
                "y_clip": (-1.0, 1.0),
                "origin_band_left": 0.0,
                "mode": "fallback_empty",
            }

        x_all = re[finite]
        y_all = im[finite]

        if points:
            px = np.asarray([p["re"] for p in points if np.isfinite(p.get("re", np.nan))], dtype=float)
            if px.size:
                x_all = np.concatenate([x_all, px])
        if critical and np.isfinite(critical.get("re", np.nan)):
            x_all = np.concatenate([x_all, np.asarray([float(critical["re"])], dtype=float)])

        xmin = float(np.min(x_all))
        xmax = float(np.max(x_all))
        xspan = max(xmax - xmin, 1e-9)
        xpad = xspan * 0.08
        xlim = (xmin - xpad, xmax + xpad)

        origin_band_fraction = 0.06
        origin_band_left = xmax - xspan * origin_band_fraction

        non_origin_mask = finite & (re < origin_band_left)
        if np.count_nonzero(non_origin_mask) >= 8:
            y_for_view = np.abs(im[non_origin_mask])
            mode = "origin_only_segment_clip"
        else:
            y_for_view = np.abs(y_all)
            mode = "fallback_full_curve_no_origin_separation"

        ymax_visible = float(np.max(y_for_view)) if y_for_view.size else float(np.max(np.abs(y_all)))
        ymax_visible = max(ymax_visible, 1e-9)

        y_clip_max = ymax_visible * 1.03
        y_window_max = ymax_visible * 1.12

        return {
            "xlim": xlim,
            "ylim": (-y_window_max, y_window_max),
            "y_clip": (-y_clip_max, y_clip_max),
            "origin_band_left": origin_band_left,
            "mode": mode,
        }

    @staticmethod
    def _clip_segment_to_horizontal_strip(x1, y1, x2, y2, ylo, yhi):
        eps = 1e-15
        dy = y2 - y1

        if abs(dy) <= eps:
            if ylo <= y1 <= yhi:
                return [(x1, y1), (x2, y2)], False
            return [], True

        t_a = (ylo - y1) / dy
        t_b = (yhi - y1) / dy
        t_enter = max(0.0, min(t_a, t_b))
        t_exit = min(1.0, max(t_a, t_b))

        if t_enter > t_exit:
            return [], True

        xa = x1 + (x2 - x1) * t_enter
        ya = y1 + dy * t_enter
        xb = x1 + (x2 - x1) * t_exit
        yb = y1 + dy * t_exit

        partial = not (np.isclose(t_enter, 0.0) and np.isclose(t_exit, 1.0))
        return [(xa, ya), (xb, yb)], partial

    @staticmethod
    def build_torsional_plot_curve(display_curve: dict, plot_policy: dict) -> dict:
        re = np.asarray(display_curve.get("re", []), dtype=float)
        im = np.asarray(display_curve.get("im", []), dtype=float)
        n = re.size

        ylo, yhi = plot_policy["y_clip"]
        origin_band_left = float(plot_policy["origin_band_left"])

        out_x = []
        out_y = []
        clipped_segments = 0

        def append_point(xv, yv):
            if out_x and np.isfinite(out_x[-1]) and np.isfinite(out_y[-1]):
                if np.isclose(out_x[-1], xv) and np.isclose(out_y[-1], yv):
                    return
            out_x.append(float(xv))
            out_y.append(float(yv))

        def append_gap():
            if not out_x or np.isfinite(out_x[-1]) or np.isfinite(out_y[-1]):
                out_x.append(np.nan)
                out_y.append(np.nan)

        if n == 0:
            return {
                "re": np.array([], dtype=float),
                "im": np.array([], dtype=float),
                "clipped_count": 0,
                "policy": "origin_only_segment_clip_no_points",
            }

        for i in range(n - 1):
            x1, y1 = re[i], im[i]
            x2, y2 = re[i + 1], im[i + 1]

            if not (np.isfinite(x1) and np.isfinite(y1) and np.isfinite(x2) and np.isfinite(y2)):
                append_gap()
                continue

            near_origin_spike_zone = max(x1, x2) >= origin_band_left

            if not near_origin_spike_zone:
                append_point(x1, y1)
                append_point(x2, y2)
                continue

            kept, partial = BoreBarModel._clip_segment_to_horizontal_strip(x1, y1, x2, y2, ylo, yhi)
            if not kept:
                clipped_segments += 1
                append_gap()
                continue

            if partial:
                clipped_segments += 1
                append_point(*kept[0])
                append_point(*kept[-1])
                append_gap()
            else:
                append_point(*kept[0])
                append_point(*kept[-1])

        return {
            "re": np.asarray(out_x, dtype=float),
            "im": np.asarray(out_y, dtype=float),
            "clipped_count": int(clipped_segments),
            "policy": "segment_clip_only_for_spikes_near_origin",
        }

    def calculate_torsional(self, params: dict) -> dict:
        """Рассчитать крутильную модель и вернуть физическую ветвь, display-ветвь и диагностику."""
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

        omega_display, sig_re_display, sig_im_display, display_metadata = self._build_torsional_display_curve(
            params=params,
            omega_pos=omega_pos,
            sigma_real_pos=sig_re_pos,
            sigma_imag_pos=sig_im_pos,
        )

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
                **display_metadata,
            },
            "model_variant": "torsional_physical_positive_plus_model_display_symmetry",
            "negative_frequency_policy": "display_curve_is_built_as_conjugate_mirror_of_positive_branch",
        }

    @staticmethod
    def _torsional_research_policy(params: dict) -> dict:
        """Параметры исследовательского окна выбора критической точки."""
        length = float(params.get("length", 0.0))
        low = float(params.get("torsional_research_root_window_low", 500.0))
        high_default = float(params.get("torsional_research_root_window_high", 2000.0))
        high_long = float(params.get("torsional_research_root_window_high_long", 1000.0))
        long_threshold = float(params.get("torsional_research_long_length_threshold", 5.5))

        high = high_long if length >= long_threshold else high_default
        if high < low:
            high = low

        return {
            "kind": "first_im0_in_research_window",
            "omega_low": low,
            "omega_high": high,
            "length": length,
            "long_length_threshold": long_threshold,
            "long_window_applied": bool(length >= long_threshold),
        }

    @staticmethod
    def _select_torsional_research_critical_point(points: list[dict], params: dict) -> tuple[dict | None, dict]:
        """Выбрать критическую точку по исследовательскому правилу окна ω."""
        policy = BoreBarModel._torsional_research_policy(params)
        if not points:
            return None, {**policy, "selection_status": "no_im0_points"}

        ordered = sorted(points, key=lambda item: (item["omega"], item["re"]))
        low = float(policy["omega_low"])
        high = float(policy["omega_high"])

        in_window = [p for p in ordered if low <= float(p["omega"]) <= high]
        if in_window:
            return dict(in_window[0]), {**policy, "selection_status": "first_point_in_window"}

        after_low = [p for p in ordered if float(p["omega"]) >= low]
        if after_low:
            return dict(after_low[0]), {**policy, "selection_status": "fallback_first_point_after_window_low"}

        return dict(ordered[0]), {**policy, "selection_status": "fallback_first_available_point"}

    @staticmethod
    def find_torsional_im0_points(params: dict) -> dict:
        """Найти пересечения Im(σ)=0 на физической положительной ветви."""
        model = BoreBarModel()
        res = model.calculate_torsional(params)

        omega = np.asarray(res["physical_omega"], dtype=float)
        sig_re = np.asarray(res["physical_sigma_real"], dtype=float)
        sig_im = np.asarray(res["physical_sigma_imag"], dtype=float)

        eps = float(params.get("im0_eps_torsional", 1e-9))
        omega_tol = float(params.get("im0_omega_tol_torsional", max(float(params.get("omega_step", 1.0)) * 0.5, 1e-9)))
        re_tol = float(params.get("im0_re_tol_torsional", 1e-6))

        def im_func(w):
            _, re_v, im_v, _ = BoreBarModel._evaluate_torsional_sigma_positive(params, np.asarray([w], dtype=float))
            return float(im_v[0])

        def re_func(w):
            _, re_v, im_v, _ = BoreBarModel._evaluate_torsional_sigma_positive(params, np.asarray([w], dtype=float))
            return float(re_v[0])

        dedup = BoreBarModel._build_zero_crossing_points(
            omega, sig_re, sig_im,
            zero_eps=eps,
            omega_tol=omega_tol,
            re_tol=re_tol,
            refine_func=im_func,
            re_eval_func=re_func,
            re_key="re",
        )

        research_critical_point, policy_meta = BoreBarModel._select_torsional_research_critical_point(dedup, params)
        minimum_re_critical_point = min(dedup, key=lambda p: p["re"]) if dedup else None

        return {
            "all_im0_points": dedup,
            "research_critical_point": research_critical_point,
            "minimum_re_critical_point": minimum_re_critical_point,
            "points": dedup,
            "source_curve": "physical_positive_branch",
            "critical_selection_policy": policy_meta,
        }

    def find_torsional_im0_points_from_result(self, params: dict, result: dict) -> dict:
        """Найти пересечения Im(σ)=0 по уже рассчитанной крутильной кривой."""
        omega = np.asarray(result.get("physical_omega", result.get("omega", [])), dtype=float)
        sig_re = np.asarray(result.get("physical_sigma_real", result.get("sigma_real", [])), dtype=float)
        sig_im = np.asarray(result.get("physical_sigma_imag", result.get("sigma_imag", [])), dtype=float)

        eps = float(params.get("im0_eps_torsional", 1e-9))
        omega_tol = float(params.get("im0_omega_tol_torsional", max(float(params.get("omega_step", 1.0)) * 0.5, 1e-9)))
        re_tol = float(params.get("im0_re_tol_torsional", 1e-6))

        def im_func(w):
            _, re_v, im_v, _ = BoreBarModel._evaluate_torsional_sigma_positive(params, np.asarray([w], dtype=float))
            return float(im_v[0])

        def re_func(w):
            _, re_v, im_v, _ = BoreBarModel._evaluate_torsional_sigma_positive(params, np.asarray([w], dtype=float))
            return float(re_v[0])

        dedup = BoreBarModel._build_zero_crossing_points(
            omega, sig_re, sig_im,
            zero_eps=eps,
            omega_tol=omega_tol,
            re_tol=re_tol,
            refine_func=im_func,
            re_eval_func=re_func,
            re_key="re",
        )

        research_critical_point, policy_meta = BoreBarModel._select_torsional_research_critical_point(dedup, params)
        minimum_re_critical_point = min(dedup, key=lambda p: p["re"]) if dedup else None

        return {
            "all_im0_points": dedup,
            "research_critical_point": research_critical_point,
            "minimum_re_critical_point": minimum_re_critical_point,
            "points": dedup,
            "source_curve": "physical_positive_branch",
            "critical_selection_policy": policy_meta,
        }

    # -------------------------------------------------------------------------
    # Продольные колебания
    # -------------------------------------------------------------------------

    @staticmethod
    def calculate_longitudinal(params: dict) -> dict:
        """Рассчитать продольную SI-модель на общей частотной сетке."""
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

        curve_data = BoreBarModel().compute_longitudinal_curve(params, omega, return_diagnostics=True)
        K1 = curve_data["K1"]
        delta = curve_data["delta"]

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
            "invalid_mask": curve_data["invalid_mask"],
            "invalid_reason_masks": curve_data["invalid_reason_masks"],
            "invalid_reason_counts": curve_data["invalid_reason_counts"],
            "invalid_point_count": curve_data["invalid_point_count"],
            "numerics_metadata": curve_data["numerics_metadata"],
            "model_variant": "si_wave_speed",
            "longitudinal_model_regime": "research_si_interpretation",
            "longitudinal_model_regime_label": "SI-интерпретация исследовательской постановки",
            "longitudinal_model_scope": "research_aligned_generalized_si_form",
            "longitudinal_model_note": (
                "Продольная часть реализована как физически согласованная SI-формулировка "
                "исследовательской модели K₁–δ; это не побуквенное копирование всех обозначений "
                "исходного Matlab-фрагмента, а согласованная инженерная интерпретация формул."
            ),
            "research_alignment_status": "si_interpretation_of_research_formulas",
            "curve_parameterization": "omega -> (K1(omega), delta(omega))",
            "zero_frequency_limit_policy": "analytic_limits_used_for_summary_only",
        }

    def compute_longitudinal_curve(self, params: dict, omega: np.ndarray, return_diagnostics: bool = False):
        """Вычислить массивы K1(ω) и δ(ω) на заданной сетке ω."""
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

        K1_max = float(params.get("K1_max_longitudinal", 1e10))
        delta_max = float(params.get("delta_max_longitudinal", 1e7))
        response_clip = (np.abs(K1) > K1_max) | (np.abs(delta) > delta_max)
        K1[response_clip] = np.nan
        delta[response_clip] = np.nan

        invalid_reason_masks = {
            "omega_nonfinite": ~np.isfinite(omega),
            "cot_singularity": ~mask_cot,
            "denominator_too_small": ~mask_denom,
            "response_clip": np.asarray(response_clip, dtype=bool),
            "response_nonfinite": ~(np.isfinite(K1) & np.isfinite(delta)),
        }
        invalid_mask = np.zeros_like(omega, dtype=bool)
        for mask in invalid_reason_masks.values():
            invalid_mask |= np.asarray(mask, dtype=bool)

        if return_diagnostics:
            return {
                "K1": K1,
                "delta": delta,
                "invalid_mask": invalid_mask,
                "invalid_reason_masks": invalid_reason_masks,
                "invalid_reason_counts": self._count_true_map(invalid_reason_masks),
                "invalid_point_count": int(np.count_nonzero(invalid_mask)),
                "numerics_metadata": {
                    "cot_eps": eps,
                    "denominator_eps": eps,
                    "K1_max_longitudinal": K1_max,
                    "delta_max_longitudinal": delta_max,
                },
            }

        return K1, delta

    def _evaluate_longitudinal_point(self, params: dict, omega_value: float) -> tuple[float, float]:
        data = self.compute_longitudinal_curve(params, np.asarray([omega_value], dtype=float), return_diagnostics=True)
        return float(data["K1"][0]), float(data["delta"][0])

    def find_longitudinal_im0_points(self, params) -> dict:
        omega = BoreBarModel.build_frequency_grid(params, include_endpoint=True)
        res = self.calculate_longitudinal({**params, "omega_override": omega})
        return self.find_longitudinal_im0_points_from_result(params, res)

    def _select_longitudinal_research_critical_point(self, points: list[dict], params: dict) -> tuple[dict | None, dict]:
        """Выбрать точку δ(ω)=0 с минимальным K1."""
        policy = {
            "kind": "minimum_K1_on_delta_zero_set",
            "criterion": "min_K1",
            "model_regime": "fixed_project_si_model",
        }
        if not points:
            return None, {**policy, "selection_status": "no_im0_points"}

        selected = min(points, key=lambda p: (float(p["K1"]), float(p["omega"])))
        chosen = dict(selected)
        chosen["delta"] = 0.0
        return chosen, {**policy, "selection_status": "minimum_K1_point_selected"}

    def find_longitudinal_im0_points_from_result(self, params: dict, result: dict) -> dict:
        """Найти пересечения δ(ω)=0 и выбрать критическую точку на общей сетке."""
        omega = np.asarray(result["omega"], dtype=float)
        K1 = np.asarray(result["K1"], dtype=float)
        delta = np.asarray(result["delta"], dtype=float)

        eps = float(params.get("im0_eps_longitudinal", 1e-9))
        omega_tol = float(params.get("im0_omega_tol_longitudinal", max(float(params.get("omega_step", 1.0)) * 0.5, 1e-9)))
        re_tol = float(params.get("im0_re_tol_longitudinal", 1e-6))

        def delta_func(w):
            return self._evaluate_longitudinal_point(params, w)[1]

        def k1_func(w):
            return self._evaluate_longitudinal_point(params, w)[0]

        dedup = self._build_zero_crossing_points(
            omega, K1, delta,
            zero_eps=eps,
            omega_tol=omega_tol,
            re_tol=re_tol,
            refine_func=delta_func,
            re_eval_func=k1_func,
            re_key="K1",
        )

        for point in dedup:
            point["delta"] = 0.0

        research_critical_point, policy_meta = self._select_longitudinal_research_critical_point(dedup, params)
        minimum_re_critical_point = min(dedup, key=lambda p: (float(p["re"]), float(p["omega"]))) if dedup else None
        if minimum_re_critical_point is not None:
            minimum_re_critical_point = dict(minimum_re_critical_point)
            minimum_re_critical_point["delta"] = 0.0

        return {
            "points": dedup,
            "research_critical_point": research_critical_point,
            "minimum_re_critical_point": minimum_re_critical_point,
            "critical": research_critical_point,
            "source_curve": "direct_longitudinal_curve",
            "critical_selection_policy": policy_meta,
        }

    def _get_transverse_modal_data(self, params: dict) -> dict:
        """Подготовить модальные коэффициенты поперечной модели."""
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
            transverse_model_regime = "project_fixed_verified_mode"
            transverse_model_regime_label = "Проектный fixed verified mode: верифицированная собственная форма"
            transverse_model_scope = "project_model_fixed_verified_mode"
            transverse_model_note = (
                "Используется только верифицированная первая собственная форма консольной балки Эйлера–Бернулли; "
                "это зафиксированная проектная одномодовая модель, исторически выросшая из исследования, "
                "но не являющаяся literal-режимом совместимости с Maple."
            )
            research_alignment_status = "project_fixed_verified_mode_grown_from_research_not_maple_reproduction"

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
        else:
            raise ValueError(
                "Поперечная модель поддерживает только verified_cantilever_first_mode_phi; "
                f"получено: {variant}"
            )

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
        if h_explicit is not None:
            h = float(h_explicit)
            beta = float(h * gamma)
            damping_source = "h"
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
            "transverse_model_regime": transverse_model_regime,
            "transverse_model_regime_label": transverse_model_regime_label,
            "transverse_model_scope": transverse_model_scope,
            "transverse_model_note": transverse_model_note,
            "research_alignment_status": research_alignment_status,
            "model_variant": "galerkin_one_mode_verified_shape",
        }

    def _evaluate_transverse_response(self, params: dict, omega: np.ndarray, modal: dict | None = None, *, return_full: bool = False):
        """Низкоуровнево вычислить W(iω) на произвольной сетке ω."""
        params = self.validate_transverse_params(params)
        if modal is None:
            modal = self._get_transverse_modal_data(params)

        mu = float(params["mu"])
        tau = float(params["tau"])
        K_cut = float(params.get("K_cut", 6e5))

        alpha = float(modal["alpha"])
        beta = float(modal["beta"])
        gamma = float(modal["gamma"])
        phi_L = float(modal["phi_L"])

        omega = np.asarray(omega, dtype=float)
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

        payload = {
            "omega": np.asarray(omega, dtype=float),
            "W_real": np.asarray(W_clean.real, dtype=float),
            "W_imag": np.asarray(W_clean.imag, dtype=float),
            "invalid_mask": invalid_mask,
            "invalid_reason_masks": invalid_reason_masks,
            "invalid_reason_counts": self._count_true_map(invalid_reason_masks),
            "invalid_point_count": int(np.count_nonzero(invalid_mask)),
            "numerics_metadata": {
                "transverse_denom_eps": denom_eps,
                "transverse_response_clip": response_clip,
            },
        }
        if return_full:
            payload.update({
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
                "transverse_model_regime": modal.get("transverse_model_regime"),
                "transverse_model_regime_label": modal.get("transverse_model_regime_label"),
                "transverse_model_scope": modal.get("transverse_model_scope"),
                "transverse_model_note": modal.get("transverse_model_note"),
                "research_alignment_status": modal.get("research_alignment_status"),
            })
        return payload

    def compute_transverse_curve(self, params: dict, omega: np.ndarray):
        """Вернуть Re(W) и Im(W) на заданной сетке ω."""
        modal = self._get_transverse_modal_data(params)
        result = self._evaluate_transverse_response(params, np.asarray(omega, dtype=float), modal=modal, return_full=False)
        return np.asarray(result["W_real"], dtype=float), np.asarray(result["W_imag"], dtype=float)

    def build_transverse_display_curve(self, params: dict) -> dict:
        base = self.calculate_transverse(params)
        return self.build_transverse_display_curve_from_result(params, base)

    def build_transverse_display_curve_from_result(self, params: dict, base: dict) -> dict:
        """Построить более плотную display-кривую без пересборки модальной части."""
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
                "solver_path": "reuse_base_curve_without_resampling",
            }

        modal = {
            "alpha": float(base["alpha"]),
            "beta": float(base["beta"]),
            "gamma": float(base["gamma"]),
            "h": float(base["h"]),
            "phi_L": float(base["phi_L"]),
            "R": float(base["R"]),
            "r": float(base["r"]),
            "J": float(base["J"]),
            "S": float(base["S"]),
            "k1": float(base["k1"]),
            "shape_scale_C": float(base["shape_scale_C"]),
            "modal_shape_variant": base["modal_shape_variant"],
            "modal_shape_source": base["modal_shape_source"],
            "modal_shape_description": base["modal_shape_description"],
            "shape_normalization": base["shape_normalization"],
            "lambda1": float(base["lambda1"]),
            "shape_eta": base["shape_eta"],
            "modal_mass_integral": float(base["modal_mass_integral"]),
            "modal_curvature_integral": float(base["modal_curvature_integral"]),
            "damping_source": base["damping_source"],
            "model_variant": base["model_variant"],
            "transverse_model_regime": base.get("transverse_model_regime"),
            "transverse_model_regime_label": base.get("transverse_model_regime_label"),
            "transverse_model_scope": base.get("transverse_model_scope"),
            "transverse_model_note": base.get("transverse_model_note"),
            "research_alignment_status": base.get("research_alignment_status"),
        }
        dense = self._evaluate_transverse_response(
            params,
            np.asarray(display_omega, dtype=float),
            modal=modal,
            return_full=False,
        )
        return {
            "omega": np.asarray(dense["omega"], dtype=float),
            "W_real": np.asarray(dense["W_real"], dtype=float),
            "W_imag": np.asarray(dense["W_imag"], dtype=float),
            "refined": True,
            "base_point_count": int(base_omega.size),
            "display_point_count": int(display_omega.size),
            "solver_path": "reuse_modal_data_with_dense_response_only",
        }

    def build_transverse_plot_im0_from_result(
        self,
        params: dict,
        result: dict,
        semantic_im0: dict | None = None,
        display_curve: dict | None = None,
    ) -> dict:
        """Подогнать plot-маркеры к той же sampled display-кривой, что рисует GUI."""
        if display_curve is None:
            display_curve = self.build_transverse_display_curve_from_result(params, result)

        omega = np.asarray(display_curve.get("omega", []), dtype=float)
        Wre = np.asarray(display_curve.get("W_real", []), dtype=float)
        Wim = np.asarray(display_curve.get("W_imag", []), dtype=float)

        eps = float(params.get("im0_eps_transverse", 1e-9))
        omega_tol = float(params.get("plot_im0_omega_tol_transverse", max(float(params.get("omega_step", 1.0)) * 0.5, 1e-9)))
        re_tol = float(params.get("plot_im0_re_tol_transverse", 1e-6))

        plot_points = self._build_zero_crossing_points(
            omega,
            Wre,
            Wim,
            zero_eps=eps,
            omega_tol=omega_tol,
            re_tol=re_tol,
            refine_func=None,
            re_eval_func=None,
            re_key="re",
        )

        semantic = semantic_im0 or {}
        semantic_research = semantic.get("research_critical_point")
        semantic_min_re = semantic.get("minimum_re_critical_point")

        def _match_plot_point(target: dict | None) -> dict | None:
            if not target or not plot_points:
                return None
            tw = float(target.get("omega", np.nan))
            tr = float(target.get("re", np.nan))
            if not (np.isfinite(tw) and np.isfinite(tr)):
                return None
            return min(
                plot_points,
                key=lambda p: (
                    abs(float(p["omega"]) - tw),
                    abs(float(p["re"]) - tr),
                ),
            )

        return {
            "points": plot_points,
            "research_critical_point": _match_plot_point(semantic_research),
            "minimum_re_critical_point": _match_plot_point(semantic_min_re),
            "source_curve": "display_curve_sampled_branch_for_plot",
        }

    def calculate_transverse(self, params: dict) -> dict:
        """Рассчитать поперечную модель на полной физической сетке с NaN-разрывами."""
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
        res = self.calculate_transverse(params)
        return self.find_transverse_im0_points_from_result(params, res)

    def _select_transverse_research_critical_point(self, points: list[dict], params: dict) -> tuple[dict | None, dict]:
        """Выбрать нетривиальное пересечение с отрицательной действительной осью."""
        omega_eps = float(params.get("transverse_research_omega_eps", max(float(params.get("omega_step", 1.0)) * 0.5, 1e-9)))
        negative_re_eps = float(params.get("transverse_research_negative_re_eps", 1e-9))
        policy = {
            "kind": "minimum_negative_ReW_on_im_zero_set",
            "criterion": "strict_negative_real_axis_intersection_only",
            "model_regime": "fixed_verified_mode",
            "omega_nontrivial_eps": omega_eps,
            "negative_re_eps": negative_re_eps,
        }
        if not points:
            return None, {**policy, "selection_status": "no_im0_points"}

        ordered = sorted(points, key=lambda p: (float(p["re"]), float(p["omega"])))
        research_candidates = [
            p for p in ordered
            if np.isfinite(float(p.get("omega", np.nan)))
            and np.isfinite(float(p.get("re", np.nan)))
            and float(p["omega"]) > omega_eps
            and float(p["re"]) < -negative_re_eps
        ]
        if not research_candidates:
            return None, {**policy, "selection_status": "no_negative_real_axis_intersection"}

        chosen = dict(research_candidates[0])
        chosen["im"] = 0.0
        return chosen, {**policy, "selection_status": "minimum_negative_re_nontrivial_point_selected"}

    def find_transverse_im0_points_from_result(self, params: dict, res: dict) -> dict:
        """Найти точки Im(W)=0 по уже рассчитанной поперечной кривой."""
        params = self.validate_transverse_params(params)
        omega = np.asarray(res["omega"], dtype=float)
        Wre = np.asarray(res["W_real"], dtype=float)
        Wim = np.asarray(res["W_imag"], dtype=float)

        eps = float(params.get("im0_eps_transverse", 1e-9))
        omega_tol = float(params.get("im0_omega_tol_transverse", max(float(params.get("omega_step", 1.0)) * 0.5, 1e-9)))
        re_tol = float(params.get("im0_re_tol_transverse", 1e-6))

        modal = {
            "alpha": float(res["alpha"]),
            "beta": float(res["beta"]),
            "gamma": float(res["gamma"]),
            "h": float(res["h"]),
            "phi_L": float(res["phi_L"]),
            "R": float(res["R"]),
            "r": float(res["r"]),
            "J": float(res["J"]),
            "S": float(res["S"]),
            "k1": float(res["k1"]),
            "shape_scale_C": float(res["shape_scale_C"]),
            "modal_shape_variant": res["modal_shape_variant"],
            "modal_shape_source": res["modal_shape_source"],
            "modal_shape_description": res["modal_shape_description"],
            "shape_normalization": res["shape_normalization"],
            "lambda1": float(res["lambda1"]),
            "shape_eta": res["shape_eta"],
            "modal_mass_integral": float(res["modal_mass_integral"]),
            "modal_curvature_integral": float(res["modal_curvature_integral"]),
            "damping_source": res["damping_source"],
            "model_variant": res["model_variant"],
            "transverse_model_regime": res.get("transverse_model_regime"),
            "transverse_model_regime_label": res.get("transverse_model_regime_label"),
            "transverse_model_scope": res.get("transverse_model_scope"),
            "transverse_model_note": res.get("transverse_model_note"),
            "research_alignment_status": res.get("research_alignment_status"),
        }

        mu = float(params["mu"])
        tau = float(params["tau"])
        K_cut = float(params.get("K_cut", 6e5))
        alpha = float(modal["alpha"])
        beta = float(modal["beta"])
        gamma = float(modal["gamma"])
        phi_L = float(modal["phi_L"])
        denom_eps = float(params.get("transverse_denom_eps", 1e-12))

        def _eval_transverse_scalar(w: float) -> tuple[float, float]:
            p = 1j * float(w)
            with np.errstate(all="ignore"):
                numerator = (phi_L ** 2) * K_cut * (1.0 - mu * np.exp(-p * tau))
                denom = alpha * p ** 2 + beta * p + gamma
                if not (np.isfinite(denom.real) and np.isfinite(denom.imag)) or abs(denom) < denom_eps:
                    return float("nan"), float("nan")
                value = numerator / denom
            if not (np.isfinite(value.real) and np.isfinite(value.imag)):
                return float("nan"), float("nan")
            return float(value.real), float(value.imag)

        def im_func(w):
            return _eval_transverse_scalar(w)[1]

        def re_func(w):
            return _eval_transverse_scalar(w)[0]

        dedup = self._build_zero_crossing_points(
            omega, Wre, Wim,
            zero_eps=eps,
            omega_tol=omega_tol,
            re_tol=re_tol,
            refine_func=im_func,
            re_eval_func=re_func,
            re_key="re",
        )
        research_critical_point, policy_meta = self._select_transverse_research_critical_point(dedup, params)
        minimum_re_critical_point = min(dedup, key=lambda p: (float(p["re"]), float(p["omega"]))) if dedup else None
        if minimum_re_critical_point is not None:
            minimum_re_critical_point = dict(minimum_re_critical_point)
            minimum_re_critical_point["im"] = 0.0
        return {
            "points": dedup,
            "research_critical_point": research_critical_point,
            "minimum_re_critical_point": minimum_re_critical_point,
            "critical": research_critical_point,
            "source_curve": "direct_transverse_curve",
            "critical_selection_policy": policy_meta,
        }
