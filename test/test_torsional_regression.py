# test/test_torsional_regression.py
import numpy as np


def _coth(z):
    # независимая реализация coth(z) = cosh(z)/sinh(z)
    return np.cosh(z) / np.sinh(z)


def _sigma_torsional_reference(omega, *, rho, G, Jp, Jr, delta1, length, multiplier):
    """
    σ(p) = -p - λ1*sqrt(1+δ1*p)*coth(λ2*p/sqrt(1+δ1*p)), p=iω
    как в Matlab-листинге исследования.
    """
    p = 1j * omega
    lam1 = np.sqrt(rho * G) * Jp / Jr
    lam2 = length * np.sqrt(rho / G)
    d1 = delta1 * multiplier

    expr = np.sqrt(1.0 + d1 * p)
    arg = lam2 * p / expr
    return -p - lam1 * expr * _coth(arg)


def test_torsional_sigma_matches_formula_at_control_points(model):
    """
    Регрессия: σ(iω) должен соответствовать формуле из исследования
    (сравниваем с независимым расчётом в нескольких ω).
    """
    params = dict(
        rho=7800.0,
        G=8.0e10,
        Jr=2.57e-2,
        Jp=1.9e-5,
        delta1=3.44e-6,
        multiplier=1,
        length=3.0,
    )

    res = model.calculate_torsional(params)
    omega_grid = np.asarray(res["omega"], dtype=float)
    sr = np.asarray(res["sigma_real"], dtype=float)
    si = np.asarray(res["sigma_imag"], dtype=float)

    # контрольные точки ω (как типовые в диапазоне исследования)
    control = [1000.0, 5000.0, 10000.0, 15000.0]

    for w in control:
        # интерполируем σ по рассчитанному массиву (в модели сетка linspace, w может не попасть в узел)
        sr_i = np.interp(w, omega_grid, sr)
        si_i = np.interp(w, omega_grid, si)
        s_model = sr_i + 1j * si_i

        s_ref = _sigma_torsional_reference(w, **params)

        # допуски — умеренные, т.к. сравнение через интерполяцию
        assert np.isfinite(s_model.real) and np.isfinite(s_model.imag)
        assert np.isfinite(s_ref.real) and np.isfinite(s_ref.imag)

        assert np.isclose(s_model.real, s_ref.real, rtol=2e-3, atol=1e-2)
        assert np.isclose(s_model.imag, s_ref.imag, rtol=2e-3, atol=1e-2)


def test_torsional_intersection_exists_and_in_expected_range(model):
    """
    Регрессия: ω* должен находиться (примерно) в тех же рамках,
    что и в Matlab (brackets ~ [500..20000]).
    """
    params = dict(
        rho=7800.0,
        G=8.0e10,
        Jr=2.57e-2,
        Jp=1.9e-5,
        delta1=3.44e-6,
        multiplier=1,
        length=3.0,
    )

    inter = model.find_intersection(params)
    assert inter is not None

    omega_star = float(inter["omega"])
    assert 500.0 < omega_star < 20000.0
    assert np.isfinite(inter["re_sigma"])
    assert np.isfinite(inter["frequency"])

def test_torsional_real_part_range_diagnostics(model):
    """
    Диагностика: смотрим диапазон Re(σ).
    Этот тест не проверяет теорию,
    а помогает понять масштаб графика.
    """
    params = dict(
        rho=7800.0,
        G=8.0e10,
        Jr=2.57e-2,
        Jp=1.9e-5,
        delta1=3.44e-6,
        multiplier=1,
        length=3.0,
        omega_start=1000.0,
        omega_end=15000.0,
        omega_step=1.0,
    )

    res = model.calculate_torsional(params)

    re_sigma = np.asarray(res["sigma_real"], dtype=float)

    re_min = np.nanmin(re_sigma)
    re_max = np.nanmax(re_sigma)

    print("\nRe(σ) min =", re_min)
    print("Re(σ) max =", re_max)

    # sanity-проверки
    assert re_min < 0
    assert re_max > re_min

def test_torsional_real_part_diagnostics(model):
    """
    Диагностика крутильных колебаний:
    вывод Re(σ) в точках пересечения Im(σ)=0.
    """

    params = dict(
        rho=7800.0,
        G=8.0e10,
        Jr=2.57e-2,
        Jp=1.9e-5,
        delta1=3.44e-6,
        multiplier=1,
        length=3.0,
        omega_start=1000.0,
        omega_end=15000.0,
        omega_step=1.0,
    )

    res = model.calculate_torsional(params)

    omega = np.asarray(res["omega"], dtype=float)
    sigma_re = np.asarray(res["sigma_real"], dtype=float)
    sigma_im = np.asarray(res["sigma_imag"], dtype=float)

    # ищем смену знака Im(σ)
    sign_changes = np.where(np.diff(np.sign(sigma_im)) != 0)[0]

    print("\n===== Крутильные =====")
    print("Количество пересечений Im(σ)=0:", len(sign_changes))

    assert len(sign_changes) > 0

    for idx in sign_changes:
        omega_star = omega[idx]
        re_star = sigma_re[idx]
        im_star = sigma_im[idx]
        f_star = omega_star / (2 * np.pi)

        print("omega* =", omega_star, "рад/с")
        print("f* =", f_star, "Гц")
        print("Re(σ*) =", re_star)
        print("Im(σ*) ≈", im_star)
        print("----------------------")

        # базовая проверка
        assert np.isfinite(re_star)
        assert np.isfinite(im_star)